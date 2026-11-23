"""
Causal Conformer Encoder for Real-time Streaming Marathi ASR.
Strictly causal architecture with zero future frame lookahead:
1. Causal 4x Convolutional Subsampling.
2. Causal Multi-Head Self-Attention with K/V caching and RoPE.
3. Causal Conformer Convolution Module with depthwise past-frame caching.
4. Frame-wise Mixture-of-Experts routing on layers 5-12.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_subsampling import CausalConv2dSubsampling4


@dataclass
class StreamingEncoderState:
    """State retained across consecutive streaming chunks."""
    subsampling_cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
    att_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    conv_caches: List[Optional[torch.Tensor]]
    subsampled_time_offset: int

    @classmethod
    def create_empty(cls, num_layers: int = 12) -> "StreamingEncoderState":
        return cls(
            subsampling_cache=(None, None),
            att_caches=[None for _ in range(num_layers)],
            conv_caches=[None for _ in range(num_layers)],
            subsampled_time_offset=0,
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding supporting streaming time offsets."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, time_offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time, d_model)
            time_offset: Position offset for streaming chunks
        """
        t = x.size(1)
        x = x + self.pe[:, time_offset : time_offset + t, :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) supporting streaming time offsets."""

    def __init__(self, d_k: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        time_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (b, n_heads, t_q, d_k)
            k: (b, n_heads, t_k, d_k)
            time_offset: Starting position index for q and k
        """
        t_req = time_offset + max(q.size(2), k.size(2))
        if t_req > self.cos_cached.size(0):
            self._build_cache(t_req + 512)

        t_q = q.size(2)
        t_k = k.size(2)

        cos_q = self.cos_cached[time_offset : time_offset + t_q, :].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
        sin_q = self.sin_cached[time_offset : time_offset + t_q, :].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)

        cos_k = self.cos_cached[time_offset : time_offset + t_k, :].to(dtype=k.dtype, device=k.device).unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cached[time_offset : time_offset + t_k, :].to(dtype=k.dtype, device=k.device).unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_rot, k_rot


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with K/V past-state caching and RoPE.
    Strictly prevents query position i from attending to key positions j > i.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        left_context: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.left_context = left_context

        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        att_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        time_offset: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch, time_q, d_model)
            att_cache: Optional tuple of (cached_k, cached_v)
                       - cached_k: (batch, n_heads, t_past, d_k)
                       - cached_v: (batch, n_heads, t_past, d_k)
            time_offset: Offset in time frames for current chunk
            mask: Optional explicit boolean mask
        Returns:
            out: (batch, time_q, d_model)
            new_cache: Updated (cached_k, cached_v)
        """
        residual = x
        x = self.layer_norm(x)
        b, t_q, _ = x.size()

        # Compute Q, K, V for current chunk
        q = self.q_proj(x).view(b, t_q, self.n_heads, self.d_k).transpose(1, 2)
        k_new = self.k_proj(x).view(b, t_q, self.n_heads, self.d_k).transpose(1, 2)
        v_new = self.v_proj(x).view(b, t_q, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE positional embeddings to newly computed Q and K
        q, k_new = self.rope(q, k_new, time_offset=time_offset)

        # Combine with K/V cache if in streaming mode
        if att_cache is not None and att_cache[0] is not None:
            past_k, past_v = att_cache
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        t_k = k.size(2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (b, n_heads, t_q, t_k)

        # Build causal mask: query i at absolute pos (time_offset + i) attends to j <= (time_offset + i)
        q_pos = torch.arange(time_offset, time_offset + t_q, device=x.device).unsqueeze(1)
        k_pos = torch.arange(0, t_k, device=x.device).unsqueeze(0)
        causal_mask = k_pos <= q_pos  # (t_q, t_k) True = allowed

        if self.left_context is not None and self.left_context > 0:
            left_mask = k_pos >= (q_pos - self.left_context)
            causal_mask = causal_mask & left_mask

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, t_q, t_k)

        if mask is not None:
            causal_mask = causal_mask & mask

        neg_inf = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~causal_mask, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (b, n_heads, t_q, d_k)
        context = context.transpose(1, 2).contiguous().view(b, t_q, self.d_model)
        out = self.out_proj(context)
        out = self.dropout(out)
        out = residual + out

        new_cache = None
        if att_cache is not None or not self.training:
            # If a left context limit is configured, discard older keys/values
            if self.left_context is not None and self.left_context > 0 and k.size(2) > self.left_context:
                k_save = k[:, :, -self.left_context :, :].detach()
                v_save = v[:, :, -self.left_context :, :].detach()
            else:
                k_save = k.detach()
                v_save = v.detach()
            new_cache = (k_save, v_save)

        return out, new_cache


class CausalConformerConvModule(nn.Module):
    """
    Causal Conformer Convolution Module.
    Uses strictly causal depthwise convolution (left padding = K - 1, right padding = 0)
    and maintains the past K - 1 input frames for incremental streaming inference.
    """

    def __init__(self, d_model: int = 512, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1  # 30 for kernel_size=31

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        # Depthwise conv has padding=0; padding is handled explicitly via past cache
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=d_model,
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)  # Per-frame LayerNorm (strictly causal, replaces temporal stats)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        conv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, time, d_model)
            conv_cache: Optional past inputs tensor of shape (batch, d_model, K - 1)
        Returns:
            out: (batch, time, d_model)
            new_cache: Updated past inputs tensor (batch, d_model, K - 1)
        """
        residual = x
        x = self.layer_norm(x)
        # (batch, channel, time)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Causal depthwise temporal convolution
        if conv_cache is not None:
            # Prepend past K - 1 frames
            x_padded = torch.cat([conv_cache, x], dim=2)
            new_cache = x_padded[:, :, -self.pad_len :].detach()
        else:
            # Full sequence: left-pad K - 1 zeros
            x_padded = F.pad(x, (self.pad_len, 0))
            new_cache = x_padded[:, :, -self.pad_len :].detach()

        x = self.depthwise_conv(x_padded)
        # Per-frame LayerNorm along channel dimension: zero temporal future leakage
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out = residual + x
        return out, new_cache


class FeedForwardModule(nn.Module):
    """Macaron-style Feed-Forward Module."""

    def __init__(self, d_model: int = 512, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class CausalConformerBlock(nn.Module):
    """
    A single Causal Conformer Block with streaming state support.
    Order:
      1. Half-step FFN1
      2. Causal Multi-Head Self-Attention (with K/V cache)
      3. Causal Convolution Module (with depthwise conv cache)
      4. Half-step FFN2 (or SparseMoELayer)
      5. Final LayerNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        conv_kernel_size: int = 31,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        left_context: Optional[int] = None,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.self_attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            left_context=left_context,
        )
        self.conv_module = CausalConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        att_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        conv_cache: Optional[torch.Tensor] = None,
        time_offset: int = 0,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        x = self.ffn1(x)
        x, new_att_cache = self.self_attn(x, att_cache=att_cache, time_offset=time_offset)
        x, new_conv_cache = self.conv_module(x, conv_cache=conv_cache)

        if type(self.ffn2).__name__ == "SparseMoELayer":
            x = self.ffn2(x, dialect_idx=dialect_idx, use_hard_routing=use_hard_routing)
        else:
            x = self.ffn2(x)

        x = self.final_norm(x)
        return x, new_att_cache, new_conv_cache


class CausalConformerEncoder(nn.Module):
    """
    Full 12-layer Causal Conformer Acoustic Encoder for Real-time Streaming ASR.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 12,
        n_heads: int = 8,
        conv_kernel_size: int = 31,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        left_context: Optional[int] = None,
        exit_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.exit_layers = exit_layers or [num_layers]

        self.subsampling = CausalConv2dSubsampling4(
            in_channels=1,
            out_dim=d_model,
            feat_dim=feat_dim,
        )
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            CausalConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                left_context=left_context,
            )
            for _ in range(num_layers)
        ])

    def init_streaming_state(self) -> StreamingEncoderState:
        """Create a fresh streaming state object for real-time inference."""
        return StreamingEncoderState.create_empty(self.num_layers)

    def forward(
        self,
        x: torch.Tensor,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
        return_all_exits: bool = False,
    ) -> Dict[str, Any]:
        """
        Full-sequence causal forward pass (for training and offline evaluation).
        Args:
            x: (batch, time_frames, feat_dim)
        """
        # Causal subsampling
        x, _ = self.subsampling(x, cache=None)
        x = self.pos_enc(x, time_offset=0)

        exit_hiddens: Dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers, start=1):
            x, _, _ = layer(
                x,
                att_cache=None,
                conv_cache=None,
                time_offset=0,
                dialect_idx=dialect_idx,
                use_hard_routing=use_hard_routing,
            )
            if idx in self.exit_layers:
                exit_hiddens[idx] = x

        return {
            "final_hidden": x,
            "exit_hiddens": exit_hiddens,
        }

    def forward_chunk(
        self,
        chunk: torch.Tensor,
        state: StreamingEncoderState,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> Tuple[torch.Tensor, StreamingEncoderState]:
        """
        Incremental streaming chunk inference forward pass.
        Args:
            chunk: Input audio spectrogram chunk of shape (batch, chunk_time, feat_dim)
                   chunk_time should be a multiple of 4 (e.g. 16 frames = 640ms, 8 frames = 320ms, 4 frames = 160ms)
            state: StreamingEncoderState retained from previous chunks
        Returns:
            out_chunk: (batch, chunk_time // 4, d_model)
            next_state: Updated StreamingEncoderState for the next chunk
        """
        # 1. Causal subsampling with past frame buffer
        x, new_sub_cache = self.subsampling(chunk, cache=state.subsampling_cache)
        chunk_sub_time = x.size(1)
        curr_offset = state.subsampled_time_offset
        x = self.pos_enc(x, time_offset=curr_offset)

        # 2. Pass chunk through 12 causal Conformer blocks
        new_att_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        new_conv_caches: List[Optional[torch.Tensor]] = []

        for idx, layer in enumerate(self.layers):
            x, new_att, new_conv = layer(
                x,
                att_cache=state.att_caches[idx],
                conv_cache=state.conv_caches[idx],
                time_offset=curr_offset,
                dialect_idx=dialect_idx,
                use_hard_routing=use_hard_routing,
            )
            new_att_caches.append(new_att)
            new_conv_caches.append(new_conv)

        next_state = StreamingEncoderState(
            subsampling_cache=new_sub_cache,
            att_caches=new_att_caches,
            conv_caches=new_conv_caches,
            subsampled_time_offset=curr_offset + chunk_sub_time,
        )

        return x, next_state
