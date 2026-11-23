"""
Unified Causal Streaming Conformer ASR Model (130M MoE).
Integrates:
- Streaming Log-Mel Spectrogram Front-End (with causal window buffer)
- 12-layer Causal Conformer Encoder (with layers 5-12 equipped with Sparse MoE dialect experts)
- CTC Decoder Head
- 100% Parameter-compatible weight loading from offline checkpoint
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torchaudio.transforms as T

from .causal_conformer import CausalConformerEncoder, StreamingEncoderState
from .moe import SparseMoELayer
from .streaming_decoder import StreamingCTCDecoder, DecoderState


class StreamingLogMelFrontEnd(nn.Module):
    """
    Log-Mel Spectrogram Frontend supporting both full waveforms and streaming chunks.
    Maintains unfinished waveform samples across chunk boundaries to eliminate window framing jitter.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,       # 25 ms window
        hop_length: int = 160,  # 10 ms hop (100 fps)
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=False,  # Causal framing: window starts at t, no future samples needed!
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Full waveform forward pass.
        Args:
            waveforms: (batch, num_samples)
        Returns:
            log_mel: (batch, time_frames, 80)
        """
        # Left-pad n_fft - hop_length to ensure first frame starts cleanly
        pad_amount = self.n_fft - self.hop_length
        padded = nn.functional.pad(waveforms, (pad_amount, 0))
        mel = self.mel_transform(padded)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.transpose(1, 2)

    def forward_chunk(
        self,
        audio_chunk: torch.Tensor,
        wav_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract log-mel frames incrementally from an incoming audio chunk.
        Args:
            audio_chunk: (batch, samples)
            wav_buffer: Residual waveform from previous chunk
        Returns:
            log_mel_chunk: (batch, new_frames, 80)
            new_buffer: Residual waveform samples to prepend to next chunk
        """
        if wav_buffer is not None and wav_buffer.size(1) > 0:
            full_wav = torch.cat([wav_buffer, audio_chunk], dim=1)
        else:
            pad_amount = self.n_fft - self.hop_length
            full_wav = nn.functional.pad(audio_chunk, (pad_amount, 0))

        num_samples = full_wav.size(1)
        if num_samples < self.n_fft:
            # Not enough samples for a single frame yet
            empty_mel = torch.empty(full_wav.size(0), 0, self.n_mels, device=audio_chunk.device)
            return empty_mel, full_wav

        # Calculate number of full hops possible
        num_frames = (num_samples - self.n_fft) // self.hop_length + 1
        usable_samples = (num_frames - 1) * self.hop_length + self.n_fft
        chunk_to_process = full_wav[:, :usable_samples]
        residual_wav = full_wav[:, usable_samples - (self.n_fft - self.hop_length):].detach()

        mel = self.mel_transform(chunk_to_process)
        log_mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)
        return log_mel, residual_wav


class CTCHead(nn.Module):
    """Linear CTC Projection Head with LayerNorm."""

    def __init__(self, d_model: int = 512, vocab_size: int = 105):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            logits: (batch, time, vocab_size)
        """
        x = self.norm(x)
        return self.proj(x)


class CausalStreamingASRModel(nn.Module):
    """
    Unified 130M Causal Conformer Streaming ASR Model.
    Supports both:
    1. 130M Sparse MoE (12 layers, layers 5-12 equipped with 3 regional dialect experts)
    2. 130M Dense (12 layers, full dense FFN with 8x expansion)
    """

    def __init__(
        self,
        feat_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 12,
        n_heads: int = 8,
        conv_kernel_size: int = 31,
        ffn_expansion: int = 4,
        enable_moe: bool = True,
        num_experts: int = 3,
        moe_start_layer: int = 5,  # Layers 5-12 are MoE
        dropout: float = 0.1,
        left_context: Optional[int] = None,
        vocab_size: int = 105,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.enable_moe = enable_moe
        self.ffn_expansion = ffn_expansion

        # 1. Front-end
        self.frontend = StreamingLogMelFrontEnd(n_mels=feat_dim)

        # 2. Causal Conformer Backbone
        self.encoder = CausalConformerEncoder(
            feat_dim=feat_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            conv_kernel_size=conv_kernel_size,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            left_context=left_context,
            exit_layers=[4, 8, 12],
        )

        # Replace FFN2 in layers moe_start_layer-1 to num_layers-1 with SparseMoELayer if enabled
        if self.enable_moe:
            for idx in range(moe_start_layer - 1, num_layers):
                self.encoder.layers[idx].ffn2 = SparseMoELayer(
                    d_model=d_model,
                    expansion_factor=ffn_expansion,
                    num_experts=num_experts,
                    dropout=dropout,
                )

        # 3. CTC Head
        self.ctc_head = CTCHead(d_model=d_model, vocab_size=vocab_size)

        # 4. Reconstruction Head & Span Masker for Self-Supervised Pretraining
        try:
            from model.heads import MaskedReconstructionHead
            from model.masking import SpanMasker
        except ImportError:
            import importlib.util
            from pathlib import Path
            dense_dir = Path(__file__).resolve().parent.parent / "marathi-conformer-130m-dense"
            heads_spec = importlib.util.spec_from_file_location("_dense_heads", dense_dir / "heads.py")
            heads_mod = importlib.util.module_from_spec(heads_spec)
            heads_spec.loader.exec_module(heads_mod)
            mask_spec = importlib.util.spec_from_file_location("_dense_masking", dense_dir / "masking.py")
            mask_mod = importlib.util.module_from_spec(mask_spec)
            mask_spec.loader.exec_module(mask_mod)
            MaskedReconstructionHead = heads_mod.MaskedReconstructionHead
            SpanMasker = mask_mod.SpanMasker
        self.span_masker = SpanMasker(mask_prob=0.40, mean_span_length=10)
        self.reconstruction_head = MaskedReconstructionHead(
            d_model=d_model, feat_dim=feat_dim, hidden_dim=512
        )

    def forward_pretrain(
        self,
        waveforms: torch.Tensor,
        custom_span_length: Optional[int] = None,
        custom_mask_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for Causal Self-Supervised Masked Reconstruction Pretraining.
        """
        spec = self.frontend(waveforms)
        t_len = spec.size(1)

        masked_spec, mask = self.span_masker(
            spec,
            custom_span_length=custom_span_length,
            custom_mask_prob=custom_mask_prob,
        )

        enc_out = self.encoder(masked_spec, return_all_exits=True)
        final_hidden = enc_out["final_hidden"]

        reconstructed_spec = self.reconstruction_head(final_hidden, target_time_len=t_len)

        return {
            "original_spec": spec,
            "masked_spec": masked_spec,
            "mask": mask,
            "reconstructed_spec": reconstructed_spec,
            "final_hidden": final_hidden,
            "exit_hiddens": enc_out["exit_hiddens"],
        }

    def forward(
        self,
        waveforms_or_spec: torch.Tensor,
        is_waveform: bool = True,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full sequence causal forward pass.
        """
        if is_waveform:
            spec = self.frontend(waveforms_or_spec)
        else:
            spec = waveforms_or_spec

        enc_out = self.encoder(
            spec,
            dialect_idx=dialect_idx,
            use_hard_routing=use_hard_routing,
            return_all_exits=True,
        )
        final_hidden = enc_out["final_hidden"]
        logits = self.ctc_head(final_hidden)

        return {
            "logits": logits,
            "final_hidden": final_hidden,
            "exit_hiddens": enc_out["exit_hiddens"],
        }

    def init_streaming_state(self) -> Dict[str, Any]:
        """Initialize all streaming caches (front-end, encoder, decoder)."""
        return {
            "wav_buffer": None,
            "encoder_state": self.encoder.init_streaming_state(),
        }

    def forward_chunk(
        self,
        audio_chunk: torch.Tensor,
        state: Dict[str, Any],
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Streaming chunk forward pass.
        Args:
            audio_chunk: (batch, num_samples) raw audio chunk
            state: Streaming state dict from init_streaming_state()
        Returns:
            logits_chunk: (batch, chunk_sub_frames, vocab_size)
            next_state: Updated state dict
        """
        # 1. Frontend feature extraction
        spec_chunk, next_wav_buf = self.frontend.forward_chunk(
            audio_chunk,
            wav_buffer=state.get("wav_buffer"),
        )

        if spec_chunk.size(1) == 0:
            # Not enough samples for a frame yet
            empty_logits = torch.empty(audio_chunk.size(0), 0, self.vocab_size, device=audio_chunk.device)
            return empty_logits, {"wav_buffer": next_wav_buf, "encoder_state": state["encoder_state"]}

        # Ensure spec chunk is a multiple of 4 for 4x subsampling
        time_rem = spec_chunk.size(1) % 4
        if time_rem != 0:
            usable_spec = spec_chunk[:, :-time_rem, :]
        else:
            usable_spec = spec_chunk

        if usable_spec.size(1) == 0:
            empty_logits = torch.empty(audio_chunk.size(0), 0, self.vocab_size, device=audio_chunk.device)
            return empty_logits, {"wav_buffer": next_wav_buf, "encoder_state": state["encoder_state"]}

        # 2. Encoder chunk forward
        hidden_chunk, next_enc_state = self.encoder.forward_chunk(
            usable_spec,
            state=state["encoder_state"],
            dialect_idx=dialect_idx,
            use_hard_routing=use_hard_routing,
        )

        # 3. CTC logits
        logits_chunk = self.ctc_head(hidden_chunk)

        next_state = {
            "wav_buffer": next_wav_buf,
            "encoder_state": next_enc_state,
        }
        return logits_chunk, next_state

    def load_offline_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Load weights from the offline-trained 130M MoE model checkpoint.
        Transfers convolution weights, attention projections, MoE combining trunk + experts,
        router weights, and CTC projection.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        raw_state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        # Normalize optional 'backbone.' prefix from wrapped checkpoints
        state_dict = {}
        for k, v in raw_state_dict.items():
            norm_k = k[len("backbone."):] if k.startswith("backbone.") else k
            state_dict[norm_k] = v

        # Map checkpoint keys to model state dict
        own_state = self.state_dict()
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []

        # Key aliases between offline Conv2dSubsampling4 (nn.Sequential) / CausalConv2dSubsampling4
        # and optional reconstruction_head naming variants
        key_aliases = {
            "encoder.subsampling.conv1.weight": ["encoder.subsampling.conv1.weight", "encoder.subsampling.conv.0.weight"],
            "encoder.subsampling.conv1.bias": ["encoder.subsampling.conv1.bias", "encoder.subsampling.conv.0.bias"],
            "encoder.subsampling.conv2.weight": ["encoder.subsampling.conv2.weight", "encoder.subsampling.conv.2.weight"],
            "encoder.subsampling.conv2.bias": ["encoder.subsampling.conv2.bias", "encoder.subsampling.conv.2.bias"],
            "reconstruction_head.net.0.weight": ["reconstruction_head.net.0.weight", "recon_head.net.0.weight"],
            "reconstruction_head.net.0.bias": ["reconstruction_head.net.0.bias", "recon_head.net.0.bias"],
            "reconstruction_head.net.2.weight": ["reconstruction_head.net.2.weight", "recon_head.net.2.weight"],
            "reconstruction_head.net.2.bias": ["reconstruction_head.net.2.bias", "recon_head.net.2.bias"],
            "reconstruction_head.net.3.weight": ["reconstruction_head.net.3.weight", "recon_head.net.3.weight"],
            "reconstruction_head.net.3.bias": ["reconstruction_head.net.3.bias", "recon_head.net.3.bias"],
        }

        used_source_keys = set()
        with torch.no_grad():
            for name, param in own_state.items():
                candidates = key_aliases.get(name, [name])
                source_name = next((c for c in candidates if c in state_dict), None)
                if source_name is not None:
                    src_val = state_dict[source_name]
                    if src_val.shape == param.shape:
                        param.copy_(src_val)
                        loaded_keys.append(name)
                        used_source_keys.add(source_name)
                    else:
                        print(f"Shape mismatch on {name}: ckpt {list(src_val.shape)} vs model {list(param.shape)}")
                        missing_keys.append(name)
                else:
                    missing_keys.append(name)

        for name in state_dict.keys():
            if name not in used_source_keys and name not in own_state:
                unexpected_keys.append(name)

        print(f"Checkpoint loaded: {len(loaded_keys)} / {len(own_state)} parameters transferred.")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")

        if strict and missing_keys:
            raise RuntimeError(f"Strict load failed with {len(missing_keys)} missing keys.")

        return {
            "loaded_keys": loaded_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
