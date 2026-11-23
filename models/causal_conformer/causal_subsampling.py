"""
Causal 4x Convolutional Subsampling block for Streaming Conformer.
Replaces symmetric temporal padding with strictly left-only padding
and provides past-frame caching for seamless chunked streaming inference.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2dSubsampling4(nn.Module):
    """
    4x Causal Convolutional Subsampling block.
    Reduces 100 fps spectrogram frames to 25 fps (40ms per frame)
    using two 2D strided convolutions with strictly causal temporal padding.
    
    Zero future frame lookahead:
    - Frequency axis: symmetric padding=1 (preserves 80 -> 40 -> 20 dim reduction).
    - Time axis: left-only padding (pad_left=2, pad_right=0) with no future leakage.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 512, feat_dim: int = 80):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.feat_dim = feat_dim

        # Convolutions defined with padding=0 so we control temporal/freq padding explicitly
        self.conv1 = nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=2, padding=0)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=0)
        self.act2 = nn.SiLU()

        subsampled_feat_dim = feat_dim // 4  # 80 // 4 = 20
        self.out_proj = nn.Linear(out_dim * subsampled_feat_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass supporting both full-sequence and chunked streaming inference.

        Args:
            x: Input spectrogram tensor of shape (batch, time, feat_dim)
            cache: Optional tuple of (cache_layer1, cache_layer2)
                   - cache_layer1: (batch, in_channels, 2, feat_dim)
                   - cache_layer2: (batch, out_dim, 2, feat_dim // 2)

        Returns:
            out: (batch, time // 4, out_dim)
            new_cache: Updated tuple of (new_cache1, new_cache2) if cache is used or requested
        """
        # (batch, 1, time, feat_dim)
        x = x.unsqueeze(1)
        b, c, t, f = x.size()

        # -------------------------------------------------------------
        # Layer 1: Conv2d (stride 2)
        # -------------------------------------------------------------
        if cache is not None:
            c1 = cache[0]
            if c1 is not None:
                # Prepend cached past 2 frames along time dimension (dim=2)
                x1_input = torch.cat([c1, x], dim=2)
            else:
                # Initial chunk: left pad with 2 zero frames
                x1_input = F.pad(x, (0, 0, 2, 0))
            new_c1 = x1_input[:, :, -2:, :].detach()
        else:
            # Full sequence mode: left-pad 2 frames along time, 0 on right
            x1_input = F.pad(x, (0, 0, 2, 0))
            new_c1 = None

        # Freq padding: pad left 1, right 1 on freq axis (dim=3)
        x1_padded = F.pad(x1_input, (1, 1, 0, 0))
        x1 = self.act1(self.conv1(x1_padded))

        # -------------------------------------------------------------
        # Layer 2: Conv2d (stride 2)
        # -------------------------------------------------------------
        if cache is not None:
            c2 = cache[1]
            if c2 is not None:
                x2_input = torch.cat([c2, x1], dim=2)
            else:
                x2_input = F.pad(x1, (0, 0, 2, 0))
            new_c2 = x2_input[:, :, -2:, :].detach()
        else:
            x2_input = F.pad(x1, (0, 0, 2, 0))
            new_c2 = None

        x2_padded = F.pad(x2_input, (1, 1, 0, 0))
        x2 = self.act2(self.conv2(x2_padded))

        # Flatten & Project: (batch, time // 4, out_dim)
        b, c_out, t_sub, f_sub = x2.size()
        x_flat = x2.transpose(1, 2).contiguous().view(b, t_sub, c_out * f_sub)
        out = self.out_proj(x_flat)

        if cache is not None:
            return out, (new_c1, new_c2)
        return out, None
