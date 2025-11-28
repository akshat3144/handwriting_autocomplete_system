"""
Multi-Scale Discriminator for HiGAN+

Implements a unified discriminator that provides:
1. Global discrimination (overall image quality)
2. Patch-level discrimination (local details)  
3. Character-level attention (per-character quality)

Shares backbone features for efficiency.
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from .utils import _len2mask


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator with shared backbone.
    More efficient than separate global + patch discriminators.
    """
    
    def __init__(self, input_nc=1, ndf=64, n_layers=4,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-12,
                 use_attention=True, attention_resolution=32):
        super().__init__()
        
        self.name = 'MultiScaleD'
        self.use_attention = use_attention
        
        # Spectral normalized convolution
        which_conv = functools.partial(
            layers.SNConv2d,
            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
            eps=SN_eps
        )
        
        # Shared backbone (first 3 layers)
        self.shared_backbone = nn.ModuleList()
        
        # Initial layer
        self.shared_backbone.append(nn.Sequential(
            which_conv(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        ))
        
        # Downsampling layers
        nf = ndf
        for i in range(1, 3):
            nf_prev = nf
            nf = min(nf * 2, 512)
            self.shared_backbone.append(nn.Sequential(
                which_conv(nf_prev, nf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False)
            ))
        
        # === Branch 1: Global Discriminator Head ===
        self.global_head = nn.ModuleList()
        nf_global = nf
        for i in range(n_layers - 3):
            nf_prev = nf_global
            nf_global = min(nf_global * 2, 512)
            self.global_head.append(nn.Sequential(
                which_conv(nf_prev, nf_global, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False)
            ))
        
        # Global output
        self.global_output = which_conv(nf_global, 1, kernel_size=4, stride=1, padding=1)
        
        # === Branch 2: Patch Discriminator Head ===
        self.patch_head = nn.Sequential(
            which_conv(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            which_conv(nf, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # === Branch 3: Character-level Attention Head ===
        if self.use_attention:
            self.char_attention = nn.Sequential(
                which_conv(nf, nf // 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                which_conv(nf // 2, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
    
    def forward(self, x, x_lens=None, y_lens=None, return_features=False):
        """
        Args:
            x: [B, C, H, W] input image
            x_lens: [B] image widths
            y_lens: [B] text lengths
            return_features: if True, return intermediate features
        
        Returns:
            dict with 'global', 'patch', 'char_attn' scores
        """
        batch_size = x.size(0)
        
        # Shared backbone
        feat = x
        shared_feats = []
        for layer in self.shared_backbone:
            feat = layer(feat)
            shared_feats.append(feat)
        
        # Create length mask for shared features
        if x_lens is not None:
            feat_lens = x_lens // 8  # After 3 downsamples
            mask = _len2mask(feat_lens.int(), feat.size(-1), torch.float32).to(x.device)
            mask = mask.view(batch_size, 1, 1, -1)
        else:
            mask = None
        
        # Branch 1: Global
        global_feat = feat
        for layer in self.global_head:
            global_feat = layer(global_feat)
        global_out = self.global_output(global_feat)
        
        # Pool global output
        if mask is not None:
            global_mask = _len2mask((feat_lens // 4).int(), global_out.size(-1), torch.float32).to(x.device)
            global_mask = global_mask.view(batch_size, 1, 1, -1)
            global_score = (global_out * global_mask).sum([2, 3]) / (global_mask.sum([2, 3]) + 1e-8)
        else:
            global_score = global_out.mean([2, 3])
        
        # Branch 2: Patch
        patch_out = self.patch_head(feat)
        if mask is not None:
            patch_score = (patch_out * mask).sum([2, 3]) / (mask.sum([2, 3]) + 1e-8)
        else:
            patch_score = patch_out.mean([2, 3])
        
        # Branch 3: Character attention (optional)
        if self.use_attention:
            char_attn = self.char_attention(feat)
            if mask is not None:
                char_attn = char_attn * mask
        else:
            char_attn = None
        
        outputs = {
            'global': global_score,
            'patch': patch_score,
            'char_attn': char_attn,
            'combined': global_score + 0.5 * patch_score  # Weighted combination
        }
        
        if return_features:
            outputs['features'] = shared_feats
        
        return outputs


class ProgressiveDiscriminator(nn.Module):
    """
    Discriminator that supports progressive growing.
    Can operate at different resolutions during training.
    """
    
    def __init__(self, input_nc=1, ndf=64, max_resolution=64,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-12):
        super().__init__()
        
        self.name = 'ProgressiveD'
        self.max_resolution = max_resolution
        self.current_resolution = max_resolution
        self.alpha = 1.0  # Blend factor for progressive growing
        
        which_conv = functools.partial(
            layers.SNConv2d,
            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
            eps=SN_eps
        )
        
        # Build layers for each resolution level
        self.from_rgb = nn.ModuleDict()
        self.blocks = nn.ModuleList()
        
        resolutions = [4, 8, 16, 32, 64]
        channels = [512, 256, 128, 64, 32]
        
        for i, (res, ch) in enumerate(zip(resolutions, channels)):
            if res <= max_resolution:
                # From RGB layer for this resolution
                self.from_rgb[str(res)] = nn.Sequential(
                    which_conv(input_nc, ch, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, inplace=False)
                )
                
                # Block for this resolution
                if i > 0:
                    prev_ch = channels[i-1] if i > 0 else ch
                    self.blocks.append(nn.Sequential(
                        which_conv(ch, prev_ch, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2, inplace=False),
                        which_conv(prev_ch, prev_ch, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=False)
                    ))
        
        # Final layers
        self.final_block = nn.Sequential(
            which_conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.output = which_conv(512, 1, kernel_size=4, stride=1, padding=0)
    
    def set_resolution(self, resolution, alpha=1.0):
        """Set current resolution and blend factor for progressive training."""
        self.current_resolution = resolution
        self.alpha = alpha
    
    def forward(self, x, x_lens=None, y_lens=None):
        # Resize input to current resolution if needed
        if x.size(-1) != self.current_resolution:
            x = F.interpolate(x, size=(self.current_resolution, x.size(-1)), 
                            mode='bilinear', align_corners=False)
        
        # Get from_rgb for current resolution
        feat = self.from_rgb[str(self.current_resolution)](x)
        
        # Process through blocks
        for block in self.blocks:
            feat = block(feat)
        
        # Final processing
        feat = self.final_block(feat)
        out = self.output(feat)
        
        # Global average pooling
        if x_lens is not None:
            out_lens = x_lens * out.size(-1) // (x.size(-1) + 1e-8)
            mask = _len2mask(out_lens.int(), out.size(-1), torch.float32).to(x.device)
            mask = mask.view(mask.size(0), 1, 1, -1)
            out = (out * mask).sum([2, 3]) / (mask.sum([2, 3]) + 1e-8)
        else:
            out = out.mean([2, 3])
        
        return out
