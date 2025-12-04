"""
=============================================================================
GENERATOR AND DISCRIMINATOR FOR HiGAN+ HANDWRITING GENERATION
=============================================================================

This file contains the core GAN models:

1. Generator:
   - Input: style_vector (32-dim) + text (character indices)
   - Output: handwriting image [B, 1, 64, W]
   - Uses BigGAN-style architecture with conditional batch norm

2. Discriminator:
   - Input: handwriting image [B, 1, 64, W]
   - Output: real/fake score (scalar)
   - Uses spectral normalization for stability

3. PatchDiscriminator:
   - Input: 32x32 patches extracted from images
   - Output: real/fake score per patch
   - Focuses on local texture quality

KEY CONCEPTS:
- Spectral Normalization: Stabilizes discriminator training
- Conditional BatchNorm: Injects style into generator layers
- Self-Attention: Captures long-range character dependencies
- Hinge Loss: More stable than vanilla GAN loss

TRAINING FLOW:
    1. D sees real + fake images → outputs real/fake scores
    2. D loss = ReLU(1 - D(real)) + ReLU(1 + D(fake))  [Hinge loss]
    3. G generates images → D judges them
    4. G loss = -D(fake) + auxiliary losses (CTC, WID, reconstruction)
=============================================================================
"""
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from .improved_layers import (
    SinusoidalPositionalEncoding, TextTransformerEncoder,
    BiGRUEncoder, StyleContentCrossAttention, AdaIN,
    ModulatedConv2d, MultiScaleStyleFusion
)
from networks.utils import init_weights, _len2mask


# =============================================================================
# GENERATOR ARCHITECTURE CONFIGURATION
# =============================================================================
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    """
    Define Generator architecture for different resolutions.
    
    The generator progressively upsamples from 4x4 to 64xW:
        4x4 → 8x8 → 16x16 → 32x32 → 64x64
    
    Args:
        ch: Base channel multiplier (64)
        attention: Which resolutions get self-attention ('32_64' = both 32 and 64)
    
    Returns:
        arch: Dict with channel configs, upsample factors, attention flags
    """
    arch = {}

    # Resolution 64 (our target height)
    arch[64] = {
        'in_channels': [ch * item for item in [8, 4, 2, 1]],      # [512, 256, 128, 64]
        'out_channels': [ch * item for item in [4, 2, 1, 1]],     # [256, 128, 64, 64]
        'upsample': [(2,1), (2,2), (2,2), (2,2)],                  # (height, width) scale factors
        'resolution': [8, 16, 32, 64],                             # Resolution at each block
        'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                      for i in range(2, 7)}  # Which resolutions get attention
    }
    return arch


# =============================================================================
# GENERATOR: Style + Text → Handwriting Image
# =============================================================================
class Generator(nn.Module):
    """
    BigGAN-style Generator for handwriting synthesis.
    
    CORE IDEA: Combine a style vector (writing style) with text embeddings
    (what to write) to generate realistic handwriting images.
    
    ARCHITECTURE:
        Input:
            z: [B, 32] style vector (from random or StyleEncoder)
            y: [B, max_len] character indices (e.g., [7, 4, 11, 11, 14] = "hello")
            y_lens: [B] actual text lengths
        
        Processing:
            1. Embed characters: y → [B, max_len, 120]
            2. Concatenate style with each character: [B, max_len, 32+120]
            3. Project to initial feature map: [B, 512, 4, max_len*4]
            4. Upsample through GBlocks with style conditioning
            5. Apply self-attention at resolutions 32 and 64
            6. Output layer: BN → ReLU → Conv → Tanh
        
        Output:
            image: [B, 1, 64, W] where W = max_len * char_width (32 pixels/char)
    
    KEY COMPONENTS:
        - text_embedding: Character → 120-dim vector
        - filter_linear: Initial projection to 4x4 feature map
        - style_linear: Style → per-block conditioning vectors
        - GBlocks: Residual blocks with conditional batch norm
        - Attention: Self-attention for long-range dependencies
    """
    def __init__(self, G_ch=64, style_dim=32, embed_dim=120,
                 bottom_width=4, bottom_height=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_class=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5, SN_eps=1e-12, G_fp16=False,
                 init='ortho', G_param='SN', norm_style='bn', bn_linear='embed', input_nc=3,
                 embed_pad_idx=0, embed_max_norm=1.0
                 ):
        super(Generator, self).__init__()
        dim_z = style_dim
        self.style_dim = style_dim      # 32-dimensional style vector
        self.name = 'G'
        
        # Channel width multiplier
        self.ch = G_ch  # Base channels (64)
        
        # Dimensionality of the latent space (style)
        self.dim_z = dim_z  # 32
        self.embed_dim = embed_dim  # Character embedding dim (120)
        
        # Initial feature map dimensions (before upsampling)
        self.bottom_width = bottom_width    # 4 pixels wide per character
        self.bottom_height = bottom_height  # 4 pixels tall
        
        # Target output resolution (height)
        self.resolution = resolution  # 64
        
        # Kernel size for convolutions
        self.kernel_size = G_kernel_size
        
        # Which resolutions get self-attention
        self.attention = G_attn  # '32_64' = attention at 32x32 and 64x64
        
        # Number of character classes (alphabet size)
        self.n_classes = n_class  # 80 (a-z, A-Z, 0-9, punctuation, blank)
        
        # Batch norm settings
        self.cross_replica = cross_replica
        self.mybn = mybn
        
        # Activation function
        self.activation = G_activation
        
        # Initialization style ('ortho' = orthogonal init)
        self.init = init
        
        # Parameterization ('SN' = spectral normalization)
        self.G_param = G_param
        self.norm_style = norm_style
        self.BN_eps = BN_eps
        self.SN_eps = SN_eps
        self.fp16 = G_fp16
        
        # Get architecture config for this resolution
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.bn_linear = bn_linear

        self.z_chunk_size = self.dim_z  # 32

        # ===== CHARACTER EMBEDDING =====
        # Maps character indices to dense vectors
        # e.g., 'a' (index 0) → [0.1, -0.3, 0.5, ..., 0.2] (120-dim)
        self.text_embedding = nn.Embedding(self.n_classes, self.embed_dim,
                                           padding_idx=embed_pad_idx,
                                           max_norm=embed_max_norm)

        # ===== CHOOSE LAYER TYPES =====
        # Use spectral normalization for stability
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        if self.bn_linear=='SN':
            bn_linear = functools.partial(self.which_linear, bias=False)
        else:
            bn_linear = nn.Linear

        # Conditional BatchNorm: injects style into each layer
        self.which_bn = functools.partial(layers.ccbn,
                                          which_linear=bn_linear,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          input_size=self.z_chunk_size,  # 32 (style dim)
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        # ===== INITIAL PROJECTION =====
        # Maps (style + char_embed) to initial feature map
        # Input: [B, max_len, 32 + 120] → Output: [B, 512 * 4 * 4, max_len]
        self.filter_linear = self.which_linear(self.embed_dim + self.z_chunk_size,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
        
        # Style projection for each GBlock
        # Splits 32-dim style into 4 chunks (one per block)
        self.style_linear = self.which_linear(self.z_chunk_size,
                                              self.z_chunk_size * len(self.arch['in_channels']))

        # ===== RESIDUAL BLOCKS (GBlocks) =====
        # Each GBlock: upsamples + applies conditional batch norm with style
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv1=self.which_conv,
                                           which_conv2=self.which_conv,
                                           which_bn=self.which_bn,
                                           activation=self.activation,
                                           upsample=(functools.partial(F.interpolate,
                                                                       scale_factor=self.arch['upsample'][index])
                                                     if index < len(self.arch['upsample']) else None))]]

            # Add self-attention at specified resolutions
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Convert to ModuleList for proper registration
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # ===== OUTPUT LAYER =====
        # Final processing: BN → ReLU → Conv → Tanh (to [-1, 1] range)
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], input_nc))

        # Initialize weights
        if self.init != 'none':
            init_weights(self, self.init)

    def forward(self, z, y, y_lens):
        """
        Generate handwriting image from style and text.
        
        Args:
            z: [B, 32] style vector (random or from StyleEncoder)
            y: [B, max_len] character indices (0-79)
            y_lens: [B] actual text lengths (for masking)
        
        Returns:
            output: [B, 1, 64, W] generated grayscale image
                    W = max_len * 32 (32 pixels per character)
        """
        # Split style into per-block conditioning vectors
        # [B, 32] → [B, 32] x 4 (one for each GBlock)
        ys = self.style_linear(z).split(32, dim=1)

        # ===== STEP 1: EMBED TEXT + CONCATENATE STYLE =====
        # y: [B, max_len] → [B, max_len, 120]
        y = self.text_embedding(y).float().to(y.device)
        
        # Concatenate style with each character embedding
        # z: [B, 32] → [B, max_len, 32] (repeat for each character)
        # Then concat: [B, max_len, 32 + 120] = [B, max_len, 152]
        z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)
        
        # ===== STEP 2: PROJECT TO INITIAL FEATURE MAP =====
        # [B, max_len, 152] → [B, max_len, 512 * 4 * 4]
        h = self.filter_linear(z)

        # Reshape to 4D: each character becomes a 4-pixel-wide column
        # [B, max_len, 8192] → [B, 512, 4, max_len * 4]
        h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
        h = h.permute(0, 3, 2, 1)  # [B, C, H, W]

        # ===== STEP 3: UPSAMPLE THROUGH GBLOCKS =====
        len_scale = 1
        x_lens = y_lens * self.bottom_width  # Track width at each resolution
        
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, layers.Attention):
                    # Self-attention: captures long-range dependencies
                    h = block(h, x_lens=x_lens * len_scale)
                else:
                    # GBlock: upsample + conditional batch norm with style
                    h = block(h, y=ys[index])
            len_scale *= self.arch['upsample'][index][1]  # Update width scale

        # ===== STEP 4: OUTPUT LAYER =====
        # BN → ReLU → Conv → Tanh (output range [-1, 1])
        output = torch.tanh(self.output_layer(h))

        # ===== STEP 5: MASK PADDING (during inference) =====
        # Zero out pixels beyond actual text length
        if not self.training:
            out_lens = y_lens * output.size(-2) // 2
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32).to(z.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)  # Padding → -1 (white)

        return output

    def _info_attention(self):
        """Get attention maps for visualization/debugging."""
        attn_index = -1
        for index in range(len(self.arch['out_channels'])):
            if self.arch['attention'][self.arch['resolution'][index]]:
                attn_index = index

        if attn_index == -1:
            return []

        attn_layer = self.blocks[attn_index][-1]
        out = []
        for l in [attn_layer.attn1, attn_layer.attn2]:
            out.append({'out': l._vis_out, 'gamma': l.gamma.item()})
        return out


# =============================================================================
# DISCRIMINATOR ARCHITECTURE CONFIGURATION
# =============================================================================
def D_arch(ch=64, attention='64', input_nc=3):
    """
    Define Discriminator architecture for different resolutions.
    
    The discriminator progressively downsamples:
        64xW → 32xW/2 → 16xW/4 → 8xW/8
    """
    arch = {}

    arch[32] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4]],
                'out_channels': [item * ch for item in [1, 2, 4, 4]],
                'downsample': [True] * 3 + [False],
                'resolution': [8, 4, 4, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 5)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 1, 2, 2, 4, 4]],
                'out_channels': [item * ch for item in [1, 1, 2, 2, 4, 4, 4]],
                'downsample': [False, True, False, True, False, True, False],
                'resolution': [8, 8, 4, 4, 4, 4, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 9)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4]],
               'out_channels': [item * ch for item in [1, 2, 4, 4]],
               'downsample': [True] * 3 + [False],
               'resolution': [32, 16, 8, 8],
               'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 7)}}

    return arch


# =============================================================================
# DISCRIMINATOR: Image → Real/Fake Score
# =============================================================================
class Discriminator(nn.Module):
    """
    BigGAN-style Discriminator for handwriting images.
    
    PURPOSE: Judge whether an image is real (from dataset) or fake (from Generator).
    Trained adversarially with the Generator.
    
    ARCHITECTURE:
        Input: [B, 1, 64, W] grayscale handwriting image
        
        Processing:
            1. DBlocks with spectral norm (downsample 2x each)
            2. Self-attention at resolution 64
            3. Global sum pooling (handle variable widths)
            4. Linear projection to scalar
        
        Output: [B, 1] real/fake score (higher = more real)
    
    LOSS (Hinge Loss):
        D_loss = E[ReLU(1 - D(real))] + E[ReLU(1 + D(fake))]
        
        Meaning:
        - D(real) should be > 1 (no loss)
        - D(fake) should be < -1 (no loss)
    
    SPECTRAL NORMALIZATION:
        - Constrains the Lipschitz constant of each layer
        - Prevents discriminator from becoming too powerful
        - More stable training than weight clipping or gradient penalty
    """
    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_class=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, output_dim=1, D_fp16=False,
                 init='ortho', D_param='SN', bn_linear='embed', input_nc=3, one_hot=False):
        super(Discriminator, self).__init__()
        self.name = 'D'
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_class
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        # self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if self.init != 'none':
            self = init_weights(self, self.init)

    def forward(self, x, x_lens=None, y_lens=None,  **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        len_scale = 1
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, x_len=x_lens // len_scale if x_lens is not None else None)
            len_scale *= 2 if self.arch['downsample'][index] else 1
        # Apply global sum pooling as in SN-GAN
        if x_lens is None:
            h = torch.sum(self.activation(h), [2, 3])
        else:
            h = self.activation(h)
            h_lens = x_lens * h.size(-1) // (x.size(-1) + 1e-8)
            mask = _len2mask(h_lens.int(), h.size(-1), torch.float32).to(x.device).detach()
            mask = mask.view(mask.size(0), 1, 1, mask.size(1))
            h = torch.sum(h * mask, [2, 3])
            h = h / y_lens.unsqueeze(dim=-1)

        # Get initial class-unconditional output
        out = self.linear(h)

        return out


# =============================================================================
# IMPROVED GENERATOR with all architectural enhancements
# =============================================================================
class ImprovedGenerator(nn.Module):
    """
    HiGAN+ Generator with architectural improvements:
    1. Transformer encoder for text (replaces linear projection)
    2. BiGRU for sequential modeling
    3. Cross-attention for style-content fusion
    4. AdaIN-based GBlocks (replaces conditional batch norm)
    5. Positional encoding for character positions
    6. Multi-scale style skip connections
    """
    
    def __init__(self, G_ch=64, style_dim=32, embed_dim=120,
                 bottom_width=4, bottom_height=4, resolution=64,
                 G_kernel_size=3, G_attn='32_64', n_class=80,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, G_fp16=False,
                 init='ortho', G_param='SN', input_nc=1,
                 embed_pad_idx=0, embed_max_norm=1.0,
                 # New parameters for improvements
                 use_transformer=True,
                 use_bigru=True,
                 use_cross_attention=True,
                 use_adain=True,
                 transformer_layers=2,
                 transformer_heads=4,
                 bigru_layers=1,
                 max_text_len=50
                 ):
        super().__init__()

        self.name = 'ImprovedG'
        self.style_dim = style_dim
        self.ch = G_ch
        self.embed_dim = embed_dim
        self.bottom_width = bottom_width
        self.bottom_height = bottom_height
        self.resolution = resolution
        self.n_classes = n_class
        self.activation = G_activation
        self.init = init
        self.fp16 = G_fp16
        
        # Flags for improvements
        self.use_transformer = use_transformer
        self.use_bigru = use_bigru
        self.use_cross_attention = use_cross_attention
        self.use_adain = use_adain
        
        # Architecture
        self.arch = G_arch(self.ch, G_attn)[resolution]
        
        # Text embedding with positional encoding
        self.text_embedding = nn.Embedding(n_class, embed_dim,
                                           padding_idx=embed_pad_idx,
                                           max_norm=embed_max_norm)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_text_len, dropout=0.1)
        
        # Convolution type
        if G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
        
        # === Improvement 2: Transformer Encoder for Text ===
        combined_dim = embed_dim + style_dim
        if self.use_transformer:
            self.text_transformer = TextTransformerEncoder(
                embed_dim=combined_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                ff_dim=combined_dim * 4,
                dropout=0.1,
                max_len=max_text_len
            )
        
        # === Improvement 7: BiGRU for Sequential Modeling ===
        if self.use_bigru:
            self.bigru = BiGRUEncoder(
                input_dim=combined_dim,
                hidden_dim=combined_dim,
                num_layers=bigru_layers,
                dropout=0.0
            )
        
        # === Improvement 1: Cross-Attention for Style-Content Fusion ===
        if self.use_cross_attention:
            self.cross_attention = StyleContentCrossAttention(
                content_dim=combined_dim,
                style_dim=style_dim,
                num_heads=4,
                dropout=0.1
            )
        
        # Linear projection to initial feature map
        self.filter_linear = self.which_linear(
            combined_dim,
            self.arch['in_channels'][0] * (bottom_width * bottom_height)
        )
        
        # Style linear for per-block style codes
        self.style_linear = self.which_linear(
            style_dim,
            style_dim * len(self.arch['in_channels'])
        )
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        for index in range(len(self.arch['out_channels'])):
            in_ch = self.arch['in_channels'][index]
            out_ch = self.arch['out_channels'][index]
            upsample = functools.partial(F.interpolate, scale_factor=self.arch['upsample'][index])
            
            # === Improvement 3: AdaIN-based GBlock ===
            if self.use_adain:
                block = layers.AdaINGBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    style_dim=style_dim,
                    which_conv=self.which_conv,
                    activation=self.activation,
                    upsample=upsample
                )
            else:
                # Fallback to original GBlock with ccbn
                which_bn = functools.partial(layers.ccbn,
                                             which_linear=nn.Linear,
                                             input_size=style_dim,
                                             norm_style='bn')
                block = layers.GBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    which_conv1=self.which_conv,
                    which_conv2=self.which_conv,
                    which_bn=which_bn,
                    activation=self.activation,
                    upsample=upsample
                )
            
            self.blocks.append(block)
            
            # Add attention at specified resolutions
            if self.arch['attention'][self.arch['resolution'][index]]:
                print(f'Adding attention layer in ImprovedG at resolution {self.arch["resolution"][index]}')
                self.blocks.append(layers.MultiHeadSelfAttention(out_ch, num_heads=4, which_conv=self.which_conv))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.InstanceNorm2d(self.arch['out_channels'][-1]),
            self.activation,
            self.which_conv(self.arch['out_channels'][-1], input_nc)
        )
        
        # Initialize weights
        if self.init != 'none':
            init_weights(self, self.init)
    
    def forward(self, z, y, y_lens):
        """
        Args:
            z: [B, style_dim] style vector
            y: [B, seq_len] text indices
            y_lens: [B] text lengths
        Returns:
            [B, 1, H, W] generated image
        """
        batch_size = z.size(0)
        seq_len = y.size(1)
        
        # Split style for each block
        styles = self.style_linear(z).split(self.style_dim, dim=1)
        
        # Text embedding with positional encoding
        y_emb = self.text_embedding(y).float()  # [B, L, embed_dim]
        y_emb = self.pos_encoding(y_emb)
        
        # Expand style and concatenate with text
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, style_dim]
        combined = torch.cat([z_expanded, y_emb], dim=2)  # [B, L, embed_dim + style_dim]
        
        # Create padding mask for transformer/bigru
        padding_mask = ~(_len2mask(y_lens, seq_len).bool())  # True for padded positions
        
        # Apply transformer encoder
        if self.use_transformer:
            combined = self.text_transformer(combined, src_key_padding_mask=padding_mask)
        
        # Apply BiGRU
        if self.use_bigru:
            combined = self.bigru(combined, y_lens)
        
        # Apply cross-attention with style
        if self.use_cross_attention:
            combined = self.cross_attention(combined, z)
        
        # Project to initial feature map
        h = self.filter_linear(combined)  # [B, L, C * bottom_h * bottom_w]
        
        # Reshape to spatial: [B, C, bottom_h, L * bottom_w]
        h = h.view(batch_size, seq_len * self.bottom_width, self.bottom_height, -1)
        h = h.permute(0, 3, 2, 1)  # [B, C, bottom_h, L * bottom_w]
        
        # Process through blocks
        style_idx = 0
        for block in self.blocks:
            if isinstance(block, (layers.AdaINGBlock, layers.GBlock)):
                if self.use_adain:
                    h = block(h, styles[style_idx])
                else:
                    h = block(h, y=styles[style_idx])
                style_idx += 1
            elif isinstance(block, layers.MultiHeadSelfAttention):
                h = block(h)
            else:
                h = block(h)
        
        # Output layer
        output = torch.tanh(self.output_layer(h))
        
        # Mask padding during inference
        if not self.training:
            out_lens = y_lens * output.size(-1) // seq_len
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32).to(z.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)
        
        return output


class PatchDiscriminator(Discriminator):
    def __init__(self, *args, **kwargs):
        super(PatchDiscriminator, self).__init__(*args, **kwargs)


# Defines the PatchGAN discriminator with the specified arguments
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538.
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, kernel_size=3, norm_layer=nn.Identity, sn=True,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-12):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.sn = sn
        self.SN_eps = SN_eps
        if self.sn:
            self.which_conv = functools.partial(layers.SNConv2d,
                                                padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)

        kw = kernel_size
        padw = 1
        sequence = [self.which_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.ReLU(inplace=False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                self.which_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                # norm_layer(ndf * nf_mult),
                nn.ReLU(inplace=False)
            ]

        nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     self.which_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
        #     # norm_layer(ndf * nf_mult),
        #     nn.ReLU(inplace=False)
        # ]

        sequence += [self.which_conv(nf_mult * ndf, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, x_lens, y_lens):
        """Standard forward."""
        h = self.model(x)
        h_lens = x_lens * h.size(-1) // (x.size(-1) + 1e-8)
        mask = _len2mask(h_lens.int(), h.size(-1), torch.float32).to(x.device).detach()
        mask = mask.view(mask.size(0), 1, 1, mask.size(1))
        h = torch.sum(h * mask, [2, 3])
        h = h / y_lens.unsqueeze(dim=-1)
        return h
