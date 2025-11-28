"""
Improved layers for HiGAN+ architecture.
Implements:
- AdaIN (Adaptive Instance Normalization)
- Modulated Convolution (StyleGAN2-style)
- Multi-Head Cross-Attention
- Positional Encoding
- Transformer Encoder for text
- Contrastive Style Loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P


# =============================================================================
# 1. Positional Encoding
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling."""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 2. Adaptive Instance Normalization (AdaIN)
# =============================================================================
class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Modulates features using style-derived scale and shift.
    """
    
    def __init__(self, num_features, style_dim, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Style to scale and shift
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        
        # Initialize
        nn.init.ones_(self.style_scale.weight)
        nn.init.zeros_(self.style_scale.bias)
        nn.init.zeros_(self.style_shift.weight)
        nn.init.zeros_(self.style_shift.bias)
    
    def forward(self, x, style):
        """
        Args:
            x: [B, C, H, W] feature maps
            style: [B, style_dim] style vector
        """
        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        x_norm = (x - mean) / std
        
        # Style modulation
        scale = self.style_scale(style).view(x.size(0), -1, 1, 1)
        shift = self.style_shift(style).view(x.size(0), -1, 1, 1)
        
        return x_norm * (1 + scale) + shift


# =============================================================================
# 3. Modulated Convolution (StyleGAN2-style)
# =============================================================================
class ModulatedConv2d(nn.Module):
    """
    StyleGAN2-style modulated convolution.
    Modulates conv weights based on style vector.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, style_dim,
                 stride=1, padding=0, demodulate=True, eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate
        self.eps = eps
        
        # Conv weight [out_ch, in_ch, kH, kW]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        
        # Style modulation
        self.style_mod = nn.Linear(style_dim, in_channels)
        nn.init.ones_(self.style_mod.weight)
        nn.init.zeros_(self.style_mod.bias)
    
    def forward(self, x, style):
        """
        Args:
            x: [B, C_in, H, W]
            style: [B, style_dim]
        Returns:
            [B, C_out, H', W']
        """
        batch_size = x.size(0)
        
        # Get style modulation [B, C_in]
        style_mod = self.style_mod(style)
        
        # Modulate weights: [B, out_ch, in_ch, kH, kW]
        weight = self.weight.unsqueeze(0) * style_mod.view(batch_size, 1, -1, 1, 1)
        
        # Demodulate (normalize by output std)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch_size, -1, 1, 1, 1)
        
        # Reshape for group conv
        weight = weight.view(
            batch_size * self.out_channels,
            self.in_channels,
            *self.kernel_size
        )
        
        # Group conv (each sample has its own weights)
        x = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))
        out = F.conv2d(x, weight, padding=self.padding, stride=self.stride,
                       groups=batch_size)
        out = out.view(batch_size, self.out_channels, out.size(2), out.size(3))
        
        return out


class ModulatedConvBlock(nn.Module):
    """Modulated conv with activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, style_dim,
                 stride=1, padding=1, upsample=False, activation='lrelu'):
        super().__init__()
        self.upsample = upsample
        
        self.conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, style_dim,
            stride=stride, padding=padding
        )
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x, style):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x, style)
        x = self.activation(x)
        return x


# =============================================================================
# 4. Multi-Head Cross-Attention
# =============================================================================
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention for style-content interaction.
    Query: text features, Key/Value: style features
    """
    
    def __init__(self, query_dim, key_dim, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [B, L_q, D_q] (text features)
            key: [B, L_k, D_k] (style features)  
            value: [B, L_k, D_k] (style features)
            key_padding_mask: [B, L_k] True for positions to mask
        Returns:
            [B, L_q, D_q]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape to [B, num_heads, L, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores [B, num_heads, L_q, L_k]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, L_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class StyleContentCrossAttention(nn.Module):
    """
    Cross-attention block with residual connection and layer norm.
    Integrates style information into content features.
    """
    
    def __init__(self, content_dim, style_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(content_dim)
        self.norm2 = nn.LayerNorm(content_dim)
        
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=content_dim,
            key_dim=style_dim,
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(content_dim, content_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim * 4, content_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, content, style, style_mask=None):
        """
        Args:
            content: [B, L, D_content] text features
            style: [B, D_style] or [B, 1, D_style] style vector
        """
        if style.dim() == 2:
            style = style.unsqueeze(1)  # [B, 1, D_style]
        
        # Cross-attention
        content = content + self.cross_attn(
            self.norm1(content), style, style, style_mask
        )
        
        # FFN
        content = content + self.ffn(self.norm2(content))
        
        return content


# =============================================================================
# 5. Transformer Encoder for Text
# =============================================================================
class TextTransformerEncoder(nn.Module):
    """
    Transformer encoder for text sequence modeling.
    Replaces simple linear projection with contextual encoding.
    """
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, 
                 ff_dim=512, dropout=0.1, max_len=50):
        super().__init__()
        
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: [B, L, D] embedded text
            src_key_padding_mask: [B, L] True for padded positions
        Returns:
            [B, L, D] contextualized embeddings
        """
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x


# =============================================================================
# 6. BiGRU for Sequential Modeling
# =============================================================================
class BiGRUEncoder(nn.Module):
    """Bidirectional GRU for sequence encoding."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, L, D]
            lengths: [B] sequence lengths
        Returns:
            [B, L, D]
        """
        if lengths is not None:
            # Pack for efficient computation
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return output


# =============================================================================
# 7. Contrastive Style Loss
# =============================================================================
class ContrastiveStyleLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for style learning.
    Encourages same-writer samples to be close, different writers to be far.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, style_vectors, writer_ids):
        """
        Args:
            style_vectors: [B, D] normalized style vectors
            writer_ids: [B] writer ID for each sample
        Returns:
            scalar loss
        """
        batch_size = style_vectors.size(0)
        device = style_vectors.device
        
        # Normalize style vectors
        style_vectors = F.normalize(style_vectors, dim=1)
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(style_vectors, style_vectors.T) / self.temperature
        
        # Create positive mask (same writer = 1, different writer = 0)
        writer_ids = writer_ids.view(-1, 1)
        positive_mask = (writer_ids == writer_ids.T).float()
        
        # Remove self-similarity from positives
        positive_mask.fill_diagonal_(0)
        
        # For each anchor, compute loss
        # Numerator: sum of exp(sim) for positive pairs
        # Denominator: sum of exp(sim) for all pairs except self
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(sim_matrix)
        logits_mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        
        # Sum of positive similarities
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sum = exp_sim.sum(dim=1)
        
        # Avoid division by zero
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute loss for samples that have positive pairs
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


# =============================================================================
# 8. Multi-Scale Feature Fusion
# =============================================================================
class MultiScaleStyleFusion(nn.Module):
    """
    Fuses multi-scale style features with skip connections.
    """
    
    def __init__(self, style_channels, target_channels):
        """
        Args:
            style_channels: list of channel dims from style backbone [c1, c2, c3]
            target_channels: list of channel dims in generator [c1', c2', c3']
        """
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sc, tc, 1, bias=False),
                nn.InstanceNorm2d(tc),
                nn.LeakyReLU(0.2, inplace=False)
            )
            for sc, tc in zip(style_channels, target_channels)
        ])
    
    def forward(self, style_feats, target_feats):
        """
        Args:
            style_feats: list of [B, C, H, W] from style encoder
            target_feats: list of [B, C', H', W'] from generator
        Returns:
            list of fused features
        """
        fused = []
        for adapter, sf, tf in zip(self.adapters, style_feats, target_feats):
            # Resize style feature to match target
            sf_resized = F.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
            sf_adapted = adapter(sf_resized)
            fused.append(tf + 0.1 * sf_adapted)  # Weighted residual
        return fused


# =============================================================================
# 9. Modulated Generator Block (combines AdaIN + ModulatedConv)
# =============================================================================
class ModulatedGBlock(nn.Module):
    """
    Generator block with modulated convolutions and AdaIN.
    Replaces standard GBlock for better style control.
    """
    
    def __init__(self, in_channels, out_channels, style_dim, upsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        
        # Modulated convolutions
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim, padding=1)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim, padding=1)
        
        # AdaIN for normalization
        self.adain1 = AdaIN(out_channels, style_dim)
        self.adain2 = AdaIN(out_channels, style_dim)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        
        # Skip connection
        self.learnable_sc = in_channels != out_channels
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x, style):
        """
        Args:
            x: [B, C_in, H, W]
            style: [B, style_dim]
        """
        # Shortcut
        if self.upsample:
            x_up = self.upsample(x)
        else:
            x_up = x
        
        if self.learnable_sc:
            shortcut = self.conv_sc(x_up)
        else:
            shortcut = x_up
        
        # Main path
        h = x
        if self.upsample:
            h = self.upsample(h)
        
        h = self.conv1(h, style)
        h = self.adain1(h, style)
        h = self.activation(h)
        
        h = self.conv2(h, style)
        h = self.adain2(h, style)
        h = self.activation(h)
        
        return h + shortcut


# =============================================================================
# 10. Improved Self-Attention with relative position
# =============================================================================
class ImprovedSelfAttention(nn.Module):
    """Self-attention with relative positional encoding."""
    
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        self.query = nn.Conv2d(in_dim, in_dim, 1)
        self.key = nn.Conv2d(in_dim, in_dim, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.out = nn.Conv2d(in_dim, in_dim, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, x_lens=None):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, self.num_heads, self.head_dim, -1)  # [B, heads, dim, HW]
        k = self.key(x).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(x).view(B, self.num_heads, self.head_dim, -1)
        
        # Attention
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(v, attn.transpose(-2, -1))  # [B, heads, dim, HW]
        out = out.view(B, C, H, W)
        out = self.out(out)
        
        return self.gamma * out + x
