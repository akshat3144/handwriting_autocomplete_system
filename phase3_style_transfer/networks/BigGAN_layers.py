"""
=============================================================================
BIGGAN LAYERS FOR HiGAN+ HANDWRITING GENERATION
=============================================================================

This file contains the core building blocks for the GAN:

1. SPECTRAL NORMALIZATION:
   - SNConv2d, SNLinear, SNEmbedding
   - Stabilizes discriminator by constraining weight matrices
   - Uses power iteration to estimate largest singular value

2. GENERATOR BLOCKS:
   - GBlock: Residual block with conditional batch norm (injects style)
   - ccbn: Class-conditional batch norm (modulates features with style)
   - AdaINGBlock: Alternative with AdaIN (StyleGAN-style)

3. DISCRIMINATOR BLOCKS:
   - DBlock: Residual block with spectral norm + downsampling

4. ATTENTION:
   - SelfAttention: Captures long-range dependencies in feature maps
   - MultiHeadSelfAttention: Multi-head version for better attention

5. NORMALIZATION:
   - bn: Standard batch normalization
   - myBN: Custom batch norm with standing stats
   - fused_bn: Fused batch norm operation for efficiency

KEY CONCEPTS:
- Spectral Norm: ||W||_2 ≤ 1 (Lipschitz constraint)
- Conditional BN: γ = f(z), β = g(z) where z is style
- Self-Attention: Allows each position to attend to all others
=============================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from .utils import _len2mask
import matplotlib.pyplot as plt

# Import improved layers (AdaIN, Modulated Conv, etc.)
from .improved_layers import (
    AdaIN, ModulatedConv2d, ModulatedGBlock, 
    MultiHeadCrossAttention, StyleContentCrossAttention,
    ImprovedSelfAttention, BiGRUEncoder,
    SinusoidalPositionalEncoding, TextTransformerEncoder
)


# =============================================================================
# HELPER FUNCTIONS FOR SPECTRAL NORMALIZATION
# =============================================================================

def proj(x, y):
    """Project vector x onto vector y."""
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """
    Orthogonalize x with respect to list of vectors ys.
    
    Used in power iteration to ensure orthogonal singular vectors.
    """
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """
    Power method to estimate top singular value of weight matrix.
    
    PURPOSE: Spectral normalization divides weights by their largest
    singular value, keeping the Lipschitz constant ≤ 1.
    
    ALGORITHM:
    1. v = W^T @ u (right singular vector)
    2. u = W @ v (left singular vector)  
    3. σ = u^T @ W @ v (singular value)
    
    The method converges to the top singular value/vectors.
    """
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            # Compute right singular vector
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            
            # Compute left singular vector
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            
            if update:
                u_[i][:] = u
        
        # Compute singular value: σ = u^T W v
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


# =============================================================================
# IDENTITY LAYER (Passthrough)
# =============================================================================
class identity(nn.Module):
    """Identity layer - just passes input through unchanged."""
    def forward(self, input):
        return input


# =============================================================================
# SPECTRAL NORMALIZATION BASE CLASS
# =============================================================================
class SN(object):
    """
    Spectral Normalization base class.
    
    PURPOSE: Constrain the spectral norm (largest singular value) of weight
    matrices to be ≤ 1. This:
    1. Stabilizes GAN training (prevents discriminator explosion)
    2. Enforces Lipschitz continuity
    3. No hyperparameter tuning needed (unlike gradient penalty)
    
    HOW IT WORKS:
    - Maintain estimates of top singular vectors (u, v)
    - Each forward pass: update u, v via power iteration
    - Divide weight by estimated singular value: W_normalized = W / σ_max
    
    Reference: "Spectral Normalization for GANs" (Miyato et al., 2018)
    """
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values to track (usually just 1)
        self.num_svs = num_svs
        # Transpose for different layer types
        self.transpose = transpose
        # Small constant for numerical stability
        self.eps = eps
        
        # Register buffers for singular vectors (not trainable)
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        """Get all left singular vectors."""
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        """Get all singular values (for logging, not training)."""
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        """
        Compute spectrally-normalized weight matrix.
        
        Returns: W / σ_max (weight divided by largest singular value)
        """
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        
        # Power iteration to estimate σ_max
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        
        # Store singular value for logging
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        
        # Normalize weight by σ_max
        return self.weight / svs[0]


# =============================================================================
# SPECTRAL NORMALIZED LAYERS
# =============================================================================

class SNConv2d(nn.Conv2d, SN):
    """
    2D Convolution with Spectral Normalization.
    
    Used in: Both Generator and Discriminator
    Ensures each conv layer has Lipschitz constant ≤ 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        # Use normalized weights instead of raw weights
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    """
    Linear layer with Spectral Normalization.
    
    Used in: Discriminator output, Generator projections
    """
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):
    """
    Embedding layer with Spectral Normalization.
    
    Used in: Character embeddings in Generator
    Note: Uses num_embeddings as dim (transposed)
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq,
                              sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())

#
# # A non-local block as used in SA-GAN
# # Note that the implementation as described in the paper is largely incorrect;
# # refer to the released code for the actual implementation.
# class Attention(nn.Module):
#     INF_VALUE = 1e8
#     def __init__(self, ch, which_conv=SNConv2d, name='attention'):
#         super(Attention, self).__init__()
#         # Channel multiplier
#         self.ch = ch
#         self.which_conv = which_conv
#         self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
#         self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
#         self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
#         self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
#         # Learnable gain parameter
#         self.gamma = P(torch.tensor(0.), requires_grad=True)
#
#     def forward(self, x, y=None, y_len=None):
#         # Apply convs
#         theta = self.theta(x)
#         phi = F.max_pool2d(self.phi(x), [2, 2])
#         g = F.max_pool2d(self.g(x), [2, 2])
#         # Perform reshapes
#         theta_mask = _len2mask(y_len, theta.shape[3]).view(x.size(0), 1, 1, theta.shape[3])
#         theta = theta * theta_mask + (theta_mask - 1) * self.INF_VALUE
#
#         for i in range(4):
#             print(y_len[i])
#             plt.subplot(4, 1, i + 1)
#             plt.imshow(theta[i, 0].detach().cpu().numpy())
#         plt.show()
#         # try to plot here
#         theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
#         try:
#             phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
#         except:
#             print(phi.shape)
#         g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
#         # Matmul and softmax to get attention maps
#         beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
#         att_map = beta.view(-1, x.shape[2] // 2, x.shape[3] // 2)
#         for i in range(4):
#             print(y_len[i])
#             plt.subplot(4, 1, i + 1)
#             plt.imshow(att_map[i].detach().cpu().numpy())
#         plt.show()
#
#         # Attention map times g path
#         o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
#         return self.gamma * o + x
#
#

# Self Attention module from self-attention gan
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    INF_VALUE = 1e8
    def __init__(self, in_dim, which_conv=SNConv2d):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(which_conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.key_conv = nn.Sequential(which_conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.value_conv = which_conv(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len=None, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # print('attention size', x.size())
        m_batchsize, C, width, height = x.size()
        # print('query_conv size', self.query_conv(x).size())

        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X (N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query.transpose(1, 2), proj_key)  # transpose check

        # if x_len is not None:
        #     mask = _len2mask(x_len, x.shape[3]).unsqueeze(dim=1).repeat(1, x.shape[2], 1)\
        #                                                         .view(m_batchsize, -1).unsqueeze(dim=1)
        #     print('mask', mask.size())
        #     mask_e = torch.bmm(mask.transpose(1, 2), mask)
        #     print('mask_e', mask_e.size())
        #     print('energy', energy.size())
        #     energy = energy * mask_e + (mask_e - 1) * self.INF_VALUE
        #     print(mask_e)
        #     print(energy)

        attention = self.softmax(energy)  # B X (N) X (N)
        # print(attention)
        # if x_len is not None:
        #     for i in range(1):
        #         print(x_len[i])
        #         plt.subplot(1, 1, i + 1)
        #         plt.imshow(attention[i].detach().cpu().numpy() * 1e5)
        #     plt.show()

        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


# from networks.cc_attention import ca_weight, ca_map
# class CrissCrossSelfAttention(nn.Module):
#     """ Criss-Cross Attention Module"""
#     INF_VALUE = 1e8
#     def __init__(self, in_dim, which_conv=SNConv2d):
#         super(CrissCrossSelfAttention,self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = which_conv(in_channels=in_dim, out_channels=in_dim // 8,
#                                      kernel_size=1, padding=0, bias=False)
#         self.key_conv = which_conv(in_channels=in_dim, out_channels=in_dim // 8,
#                                    kernel_size=1, padding=0, bias=False)
#         self.value_conv = which_conv(in_channels=in_dim, out_channels=in_dim,
#                                      kernel_size=1, padding=0, bias=False)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self._vis_out = None
#
#     def forward(self, x, energy_mask=None, **kwargs):
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         proj_value = self.value_conv(x)
#
#         energy = ca_weight(proj_query, proj_key)
#         if energy_mask is not None:
#             energy = energy * energy_mask + (energy_mask - 1) * self.INF_VALUE
#         attention = F.softmax(energy, 1)
#         out = ca_map(attention, proj_value)
#
#         self._vis_out = out[0, 0].detach().cpu().numpy()
#
#         for j in range(2):
#             plt.subplot(411)
#             energy_mask = energy_mask.view(energy.size(0), energy.size(1), -1)
#             plt.imshow(energy_mask.detach().cpu().numpy()[j])
#
#             plt.subplot(412)
#             energy = energy.view(energy.size(0), energy.size(1), -1)
#             plt.imshow(energy.detach().cpu().numpy()[j])
#
#             attention = attention.view(attention.size(0), attention.size(1), -1)
#             plt.subplot(413)
#             plt.imshow(attention.detach().cpu().numpy()[j])
#
#             plt.subplot(414)
#             plt.imshow(out.detach().cpu().numpy()[j, 0])
#             plt.show()
#
#         out = self.gamma*out + x
#         return out
#
#     @staticmethod
#     def calc_energy_mask(x, y_len=None):
#         if y_len is None:
#             energy_mask = ca_weight(seq_mask_expand, None)
#         else:
#             y_len_mask = _len2mask(y_len, x.size(3)).unsqueeze(1).unsqueeze(1)
#             seq_mask_expand = y_len_mask.repeat(1, 1, x.size(2), 1)
#             energy_mask = ca_weight(seq_mask_expand, seq_mask_expand)
#         return energy_mask
#
#
# class DualCrossAttention(nn.Module):
#     def __init__(self, in_dim, which_conv=SNConv2d):
#         super(DualCrossAttention, self).__init__()
#         self.attn1 = CrissCrossSelfAttention(in_dim, which_conv)
#         self.attn2 = CrissCrossSelfAttention(in_dim, which_conv)
#
#     def forward(self, x, y_len=None, **kwargs):
#         energy_mask = CrissCrossSelfAttention.calc_energy_mask(x, y_len)
#         out = self.attn1(x, energy_mask)
#         out = self.attn2(out, energy_mask)
#         return out
#
# Attention = DualCrossAttention

Attention = SelfAttention

# Multi-Head Self-Attention with improved architecture
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for feature maps."""
    
    def __init__(self, in_dim, num_heads=4, which_conv=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        conv_cls = which_conv if which_conv else SNConv2d
        self.query = conv_cls(in_dim, in_dim, kernel_size=1, padding=0)
        self.key = conv_cls(in_dim, in_dim, kernel_size=1, padding=0)
        self.value = conv_cls(in_dim, in_dim, kernel_size=1, padding=0)
        self.out = conv_cls(in_dim, in_dim, kernel_size=1, padding=0)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, x_lens=None, **kwargs):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # [B, heads, HW, HW]
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        
        # Optional masking for variable length
        if x_lens is not None:
            mask = _len2mask(x_lens, W).view(B, 1, 1, W)
            mask = mask.repeat(1, 1, H, 1).view(B, 1, H * W, 1)
            mask = mask * mask.transpose(-2, -1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(B, C, H, W)
        out = self.out(out)
        
        return self.gamma * out + x


# AdaIN-based Generator Block (StyleGAN-style)
class AdaINGBlock(nn.Module):
    """Generator block with AdaIN for style modulation."""
    
    def __init__(self, in_channels, out_channels, style_dim,
                 which_conv=nn.Conv2d, activation=None, upsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.activation = activation if activation else nn.ReLU(inplace=False)
        
        # Convolutions
        self.conv1 = which_conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = which_conv(out_channels, out_channels, kernel_size=3, padding=1)
        
        # AdaIN layers
        self.adain1 = AdaIN(out_channels, style_dim)
        self.adain2 = AdaIN(out_channels, style_dim)
        
        # Skip connection
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = which_conv(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, style, **kwargs):
        """
        Args:
            x: [B, C, H, W] input features
            style: [B, style_dim] style vector
        """
        # Shortcut path
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
        
        h = self.conv1(h)
        h = self.adain1(h, style)
        h = self.activation(h)
        
        h = self.conv2(h)
        h = self.adain2(h, style)
        h = self.activation(h)
        
        return h + shortcut


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift
    # return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = (m2 - m ** 2)
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var', torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# =============================================================================
# CLASS-CONDITIONAL BATCH NORMALIZATION (ccbn)
# =============================================================================
class ccbn(nn.Module):
    """
    Class-Conditional Batch Normalization.
    
    PURPOSE: Inject style information into the Generator.
    Standard BN normalizes, then applies learnable γ and β.
    CCBN makes γ and β FUNCTIONS of the style vector.
    
    FORMULA:
        Standard BN: y = γ * normalize(x) + β
        CCBN:        y = γ(z) * normalize(x) + β(z)
        
        Where z is the 32-dim style vector.
    
    WHY IT WORKS:
    - Different styles → different γ and β
    - This modulates feature statistics based on writing style
    - Similar to AdaIN but with batch normalization
    
    Used in: Generator GBlocks (every upsampling layer)
    """
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn', ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        
        # Linear layers to produce γ and β from style vector
        # Input: style vector [B, 32]
        # Output: per-channel scale [B, C] and shift [B, C]
        self.gain = which_linear(input_size, output_size)  # γ(z)
        self.bias = which_linear(input_size, output_size)  # β(z)
        
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.norm_style = norm_style

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        """
        Args:
            x: [B, C, H, W] feature maps
            y: [B, 32] style vector
        
        Returns:
            [B, C, H, W] modulated features
        """
        # Compute style-dependent scale and shift
        # gain = 1 + γ(z) to make it multiplicative around 1
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                                   self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                      self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


# =============================================================================
# STANDARD BATCH NORMALIZATION (bn)
# =============================================================================
class bn(nn.Module):
    """
    Standard Batch Normalization (not class-conditional).
    
    Used in: Generator output layer
    """
    def __init__(self, output_size, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        # Learnable scale and shift
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


# =============================================================================
# GENERATOR BLOCK (GBlock)
# =============================================================================
class GBlock(nn.Module):
    """
    Generator Residual Block with Conditional Batch Normalization.
    
    PURPOSE: Upsample features while conditioning on style.
    
    ARCHITECTURE:
        Input: [B, C_in, H, W] + style vector
        
        Main path:
            BN(style) → ReLU → Upsample → Conv → BN(style) → ReLU → Conv
        
        Skip path:
            Upsample → 1x1 Conv (if channels change)
        
        Output: main + skip = [B, C_out, 2H, 2W]
    
    KEY: Conditional BN injects style at every layer!
    
    Used in: Generator (4 GBlocks for 4x → 64 upsampling)
    """
    def __init__(self, in_channels, out_channels,
                 which_conv1=nn.Conv2d, which_conv2=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv1, self.which_conv2, self.which_bn = which_conv1, which_conv2, which_bn
        self.activation = activation
        self.upsample = upsample
        
        # Main path convolutions
        self.conv1 = self.which_conv1(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv2(self.out_channels, self.out_channels)
        
        # Skip connection (1x1 conv if channels change)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv1(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        
        # Conditional batch norm layers (take style as input)
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        
        self.upsample = upsample

    def forward(self, x, y, **kwargs):
        """
        Args:
            x: [B, C_in, H, W] input features
            y: [B, 32] style vector for conditional BN
        
        Returns:
            [B, C_out, 2H, 2W] upsampled, style-conditioned features
        """
        # Main path: BN(style) → ReLU → Upsample → Conv
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        
        # Second conv: BN(style) → ReLU → Conv
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        
        # Skip connection
        if self.learnable_sc:
            x = self.conv_sc(x)
        
        return h + x  # Residual connection


# =============================================================================
# DISCRIMINATOR BLOCK (DBlock)
# =============================================================================
class DBlock(nn.Module):
    """
    Discriminator Residual Block with Spectral Normalization.
    
    PURPOSE: Downsample features while maintaining stability.
    
    ARCHITECTURE:
        Input: [B, C_in, H, W]
        
        Main path:
            [ReLU] → Conv → ReLU → Conv → [Downsample]
        
        Skip path:
            [Downsample] → 1x1 Conv (if channels change)
        
        Output: main + skip = [B, C_out, H/2, W/2]
    
    KEY DIFFERENCES FROM GBlock:
    - No batch norm (discriminator shouldn't use BN)
    - Spectral normalized convolutions
    - Downsamples instead of upsamples
    - No style conditioning
    
    Used in: Discriminator (4 DBlocks for 64 → 4 downsampling)
    """
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # Wide D: more channels in hidden layer (as in BigGAN)
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation  # ReLU before first conv?
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x, **kwargs):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


# =============================================================================
# Improved GBlock with Modulated Convolutions (StyleGAN2-style)
# =============================================================================
class ModulatedGBlockV2(nn.Module):
    """
    Advanced Generator block with modulated convolutions.
    Provides finer control over style injection compared to AdaIN.
    """
    
    def __init__(self, in_channels, out_channels, style_dim,
                 which_conv=None, activation=None, upsample=None, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.activation = activation if activation else nn.LeakyReLU(0.2, inplace=False)
        
        # Modulated convolutions
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim, 
                                      padding=1, demodulate=demodulate)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim,
                                      padding=1, demodulate=demodulate)
        
        # Skip connection (not modulated)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x, style, **kwargs):
        # Shortcut
        if self.upsample:
            x_up = self.upsample(x)
        else:
            x_up = x
        
        shortcut = self.conv_sc(x_up) if self.learnable_sc else x_up
        
        # Main path
        h = x
        if self.upsample:
            h = self.upsample(h)
        
        h = self.conv1(h, style)
        h = self.activation(h)
        h = self.conv2(h, style)
        h = self.activation(h)
        
        return h + shortcut


# =============================================================================
# Cross-Attention Block for Style-Content Fusion
# =============================================================================
class StyleContentAttentionBlock(nn.Module):
    """
    Applies cross-attention between content features and style.
    Used in generator to inject style information spatially.
    """
    
    def __init__(self, content_dim, style_dim, num_heads=4):
        super().__init__()
        self.cross_attn = StyleContentCrossAttention(
            content_dim, style_dim, num_heads=num_heads
        )
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, style, **kwargs):
        """
        Args:
            x: [B, C, H, W] feature maps
            style: [B, style_dim] style vector
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: [B, H*W, C]
        x_seq = x.view(B, C, -1).permute(0, 2, 1)
        
        # Apply cross-attention
        out_seq = self.cross_attn(x_seq, style)
        
        # Reshape back: [B, C, H, W]
        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.gamma * out

# dogball