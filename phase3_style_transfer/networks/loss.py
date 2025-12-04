"""
=============================================================================
LOSS FUNCTIONS FOR HiGAN+ HANDWRITING GENERATION
=============================================================================

This file contains all the loss functions used in training:

1. RECONSTRUCTION LOSSES:
   - recn_l1_loss: Pixel-wise L1 loss for reconstruction
   - tv_loss: Total Variation loss for smoothness
   - calc_loss_perceptual: Multi-scale perceptual loss

2. REGULARIZATION LOSSES:
   - r1_reg: R1 gradient penalty for discriminator
   - KLloss: KL divergence for VAE latent space

3. STYLE LOSSES:
   - CXLoss: Contextual loss for feature matching
   - GramStyleLoss: Gram matrix style matching
   - ContrastiveStyleLoss: InfoNCE for writer clustering

The total Generator loss is:
    L_G = L_adv + 3.0*L_ctc + 1.5*L_wid + 5.0*L_recn + λ_ctx*L_cx + λ_kl*L_kl
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# UTILITY: Length to Mask Conversion
# =============================================================================
def _len2mask(length, max_len, dtype=torch.float32):
    """
    Convert sequence lengths to binary mask.
    
    WHY NEEDED: Images have variable widths (different word lengths).
    This mask zeros out padded regions so losses only consider valid pixels.
    
    Args:
        length: [B] tensor of actual lengths
        max_len: Maximum sequence length
        dtype: Output dtype
    
    Returns:
        mask: [B, max_len] binary mask (1 for valid, 0 for padding)
    
    Example:
        lengths = [3, 5, 2], max_len = 5
        mask = [[1,1,1,0,0],
                [1,1,1,1,1],
                [1,1,0,0,0]]
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


# =============================================================================
# R1 GRADIENT PENALTY (Discriminator Regularization)
# =============================================================================
def r1_reg(d_out, x_in):
    """
    R1 zero-centered gradient penalty for real images.
    
    PURPOSE: Prevents discriminator from becoming too confident.
    Penalizes large gradients of D w.r.t. real images.
    
    FORMULA: R1 = 0.5 * E[||∇D(x_real)||²]
    
    Used in: Discriminator training (optional regularization)
    """
    batch_size = x_in.size(0)
    # Compute gradients of discriminator output w.r.t. input
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


# =============================================================================
# TOTAL VARIATION LOSS (Smoothness Regularization)
# =============================================================================
def tv_loss(img, img_lens):
    """
    Total Variation loss for image smoothness.
    
    PURPOSE: Reduces noise/artifacts by penalizing high-frequency changes.
    Encourages neighboring pixels to have similar values.
    
    FORMULA: TV = |img[x+1,y] - img[x,y]| + |img[x,y+1] - img[x,y]|
    
    Used in: Optional regularization during generator training
    """
    # Horizontal difference + Vertical difference
    loss = (recn_l1_loss(img[:, :, 1:, :], img[:, :, :-1, :], img_lens) +
            recn_l1_loss(img[:, :, :, 1:], img[:, :, :, :-1], img_lens - 1)) / 2
    return loss


# =============================================================================
# RECONSTRUCTION L1 LOSS (Primary Pixel Loss)
# =============================================================================
def recn_l1_loss(img1, img2, img_lens):
    """
    Masked L1 reconstruction loss.
    
    PURPOSE: Measures pixel-wise difference between generated and real images.
    The mask ensures we only compare valid (non-padded) regions.
    
    FORMULA: L1 = mean(|img1 - img2| * mask)
    
    WEIGHT IN TOTAL LOSS: 5.0 (highest weight = most important)
    
    Args:
        img1: Generated image [B, 1, H, W]
        img2: Real image [B, 1, H, W]
        img_lens: Actual widths [B]
    
    Used in: Generator training for reconstruction path
    """
    # Create mask for valid pixels only
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    # Apply mask to difference image
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    # Normalize by actual number of pixels
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * img_lens.sum())
    return loss


# =============================================================================
# PERCEPTUAL LOSS (Multi-Scale Feature Matching)
# =============================================================================
def calc_loss_perceptual(hout, hgt, img_lens):
    """
    Multi-scale perceptual loss using intermediate features.
    
    PURPOSE: Compares high-level features, not just pixels.
    Captures structural similarity at different scales.
    
    Args:
        hout: List of feature maps from generated image [scale1, scale2, scale3]
        hgt: List of feature maps from real image
        img_lens: Image lengths for masking
    
    Used in: Optional perceptual similarity objective
    """
    loss = 0
    for j in range(3):
        scale = 2 ** (3 - j)  # 8, 4, 2 for different resolutions
        loss += recn_l1_loss(hout[j], hgt[j], img_lens // scale) / scale
    return loss


# =============================================================================
# GRAM MATRIX (Style Representation)
# =============================================================================
def gram_matrix(feat):
    """
    Compute Gram matrix for style representation.
    
    PURPOSE: Captures texture/style by computing feature correlations.
    Two images with similar Gram matrices have similar style.
    
    FORMULA: G = F × F^T / (C × H × W)
    
    Reference: Neural Style Transfer (Gatys et al.)
    """
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)       # Flatten spatial dims
    feat_t = feat.transpose(1, 2)         # Transpose for matrix multiply
    gram = torch.bmm(feat, feat_t) / (ch * h * w)  # Normalized Gram matrix
    return gram


# =============================================================================
# KL DIVERGENCE LOSS (VAE Regularization)
# =============================================================================
def KLloss(mu, logvar):
    """
    KL Divergence between learned latent distribution and standard normal.
    
    PURPOSE: Regularizes the VAE latent space (StyleEncoder).
    Forces the encoded style distribution to be close to N(0,1).
    This enables smooth interpolation between styles.
    
    FORMULA: KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    
    Args:
        mu: Mean of latent distribution [B, style_dim]
        logvar: Log variance of latent distribution [B, style_dim]
    
    WEIGHT IN TOTAL LOSS: λ_kl (typically 0.01-0.1)
    
    Used in: Generator training when vae_mode=True
    """
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


##############################################################################
# CONTRASTIVE STYLE LOSS (InfoNCE)
##############################################################################
class ContrastiveStyleLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for style learning.
    
    PURPOSE: Learns a style space where:
    - Same writer → similar style vectors (pulled together)
    - Different writers → dissimilar vectors (pushed apart)
    
    HOW IT WORKS:
    1. Normalize all style vectors to unit sphere
    2. Compute pairwise cosine similarities
    3. For each sample, maximize similarity to same-writer samples
       relative to all other samples
    
    FORMULA: L = -log(Σ exp(sim(i,j)/τ) for j∈same_writer) / (Σ exp(sim(i,k)/τ) for all k)
    
    Temperature (τ): Lower = sharper distinctions, higher = softer
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature  # Controls sharpness of similarity
    
    def forward(self, style_vectors, writer_ids):
        """
        Args:
            style_vectors: [B, D] style vectors from StyleEncoder
            writer_ids: [B] writer ID for each sample (0-371 for IAM)
        Returns:
            scalar loss (lower = better clustering by writer)
        """
        batch_size = style_vectors.size(0)
        device = style_vectors.device
        
        # Step 1: Normalize style vectors to unit sphere
        style_vectors = F.normalize(style_vectors, dim=1)
        
        # Step 2: Compute similarity matrix [B, B]
        # sim[i,j] = cosine similarity between sample i and j
        sim_matrix = torch.matmul(style_vectors, style_vectors.T) / self.temperature
        
        # Step 3: Create positive mask (same writer = positive pair)
        writer_ids = writer_ids.view(-1, 1)
        positive_mask = (writer_ids == writer_ids.T).float()
        
        # Remove self-similarity from positives (diagonal)
        positive_mask.fill_diagonal_(0)
        
        # Mask out self-similarity for denominator
        logits_mask = torch.ones_like(sim_matrix)
        logits_mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        
        # Sum of positive similarities (same writer)
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sum = exp_sim.sum(dim=1)
        
        # InfoNCE loss: -log(positive / all)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute for samples with positive pairs in batch
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


##############################################################################
# MULTI-SCALE PERCEPTUAL LOSS (Learned Weights)
##############################################################################
class MultiScalePerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss with learnable scale weights.
    
    PURPOSE: Compares features at multiple resolutions.
    The network learns which scales are most important.
    
    Scales: Usually 3 levels (8x8, 16x16, 32x32)
    """
    
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # Learnable weights for each scale (trained with the model)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
    
    def forward(self, fake_feats, real_feats, img_lens):
        """
        Args:
            fake_feats: List of feature maps from generated image
            real_feats: List of feature maps from real image
            img_lens: [B] image lengths for masking
        """
        # Softmax ensures weights sum to 1
        weights = F.softmax(self.scale_weights, dim=0)
        loss = 0
        
        for i, (fake_feat, real_feat) in enumerate(zip(fake_feats[:self.num_scales], 
                                                        real_feats[:self.num_scales])):
            scale = 2 ** (self.num_scales - i)
            scale_loss = recn_l1_loss(fake_feat, real_feat, img_lens // scale)
            loss += weights[i] * scale_loss
        
        return loss


##############################################################################
# CONTEXTUAL LOSS (Feature Matching for Style Transfer)
##############################################################################
class CXLoss(nn.Module):
    """
    Contextual Loss for style/feature matching.
    
    PURPOSE: Measures similarity between feature distributions rather than
    pixel-by-pixel matching. More robust to misalignment.
    
    HOW IT WORKS:
    1. Extract patches from both real and generated features
    2. For each generated patch, find best matching real patch
    3. Use soft-matching (not argmax) for differentiability
    
    KEY INSIGHT: Good for style transfer because it doesn't require
    exact spatial correspondence - just similar features somewhere.
    
    Reference: "The Contextual Loss for Image Transformation" (Mechrez et al.)
    
    Args:
        sigma: Controls softness of matching (lower = sharper)
        b: Baseline offset for distance calculation
    """
    def __init__(self, sigma=0.5, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        """Center features by target mean (domain normalization)."""
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        """Normalize features along channel dimension for cosine similarity."""
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        """Convert feature map to patches for convolution-based matching."""
        N, C, H, W = features.shape
        assert N == 1
        P = H * W  # Number of patches
        # Reshape: NCHW --> HWxCx1x1 (patches as conv kernels)
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        """Compute relative distances (normalize by minimum)."""
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)
        CX = torch.mean(CX.max(dim=3)[0].max(dim=2)[0], dim=1)
        CX = torch.mean(-torch.log(CX + 1e-5))
        return CX



##############################################################################
# Gram style loss
##############################################################################
class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input_feat, target_feat, feat_len=None):
        input_gram = self.gram(input_feat, feat_len)
        target_gram = self.gram(target_feat, feat_len)
        loss = self.criterion(input_gram, target_gram)
        return loss


class GramMatrix(nn.Module):
    def forward(self, input, feat_len=None):
        a, b, c, d = input.size()

        if feat_len is not None:
            # mask for varying lengths
            mask = _len2mask(feat_len, d).view(a, 1, 1, d)
            input = input * mask

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        return G.div(a * b * c * d)
