"""
=============================================================================
AUXILIARY NETWORKS FOR HiGAN+ HANDWRITING GENERATION
=============================================================================

This file contains the ENCODER networks and auxiliary models:

1. StyleBackbone:
   - Shared CNN feature extractor for style
   - Used by both StyleEncoder and WriterIdentifier
   - Outputs multi-scale features for style extraction

2. StyleEncoder:
   - Converts a handwriting image → 32-dim style vector
   - Supports VAE mode (samples from distribution) for regularization
   - The style vector captures: slant, thickness, spacing, etc.

3. WriterIdentifier:
   - Classifies which of 372 writers produced an image
   - Used as auxiliary loss to ensure style consistency
   - Frozen during generator training (pretrained)

4. Recognizer (OCR):
   - CNN + BiLSTM for text recognition
   - Uses CTC loss to ensure generated text is readable
   - Frozen during generator training (pretrained)

ARCHITECTURE FLOW:
    Image → StyleBackbone → features → StyleEncoder → style_vector (32-dim)
                          ↘ features → WriterIdentifier → writer_class (372)
                          
    Image → Recognizer → character_probabilities → CTC decode → text
=============================================================================
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights


# =============================================================================
# STYLE BACKBONE: Shared Feature Extractor
# =============================================================================
class StyleBackbone(nn.Module):
    """
    Shared CNN backbone for style extraction.
    
    PURPOSE: Extracts hierarchical visual features from handwriting images.
    These features are shared between StyleEncoder and WriterIdentifier.
    
    ARCHITECTURE:
        Input: [B, 1, 64, W] grayscale handwriting image
        → 5 ResBlocks with progressive downsampling
        → Output: [B, 256, 4, W/16] feature maps
    
    KEY FEATURES:
    - Multi-scale feature extraction (feat2, feat3, feat4)
    - Width-preserving (height reduces, width scales with text length)
    - Captures stroke patterns, textures, spacing
    
    WHY SHARED: Same features useful for both:
    - Style encoding (overall writing style)
    - Writer identification (who wrote this)
    """
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn'):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16  # Width is reduced by factor of 16
        nf = resolution  # Starting number of filters (16)
        
        # ===== INITIAL CONV BLOCK =====
        # Pad with -1 (white in normalized space), then 5x5 conv
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,  # 64→32 height
                             norm='none',
                             activation='none')]
        
        # ===== DOWNSAMPLING BLOCKS (2x) =====
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]  # Pad width only
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]  # Downsample 2x
            nf = min([nf_out, max_dim])

        # ===== DEEPER BLOCKS (2x) =====
        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])
        
        self.cnn_backbone = nn.Sequential(*cnn_f)
        
        # Layer indices for extracting intermediate features
        # Used for contextual loss (feature matching at multiple scales)
        self.layer_name_mapping = {
            '9': "feat2",   # 16x16 resolution features
            '13': "feat3",  # 8x8 resolution features
            '16': "feat4",  # 4x4 resolution features
        }

        # Final processing for width-pooled output
        self.cnn_ctc = nn.Sequential(
            nn.ReLU(),
            Conv2dBlock(df, df, 3, 1, 0,
                        norm=norm,
                        activation='relu')
        )
        if init != 'none':
            init_weights(self, init)

    def forward(self, x, ret_feats=False):
        """
        Args:
            x: [B, 1, 64, W] input image
            ret_feats: If True, return intermediate features for CXLoss
        
        Returns:
            out: [B, 256, W/16] width-wise features (height collapsed)
            feats: List of [feat2, feat3, feat4] if ret_feats=True
        """
        feats = []
        for name, layer in self.cnn_backbone._modules.items():
            x = layer(x)
            if ret_feats and name in self.layer_name_mapping:
                feats.append(x)

        out = self.cnn_ctc(x).squeeze(-2)  # Squeeze height dimension

        return out, feats


# =============================================================================
# STYLE ENCODER: Image → Style Vector
# =============================================================================
class StyleEncoder(nn.Module):
    """
    Encodes a handwriting image into a 32-dimensional style vector.
    
    PURPOSE: Captures the "writing style" - slant, thickness, spacing,
    character shapes, etc. - as a compact vector.
    
    ARCHITECTURE:
        StyleBackbone features [B, 256, W/16]
        → Global average pooling over width
        → MLP: 256 → 256 → 256
        → Linear: 256 → 32 (mu) and 256 → 32 (logvar)
        → Reparameterization: z = mu + eps * exp(0.5 * logvar)
    
    VAE MODE (vae_mode=True):
        - Returns (z, mu, logvar) for KL loss
        - Enables smooth style interpolation
        - z is sampled from N(mu, sigma²)
    
    NON-VAE MODE (vae_mode=False):
        - Returns mu directly (deterministic)
        - Used during inference
    
    The 32-dim style vector is then:
        1. Fed to Generator to condition image synthesis
        2. Used for style transfer: encode ref_image → z → generate new text
    """
    def __init__(self, style_dim=32, in_dim=256, init='N02', use_contrastive=False):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim  # Output dimension (32)
        self.use_contrastive = use_contrastive

        # ===== MLP for style processing =====
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),    # 256 → 256
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),    # 256 → 256
            nn.LeakyReLU(),
        )

        # ===== VAE outputs =====
        self.mu = nn.Linear(in_dim, style_dim)       # Mean of latent distribution
        self.logvar = nn.Linear(in_dim, style_dim)   # Log-variance of latent distribution
        
        # Optional projection head for contrastive learning
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(style_dim, style_dim * 2),
                nn.ReLU(),
                nn.Linear(style_dim * 2, style_dim),
            )
        
        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        """
        Args:
            img: [B, 1, 64, W] input handwriting image
            img_len: [B] actual widths (for masking padding)
            cnn_backbone: StyleBackbone network
            ret_feats: Return intermediate features for CXLoss
            vae_mode: If True, return (z, mu, logvar) for VAE training
        
        Returns:
            style: [B, 32] style vector (or tuple if vae_mode)
            feats: List of feature maps (if ret_feats=True)
        """
        # Step 1: Extract features using shared backbone
        feat, all_feats = cnn_backbone(img, ret_feats)
        
        # Step 2: Create mask for valid positions (handle variable widths)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        
        # Step 3: Global average pooling over width (masked)
        # Only average over valid positions, not padding
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        
        # Step 4: MLP processing
        style = self.linear_style(style)
        
        # Step 5: Get mean (always needed)
        mu = self.mu(style)

        if vae_mode:
            # VAE: Sample from distribution
            logvar = self.logvar(style)
            logvar = torch.clamp(logvar, -10, 2)  # Stability clamp
            style = self.reparameterize(mu, logvar)
            style = (style, mu, logvar)  # Return tuple for KL loss
        else:
            # Non-VAE: Just use mean
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style
    
    def get_contrastive_embedding(self, style):
        """Get normalized embedding for contrastive loss."""
        if isinstance(style, tuple):
            style = style[0]  # Use sampled style if VAE mode
        if self.use_contrastive:
            proj = self.projection_head(style)
            return F.normalize(proj, dim=1)
        return F.normalize(style, dim=1)

    @staticmethod
    def reparameterize(mu, logvar):
        """
        VAE reparameterization trick.
        
        Enables backprop through sampling by:
        z = mu + eps * sigma, where eps ~ N(0,1)
        
        This makes the random sampling a deterministic function of
        (mu, logvar, eps), allowing gradients to flow.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


# =============================================================================
# WRITER IDENTIFIER: Image → Writer Class
# =============================================================================
class WriterIdentifier(nn.Module):
    """
    Classifies handwriting images by writer identity.
    
    PURPOSE: Auxiliary loss to ensure generated images match the
    reference writer's style. If the classifier can identify the
    correct writer from generated images, the style is captured well.
    
    ARCHITECTURE:
        StyleBackbone features [B, 256, W/16]
        → Global average pooling over width
        → MLP: 256 → 256 → 372 (372 writers in IAM dataset)
        → Softmax → writer class probabilities
    
    TRAINING:
        - Pretrained on real IAM data
        - FROZEN during generator training
        - Generator is trained to fool this classifier (match styles)
    
    LOSS (in generator training):
        L_wid = CrossEntropy(WriterIdentifier(generated_img), target_writer_id)
        Weight: 1.5
    """
    def __init__(self, n_writer=372, in_dim=256, init='N02'):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32

        # MLP classifier: features → writer class
        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),    # 256 → 256
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),  # 256 → 372 (number of writers)
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        """
        Args:
            img: [B, 1, 64, W] handwriting image
            img_len: [B] actual widths
            cnn_backbone: StyleBackbone (shared with StyleEncoder)
            ret_feats: Return features for additional losses
        
        Returns:
            wid_logits: [B, 372] logits for each writer class
        """
        # Extract features using shared backbone
        feat, all_feats = cnn_backbone(img, ret_feats)
        
        # Global average pooling (masked for variable lengths)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        
        # Classify writer
        wid_logits = self.linear_wid(wid_feat)
        
        if ret_feats:
            return wid_logits, all_feats
        else:
            return wid_logits

    def return_feat(self, img, img_len):
        """Return intermediate features (for debugging/visualization)."""
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        for j in range(2):
            wid_feat = self.linear_wid[j](wid_feat)
        return wid_feat


# =============================================================================
# RECOGNIZER: OCR Network (Text Recognition)
# =============================================================================
class Recognizer(nn.Module):
    """
    Optical Character Recognition (OCR) network.
    
    PURPOSE: Ensures generated text is READABLE. If the OCR can correctly
    read the generated text, the generator is producing valid handwriting.
    
    ARCHITECTURE:
        Input: [B, 1, 64, W] handwriting image
        → CNN backbone (similar to StyleBackbone)
        → BiLSTM for sequence modeling
        → Linear → 80 character classes (a-z, A-Z, 0-9, punctuation)
        → CTC loss for sequence alignment
    
    CTC LOSS (Connectionist Temporal Classification):
        - Handles variable-length alignment between image and text
        - No need for character-level segmentation
        - Allows repeated characters and blanks
    
    TRAINING:
        - Pretrained on real IAM data
        - FROZEN during generator training
        - Generator is trained to produce readable text
    
    LOSS (in generator training):
        L_ctc = CTC(Recognizer(generated_img), target_text)
        Weight: 3.0 (high weight = readability is important!)
    """
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 16  # Output width = input_width / 16
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        # ===== CNN BACKBONE =====
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
                             norm='none',
                             activation='none')]
        
        # Downsampling blocks
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len.cpu())
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits

    def frozen_bn(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fix_bn)

