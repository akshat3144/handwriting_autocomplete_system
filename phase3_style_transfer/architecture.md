# HiGAN+ Architecture Documentation

## Complete Network Architecture & Data Flow

This document provides a comprehensive explanation of the **Modified HiGAN+ (Handwriting Imitation GAN Plus)** architecture, including all eight major architectural novelties, dimension transformations at each layer, and the intuition behind each component.

---

## 📂 File Structure Reference

| Component | File Path |
|-----------|-----------|
| Main Model & Training Loop | `networks/model.py` |
| Generator (Original + Improved) | `networks/BigGAN_networks.py` |
| Discriminator (Standard + Multi-scale) | `networks/BigGAN_networks.py`, `networks/multi_scale_discriminator.py` |
| Core Layers (GBlock, DBlock, Attention) | `networks/BigGAN_layers.py` |
| Improved Layers (AdaIN, ModConv, CrossAttn) | `networks/improved_layers.py` |
| Style Encoder, Recognizer, Writer ID | `networks/module.py` |
| Loss Functions | `networks/loss.py` |
| Dataset & HDF5 Loading | `lib/datasets.py` |
| Trained Model Checkpoint | `models/higanplus_trained.pth` |
| Training Configuration | `configs/gan_iam_improved.yml` |

---

## 🎯 Task Overview

**Handwriting Style Transfer / Imitation**: Given a reference handwriting sample from a writer and arbitrary text content, generate a new handwritten image that:
1. Contains the specified text content
2. Mimics the visual style (slant, stroke width, letter shapes, spacing) of the reference writer

---

## 📊 Data Pipeline

### 1. IAM Handwriting Dataset

The IAM Handwriting Database contains handwritten English text from 657 different writers.

**Original Dataset Structure:**
- Forms → Lines → Words
- Total: ~115,000 word images
- Writers: 657 unique writers
- Image height normalized to **64 pixels**
- Variable width based on text length

### 2. HDF5 Preprocessing

**File**: `data/iam/trnvalset_words64_OrgSz.hdf5`, `data/iam/testset_words64_OrgSz.hdf5`

The dataset is stored in HDF5 format for efficient loading:

```
HDF5 File Structure:
├── imgs          : uint8 array [H=64, total_width]  - Concatenated images
├── lbs           : uint8 array [total_chars]        - Character labels (ASCII)
├── img_seek_idxs : int32 array [N_samples]          - Start index for each image
├── lb_seek_idxs  : int32 array [N_samples]          - Start index for each label
├── img_lens      : int32 array [N_samples]          - Width of each image
├── lb_lens       : int32 array [N_samples]          - Length of each text
└── wids          : int32 array [N_samples]          - Writer ID (0-656)
```

**Dimensions per sample:**
| Field | Shape | Description |
|-------|-------|-------------|
| Image | `[1, 64, W]` | Grayscale, H=64, W varies (typically 32-400 pixels) |
| Label | `[L]` | Character indices, L = text length (1-20 chars) |
| Writer ID | scalar | Integer identifying writer (0-656) |

### 3. Data Loading & Batching

**File**: `lib/datasets.py` → `Hdf5Dataset` class

```
Input Batch (after padding):
├── style_imgs     : [B, 1, 64, W_max]    - Padded to max width in batch
├── style_img_lens : [B]                   - Original widths before padding
├── lbs            : [B, L_max]            - Text indices (padded)
├── lb_lens        : [B]                   - Original text lengths
└── wids           : [B]                   - Writer IDs
```

**Typical batch:**
- Batch size: 24
- Image shape: `[24, 1, 64, 320]` (padded with -1)
- Labels: `[24, 20]` (max 20 characters)

---

## 🏗️ Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

     STYLE REFERENCE                          TEXT CONTENT
           │                                       │
           ▼                                       ▼
    ┌─────────────┐                         ┌───────────┐
    │ Style Image │                         │ "hello"   │
    │ [B,1,64,W]  │                         │ [B, L]    │
    └──────┬──────┘                         └─────┬─────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                         ┌───────────────┐
    │   STYLE     │                         │Text Embedding │
    │  BACKBONE   │                         │[B,L,embed_dim]│
    │  (CNN)      │                         └───────┬───────┘
    └──────┬──────┘                                 │
           │                                        │
           ▼                                        ▼
    ┌─────────────┐                    ┌────────────────────┐
    │   STYLE     │                    │ Positional Encoding │ ◄── NOVELTY 8
    │  ENCODER    │                    │   (Sinusoidal PE)   │
    │   (VAE)     │                    └──────────┬─────────┘
    └──────┬──────┘                               │
           │                                      │
           │ style_z                              │
           │ [B, 32]                              ▼
           │                           ┌──────────────────┐
           │                           │ Transformer Enc  │ ◄── NOVELTY 4
           │                           │ (Global Context) │
           │                           └────────┬─────────┘
           │                                    │
           │                                    ▼
           │                           ┌──────────────────┐
           │                           │    BiGRU Enc     │ ◄── NOVELTY 3
           │                           │ (Sequence Model) │
           │                           └────────┬─────────┘
           │                                    │
           └──────────────┬─────────────────────┘
                          │
                          ▼
                  ┌───────────────────┐
                  │ CROSS-ATTENTION   │ ◄── NOVELTY 2
                  │ Style ↔ Content   │
                  │ Fusion            │
                  └─────────┬─────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │       GENERATOR           │
              │  ┌─────────────────────┐  │
              │  │ Linear Projection   │  │
              │  │ [B,L,C×H×W]         │  │
              │  └──────────┬──────────┘  │
              │             ▼             │
              │  ┌─────────────────────┐  │
              │  │   Reshape to 4D     │  │
              │  │  [B,512,4,L×4]      │  │
              │  └──────────┬──────────┘  │
              │             ▼             │
              │  ┌─────────────────────┐  │
              │  │ AdaIN GBlock ×4     │ ◄┼── NOVELTY 1 (StyleGAN2 Control)
              │  │ + MultiHead Attn    │  │
              │  │ + Skip Connections  │ ◄┼── NOVELTY 7
              │  └──────────┬──────────┘  │
              │             ▼             │
              │  ┌─────────────────────┐  │
              │  │   Output Layer      │  │
              │  │  [B,1,64,W_out]     │  │ 
              │  └─────────────────────┘  │
              └───────────────────────────┘
                            │
                            ▼
                   GENERATED IMAGE
                    [B, 1, 64, W]
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
   ┌─────────────┐                    ┌───────────────┐
   │   OCR       │                    │  MULTI-SCALE  │ ◄── NOVELTY 6
   │ RECOGNIZER  │                    │ DISCRIMINATOR │
   └──────┬──────┘                    └───────┬───────┘
          │                                   │
          ▼                                   ▼
    CTC Loss                          Adversarial Loss
      (Text)                          + Contrastive Loss ◄── NOVELTY 5
                                      (Writer Disentangle)
```

---

## 🔧 Core Building Blocks

Before diving into the novelties, these are the fundamental components used throughout the architecture:

### Spectral Normalization (SN)

**File**: `networks/BigGAN_layers.py` → `SN`, `SNConv2d`, `SNLinear`

**Purpose**: Stabilize GAN training by constraining the Lipschitz constant of the discriminator.

```python
class SpectralNorm:
    """
    Constrains weight matrix W so that ||W||_spectral ≤ 1
    
    The spectral norm is the largest singular value of W.
    By dividing W by this value, we ensure the layer doesn't
    amplify inputs too much → prevents exploding gradients.
    
    Uses Power Iteration to estimate largest singular value:
    1. v = W^T @ u / ||W^T @ u||
    2. u = W @ v / ||W @ v||
    3. σ = u^T @ W @ v (singular value)
    """
    def forward(self, weight):
        # Estimate largest singular value via power iteration
        sigma = power_iteration(weight, u, v, num_iters=1)
        # Normalize weight
        return weight / sigma
```

**Where Used**:
- All Conv2d layers in Generator (`SNConv2d`)
- All Linear layers in Generator (`SNLinear`)
- All layers in Discriminator
- Embedding layers (`SNEmbedding`)

### Self-Attention Layer

**File**: `networks/BigGAN_layers.py` → `SelfAttention`

**Purpose**: Allow spatial positions to attend to each other for global coherence.

```python
class SelfAttention(nn.Module):
    """
    Self-attention for 2D feature maps.
    Each pixel can attend to every other pixel.
    
    Dimensions:
        Input:  [B, C, H, W]
        Query:  [B, C//8, H×W]  ← Reduced channels for efficiency
        Key:    [B, C//8, H×W]
        Value:  [B, C, H×W]     ← Full channels for output
    """
    def __init__(self, in_dim):
        self.query_conv = SNConv2d(in_dim, in_dim//8, 1)  # 1×1 conv
        self.key_conv = SNConv2d(in_dim, in_dim//8, 1)
        self.value_conv = SNConv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scale
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Project to Q, K, V
        Q = self.query_conv(x).view(B, -1, H*W)  # [B, C/8, N]
        K = self.key_conv(x).view(B, -1, H*W)    # [B, C/8, N]
        V = self.value_conv(x).view(B, -1, H*W)  # [B, C, N]
        
        # Attention: softmax(Q^T @ K)
        attention = softmax(Q.transpose(1,2) @ K)  # [B, N, N]
        
        # Apply attention to values
        out = V @ attention.transpose(1,2)  # [B, C, N]
        out = out.view(B, C, H, W)
        
        # Residual with learnable scale
        return self.gamma * out + x
```

**Dimension Example** (at 64×320 resolution):
```
Input:  [B, 64, 64, 320]
Query:  [B, 8, 20480]   (64/8 = 8 channels, 64×320 = 20480 positions)
Key:    [B, 8, 20480]
Value:  [B, 64, 20480]
Attention: [B, 20480, 20480]  ← Each position attends to all others
Output: [B, 64, 64, 320]
```

**Why for Handwriting**: Ensures consistent stroke style across the entire word.

### GBlock (Generator Block)

**File**: `networks/BigGAN_layers.py` → `GBlock`

**Purpose**: Main building block of Generator. Upsamples and transforms features.

```python
class GBlock(nn.Module):
    """
    Residual block with conditional batch normalization.
    
    Structure:
        x → BN → ReLU → Upsample → Conv → BN → ReLU → Conv → + residual
                  ↓                                         ↑
              (style conditioning via ccbn)         (skip connection)
    """
    def __init__(self, in_ch, out_ch, style_dim, upsample=True):
        # Conditional batch norm layers (style-conditioned)
        self.bn1 = ccbn(in_ch, style_dim)   # γ, β from style
        self.bn2 = ccbn(out_ch, style_dim)
        
        # Convolutions with spectral norm
        self.conv1 = SNConv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, 3, padding=1)
        
        # Skip connection (learnable if channels change)
        self.skip = SNConv2d(in_ch, out_ch, 1) if in_ch != out_ch else Identity
    
    def forward(self, x, style):
        h = self.bn1(x, style)           # Style-conditioned norm
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2)  # Upsample 2×
        h = self.conv1(h)
        h = self.bn2(h, style)
        h = F.relu(h)
        h = self.conv2(h)
        
        # Skip connection (also upsampled)
        skip = F.interpolate(x, scale_factor=2)
        skip = self.skip(skip)
        
        return h + skip
```

### DBlock (Discriminator Block)

**File**: `networks/BigGAN_layers.py` → `DBlock`

**Purpose**: Main building block of Discriminator. Downsamples and extracts features.

```python
class DBlock(nn.Module):
    """
    Residual block for discriminator (no conditioning).
    
    Structure:
        x → ReLU → Conv → ReLU → Conv → Downsample → + residual
                                                      ↑
                                             (skip connection)
    """
    def __init__(self, in_ch, out_ch, downsample=True):
        self.conv1 = SNConv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, 3, padding=1)
        self.skip = SNConv2d(in_ch, out_ch, 1) if in_ch != out_ch else Identity
        self.downsample = downsample
    
    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)  # Downsample 2×
        
        # Skip connection
        skip = self.skip(x)
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)
        
        return h + skip
```

### Conditional Batch Normalization (ccbn)

**File**: `networks/BigGAN_layers.py` → `ccbn`

**Purpose**: Style-conditioned normalization. The style vector controls scale (γ) and shift (β).

```python
class ccbn(nn.Module):
    """
    Conditional Batch Normalization.
    
    Standard BN: y = γ * (x - μ) / σ + β  (γ, β are learned)
    Conditional BN: γ, β come from a linear projection of the style vector
    
    This allows different styles to normalize features differently.
    """
    def __init__(self, num_features, style_dim):
        self.bn = nn.BatchNorm2d(num_features, affine=False)  # No learnable params
        self.gain = SNLinear(style_dim, num_features)  # γ from style
        self.bias = SNLinear(style_dim, num_features)  # β from style
    
    def forward(self, x, style):
        # Standard batch normalization (no affine)
        x = self.bn(x)
        
        # Style-specific scale and shift
        gain = self.gain(style).view(-1, C, 1, 1)  # [B, C, 1, 1]
        bias = self.bias(style).view(-1, C, 1, 1)
        
        return x * (1 + gain) + bias
```

**Dimension Flow**:
```
Input x: [B, 256, 16, 80]
Style:   [B, 32]
     ↓
BN(x):   [B, 256, 16, 80]  (normalized, zero mean, unit var)
     ↓
gain = Linear(style): [B, 32] → [B, 256] → [B, 256, 1, 1]
bias = Linear(style): [B, 32] → [B, 256] → [B, 256, 1, 1]
     ↓
Output:  x * (1 + gain) + bias  → [B, 256, 16, 80]
```

---

## 🖼️ Three Types of Generated Images

During training, the Generator produces **three different types** of images, each serving a specific purpose:

### 1. Random Style Images (`fake_imgs`)
```python
z_dist.sample_()  # Sample random style z ~ N(0,1)
fake_imgs = generator(z_dist, fake_lbs, fake_lb_lens)
```
- **Style Source**: Random vector from standard normal distribution
- **Text Source**: Random words from lexicon
- **Purpose**: Test if discriminator can detect "invented" styles
- **Losses Applied**: Adversarial, CTC, Info Loss

### 2. Style Transfer Images (`style_imgs`)
```python
enc_z = style_encoder(real_imgs, ...)  # Encode real image style
style_imgs = generator(enc_z, fake_lbs, fake_lb_lens)  # Different text!
```
- **Style Source**: Encoded from real handwriting sample
- **Text Source**: Random words (different from original)
- **Purpose**: Generate new text in existing writer's style
- **Losses Applied**: Adversarial, CTC, Writer ID, Contextual

### 3. Reconstruction Images (`recn_imgs`)
```python
enc_z = style_encoder(real_imgs, ...)  # Encode real image style
recn_imgs = generator(enc_z, real_lbs, real_lb_lens)  # SAME text!
```
- **Style Source**: Encoded from real handwriting sample
- **Text Source**: Same text as original image
- **Purpose**: Should perfectly reconstruct the input
- **Losses Applied**: Adversarial, CTC, Writer ID, **Reconstruction L1 (×5.0)**

### Visual Summary
```
Real Image: "hello" by Writer A
     │
     ├─→ Encode Style ─→ z_A [32-dim]
     │                        │
     │    ┌───────────────────┼───────────────────┐
     │    │                   │                   │
     │    ▼                   ▼                   ▼
     │  "world"            "hello"           z_random
     │    │                   │                   │
     │    ▼                   ▼                   ▼
     │ style_imgs         recn_imgs          fake_imgs
     │ (style transfer)   (reconstruction)  (random style)
     │                         │
     └─────────────────────────┴──→ Should match original!
```

---

## 🔍 Auxiliary Networks (Frozen During Training)

### Recognizer (OCR Network)

**File**: `networks/module.py` → `Recognizer`

**Purpose**: Ensures generated text is **readable**. Acts as a "text critic".

**Architecture**:
```
Input Image: [B, 1, 64, W]
     ↓
CNN Backbone (same structure as StyleBackbone):
  ResBlocks + MaxPool × 4
  [B, 1, 64, W] → [B, 256, 4, W/16]
     ↓
Squeeze height: [B, 256, W/16]
     ↓
BiLSTM (2 layers, bidirectional):
  [B, W/16, 256] → [B, W/16, 256]
  Captures left-to-right and right-to-left context
     ↓
Linear: [B, W/16, 80]  (80 character classes)
     ↓
Log-Softmax: [B, W/16, 80]
     ↓
CTC Decode → "hello"
```

**Key Parameters**:
- `n_class`: 80 (a-z, A-Z, 0-9, punctuation, blank token)
- `rnn_depth`: 2 (BiLSTM layers)
- `len_scale`: 16 (output width = input_width / 16)

**CTC Loss Details**:
```python
ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')

# For each generated image type:
loss = ctc_loss(
    log_probs,      # [T, B, 80] from recognizer
    targets,        # [B, L] ground truth characters
    input_lengths,  # [B] actual output sequence lengths
    target_lengths  # [B] actual text lengths
)
```

### Writer Identifier

**File**: `networks/module.py` → `WriterIdentifier`

**Purpose**: Classifies which of 372 writers produced an image.

**Architecture**:
```
Input Image: [B, 1, 64, W]
     ↓
StyleBackbone (shared with StyleEncoder):
  [B, 1, 64, W] → [B, 256, W/16]
     ↓
Global Average Pool (masked):
  [B, 256, W/16] → [B, 256]
     ↓
MLP: Linear(256→256) + LeakyReLU + Linear(256→372)
     ↓
Output: [B, 372] (logits for each writer)
```

**Loss**:
```python
wid_loss = CrossEntropyLoss()(
    writer_identifier(generated_img),  # [B, 372]
    real_writer_ids                     # [B] ground truth
)
```

---

## 🔬 Eight Major Architectural Novelties (Detailed)

### 1️⃣ StyleGAN2-Style Control (AdaIN + ModConv)

**File**: `networks/improved_layers.py` → `AdaIN`, `ModulatedConv2d`

**Problem Solved**: Standard conditional batch normalization (cBN) in BigGAN provides limited style control—it only learns scale and shift parameters, treating style globally.

**Solution**: Adaptive Instance Normalization + Weight Modulation from StyleGAN2

#### AdaIN (Adaptive Instance Normalization)

```python
class AdaIN(nn.Module):
    """
    Instead of learning fixed BN statistics, AdaIN:
    1. Instance-normalizes each sample independently
    2. Applies style-specific scale (γ) and shift (β)
    """
    def forward(self, x, style):
        # x: [B, C, H, W] feature maps
        # style: [B, style_dim] style vector
        
        # Step 1: Instance Normalization
        mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True)    # [B, C, 1, 1]
        x_norm = (x - mean) / (std + ε)          # [B, C, H, W]
        
        # Step 2: Style Modulation
        scale = self.style_scale(style)  # Linear: [B, style_dim] → [B, C]
        shift = self.style_shift(style)  # Linear: [B, style_dim] → [B, C]
        
        # Output: apply style-specific transformation
        return x_norm * (1 + scale) + shift  # [B, C, H, W]
```

**Dimension Flow:**
```
Input:  x [B, 256, 16, 64], style [B, 32]
        ↓
Instance Norm: x_norm [B, 256, 16, 64]
        ↓
Style Linear:  scale [B, 256], shift [B, 256]
        ↓
Output: [B, 256, 16, 64] (same shape, style-modulated)
```

#### Modulated Convolution

```python
class ModulatedConv2d(nn.Module):
    """
    StyleGAN2's key innovation: modulate conv WEIGHTS, not features.
    This provides more expressive style control at the convolution level.
    """
    def forward(self, x, style):
        # x: [B, C_in, H, W]
        # style: [B, style_dim]
        
        # Step 1: Get style modulation
        style_mod = self.style_mod(style)  # [B, C_in]
        
        # Step 2: Modulate weights
        # Original weight: [C_out, C_in, kH, kW]
        # Expand: [B, C_out, C_in, kH, kW]
        weight = self.weight * style_mod.view(B, 1, C_in, 1, 1)
        
        # Step 3: Demodulate (normalize by output std)
        # This prevents magnitude explosion
        demod = rsqrt(weight.pow(2).sum([2,3,4]) + ε)
        weight = weight * demod.view(B, C_out, 1, 1, 1)
        
        # Step 4: Group convolution (each sample uses its own weights)
        return group_conv(x, weight, groups=B)
```

**Why for Handwriting?**
- Different writers have distinct stroke weights, letter proportions, and curvatures
- AdaIN allows the generator to adapt normalization statistics per-writer
- ModConv enables style-specific feature extraction at every layer

---

### 2️⃣ Cross-Attention Fusion of Style + Content

**File**: `networks/improved_layers.py` → `StyleContentCrossAttention`, `MultiHeadCrossAttention`

**Problem Solved**: In original HiGAN+, style is simply concatenated with text features. This doesn't allow the model to learn which style aspects are relevant for which characters.

**Solution**: Multi-head cross-attention where text queries attend to style information.

#### Exact Cross-Attention Dimensions (from config)

| Parameter | Value | Source/Calculation |
|-----------|-------|-------------------|
| **Query Dimension (content_dim)** | **152** | `embed_dim + style_dim = 120 + 32` |
| **Key/Value Dimension (style_dim)** | **32** | Style vector dimension from config |
| **Embedding Dimension** | **152** | Same as content_dim (projects to same space) |
| **Number of Heads** | **4** | Configured in Generator |
| **Head Dimension** | **38** | `embed_dim / num_heads = 152 / 4` |
| **Dropout** | **0.1** | Regularization |

```python
class StyleContentCrossAttention(nn.Module):
    """
    Query: Text/content features (what to write)
    Key/Value: Style features (how to write)
    
    The model learns to attend to relevant style information
    for each character position.
    """
    def __init__(self, content_dim=152, style_dim=32, num_heads=4):
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=content_dim,   # 152 (text+style concat)
            key_dim=style_dim,       # 32 (pure style vector)
            embed_dim=content_dim,   # 152 (projection space)
            num_heads=num_heads      # 4 heads
        )
    
    def forward(self, content, style):
        # content: [B, L, 152] - text+style embeddings
        # style: [B, 32] - global style vector
        
        # Expand style for attention: [B, 32] → [B, 1, 32]
        style = style.unsqueeze(1)
        
        # Cross-attention: Q from content, K/V from style
        Q = self.q_proj(content)  # [B, L, 152] → [B, L, 152]
        K = self.k_proj(style)    # [B, 1, 32] → [B, 1, 152]
        V = self.v_proj(style)    # [B, 1, 32] → [B, 1, 152]
        
        # Split into heads: [B, L, 152] → [B, 4, L, 38]
        # Attention: Q @ K^T / sqrt(38)
        attn = softmax(Q @ K.T / sqrt(38))  # [B, 4, L, 1]
        
        # Apply attention and merge heads
        attended = attn @ V  # [B, 4, L, 38] → [B, L, 152]
        
        # Output projection back to content_dim
        return self.out_proj(attended) + content  # Residual

```

#### Multi-Head Cross-Attention Internals

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Detailed dimension flow for 4-head cross-attention.
    """
    def __init__(self, query_dim=152, key_dim=32, embed_dim=152, num_heads=4):
        self.head_dim = embed_dim // num_heads  # 152/4 = 38
        
        # Projection layers
        self.q_proj = nn.Linear(query_dim, embed_dim)   # 152 → 152
        self.k_proj = nn.Linear(key_dim, embed_dim)     # 32 → 152
        self.v_proj = nn.Linear(key_dim, embed_dim)     # 32 → 152
        self.out_proj = nn.Linear(embed_dim, query_dim) # 152 → 152
        
        self.scale = self.head_dim ** -0.5  # 1/sqrt(38) ≈ 0.162
```

**Dimension Flow (Step-by-Step):**
```
Query (Text+Style features):
    Input:  [B, L_text, 152]  ← (embed_dim=120) + (style_dim=32)
    Q Proj: [B, L_text, 152]  ← Linear(152→152)
    Reshape: [B, 4, L_text, 38]  ← 4 heads × 38 dims each

Key/Value (Style vector):
    Input:  [B, 1, 32]   ← style_dim, expanded to sequence
    K Proj: [B, 1, 152]  ← Linear(32→152)
    V Proj: [B, 1, 152]  ← Linear(32→152)
    Reshape: [B, 4, 1, 38]  ← 4 heads × 38 dims each

Attention Computation:
    Scores: Q @ K^T = [B, 4, L_text, 38] @ [B, 4, 38, 1]
          = [B, 4, L_text, 1]  ← attention weights per head
    Softmax: [B, 4, L_text, 1]  ← normalized (sum to 1)
    
Apply Attention:
    Weighted V: [B, 4, L_text, 1] @ [B, 4, 1, 38]
              = [B, 4, L_text, 38]  ← attended values

Merge Heads:
    Concat: [B, L_text, 152]  ← 4 × 38 = 152
    Out Proj: [B, L_text, 152]  ← Linear(152→152)

Output: [B, L_text, 152] (style-informed content features)
```

**Intuition**: Each of the 4 attention heads can learn different style aspects:
- Head 1: Stroke thickness information
- Head 2: Slant/angle information  
- Head 3: Letter spacing patterns
- Head 4: Character shape variations

**Why for Handwriting?**
- Different characters may need different style emphasis (e.g., 'o' vs 'l' have different stroke patterns)
- Allows the model to selectively apply style features where relevant
- Never done before in HiGAN+ architecture

---

### 3️⃣ Sequence Modeling using BiGRU

**File**: `networks/improved_layers.py` → `BiGRUEncoder`

**Problem Solved**: Handwriting is inherently sequential. Standard feed-forward processing loses the temporal dependencies between characters (e.g., ligatures, consistent slant).

**Solution**: Bidirectional GRU captures both left-to-right and right-to-left dependencies.

```python
class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU for capturing:
    - Left-to-right: How previous characters affect current
    - Right-to-left: How following characters affect current
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim // 2,  # Half for each direction
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x, lengths):
        # x: [B, L, D]
        # Pack for efficient variable-length processing
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output  # [B, L, D]
```

**Dimension Flow:**
```
Input:  [B, 20, 152] (sequence of char+style embeddings)
     ↓
Forward GRU:  hidden_dim=76, output [B, 20, 76]
Backward GRU: hidden_dim=76, output [B, 20, 76]
     ↓
Concat: [B, 20, 152] (bidirectional features)
```

**Why for Handwriting?**
- Captures ligatures: in cursive, 'th' connects differently than 'ot'
- Maintains consistent slant across the word
- Models natural spacing patterns based on surrounding characters

---

### 4️⃣ Global Context Modeling using Transformers

**File**: `networks/improved_layers.py` → `TextTransformerEncoder`

**Problem Solved**: GRU processes sequentially, which can be slow and may not capture global word-level patterns effectively.

**Solution**: Transformer encoder with self-attention for global context.

```python
class TextTransformerEncoder(nn.Module):
    """
    Multi-layer Transformer encoder for global text understanding.
    Self-attention allows each character to see all other characters.
    """
    def __init__(self, embed_dim, num_layers=2, num_heads=4):
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu'
            ),
            num_layers=num_layers
        )
    
    def forward(self, x, padding_mask):
        # x: [B, L, D]
        x = self.pos_encoding(x)  # Add position info
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x
```

**Dimension Flow:**
```
Input:  [B, 20, 152]
     ↓
Pos Encoding: [B, 20, 152] + PE
     ↓
Transformer Layer 1:
  - Self-Attention [B, 20, 152] → [B, 20, 152]
  - FFN [B, 20, 152] → [B, 20, 608] → [B, 20, 152]
     ↓
Transformer Layer 2:
  - Same structure
     ↓
Output: [B, 20, 152]
```

**Why for Handwriting?**
- Global word structure: 'b' in 'big' vs 'b' in 'amb' have different contexts
- Parallel processing (faster than GRU during training)
- Better gradient flow for learning long-range dependencies

---

### 5️⃣ Writer-Disentangled Style via Contrastive Learning

**File**: `networks/loss.py` → `ContrastiveStyleLoss`, `networks/module.py` → `StyleEncoder`

**Problem Solved**: The style encoder might entangle writer identity with other factors (content, random variation). We want a style space where same-writer samples cluster together.

**Solution**: InfoNCE contrastive loss that pulls same-writer samples together and pushes different writers apart.

```python
class ContrastiveStyleLoss(nn.Module):
    """
    InfoNCE loss for style disentanglement.
    
    Positive pairs: samples from the same writer
    Negative pairs: samples from different writers
    """
    def __init__(self, temperature=0.07):
        self.temperature = temperature
    
    def forward(self, style_vectors, writer_ids):
        # style_vectors: [B, D] normalized style vectors
        # writer_ids: [B] writer ID for each sample
        
        # Normalize
        style_vectors = F.normalize(style_vectors, dim=1)
        
        # Similarity matrix [B, B]
        sim = (style_vectors @ style_vectors.T) / self.temperature
        
        # Positive mask: same writer
        pos_mask = (writer_ids.view(-1,1) == writer_ids.view(1,-1)).float()
        pos_mask.fill_diagonal_(0)  # Exclude self
        
        # InfoNCE loss
        exp_sim = torch.exp(sim) * (1 - torch.eye(B))  # Exclude self
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        
        loss = -torch.log(pos_sum / all_sum)
        return loss.mean()
```

**Style Encoder with Projection Head:**
```python
class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, use_contrastive=True):
        # Main encoder
        self.linear_style = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(256, style_dim)  # VAE mean
        self.logvar = nn.Linear(256, style_dim)  # VAE variance
        
        # Contrastive projection head
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(style_dim, style_dim * 2),
                nn.ReLU(),
                nn.Linear(style_dim * 2, style_dim)
            )
    
    def get_contrastive_embedding(self, style):
        # Used for contrastive loss
        proj = self.projection_head(style)
        return F.normalize(proj, dim=1)
```

**Dimension Flow:**
```
Style Image: [B, 1, 64, 320]
     ↓
StyleBackbone (CNN): [B, 256, 1, W/16]
     ↓
Global Avg Pool: [B, 256]
     ↓
Linear Style: [B, 256]
     ↓
μ, log(σ²): [B, 32], [B, 32]
     ↓
Reparameterize: z = μ + σ·ε, [B, 32]
     ↓
Projection Head (for contrastive): [B, 32]
```

**Why for Handwriting?**
- Forces style encoder to capture writer-specific characteristics
- Disentangles content (what) from style (how)
- Enables better writer transfer to unseen text

---

### 6️⃣ Multi-Scale, Multi-Head Discriminator

**File**: `networks/multi_scale_discriminator.py` → `MultiScaleDiscriminator`

**Problem Solved**: A single discriminator may focus on either global structure or local details, but not both. Handwriting quality depends on both levels.

**Solution**: Shared backbone with three specialized heads.

```python
class MultiScaleDiscriminator(nn.Module):
    """
    Three-branch discriminator:
    1. Global: Overall image quality and structure
    2. Patch: Local texture and stroke details
    3. Character: Per-character attention
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=4):
        # Shared backbone (3 downsampling layers)
        self.shared_backbone = nn.ModuleList([
            SNConv(1, 64, stride=2),    # 64x320 → 32x160
            SNConv(64, 128, stride=2),  # 32x160 → 16x80
            SNConv(128, 256, stride=2)  # 16x80 → 8x40
        ])
        
        # Branch 1: Global
        self.global_head = nn.ModuleList([
            SNConv(256, 512, stride=2),  # 8x40 → 4x20
        ])
        self.global_output = SNConv(512, 1)
        
        # Branch 2: Patch
        self.patch_head = nn.Sequential(
            SNConv(256, 256),
            SNConv(256, 1)
        )
        
        # Branch 3: Character attention
        self.char_attention = nn.Sequential(
            SNConv(256, 128),
            SNConv(128, 1),
            nn.Sigmoid()  # Attention weights
        )
    
    def forward(self, x):
        # Shared processing
        feat = x
        for layer in self.shared_backbone:
            feat = layer(feat)
        
        # Global score
        global_feat = feat
        for layer in self.global_head:
            global_feat = layer(global_feat)
        global_score = self.global_output(global_feat).mean([2,3])
        
        # Patch score
        patch_score = self.patch_head(feat).mean([2,3])
        
        # Character attention
        char_attn = self.char_attention(feat)
        
        return {
            'global': global_score,
            'patch': patch_score,
            'char_attn': char_attn,
            'combined': global_score + 0.5 * patch_score
        }
```

**Dimension Flow:**
```
Input: [B, 1, 64, 320]
     ↓
Shared Backbone:
  Layer 1: [B, 64, 32, 160]
  Layer 2: [B, 128, 16, 80]
  Layer 3: [B, 256, 8, 40]
     ↓
Global Branch:
  [B, 512, 4, 20] → [B, 1, 4, 20] → mean → [B, 1]
     ↓
Patch Branch:
  [B, 256, 8, 40] → [B, 1, 8, 40] → mean → [B, 1]
     ↓
Char Attention:
  [B, 256, 8, 40] → [B, 1, 8, 40] (attention map)
```

**Why for Handwriting?**
- Global: Correct word structure, consistent slant, proper baseline
- Patch: Stroke quality, ink texture, anti-aliasing
- Character: Per-character quality control, helps with difficult letters

---

### 7️⃣ Skip-Connections for Multi-Scale Style Retention

**File**: `networks/improved_layers.py` → `MultiScaleStyleFusion`

**Problem Solved**: As features pass through deep generator layers, fine-grained style information (thin strokes, serifs) may be lost.

**Solution**: Skip connections that inject multi-scale style features at different generator stages.

```python
class MultiScaleStyleFusion(nn.Module):
    """
    Injects style features from encoder at multiple scales.
    
    Style Encoder produces features at 3 scales:
    - feat2: After 2nd ResBlock (low-level: edges, strokes)
    - feat3: After 3rd ResBlock (mid-level: letter parts)
    - feat4: After 4th ResBlock (high-level: letter shapes)
    
    These are adapted and added to generator features.
    """
    def __init__(self, style_channels, target_channels):
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sc, tc, 1),  # 1x1 conv to match channels
                nn.InstanceNorm2d(tc),
                nn.LeakyReLU(0.2)
            )
            for sc, tc in zip(style_channels, target_channels)
        ])
    
    def forward(self, style_feats, target_feats):
        fused = []
        for adapter, sf, tf in zip(self.adapters, style_feats, target_feats):
            # Resize style feature to match target spatial size
            sf_resized = F.interpolate(sf, size=tf.shape[2:])
            sf_adapted = adapter(sf_resized)
            # Weighted residual connection
            fused.append(tf + 0.1 * sf_adapted)
        return fused
```

**Style Backbone Multi-Scale Features:**
```
Input: [B, 1, 64, W]
     ↓
ResBlock 1-2: feat2 [B, 64, 16, W/4]   (low-level strokes)
     ↓
ResBlock 3-4: feat3 [B, 128, 8, W/8]   (mid-level parts)
     ↓
ResBlock 5-6: feat4 [B, 256, 4, W/16]  (high-level shapes)
```

**Fusion in Generator:**
```
Generator Block 2: [B, 256, 16, L×8]  + adapted feat2
Generator Block 3: [B, 128, 32, L×16] + adapted feat3
Generator Block 4: [B, 64, 64, L×32]  + adapted feat4
```

**Why for Handwriting?**
- Preserves fine stroke details (serifs, stroke endings)
- Maintains consistent style across resolutions
- Prevents "style washing" in deep layers

---

### 8️⃣ Positional Encoding for Stable Sequence Handwriting

**File**: `networks/improved_layers.py` → `SinusoidalPositionalEncoding`, `LearnablePositionalEncoding`

**Problem Solved**: Text embeddings have no inherent notion of position. The model needs to know character order for proper spacing and flow.

**Solution**: Sinusoidal positional encoding (from Transformers) that provides unique position signatures.

```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects position information using sine/cosine functions.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Properties:
    - Each position has unique encoding
    - Relative positions can be computed via linear transformation
    - Generalizes to longer sequences than seen during training
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**Dimension Flow:**
```
Input:  x [B, 20, 152] (text + style embeddings)
     ↓
PE:     [1, 20, 152] (precomputed, added element-wise)
     ↓
Output: [B, 20, 152] (position-aware embeddings)
```

**Visualization of PE patterns:**
```
Position 0:  [sin(0), cos(0), sin(0/k), cos(0/k), ...]
Position 1:  [sin(1), cos(1), sin(1/k), cos(1/k), ...]
Position 2:  [sin(2), cos(2), sin(2/k), cos(2/k), ...]
...
(Each position has unique "fingerprint")
```

**Why for Handwriting?**
- Ensures consistent left-to-right flow
- Helps with proper character spacing
- Critical for Transformer self-attention to understand sequence order
- Enables generation of variable-length words

---

## 🔄 Complete Forward Pass (Step-by-Step)

### Phase 1: Style Encoding

```
STYLE IMAGE INPUT
[B, 1, 64, W]  (e.g., [24, 1, 64, 320])
      │
      ▼
┌─────────────────────────────────────┐
│         STYLE BACKBONE              │
│  (networks/module.py → StyleBackbone)│
├─────────────────────────────────────┤
│ ConstantPad2d + Conv 5×5            │
│ [24, 1, 64, 320] → [24, 16, 32, 160]│
│                                     │
│ ResBlock ×2 + MaxPool               │
│ [24, 16, 32, 160] → [24, 64, 16, 80]│
│                                     │
│ ResBlock ×2 + MaxPool               │
│ [24, 64, 16, 80] → [24, 128, 8, 40] │
│                                     │
│ ResBlock ×2 + MaxPool               │
│ [24, 128, 8, 40] → [24, 256, 4, 20] │
│                                     │
│ Conv 3×3 (final)                    │
│ [24, 256, 4, 20] → [24, 256, 1, 20] │
│                                     │
│ Squeeze + Transpose                 │
│ [24, 256, 20] (sequence format)     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│         STYLE ENCODER               │
│  (networks/module.py → StyleEncoder) │
├─────────────────────────────────────┤
│ Global Average Pool over sequence   │
│ [24, 256, 20] → [24, 256]           │
│                                     │
│ Linear 256→256 + LeakyReLU ×2       │
│ [24, 256] → [24, 256]               │
│                                     │
│ VAE heads:                          │
│ μ = Linear(256, 32)  → [24, 32]     │
│ logσ² = Linear(256, 32) → [24, 32]  │
│                                     │
│ Reparameterize:                     │
│ z = μ + σ·ε  → [24, 32]             │
└─────────────────────────────────────┘
      │
      ▼
STYLE VECTOR z: [24, 32]
```

### Phase 2: Text Processing

```
TEXT INPUT
[B, L] (e.g., [24, 10] for 10-char words)
      │
      ▼
┌─────────────────────────────────────┐
│       TEXT EMBEDDING                │
│ (nn.Embedding)                      │
├─────────────────────────────────────┤
│ Embedding(80, 120)                  │
│ [24, 10] → [24, 10, 120]            │
│ (80 = alphabet size, 120 = embed_dim)│
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│    POSITIONAL ENCODING (Novelty 8) │
├─────────────────────────────────────┤
│ SinusoidalPositionalEncoding        │
│ [24, 10, 120] + PE → [24, 10, 120]  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│    STYLE-TEXT CONCATENATION        │
├─────────────────────────────────────┤
│ Expand style: [24, 32] → [24, 10, 32]│
│ Concat: [24, 10, 120] || [24, 10, 32]│
│ Result: [24, 10, 152]               │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  TRANSFORMER ENCODER (Novelty 4)   │
├─────────────────────────────────────┤
│ 2 layers, 4 heads, FFN=608          │
│                                     │
│ Self-Attention (each char sees all) │
│ [24, 10, 152] → [24, 10, 152]       │
│                                     │
│ Feed-Forward Network                │
│ [24, 10, 152] → [24, 10, 608]       │
│              → [24, 10, 152]        │
│                                     │
│ ×2 layers with residual connections │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│     BIGRU ENCODER (Novelty 3)      │
├─────────────────────────────────────┤
│ BiGRU(152, 76, bidirectional=True)  │
│ [24, 10, 152] → [24, 10, 152]       │
│ (76 forward + 76 backward)          │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ CROSS-ATTENTION FUSION (Novelty 2) │
├─────────────────────────────────────┤
│ Q = content (text), K/V = style     │
│                                     │
│ Q: [24, 10, 152]                    │
│ K: [24, 1, 32] (style expanded)     │
│ V: [24, 1, 32]                      │
│                                     │
│ Attention + FFN                     │
│ [24, 10, 152] → [24, 10, 152]       │
└─────────────────────────────────────┘
      │
      ▼
FUSED CONTENT+STYLE: [24, 10, 152]
```

### Phase 3: Image Generation

```
FUSED FEATURES: [24, 10, 152]
      │
      ▼
┌─────────────────────────────────────┐
│       LINEAR PROJECTION             │
├─────────────────────────────────────┤
│ Linear(152, 512×4×4)                │
│ [24, 10, 152] → [24, 10, 8192]      │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│         RESHAPE TO 4D              │
├─────────────────────────────────────┤
│ View + Permute                      │
│ [24, 10, 8192]                      │
│  → [24, 40, 4, 512] (L×W, H, C)     │
│  → [24, 512, 4, 40]  (B, C, H, W)   │
│                                     │
│ Initial: 4 pixels height, 4×L width │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   ADAIN GBLOCK 1 (Novelty 1)       │
├─────────────────────────────────────┤
│ Upsample: ×(2,1) → [24, 512, 8, 40] │
│ Conv + AdaIN + Conv + AdaIN         │
│ [24, 512, 8, 40] → [24, 256, 8, 40] │
│                                     │
│ Style injection via:                │
│ - Instance norm per sample          │
│ - Style-specific scale/shift        │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   ADAIN GBLOCK 2 + ATTENTION       │
├─────────────────────────────────────┤
│ Upsample: ×(2,2) → [24,256,16,80]   │
│ Conv + AdaIN + Conv + AdaIN         │
│ [24, 256, 16, 80] → [24, 128, 16, 80]│
│                                     │
│ Multi-Head Self-Attention           │
│ [24, 128, 16, 80] → [24, 128, 16, 80]│
│ (Spatial attention for coherence)   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   ADAIN GBLOCK 3                   │
├─────────────────────────────────────┤
│ Upsample: ×(2,2) → [24,128,32,160]  │
│ Conv + AdaIN + Conv + AdaIN         │
│ [24, 128, 32, 160] → [24, 64, 32, 160]│
│                                     │
│ Skip connection from style backbone │
│ (Novelty 7: Multi-scale retention)  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   ADAIN GBLOCK 4 + ATTENTION       │
├─────────────────────────────────────┤
│ Upsample: ×(2,2) → [24,64,64,320]   │
│ Conv + AdaIN + Conv + AdaIN         │
│ [24, 64, 64, 320] → [24, 64, 64, 320]│
│                                     │
│ Multi-Head Self-Attention           │
│ (Final coherence check)             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│        OUTPUT LAYER                │
├─────────────────────────────────────┤
│ InstanceNorm2d                      │
│ ReLU                                │
│ Conv2d(64, 1, 3×3)                  │
│ Tanh                                │
│                                     │
│ [24, 64, 64, 320] → [24, 1, 64, 320]│
└─────────────────────────────────────┘
      │
      ▼
GENERATED IMAGE: [24, 1, 64, 320]
(Range: [-1, 1], normalized grayscale)
```

### Phase 4: Discrimination

```
REAL/FAKE IMAGE: [24, 1, 64, 320]
      │
      ▼
┌─────────────────────────────────────┐
│    MULTI-SCALE DISCRIMINATOR       │
│        (Novelty 6)                 │
├─────────────────────────────────────┤
│                                     │
│ ┌─────────────────────────────────┐ │
│ │     SHARED BACKBONE             │ │
│ ├─────────────────────────────────┤ │
│ │ SNConv stride=2                 │ │
│ │ [24,1,64,320]→[24,64,32,160]    │ │
│ │                                 │ │
│ │ SNConv stride=2                 │ │
│ │ [24,64,32,160]→[24,128,16,80]   │ │
│ │                                 │ │
│ │ SNConv stride=2                 │ │
│ │ [24,128,16,80]→[24,256,8,40]    │ │
│ └─────────────────────────────────┘ │
│         │                           │
│    ┌────┼────┬────────┐             │
│    ▼    ▼    ▼        ▼             │
│ ┌─────┐ ┌─────┐ ┌──────────┐       │
│ │GLOBAL│ │PATCH│ │CHAR ATTN│       │
│ │HEAD │ │HEAD │ │  HEAD   │       │
│ ├─────┤ ├─────┤ ├──────────┤       │
│ │SNConv│ │Conv │ │Conv+Sig │       │
│ │↓    │ │     │ │         │       │
│ │[B,1]│ │[B,1]│ │[B,1,8,40]│      │
│ └─────┘ └─────┘ └──────────┘       │
│    │       │          │             │
│    └───────┼──────────┘             │
│            ▼                        │
│   Combined Score: global + 0.5×patch│
└─────────────────────────────────────┘
```

---

## 📉 Loss Functions

### Generator Losses (Actual Training Weights)

| Loss | Weight (λ) | Purpose | Formula |
|------|--------|---------|---------|
| **Adversarial (Hinge)** | 1.0 | Fool discriminator | $-\mathbb{E}[D(G(z,y))]$ |
| **Patch Adversarial** | 1.0 | Local realism | $-\mathbb{E}[D_{patch}(G(z,y))]$ |
| **CTC (OCR)** | **3.0** | Text readability | $CTC(Recognizer(fake), text)$ |
| **Writer ID** | **1.5** | Style consistency | $CE(WriterID(fake), writer_{id})$ |
| **Reconstruction L1** | **5.0** | Pixel fidelity | $\|G(z,y) - x_{real}\|_1$ |
| **Info Loss** | **1.5** | Style cycle consistency | $\|Encoder(G(z)) - z\|_1$ |
| **Contextual (CX)** | λ_ctx (1.0) | Feature matching | $-\log(CX(fake, real))$ |
| **KL Divergence** | λ_kl (0.0001) | VAE regularization | $D_{KL}(q(z\|x) \|\| p(z))$ |

**Total Generator Loss:**
$$L_G = L_{adv} + L_{patch} + 3.0 \cdot L_{CTC} + 1.5 \cdot L_{info} + 1.5 \cdot L_{WID} + 5.0 \cdot L_{recn} + \lambda_{ctx} \cdot L_{CX} + \lambda_{kl} \cdot L_{KL}$$

**Weight Interpretation (Why These Values?):**
- **Reconstruction (5.0)**: Highest weight because exact pixel match is critical for style learning
- **CTC (3.0)**: Second highest because text must be readable
- **Writer ID & Info (1.5)**: Medium weight for style consistency  
- **Adversarial (1.0)**: Baseline weight for realism
- **KL (0.0001)**: Very low to avoid over-regularization of latent space

### Discriminator Losses

| Loss | Weight | Purpose | Formula |
|------|--------|---------|---------|
| **Hinge (Real)** | 1.0 | Push real scores > 1 | $\mathbb{E}[\max(0, 1 - D(x_{real}))]$ |
| **Hinge (Fake)** | 1.0 | Push fake scores < -1 | $\mathbb{E}[\max(0, 1 + D(G(z)))]$ |
| **Patch (Real)** | 1.0 | Local real detection | Same, on patches |
| **Patch (Fake)** | 1.0 | Local fake detection | Same, on patches |

**Hinge Loss Explained:**
```python
# Discriminator wants: D(real) > 1 and D(fake) < -1
d_loss = (
    F.relu(1 + D(fake)).mean() +      # Penalize if fake > -1
    F.relu(1 - D(real)).mean()        # Penalize if real < 1
)

# Generator wants: D(fake) as high as possible
g_loss = -D(fake).mean()              # Maximize discriminator score
```

### Loss Function Details

#### CTC Loss (Connectionist Temporal Classification)
```python
ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')

# Applied to 3 types of generated images:
ctc_rand = ctc_loss(recognizer(fake_imgs), fake_labels, ...)    # Random style
ctc_style = ctc_loss(recognizer(style_imgs), fake_labels, ...)  # Style transfer
ctc_recn = ctc_loss(recognizer(recn_imgs), real_labels, ...)    # Reconstruction

ctc_total = ctc_rand + ctc_style + ctc_recn  # All must be readable
```

**Why CTC?**
- Handles variable-length alignment (image width ≠ text length)
- Allows blank tokens and repeated characters
- No character-level segmentation needed

#### Info Loss (Style Cycle Consistency)
```python
# If Generator uses style z, encoding the output should recover z
info_loss = |Encoder(Generator(z)) - z|

# Ensures Generator actually uses the style vector
# Prevents mode collapse where style is ignored
```

#### Contextual Loss (Feature Matching)
```python
class CXLoss:
    """
    Matches feature DISTRIBUTIONS, not exact positions.
    For each generated patch, find best matching real patch.
    """
    def forward(self, real_feat, fake_feat):
        # Cosine similarity between all patch pairs
        similarity = cosine_sim(real_patches, fake_patches)
        
        # Soft matching (differentiable argmax)
        weights = softmax(-distance / temperature)
        
        # Loss: negative log of best matches
        return -log(max_similarity)
```

---

## 🎛️ Hyperparameters

### Actual Configuration (from `configs/gan_iam.yml`)

```yaml
# =============================================================================
# CORE DIMENSIONS
# =============================================================================
img_height: 64          # Fixed image height (pixels)
char_width: 32          # Pixels per character (output width = L × 32)
style_dim: 32           # Style vector dimension (z)
embed_dim: 120          # Character embedding dimension
n_class: 80             # Vocabulary size (a-z, A-Z, 0-9, punctuation, blank)
max_word_len: 20        # Maximum text length

# =============================================================================
# GENERATOR ARCHITECTURE
# =============================================================================
GenModel:
  G_ch: 64              # Base channel multiplier
  style_dim: 32         # Style vector size
  embed_dim: 120        # Text embedding size
  bottom_width: 4       # Initial spatial width
  bottom_height: 4      # Initial spatial height
  resolution: 64        # Output height
  G_kernel_size: 3      # Convolution kernel size
  G_attn: '0'           # Attention layer positions (disabled in base)
  num_G_SVs: 1          # Spectral norm singular values
  num_G_SV_itrs: 1      # Power iteration steps
  G_param: 'SN'         # Use Spectral Normalization
  init: 'N02'           # Normal initialization (std=0.02)

# =============================================================================
# DISCRIMINATOR ARCHITECTURE  
# =============================================================================
DiscModel:
  D_ch: 64              # Base channel multiplier
  D_wide: true          # Use wider architecture
  resolution: 64        # Input height
  D_kernel_size: 3      # Convolution kernel size
  num_D_SVs: 1          # Spectral norm singular values
  D_param: 'SN'         # Use Spectral Normalization
  output_dim: 1         # Scalar output (real/fake score)
  one_hot: true         # Use one-hot text conditioning

# Patch Discriminator (for local details)
PatchDiscModel:
  resolution: 32        # Operates on 32×32 patches
  D_ch: 64
  D_wide: true

# =============================================================================
# ENCODER & AUXILIARY NETWORKS
# =============================================================================
StyBackbone:
  resolution: 16        # Downsampling factor
  max_dim: 256          # Maximum channel dimension
  in_channel: 1         # Grayscale input
  norm: 'bn'            # Batch normalization

EncModel:
  style_dim: 32         # Output dimension
  in_dim: 256           # Input from backbone

WidModel:
  n_writer: 372         # Number of writers in IAM dataset
  in_dim: 256           # Input from backbone

OcrModel:
  n_class: 80           # Character classes
  rnn_depth: 2          # BiLSTM layers
  bidirectional: true   # Bidirectional LSTM
  max_dim: 256          # Feature dimension

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
training:
  epochs: 70            # Total training epochs
  batch_size: 24        # Batch size (actual, may differ from config)
  lr: 2.0e-4            # Learning rate
  adam_b1: 0.5          # Adam beta1 (momentum)
  adam_b2: 0.999        # Adam beta2
  
  # Learning rate schedule
  lr_policy: 'linear'   # Linear decay
  start_decay_epoch: 25 # Start decaying LR
  n_epochs_decay: 46    # Epochs to decay over
  
  # GAN training dynamics
  num_critic_train: 4   # D updates per G update (D:G = 4:1)
  
  # VAE mode
  vae_mode: true        # Enable VAE for style encoder
  lambda_kl: 0.0001     # KL divergence weight
  lambda_ctx: 1.0       # Contextual loss weight
  lambda_gram: 2.0      # Gram style loss weight
  
  # Text generation
  capitalize_ratio: 0.5 # 50% chance to capitalize
  blank_ratio: 0.0      # No blank words
  
  # Pretrained models (frozen during training)
  pretrained_w: './pretrained/wid_iam_new.pth'  # Writer ID
  pretrained_r: './pretrained/ocr_iam_new.pth'  # OCR

# =============================================================================
# IMPROVEMENT-SPECIFIC PARAMETERS
# =============================================================================
# Transformer (Novelty 4)
transformer_layers: 2
transformer_heads: 4
transformer_ff_dim: 608  # 4 × combined_dim = 4 × 152

# BiGRU (Novelty 3)
bigru_layers: 1
bigru_hidden: 152       # Same as combined_dim

# Cross-Attention (Novelty 2)
cross_attn_heads: 4
cross_attn_dropout: 0.1

# Self-Attention in Generator
# Applied at resolution 32 and 64 for spatial coherence
```

### Key Dimension Relationships

| Dimension | Value | Derivation |
|-----------|-------|------------|
| `combined_dim` | 152 | `embed_dim + style_dim = 120 + 32` |
| `transformer_ff` | 608 | `4 × combined_dim = 4 × 152` |
| `head_dim` | 38 | `combined_dim / num_heads = 152 / 4` |
| `output_width` | L × 32 | `num_chars × char_width` |
| `bottleneck_channels` | 512 | `8 × G_ch = 8 × 64` |

### Training Dynamics

**Discriminator:Generator Update Ratio = 4:1**
```
Iteration 1: Train D only
Iteration 2: Train D only  
Iteration 3: Train D only
Iteration 4: Train D only
Iteration 5: Train D + Train G  ← G trains every 4th iteration
...
```

**Why 4:1?**
- Gives D time to provide meaningful gradients
- Prevents G from outpacing D (mode collapse)
- Stabilizes adversarial training

---

## 📊 Output Examples

### Dimension Summary

| Stage | Tensor Shape | Description |
|-------|--------------|-------------|
| Input Image | `[B, 1, 64, W]` | Grayscale, height=64, variable width |
| Style Vector | `[B, 32]` | Compressed style representation |
| Text Indices | `[B, L]` | Character indices (max 20) |
| Text Embedding | `[B, L, 120]` | Dense embeddings |
| Fused Features | `[B, L, 152]` | Style + text combined |
| Initial Feature Map | `[B, 512, 4, 4L]` | Before upsampling |
| Block 1 Output | `[B, 256, 8, 4L]` | After first GBlock |
| Block 2 Output | `[B, 128, 16, 8L]` | After second GBlock |
| Block 3 Output | `[B, 64, 32, 16L]` | After third GBlock |
| Block 4 Output | `[B, 64, 64, 32L]` | After fourth GBlock |
| Generated Image | `[B, 1, 64, 32L]` | Final output |

### Width Calculation

For a word with $L$ characters:
$$W_{out} = L \times char\_width = L \times 32$$

Example: "hello" (5 chars) → 160 pixels wide

---

## 🔍 Why These Novelties for Handwriting?

| Novelty | Handwriting Problem | Solution |
|---------|---------------------|----------|
| AdaIN + ModConv | Writers have unique stroke weights, slants | Per-sample style modulation |
| Cross-Attention | Different chars need different style emphasis | Selective style application |
| BiGRU | Ligatures, consistent flow | Sequential dependencies |
| Transformer | Global word structure | All-to-all character relations |
| Contrastive | Mixing writer identity with content | Disentangled style space |
| Multi-Scale D | Both global structure and local texture matter | Specialized discrimination |
| Skip-Connections | Fine strokes lost in deep layers | Multi-scale style injection |
| Positional Encoding | Character order and spacing | Position-aware generation |

---

## 📚 References

- **Original HiGAN**: Handwriting Imitation GAN (CVPR 2021)
- **StyleGAN2**: Analyzing and Improving the Image Quality of StyleGAN
- **BigGAN**: Large Scale GAN Training for High Fidelity Natural Image Synthesis
- **Transformer**: Attention Is All You Need
- **InfoNCE**: Representation Learning with Contrastive Predictive Coding
- **CTC Loss**: Connectionist Temporal Classification (Graves et al.)
- **Spectral Normalization**: Spectral Normalization for GANs (Miyato et al.)
- **AdaIN**: Arbitrary Style Transfer in Real-time (Huang & Belongie)

---

## 🔢 Quick Reference: All Dimensions

| Component | Input Shape | Output Shape | Key Parameters |
|-----------|-------------|--------------|----------------|
| **Text Embedding** | `[B, L]` | `[B, L, 120]` | vocab=80, dim=120 |
| **Style Backbone** | `[B, 1, 64, W]` | `[B, 256, W/16]` | 6 ResBlocks |
| **Style Encoder** | `[B, 256]` | `[B, 32]` | VAE: μ, σ |
| **Cross-Attention** | Q:`[B, L, 152]`, K/V:`[B, 1, 32]` | `[B, L, 152]` | 4 heads, dim=38 |
| **Transformer** | `[B, L, 152]` | `[B, L, 152]` | 2 layers, 4 heads |
| **BiGRU** | `[B, L, 152]` | `[B, L, 152]` | bidirectional |
| **Generator** | `[B, L, 152]` + style | `[B, 1, 64, 32L]` | 4 GBlocks |
| **Discriminator** | `[B, 1, 64, W]` | `[B, 1]` | 5 DBlocks |
| **Recognizer** | `[B, 1, 64, W]` | `[B, W/16, 80]` | BiLSTM+CTC |
| **Writer ID** | `[B, 1, 64, W]` | `[B, 372]` | 372 writers |

---

## 📝 VAE Reparameterization Trick

The Style Encoder uses VAE mode for smooth style interpolation:

```python
def reparameterize(mu, logvar):
    """
    The reparameterization trick enables backprop through sampling.
    
    Problem: We want z ~ N(μ, σ²), but sampling is not differentiable.
    
    Solution: z = μ + σ × ε, where ε ~ N(0, 1)
    
    Now z is a deterministic function of (μ, σ, ε), and gradients
    can flow through μ and σ to the encoder.
    """
    std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log(σ²))
    eps = torch.randn_like(std)     # ε ~ N(0, 1)
    return mu + eps * std           # z = μ + σε
```

**Why VAE for Handwriting?**
- Enables smooth interpolation between writer styles
- Regularizes latent space to prevent overfitting
- Allows random sampling: z ~ N(0,1) generates plausible styles

**KL Loss** keeps the latent distribution close to N(0,1):
$$L_{KL} = -\frac{1}{2}\sum_{j=1}^{32}(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$$

---

*Architecture documentation generated for the HiGAN+ handwriting style transfer system.*
*Last updated: December 2025*
