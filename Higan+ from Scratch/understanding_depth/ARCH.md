# HiGAN+ Architecture Deep Dive

## 🏗️ Architecture Overview

HiGAN+ generates handwritten text by combining **content** (what to write) and **style** (how to write it). Here's the complete architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     HiGAN+ ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GENERATION PATH:                                            │
│  Style Vector (z) + Text Labels (y) → Generator → Fake Image│
│                                                              │
│  DISCRIMINATION PATH:                                         │
│  Real/Fake Image → Discriminator → Real/Fake Score          │
│  Real/Fake Image → Patch Discriminator → Local Scores       │
│                                                              │
│  AUXILIARY NETWORKS:                                         │
│  Image → Style Encoder → Style Vector                       │
│  Image → Recognizer (OCR) → Text Prediction                 │
│  Image → Writer Identifier → Writer ID                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ **Generator** (Creates Handwriting)

### **Purpose**: Generate handwritten text images from style vectors and text labels

### **Architecture Flow**:

```python
Input:
  - z: Style vector [batch_size, 32] (controls handwriting style)
  - y: Text labels [batch_size, seq_len] (word to write, e.g., "hello")
  - y_lens: Text lengths [batch_size] (number of characters)

Step 1: Text Embedding
  y → Embedding Layer → [batch_size, seq_len, 120]

Step 2: Style Conditioning
  z → Linear Layer → [batch_size, 32 * num_blocks]
  Split into multiple z chunks for each resolution block

Step 3: Combine Style + Text
  z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # [B, L, 32]
  combined = concat(z_expanded, text_embedding)       # [B, L, 152]

Step 4: Initial Feature Map
  combined → Linear → Reshape to [B, 512, 4, 4*seq_len]
  # Creates a vertical "strip" for each character

Step 5: Upsampling Blocks (4 blocks)
  [B, 512, 4, 4L] → GBlock + ConditionalBatchNorm → [B, 256, 8, 4L]
  [B, 256, 8, 4L] → GBlock + Upsample(2,1) → [B, 128, 16, 4L]
  [B, 128, 16, 4L] → GBlock + Upsample(2,2) → [B, 64, 32, 8L]
  [B, 64, 32, 8L] → GBlock + Upsample(2,2) → [B, 64, 64, 16L]
  # Gradually increases spatial resolution while decreasing channels

Step 6: Output Layer
  [B, 64, 64, 16L] → BatchNorm → ReLU → Conv(64→1) → Tanh
  Output: [B, 1, 64, width] where width depends on text length

Step 7: Masking (Inference only)
  Apply length mask to blank out padding regions
```

### **Key Components**:

#### **GBlock (Generator Block)**:
```python
class GBlock:
    def forward(h, y):
        # y is the style vector for this block
        h1 = ConditionalBatchNorm(h, y)  # Style modulation
        h1 = ReLU(h1)
        h1 = Upsample(h1)  # Double height/width
        h1 = Conv3x3(h1)
        
        h2 = ConditionalBatchNorm(h1, y)
        h2 = ReLU(h2)
        h2 = Conv3x3(h2)
        
        # Skip connection
        h_skip = Upsample(h)
        h_skip = Conv1x1(h_skip)
        
        return h2 + h_skip  # Residual connection
```

#### **Conditional BatchNorm**:
```python
# Modulates features based on style vector
def ConditionalBatchNorm(x, z):
    x_normalized = BatchNorm(x)
    gamma = Linear(z)  # Scale parameter from style
    beta = Linear(z)   # Shift parameter from style
    return gamma * x_normalized + beta
```

### **Example Dimensions**:
```
Text: "hello" (5 characters)
z: [8, 32] (batch=8, style_dim=32)
y: [8, 5] (batch=8, seq_len=5)

After embedding: [8, 5, 120]
After combine: [8, 5, 152]
After linear: [8, 512, 4, 20]  # 4*5=20 width
After 4 GBlocks: [8, 64, 64, 80]  # 16*5=80 width
After output: [8, 1, 64, 80]  # Final handwriting image
```

---

## 2️⃣ **Discriminator** (Global Realism Checker)

### **Purpose**: Classify entire images as real or fake

### **Architecture Flow**:

```python
Input:
  - x: Image [batch_size, 1, 64, width]
  - x_lens: Image widths [batch_size]
  - y_lens: Text lengths [batch_size]

Step 1: Convolutional Downsampling (4 DBlocks)
  [B, 1, 64, W] → DBlock + AvgPool → [B, 64, 32, W/2]
  [B, 64, 32, W/2] → DBlock + AvgPool → [B, 128, 16, W/4]
  [B, 128, 16, W/4] → DBlock + AvgPool → [B, 256, 8, W/8]
  [B, 256, 8, W/8] → DBlock → [B, 256, 8, W/8]

Step 2: Optional Attention
  # At resolution 64x64, apply self-attention
  [B, 256, 8, W/8] → SelfAttention → [B, 256, 8, W/8]

Step 3: Global Pooling (Length-Aware)
  # Apply mask to handle variable-length images
  mask = create_mask_from_lengths(x_lens)
  h_masked = h * mask
  h_pooled = sum(h_masked) / y_lens  # Average over valid regions

Step 4: Classification
  h_pooled → Linear(256 → 1) → Real/Fake score

Output: [batch_size, 1] (probability of being real)
```

### **DBlock (Discriminator Block)**:
```python
class DBlock:
    def forward(h):
        h1 = ReLU(h)
        h1 = Conv3x3(h1)  # With Spectral Normalization
        h1 = ReLU(h1)
        h1 = Conv3x3(h1)
        h1 = AvgPool2d(h1)  # Downsample
        
        # Skip connection
        h_skip = Conv1x1(h)
        h_skip = AvgPool2d(h_skip)
        
        return h1 + h_skip
```

### **Spectral Normalization**:
```python
# Stabilizes training by constraining weight matrix norms
def SpectralNorm(W):
    u, sigma, v = SVD(W)  # Singular Value Decomposition
    W_normalized = W / sigma_max  # Divide by largest singular value
    return W_normalized
```

---

## 3️⃣ **Patch Discriminator** (Local Realism Checker)

### **Purpose**: Check if small patches look realistic (catches local artifacts)

### **Architecture**:

```python
Input: Extracted patches [batch_size, 1, patch_h, patch_w]

Step 1: Patch Extraction (done externally)
  extract_all_patches(image, img_lens)
  # Extracts overlapping 70x70 patches from image

Step 2: Multi-Scale Conv Layers (3 layers)
  [B, 1, 70, 70] → Conv(1→64, k=3, s=2) → ReLU → [B, 64, 35, 35]
  [B, 64, 35, 35] → Conv(64→128, k=3, s=2) → ReLU → [B, 128, 17, 17]
  [B, 128, 17, 17] → Conv(128→256, k=3, s=2) → ReLU → [B, 256, 8, 8]

Step 3: Score Map
  [B, 256, 8, 8] → Conv(256→1, k=3, s=1) → [B, 1, 8, 8]
  # Each spatial location = score for that receptive field

Step 4: Aggregate Scores
  score_map → Global Average Pool → [B, 1]

Output: [batch_size, 1] (average patch realism score)
```

### **Why Two Discriminators?**
- **Global D**: Checks overall structure, layout, consistency
- **Patch D**: Checks fine details, texture, local quality
- **Together**: Prevents blurry images AND weird local artifacts

---

## 4️⃣ **Style Backbone** (Feature Extractor)

### **Purpose**: Extract hierarchical features from handwriting images (shared by Style Encoder and Writer Identifier)

### **Architecture**:

```python
Input: Image [batch_size, 1, 64, width]

Step 1: Initial Conv
  [B, 1, 64, W] → ConstantPad2d(-1) → Conv(1→16, k=5, s=2) → [B, 16, 32, W/2]

Step 2: Early Feature Blocks (Resolution 32→8)
  [B, 16, 32, W/2] → 2x ResBlock → MaxPool → [B, 32, 16, W/4]
  [B, 32, 16, W/4] → 2x ResBlock → MaxPool → [B, 64, 8, W/8]

Step 3: Deep Feature Blocks (Resolution 8)
  [B, 64, 8, W/8] → 2x ResBlock → [B, 128, 8, W/8]
  [B, 128, 8, W/8] → 2x ResBlock → [B, 256, 8, W/8]

Step 4: CTC Head (for sequence modeling)
  [B, 256, 8, W/8] → ReLU → Conv(256→256, k=3) → Squeeze height
  Output: [B, 256, W/16] (sequence of feature vectors)

Intermediate Outputs (for loss):
  feat2: [B, 32, 16, W/4]   # Early features
  feat3: [B, 64, 8, W/8]    # Mid features  
  feat4: [B, 128, 8, W/8]   # Deep features
```

### **ResBlock (Residual Block)**:
```python
class ActFirstResBlock:
    def forward(x):
        h = ReLU(x)
        h = Conv3x3(h)
        h = Dropout(h)
        h = ReLU(h)
        h = Conv3x3(h)
        return h + x  # Skip connection
```

---

## 5️⃣ **Style Encoder** (Extracts Style Vector)

### **Purpose**: Convert handwriting image into a style vector that captures writing characteristics

### **Architecture**:

```python
Input: 
  - img: [batch_size, 1, 64, width]
  - img_lens: [batch_size] (valid image widths)

Step 1: Extract Features
  img → StyleBackbone → feat: [B, 256, W/16]

Step 2: Length-Aware Pooling
  # Average only over valid (non-padding) regions
  mask = create_mask_from_lengths(img_lens // 16)
  feat_masked = feat * mask
  style_feat = sum(feat_masked, dim=-1) / img_lens
  # Output: [B, 256]

Step 3: Style MLP
  [B, 256] → Linear(256→256) → LeakyReLU
  [B, 256] → Linear(256→256) → LeakyReLU
  [B, 256] → Linear(256→32) → mu (mean)

Step 4: VAE Mode (Optional)
  [B, 256] → Linear(256→32) → logvar (log variance)
  z = mu + eps * exp(0.5 * logvar)  # Reparameterization trick
  Output: (z, mu, logvar)

Non-VAE Mode:
  Output: mu (just use mean as style vector)
```

### **Why Length-Aware Pooling?**
```python
# Example:
# Word "hi" → image width = 40
# Word "hello" → image width = 100
# Without masking: padding affects style vector
# With masking: only actual handwriting contributes
```

---

## 6️⃣ **Recognizer** (OCR Network)

### **Purpose**: Ensure generated text is readable (acts as quality controller)

### **Architecture**:

```python
Input: Image [batch_size, 1, 64, width]

Step 1: CNN Backbone (Same as StyleBackbone)
  [B, 1, 64, W] → CNN layers → [B, 256, 8, W/16]

Step 2: CTC Head
  [B, 256, 8, W/16] → Conv(256→256) → Squeeze height
  [B, 256, W/16] → Transpose → [B, W/16, 256]

Step 3: Bidirectional LSTM (Optional)
  [B, W/16, 256] → BiLSTM(256→256) → [B, W/16, 256]
  # Captures sequential dependencies

Step 4: Character Classification
  [B, W/16, 256] → Linear(256→80) → [B, W/16, 80]
  # 80 classes = alphabet + blank token

Step 5: CTC Loss
  predictions: [B, W/16, 80]
  targets: [B, max_label_len]
  ctc_loss(predictions, targets, pred_lens, target_lens)

Output: Log probabilities [B, W/16, 80]
```

### **CTC (Connectionist Temporal Classification)**:
```python
# Problem: Alignment between image and text unknown
# Solution: CTC learns all possible alignments

Example:
Image: "hello" (width = 80 pixels → 5 time steps after CNN)
CTC outputs: [blank, h, e, l, l, l, o, blank, ...]
Collapse repeats and blanks → "hello"

# Allows variable-length outputs from fixed-length inputs
```

---

## 7️⃣ **Writer Identifier** (Who Wrote This?)

### **Purpose**: Classify which writer produced the handwriting (preserves individual style)

### **Architecture**:

```python
Input:
  - img: [batch_size, 1, 64, width]
  - img_lens: [batch_size]

Step 1: Feature Extraction
  img → StyleBackbone → feat: [B, 256, W/16]

Step 2: Length-Aware Pooling
  mask = create_mask_from_lengths(img_lens // 16)
  wid_feat = sum(feat * mask, dim=-1) / img_lens
  # Output: [B, 256]

Step 3: Writer Classification
  [B, 256] → Linear(256→256) → LeakyReLU
  [B, 256] → Linear(256→372)  # 372 writers in IAM dataset

Output: [batch_size, 372] (writer ID logits)
```

---

## 🔄 **Complete Training Flow**

### **1. Discriminator Training Step**:

```python
# Get real images
real_imgs, real_lbs, real_lb_lens = batch

# Generate fake images (3 types)
z_random = sample_noise()
fake_imgs_random = Generator(z_random, random_text)

z_style = StyleEncoder(real_imgs)
fake_imgs_style = Generator(z_style, random_text)

fake_imgs_recn = Generator(z_style, real_lbs)  # Reconstruction

# Discriminate
real_score = Discriminator(real_imgs)
fake_score = Discriminator(cat([fake_imgs_random, fake_imgs_style, fake_imgs_recn]))

# Hinge loss
d_loss = relu(1 - real_score) + relu(1 + fake_score)

# Patch discriminator (same logic)
real_patches = extract_patches(real_imgs)
fake_patches = extract_patches(fake_imgs)
patch_d_loss = similar_hinge_loss(real_patches, fake_patches)

# Total discriminator loss
total_d_loss = d_loss + patch_d_loss
```

### **2. Generator Training Step**:

```python
# Generate fake images
z = sample_noise()
fake_imgs = Generator(z, text)

z_style = StyleEncoder(real_imgs)
style_imgs = Generator(z_style, text)
recn_imgs = Generator(z_style, real_text)

# Adversarial loss (fool discriminators)
adv_loss = -Discriminator(fake_imgs) - PatchDiscriminator(patches)

# CTC loss (readability)
ocr_pred = Recognizer(fake_imgs)
ctc_loss = CTC(ocr_pred, text)

# Writer ID loss (preserve style)
wid_pred = WriterIdentifier(style_imgs)
wid_loss = CrossEntropy(wid_pred, real_writer_id)

# Reconstruction loss (content fidelity)
recn_loss = L1(recn_imgs, real_imgs)

# Style consistency loss
z_reconstructed = StyleEncoder(fake_imgs)
info_loss = L1(z_reconstructed, z)

# Contextual loss (texture matching)
real_feats = StyleBackbone(real_imgs, ret_feats=True)
fake_feats = StyleBackbone(fake_imgs, ret_feats=True)
ctx_loss = ContextualLoss(real_feats, fake_feats)

# Total generator loss (weighted sum)
g_loss = adv_loss + λ_ctc*ctc_loss + λ_wid*wid_loss + 
         λ_recn*recn_loss + λ_info*info_loss + λ_ctx*ctx_loss
```

---

## 📊 **Architecture Summary Table**

| **Component** | **Input** | **Output** | **Purpose** |
|---------------|-----------|------------|-------------|
| **Generator** | Style vector (32D) + Text labels | Image (1×64×W) | Create handwriting |
| **Discriminator** | Image (1×64×W) | Real/Fake score | Check global realism |
| **Patch Discriminator** | Image patches | Patch scores | Check local details |
| **Style Backbone** | Image (1×64×W) | Features (256×W/16) | Extract features (shared) |
| **Style Encoder** | Image (1×64×W) | Style vector (32D) | Extract writing style |
| **Recognizer (OCR)** | Image (1×64×W) | Character probs (80×W/16) | Verify readability |
| **Writer Identifier** | Image (1×64×W) | Writer ID (372D) | Identify writer |

---

## 🎯 **Key Innovations**

1. **Hierarchical Style Injection**: Different style vectors for each resolution in Generator
2. **Dual Discrimination**: Global + Patch discriminators catch different types of artifacts
3. **Length-Aware Operations**: Handles variable-length text naturally
4. **Multi-Task Learning**: OCR + Writer ID losses improve quality
5. **Style Consistency**: Generated images must fool the Style Encoder too

This architecture enables:
- ✅ Generating realistic handwriting
- ✅ Controlling writing style independently from content
- ✅ Copying style from reference images
- ✅ Maintaining text readability
- ✅ Preserving individual writer characteristics