# Higan+ from Scratch: Essential Guide

## What is Higan+?

Higan+ is a **Generative Adversarial Network (GAN)** that generates realistic handwritten text by learning and mimicking a person's handwriting style. Built from scratch using **PyTorch** and trained on the **IAM handwriting dataset** (372 writers).

## Key Libraries
- **PyTorch**: Neural network framework (GPU-accelerated)
- **HDF5/h5py**: Efficient dataset storage
- **OpenCV, NumPy**: Image processing
- **YAML**: Configuration management
- **Matplotlib, TensorBoard**: Visualization and logging

## Dataset & Preprocessing
- **IAM Dataset**: 372 writers, stored in HDF5 format
- **Image Size**: 64px height, variable width (32px per character)
- **Normalization**: Pixel values scaled to [-1, 1]
- **Text Encoding**: 80-character CTC alphabet
- **Augmentation**: Random scaling and clipping

## Model Architecture (7 Neural Networks)

### 1. **Style Backbone** (Feature Extractor)
- Conv2D layers (3×3 kernels, stride 2) with BatchNorm + LeakyReLU
- Channels: 64 → 128 → 256
- Output: 256-dim feature vector

### 2. **Style Encoder** (32-dim style code)
- Gated linear layers (MLP)
- VAE mode: outputs μ and σ for style sampling
- Compresses 256-dim features → 32-dim style vector

### 3. **Generator** (The Artist - BigGAN-inspired)
- Input: 32-dim style + 120-dim character embeddings
- Residual blocks with upsampling (3×3 kernels)
- **AdaIN (Adaptive Instance Normalization)** for style control
- Output: 1-channel grayscale image (64×width pixels)

### 4. **Discriminator** (Global Critic)
- Downsampling ResBlocks (3×3 kernels, stride 2)
- **Spectral Normalization** for training stability
- Output: Real/fake score

### 5. **Patch Discriminator** (Local Detail Critic)
- Operates on 32×32 patches
- Ensures fine texture quality

### 6. **Recognizer/OCR** (Readability Checker)
- CNN + BiLSTM (2 layers)
- CTC decoding for text recognition
- Ensures generated text is legible

### 7. **Writer Identifier** (Style Verifier)
- Classification head (256 → 372 writer classes)
- Ensures style matches reference writer

## Training Configuration

### Optimizer
- **Adam**: lr=2e-4, β1=0.5, β2=0.999
- **LR Schedule**: Linear decay from epoch 25 to 70
- **Training Ratio**: Discriminator trained 4× per Generator step

### Loss Functions (8 Total)
1. **Hinge GAN Loss**: Adversarial training (real vs fake)
2. **CTC Loss**: Ensures text readability
3. **Writer ID Loss**: Matches writer style (cross-entropy)
4. **Reconstruction Loss**: L1 pixel difference
5. **Contextual Loss**: High-level feature matching (λ=2.0)
6. **KL Loss**: VAE regularization (λ=0.0001)
7. **Info Loss**: Style preservation
8. **Patch Loss**: Local texture quality

### Normalization
- **BatchNorm**: In backbone and base layers
- **AdaIN**: In generator for style control
- **Spectral Norm**: In discriminators for stability

### Key Metrics
- **CER**: 2.96% (Character Error Rate)
- **WER**: 10% (Word Error Rate)
- **FID, KID, IS**: Image quality scores
- **MSSIM, PSNR**: Reconstruction similarity

## Training Process

**Setup**: 70 epochs, batch size 8  
**Each Iteration**:
1. Train Discriminator (4 steps): Judge real vs fake images
2. Train Generator (1 step): Create realistic handwriting with 8 combined losses
3. Update learning rates, save checkpoints every 2 epochs

## Inference Modes
- **Style-Guided**: Use reference handwriting to generate new text
- **Random Style**: Generate text with random handwriting styles
- **Interpolation**: Smoothly blend between two writing styles
- **Custom Transfer**: Write any text in a specific person's handwriting

## Configuration
- All hyperparameters in `gan_iam.yml`
- Image: 64px height, 32px per character
- Style vector: 32-dim, Character embedding: 120-dim
- Total parameters: ~11 million

---

## How The Model Works (In Brief)

**Data Flow**: The model reads handwritten word images from the IAM dataset stored in HDF5 format. Each image (64×width pixels) is normalized to [-1,1] and paired with its text label and writer ID. Images undergo random augmentation (scaling/clipping) for robustness.

**Training**: During training, the system operates as an adversarial game between 7 neural networks. First, the Style Backbone extracts 256-dimensional features from a reference handwriting image, which the Style Encoder compresses into a compact 32-dimensional "style code" that captures the writer's unique characteristics (slant, thickness, spacing). The Generator then takes this style code plus character embeddings (learned 120-dim vectors for each letter) and synthesizes a handwriting image using hierarchical residual blocks with AdaIN layers that modulate features at each level to inject the style. Meanwhile, two Discriminators (global and patch-based) judge whether images look real or fake, providing adversarial training signal. The Recognizer (OCR network with BiLSTM) ensures the generated text is readable by computing CTC loss against the target text, while the Writer Identifier verifies that the style matches the reference writer. The Generator is trained with 8 combined losses: adversarial (fool discriminators), CTC (be readable), writer ID (match style), reconstruction (reproduce input accurately), contextual (match high-level features), KL divergence (regularize style space), info loss (preserve style), and patch quality. The Discriminator is trained 4 times per Generator step to maintain balance. Adam optimizer with learning rate 2e-4 decays linearly after epoch 25. All networks use 3×3 convolutional kernels, BatchNorm for stable features, AdaIN for style control, and Spectral Normalization in discriminators for training stability.

**Output Generation**: After training, to generate handwriting, you provide a reference image (the Style Encoder extracts its 32-dim style code) and target text (converted to character embeddings). The Generator fuses these through its AdaIN-modulated residual blocks to produce a grayscale image where the text appears in the reference writer's style. You can also sample random style vectors to create novel handwriting styles, or interpolate between two style codes to morph between writing styles smoothly. The model achieves 2.96% character error rate and 10% word error rate on the test set, demonstrating it generates both realistic and legible handwriting.