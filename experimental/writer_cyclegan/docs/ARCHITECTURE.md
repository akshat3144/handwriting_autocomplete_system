**Architecture Overview — CycleGAN (writer-aware) with OCR & Style Loss**

This document summarizes the project's high-level architecture, data flow, and training/inference loop. It focuses on the modifications that make the model writer-aware and preserve text readability.

---

## How CycleGAN Works (Simple Explanation)

CycleGAN learns to translate images between two domains (A and B) without paired examples.

**Core idea:**
- Two generators: G_A (A→B) and G_B (B→A)
- Two discriminators: D_A (judges if image looks like domain A) and D_B (judges if image looks like domain B)
- Cycle-consistency: If you translate A→B→A, you should get back the original A

**Training loop (simplified):**
```
1. Take real_A from domain A
2. Generate fake_B = G_A(real_A)           # translate A to B
3. Reconstruct rec_A = G_B(fake_B)         # translate back to A
4. Loss = |rec_A - real_A|                 # cycle loss forces consistency
5. D_B tries to distinguish fake_B from real_B  # adversarial loss
6. Same process in reverse for B→A→B
```

**Why it works:** The cycle-consistency constraint prevents the generator from producing arbitrary outputs — it must preserve enough information to reconstruct the original.

---

## Network Architecture Details

### CNN Specifications Summary

| Component | Kernel Size | Stride | Padding | Filters/Channels | Activation |
|-----------|-------------|--------|---------|------------------|------------|
| Generator first conv | 7×7 | 1 | 3 (reflect) | 64 (ngf) | ReLU |
| Generator downsample | 3×3 | 2 | 1 | 64→128→256 | ReLU |
| ResNet blocks | 3×3 | 1 | 1 (reflect) | 256 | ReLU |
| Generator upsample | 3×3 (transposed) | 2 | 1 | 256→128→64 | ReLU |
| Generator final conv | 7×7 | 1 | 3 (reflect) | 3 (output_nc) | Tanh |
| Discriminator conv | 4×4 | 2 | 1 | 64→128→256→512 | LeakyReLU(0.2) |
| Discriminator final | 4×4 | 1 | 1 | 1 | None (logits) |
| StyleEncoder conv | 4×4 | 2 | 1 | 64→128→256→512 | LeakyReLU(0.2) |
| StyleEncoder FC | - | - | - | 512→256→128 | ReLU→Linear |

### Activation Functions Used

| Activation | Where Used | Formula |
|------------|------------|---------|
| ReLU | Generator (all layers except output) | max(0, x) |
| LeakyReLU(0.2) | Discriminator, StyleEncoder | max(0.2x, x) |
| Tanh | Generator output layer | (e^x - e^-x)/(e^x + e^-x), outputs [-1,1] |

### Normalization Layers

| Type | Where Used | Properties |
|------|------------|------------|
| InstanceNorm2d | Generator (default) | Per-sample, per-channel normalization; no learnable params |
| BatchNorm2d | Discriminator (optional) | Cross-batch statistics; has learnable affine params |

### Optimizer Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Adaptive learning rate per parameter |
| Learning rate | 0.0002 | Standard for GANs |
| beta1 | 0.5 | Momentum term (lower than default 0.9 for GAN stability) |
| beta2 | 0.999 | Second moment term |
| LR policy | linear | Constant for n_epochs, then linear decay to 0 |

### Weight Initialization

| Method | Where Used | Distribution |
|--------|------------|--------------|
| Normal | Conv, Linear layers | N(0, 0.02) |
| Xavier | Alternative option | Scaled by fan_in/fan_out |
| Kaiming | Alternative option | Scaled for ReLU activations |

---

1) High-level components
- Generator (ResnetGenerator): ResNet-based image-to-image network (default `resnet_9blocks`). When `--embed_dim>0` the generator concatenates a tiled style embedding to the input channels before the first convolution. Input: `[B, C, H, W]` (+ style channels). Output: `[B, C, H, W]` (Tanh in [-1,1]).
- Discriminator (NLayerDiscriminator / PixelDiscriminator): PatchGAN-style discriminators that output a prediction map for real/fake patches.
- StyleEncoder: CNN that converts 1 or N reference images into a per-sample style vector `[B, embed_dim]`. Multiple reference images are averaged to form a stable style.
- WriterEmbedding (legacy): `nn.Embedding` for datasets with fixed writer IDs (useful when labels exist; replaced by `StyleEncoder` for unseen writers).
- OCRLoss: lightweight OCR consistency loss using EasyOCR (if available) — currently uses character-count proxy features and normalized L1 between fake/real.
- StyleLoss: currently a pixel-level L1 between generated image and style reference (cheap and stable; can be swapped for VGG/Gram-based losses).
- ImagePool: history buffer for fake images used to stabilize discriminator updates.
- Visualizer / HTML / WandB: logging, image gallery and remote experiment tracking.

2) Dataflow (training iteration)
1. `set_input(data)` in `cycle_gan_model.py` loads `real_A`, `real_B` and optional `ref_images` or `writer_ids`.
2. If `embed_dim>0`, `netstyle_encoder(reference_images)` -> `style_vector` with shape `[B, embed_dim]`.
3. Forward pass:
   - `fake_B = netG_A(real_A, writer_style=style_vector)` (A→B conditional on style)
   - `fake_A = netG_B(real_B, writer_style=style_vector_other)` (B→A)
4. Cycle: `rec_A = netG_B(fake_B, writer_style_of_A)` and `rec_B = netG_A(fake_A, writer_style_of_B)`.
5. Compute losses:
   - GAN loss: `GANLoss(netD(fake), True/False)`
   - Cycle-consistency: `L1(rec_A, real_A)` + `L1(rec_B, real_B)`
   - Identity (optional): `L1(netG_A(real_B), real_B)`
   - Style loss: `style_loss.compute_loss(generated, style_reference) * lambda_style`
   - OCR loss: `ocr_loss.compute_loss(fake, real) * lambda_OCR` (may be computed every K steps)
6. Backprop: `backward_G()` (sum of above), update G & style encoder params; then `backward_D()` updates discriminators (possibly using `ImagePool.query` for fakes).

3) Key design choices & rationale
- Conditioning by concatenation: The style vector is tiled spatially and concatenated to the input channels. This is simple and effective for global style signals like handwriting stroke width and slant.
- Dynamic StyleEncoder vs. fixed embeddings: `StyleEncoder` supports unseen writers at test time by extracting style from reference images; `WriterEmbedding` is only useful when writer IDs are known and fixed.
- OCR consistency: adds an explicit readability objective to prevent style transfer from distorting characters; implemented as a lightweight proxy to avoid full feature matching overhead.
- Style loss L1 (pixel-space): chosen for stability; Gram/VGG-based alternatives give richer texture transfer but require more compute and tuning.

4) Checkpoints & outputs
- Checkpoints saved under `checkpoints/<name>/` with files like `latest_net_G_A.pth`, `latest_net_style_encoder.pth`.
- Visual outputs and HTML saved under `checkpoints/<name>/web/` and `checkpoints/<name>/paragraph_test/` from test scripts.
- Metrics JSON (`metrics.json`) is written by `util/metrics.py` in the checkpoint folder.

5) Training tips & recommended hyperparameters (starting point)
- Image size: `--load_size 512 --crop_size 512` for better style fidelity.
- Batch size: `1` (typical for image-to-image CycleGAN runs).
- LR: `0.0002` with Adam (`beta1=0.5`, `beta2=0.999`).
- `--embed_dim 128` for style vector dimension.
- Loss weights: `--lambda_style 1.0`, `--lambda_OCR 0.05` (start small for OCR so readability remains strong).
- If OOM at high resolution: reduce load/crop size or use mixed precision (AMP) and/or gradient accumulation.

6) Inference / test-time
- Supply one or more reference images to `StyleEncoder` to extract writer style.
- Call `netG_A(real_A, writer_style=style_vector)` to produce stylized output.
- Test scripts `test_paragraph.py`, `test_custom_style.py`, and `test_three_lines.py` show minimal examples of generating paragraphs and custom-style images.

7) Extension points
- Replace `StyleLoss` with a VGG-feature Gram loss for richer texture transfer.
- Replace OCR proxy with feature-based OCR loss (e.g., compare intermediate OCR network features) for better semantics.
- Experiment with conditional normalization (AdaIN) instead of channel concatenation to inject style.

References (where to look in code)
- Generator/Discriminator: `models/networks.py` (look at `ResnetGenerator` and `NLayerDiscriminator`).
- Main logic: `models/cycle_gan_model.py` (forward, backward_G, optimize_parameters).
- Style extraction: `models/style_encoder.py`.
- Loss helpers: `models/style_loss.py` and `models/ocr_loss.py`.
- Utilities: `util/visualizer.py`, `util/metrics.py`, `util/image_pool.py`.

---

## Loss Functions Summary

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| GAN (LSGAN) | MSE(D(fake), 1) | 1.0 | Make generated images realistic |
| Cycle | L1(rec_A, real_A) + L1(rec_B, real_B) | 10.0 | Preserve content through cycle |
| Identity | L1(G_A(real_B), real_B) | 5.0 | Preserve color/texture when input is already target domain |
| Style | L1(generated, style_ref) | lambda_style | Match handwriting style |
| OCR | L1(ocr_features(fake), ocr_features(real)) | lambda_OCR | Preserve text readability |

---

## Generator Architecture (ResNet-9blocks)

```
Input [B, 3, 256, 256]
    ↓
ReflectionPad2d(3) + Conv2d(3→64, k=7) + InstanceNorm + ReLU
    ↓ [B, 64, 256, 256]
Conv2d(64→128, k=3, s=2) + InstanceNorm + ReLU  (downsample)
    ↓ [B, 128, 128, 128]
Conv2d(128→256, k=3, s=2) + InstanceNorm + ReLU (downsample)
    ↓ [B, 256, 64, 64]
9× ResNet Blocks (Conv-Norm-ReLU-Conv-Norm + skip connection)
    ↓ [B, 256, 64, 64]
ConvTranspose2d(256→128, k=3, s=2) + InstanceNorm + ReLU (upsample)
    ↓ [B, 128, 128, 128]
ConvTranspose2d(128→64, k=3, s=2) + InstanceNorm + ReLU (upsample)
    ↓ [B, 64, 256, 256]
ReflectionPad2d(3) + Conv2d(64→3, k=7) + Tanh
    ↓
Output [B, 3, 256, 256] in range [-1, 1]
```

---

## Discriminator Architecture (PatchGAN)

```
Input [B, 3, 256, 256]
    ↓
Conv2d(3→64, k=4, s=2) + LeakyReLU(0.2)
    ↓ [B, 64, 128, 128]
Conv2d(64→128, k=4, s=2) + InstanceNorm + LeakyReLU(0.2)
    ↓ [B, 128, 64, 64]
Conv2d(128→256, k=4, s=2) + InstanceNorm + LeakyReLU(0.2)
    ↓ [B, 256, 32, 32]
Conv2d(256→512, k=4, s=1) + InstanceNorm + LeakyReLU(0.2)
    ↓ [B, 512, 31, 31]
Conv2d(512→1, k=4, s=1)
    ↓
Output [B, 1, 30, 30] (patch predictions)
```

The output is a 30×30 grid where each value indicates real/fake for a 70×70 receptive field patch.

---

This overview is meant to give you a focused mental map of the repository and where to make future changes. 
