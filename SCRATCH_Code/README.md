# Handwriting Autocomplete GAN

Documentation for the architecture implemented in `gan.ipynb` for style-conditioned handwriting imitation.

## Project Overview
- Goal: synthesize target words that mimic a specific writer9s penmanship using the IAM handwriting dataset.
- Approach: conditional GAN that blends visual style cues from exemplars with textual content embeddings.
- Key ideas: multi-word style context, character-aware text encoding, style-conditioned generation via AdaIN, and patch-wise adversarial supervision.

## Data Pipeline
- **Vocabulary & Tokenization**: characters mapped to integer ids using helper `text_to_indices`. Padding (`<PAD>`) and unknown (`<UNK>`) tokens keep fixed sequence length per batch.
- **ImprovedIAMWordStyleDataset** (`gan.ipynb`, Cell 12):
  - Builds line-level index so style examples share writer/form/line with the target.
  - Samples up to three consecutive context words, concatenating their glyphs horizontally to stabilize style cues.
  - Pads all images to `64x256` while preserving aspect ratio; converts to tensors in `[-1,1]` for GAN stability.
  - Returns paired tensors: style image, style text indices, target image, target text indices, plus writer id for pretraining.

## Model Architecture

### Text Encoder (BiLSTM)
- Location: `gan.ipynb`, Cell 34 (`TextEncoder`).
- Layers: `nn.Embedding` (128-d), 2-layer bidirectional LSTM (hidden size 256 per direction, dropout 0.3), mean pooling across time, linear projection to 256-d embedding.
- Rationale: bidirectionality captures left/right context in character sequences; mean pooling enforces order-invariant summary suitable for conditioning both generator and discriminator.

### Improved Style Encoder (CNN + Residual)
- Location: Cell 13 (`ImprovedStyleEncoder`).
- Stem: `7x7` conv (stride 2) + BN + ReLU + max pool for coarse stroke capture.
- Downsampling: three `_make_layer` stages, each halving spatial resolution with stride-2 conv followed by residual blocks (`ResidualBlock`) to refine features without losing style identity.
- Global Average Pooling reduces spatial map to 512-d vector.
- MLP head: `512->1024->style_dim(512)` with dropout 0.3 to form the style embedding.
- Optional writer classifier head enables supervised pretraining, promoting discriminative style features.

### Adaptive Instance Normalization (AdaIN) Stack
- Location: Cell 14 (`AdaIN`, `ModulatedResBlock`).
- AdaIN normalizes generator activations channel-wise then re-scales/shifts them using affine transforms derived from the 512-d style embedding.
- `ModulatedResBlock` pairs two conv layers with AdaIN, preserving residual connections so style cues modulate, rather than overwrite, content structure.

### Improved Generator (Style-Conditioned Decoder)
- Location: Cell 14 (`ImprovedGenerator`).
- Text fusion MLP combines style-text and target-text embeddings (`512 -> 256`) to separate handwriting style vocabulary from requested content.
- Initial projection maps concatenated (style + fused text) vector into a `(256, 4, 16)` feature map.
- Encoder pre-block refines the base grid before modulation.
- Six `ModulatedResBlock`s inject style statistics at every depth, maintaining consistency across the generated word.
- Decoder: sequence of transpose-conv blocks (stride 2) with InstanceNorm + ReLU gradually upsamples to `(1, 64, 256)` and final `Tanh` outputs normalized grayscale.
- Reasoning: AdaIN ensures writer-specific stroke patterns persist, while text fusion keeps content faithful to requested spelling.

### Patch Discriminator (Text-Conditioned PatchGAN)
- Location: Cell 15 (`PatchDiscriminator`).
- Input: concatenation of style image and target/generated image across channels (2-channel tensor).
- CNN backbone: stride-2 `4x4` conv blocks with LeakyReLU and BatchNorm, producing `(batch, 512, 4, 16)` feature map (patch granularity).
- Text conditioning: separate MLP maps target text embedding to 512-d, then spatially broadcasts via linear layer to match the feature map size, appended as an extra channel.
- Final head: conv stack ending in sigmoid map `(1, 4, 16)` measuring authenticity per patch.
- Motivation: local patch judgments better capture pen stroke fidelity and force generator to maintain coherence with style cues.

## Training Strategy
- **Stage 1  Style Encoder Pretraining** (Cell 18): cross-entropy writer classification builds robust style embeddings; employs Adam (lr=1e-3).
- **Stage 2  Generator Pretraining** (Cell 19): freezes style encoder, optimizes text encoder + generator using weighted L1 (10x) and L2 losses to match ground-truth images; visualizes progress every 5 epochs.
- **Stage 3  Full GAN Training** (Cells 202):
  - Discriminator uses BCE over patch logits with real/fake labels shaped to `(1,4,16)`.
  - Generator loss mixes adversarial term (weight anneals from 0.1 to 1.0) and content L1 (scaled by 50); optional feature matching aligns discriminator feature maps (`conv_layers`).
  - Adaptive schedule skips generator updates when discriminator accuracy exceeds 85% to avert overfitting.
  - Checkpoints saved every `save_interval` epochs with optimizer states and history.

## Loss Functions
- `L_adv`: patch-wise binary cross-entropy versus real labels.
- `L_content`: primarily L1 distance for sharp glyph reproduction (Stage 2 and 3).
- `L_fm`: feature-matching L1 between discriminator activations (scaled by 10) stabilizes gradients.
- Stage-specific weighting ensures smooth transition from supervised reconstruction to adversarial refinement.

## Inference Workflow
- After training, load encoder/generator weights from checkpoints (Stage 3). 
- Encode style image(s) and desired target text via `style_encoder` and `text_encoder`.
- Run `generator(style_embed, style_text_embed, target_text_embed)`; rescale from `[-1,1]` back to `[0,1]` or `[0,255]` for visualization.
- Notebook cells labeled "Inference and Testing" (after Cell 29) provide plotting utilities for generated vs. reference comparisons.

## Design Rationale Summary
- Multi-word context and residual CNN encoder capture consistent handwriting traits beyond single glyph noise.
- BiLSTM text encoder preserves character order while producing fixed-length conditioning vectors.
- AdaIN-based generator allows continuous style manipulation without retraining per writer.
- PatchGAN discriminator plus feature matching maintains local stroke realism and stabilizes GAN dynamics.
- Progressive training and adaptive update gating mitigate mode collapse and handwriting legibility loss.

Refer to `gan.ipynb` for executable code, dataset preparation steps, and visualization helpers aligned with this architecture description.
