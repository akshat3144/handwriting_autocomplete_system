# 🏗️ Phase 2 Architecture: Next Word Prediction

This document details the architecture and implementation of the Next Word Prediction module, which utilizes a GPT-2 based Transformer model to predict subsequent text based on OCR output.

## 🧠 Model Architecture

The core of this phase is a custom implementation of the **GPT-2 (124M)** language model, defined in `model.py`.

### 1. Configuration (`GPTConfig`)
The model is configured with the following hyperparameters, matching the standard GPT-2 small architecture:
- **Parameters**: ~124 Million
- **Context Length**: 1024 tokens
- **Vocabulary Size**: 50,257 (Byte Pair Encoding tokens)
- **Embedding Dimension**: 768
- **Layers**: 12 Transformer blocks
- **Attention Heads**: 12

### 2. Core Components

#### **Causal Self-Attention (`CausalSelfAttention`)**
- Implements multi-head self-attention.
- **Flash Attention**: Utilizes `torch.nn.functional.scaled_dot_product_attention` for optimized, memory-efficient attention computation.
- **Causal Masking**: Ensures predictions only depend on past tokens (autoregressive property).
- **Projections**: Linear projections for Query, Key, and Value, and a final output projection.

#### **Feed-Forward Network (`MLP`)**
- Consists of two linear transformations with a GELU activation in between.
- **Expansion Factor**: The inner dimension is 4x the embedding size (4 * 768).
- **Activation**: Uses `GELU(approximate='tanh')` to closely match the original GPT-2 paper implementation.

#### **Transformer Block (`Block`)**
- Composes the Attention and MLP layers.
- **Pre-Norm Architecture**: Layer Normalization (`LayerNorm`) is applied *before* the attention and MLP sub-layers.
- **Residual Connections**: Adds the input to the output of each sub-layer.

#### **Main Model (`GPT`)**
- **Embeddings**:
  - `wte`: Token embeddings (Vocab Size × Embedding Dim).
  - `wpe`: Positional embeddings (Context Length × Embedding Dim).
- **Weight Tying**: The weights of the output language modeling head (`lm_head`) are tied to the input token embeddings (`wte`). This significantly reduces the parameter count and improves training stability.
- **Initialization**: Custom weight initialization (`_init_weights`) following the GPT-2 specification (normal distribution with specific scaling for residual layers).

---

## 🔄 Training Pipeline

The training logic is encapsulated in the `Trainer` class within `train.py`.

### 1. Optimization
- **Optimizer**: `AdamW` (Adam with Weight Decay).
- **Weight Decay**: Applied only to 2D parameters (weights), excluding biases and LayerNorm parameters.
- **Fused Optimizer**: Automatically uses the fused kernel version of AdamW if available on CUDA devices for faster updates.
- **Precision**: Uses **Mixed Precision Training** with `bfloat16` via `torch.autocast`. This reduces memory usage and speeds up computation on supported hardware (e.g., Ampere GPUs) without sacrificing convergence stability.

### 2. Distributed Training
- **DDP (Distributed Data Parallel)**: The code is designed to run on multiple GPUs.
- **Gradient Accumulation**: Supports simulating larger batch sizes by accumulating gradients over multiple mini-batches before an optimizer step.

### 3. Loop & Monitoring
- **Evaluation**: Periodically evaluates on a validation set.
- **HellaSwag**: Includes integration for evaluating common-sense reasoning capabilities using the HellaSwag benchmark (`hellaswag_eval.py`).
- **Generation**: Periodically generates sample text during training to visually monitor model progress.

---

## 💾 Data Pipeline

Data handling is managed by `dataloader.py` and `prepare_dataset.py`.

### 1. Dataset
- The system is designed to train on the **FineWebEdu-10B** dataset.
- Data is stored in sharded binary files (`.npy`) for efficient streaming.

### 2. DataLoader (`DataLoaderLite`)
- **Streaming**: Loads data shards on-demand to handle datasets larger than memory.
- **Tokenization**: Uses `tiktoken` (OpenAI's fast BPE tokenizer) with the `gpt2` encoding.
- **Batching**: Generates batches of shape `(B, T)` where `B` is batch size and `T` is sequence length.
- **Distributed Awareness**: Splits the dataset across different processes (ranks) when running in a distributed setting.

---

## 🔮 Inference

Inference is handled by `inference.py` and `infer_pretrained.py`.

### 1. Generation Strategy
- **Autoregressive Generation**: Predicts one token at a time, appending it to the context for the next prediction.
- **Top-k Sampling**: Restricts sampling to the top `k` (default 50) most likely next tokens to balance creativity and coherence.
- **Temperature**: Optional scaling of logits to control randomness (implemented in `infer_pretrained.py`).

### 2. Implementation
- **`GPT2Inference` Class**: A wrapper for easy generation from other modules.
- **Pretrained Support**: The system can load official GPT-2 weights (gpt2, gpt2-medium, etc.) via `GPT.from_pretrained()` for immediate usage or fine-tuning.

---

## 📂 File Structure Summary

| File | Description |
| :--- | :--- |
| `model.py` | GPT-2 model definition, layers, and configuration. |
| `train.py` | Main training script with `Trainer` class and DDP setup. |
| `dataloader.py` | `DataLoaderLite` for efficient sharded data loading. |
| `inference.py` | Inference class for generating text from trained models. |
| `infer_pretrained.py` | Script to run inference using pretrained GPT-2 weights. |
| `hellaswag_eval.py` | Evaluation logic for the HellaSwag benchmark. |
| `prepare_dataset.py` | Utilities for downloading and tokenizing datasets. |
