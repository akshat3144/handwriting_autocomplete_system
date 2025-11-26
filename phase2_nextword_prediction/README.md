# 📝 Phase 2: Next Word Prediction

This module implements **GPT-2 based next word prediction** for the handwriting autocomplete system. Given recognized text from Phase 1 (OCR), this phase predicts the most likely next words to complete the sentence.

---

## 🏗️ Architecture

The implementation uses the **GPT-2 Transformer architecture** with the following specifications:

| Parameter           | Value             |
| ------------------- | ----------------- |
| Model Size          | 124M parameters   |
| Context Length      | 1024 tokens       |
| Vocabulary Size     | 50,257 BPE tokens |
| Embedding Dimension | 768               |
| Number of Layers    | 12                |
| Attention Heads     | 12                |

### Model Components

```
GPT-2
├── Token Embedding (wte)
├── Position Embedding (wpe)
├── Transformer Blocks (×12)
│   ├── Layer Norm
│   ├── Causal Self-Attention (with Flash Attention)
│   ├── Layer Norm
│   └── MLP (Feed-Forward Network)
├── Final Layer Norm
└── Language Model Head
```

---

## 📁 Directory Structure

```
phase2_nextword_prediction/
├── README.md                    # This file
├── inference.py                 # Inference with trained checkpoint
├── infer_pretrained.py          # Inference with pretrained GPT-2
│
├── src/                         # Core modules
│   ├── model.py                 # GPT-2 model architecture
│   ├── dataloader.py            # Data loading for training
│   └── prepare_dataset.py       # Dataset preparation utilities
│
├── training/                    # Training scripts
│   └── train.py                 # Distributed training script
│
└── evaluation/                  # Evaluation scripts
    └── hellaswag_eval.py        # HellaSwag benchmark evaluation
```

---

## 🚀 Quick Start

### Using Pretrained GPT-2 (Recommended)

The simplest way to run next word prediction using pretrained GPT-2 weights:

```bash
cd phase2_nextword_prediction

# Basic usage
python infer_pretrained.py --prompt "Hello, I am"

# With more options
python infer_pretrained.py \
    --prompt "The quick brown fox" \
    --model_type gpt2 \
    --num_seq 3 \
    --max_new_tokens 50 \
    --temperature 0.8 \
    --top_k 50
```

### Using Trained Checkpoint

If you have trained your own model:

```bash
python inference.py --prompt "Hello, I am a language model," --num_seq 5 --max_tokens 50
```

---

## ⚙️ Configuration Options

### Inference Parameters

| Parameter          | Default       | Description                                                   |
| ------------------ | ------------- | ------------------------------------------------------------- |
| `--prompt`         | "Hello, I am" | Input text to complete                                        |
| `--model_type`     | gpt2          | Model variant: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| `--num_seq`        | 1             | Number of completions to generate                             |
| `--max_new_tokens` | 50            | Maximum tokens to generate                                    |
| `--temperature`    | 1.0           | Sampling temperature (lower = more deterministic)             |
| `--top_k`          | 50            | Top-k sampling (0 to disable)                                 |
| `--device`         | auto          | Device: `cpu`, `cuda`, or `mps`                               |
| `--seed`           | 42            | Random seed for reproducibility                               |

### Model Variants

| Model         | Parameters | Description          |
| ------------- | ---------- | -------------------- |
| `gpt2`        | 124M       | Base model (default) |
| `gpt2-medium` | 350M       | Medium model         |
| `gpt2-large`  | 774M       | Large model          |
| `gpt2-xl`     | 1.56B      | Extra-large model    |

---

## 🏋️ Training (Optional)

To train GPT-2 from scratch on FineWebEdu-10B dataset:

### Prerequisites

1. Download and prepare the FineWebEdu-10B dataset
2. Place tokenized shards in `data/edu_fineweb10B/`

### Single GPU Training

```bash
cd training
python train.py \
    --total_batch_size 524288 \
    --mini_batch_size 32 \
    --num_epochs 5 \
    --max_lr 1e-3 \
    --logdir ./logs/
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py \
    --total_batch_size 524288 \
    --mini_batch_size 32 \
    --num_epochs 5
```

### Training Hyperparameters

| Parameter            | Default | Description                      |
| -------------------- | ------- | -------------------------------- |
| `--total_batch_size` | 524,288 | Tokens per weight update (~0.5M) |
| `--mini_batch_size`  | 32      | Batch size per GPU               |
| `--context_length`   | 1024    | Maximum sequence length          |
| `--max_lr`           | 1e-3    | Maximum learning rate            |
| `--min_lr`           | 1e-4    | Minimum learning rate            |
| `--warmup_steps`     | 715     | LR warmup steps                  |
| `--weight_decay`     | 0.1     | Weight decay for regularization  |
| `--num_epochs`       | 5       | Number of training epochs        |
| `--eval_freq`        | 250     | Evaluation frequency (steps)     |

---

## 📊 Evaluation

### HellaSwag Benchmark

Evaluate model performance on the HellaSwag common-sense reasoning benchmark:

```bash
cd evaluation
python hellaswag_eval.py --model_type gpt2 --device cuda
```

### Expected Results

| Model            | Accuracy (normalized) |
| ---------------- | --------------------- |
| GPT-2 (124M)     | ~29.5%                |
| GPT-2-XL (1.56B) | ~48.9%                |

---

## 🔗 Pipeline Integration

This module integrates with the complete pipeline in `pipeline/complete_pipeline.py`:

```python
from phase2_nextword_prediction.src.model import GPT

# Load pretrained GPT-2
model = GPT.from_pretrained('gpt2')
model.eval()

# Generate predictions
tokenizer = tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode("recognized handwritten text")
# ... generate next words
```

### Integration Flow

```
Phase 1 (OCR)          Phase 2 (Prediction)         Phase 3 (Style Transfer)
     │                        │                            │
     ▼                        ▼                            ▼
[Word Images] ──► [Recognized Text] ──► [Next Word] ──► [Styled Output]
```

---

## 📦 Dependencies

```
torch>=2.0.0
tiktoken
transformers
numpy
tqdm
requests
```

Install with:

```bash
pip install torch tiktoken transformers numpy tqdm requests
```

---

## 💡 Usage Examples

### Example 1: Basic Completion

```python
from infer_pretrained import generate, GPT
import tiktoken

# Load model
model = GPT.from_pretrained('gpt2')
model.to('cuda')
enc = tiktoken.get_encoding('gpt2')

# Generate
completions = generate(
    model=model,
    enc=enc,
    prompt="The weather today is",
    num_seq=3,
    max_new_tokens=20
)

for i, text in enumerate(completions):
    print(f"{i+1}. {text}")
```

### Example 2: Integration with OCR Output

```python
# After Phase 1 OCR recognition
recognized_text = "I am writing to"

# Generate next word predictions
predictions = generate(
    model=gpt_model,
    enc=tokenizer,
    prompt=recognized_text,
    num_seq=5,
    max_new_tokens=10,
    temperature=0.7
)

print(f"Input: '{recognized_text}'")
print("Predicted completions:")
for pred in predictions:
    print(f"  → {pred}")
```

---

## 🔧 Technical Details

### Flash Attention

The model uses PyTorch's `scaled_dot_product_attention` for efficient attention computation:

```python
# Flash Attention (faster than manual implementation)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Weight Sharing

Token embedding and language model head share weights to reduce parameters:

```python
self.transformer.wte.weight = self.lm_head.weight  # ~40M params saved
```

### Cosine Learning Rate Schedule

Training uses cosine decay with linear warmup:

```
LR
 ↑
max_lr ─────╮
            │╲
            │  ╲
            │    ╲____
min_lr ─────┼─────────────► steps
        warmup  decay
```

---

## 📈 Performance Notes

- **Inference Speed**: ~50 tokens/second on CPU, ~500+ tokens/second on CUDA
- **Memory Usage**: ~500MB for GPT-2 base model
- **First Run**: Initial run downloads pretrained weights (~500MB)

---

## 📚 References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 Paper)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer Architecture)
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) (Evaluation Benchmark)

---

## 📄 License

This module is part of the Handwriting Autocomplete System, licensed under the MIT License.
