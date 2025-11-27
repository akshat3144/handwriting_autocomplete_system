# Transfer Learning Pipeline

An end-to-end handwriting autocomplete system that combines three state-of-the-art models: TrOCR for text recognition, GPT-2 for text prediction, and HiGAN+ for handwriting generation.

## Overview

This pipeline implements a complete handwriting autocomplete workflow:

1. **Text Extraction**: Recognizes handwritten text from images using TrOCR
2. **Text Prediction**: Predicts next words using GPT-2
3. **Style Transfer**: Generates predicted text in the original handwriting style using HiGAN+

## Features

- 🔍 **Accurate OCR**: Microsoft's TrOCR for handwriting recognition
- 🤖 **Smart Predictions**: GPT-2 medium for contextual next-word prediction
- ✍️ **Style Preservation**: HiGAN+ maintains the writer's unique handwriting style
- 🚀 **GPU Accelerated**: Automatic CUDA detection and optimization
- 🔧 **Flexible Input**: Supports manual text input or OCR extraction
- 📊 **Customizable**: Adjustable prediction length and style sources

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `transformers>=4.30.0` - For TrOCR and GPT-2 models
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Image preprocessing
- `numpy>=1.24.0` - Numerical operations
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image I/O
- `munch>=4.0.0` - Configuration management
- `PyYAML>=6.0` - YAML config parsing

## Model Weights

### Required Files

1. **HiGAN+ Checkpoint**: `epoch_70.pth` (included)

   - Pre-trained generator and style encoder weights

2. **Phase 3 Dependencies** (from `../phase3_style_transfer/`):

   - `wid_iam_new.pth` - StyleBackbone weights
   - `configs/gan_iam.yml` - Model configuration

### Automatic Downloads

The following models will be downloaded automatically on first run:

- `microsoft/trocr-base-handwritten` (~1.2 GB)
- `gpt2-medium` (~1.5 GB)

## Usage

### Command Line Interface

Basic usage:

```bash
python pipeline.py --image path/to/handwriting.jpg
```

With all options:

```bash
python pipeline.py \
  --image path/to/handwriting.jpg \
  --num_words 5 \
  --output path/to/output.jpg \
  --checkpoint path/to/custom_checkpoint.pth
```

Skip OCR with manual text input:

```bash
python pipeline.py \
  --image style_reference.jpg \
  --text "The quick brown fox" \
  --num_words 3
```

Use separate style reference:

```bash
python pipeline.py \
  --image ocr_source.jpg \
  --style_image style_reference.jpg \
  --num_words 4
```

### Python API

```python
from pipeline import HandwritingAutocompletePipeline

# Initialize pipeline
pipeline = HandwritingAutocompletePipeline(
    checkpoint_path="epoch_70.pth",  # Optional
    device="cuda"  # Optional: "cuda" or "cpu"
)

# Run complete pipeline
result = pipeline.run(
    image_path="handwriting.jpg",
    num_words=3,
    output_path="generated.jpg"
)

print(f"Original: {result['original_text']}")
print(f"Predicted: {result['predicted_words']}")
print(f"Complete: {result['completed_text']}")
```

### Advanced Usage

**Extract components separately:**

```python
# Step 1: Extract text
text = pipeline.extract_text("handwriting.jpg")

# Step 2: Predict next words
predictions = pipeline.predict_next_words(text, num_words=5)

# Step 3: Extract style
style_vector = pipeline.extract_style("handwriting.jpg")

# Step 4: Generate handwriting
generated_tensor = pipeline.generate_handwriting(style_vector, predictions)
image = pipeline.tensor_to_image(generated_tensor)
```

## Command Line Arguments

| Argument        | Type | Required | Default      | Description                    |
| --------------- | ---- | -------- | ------------ | ------------------------------ |
| `--image`       | str  | Yes      | -            | Input handwriting image path   |
| `--num_words`   | int  | No       | 3            | Number of words to predict     |
| `--output`      | str  | No       | None         | Output image save path         |
| `--checkpoint`  | str  | No       | epoch_70.pth | HiGAN checkpoint path          |
| `--text`        | str  | No       | None         | Manual text input (skips OCR)  |
| `--style_image` | str  | No       | None         | Separate style reference image |

## Pipeline Components

### 1. TrOCR (Text Recognition)

- **Model**: `microsoft/trocr-base-handwritten`
- **Input**: RGB handwriting images
- **Output**: Recognized text string
- **Max Tokens**: 128 characters

### 2. GPT-2 (Text Prediction)

- **Model**: `gpt2-medium` (355M parameters)
- **Strategy**: Top-k (k=40) and nucleus sampling (p=0.9)
- **Temperature**: 0.7 for controlled creativity
- **Features**:
  - No-repeat n-gram (size=3)
  - Repetition penalty (1.2)
  - Smart continuation extraction

### 3. HiGAN+ (Style Transfer)

- **Generator**: BigGAN architecture
- **Style Encoder**: Custom VAE-based encoder
- **Style Backbone**: Pre-trained writer identification network
- **Image Height**: 64 pixels
- **Character Width**: 32 pixels
- **Training**: Transfer learning on IAM dataset (epoch 70)

## Architecture

```
Input Image → TrOCR → Recognized Text → GPT-2 → Predicted Words
     ↓                                                  ↓
Style Vector ← Style Encoder ← Style Backbone      Generator
                                                       ↓
                                               Generated Handwriting
```

## Performance

- **OCR Accuracy**: ~95% on IAM handwriting dataset
- **Prediction Quality**: Contextually relevant with low repetition
- **Style Fidelity**: Maintains writer-specific characteristics
- **Speed** (with GPU):
  - OCR: ~0.1s per image
  - Prediction: ~0.2s per sequence
  - Generation: ~0.3s per text

## Examples

### Example 1: Basic Autocomplete

```bash
python pipeline.py --image samples/letter.jpg --num_words 3
```

**Output:**

```
Extracted text: 'Dear Sir, I am writing to'
Predicted words: 'inform you about'
Complete: 'Dear Sir, I am writing to inform you about'
```

### Example 2: Manual Text with Style

```bash
python pipeline.py \
  --image writer_sample.jpg \
  --text "Machine learning is" \
  --num_words 5
```

**Output:**

```
Using input text: 'Machine learning is'
Predicted words: 'a subset of artificial intelligence that'
```

## Troubleshooting

### CUDA Out of Memory

Use CPU mode:

```python
pipeline = HandwritingAutocompletePipeline(device="cpu")
```

Or reduce batch sizes in the models.

### Missing Dependencies

Ensure phase3 files are present:

```
../phase3_style_transfer/
  ├── wid_iam_new.pth
  ├── configs/gan_iam.yml
  └── lib/
```

### Poor OCR Quality

- Ensure input images have good contrast
- Recommended resolution: minimum 64px height
- Use grayscale or RGB (automatic conversion)

### Repetitive Predictions

Adjust GPT-2 parameters:

```python
# In pipeline.py, modify predict_next_words():
temperature=0.8,        # More randomness
top_k=50,              # Wider selection
repetition_penalty=1.5  # Stronger penalty
```

## Project Structure

```
transfer_learning_pipeline/
├── README.md              # This file
├── pipeline.py            # Main pipeline implementation
├── epoch_70.pth          # HiGAN+ checkpoint
├── requirements.txt       # Python dependencies
└── ocr-next-tl.ipynb     # Interactive notebook
```

## Integration with Project

This pipeline integrates three phases of the handwriting autocomplete system:

- **Phase 1** (`../phase1_ocr/`): OCR foundation
- **Phase 2** (`../phase2_nextword_prediction/`): Language modeling
- **Phase 3** (`../phase3_style_transfer/`): HiGAN+ generation

For complete system integration, see `../pipeline/complete_pipeline.py`.

## **Component Models:**

- TrOCR: [Microsoft Research](https://arxiv.org/abs/2109.10282)
- GPT-2: [OpenAI](https://openai.com/research/gpt-2)
- HiGAN+: Based on [HiGAN](https://github.com/ganji15/HiGAN)
