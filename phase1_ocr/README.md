# Phase 1: Optical Character Recognition (OCR)

A complete handwritten text recognition (HTR) system that segments handwritten sentence images into individual words and recognizes them using a CNN-LSTM deep learning model trained on the IAM Handwriting Dataset.

## 📁 Project Structure

```
phase1_ocr/
├── sentence_recognizer.py      # Main inference script for sentence recognition
├── test.jpg
├── model_weights/              # Pre-trained model files
│   ├── htr_model_20251020_084444_base.h5
│   ├── htr_model_20251020_084444.weights.weights.h5
│   └── encoder_20251020_084444.pkl
├── outputs/                    # Recognition results and segmented images
│   ├── recognition_results.txt
│   ├── segmented_visualization.jpg
│   └── segmented/              # Individual word images
├── segmentation/               # Word segmentation module
│   ├── segmenter.py           # Core word segmentation logic
│   ├── visualize.py           # Pipeline visualization tool
│   ├── test_image.png
│   ├── output.png
│   ├── pipeline_steps.png
│   └── README.md
├── training/                   # Model training scripts
│   ├── htr_cnn_lstm.py        # CNN-LSTM model architecture
│   ├── train_htr_iam.py       # IAM dataset training script
│   └── htr_nb_kaggle.ipynb    # Kaggle training notebook
└── tests/                      # Test scripts
    ├── test_recognition.py    # Quick recognition test
    └── test_dataset.py        # Dataset structure validation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy opencv-python matplotlib pillow scikit-learn pandas
```

### Run Sentence Recognition

```bash
cd phase1_ocr
python sentence_recognizer.py --image path/to/your/handwritten_sentence.jpg
```

### Command Line Options

```bash
python sentence_recognizer.py --help

Options:
  --image, -i        Path to the sentence image to recognize
  --model, -m        Path to the trained HTR model (.h5 file)
  --encoder, -e      Path to the character encoder (.pkl file)
  --output-dir, -o   Directory to save segmented word images
  --no-visualize     Do not create segmentation visualization
  --no-save-words    Do not save individual word images
```

### Example

```bash
python sentence_recognizer.py --image my_handwriting.jpg
```

**Output:**

- Recognition results printed to console
- Results saved to `outputs/recognition_results.txt`
- Segmented word images saved to `outputs/segmented/`
- Visualization saved to `outputs/segmented_visualization.jpg`

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Input Image    │ ──▶ │  Word Segmentation│ ──▶ │  Preprocessing  │ ──▶ │   CNN-LSTM HTR   │
│  (Sentence)     │     │  (7-step pipeline)│     │  (32×128 resize)│     │   Recognition    │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                   │
                                                                                   ▼
                                                                         ┌──────────────────┐
                                                                         │  Recognized Text │
                                                                         │  (CTC Decoding)  │
                                                                         └──────────────────┘
```

### 1. Word Segmentation Module

Located in `segmentation/segmenter.py`, this module uses a **7-step computer vision pipeline**:

| Step | Operation             | Description                    |
| ---- | --------------------- | ------------------------------ |
| 1    | Original Image        | Input handwritten text         |
| 2    | Gaussian Blur         | Minimal noise reduction        |
| 3    | Edge Detection        | Sobel operator on RGB channels |
| 4    | Binary Thresholding   | Black & white conversion       |
| 5    | Morphological Closing | Fill small gaps in strokes     |
| 6    | Minimal Dilation      | Preserve word boundaries       |
| 7    | Contour Detection     | Extract bounding boxes         |

**Usage:**

```python
from segmentation.segmenter import WordSegmenter, segment_words

# Quick usage
result = segment_words('your_image.png')
word_images = result['word_images']  # List of cropped word images
boxes = result['boxes']              # [[x1, y1, x2, y2], ...]

# With custom parameters
segmenter = WordSegmenter(
    blur_kernel=(9, 9),
    blur_sigma=4,
    morph_kernel=(9, 9),
    dilation_kernel=(7, 17),
    min_width=50,
    min_height=25
)
result = segmenter.segment(image)
```

### 2. CNN-LSTM Recognition Model

The core recognition model (`training/htr_cnn_lstm.py`) implements a **CRNN (Convolutional Recurrent Neural Network)** architecture:

```
Input (32 × 128 × 1)
        │
    ┌───▼───┐
    │ Conv2D │ 64 filters, 3×3
    │ + Pool │ 2×2
    └───┬───┘
        │
    ┌───▼───┐
    │ Conv2D │ 128 filters, 3×3
    │ + Pool │ 2×2
    └───┬───┘
        │
    ┌───▼───┐
    │ Conv2D │ 256 filters, 3×3 (×2)
    │ + Pool │ 2×1
    └───┬───┘
        │
    ┌───▼───┐
    │ Conv2D │ 512 filters, 3×3 (×2)
    │ + BN   │ Batch Normalization
    │ + Pool │ 2×1
    └───┬───┘
        │
    ┌───▼───┐
    │ Conv2D │ 512 filters, 2×2
    └───┬───┘
        │
    ┌───▼───┐
    │ BiLSTM │ 256 units (×2)
    └───┬───┘
        │
    ┌───▼───┐
    │ Dense  │ num_classes (softmax)
    └───┬───┘
        │
    ┌───▼───┐
    │  CTC   │ Decoding
    └───────┘
```

**Key Features:**

- **7 Convolutional Layers** with increasing filter sizes (64 → 128 → 256 → 512)
- **Batch Normalization** for stable training
- **2 Bidirectional LSTM Layers** with 256 units each
- **CTC (Connectionist Temporal Classification)** loss for sequence-to-sequence learning
- **79 output classes** (lowercase + uppercase + digits + punctuation + space + blank)

### 3. Preprocessing Pipeline

Images are preprocessed before recognition:

```python
def preprocess_image(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Binarization (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Resize with aspect ratio preservation
    resized = resize_and_pad(binary, target_height=32, target_width=128)

    # 4. Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized
```

---

## 🎓 Training

### Dataset: IAM Handwriting Database

The model is trained on the [IAM Handwriting Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), which contains ~115,000 isolated handwritten words.

**Expected Dataset Structure:**

```
iam-handwriting-word-database/
├── iam_words/
│   └── words/
│       ├── a01/
│       │   ├── a01-000u/
│       │   │   ├── a01-000u-00-00.png
│       │   │   └── ...
│       └── words.txt
└── words_new.txt (alternative annotation file)
```

### Validate Dataset Structure

Before training, verify your dataset is correctly structured:

```bash
cd phase1_ocr/tests
python test_dataset.py
```

### Run Training

```bash
cd phase1_ocr/training
python train_htr_iam.py
```

**Training Configuration:**

| Parameter       | Default | Description                     |
| --------------- | ------- | ------------------------------- |
| `BATCH_SIZE`    | 32      | Training batch size             |
| `EPOCHS`        | 50      | Maximum training epochs         |
| `IMG_HEIGHT`    | 32      | Input image height              |
| `IMG_WIDTH`     | 128     | Input image width               |
| `LEARNING_RATE` | 0.001   | Adam optimizer LR               |
| `MAX_SAMPLES`   | None    | Limit dataset size (None = all) |

**Training Callbacks:**

- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Stops training after 10 epochs without improvement
- **ReduceLROnPlateau**: Reduces learning rate by 0.5 after 5 epochs without improvement

### Training Output

Models are saved to `model_weights/`:

```
model_weights/
├── htr_model_YYYYMMDD_HHMMSS_base.h5      # Base model for inference
├── htr_model_YYYYMMDD_HHMMSS.weights.h5   # Model weights
└── encoder_YYYYMMDD_HHMMSS.pkl            # Character encoder
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

| Metric                      | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| **CER**                     | Character Error Rate - Levenshtein distance at character level |
| **WER**                     | Word Error Rate - Levenshtein distance at word level           |
| **Jaro-Winkler Similarity** | String similarity score (0-1)                                  |
| **Exact Match Accuracy**    | Percentage of perfectly recognized words                       |

### Sample Output

```
================================================================================
EVALUATION RESULTS
================================================================================
Samples evaluated: 100
Exact match accuracy: 45.00%
Average similarity: 87.32%
Median similarity: 92.50%
Character Error Rate (CER): 12.45%
Word Error Rate (WER): 25.67%
================================================================================
```

---

## 🔧 Character Encoder

The `CharacterEncoder` class handles text encoding/decoding:

```python
from sentence_recognizer import CharacterEncoder

encoder = CharacterEncoder()

# Default character set
# " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-"

# Encode text to numerical indices
encoded = encoder.encode("Hello World")  # [8, 5, 12, 12, 15, 0, 33, 15, 18, 12, 4]

# Decode indices back to text
decoded = encoder.decode(encoded)  # "Hello World"

print(encoder.vocab_size)  # 79 (68 characters + 1 CTC blank)
```

---

## 🧪 Testing

### Quick Recognition Test

```bash
cd phase1_ocr/tests
python test_recognition.py
```

This script:

1. Verifies all required files exist (model, encoder, segmenter)
2. Checks for a test image
3. Runs recognition and displays results

### Run Segmentation Visualization

```bash
cd phase1_ocr/segmentation
python visualize.py
```

**Output:**

- `pipeline_steps.png` - All 7 processing steps visualized
- `output.png` - Final result with bounding boxes

---

## 📝 API Reference

### WordSegmenter Class

```python
class WordSegmenter:
    def __init__(self,
                 blur_kernel=(3, 3),      # Gaussian blur kernel size
                 blur_sigma=1,            # Gaussian blur sigma
                 morph_kernel=(3, 3),     # Morphological closing kernel
                 dilation_kernel=(1, 3),  # Dilation kernel
                 min_width=15,            # Minimum word width (pixels)
                 min_height=10,           # Minimum word height (pixels)
                 max_width_ratio=0.9,     # Max word width as ratio of image
                 max_height_ratio=0.5,    # Max word height as ratio of image
                 min_fill_ratio=0.1):     # Minimum pixel fill ratio
        ...

    def segment(self, image) -> dict:
        """Returns dict with 'boxes', 'word_images', 'sorted_lines', 'num_words'"""

    def segment_from_path(self, image_path) -> dict:
        """Same as segment(), but reads image from file path"""

    def save_word_images(self, result, output_dir):
        """Saves individual word images to directory"""
```

### Main Recognition Function

```python
def main(image_path=None, visualize=True, save_words=True) -> str:
    """
    Main function to segment and recognize handwritten sentence.

    Args:
        image_path: Path to sentence image
        visualize: Create segmentation visualization
        save_words: Save individual word images

    Returns:
        Recognized sentence as string
    """
```

---

## ⚠️ Known Limitations

1. **Single-line vs Multi-line**: Works best with clearly separated lines
2. **Handwriting Quality**: Performance varies with handwriting clarity
3. **Vocabulary**: Limited to trained character set (no special Unicode characters)
4. **Word Spacing**: May merge or split words if spacing is inconsistent
5. **Slanted Text**: Heavy slant may affect segmentation accuracy

---

## 🔗 Dependencies

```
tensorflow>=2.10.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.4.0
pillow>=8.0.0
scikit-learn>=0.24.0
pandas>=1.2.0
```

---

## 📚 References

- **IAM Handwriting Database**: [fki.tic.heia-fr.ch](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- **CRNN for OCR**: [Shi et al., 2015 - An End-to-End Trainable Neural Network](https://arxiv.org/abs/1507.05717)

---

## 📄 License

This project is part of the Handwriting Autocomplete System. See the root [LICENSE](../LICENSE) file for details.
