# Complete Handwriting Recognition Pipeline

This script combines:
1. **Word Segmentation** - Detects individual words in handwritten text
2. **OCR Recognition** - Recognizes each word using a trained HTR model
3. **Next Word Prediction** - Suggests next words using GPT-2

## Usage

### Basic usage:
```bash
python complete_pipeline.py --image handwriting.png
```

### With more predictions:
```bash
python complete_pipeline.py --image handwriting.png --predictions 10
```

### Command line options:
- `--image, -i`: Path to the handwriting image (required)
- `--predictions, -p`: Number of next word predictions (default: 5)
- `--visualize, -v`: Show visualizations (optional)

## Example:

```bash
cd boundbox-ocr-nextword-pipeline
python complete_pipeline.py --image test.jpg --predictions 10
```

## Output:

The script will:
1. Display the recognized text from your handwriting
2. Show the top N next word predictions with probabilities
3. Save all results to `pipeline_results.txt`

## Requirements:

Make sure you have all dependencies installed:
- tensorflow
- opencv-python
- torch
- tiktoken
- numpy

The script automatically uses the trained models from the parent directory.
