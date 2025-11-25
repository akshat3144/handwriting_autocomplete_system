# Word Segmenter

Standalone module for segmenting handwritten text images into individual words.

## Files

- `segmenter.py` - Core segmentation logic (main module)
- `visualize.py` - Visualization script showing all 7 processing steps
- `test_image.png` - Sample handwritten text image
- `requirements.txt` - Dependencies

## Quick Usage

```python
from segmenter import segment_words

# Segment an image
result = segment_words('your_image.png')

# Access results
word_images = result['word_images']  # List of cropped word images
boxes = result['boxes']              # [[x1, y1, x2, y2], ...]
num_words = result['num_words']      # Total word count
```

## Run Visualization

```bash
python visualize.py
```

**Outputs:**
- `pipeline_steps.png` - All 7 processing steps visualized
- `output.png` - Final result with bounding boxes only

## The 7 Steps

1. **Original Image** - Input
2. **Gaussian Blur** - Minimal noise reduction
3. **Edge Detection** - Sobel operator on RGB
4. **Binary Thresholding** - Black & white conversion
5. **Morphological Closing** - Fill small gaps
6. **Minimal Dilation** - Preserve word boundaries
7. **Final Result** - Words with bounding boxes

## Pipeline Integration

```python
from segmenter import WordSegmenter

segmenter = WordSegmenter()
result = segmenter.segment_from_path('image.png')

for word_img in result['word_images']:
    # Your OCR processing here
    pass
```
