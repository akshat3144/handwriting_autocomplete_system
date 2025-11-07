"""
Visualization script for word segmentation.
Shows all 7 processing steps and final results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from segmenter import WordSegmenter


def visualize_pipeline(image_path='test_image.png', output_path='pipeline_steps.png'):
    """
    Visualize all 7 steps of the segmentation pipeline.
    
    Args:
        image_path: Path to input image
        output_path: Path to save visualization
    """
    print(f"Processing: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Step 1: Gaussian Blur
    blurred = cv2.GaussianBlur(image, (3, 3), 1)
    
    # Step 2: Edge Detection (Sobel)
    def sobel_detect(channel):
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)
        sobel[sobel > 255] = 255
        return np.uint8(sobel)
    
    edge_img = np.max(np.array([
        sobel_detect(blurred[:, :, 0]),
        sobel_detect(blurred[:, :, 1]),
        sobel_detect(blurred[:, :, 2])
    ]), axis=0)
    
    # Step 3: Thresholding
    _, thresh_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
    
    # Step 4: Morphological Closing
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Step 5: Minimal Dilation
    dilated_img = cv2.dilate(morph_img, np.ones((1, 3), np.uint8), iterations=1)
    
    # Step 6: Segment words using the full segmenter
    segmenter = WordSegmenter()
    result = segmenter.segment_from_path(image_path)
    
    print(f"✓ Detected {result['num_words']} words in {result['num_lines']} lines")
    
    # Create visualization
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Original processing steps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('1. Original Image', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax2.set_title('2. Gaussian Blur\n(minimal noise reduction)', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(edge_img, cmap='gray')
    ax3.set_title('3. Edge Detection\n(Sobel operator)', fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # Row 2: Processing steps continued
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(thresh_img, cmap='gray')
    ax4.set_title('4. Binary Thresholding\n(black & white)', fontsize=14, fontweight='bold', pad=10)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(morph_img, cmap='gray')
    ax5.set_title('5. Morphological Closing\n(fills small gaps)', fontsize=14, fontweight='bold', pad=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(dilated_img, cmap='gray')
    ax6.set_title('6. Minimal Dilation\n(preserve word boundaries)', fontsize=14, fontweight='bold', pad=10)
    ax6.axis('off')
    
    # Row 3: Final result (spans all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.imshow(cv2.cvtColor(result['image_with_boxes'], cv2.COLOR_BGR2RGB))
    ax7.set_title(f'7. FINAL: {result["num_words"]} Words Detected with Bounding Boxes', 
                 fontsize=18, fontweight='bold', color='darkgreen', pad=15)
    ax7.axis('off')
    
    plt.suptitle('WORD SEGMENTATION PIPELINE - 7 STEPS', 
                fontsize=20, fontweight='bold', y=0.985)
    
    # Save
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved pipeline visualization: {output_path}")
    plt.close()
    
    # Also save just the final result
    final_output = 'output.png'
    cv2.imwrite(final_output, result['image_with_boxes'])
    print(f"✓ Saved final result: {final_output}")
    
    return result


if __name__ == "__main__":
    visualize_pipeline()
