"""
Word Segmentation Core Module
Standalone implementation for segmenting text images into individual words.
No external dependencies except OpenCV and NumPy.
"""

import cv2
import numpy as np


class WordSegmenter:
    """
    A class for segmenting handwritten text images into individual words.
    
    This uses computer vision techniques including edge detection, morphological
    operations, and contour detection to find word boundaries.
    """
    
    def __init__(self, 
                 blur_kernel=(3, 3),
                 blur_sigma=1,
                 morph_kernel=(3, 3),
                 dilation_kernel=(1, 3),
                 min_width=15,
                 min_height=10,
                 max_width_ratio=0.9,
                 max_height_ratio=0.5,
                 min_fill_ratio=0.1):
        """
        Initialize the word segmenter with configurable parameters.
        
        Args:
            blur_kernel: Gaussian blur kernel size (width, height)
            blur_sigma: Gaussian blur sigma value
            morph_kernel: Morphological closing kernel size
            dilation_kernel: Dilation kernel size for connecting letters
            min_width: Minimum word width in pixels
            min_height: Minimum word height in pixels
            max_width_ratio: Maximum word width as ratio of image width
            max_height_ratio: Maximum word height as ratio of image height
            min_fill_ratio: Minimum fill ratio (non-zero pixels / total pixels)
        """
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.morph_kernel = morph_kernel
        self.dilation_kernel = dilation_kernel
        self.min_width = min_width
        self.min_height = min_height
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio
        self.min_fill_ratio = min_fill_ratio
    
    def _apply_sobel(self, channel):
        """Apply Sobel edge detection to a single channel."""
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)
        sobel[sobel > 255] = 255
        return np.uint8(sobel)
    
    def segment(self, image):
        """
        Segment an image into individual words.
        
        Args:
            image: Input image (BGR format, as read by cv2.imread)
            
        Returns:
            dict with keys:
                - 'boxes': List of bounding boxes [[x1, y1, x2, y2], ...]
                - 'sorted_lines': List of lists, words grouped by lines
                - 'word_images': List of cropped word images
                - 'image_with_boxes': Original image with boxes drawn
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Step 1: Gaussian Blur (minimal noise reduction)
        blurred = cv2.GaussianBlur(image, self.blur_kernel, self.blur_sigma)
        
        # Step 2: Edge Detection (Sobel on each RGB channel)
        edge_img = np.max(np.array([
            self._apply_sobel(blurred[:, :, 0]),
            self._apply_sobel(blurred[:, :, 1]),
            self._apply_sobel(blurred[:, :, 2])
        ]), axis=0)
        
        # Step 3: Binary Thresholding
        _, thresh_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
        
        # Step 4: Morphological Closing (minimal gap filling)
        morph_img = cv2.morphologyEx(
            thresh_img,
            cv2.MORPH_CLOSE,
            np.ones(self.morph_kernel, np.uint8)
        )
        
        # Step 5: Minimal Dilation (preserve word boundaries)
        dilated_img = cv2.dilate(
            morph_img,
            np.ones(self.dilation_kernel, np.uint8),
            iterations=1
        )
        
        # Step 6: Find Contours
        contours, _ = cv2.findContours(
            dilated_img.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and extract bounding boxes
        boxes = []
        img_height, img_width = image.shape[:2]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate fill ratio
            mask_roi = dilated_img[y:y+h, x:x+w]
            if mask_roi.size > 0:
                fill_ratio = cv2.countNonZero(mask_roi) / (w * h)
            else:
                fill_ratio = 0
            
            # Filter based on size, aspect ratio, and fill ratio
            if (fill_ratio > self.min_fill_ratio and
                w > self.min_width and h > self.min_height and
                w < img_width * self.max_width_ratio and
                h < img_height * self.max_height_ratio and
                h / w < 3 and  # Not too vertical
                w / h < 15):   # Not too horizontal
                boxes.append([x, y, x + w, y + h])
        
        # Convert to numpy array
        boxes = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
        
        # Sort words into lines
        sorted_lines = self._sort_words(boxes)
        
        # Extract word images
        word_images = []
        flat_boxes = []
        for line in sorted_lines:
            for box in line:
                flat_boxes.append(box)
                x1, y1, x2, y2 = box
                word_img = image[y1:y2, x1:x2]
                word_images.append(word_img)
        
        # Draw boxes on image
        image_with_boxes = image.copy()
        for box in flat_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return {
            'boxes': flat_boxes,
            'sorted_lines': sorted_lines,
            'word_images': word_images,
            'image_with_boxes': image_with_boxes,
            'num_words': len(flat_boxes),
            'num_lines': len(sorted_lines)
        }
    
    def _sort_words(self, boxes):
        """
        Sort bounding boxes into lines (top to bottom) and 
        within each line (left to right).
        
        Args:
            boxes: numpy array of shape (N, 4) with format [x1, y1, x2, y2]
            
        Returns:
            List of lists, where each inner list contains boxes for one line
        """
        if len(boxes) == 0:
            return []
        
        # Calculate mean height
        heights = boxes[:, 3] - boxes[:, 1]
        mean_height = np.mean(heights) if len(heights) > 0 else 0
        
        # Sort by y-coordinate first
        boxes = boxes[boxes[:, 1].argsort()]
        
        # Group into lines
        lines = []
        current_line = []
        current_y = boxes[0][1]
        
        for box in boxes:
            y1 = box[1]
            # If this box is significantly below the current line, start a new line
            if y1 > current_y + mean_height:
                if current_line:
                    lines.append(current_line)
                current_line = [box]
                current_y = y1
            else:
                current_line.append(box)
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Sort each line by x-coordinate (left to right)
        for line in lines:
            line.sort(key=lambda box: box[0])
        
        return lines
    
    def segment_from_path(self, image_path):
        """
        Convenience method to segment from an image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Same dict as segment() method
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return self.segment(image)
    
    def save_word_images(self, result, output_dir):
        """
        Save individual word images to a directory.
        
        Args:
            result: Dictionary returned by segment() or segment_from_path()
            output_dir: Directory to save word images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, word_img in enumerate(result['word_images']):
            filename = os.path.join(output_dir, f'word_{idx+1:03d}.png')
            cv2.imwrite(filename, word_img)
        
        return output_dir


# Convenience function for quick usage
def segment_words(image_or_path, **kwargs):
    """
    Convenience function to quickly segment an image into words.
    
    Args:
        image_or_path: Either a numpy array (image) or string (file path)
        **kwargs: Optional parameters to pass to WordSegmenter constructor
        
    Returns:
        Dictionary with segmentation results
    """
    segmenter = WordSegmenter(**kwargs)
    
    if isinstance(image_or_path, str):
        return segmenter.segment_from_path(image_or_path)
    else:
        return segmenter.segment(image_or_path)
