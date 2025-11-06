"""
For Testing purposes
    Take image from user, crop the background and transform perspective
    from the perspective detect the word and return the array of word's
    bounding boxes
"""

import page
import words
from PIL import Image
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "test.jpg")

# Read image and check it loaded correctly
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at {img_path}. Make sure the file exists.")
image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Crop image and get bounding boxes
crop = page.detection(image)
boxes = words.detection(crop)
lines = words.sort_words(boxes)

# Create segmented directory if it doesn't exist
segmented_dir = os.path.join(script_dir, "segmented")
os.makedirs(segmented_dir, exist_ok=True)

# Saving the bounded words from the page image in sorted way
i = 0
for line in lines:
    text = crop.copy()
    for (x1, y1, x2, y2) in line:
        # roi = text[y1:y2, x1:x2]
        save = Image.fromarray(text[y1:y2, x1:x2])
        # print(i)
        save.save(os.path.join(segmented_dir, f"segment{i}.png"))
        i += 1



