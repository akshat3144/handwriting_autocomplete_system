"""
Quick Test Script for Sentence Recognition
This is a simple script to quickly test the sentence recognizer.
"""

import os
import sys

# Check if required files exist
print("Checking required files...")

required_files = {
    'Model': 'htr_model_20251020_084444_base.h5',
    'Encoder': 'encoder_20251020_084444.pkl',
    'Segmentation code': 'word_segmentation-master/words.py'
}

missing_files = []
for name, path in required_files.items():
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path} NOT FOUND")
        missing_files.append(path)

if missing_files:
    print("\n⚠️  Missing files detected!")
    print("Please ensure all required files are in the correct location.")
    print("\nExpected structure:")
    print("  - htr_model_20251020_084444_base.h5")
    print("  - encoder_20251020_084444.pkl")
    print("  - word_segmentation-master/")
    print("      - words.py")
    print("      - utils.py")
    print("      - page.py")
    sys.exit(1)

print("\n" + "="*80)
print("All required files found!")
print("="*80)

# Check for test image
test_images = ['test_sentence.jpg', 'test.jpg', 'sentence.jpg', 'sample.jpg']
found_image = None

for img in test_images:
    if os.path.exists(img):
        found_image = img
        break

if found_image:
    print(f"\n✓ Found test image: {found_image}")
    print("\nRunning sentence recognition...")
    print("="*80)
    
    # Import and run the main function
    try:
        # Import CharacterEncoder first so pickle can find it
        from sentence_recognizer import CharacterEncoder, main
        result = main(image_path=found_image, visualize=True, save_words=True)
        
        if result:
            print("\n" + "="*80)
            print("SUCCESS! ✓")
            print(f"Recognized text: {result}")
            print("="*80)
        else:
            print("\n⚠️  Recognition completed but no result returned.")
            
    except Exception as e:
        print(f"\n❌ Error during recognition: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("\n⚠️  No test image found!")
    print(f"Please provide an image file named one of: {', '.join(test_images)}")
    print("\nOr run the script directly with your image:")
    print("  python sentence_recognizer.py --image path/to/your/image.jpg")
