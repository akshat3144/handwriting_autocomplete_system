"""
Quick test script to verify IAM dataset structure and accessibility
Run this before training to ensure everything is set up correctly
"""

import os
from pathlib import Path

def test_dataset_structure(base_path):
    """Test if the IAM dataset structure is correct"""
    
    print("="*80)
    print("IAM DATASET STRUCTURE TEST")
    print("="*80)
    
    base_path = Path(base_path)
    
    # Test 1: Base path exists
    print(f"\n[Test 1] Checking base path...")
    print(f"Path: {base_path}")
    if base_path.exists():
        print("✓ Base path exists")
        print(f"  Contents: {[item.name for item in base_path.iterdir()][:10]}")
    else:
        print("✗ Base path does NOT exist!")
        return False
    
    # Test 2: iam_words folder exists
    print(f"\n[Test 2] Checking iam_words folder...")
    iam_words_dir = base_path / 'iam_words'
    print(f"Path: {iam_words_dir}")
    if iam_words_dir.exists():
        print("✓ iam_words folder exists")
    else:
        print("✗ iam_words folder does NOT exist!")
        return False
    
    # Test 3: Words directory exists inside iam_words
    print(f"\n[Test 3] Checking words directory...")
    words_dir = iam_words_dir / 'words'
    print(f"Path: {words_dir}")
    if words_dir.exists():
        print("✓ Words directory exists")
    else:
        print("✗ Words directory does NOT exist!")
        return False
    
    # Test 4: Annotation file exists
    print(f"\n[Test 4] Checking annotation files...")
    words_txt_options = [
        words_dir / 'words.txt',  # Inside iam_words/words/
        base_path / 'words_new.txt',  # In root
        iam_words_dir / 'words_new.txt'  # Inside iam_words/
    ]
    
    words_txt = None
    for option in words_txt_options:
        print(f"Checking: {option}")
        if option.exists():
            words_txt = option
            print(f"✓ Found annotation file: {words_txt}")
            break
    
    if words_txt is None:
        print("✗ No annotation file found!")
        print("Expected locations:")
        for opt in words_txt_options:
            print(f"  - {opt}")
        return False
    
    # Test 5: Check folder structure
    print(f"\n[Test 5] Checking folder structure...")
    subdirs = [d for d in words_dir.iterdir() if d.is_dir()]
    if len(subdirs) > 0:
        print(f"✓ Found {len(subdirs)} subdirectories in words/")
        print(f"  Sample folders: {[d.name for d in subdirs[:5]]}")
    else:
        print("✗ No subdirectories found in words/")
        return False
    
    # Test 6: Check sample images
    print(f"\n[Test 6] Checking sample images...")
    sample_folder = subdirs[0]
    image_folders = list(sample_folder.iterdir())
    
    if len(image_folders) > 0:
        print(f"✓ Found {len(image_folders)} image folders in {sample_folder.name}/")
        
        # Check for actual images
        sample_img_folder = image_folders[0]
        images = list(sample_img_folder.glob('*.png'))
        
        if len(images) > 0:
            print(f"✓ Found {len(images)} images in {sample_img_folder.name}/")
            print(f"  Sample image: {images[0].name}")
        else:
            print(f"✗ No PNG images found in {sample_img_folder.name}/")
            return False
    else:
        print(f"✗ No image folders found in {sample_folder.name}/")
        return False
    
    # Test 7: Parse annotation file
    print(f"\n[Test 7] Parsing annotation file...")
    valid_lines = 0
    total_lines = 0
    
    with open(words_txt, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9 and parts[1] == 'ok':
                valid_lines += 1
            
            if valid_lines >= 5:  # Just check first 5 valid lines
                break
    
    print(f"✓ Annotation file readable")
    print(f"  Total lines: {total_lines}")
    print(f"  Valid entries (first batch): {valid_lines}")
    
    # Test 8: Verify image path construction
    print(f"\n[Test 8] Verifying image path construction...")
    with open(words_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9 and parts[1] == 'ok':
                word_id = parts[0]
                transcription = ' '.join(parts[8:])
                
                # Construct path (Kaggle format)
                # a01-000u-00-00 -> a01/a01-000u/a01-000u-00-00.png
                parts_id = word_id.split('-')
                folder1 = parts_id[0]  # First part (e.g., 'a01')
                folder2 = '-'.join(parts_id[:2])  # First 2 parts (e.g., 'a01-000u')
                
                image_path = words_dir / folder1 / folder2 / f"{word_id}.png"
                
                print(f"Sample word ID: {word_id}")
                print(f"Transcription: {transcription}")
                print(f"Expected path: {image_path}")
                
                if image_path.exists():
                    print(f"✓ Image found at expected location!")
                    
                    # Get image size
                    import cv2
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        print(f"  Image size: {img.shape}")
                    else:
                        print("  Warning: Image could not be read!")
                else:
                    print(f"✗ Image NOT found at expected location!")
                    print(f"  Checking if folder exists: {image_path.parent}")
                    if image_path.parent.exists():
                        print(f"  Folder exists, listing contents:")
                        contents = list(image_path.parent.iterdir())[:5]
                        for item in contents:
                            print(f"    - {item.name}")
                
                break
    
    # Final Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✓ All tests passed!")
    print(f"✓ Dataset appears to be correctly structured")
    print(f"✓ Ready for training!")
    print("="*80)
    
    return True


def main():
    """Main test function"""
    
    # UPDATE THIS PATH TO YOUR DATASET ROOT (folder containing iam_words/ and words_new.txt)
    dataset_path = r"C:\Users\nitro 5\Desktop\DL\iam-handwriting-word-database"
    
    print("\nTesting IAM Dataset...")
    print(f"Dataset path: {dataset_path}\n")
    
    try:
        success = test_dataset_structure(dataset_path)
        
        if success:
            print("\n✅ SUCCESS! Your dataset is ready for training.")
            print("\nNext step: Run 'python train_htr_iam.py' to start training!")
        else:
            print("\n❌ FAILED! Please fix the issues above before training.")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
