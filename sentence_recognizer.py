"""
Handwritten Sentence Recognition Script
This script segments a sentence image into words and recognizes each word using a trained HTR model.

Usage:
    python sentence_recognizer.py --image path/to/sentence_image.jpg
    
    Or modify the IMAGE_PATH variable directly in the script.
"""

import os
import sys
import numpy as np
import cv2
import pickle
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Add word_segmentation module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'word_segmentation-master'))

# Import word segmentation functions
try:
    from words import detection, sort_words
    from utils import implt
except ImportError:
    print("Warning: Could not import word segmentation modules. Make sure word_segmentation-master folder exists.")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (update these paths according to your saved model)
MODEL_PATH = 'htr_model_20251020_084444_base.h5'  # Base model for prediction
ENCODER_PATH = 'encoder_20251020_084444.pkl'  # Character encoder

# Image path (can be overridden by command line argument)
IMAGE_PATH = 'test_sentence.jpg'  # Path to sentence image

# Output settings
SAVE_SEGMENTED_WORDS = True  # Save individual word images
OUTPUT_DIR = 'segmented'  # Directory to save segmented words


# ============================================================================
# CHARACTER ENCODER CLASS
# ============================================================================

class CharacterEncoder:
    """Encode and decode characters for model training/inference"""
    
    def __init__(self, characters=None):
        if characters is None:
            # Default character set
            self.characters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-"
        else:
            self.characters = characters
        
        # Create character to index mapping
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        
        # Create index to character mapping
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
        
        # Vocab size includes all characters + blank token
        self.vocab_size = len(self.characters) + 1
        self.blank_token_idx = len(self.characters)
    
    def encode(self, text):
        """Encode text to numerical indices"""
        encoded = []
        for char in text:
            if char in self.char_to_num:
                encoded.append(self.char_to_num[char])
        return encoded
    
    def decode(self, indices):
        """Decode numerical indices to text"""
        decoded = []
        for idx in indices:
            if idx < len(self.characters) and idx in self.num_to_char:
                decoded.append(self.num_to_char[idx])
        return ''.join(decoded)


# ============================================================================
# PREPROCESSING FUNCTIONS (from notebook)
# ============================================================================

def grayscale_conversion(image):
    """Convert RGB image to grayscale"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray


def binarization(gray_image, method='otsu'):
    """Apply thresholding to convert grayscale to binary"""
    if method == 'otsu':
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'huang':
        threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]
        _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary


def resize_and_pad(image, target_height=32, target_width=128):
    """Resize image to fixed dimensions with padding"""
    h, w = image.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image with white background
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    
    # Calculate padding offsets
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def normalize_image(image):
    """Normalize pixel values to range [0, 1]"""
    normalized = image.astype(np.float32) / 255.0
    return normalized


def preprocess_image_for_recognition(image, target_height=32, target_width=128):
    """Complete preprocessing pipeline for word recognition"""
    # If it's a path, read the image
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    
    # Step 1: Grayscale conversion
    gray = grayscale_conversion(image)
    
    # Step 2: Binarization
    binary = binarization(gray, method='otsu')
    
    # Step 3: Resize and pad
    resized = resize_and_pad(binary, target_height, target_width)
    
    # Step 4: Normalization
    normalized = normalize_image(resized)
    
    # Add channel dimension
    preprocessed = np.expand_dims(normalized, axis=-1)
    
    return preprocessed


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def build_crnn_model(input_shape=(32, 128, 1), num_classes=79):
    """
    Build CRNN model architecture matching the trained model
    This is needed to properly load models with Lambda layers
    """
    from tensorflow.keras import layers, Model
    
    # Input layer
    input_layer = layers.Input(shape=input_shape, name='input_1')
    
    # Convolutional Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d')(input_layer)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)
    
    # Convolutional Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    # Convolutional Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    
    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_2')(x)
    
    # Convolutional Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    
    # Convolutional Block 6
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_3')(x)
    
    # Convolutional Block 7
    x = layers.Conv2D(512, (2, 2), activation='relu', name='conv2d_6')(x)
    
    # Reshape for LSTM (Lambda layer with proper output_shape)
    x = layers.Lambda(lambda x: tf.squeeze(x, axis=1), 
                     output_shape=lambda s: (s[0], s[2], s[3]),
                     name='lambda')(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional_1')(x)
    
    # Dense output layer
    output = layers.Dense(num_classes, activation='softmax', name='dense')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output, name='CRNN_HTR')
    
    return model


def load_model_with_weights(model_path, encoder):
    """
    Load model by rebuilding architecture and loading weights
    This avoids Lambda layer deserialization issues
    """
    try:
        # Try to load the model directly first
        print("  Attempting direct model loading...")
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        print("  ✓ Direct loading successful")
        return model
    except Exception as e:
        print(f"  Direct loading failed: {str(e)[:100]}...")
        print("  Attempting to rebuild model and load weights...")
        
        try:
            # Build model architecture
            model = build_crnn_model(input_shape=(32, 128, 1), num_classes=encoder.vocab_size)
            
            # Load weights
            weights_path = model_path.replace('_base.h5', '.weights.h5')
            if not os.path.exists(weights_path):
                # Try with just .weights.h5
                weights_path = model_path.replace('.h5', '.weights.h5')
            
            if os.path.exists(weights_path):
                print(f"  Loading weights from: {weights_path}")
                model.load_weights(weights_path)
                print("  ✓ Model rebuilt and weights loaded successfully")
                return model
            else:
                # Try loading weights from the base model file itself
                print(f"  Loading weights from base model file...")
                model.load_weights(model_path)
                print("  ✓ Model rebuilt and weights loaded successfully")
                return model
                
        except Exception as e2:
            print(f"  ✗ Rebuild method failed: {e2}")
            raise Exception(f"Could not load model: {e2}")


# ============================================================================
# DECODER FUNCTIONS
# ============================================================================

def decode_predictions(predictions, encoder):
    """Decode CTC predictions to text"""
    decoded_texts = []
    
    batch_size = predictions.shape[0]
    time_steps = predictions.shape[1]
    
    # Create input_length for all samples in batch
    input_lengths = np.full((batch_size,), time_steps, dtype=np.int32)
    
    # Decode all predictions at once
    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_lengths,
        greedy=True
    )
    
    # Convert to text
    decoded = decoded[0].numpy()
    for i in range(batch_size):
        seq = decoded[i]
        text = encoder.decode(seq)
        decoded_texts.append(text)
    
    return decoded_texts


# ============================================================================
# WORD SEGMENTATION
# ============================================================================

def segment_sentence(image_path, visualize=False):
    """
    Segment sentence image into individual words
    
    Args:
        image_path: Path to sentence image
        visualize: Whether to display segmentation results
        
    Returns:
        List of word images and their bounding boxes
    """
    print(f"\n{'='*80}")
    print("STEP 1: SEGMENTING SENTENCE INTO WORDS")
    print(f"{'='*80}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Detect word bounding boxes
    boxes = detection(image, join=False)
    print(f"✓ Detected {len(boxes)} word regions")
    
    # Sort words from left to right, top to bottom
    sorted_lines = sort_words(boxes)
    
    # Flatten sorted lines into single list
    sorted_boxes = []
    for line in sorted_lines:
        sorted_boxes.extend(line)
    
    print(f"✓ Words sorted into {len(sorted_lines)} line(s)")
    
    # Extract word images
    word_images = []
    for i, box in enumerate(sorted_boxes):
        x1, y1, x2, y2 = box
        word_img = image[y1:y2, x1:x2]
        word_images.append({
            'image': word_img,
            'box': box,
            'index': i
        })
    
    # Visualize if requested
    if visualize:
        vis_image = image.copy()
        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i+1), (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite('segmented_visualization.jpg', vis_image)
        print("✓ Visualization saved to 'segmented_visualization.jpg'")
    
    return word_images, sorted_lines


# ============================================================================
# WORD RECOGNITION
# ============================================================================

def recognize_words(word_images, model, encoder, save_words=False, output_dir='segmented'):
    """
    Recognize text from word images
    
    Args:
        word_images: List of word image dictionaries
        model: Trained HTR model
        encoder: Character encoder
        save_words: Whether to save individual word images
        output_dir: Directory to save word images
        
    Returns:
        List of recognized words
    """
    print(f"\n{'='*80}")
    print("STEP 2: RECOGNIZING WORDS")
    print(f"{'='*80}")
    
    # Create output directory if needed
    if save_words:
        os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess all word images
    preprocessed_words = []
    for word_data in word_images:
        try:
            preprocessed = preprocess_image_for_recognition(word_data['image'])
            preprocessed_words.append(preprocessed)
            
            # Save word image if requested
            if save_words:
                word_path = os.path.join(output_dir, f"word_{word_data['index']:03d}.jpg")
                cv2.imwrite(word_path, word_data['image'])
                
        except Exception as e:
            print(f"Warning: Could not preprocess word {word_data['index']}: {e}")
            # Add a blank image as placeholder
            preprocessed_words.append(np.zeros((32, 128, 1), dtype=np.float32))
    
    # Convert to numpy array
    X = np.array(preprocessed_words)
    print(f"✓ Preprocessed {len(X)} word images")
    
    # Make predictions
    print("✓ Running model predictions...")
    predictions = model.predict(X, batch_size=32, verbose=0)
    
    # Decode predictions
    print("✓ Decoding predictions...")
    recognized_words = decode_predictions(predictions, encoder)
    
    return recognized_words


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(image_path=None, visualize=True, save_words=True):
    """
    Main function to segment and recognize handwritten sentence
    
    Args:
        image_path: Path to sentence image (if None, uses IMAGE_PATH)
        visualize: Whether to display segmentation visualization
        save_words: Whether to save individual word images
    """
    # Use default path if none provided
    if image_path is None:
        image_path = IMAGE_PATH
    
    print("="*80)
    print("HANDWRITTEN SENTENCE RECOGNITION")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Model: {MODEL_PATH}")
    print(f"Encoder: {ENCODER_PATH}")
    print("="*80)
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image not found: {image_path}")
        print("\nPlease provide a valid image path:")
        print("  python sentence_recognizer.py --image path/to/your/image.jpg")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found: {MODEL_PATH}")
        print(f"Please update MODEL_PATH in the script to point to your trained model.")
        return
    
    if not os.path.exists(ENCODER_PATH):
        print(f"\n❌ ERROR: Encoder not found: {ENCODER_PATH}")
        print(f"Please update ENCODER_PATH in the script to point to your encoder file.")
        return
    
    # Load encoder first (needed for model loading)
    print("\nLoading encoder...")
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✓ Encoder loaded (vocab size: {encoder.vocab_size})")
    except Exception as e:
        print(f"❌ ERROR loading encoder: {e}")
        return
    
    # Load model
    print("\nLoading model...")
    try:
        model = load_model_with_weights(MODEL_PATH, encoder)
        print(f"✓ Model ready for inference")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Segment sentence into words
    try:
        word_images, sorted_lines = segment_sentence(image_path, visualize=visualize)
    except Exception as e:
        print(f"\n❌ ERROR during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if len(word_images) == 0:
        print("\n❌ No words detected in the image!")
        return
    
    # Recognize words
    try:
        recognized_words = recognize_words(
            word_images, 
            model, 
            encoder, 
            save_words=save_words,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"\n❌ ERROR during recognition: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print(f"\n{'='*80}")
    print("RECOGNITION RESULTS")
    print(f"{'='*80}")
    
    # Print word by word
    print("\nIndividual words:")
    for i, word in enumerate(recognized_words):
        print(f"  Word {i+1:2d}: '{word}'")
    
    # Reconstruct sentence (respecting lines)
    print("\nRecognized sentence:")
    current_word_idx = 0
    for line_idx, line in enumerate(sorted_lines):
        line_words = []
        for _ in range(len(line)):
            if current_word_idx < len(recognized_words):
                line_words.append(recognized_words[current_word_idx])
                current_word_idx += 1
        
        line_text = ' '.join(line_words)
        print(f"  Line {line_idx+1}: {line_text}")
    
    # Full sentence
    full_sentence = ' '.join(recognized_words)
    print(f"\nFull text: {full_sentence}")
    
    # Save results to file
    output_file = 'recognition_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HANDWRITTEN SENTENCE RECOGNITION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Total words detected: {len(recognized_words)}\n")
        f.write(f"Total lines: {len(sorted_lines)}\n\n")
        f.write("Individual words:\n")
        for i, word in enumerate(recognized_words):
            f.write(f"  Word {i+1:2d}: '{word}'\n")
        f.write("\nRecognized sentence:\n")
        current_word_idx = 0
        for line_idx, line in enumerate(sorted_lines):
            line_words = []
            for _ in range(len(line)):
                if current_word_idx < len(recognized_words):
                    line_words.append(recognized_words[current_word_idx])
                    current_word_idx += 1
            line_text = ' '.join(line_words)
            f.write(f"  Line {line_idx+1}: {line_text}\n")
        f.write(f"\nFull text: {full_sentence}\n")
    
    print(f"\n✓ Results saved to '{output_file}'")
    
    if save_words:
        print(f"✓ Individual word images saved to '{OUTPUT_DIR}/' directory")
    
    print(f"\n{'='*80}")
    print("RECOGNITION COMPLETE!")
    print(f"{'='*80}\n")
    
    return full_sentence


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Recognize handwritten sentences by segmenting into words',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentence_recognizer.py --image my_sentence.jpg
  python sentence_recognizer.py --image test.jpg --no-visualize --no-save-words
  
Make sure to update MODEL_PATH and ENCODER_PATH in the script if your model files have different names.
        """
    )
    
    parser.add_argument('--image', '-i', type=str, default=IMAGE_PATH,
                       help='Path to the sentence image to recognize')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                       help='Path to the trained HTR model (.h5 file)')
    parser.add_argument('--encoder', '-e', type=str, default=ENCODER_PATH,
                       help='Path to the character encoder (.pkl file)')
    parser.add_argument('--output-dir', '-o', type=str, default=OUTPUT_DIR,
                       help='Directory to save segmented word images')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Do not create segmentation visualization')
    parser.add_argument('--no-save-words', action='store_true',
                       help='Do not save individual word images')
    
    args = parser.parse_args()
    
    # Update global variables from arguments
    IMAGE_PATH = args.image
    MODEL_PATH = args.model
    ENCODER_PATH = args.encoder
    OUTPUT_DIR = args.output_dir
    
    # Run main function
    main(
        image_path=args.image,
        visualize=not args.no_visualize,
        save_words=not args.no_save_words
    )
