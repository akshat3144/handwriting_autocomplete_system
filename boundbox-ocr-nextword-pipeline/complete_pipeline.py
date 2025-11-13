"""
Complete Handwriting Recognition and Next Word Prediction Pipeline
Combines word segmentation, OCR recognition, and GPT-2 next word prediction.

Usage:
    python complete_pipeline.py --image path/to/sentence_image.jpg --predictions 5
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
import torch
import torch.nn.functional as F
import tiktoken

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'word_segmentation'))
sys.path.insert(0, str(parent_dir / 'gpt-2-train'))

# Import word segmentation and GPT-2 model
from segmenter import WordSegmenter
from model import GPT, GPTConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

# OCR Model paths
OCR_MODEL_PATH = parent_dir / 'ocr_weights' / 'htr_model_20251020_084444_base.h5'
ENCODER_PATH = parent_dir / 'ocr_weights' / 'encoder_20251020_084444.pkl'

# GPT-2 paths
GPT2_CHECKPOINTS_DIR = parent_dir / 'gpt-2-124M_checkpoints'

# Default image path
IMAGE_PATH = 'test.jpg'

# Default number of predictions
NUM_PREDICTIONS = 5


# ============================================================================
# CHARACTER ENCODER CLASS
# ============================================================================

class CharacterEncoder:
    """Encode and decode characters for model training/inference"""
    
    def __init__(self, characters=None):
        if characters is None:
            self.characters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-"
        else:
            self.characters = characters
        
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
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
# OCR FUNCTIONS
# ============================================================================

def preprocess_image_for_recognition(image, target_height=32, target_width=128):
    """Complete preprocessing pipeline for word recognition"""
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize and pad
    h, w = binary.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize
    normalized = padded.astype(np.float32) / 255.0
    preprocessed = np.expand_dims(normalized, axis=-1)
    
    return preprocessed


def build_crnn_model(input_shape=(32, 128, 1), num_classes=79):
    """Build CRNN model architecture"""
    from tensorflow.keras import layers, Model
    
    input_layer = layers.Input(shape=input_shape, name='input_1')
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d')(input_layer)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_3')(x)
    x = layers.Conv2D(512, (2, 2), activation='relu', name='conv2d_6')(x)
    x = layers.Lambda(lambda x: tf.squeeze(x, axis=1), 
                     output_shape=lambda s: (s[0], s[2], s[3]),
                     name='lambda')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional_1')(x)
    output = layers.Dense(num_classes, activation='softmax', name='dense')(x)
    
    model = Model(inputs=input_layer, outputs=output, name='CRNN_HTR')
    return model


def load_ocr_model(model_path, encoder):
    """Load OCR model - rebuild and load from base h5 file"""
    print("  Building model architecture...")
    model = build_crnn_model(input_shape=(32, 128, 1), num_classes=encoder.vocab_size)
    
    print(f"  Loading weights from: {model_path.name}")
    try:
        # Load weights from the base h5 model file
        model.load_weights(str(model_path))
        print("  ✓ Model weights loaded successfully")
        return model
    except Exception as e:
        print(f"  ✗ Weight loading failed: {e}")
        raise


def decode_predictions(predictions, encoder):
    """Decode CTC predictions to text"""
    batch_size = predictions.shape[0]
    time_steps = predictions.shape[1]
    input_lengths = np.full((batch_size,), time_steps, dtype=np.int32)
    
    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_lengths,
        greedy=True
    )
    
    decoded_texts = []
    decoded = decoded[0].numpy()
    for i in range(batch_size):
        seq = decoded[i]
        text = encoder.decode(seq)
        decoded_texts.append(text)
    
    return decoded_texts


# ============================================================================
# GPT-2 FUNCTIONS
# ============================================================================

def load_gpt2_model(device='cpu'):
    """Load GPT-2 model and tokenizer"""
    print("\nLoading GPT-2 model...")
    
    # Load model
    gpt_model = GPT.from_pretrained('gpt2')
    gpt_model = gpt_model.to(device)
    gpt_model.eval()
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    print(f"✓ GPT-2 model loaded on {device}")
    return gpt_model, tokenizer


def generate_next_words(gpt_model, tokenizer, prompt, num_predictions=5, temperature=1.0, top_k=50, device='cpu'):
    """Generate next word predictions using GPT-2"""
    gpt_model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Clamp to context length
    context_len = gpt_model.config.context_length
    if idx.shape[1] > context_len:
        idx = idx[:, -context_len:]
    
    # Generate
    with torch.no_grad():
        logits, _ = gpt_model(idx)
        logits = logits[:, -1, :]
        
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        
        topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        
        top_n_probs, top_n_indices = torch.topk(probs, k=min(num_predictions, probs.shape[-1]), dim=-1)
        top_n_tokens = torch.gather(topk_idx, -1, top_n_indices)
        
        predictions = []
        for i in range(top_n_tokens.shape[1]):
            token_id = top_n_tokens[0, i].item()
            token_text = tokenizer.decode([token_id])
            prob = top_n_probs[0, i].item()
            predictions.append((token_text, prob))
    
    return predictions


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def complete_pipeline(image_path, num_predictions=5, visualize=False):
    """
    Complete pipeline: Word Segmentation → OCR → Next Word Prediction
    """
    print("="*80)
    print("COMPLETE HANDWRITING RECOGNITION AND NEXT WORD PREDICTION PIPELINE")
    print("="*80)
    print(f"Image: {image_path}")
    print("="*80)
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image not found: {image_path}")
        return
    
    # Load encoder
    print("\nLoading character encoder...")
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✓ Encoder loaded (vocab size: {encoder.vocab_size})")
    except Exception as e:
        print(f"❌ ERROR loading encoder: {e}")
        return
    
    # Load OCR model
    print("\nLoading OCR model...")
    try:
        ocr_model = load_ocr_model(OCR_MODEL_PATH, encoder)
        print("✓ OCR model ready")
    except Exception as e:
        print(f"❌ ERROR loading OCR model: {e}")
        return
    
    # Initialize word segmenter
    print("\nInitializing word segmenter...")
    word_segmenter = WordSegmenter(
        blur_kernel=(3, 3),
        blur_sigma=1,
        morph_kernel=(3, 3),
        dilation_kernel=(1, 3),
        min_width=15,
        min_height=10,
        max_width_ratio=0.9,
        max_height_ratio=0.5,
        min_fill_ratio=0.1
    )
    print("✓ Word segmenter ready")
    
    # STEP 1: Word Segmentation
    print(f"\n{'='*80}")
    print("STEP 1: WORD SEGMENTATION")
    print(f"{'='*80}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image")
        return
    
    print(f"✓ Image loaded: {image.shape}")
    
    seg_results = word_segmenter.segment(image)
    boxes = seg_results['boxes']
    word_images = seg_results['word_images']
    
    print(f"✓ Detected {len(boxes)} words in {seg_results['num_lines']} lines")
    
    if len(word_images) == 0:
        print("❌ No words detected!")
        return
    
    # Save image with boxes for visualization later
    image_with_boxes = seg_results['image_with_boxes'].copy()
    
    # STEP 2: OCR Recognition
    print(f"\n{'='*80}")
    print("STEP 2: OCR RECOGNITION")
    print(f"{'='*80}")
    
    preprocessed_words = []
    for word_img in word_images:
        preprocessed = preprocess_image_for_recognition(word_img)
        preprocessed_words.append(preprocessed)
    
    batch = np.array(preprocessed_words)
    print(f"✓ Preprocessed {len(batch)} word images")
    
    print("✓ Running OCR predictions...")
    predictions = ocr_model.predict(batch, batch_size=32, verbose=0)
    recognized_words = decode_predictions(predictions, encoder)
    
    print(f"\nRecognized words:")
    for i, word in enumerate(recognized_words, 1):
        print(f"  {i:3d}. '{word}'")
    
    full_text = ' '.join(recognized_words)
    print(f"\n✓ Full recognized text: '{full_text}'")
    
    # Draw recognized text on the bounding boxes
    print("\n✓ Creating visualization with recognized text...")
    output_image = image.copy()
    
    for i, (box, word) in enumerate(zip(boxes, recognized_words)):
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate text size to position it properly
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(word, font, font_scale, thickness)
        
        # Position text above the box (or below if not enough space)
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + text_size[1] + 10
        text_x = x1
        
        # Draw background rectangle for text
        bg_x1 = text_x
        bg_y1 = text_y - text_size[1] - 5
        bg_x2 = text_x + text_size[0] + 5
        bg_y2 = text_y + 5
        cv2.rectangle(output_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(output_image, word, (text_x, text_y), 
                   font, font_scale, (0, 0, 255), thickness)
    
    # Save annotated image
    output_image_path = 'recognized_output.jpg'
    cv2.imwrite(output_image_path, output_image)
    print(f"✓ Annotated image saved to '{output_image_path}'")
    
    # STEP 3: Next Word Prediction with GPT-2
    print(f"\n{'='*80}")
    print("STEP 3: NEXT WORD PREDICTION")
    print(f"{'='*80}")
    
    # Determine device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    
    try:
        gpt_model, tokenizer = load_gpt2_model(device)
        
        print(f"\nGenerating top {num_predictions} next word predictions for:")
        print(f"  '{full_text}'")
        
        predictions = generate_next_words(
            gpt_model, 
            tokenizer, 
            full_text, 
            num_predictions=num_predictions,
            temperature=1.0,
            top_k=50,
            device=device
        )
        
        print(f"\n✓ Top {len(predictions)} next word predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i:2d}. '{word}' (probability: {prob:.4f})")
        
    except Exception as e:
        print(f"❌ ERROR in next word prediction: {e}")
        import traceback
        traceback.print_exc()
        predictions = []
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    output_file = 'pipeline_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE PIPELINE RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Words detected: {len(recognized_words)}\n\n")
        f.write("Recognized text:\n")
        f.write(f"  {full_text}\n\n")
        f.write(f"Top {len(predictions)} next word predictions:\n")
        for i, (word, prob) in enumerate(predictions, 1):
            f.write(f"  {i:2d}. '{word}' (probability: {prob:.4f})\n")
    
    print(f"✓ Results saved to '{output_file}'")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}\n")
    
    return {
        'recognized_text': full_text,
        'recognized_words': recognized_words,
        'next_word_predictions': predictions
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Run with defaults - no arguments required
    complete_pipeline(
        image_path=IMAGE_PATH,
        num_predictions=NUM_PREDICTIONS,
        visualize=False
    )
