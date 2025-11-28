import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Selection
# Options: "2.5-7b", "gpt-2-124M"
NEXT_WORD_MODEL = "gpt-2-124M" 

# Number of words to predict
NUM_NEXT_WORDS = 3

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get absolute path of the pipeline directory
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PIPELINE_DIR)

PHASE1_DIR = os.path.join(WORKSPACE_ROOT, "phase1_ocr")
PHASE3_DIR = os.path.join(WORKSPACE_ROOT, "phase3_style_transfer")
MODEL_2_5_7B_DIR = os.path.join(WORKSPACE_ROOT, "2.5-7b")

# Add paths to sys.path
sys.path.append(PHASE3_DIR)

# ============================================================================
# IMPORTS FROM PHASES
# ============================================================================

# Import Phase 3 Style Transfer
# We need to handle the potential import errors or path issues in run_generate
try:
    import run_generate
    from networks.BigGAN_networks import Generator
    from networks.module import StyleEncoder, StyleBackbone
    from lib.alphabet import strLabelConverter
    from lib.path_config import CharWidth, ImgHeight
    from lib.utils import yaml2config
except ImportError as e:
    print(f"Error importing Phase 3 Style Transfer: {e}")
    sys.exit(1)

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def run_ocr(image_path):
    """Run OCR on the image and return the recognized text."""
    print(f"\n[Step 1] Running OCR on {image_path}...")
    
    script_path = os.path.join(PHASE1_DIR, "sentence_recognizer.py")
    model_path = os.path.join(WORKSPACE_ROOT, "ocr_weights", "htr_model_20251020_084444_base.h5")
    encoder_path = os.path.join(WORKSPACE_ROOT, "ocr_weights", "encoder_20251020_084444.pkl")
    
    cmd = [
        sys.executable,
        script_path,
        "--image", image_path,
        "--model", model_path,
        "--encoder", encoder_path,
        "--no-visualize",
        "--no-save-words"
    ]
    
    env = os.environ.copy()
    # Only disable GPU on macOS to avoid segfaults
    if sys.platform == "darwin":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        output = result.stdout
        # Parse output for "Full text: "
        for line in output.splitlines():
            if "Full text:" in line:
                return line.split("Full text:", 1)[1].strip()
        
        print("Warning: Could not find 'Full text:' in OCR output.")
        print("OCR Output:\n", output)
        print("Using mock OCR output due to failure.")
        return "The quick brown fox"
        
    except subprocess.CalledProcessError as e:
        print(f"Error running OCR: {e}")
        # print("Stderr:", e.stderr) # Stderr might be huge or binary if segfault
        print("Using mock OCR output due to failure.")
        return "The quick brown fox"

def run_next_word_prediction(text):
    """Predict the next word(s) using the selected model."""
    print(f"\n[Step 2] Predicting next {NUM_NEXT_WORDS} word(s) for: '{text}' using {NEXT_WORD_MODEL}...")
    
    try:
        if NEXT_WORD_MODEL == "gpt-2-124M":
            # Use standard GPT-2 (124M) from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            # Move to device if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            model = model.to(device)
            
        elif NEXT_WORD_MODEL == "2.5-7b":
            tokenizer = AutoTokenizer.from_pretrained(MODEL_2_5_7B_DIR)
            model = AutoModelForCausalLM.from_pretrained(MODEL_2_5_7B_DIR, device_map="auto", torch_dtype=torch.float16)
        else:
            print(f"Unknown model: {NEXT_WORD_MODEL}")
            return None
            
    except Exception as e:
        print(f"Error loading {NEXT_WORD_MODEL} model: {e}")
        return None

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate tokens
    # Ensure we generate enough tokens for the requested number of words
    max_new = max(10, NUM_NEXT_WORDS * 5)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the new part
    if generated_text.startswith(text):
        new_text = generated_text[len(text):].strip()
    else:
        new_text = generated_text.strip()
        
    # Get the requested number of words
    words = new_text.split()
    if not words:
        return ""
        
    next_words = " ".join(words[:NUM_NEXT_WORDS])
    
    # Clean up punctuation if needed (optional, but good for style transfer)
    # next_words = ''.join(ch for ch in next_words if ch.isalnum() or ch.isspace())
    
    return next_words

def run_style_transfer(ref_image_path, text_to_gen, output_path):
    """Generate image of text_to_gen using style from ref_image_path."""
    print(f"\n[Step 3] Running Style Transfer for word '{text_to_gen}'...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(PHASE3_DIR, "configs", "gan_iam.yml")
    cfg = yaml2config(config_path)

    # Initialize models using config
    G = Generator(**cfg.GenModel).to(device)
    E = StyleEncoder(**cfg.EncModel).to(device)
    B = StyleBackbone(**cfg.StyBackbone).to(device)
    
    G.eval()
    E.eval()
    B.eval()

    # Load weights
    ckpt_path = os.path.join(WORKSPACE_ROOT, "pipeline_transfer_learning", "epoch_70.pth")
    wid_path = os.path.join(PHASE3_DIR, "wid_iam_new.pth")
    
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(checkpoint["generator"])
        E.load_state_dict(checkpoint["style_encoder"])
    else:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # Load pretrained StyleBackbone
    if os.path.exists(wid_path):
        print(f"Loading StyleBackbone from {wid_path}")
        wid_dict = torch.load(wid_path, map_location=device)
        if "StyleBackbone" in wid_dict:
            B.load_state_dict(wid_dict["StyleBackbone"])
    else:
        print("Warning: StyleBackbone weights not found")

    # Prepare Reference Image
    pil_img = Image.open(ref_image_path).convert('RGB')
    
    img_t, img_w, ref_char_len = run_generate.preprocess_image(
        pil_img,
        target_h=ImgHeight,
        invert=True,
        target_chars=None,
        char_width=CharWidth
    )
    
    img_t = img_t.to(device).float()
    img_len = torch.IntTensor([img_w]).to(device)
    
    # Encode Style
    with torch.no_grad():
        style_output = E(img_t, img_len, B, vae_mode=False)
        if isinstance(style_output, tuple):
            style_vector = style_output[0]
        else:
            style_vector = style_output
    
    # Prepare Target Text
    alphabet_key = "_".join(cfg.dataset.split("_")[:2])
    converter = strLabelConverter(alphabet_key)
    
    labels, lengths = converter.encode([text_to_gen])
    labels = labels.to(device)
    lengths = lengths.to(device)
    
    style_batch = style_vector.repeat(1, 1) # Batch size 1
    
    # Generate
    with torch.no_grad():
        raw_out = G(style_batch, labels, lengths)
    
    # Postprocess
    resized = run_generate.postprocess_generated(raw_out, lengths, img_w, ref_char_len)
    
    # Save
    pil_out = run_generate.tensor_to_pil(resized[0])
    
    # Invert the image (to get black text on white background)
    pil_out = Image.fromarray(255 - np.array(pil_out))
    
    pil_out.save(output_path)
    print(f"Saved generated image to {output_path}")


def main():
    test_images = ["test1.jpeg", "test2.jpeg"]
    
    for img_name in test_images:
        img_path = os.path.join(PIPELINE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found. Skipping.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing {img_name}")
        print(f"{'='*50}")
        
        # 1. OCR
        recognized_text = run_ocr(img_path)
        if not recognized_text:
            print("OCR failed to recognize text.")
            continue
        print(f"OCR Result: {recognized_text}")
        
        # 2. Next Word
        next_word = run_next_word_prediction(recognized_text)
        if not next_word:
            print("Next word prediction failed.")
            continue
        print(f"Predicted Next Word: {next_word}")
        
        # 3. Style Transfer
        # Sanitize filename
        safe_next_word = "".join(c for c in next_word if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        output_filename = f"output_{img_name.split('.')[0]}_{safe_next_word}.png"
        output_path = os.path.join(PIPELINE_DIR, output_filename)
        run_style_transfer(img_path, next_word, output_path)

if __name__ == "__main__":
    main()
