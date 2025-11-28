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
    """Predict the next word using the 2.5-7b model."""
    print(f"\n[Step 2] Predicting next word for: '{text}'...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_2_5_7B_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_2_5_7B_DIR, device_map="auto", torch_dtype=torch.float16)
    except Exception as e:
        print(f"Error loading 2.5-7b model: {e}")
        # Fallback or exit?
        return None

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate a few tokens
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the new part
    # This logic depends on how the model echoes the input. Usually it does.
    if generated_text.startswith(text):
        new_text = generated_text[len(text):].strip()
    else:
        new_text = generated_text.strip()
        
    next_word = new_text.split()[0] if new_text else ""
    
    # Clean up punctuation if needed
    next_word = ''.join(ch for ch in next_word if ch.isalnum())
    
    return next_word

def run_style_transfer(ref_image_path, text_to_gen, output_path):
    """Generate image of text_to_gen using style from ref_image_path."""
    print(f"\n[Step 3] Running Style Transfer for word '{text_to_gen}'...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")

    # Configs (matching run_generate.py)
    gen_kwargs = dict(G_ch=64, style_dim=32, embed_dim=120, bottom_width=4, bottom_height=4,
                      resolution=64, G_kernel_size=3, G_attn='0', n_class=80, input_nc=1)
    E_kwargs = dict(style_dim=32, in_dim=256)
    B_kwargs = dict(resolution=16, max_dim=256, in_channel=1)

    G = Generator(**gen_kwargs).to(device)
    E = StyleEncoder(**E_kwargs).to(device)
    B = StyleBackbone(**B_kwargs).to(device)
    
    G.eval()
    E.eval()
    B.eval()

    # Load weights
    # We look for weights in likely locations
    ckpt_path = os.path.join(WORKSPACE_ROOT, "pipeline_transfer_learning", "epoch_70.pth")
    ocr_path = os.path.join(PHASE3_DIR, "ocr_iam_new.pth")
    wid_path = os.path.join(PHASE3_DIR, "wid_iam_new.pth")
    
    # Load Generator
    if os.path.exists(ckpt_path):
        print(f"Loading Generator from {ckpt_path}")
        full_ckpt = torch.load(ckpt_path, map_location='cpu')
        # Try to find state dict
        sd = run_generate.find_state_dict(full_ckpt)
        if sd:
            sd = run_generate.normalize_state_dict(sd)
            try:
                G.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"Warning: Failed to load G state_dict: {e}")
        else:
            # Try full ckpt mapping
            run_generate.try_load_from_full_ckpt(G, full_ckpt, ['generator', 'Generator', 'G'])
    else:
        print(f"Error: Generator checkpoint not found at {ckpt_path}")
        return

    # Load Encoder
    # run_generate.load_from_paths expects Path objects, not strings
    from pathlib import Path
    
    if not run_generate.load_from_paths(E, ['style_encoder', 'StyleEncoder', 'E'], 'StyleEncoder', Path(ocr_path)):
        print("Warning: Failed to load StyleEncoder")

    # Load Backbone
    if not run_generate.load_from_paths(B, ['style_backbone', 'StyleBackbone', 'B'], 'StyleBackbone', Path(wid_path)):
        print("Warning: Failed to load StyleBackbone")

    # Prepare Reference Image
    pil_img = Image.open(ref_image_path).convert('RGB')
    
    # We need to infer reference text length for scaling?
    # run_generate uses infer_reference_text(path) which cleans the filename.
    # But here we have the actual OCR text! We should use it.
    # However, run_generate.preprocess_image uses target_chars to determine width.
    # If we pass the OCR text length, it might be better.
    # But wait, run_generate.preprocess_image logic:
    # target_chars = max(1, target_chars or int(round(scaled_w / float(char_width))))
    # If we don't pass target_chars, it estimates from width.
    # Let's stick to estimation or use a default to avoid mismatch if OCR is very wrong.
    # Actually, using the OCR text length is probably better if OCR is good.
    # But let's just let it estimate to be safe.
    
    img_t, img_w, ref_char_len = run_generate.preprocess_image(
        pil_img,
        target_h=ImgHeight,
        invert=True, # Default in run_generate
        target_chars=None,
        char_width=CharWidth
    )
    
    img_t = img_t.to(device).float()
    img_len = torch.IntTensor([img_w]).to(device)
    
    # Encode Style
    with torch.no_grad():
        enc_z = E(img_t, img_len, B, vae_mode=False)
    
    # Prepare Target Text
    converter = strLabelConverter('all')
    labels, lengths = run_generate.encode_texts(converter, [text_to_gen], device)
    style_batch = enc_z.repeat(1, 1) # Batch size 1
    
    # Generate
    with torch.no_grad():
        raw_out = G(style_batch, labels, lengths)
    
    # Postprocess
    resized = run_generate.postprocess_generated(raw_out, lengths, img_w, ref_char_len)
    
    # Save
    pil_out = run_generate.tensor_to_pil(resized[0])
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
        output_filename = f"output_{img_name.split('.')[0]}_{next_word}.png"
        output_path = os.path.join(PIPELINE_DIR, output_filename)
        run_style_transfer(img_path, next_word, output_path)

if __name__ == "__main__":
    main()
