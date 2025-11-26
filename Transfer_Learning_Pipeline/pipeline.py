"""
Handwriting Autocomplete Pipeline
==================================
1. TrOCR: Extract text from handwriting image
2. GPT-2: Predict next word(s)
3. HiGAN+: Generate handwritten text in extracted style
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Add phase3 to path for HiGAN imports
PHASE3_PATH = Path(__file__).parent.parent / "phase3_style_transfer"
sys.path.insert(0, str(PHASE3_PATH))

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GPT2LMHeadModel, GPT2Tokenizer
from lib.utils import yaml2config
from lib.alphabet import strLabelConverter
from networks.BigGAN_networks import Generator
from networks.module import StyleEncoder, StyleBackbone


class HandwritingAutocompletePipeline:
    """End-to-end pipeline for handwriting autocomplete."""
    
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (Path(__file__).parent / "epoch_70.pth")
        self.img_height = 64
        self.char_width = 32
        
        self._load_ocr_model()
        self._load_gpt_model()
        self._load_higan_model()
        
        print(f"Pipeline initialized on {self.device}")
    
    def _load_ocr_model(self):
        """Load TrOCR for handwriting recognition."""
        print("Loading TrOCR...")
        self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(self.device)
        self.ocr_model.eval()
    
    def _load_gpt_model(self):
        """Load GPT-2 for text completion."""
        print("Loading GPT-2...")
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.gpt_model = GPT2LMHeadModel.from_pretrained(
            "gpt2-medium",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.gpt_model.eval()
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
    
    def _load_higan_model(self):
        """Load HiGAN+ for handwriting generation."""
        print("Loading HiGAN+...")
        
        # Load config
        config_path = PHASE3_PATH / "configs" / "gan_iam.yml"
        cfg = yaml2config(str(config_path))
        
        # Initialize label converter
        alphabet_key = "_".join(cfg.dataset.split("_")[:2])
        self.label_converter = strLabelConverter(alphabet_key)
        
        # Initialize models
        self.generator = Generator(**cfg.GenModel).to(self.device)
        self.style_backbone = StyleBackbone(**cfg.StyBackbone).to(self.device)
        self.style_encoder = StyleEncoder(**cfg.EncModel).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.style_encoder.load_state_dict(checkpoint["style_encoder"])
        
        # Load pretrained StyleBackbone
        wid_path = PHASE3_PATH / "wid_iam_new.pth"
        if wid_path.exists():
            wid_dict = torch.load(wid_path, map_location=self.device)
            if "StyleBackbone" in wid_dict:
                self.style_backbone.load_state_dict(wid_dict["StyleBackbone"])
        
        self.generator.eval()
        self.style_encoder.eval()
        self.style_backbone.eval()
    
    def extract_text(self, image_path):
        """Extract text from handwriting image using TrOCR."""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.ocr_model.generate(pixel_values, max_new_tokens=128)
        
        text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    
    def predict_next_words(self, text, num_words=3):
        """Predict next words using GPT-2."""
        inputs = self.gpt_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt_model.generate(
                inputs.input_ids,
                max_new_tokens=num_words * 3,
                min_new_tokens=2,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                pad_token_id=self.gpt_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
        
        generated = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only new words
        if generated.startswith(text):
            continuation = generated[len(text):].strip()
        else:
            continuation = generated.strip()
        
        continuation = continuation.lstrip(".,!?;:")
        for i, char in enumerate(continuation):
            if char in ".!?" and i > 0:
                continuation = continuation[:i]
                break
        
        words = continuation.split()[:num_words]
        return " ".join(words).rstrip(".,!?;:")
    
    def _preprocess_image(self, image_path):
        """Preprocess image for style extraction."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        r = self.img_height / float(h)
        new_w = max(int(w * r), self.char_width * 2)
        new_w = new_w + (self.char_width - new_w % self.char_width) % self.char_width
        
        img = cv2.resize(img, (new_w, self.img_height), 
                        interpolation=cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR)
        
        if img.mean() > 127:
            img = 255 - img
        
        img_float = img.astype(np.float32) / 255.0
        img_normalized = (img_float - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()
        
        return img_tensor, new_w
    
    def extract_style(self, image_path):
        """Extract style vector from handwriting image."""
        img_tensor, img_width = self._preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        img_len = torch.tensor([img_width], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            style_vector = self.style_encoder(img_tensor, img_len, self.style_backbone, vae_mode=False)
        
        return style_vector
    
    def generate_handwriting(self, style_vector, text):
        """Generate handwritten text using extracted style."""
        labels, label_lens = self.label_converter.encode([text])
        labels = labels.to(self.device)
        label_lens = label_lens.to(self.device)
        
        with torch.no_grad():
            fake_img = self.generator(style_vector, labels, label_lens)
        
        return fake_img
    
    @staticmethod
    def tensor_to_image(tensor):
        """Convert output tensor to displayable numpy image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        
        img_np = tensor.cpu().numpy()
        img_np = (img_np + 1) / 2.0
        img_np = 1.0 - img_np  # Invert for white background
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        return img_np
    
    def run(self, image_path, num_words=3, output_path=None):
        """
        Run the complete pipeline.
        
        Args:
            image_path: Path to input handwriting image
            num_words: Number of words to predict
            output_path: Optional path to save generated image
        
        Returns:
            dict with original_text, predicted_words, generated_image
        """
        print(f"Processing: {image_path}")
        
        # Step 1: OCR
        original_text = self.extract_text(image_path)
        print(f"Extracted text: '{original_text}'")
        
        # Step 2: Predict next words
        predicted_words = self.predict_next_words(original_text, num_words)
        print(f"Predicted words: '{predicted_words}'")
        
        # Step 3: Extract style and generate
        style_vector = self.extract_style(image_path)
        generated_tensor = self.generate_handwriting(style_vector, predicted_words)
        generated_image = self.tensor_to_image(generated_tensor)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), generated_image)
            print(f"Saved to: {output_path}")
        
        return {
            "original_text": original_text,
            "predicted_words": predicted_words,
            "completed_text": f"{original_text} {predicted_words}",
            "generated_image": generated_image
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Handwriting Autocomplete Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Input handwriting image")
    parser.add_argument("--num_words", type=int, default=3, help="Number of words to predict")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--checkpoint", type=str, default=None, help="HiGAN checkpoint path")
    args = parser.parse_args()
    
    pipeline = HandwritingAutocompletePipeline(checkpoint_path=args.checkpoint)
    result = pipeline.run(args.image, num_words=args.num_words, output_path=args.output)
    
    print("\nResult:")
    print(f"  Original: {result['original_text']}")
    print(f"  Predicted: {result['predicted_words']}")
    print(f"  Complete: {result['completed_text']}")


if __name__ == "__main__":
    main()
