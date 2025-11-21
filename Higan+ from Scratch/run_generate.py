import argparse
import math
import os
import sys
from pathlib import Path
import traceback

ROOT = r"B:\College\DL\handwriting_autocomplete_system\Higan+ from Scratch"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
from PIL import Image

# Import Generator and encoders
from networks.BigGAN_networks import Generator
from networks.module import StyleEncoder, StyleBackbone
from networks.utils import rescale_images2
from lib.alphabet import strLabelConverter
from lib.path_config import CharWidth, ImgHeight

OUT_DIR = Path(ROOT)
DEFAULT_REF_IMAGE = OUT_DIR / 'image.png'


def find_state_dict(ckpt):
    """Return a state_dict if ckpt wraps one, otherwise return ckpt itself."""
    if isinstance(ckpt, dict):
        for k in ['state_dict', 'model', 'G', 'Generator', 'generator', 'netG']:
            if k in ckpt:
                return ckpt[k]
        # look for a dict that looks like a state dict
        for v in ckpt.values():
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                return v
    return ckpt


def normalize_state_dict(sd):
    new = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            new[k[len('module.'):]] = v
        else:
            new[k] = v
    return new


def try_load_from_full_ckpt(model, full_ckpt, possible_keys):
    """Try to load model state from various keys in a full checkpoint dict."""
    for k in possible_keys:
        if k in full_ckpt:
            try:
                model.load_state_dict(full_ckpt[k])
                print(f'Loaded {k} into {type(model).__name__}')
                return True
            except Exception:
                pass
    # fuzzy match
    for k in full_ckpt.keys():
        if k.lower().find(type(model).__name__.lower()) >= 0:
            try:
                model.load_state_dict(full_ckpt[k])
                print(f'Loaded {k} (fuzzy) into {type(model).__name__}')
                return True
            except Exception:
                pass
    return False


def tensor_to_pil(img_tensor):
    arr = img_tensor.detach().cpu().numpy() if isinstance(img_tensor, torch.Tensor) else np.array(img_tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.min() < -0.5 and arr.max() <= 1.0:
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def ensure_multiple(value, base):
    return int(math.ceil(value / float(base))) * base


def preprocess_image(pil_img, target_h=ImgHeight, invert=True, target_chars=None, char_width=CharWidth):
    """Convert PIL image to model input: grayscale, resize height, and align width to char multiples."""
    from torchvision.transforms import ToTensor, Normalize, Compose

    pil_img = pil_img.convert('L')
    w, h = pil_img.size
    scaled_w = max(1, int(round(w * (target_h / float(h)))))
    target_chars = max(1, target_chars or int(round(scaled_w / float(char_width))))
    aligned_w = ensure_multiple(max(scaled_w, target_chars * char_width), char_width)
    pil_img = pil_img.resize((aligned_w, target_h), Image.BILINEAR)
    arr = np.array(pil_img).astype(np.uint8)
    if invert:
        arr = 255 - arr
    pil_img = Image.fromarray(arr, mode='L')
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
    t = transform(pil_img)
    final_chars = max(1, aligned_w // char_width)
    return t.unsqueeze(0), aligned_w, final_chars


def encode_texts(converter, texts, device):
    if len(texts) == 1:
        encoded = converter.encode(texts[0])
        labels = torch.LongTensor(encoded).unsqueeze(0)
        lengths = torch.IntTensor([len(encoded)])
    else:
        labels, lengths = converter.encode(texts)
        if isinstance(labels, list):
            labels = torch.LongTensor(labels)
    labels = labels.to(device)
    lengths = lengths.to(device)
    return labels, lengths


def infer_reference_text(path):
    base = path.stem
    allowed = strLabelConverter('all').alphabet
    cleaned = ''.join(ch for ch in base if ch in allowed)
    return cleaned or 'hello'


def load_from_paths(model, possible_keys, description, *candidate_paths):
    for cand in candidate_paths:
        if cand is None or not cand.exists():
            continue
        try:
            ckpt = torch.load(str(cand), map_location='cpu')
        except Exception as exc:
            print(f'Warning: failed to load {description} fallback {cand}: {exc}')
            continue
        if isinstance(ckpt, dict):
            if try_load_from_full_ckpt(model, ckpt, possible_keys):
                print(f'Loaded {description} from {cand}')
                return True
            sd_candidate = find_state_dict(ckpt)
            if isinstance(sd_candidate, dict):
                try:
                    model.load_state_dict(normalize_state_dict(sd_candidate), strict=False)
                    print(f'Loaded {description} state_dict from {cand}')
                    return True
                except Exception as exc:
                    print(f'Warning: failed to apply {description} state_dict from {cand}: {exc}')
        if hasattr(ckpt, 'state_dict'):
            try:
                model.load_state_dict(ckpt.state_dict())
                print(f'Loaded {description} module from {cand}')
                return True
            except Exception as exc:
                print(f'Warning: failed to load {description} module from {cand}: {exc}')
    print(f'Warning: unable to locate weights for {description}; generation quality may degrade.')
    return False


def postprocess_generated(imgs, text_lens, ref_img_w, ref_char_len):
    img_lens = torch.full((imgs.size(0),), imgs.size(-1), dtype=torch.int, device=imgs.device)
    target_ref_w = torch.full((imgs.size(0),), ref_img_w, dtype=torch.int, device=imgs.device)
    target_ref_chars = torch.full((imgs.size(0),), ref_char_len, dtype=torch.int, device=imgs.device)
    resized_imgs, _ = rescale_images2(imgs, img_lens, text_lens, target_ref_w, target_ref_chars)
    return resized_imgs


def parse_args():
    parser = argparse.ArgumentParser(description='Generate HiGAN+ samples from a checkpoint.')
    parser.add_argument('--ckpt', default=str(OUT_DIR / 'server_files' / 'best_e_20.pth'))
    parser.add_argument('--reference-image', default=str(DEFAULT_REF_IMAGE))
    parser.add_argument('--reference-text', default=None)
    parser.add_argument('--target-texts', nargs='*', default=None)
    parser.add_argument('--invert-reference', dest='invert_reference', action='store_true',
                        help='Invert reference pixels (matching training setup).')
    parser.add_argument('--no-invert-reference', dest='invert_reference', action='store_false',
                        help='Disable pixel inversion for reference image.')
    parser.add_argument('--img-height', type=int, default=ImgHeight)
    parser.add_argument('--char-width', type=int, default=CharWidth)
    parser.add_argument('--output-dir', default=str(OUT_DIR))
    parser.add_argument('--device', default=None)
    parser.set_defaults(invert_reference=True)
    return parser.parse_args()


def main(cli_args=None):
    args = parse_args() if cli_args is None else cli_args
    ckpt_path = Path(args.ckpt)
    print('Loading checkpoint:', ckpt_path)
    full_ckpt = torch.load(str(ckpt_path), map_location='cpu')

    # Determine generator state dict
    sd_candidate = find_state_dict(full_ckpt)
    sd = sd_candidate if isinstance(sd_candidate, dict) and any(isinstance(x, torch.Tensor) for x in sd_candidate.values()) else None
    if sd is None:
        # maybe full_ckpt itself is state dict
        sd = full_ckpt if isinstance(full_ckpt, dict) and any(isinstance(x, torch.Tensor) for x in full_ckpt.values()) else None

    if sd is not None:
        print('state_dict sample keys:', list(sd.keys())[:10])
        sd = normalize_state_dict(sd)

    # Instantiate models using config values from configs/gan_iam.yml
    gen_kwargs = dict(G_ch=64, style_dim=32, embed_dim=120, bottom_width=4, bottom_height=4,
                      resolution=64, G_kernel_size=3, G_attn='0', n_class=80, input_nc=1)
    E_kwargs = dict(style_dim=32, in_dim=256)
    B_kwargs = dict(resolution=16, max_dim=256, in_channel=1)

    print('Instantiating Generator with', gen_kwargs)
    G = Generator(**gen_kwargs)
    E = StyleEncoder(**E_kwargs)
    B = StyleBackbone(**B_kwargs)

    # Load weights: prefer full checkpoint keys like 'generator','style_encoder','style_backbone'
    loaded_G = False
    if isinstance(full_ckpt, dict):
        # try full checkpoint mapping
        loaded_G = try_load_from_full_ckpt(G, full_ckpt, ['generator', 'Generator', 'G'])
    if not loaded_G and sd is not None:
        try:
            G.load_state_dict(sd, strict=False)
            print('Loaded generator state_dict (fallback)')
            loaded_G = True
        except Exception as e:
            print('Failed to load generator from state_dict:', e)

    # Try to load encoder/backbone from full ckpt
    loaded_E = try_load_from_full_ckpt(E, full_ckpt, ['style_encoder', 'StyleEncoder', 'E'])
    loaded_B = try_load_from_full_ckpt(B, full_ckpt, ['style_backbone', 'StyleBackbone', 'B'])

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    G = G.to(device); G.eval()
    E = E.to(device); E.eval()
    B = B.to(device); B.eval()

    if not loaded_E:
        loaded_E = load_from_paths(E, ['style_encoder', 'StyleEncoder', 'E'], 'StyleEncoder',
                                   Path(ROOT) / 'server_files' / 'best_e_20.pth',
                                   Path(ROOT) / 'pretrained' / 'ocr_iam_new.pth')
    if not loaded_B:
        loaded_B = load_from_paths(B, ['style_backbone', 'StyleBackbone', 'B'], 'StyleBackbone',
                                   Path(ROOT) / 'server_files' / 'best_e_20.pth',
                                   Path(ROOT) / 'pretrained' / 'wid_iam_new.pth')

    # If the repository provided an input image, try to use it as reference style
    img_path = Path(args.reference_image)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if img_path.exists() and loaded_E and loaded_B:
        print('Found reference image and encoder/backbone loaded -> performing image-conditioned generation')
        pil = Image.open(str(img_path))
        ref_text = args.reference_text or infer_reference_text(img_path)
        img_t, img_w, ref_char_len = preprocess_image(
            pil,
            target_h=args.img_height,
            invert=args.invert_reference,
            target_chars=len(ref_text) if args.reference_text else None,
            char_width=args.char_width,
        )
        img_t = img_t.to(device).float()
        img_len = torch.IntTensor([img_w]).to(device)
        with torch.no_grad():
            enc_z = E(img_t, img_len, B, vae_mode=False)
        print('enc_z.shape =', getattr(enc_z, 'shape', None))

        converter = strLabelConverter('all')
        target_texts = args.target_texts or []
        if ref_text not in target_texts:
            target_texts = [ref_text] + target_texts

        labels, lengths = encode_texts(converter, target_texts, device)
        style_batch = enc_z.repeat(len(target_texts), 1)
        with torch.no_grad():
            raw_out = G(style_batch, labels, lengths)
        resized = postprocess_generated(raw_out, lengths, img_w, ref_char_len)
        for text, tensor_img in zip(target_texts, resized):
            pil_out = tensor_to_pil(tensor_img)
            safe_text = ''.join(ch for ch in text if ch.isalnum()) or 'sample'
            out_path = output_root / f'image_from_ref_{safe_text}.png'
            pil_out.save(str(out_path))
            print('Saved', out_path)
    else:
        print('Reference image or encoder/backbone not available - falling back to random sampling')
        # fallback: sample random styles as before
        saved = []
        seeds = [0, 1, 2, 3]
        for i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            z = torch.randn(1, G.style_dim, device=device)
            seq_len = 8
            y = torch.randint(0, G.n_classes, (1, seq_len), dtype=torch.long, device=device)
            y_lens = torch.tensor([seq_len], device=device)
            with torch.no_grad():
                out = G(z, y, y_lens)
            pil = tensor_to_pil(out[0])
            fname = output_root / (f'image_seed{seed}.png' if i > 0 else 'image.png')
            pil.save(str(fname))
            print('Saved', fname)
            saved.append(str(fname))
        print('Done. Saved files:', saved)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Error during generation:')
        traceback.print_exc()
        sys.exit(1)
