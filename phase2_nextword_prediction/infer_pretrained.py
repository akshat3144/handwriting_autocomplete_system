import argparse
import torch
import torch.nn.functional as F
import tiktoken
from model import GPT


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(
    model: GPT,
    enc: tiktoken.core.Encoding,
    prompt: str,
    num_seq: int = 1,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
    seed: int = 42,
):
    model.eval()
    # Encode prompt and create batch
    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_seq, 1).to(device)

    # Clamp to context length if needed
    context_len = model.config.context_length
    if idx.shape[1] > context_len:
        idx = idx[:, -context_len:]

    # Dedicated RNG so we don't disturb global state
    g = torch.Generator(device=device).manual_seed(seed)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(idx)
            logits = logits[:, -1, :]  # (B, vocab)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                # Top-k filter
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                next_idx_in_topk = torch.multinomial(probs, num_samples=1, generator=g)
                next_token = torch.gather(topk_idx, -1, next_idx_in_topk)  # (B,1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=g)
        idx = torch.cat([idx, next_token], dim=1)
        if idx.shape[1] > context_len:
            idx = idx[:, -context_len:]

    # Decode
    outs = []
    for i in range(num_seq):
        toks = idx[i].tolist()
        text = enc.decode(toks)
        outs.append(text)
    return outs


def parse_args():
    p = argparse.ArgumentParser(description="Generate text completions using pretrained GPT-2 weights")
    p.add_argument("--prompt", type=str, default="Hello, I am", help="Prompt to complete")
    p.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="Pretrained GPT-2 variant")
    p.add_argument("--num_seq", type=int, default=1, help="Number of samples to generate")
    p.add_argument("--max_new_tokens", type=int, default=50, help="Number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling; set 0 to disable")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto-detected if omitted)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"using device: {device}")

    # Load pretrained GPT-2 weights into our GPT implementation
    model = GPT.from_pretrained(args.model_type)
    model = model.to(device)

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    outs = generate(
        model=model,
        enc=enc,
        prompt=args.prompt,
        num_seq=args.num_seq,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )

    for i, text in enumerate(outs):
        print(f"\n> completion {i}:\n{text}")


if __name__ == "__main__":
    main()
