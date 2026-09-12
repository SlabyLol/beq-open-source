"""
Simple command-line generation with Beq.
"""

import argparse
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BeqTransformer, CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/beq_best.pt")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_path = ckpt_path.parent / "tokenizer.json"

    if not ckpt_path.exists():
        print("Checkpoint not found. Train the model first:")
        print("  python train/train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer.load(tok_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = BeqTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tokenizer.decode(out[0].tolist())
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
