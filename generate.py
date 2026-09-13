"""
Simple command-line generation with Beq.
"""

import argparse
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BeqTransformer, CharTokenizer
from web.mathtool import try_math_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/beq_best.pt")
    args = parser.parse_args()

    # Simple arithmetic never needs the (tiny, from-scratch) language model —
    # answer it directly and correctly instead of letting Beq guess.
    math_answer = try_math_answer(args.prompt)
    if math_answer is not None:
        print("\n" + "=" * 60)
        print(math_answer)
        print("=" * 60)
        return

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
            repetition_penalty=args.repetition_penalty,
        )

    text = tokenizer.decode(out[0].tolist())
    # Cut off once the model starts hallucinating a fake "User:" turn or
    # rambles into a second paragraph — cosmetic only, doesn't fix wording.
    for marker in ("\nUser:", "\nuser:", "\n\n"):
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut]

    print("\n" + "=" * 60)
    print(text.strip())
    print("=" * 60)
    print(
        "\nNote: Beq is a tiny character-level model trained on ~11KB of "
        "text. Coherent, factual answers need a much bigger and more "
        "diverse training corpus — see README 'Warum redet Beq wirr?'."
    )


if __name__ == "__main__":
    main()
