"""
Train Beq - Your own language model from scratch.
Pure PyTorch. No external AI APIs.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import BeqTransformer, CharTokenizer, count_parameters


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train Beq language model")
    parser.add_argument("--data", type=str, default="data/input.txt", help="Path to training text")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Where to save model")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=1000)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Creating a small sample dataset so you can test immediately...")
        sample = (
            "Beq is an open-source AI. User: hi\nBeq: Hello! How can I help?\n"
            "User: Who are you?\nBeq: I am Beq, a pure PyTorch language model.\n"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(sample * 200, encoding="utf-8")
        print(f"Sample data written to {data_path}")

    text = data_path.read_text(encoding="utf-8")
    extra_path = data_path.parent / "input_extra.txt"
    if extra_path.exists() and extra_path.resolve() != data_path.resolve():
        text = text + "\n" + extra_path.read_text(encoding="utf-8")
        print(f"Also loaded extra data: {extra_path} ({extra_path.stat().st_size} bytes)")

    print(f"Loaded {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_loader = DataLoader(
        TextDataset(train_data, args.block_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TextDataset(val_data, args.block_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.block_size,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")

    best_val_loss = float("inf")
    step = 0
    model.train()
    start_time = time.time()

    print("\nStarting training...")
    print("-" * 60)

    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                if len(val_loader) > 0:
                    for i, (vx, vy) in enumerate(val_loader):
                        if i >= 20:
                            break
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, vy)
                        val_losses.append(vloss.item())
            if not val_losses:
                avg_val = float("inf")
                print("--> val loss: skipped (val set too small for block_size)")
            else:
                avg_val = sum(val_losses) / len(val_losses)
                print(f"--> val loss: {avg_val:.4f}")

            if val_losses and avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt = {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "max_seq_len": args.block_size,
                    },
                    "step": step,
                    "val_loss": avg_val,
                }
                torch.save(ckpt, out_dir / "beq_best.pt")
                print(f"    saved best checkpoint (val_loss={avg_val:.4f})")

            model.train()

        if step % args.save_interval == 0 and step > 0:
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "max_seq_len": args.block_size,
                },
                "step": step,
            }
            torch.save(ckpt, out_dir / f"beq_step_{step}.pt")

        step += 1

    ckpt = {
        "model": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_seq_len": args.block_size,
        },
        "step": step,
    }
    torch.save(ckpt, out_dir / "beq_final.pt")
    print(f"\nTraining finished. Model saved to {out_dir}")


if __name__ == "__main__":
    main()
