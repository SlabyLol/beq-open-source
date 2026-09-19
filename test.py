#!/usr/bin/env python3
"""
Load and run modul.safetensors

Install:
    pip install safetensors torch transformers

Usage:
    # 1) Inspect the file (tensor names, shapes, dtypes, metadata)
    python run_safetensors.py modul.safetensors

    # 2) Run it as a text-generation model (needs config.json + tokenizer files in a folder)
    python run_safetensors.py modul.safetensors --config-dir ./model_folder --prompt "Hello"
"""
import argparse
from pathlib import Path

from safetensors import safe_open


def inspect(path: Path, limit: int = 40) -> None:
    """Print metadata and tensor info without loading weights into memory."""
    total_params = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        print(f"File: {path}")
        print(f"Metadata: {f.metadata()}\n")
        keys = list(f.keys())
        for i, key in enumerate(keys):
            sl = f.get_slice(key)
            shape = sl.get_shape()
            n = 1
            for d in shape:
                n *= d
            total_params += n
            if i < limit:
                print(f"{key:70s} {str(shape):25s} {sl.get_dtype()}")
        if len(keys) > limit:
            print(f"... and {len(keys) - limit} more tensors")
    print(f"\nTensors: {len(keys)} | Parameters: {total_params:,}")


def run_causal_lm(path: Path, config_dir: Path, prompt: str, max_new_tokens: int) -> None:
    """Build a model from config.json, load the safetensors weights, generate text."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(config_dir)
    tokenizer = AutoTokenizer.from_pretrained(config_dir)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    state_dict = load_file(str(path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    model.to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or run a .safetensors model")
    parser.add_argument("path", nargs="?", default="modul.safetensors", help="Path to .safetensors file")
    parser.add_argument("--config-dir", type=Path, help="Folder with config.json and tokenizer files")
    parser.add_argument("--prompt", default="Hello, my name is", help="Prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if args.config_dir:
        run_causal_lm(path, args.config_dir, args.prompt, args.max_new_tokens)
    else:
        inspect(path)


if __name__ == "__main__":
    main()
