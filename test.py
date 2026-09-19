#!/usr/bin/env python3
"""
Interaktiver Chat mit modul.safetensors

Installation:
    pip install safetensors torch transformers

Start:
    python chat_safetensors.py modul.safetensors --config-dir ./model_folder

Der Ordner (--config-dir) braucht config.json und die Tokenizer-Dateien
(tokenizer.json / tokenizer_config.json usw.).

Befehle im Chat:
    /reset  -> Verlauf löschen
    /exit   -> Beenden
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)


def load_model(path: Path, config_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(config_dir)
    tokenizer = AutoTokenizer.from_pretrained(config_dir)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    state_dict = load_file(str(path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warnung: {len(missing)} fehlende Keys (z.B. {missing[:3]})")
    if unexpected:
        print(f"Warnung: {len(unexpected)} unerwartete Keys (z.B. {unexpected[:3]})")

    model.to(device).eval()
    return model, tokenizer, device


def build_inputs(tokenizer, history, device):
    """Nutzt das Chat-Template des Modells, falls vorhanden."""
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids, torch.Tensor):  # neuere transformers-Versionen
            ids = ids["input_ids"]
        return ids.to(device)

    # Fallback ohne Template: einfacher Text-Prompt
    text = ""
    for m in history:
        who = "User" if m["role"] == "user" else "Assistant"
        text += f"{who}: {m['content']}\n"
    text += "Assistant:"
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def chat(model, tokenizer, device, system, max_new_tokens, temperature):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": system}] if system else []

    print("Chat gestartet. /reset = Verlauf löschen, /exit = Beenden\n")
    while True:
        try:
            user = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": system}] if system else []
            print("Verlauf gelöscht.\n")
            continue

        history.append({"role": "user", "content": user})
        input_ids = build_inputs(tokenizer, history, device)

        print("Modell: ", end="", flush=True)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        history.append({"role": "assistant", "content": reply})
        print()


def main():
    p = argparse.ArgumentParser(description="Chat mit einer .safetensors-Datei")
    p.add_argument("path", nargs="?", default="modul.safetensors")
    p.add_argument("--config-dir", type=Path, required=True,
                   help="Ordner mit config.json und Tokenizer-Dateien")
    p.add_argument("--system", default="Du bist ein hilfreicher Assistent.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Datei nicht gefunden: {path}")

    model, tokenizer, device = load_model(path, args.config_dir)
    chat(model, tokenizer, device, args.system, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
