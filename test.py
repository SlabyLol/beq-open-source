#!/usr/bin/env python3
"""
Chat mit modul.safetensors – einfach starten:

    python chat.py

Alles ist voreingestellt. Lege einfach diese Datei, modul.safetensors,
config.json und die Tokenizer-Dateien in denselben Ordner.

Installation (einmalig):
    pip install safetensors torch transformers huggingface_hub
"""
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

# ============================ EINSTELLUNGEN ============================
BASE_DIR = Path(__file__).resolve().parent          # Ordner dieses Skripts
MODEL_FILE = BASE_DIR / "modul.safetensors"         # Gewichte
CONFIG_DIR = BASE_DIR                               # config.json + Tokenizer

# Optional: Hugging-Face-Modellname (z.B. "Qwen/Qwen2.5-0.5B-Instruct").
# Fehlen config.json/Tokenizer im Ordner, werden sie automatisch von dort geladen.
MODEL_ID = None

SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent. Antworte auf Deutsch."
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# =======================================================================


def ensure_config_files():
    """Lädt config/Tokenizer automatisch herunter, falls sie fehlen."""
    if (CONFIG_DIR / "config.json").exists():
        return
    if not MODEL_ID:
        raise SystemExit(
            "config.json fehlt in " + str(CONFIG_DIR) + ".\n"
            "Lege config.json und die Tokenizer-Dateien in den Ordner ODER "
            "trage oben in chat.py bei MODEL_ID den Hugging-Face-Namen des Modells ein."
        )
    from huggingface_hub import snapshot_download

    print(f"Lade config und Tokenizer von {MODEL_ID} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=CONFIG_DIR,
        allow_patterns=[
            "config.json", "generation_config.json", "tokenizer*",
            "special_tokens_map.json", "vocab*", "merges.txt", "*.model",
        ],
    )


def load_model():
    if not MODEL_FILE.exists():
        raise SystemExit(f"Datei nicht gefunden: {MODEL_FILE}")
    ensure_config_files()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Gerät: {device} | Lade Modell ...")

    config = AutoConfig.from_pretrained(CONFIG_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    state_dict = load_file(str(MODEL_FILE))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warnung: {len(missing)} fehlende Keys (z.B. {missing[:3]})")
    if unexpected:
        print(f"Warnung: {len(unexpected)} unerwartete Keys (z.B. {unexpected[:3]})")
    # Modelle mit geteilten Embeddings (tied weights) korrekt verbinden
    model.tie_weights()

    return model.to(device).eval(), tokenizer, device


def build_inputs(tokenizer, history, device):
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        return ids.to(device)

    text = ""
    for m in history:
        if m["role"] == "system":
            text += m["content"] + "\n"
        else:
            who = "User" if m["role"] == "user" else "Assistant"
            text += f"{who}: {m['content']}\n"
    text += "Assistant:"
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def main():
    model, tokenizer, device = load_model()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def fresh():
        return [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []

    history = fresh()
    print("\nBereit! /reset = Verlauf löschen, /exit = Beenden\n")

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
            history = fresh()
            print("Verlauf gelöscht.\n")
            continue

        history.append({"role": "user", "content": user})
        input_ids = build_inputs(tokenizer, history, device)

        print("Modell: ", end="", flush=True)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
