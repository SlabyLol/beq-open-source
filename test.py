#!/usr/bin/env python3
"""
Chat mit modul.safetensors – nur modul.safetensors + tokenizer.json nötig.

    python chat.py

Die fehlende config.json wird aus den Tensor-Formen der Datei erraten
(unterstützt: Llama-Stil, Qwen2-Stil, GPT-2-Stil).

Installation (einmalig):
    pip install safetensors torch transformers
"""
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    TextStreamer,
)

# ============================ EINSTELLUNGEN ============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modul.safetensors"
TOKENIZER_FILE = BASE_DIR / "tokenizer.json"

SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent. Antworte auf Deutsch."
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Werte, die sich NICHT sicher aus der Datei lesen lassen (None = automatisch raten).
# Nur ändern, wenn die Antworten Unsinn sind:
HEAD_DIM = None        # meist 64 oder 128
ROPE_THETA = None      # z.B. 10000, 500000, 1000000
MAX_POSITIONS = None   # z.B. 2048, 4096, 32768
CHAT_FORMAT = "auto"   # "auto", "chatml" oder "plain"
# =======================================================================

EOS_CANDIDATES = ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>", "<|eot_id|>", "<|end_of_text|>"]


def read_shapes(path):
    with safe_open(str(path), framework="pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def layer_count(keys, pattern):
    nums = [int(m.group(1)) for k in keys if (m := re.search(pattern, k))]
    return max(nums) + 1


def build_config(shapes):
    """Errät die Modell-Konfiguration aus den Tensor-Formen."""
    keys = list(shapes)

    # ---------- Llama- / Qwen2-Stil ----------
    suffix = "layers.0.self_attn.q_proj.weight"
    q_keys = [k for k in keys if k.endswith(suffix)]
    if q_keys:
        prefix = q_keys[0][: -len(suffix)]
        vocab, hidden = shapes[prefix + "embed_tokens.weight"]
        n_layers = layer_count(keys, r"layers\.(\d+)\.self_attn\.q_proj\.weight$")
        inter = shapes[prefix + "layers.0.mlp.gate_proj.weight"][0]
        q_out = shapes[prefix + "layers.0.self_attn.q_proj.weight"][0]
        kv_out = shapes[prefix + "layers.0.self_attn.k_proj.weight"][0]
        head_dim = HEAD_DIM or (128 if hidden >= 2048 and q_out % 128 == 0 else 64)
        n_heads = q_out // head_dim
        n_kv = kv_out // head_dim
        tied = "lm_head.weight" not in keys
        has_bias = (prefix + "layers.0.self_attn.q_proj.bias") in shapes

        cfg_cls = Qwen2Config if has_bias else LlamaConfig
        kind = "qwen2" if has_bias else "llama"
        config = cfg_cls(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            max_position_embeddings=MAX_POSITIONS or 4096,
            rope_theta=ROPE_THETA or (1000000.0 if has_bias else 10000.0),
            tie_word_embeddings=tied,
        )
        print(f"Erkannt: {kind} | {n_layers} Layer | hidden {hidden} | "
              f"Heads {n_heads}/{n_kv} | Vocab {vocab} | tied={tied}")
        return config, "model.", prefix

    # ---------- GPT-2-Stil ----------
    wte = [k for k in keys if k.endswith("wte.weight")]
    if wte:
        prefix = wte[0][: -len("wte.weight")]
        vocab, n_embd = shapes[wte[0]]
        n_pos = MAX_POSITIONS or shapes[prefix + "wpe.weight"][0]
        n_layer = layer_count(keys, r"h\.(\d+)\.attn\.c_attn\.weight$")
        n_inner = shapes[prefix + "h.0.mlp.c_fc.weight"][1]
        config = GPT2Config(
            vocab_size=vocab, n_embd=n_embd, n_layer=n_layer,
            n_head=n_embd // (HEAD_DIM or 64), n_positions=n_pos, n_inner=n_inner,
        )
        print(f"Erkannt: gpt2 | {n_layer} Layer | n_embd {n_embd} | Vocab {vocab}")
        return config, "transformer.", prefix

    print("Unbekannte Architektur. Die ersten Tensor-Namen:")
    for k in keys[:25]:
        print(f"  {k}  {shapes[k]}")
    raise SystemExit("Diese Architektur kann ich nicht automatisch erkennen. "
                     "Schick mir die Namen oben, dann passe ich das Skript an.")


def load_model():
    for f in (MODEL_FILE, TOKENIZER_FILE):
        if not f.exists():
            raise SystemExit(f"Datei nicht gefunden: {f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"Gerät: {device}")

    config, target, prefix = build_config(read_shapes(MODEL_FILE))
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    def rename(k):
        if k == "lm_head.weight":
            return k
        if prefix and k.startswith(prefix):
            k = k[len(prefix):]
        return target + k

    state = {rename(k): v for k, v in load_file(str(MODEL_FILE)).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing
               if not any(s in k for s in ("rotary_emb", "attn.bias", "masked_bias", "lm_head"))]
    if missing:
        print(f"Warnung: {len(missing)} fehlende Gewichte (z.B. {missing[:3]}) – "
              "Antworten könnten schlecht sein.")
    if unexpected:
        print(f"Warnung: {len(unexpected)} unerwartete Gewichte (z.B. {unexpected[:3]})")
    model.tie_weights()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_FILE))
    return model.to(device).eval(), tokenizer, device


def build_prompt(history, fmt):
    if fmt == "chatml":
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in history
        )
        return text + "<|im_start|>assistant\n"
    text = ""
    for m in history:
        if m["role"] == "system":
            text += m["content"] + "\n\n"
        else:
            who = "User" if m["role"] == "user" else "Assistant"
            text += f"{who}: {m['content']}\n"
    return text + "Assistant:"


def main():
    model, tokenizer, device = load_model()
    vocab = tokenizer.get_vocab()
    fmt = CHAT_FORMAT if CHAT_FORMAT != "auto" else ("chatml" if "<|im_start|>" in vocab else "plain")
    eos_ids = [vocab[t] for t in EOS_CANDIDATES if t in vocab]
    pad_id = eos_ids[0] if eos_ids else 0
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"Chat-Format: {fmt}")

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
        prompt = build_prompt(history, fmt)
        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=(fmt != "chatml")
        ).input_ids.to(device)

        gen_kwargs = {}
        if fmt == "plain":
            gen_kwargs = dict(stop_strings=["\nUser:"], tokenizer=tokenizer)

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
                eos_token_id=eos_ids or None,
                pad_token_id=pad_id,
                streamer=streamer,
                **gen_kwargs,
            )
        reply = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).split("\nUser:")[0].strip()
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
