#!/usr/bin/env python3
"""
Semantische Suche mit modul.safetensors (BERT-Embedding-Modell, 384 Dimensionen).

Benötigt im selben Ordner:
    modul.safetensors, tokenizer.json
Optional:
    texte.txt  (ein Text pro Zeile, wird durchsucht)

Start:
    python search.py

Installation (einmalig):
    pip install safetensors torch transformers
"""
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

# ============================ EINSTELLUNGEN ============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modul.safetensors"
TOKENIZER_FILE = BASE_DIR / "tokenizer.json"
TEXTS_FILE = BASE_DIR / "texte.txt"

NUM_HEADS = 12       # bei hidden=384 üblich (MiniLM, bge-small); nicht aus der Datei lesbar
POOLING = "mean"     # "mean" (MiniLM) oder "cls" (bge)
TOP_K = 3

# Wird automatisch heruntergeladen, falls tokenizer.json kaputt ist oder fehlt
# (alle MiniLM-L6-Modelle nutzen dasselbe Vokabular mit 30522 Einträgen).
TOKENIZER_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
# =======================================================================

EXAMPLE_TEXTS = [
    "Die Katze schläft auf dem Sofa.",
    "Ein Hund läuft im Park.",
    "Python ist eine Programmiersprache.",
    "Das Wetter ist heute sonnig und warm.",
    "Ich habe Hunger und möchte Pizza essen.",
]


def build_config():
    with safe_open(str(MODEL_FILE), framework="pt") as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    vocab, hidden = shapes["embeddings.word_embeddings.weight"]
    max_pos = shapes["embeddings.position_embeddings.weight"][0]
    types = shapes["embeddings.token_type_embeddings.weight"][0]
    layers = 1 + max(
        int(m.group(1))
        for k in shapes
        if (m := re.match(r"encoder\.layer\.(\d+)\.", k))
    )
    inter = shapes["encoder.layer.0.intermediate.dense.weight"][0]
    print(f"Erkannt: BERT | {layers} Layer | hidden {hidden} | Vocab {vocab}")
    return BertConfig(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=NUM_HEADS, intermediate_size=inter,
        max_position_embeddings=max_pos, type_vocab_size=types,
    )


def load_tokenizer():
    try:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(TOKENIZER_FILE),
            unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
    except Exception as e:
        print(f"Warnung: tokenizer.json ist kaputt oder fehlt ({e}).")
        print(f"Lade stattdessen den passenden Tokenizer von {TOKENIZER_FALLBACK} ...")
        try:
            return AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)
        except Exception as e2:
            raise SystemExit(
                "Auch der Download hat nicht geklappt (Internetverbindung?):\n" + str(e2)
            )


def load():
    if not MODEL_FILE.exists():
        raise SystemExit(f"Datei nicht gefunden: {MODEL_FILE}")
    model = BertModel(build_config(), add_pooling_layer=False)
    state = load_file(str(MODEL_FILE))
    state = {k.removeprefix("bert."): v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warnung: fehlende Gewichte: {missing[:3]}")
    return model.eval(), load_tokenizer()


@torch.no_grad()
def embed(model, tokenizer, texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = model(**enc).last_hidden_state
    if POOLING == "cls":
        vec = out[:, 0]
    else:
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(vec, dim=-1)


def main():
    model, tokenizer = load()

    if TEXTS_FILE.exists():
        texts = [t.strip() for t in TEXTS_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]
    else:
        texts = EXAMPLE_TEXTS
        print(f"(Keine {TEXTS_FILE.name} gefunden – nutze Beispieltexte)")
    print(f"{len(texts)} Texte werden durchsucht ...")
    corpus = embed(model, tokenizer, texts)

    print("\nBereit! Tippe einen Satz, ich zeige die ähnlichsten Texte. /exit = Beenden\n")
    while True:
        try:
            query = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query == "/exit":
            break
        scores = (embed(model, tokenizer, [query]) @ corpus.T)[0]
        top = scores.topk(min(TOP_K, len(texts)))
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {score:.3f}  {texts[idx]}")
        print()


if __name__ == "__main__":
    main()
