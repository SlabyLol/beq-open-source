# Beq – Open Source Language Model

**Beq** is a small but real GPT-style language model written entirely in pure PyTorch.  
No external AI APIs. No closed weights. Everything is yours.

- Custom Transformer architecture (from scratch)
- Character-level tokenizer
- Training script included
- Clean web interface with real HTML pages
- Fully open source

---

## Features

- **Pure PyTorch** – the model is implemented by hand (attention, feed-forward, residual connections, etc.)
- **No external AI services** – runs completely locally
- **Web UI** – chat page + generate page + about page
- **Easy to train** – just put your text in `data/input.txt` and run the training script
- **MIT License** – free to use, modify and distribute

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Beq

```bash
# Uses a small sample text if data/input.txt does not exist
python train/train.py
```

Useful options:

```bash
python train/train.py \
  --data data/input.txt \
  --d_model 256 \
  --n_layers 6 \
  --n_heads 8 \
  --block_size 128 \
  --batch_size 32 \
  --max_steps 5000 \
  --out_dir checkpoints
```

After training you will have:
- `checkpoints/beq_best.pt`
- `checkpoints/tokenizer.json`

### 3. Start the web interface

```bash
python web/app.py
```

Then open: [http://localhost:8000](http://localhost:8000)

Pages:
- `/` – Chat with Beq
- `/generate` – API-style generation
- `/about` – Project information

---

## Project Structure

```
beq-open-source/
├── model/
│   ├── transformer.py      # Beq Transformer (pure PyTorch)
│   └── tokenizer.py        # Character tokenizer
├── train/
│   └── train.py            # Training loop
├── web/
│   ├── app.py              # FastAPI backend
│   ├── templates/          # HTML pages
│   └── static/             # CSS
├── data/                   # Put your training text here
├── checkpoints/            # Saved models (created after training)
├── requirements.txt
└── README.md
```

---

## Model Architecture

Beq is a classic decoder-only Transformer:

- Token + positional embeddings
- Multi-head causal self-attention
- Feed-forward network with GELU
- LayerNorm + residual connections
- Weight tying between embedding and output head

Default size (~10–15M parameters depending on vocab):
- `d_model = 256`
- `n_layers = 6`
- `n_heads = 8`
- `block_size = 128`

You can scale it up by changing the arguments.

---

## Using your own data

1. Create a plain text file: `data/input.txt`
2. Put as much text as you want (books, conversations, code, your own writing…)
3. Run training again

The more high-quality text you give it, the better Beq becomes.

---

## API Endpoint

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100, "temperature": 0.8}'
```

---

## License

MIT License – do whatever you want with Beq.

---

**Beq is yours.** Train it, improve it, share it.
