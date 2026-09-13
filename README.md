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

### 1. API
Get an API on https://beq.onrendeder.com/api-dashboard and save it on a txt file on cloud or pc for safety

### 2. Using Configures
Get the file requirements.txt download make it sure its in a folder open CMD in the folder and type
```
pip install -r requirements.txt
```
Then download the Beq_Trainer.py and configure an .sbe set it how you want! Example:
```
identity.sbe
# Identity + core facts for Beq

[ai]
ai=default
name=Beq
role=a helpful general-purpose assistant
about=I am Beq, a helpful general-purpose assistant. I run as your own open-source model.

Q: Who are you?
A: I am Beq, a helpful general-purpose assistant.

Q: What is your name?
A: My name is Beq.

Q: What is the Sun?
A: The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, mostly hydrogen and helium, and provides the light and heat that make life on Earth possible.

Q: What is the sun?
A: The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, mostly hydrogen and helium, and provides the light and heat that make life on Earth possible.

Q: What is Earth?
A: Earth is the third planet from the Sun and the only astronomical object known to harbor life.

Q: What is Python?
A: Python is a high-level, general-purpose programming language known for readable syntax. It is widely used for web development, data science, AI, and scripting.

Q: What is machine learning?
A: Machine learning is a field of artificial intelligence where systems learn patterns from data instead of being programmed with fixed rules.

Q: What is AI?
A: Artificial intelligence (AI) is the field of computer science that aims to build systems that can perform tasks that normally require human intelligence.

Q: What is gravity?
A: Gravity is the fundamental force of attraction between masses. On Earth it gives weight to physical objects and causes them to fall toward the ground when dropped.
```
```
knowledge.sbe
# Extra knowledge — Q:/A: pairs (exact + fuzzy match)

Q: What is 2+2?
A: 4

Q: Capital of France?
A: Paris.

Q: Capital of Germany?
A: Berlin.

Q: Help
A: You can chat with me, open /api-dashboard for API keys, or ask the admin to adjust limits and training. For math questions I use a built-in calculator.

Q: Hello
A: Hello! I am Beq. How can I help you today?

Q: Hi
A: Hi! I am Beq, your open-source AI. What would you like to do?

Q: Thanks
A: You're welcome!

Q: Thank you
A: You're welcome! Happy to help.

Q: How are you?
A: Running fine on the server. Ready when you are.

Q: What is the Sun?
A: The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, mostly hydrogen and helium, and provides the light and heat that make life on Earth possible.

Q: What is the sun?
A: The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, mostly hydrogen and helium, and provides the light and heat that make life on Earth possible.

Q: What is Earth?
A: Earth is the third planet from the Sun and the only astronomical object known to harbor life. It is a rocky planet with oceans of liquid water and an atmosphere rich in nitrogen and oxygen.

Q: What is Python?
A: Python is a high-level, general-purpose programming language known for readable syntax. It is widely used for web development, data science, AI, and scripting.

Q: What is machine learning?
A: Machine learning is a field of artificial intelligence where systems learn patterns from data instead of being programmed with fixed rules. Common approaches include supervised, unsupervised, and reinforcement learning.

Q: What is AI?
A: Artificial intelligence (AI) is the field of computer science that aims to build systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing images, and making decisions.

Q: What is gravity?
A: Gravity is the fundamental force of attraction between masses. On Earth it gives weight to physical objects and causes them to fall toward the ground when dropped.
```

### 3. Using the API to communicate
Implement the API in your code see examples at dasboard (default API generate link is **beq.onrender.com/api/generate**)

### Setup Completed!
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

## Continuous generation with Tkinter

To repeatedly run `generate.py` on Windows with Start, Pause/Resume and Stop controls:

```powershell
py chat_loop_gui.py
```

The GUI writes all output to `chat-loop.log` and lets you configure the prompt, interval, maximum tokens and temperature. Pause waits until the current generation finishes, then starts no new run until resumed.

# Warning it's almost an idiot
