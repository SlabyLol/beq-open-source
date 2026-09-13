# beq-client

Python package / plugin to call your **Beq** API.

## Install

```bash
cd python && pip install -e .
```

## Quick start

```python
from beq_client import Beq

beq = Beq(
    base_url="https://beq.onrender.com",
    api_key="beq_YOUR_KEY",   # from /api-dashboard
    language="de",
)

print(beq.generate("Who are you?"))
print(beq.chat("Hello", language="en"))
print(beq.status())
print(Beq.languages())
```

## Languages

en, de, fr, es, it, pt, nl, pl, ru, ja, zh, ko, tr, ar — full list via `Beq.languages()`.

```python
beq.set_language("fr")
beq.generate("Bonjour")
```

## CLI

```bash
export BEQ_URL=https://beq.onrender.com
export BEQ_API_KEY=beq_...
beq languages
beq status
beq generate -l de "Hello Beq"
```

## Env

| Variable | Meaning |
|----------|---------|
| `BEQ_URL` | Base URL |
| `BEQ_API_KEY` | API key from dashboard |
| `BEQ_LANG` | Default language |
