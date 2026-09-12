"""
Beq Web Interface
Simple FastAPI backend + HTML pages.
Runs your own model locally. No external AI APIs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import BeqTransformer, CharTokenizer

# ---------- Config ----------
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "beq_best.pt"
TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "tokenizer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Beq", description="Your own open-source language model")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global model & tokenizer
model = None
tokenizer = None


def quick_train():
    """Train a tiny model automatically so Render works out of the box."""
    print("No checkpoint found – starting quick training (approx. 1–2 min)...")
    import subprocess

    cmd = [
        sys.executable, "train/train.py",
        "--data", "data/input.txt",
        "--out_dir", "checkpoints",
        "--d_model", "128",
        "--n_layers", "4",
        "--n_heads", "4",
        "--block_size", "64",
        "--batch_size", "8",
        "--max_steps", "300",
        "--eval_interval", "150",
        "--save_interval", "300",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Quick training failed")
    print("Quick training finished.")


def load_model():
    global model, tokenizer

    if not CHECKPOINT_PATH.exists():
        try:
            quick_train()
        except Exception as e:
            print(f"Auto-training failed: {e}")
            return False

    if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
        print("Still no checkpoint after training attempt.")
        return False

    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]

    model = BeqTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Beq loaded on {DEVICE} | params: {sum(p.numel() for p in model.parameters()):,}")
    return True


@app.on_event("startup")
async def startup():
    load_model()


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_loaded": model is not None,
        },
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    return templates.TemplateResponse(
        "generate.html",
        {"request": request, "model_loaded": model is not None},
    )


# ---------- API ----------

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 120
    temperature: float = 0.8
    top_k: int = 40


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        return {"error": "Model not loaded. Train it first with: python train/train.py"}

    prompt = req.prompt.strip()
    if not prompt:
        return {"error": "Prompt cannot be empty"}

    ids = tokenizer.encode(prompt)
    if not ids:
        return {"error": "Could not encode prompt"}

    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=min(req.max_tokens, 200),
            temperature=max(0.1, min(req.temperature, 2.0)),
            top_k=req.top_k if req.top_k > 0 else None,
        )

    generated = tokenizer.decode(out[0].tolist())
    return {
        "prompt": prompt,
        "generated": generated,
        "new_text": generated[len(prompt) :],
    }


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    prompt: str = Form(...),
    max_tokens: int = Form(100),
    temperature: float = Form(0.8),
):
    result = None
    error = None

    if model is None:
        error = "Model not loaded. Please train Beq first."
    else:
        ids = tokenizer.encode(prompt)
        if not ids:
            error = "Could not encode the prompt."
        else:
            idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                out = model.generate(
                    idx,
                    max_new_tokens=min(max_tokens, 200),
                    temperature=max(0.1, min(temperature, 2.0)),
                    top_k=40,
                )
            generated = tokenizer.decode(out[0].tolist())
            result = {
                "prompt": prompt,
                "full": generated,
                "completion": generated[len(prompt) :],
            }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_loaded": model is not None,
            "result": result,
            "error": error,
            "prompt": prompt,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
