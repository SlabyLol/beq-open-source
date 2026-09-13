# Free / open text data people share for training

Use these to grow `data/input.txt` (or train offline and upload a checkpoint). Prefer plain text; keep files reasonable for free hosting.

## Best for small models (start here)

| Dataset | License / notes | Link |
|---------|-----------------|------|
| **TinyStories** | Synthetic kids-level stories; great for tiny LMs | https://huggingface.co/datasets/roneneldan/TinyStories |
| **TinyStories GPT-4 clean** | Cleaned plain stories | https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean |
| **Project Gutenberg** | Public domain books | https://www.gutenberg.org/ and HF `common-pile/project_gutenberg` |
| **Wikipedia dumps** | CC BY-SA — use extracts, not full dump on free tier | https://dumps.wikimedia.org/ |

## Larger open pretraining sets

| Dataset | Notes | Link |
|---------|-------|------|
| **Common Corpus** | Large public-domain / open-license multilingual | https://huggingface.co/blog/Pclanglais/common-corpus |
| **Common Pile** | Public domain + open licenses | search Hugging Face for common-pile |
| **DCLM-baseline** | Research web text (check terms) | https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 |
| **SmallCorpus** | Built for small LMs; Apache-2.0 | https://huggingface.co/datasets/SmallDoge/SmallCorpus |
| **llm-datasets** | Download scripts for many corpora | https://github.com/malteos/llm-datasets |

## How to use with Beq

1. Download a **small sample** (e.g. a few MB of TinyStories text), not multi-GB dumps, if you deploy on free Render.
2. Convert to plain UTF-8 text.
3. Append to `data/input.txt` or point Beq-Trainer at the new file.
4. Train (GitHub Actions / Beq-Trainer / Admin), then deploy the checkpoint.
5. Put exact FAQ answers in `configs/*.sbe` (online **SBE Builder** or Beq-Trainer) — those override the neural model.

## Legal

Always respect each dataset’s license and robots/terms. Public domain and clearly open-licensed sources are safest for open-source Beq.
