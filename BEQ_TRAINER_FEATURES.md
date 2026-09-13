# Beq Trainer – feature list (v2)

Run on a PC/laptop (needs Python + Tkinter):

```bash
python Beq-Trainer-full.py
# or
python Beq-Trainer.py   # loads Beq-Trainer-full.py if present
```

## Tabs (full version)

| Tab | Features |
|-----|----------|
| **Train** | Run name, display name, data path, checkpoint folder, resume path, all hyperparams, 6 presets (Tiny → Stronger + Quick test + Long context), command preview, live log, progress bar, elapsed time, estimate time, validate, load/save JSON settings |
| **Config & name** | YAML load/save, new/duplicate config, raw YAML editor, sync name → identity.sbe, export settings pack |
| **Knowledge** | List/edit/delete `configs/*.sbe`, Q/A + alias templates, validate format, count entries |
| **Identity** | name / role / about → identity.sbe |
| **Data** | Edit `input.txt` / `input_extra.txt`, append file, size stats, save |
| **Checkpoints** | List all `.pt`, promote to `beq_best.pt`, delete, export zip, open folder |
| **Generate** | Local test generation from a checkpoint + tokenizer |
| **Tools** | Open folders, pip install requirements, check packages, validate setup, dark mode, open site/GitHub, training history |
| **Help & datasets** | Guide + links to TinyStories, Gutenberg, Common Corpus, SmallCorpus, Wikipedia dumps |

## Presets

- Tiny (free tier / fast)
- Default
- Balanced
- Stronger (more RAM)
- Long context
- Quick test (~1 min)

## Notes

- Training history is stored in `checkpoints/trainer_history.json`.
- Exact FAQ answers still belong in `.sbe` (or the online SBE Builder).
- Math is handled on the server by `mathtool`, not the tiny neural net.

If `Beq-Trainer-full.py` is missing from the repo, upload the full script from your build artifacts or ask Grok to push it again.
