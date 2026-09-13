"""Beq web entrypoint - loads full app implementation."""
from pathlib import Path
import runpy
_impl = Path(__file__).resolve().parent / "app_impl.py"
if not _impl.exists():
    raise RuntimeError("web/app_impl.py missing")
# Export app for uvicorn web.app:app
ns = runpy.run_path(str(_impl), run_name="web.app_impl")
app = ns["app"]
