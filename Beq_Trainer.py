#!/usr/bin/env python3
"""Beq Trainer – full desktop control panel for Beq.

Tabs: Train | Config | Knowledge | Identity | Data | Checkpoints | Generate | Tools | Help
Single file — real Tkinter code, no stubs, no parts, no payload.
"""

from __future__ import annotations

import json
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

DEFAULTS = {
    "d_model": "256",
    "n_layers": "6",
    "n_heads": "8",
    "block_size": "128",
    "batch_size": "32",
    "learning_rate": "0.0003",
    "max_steps": "5000",
    "eval_interval": "250",
    "save_interval": "1000",
}

PRESETS = {
    "Tiny (free tier / fast)": {
        "d_model": "128", "n_layers": "4", "n_heads": "4", "block_size": "64",
        "batch_size": "8", "learning_rate": "0.0003", "max_steps": "500",
        "eval_interval": "100", "save_interval": "250",
    },
    "Default": dict(DEFAULTS),
    "Balanced": {
        "d_model": "256", "n_layers": "6", "n_heads": "8", "block_size": "128",
        "batch_size": "32", "learning_rate": "0.0003", "max_steps": "5000",
        "eval_interval": "250", "save_interval": "1000",
    },
    "Stronger": {
        "d_model": "384", "n_layers": "8", "n_heads": "8", "block_size": "256",
        "batch_size": "16", "learning_rate": "0.0002", "max_steps": "10000",
        "eval_interval": "250", "save_interval": "1000",
    },
    "Quick test": {
        "d_model": "64", "n_layers": "2", "n_heads": "2", "block_size": "32",
        "batch_size": "4", "learning_rate": "0.001", "max_steps": "50",
        "eval_interval": "25", "save_interval": "50",
    },
}


class BeqTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Beq Trainer — full control panel")
        self.geometry("1200x880")
        self.minsize(960, 700)
        self.root_dir = Path(__file__).resolve().parent
        self.train_script = self.root_dir / "train" / "train.py"
        self.configs_dir = self.root_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root_dir / "checkpoints" / "trainer_history.json"

        self.run_name = tk.StringVar(value="beq-default")
        self.model_display_name = tk.StringVar(value="Beq")
        inp = self.data_dir / "input.txt"
        self.input_path = tk.StringVar(value=str(inp) if inp.exists() else "")
        self.output_dir = tk.StringVar(value=str(self.root_dir / "checkpoints"))
        self.python_executable = tk.StringVar(value=sys.executable)
        self.config_path = tk.StringVar(value=str(self.configs_dir / "default.yaml"))
        self.resume_path = tk.StringVar(value="")
        self.preset_name = tk.StringVar(value="Default")
        self.gen_prompt = tk.StringVar(value="Who are you?")
        self.gen_tokens = tk.StringVar(value="80")
        self.gen_temp = tk.StringVar(value="0.8")
        self.gen_ckpt = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="")
        self.elapsed_text = tk.StringVar(value="Elapsed: 00:00")
        self.current_action = tk.StringVar(value="")
        self.data_stats = tk.StringVar(value="")
        self.settings = {k: tk.StringVar(value=v) for k, v in DEFAULTS.items()}
        self.process = None
        self.output_queue: queue.Queue = queue.Queue()
        self.stop_requested = False
        self.start_time = None
        self._sbe_path = None
        self._data_file = tk.StringVar(value="input.txt")

        self._build_ui()
        self._refresh_command_preview()
        self._refresh_data_stats()
        self._refresh_checkpoints()
        self.after(200, self._poll_output)
        self.after(500, self._tick_elapsed)
        self._load_yaml_into_form(self.config_path.get())
        self._load_sbe_list()

    def _build_ui(self) -> None:
        try:
            style = ttk.Style(self)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
            style.configure("Sub.TLabel", font=("Segoe UI", 9))
        except tk.TclError:
            pass

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(header, text="Beq Trainer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Train · configs · knowledge · data · checkpoints · generate · tools",
            style="Sub.TLabel",
        ).grid(row=1, column=0, sticky="w")

        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew")

        tabs = {}
        for key, title in [
            ("train", "Train"),
            ("config", "Config"),
            ("knowledge", "Knowledge"),
            ("identity", "Identity"),
            ("data", "Data"),
            ("ckpt", "Checkpoints"),
            ("gen", "Generate"),
            ("tools", "Tools"),
            ("help", "Help"),
        ]:
            frame = ttk.Frame(notebook, padding=8)
            notebook.add(frame, text=f"  {title}  ")
            tabs[key] = frame

        self._build_train_tab(tabs["train"])
        self._build_config_tab(tabs["config"])
        self._build_knowledge_tab(tabs["knowledge"])
        self._build_identity_tab(tabs["identity"])
        self._build_data_tab(tabs["data"])
        self._build_ckpt_tab(tabs["ckpt"])
        self._build_gen_tab(tabs["gen"])
        self._build_tools_tab(tabs["tools"])
        self._build_help_tab(tabs["help"])

        status = ttk.Frame(root)
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(status, textvariable=self.status_text).pack(side="left")
        ttk.Label(status, textvariable=self.elapsed_text).pack(side="left", padx=12)
        ttk.Label(status, textvariable=self.progress_text).pack(side="left")
        ttk.Label(status, textvariable=self.current_action).pack(side="right")
        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.grid(row=3, column=0, sticky="ew", pady=(4, 0))

    # ── Train ───────────────────────────────────────────────────────────
    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)

        idf = ttk.LabelFrame(parent, text="Run identity", padding=8)
        idf.grid(row=0, column=0, sticky="ew")
        idf.columnconfigure(1, weight=1)
        idf.columnconfigure(3, weight=1)
        ttk.Label(idf, text="Run name").grid(row=0, column=0, sticky="w")
        ttk.Entry(idf, textvariable=self.run_name).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(idf, text="Display name").grid(row=0, column=2, sticky="w")
        ttk.Entry(idf, textvariable=self.model_display_name).grid(row=0, column=3, sticky="ew", padx=4)

        paths = ttk.LabelFrame(parent, text="Paths", padding=8)
        paths.grid(row=1, column=0, sticky="ew", pady=6)
        paths.columnconfigure(1, weight=1)
        self._add_path_row(paths, 0, "Training text", self.input_path, "Main corpus (input.txt).", self._browse_input)
        self._add_path_row(paths, 1, "Output dir", self.output_dir, "Checkpoints folder.", self._browse_output)
        self._add_path_row(paths, 2, "Resume .pt", self.resume_path, "Optional checkpoint to continue.", self._browse_resume)

        model_frame = ttk.LabelFrame(parent, text="Model", padding=8)
        model_frame.grid(row=2, column=0, sticky="ew")
        for i, key in enumerate(["d_model", "n_layers", "n_heads", "block_size"]):
            ttk.Label(model_frame, text=key).grid(row=0, column=i * 2, sticky="w", padx=2)
            ttk.Entry(model_frame, textvariable=self.settings[key], width=10).grid(row=0, column=i * 2 + 1, sticky="w")

        training_frame = ttk.LabelFrame(parent, text="Training", padding=8)
        training_frame.grid(row=3, column=0, sticky="ew", pady=6)
        for i, key in enumerate(["batch_size", "learning_rate", "max_steps", "eval_interval", "save_interval"]):
            r, c = divmod(i, 5)
            ttk.Label(training_frame, text=key).grid(row=r, column=c * 2, sticky="w", padx=2)
            ttk.Entry(training_frame, textvariable=self.settings[key], width=10).grid(row=r, column=c * 2 + 1, sticky="w")

        runtime = ttk.LabelFrame(parent, text="Runtime", padding=8)
        runtime.grid(row=4, column=0, sticky="ew")
        runtime.columnconfigure(1, weight=1)
        ttk.Label(runtime, text="Python").grid(row=0, column=0, sticky="w")
        ttk.Entry(runtime, textvariable=self.python_executable).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(runtime, text="Browse", command=self._browse_python).grid(row=0, column=2)
        ttk.Label(runtime, text="Preset").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(runtime, textvariable=self.preset_name, values=list(PRESETS.keys()), state="readonly").grid(
            row=1, column=1, sticky="ew", padx=4
        )
        ttk.Button(runtime, text="Apply preset", command=self._apply_preset).grid(row=1, column=2)

        bottom = ttk.Panedwindow(parent, orient="vertical")
        bottom.grid(row=5, column=0, sticky="nsew", pady=6)
        parent.rowconfigure(5, weight=1)

        cmd_frame = ttk.LabelFrame(bottom, text="Command preview", padding=4)
        self.cmd_preview = tk.Text(cmd_frame, height=3, font=("Consolas", 10), wrap="word")
        self.cmd_preview.pack(fill="both", expand=True)
        bottom.add(cmd_frame, weight=1)

        log_frame = ttk.LabelFrame(bottom, text="Training output", padding=6)
        self.log = ScrolledText(log_frame, height=12, font=("Consolas", 10), state="disabled")
        self.log.pack(fill="both", expand=True)
        bottom.add(log_frame, weight=3)

        actions = ttk.Frame(parent)
        actions.grid(row=6, column=0, sticky="ew", pady=4)
        ttk.Button(actions, text="Validate", command=self._validate).pack(side="left", padx=2)
        ttk.Button(actions, text="Clear log", command=self._clear_log).pack(side="left", padx=2)
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop_training, state="disabled")
        self.stop_button.pack(side="right", padx=2)
        self.start_button = ttk.Button(actions, text="Start training", command=self._start_training)
        self.start_button.pack(side="right")

    def _add_path_row(self, parent, row, label, var, tip, browse_cmd):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(parent, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(parent, text="…", width=3, command=browse_cmd).grid(row=row, column=2)

    # ── Config ──────────────────────────────────────────────────────────
    def _build_config_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top = ttk.LabelFrame(parent, text="YAML config", padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Entry(top, textvariable=self.config_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_config).grid(row=0, column=1, padx=4)
        btns = ttk.Frame(top)
        btns.grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(btns, text="Load YAML → form", command=lambda: self._load_yaml_into_form(self.config_path.get())).pack(side="left", padx=2)
        ttk.Button(btns, text="Save form → YAML", command=self._save_form_to_yaml).pack(side="left", padx=2)

        ed = ttk.LabelFrame(parent, text="YAML editor", padding=4)
        ed.grid(row=1, column=0, sticky="nsew", pady=6)
        ed.columnconfigure(0, weight=1)
        ed.rowconfigure(0, weight=1)
        self.yaml_editor = ScrolledText(ed, font=("Consolas", 10))
        self.yaml_editor.grid(row=0, column=0, sticky="nsew")

    # ── Knowledge ───────────────────────────────────────────────────────
    def _build_knowledge_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(parent, text="*.sbe files", padding=4)
        left.grid(row=0, column=0, sticky="nsw")
        self.sbe_list = tk.Listbox(left, height=18, width=28, exportselection=False)
        self.sbe_list.pack(fill="both", expand=True)
        self.sbe_list.bind("<<ListboxSelect>>", self._on_sbe_select)
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="Refresh", command=self._load_sbe_list).pack(side="left")
        ttk.Button(bf, text="New", command=self._new_sbe).pack(side="left", padx=2)

        right = ttk.LabelFrame(parent, text="Editor (exact Q:/A: knowledge)", padding=4)
        right.grid(row=0, column=1, sticky="nsew", padx=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.sbe_editor = ScrolledText(right, font=("Consolas", 10))
        self.sbe_editor.grid(row=0, column=0, sticky="nsew")
        ttk.Button(right, text="Save .sbe", command=self._save_sbe).grid(row=1, column=0, sticky="w", pady=4)

    # ── Identity ────────────────────────────────────────────────────────
    def _build_identity_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self.id_name = tk.StringVar(value="Beq")
        self.id_role = tk.StringVar(value="helpful open-source assistant")
        form = ttk.LabelFrame(parent, text="Identity fields → identity.sbe", padding=8)
        form.grid(row=0, column=0, columnspan=2, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.id_name).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(form, text="Role").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.id_role).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(form, text="About").grid(row=2, column=0, sticky="nw")
        self.id_about = ScrolledText(form, height=8, font=("Segoe UI", 10))
        self.id_about.grid(row=2, column=1, sticky="ew", padx=4)
        self.id_about.insert("1.0", "I am Beq, an open-source PyTorch AI. Everything runs locally. No external AI APIs.")
        ttk.Button(parent, text="Save identity.sbe", command=self._save_identity).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(
            parent,
            text="Tip: keep name short. Edit Knowledge so common questions never fall through to gibberish.",
            wraplength=700,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self._load_identity()

    # ── Data ────────────────────────────────────────────────────────────
    def _build_data_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, textvariable=self.data_stats).pack(side="left")
        for name in ("input.txt", "input_extra.txt", "input_stories.txt"):
            ttk.Button(top, text=name, command=lambda n=name: self._load_data_file(n)).pack(side="left", padx=2)
        ttk.Button(top, text="Save current", command=self._save_data_file).pack(side="left", padx=6)
        ttk.Button(top, text="Refresh stats", command=self._refresh_data_stats).pack(side="left")
        self.data_editor = ScrolledText(parent, font=("Consolas", 10))
        self.data_editor.grid(row=1, column=0, sticky="nsew", pady=6)
        ttk.Label(
            parent,
            text="Use Knowledge .sbe for exact facts. See Help tab for public datasets.",
        ).grid(row=2, column=0, sticky="w")

    # ── Checkpoints ─────────────────────────────────────────────────────
    def _build_ckpt_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Refresh", command=self._refresh_checkpoints).pack(side="left")
        ttk.Button(bar, text="Open folder", command=self._open_checkpoints_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="Promote selected → beq_best.pt", command=self._promote_checkpoint).pack(side="left", padx=4)
        self.ckpt_list = tk.Listbox(parent, font=("Consolas", 10))
        self.ckpt_list.grid(row=1, column=0, sticky="nsew", pady=6)

    # ── Generate ────────────────────────────────────────────────────────
    def _build_gen_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(5, weight=1)
        ttk.Label(parent, text="Checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gen_ckpt).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse", command=self._browse_gen_ckpt).grid(row=0, column=2)
        ttk.Label(parent, text="Prompt").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.gen_prompt).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(parent, text="Max tokens").grid(row=2, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gen_tokens, width=10).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(parent, text="Temperature").grid(row=3, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gen_temp, width=10).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Button(parent, text="Generate", command=self._run_generate).grid(row=4, column=1, sticky="w", pady=8)
        self.gen_output = ScrolledText(parent, font=("Consolas", 10))
        self.gen_output.grid(row=5, column=0, columnspan=3, sticky="nsew")

    # ── Tools ───────────────────────────────────────────────────────────
    def _build_tools_tab(self, parent: ttk.Frame) -> None:
        for label, cmd in [
            ("Open project folder", lambda: self._open_path(self.root_dir)),
            ("Open data/", lambda: self._open_path(self.data_dir)),
            ("Open configs/", lambda: self._open_path(self.configs_dir)),
            ("Open checkpoints/", lambda: self._open_path(self.root_dir / "checkpoints")),
            ("pip install -r requirements.txt", self._run_pip_install),
            ("Open https://beq.onrender.com", lambda: webbrowser.open("https://beq.onrender.com")),
            ("Open GitHub repo", lambda: webbrowser.open("https://github.com/SlabyLol/beq-open-source")),
            ("Open admin crawler (web)", lambda: webbrowser.open("https://beq.onrender.com/admin")),
        ]:
            ttk.Button(parent, text=label, command=cmd).pack(anchor="w", pady=3, fill="x")

    # ── Help ────────────────────────────────────────────────────────────
    def _build_help_tab(self, parent: ttk.Frame) -> None:
        t = ScrolledText(parent, wrap="word", font=("Segoe UI", 10))
        t.pack(fill="both", expand=True)
        t.insert(
            "1.0",
            "Beq Trainer — single complete file (9 tabs)\n\n"
            "1. Train tab: set model size, steps, data path → Start training.\n"
            "2. Config tab: edit / save YAML presets.\n"
            "3. Knowledge tab: edit .sbe FAQ files — Beq returns A: text EXACTLY.\n"
            "4. Identity tab: name/role/about for identity.sbe.\n"
            "5. Data tab: edit input.txt / input_extra.txt / input_stories.txt.\n"
            "6. Checkpoints: list, promote best.\n"
            "7. Generate: quick local sample from a .pt checkpoint.\n"
            "8. Tools: open folders, install deps, open website.\n"
            "9. Help: this page.\n\n"
            "Tips\n"
            "• Train on clean User:/Beq: dialogues + TinyStories samples.\n"
            "• SBE is optional; without it the model still works.\n"
            "• Crawler / search lives on the website admin panel (not here).\n"
            "• Keep model small (Tiny / Default) if you train on limited hardware.\n",
        )
        t.configure(state="disabled")

    # ── Browse helpers ──────────────────────────────────────────────────
    def _browse_input(self) -> None:
        s = filedialog.askopenfilename(
            title="Training text",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
            initialdir=str(self.data_dir),
        )
        if s:
            self.input_path.set(s)
            self._refresh_command_preview()

    def _browse_output(self) -> None:
        s = filedialog.askdirectory(title="Output dir", initialdir=str(self.root_dir / "checkpoints"))
        if s:
            self.output_dir.set(s)
            self._refresh_command_preview()

    def _browse_resume(self) -> None:
        s = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if s:
            self.resume_path.set(s)
            self._refresh_command_preview()

    def _browse_python(self) -> None:
        s = filedialog.askopenfilename(title="Python executable")
        if s:
            self.python_executable.set(s)
            self._refresh_command_preview()

    def _browse_config(self) -> None:
        s = filedialog.askopenfilename(
            initialdir=str(self.configs_dir),
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if s:
            self.config_path.set(s)
            self._load_yaml_into_form(s)

    def _browse_gen_ckpt(self) -> None:
        s = filedialog.askopenfilename(
            filetypes=[("PyTorch", "*.pt")],
            initialdir=str(self.root_dir / "checkpoints"),
        )
        if s:
            self.gen_ckpt.set(s)

    def _apply_preset(self) -> None:
        preset = PRESETS.get(self.preset_name.get()) or {}
        for k, v in preset.items():
            if k in self.settings:
                self.settings[k].set(str(v))
        self._refresh_command_preview()

    def _resolved_out_dir(self) -> Path:
        base = Path(self.output_dir.get() or "checkpoints")
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.run_name.get().strip()) or "beq"
        if name in ("beq", "beq-default"):
            return base
        return base / name

    def _build_command(self) -> list[str]:
        cmd = [
            self.python_executable.get().strip() or sys.executable,
            str(self.train_script),
            "--data", self.input_path.get().strip(),
            "--out_dir", str(self._resolved_out_dir()),
            "--d_model", self.settings["d_model"].get(),
            "--n_layers", self.settings["n_layers"].get(),
            "--n_heads", self.settings["n_heads"].get(),
            "--block_size", self.settings["block_size"].get(),
            "--batch_size", self.settings["batch_size"].get(),
            "--lr", self.settings["learning_rate"].get(),
            "--max_steps", self.settings["max_steps"].get(),
            "--eval_interval", self.settings["eval_interval"].get(),
            "--save_interval", self.settings["save_interval"].get(),
        ]
        resume = self.resume_path.get().strip()
        if resume:
            cmd += ["--resume", resume]
        return cmd

    def _refresh_command_preview(self) -> None:
        try:
            text = " ".join(shlex.quote(x) for x in self._build_command())
        except Exception as exc:
            text = f"# error: {exc}"
        self.cmd_preview.delete("1.0", "end")
        self.cmd_preview.insert("1.0", text)

    def _load_yaml_into_form(self, path_str: str) -> None:
        path = Path(path_str)
        if not path.exists():
            return
        raw = path.read_text(encoding="utf-8")
        self.yaml_editor.delete("1.0", "end")
        self.yaml_editor.insert("1.0", raw)
        if yaml is None:
            return
        try:
            data = yaml.safe_load(raw) or {}
        except Exception:
            return
        if data.get("name"):
            self.run_name.set(str(data["name"]))
        if data.get("display_name"):
            self.model_display_name.set(str(data["display_name"]))
        model = data.get("model") or {}
        train = data.get("training") or {}
        mapping = {
            "d_model": model, "n_layers": model, "n_heads": model, "block_size": model,
            "batch_size": train, "learning_rate": train, "max_steps": train,
            "eval_interval": train, "save_interval": train,
        }
        for key, src in mapping.items():
            if key in src and key in self.settings:
                self.settings[key].set(str(src[key]))
        self._refresh_command_preview()

    def _save_form_to_yaml(self) -> None:
        path = Path(self.config_path.get())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.run_name.get().strip() or "beq",
            "display_name": self.model_display_name.get().strip() or "Beq",
            "model": {
                "d_model": int(self.settings["d_model"].get() or 256),
                "n_layers": int(self.settings["n_layers"].get() or 6),
                "n_heads": int(self.settings["n_heads"].get() or 8),
                "block_size": int(self.settings["block_size"].get() or 128),
            },
            "training": {
                "batch_size": int(self.settings["batch_size"].get() or 32),
                "learning_rate": float(self.settings["learning_rate"].get() or 3e-4),
                "max_steps": int(self.settings["max_steps"].get() or 5000),
                "eval_interval": int(self.settings["eval_interval"].get() or 250),
                "save_interval": int(self.settings["save_interval"].get() or 1000),
            },
        }
        if yaml is not None:
            text = yaml.safe_dump(data, sort_keys=False)
        else:
            text = json.dumps(data, indent=2)
        path.write_text(text, encoding="utf-8")
        self.yaml_editor.delete("1.0", "end")
        self.yaml_editor.insert("1.0", text)
        messagebox.showinfo("Config", f"Saved:\n{path}")

    # Knowledge
    def _load_sbe_list(self) -> None:
        self.sbe_list.delete(0, "end")
        for p in sorted(self.configs_dir.glob("*.sbe")):
            self.sbe_list.insert("end", p.name)

    def _on_sbe_select(self, _event=None) -> None:
        sel = self.sbe_list.curselection()
        if not sel:
            return
        name = self.sbe_list.get(sel[0])
        path = self.configs_dir / name
        self._sbe_path = path
        self.sbe_editor.delete("1.0", "end")
        if path.exists():
            self.sbe_editor.insert("1.0", path.read_text(encoding="utf-8"))

    def _new_sbe(self) -> None:
        name = simpledialog.askstring("New .sbe", "Filename (e.g. facts.sbe):")
        if not name:
            return
        if not name.endswith(".sbe"):
            name += ".sbe"
        path = self.configs_dir / name
        if not path.exists():
            path.write_text("# Q: question\n# A: exact answer\n\nQ: Who are you?\nA: I am Beq.\n", encoding="utf-8")
        self._load_sbe_list()
        self._sbe_path = path
        self.sbe_editor.delete("1.0", "end")
        self.sbe_editor.insert("1.0", path.read_text(encoding="utf-8"))

    def _save_sbe(self) -> None:
        if self._sbe_path is None:
            messagebox.showerror("Knowledge", "Select or create a .sbe file first.")
            return
        self._sbe_path.write_text(self.sbe_editor.get("1.0", "end-1c"), encoding="utf-8")
        messagebox.showinfo("Knowledge", f"Saved:\n{self._sbe_path}")

    # Identity
    def _load_identity(self) -> None:
        path = self.configs_dir / "identity.sbe"
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        # simple parse
        for line in text.splitlines():
            low = line.strip().lower()
            if low.startswith("name:"):
                self.id_name.set(line.split(":", 1)[1].strip())
            elif low.startswith("role:"):
                self.id_role.set(line.split(":", 1)[1].strip())
            elif low.startswith("about:"):
                self.id_about.delete("1.0", "end")
                self.id_about.insert("1.0", line.split(":", 1)[1].strip())

    def _save_identity(self) -> None:
        path = self.configs_dir / "identity.sbe"
        about = self.id_about.get("1.0", "end-1c").strip()
        text = (
            f"[ai] ai=default name={self.id_name.get().strip() or 'Beq'}\n"
            f"name: {self.id_name.get().strip() or 'Beq'}\n"
            f"role: {self.id_role.get().strip() or 'assistant'}\n"
            f"about: {about}\n"
        )
        path.write_text(text, encoding="utf-8")
        messagebox.showinfo("Identity", f"Saved:\n{path}")

    # Data
    def _load_data_file(self, name: str) -> None:
        path = self.data_dir / name
        self._data_file.set(name)
        self.data_editor.delete("1.0", "end")
        if path.exists():
            self.data_editor.insert("1.0", path.read_text(encoding="utf-8"))
        self._refresh_data_stats()

    def _save_data_file(self) -> None:
        name = self._data_file.get() or "input.txt"
        path = self.data_dir / name
        path.write_text(self.data_editor.get("1.0", "end-1c"), encoding="utf-8")
        self._refresh_data_stats()
        messagebox.showinfo("Data", f"Saved:\n{path}")

    def _refresh_data_stats(self) -> None:
        parts = []
        for name in ("input.txt", "input_extra.txt", "input_stories.txt"):
            p = self.data_dir / name
            if p.exists():
                n = len(p.read_text(encoding="utf-8"))
                parts.append(f"{name}: {n:,} chars")
            else:
                parts.append(f"{name}: missing")
        self.data_stats.set("  |  ".join(parts))

    # Checkpoints
    def _refresh_checkpoints(self) -> None:
        self.ckpt_list.delete(0, "end")
        ckpt_dir = self.root_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        files = sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:80]:
            rel = p.relative_to(self.root_dir)
            size_mb = p.stat().st_size / (1024 * 1024)
            self.ckpt_list.insert("end", f"{rel}  ({size_mb:.2f} MB)")

    def _open_checkpoints_folder(self) -> None:
        self._open_path(self.root_dir / "checkpoints")

    def _promote_checkpoint(self) -> None:
        sel = self.ckpt_list.curselection()
        if not sel:
            messagebox.showwarning("Checkpoints", "Select a checkpoint first.")
            return
        line = self.ckpt_list.get(sel[0])
        rel = line.split("  (")[0].strip()
        src = self.root_dir / rel
        if not src.exists():
            messagebox.showerror("Checkpoints", f"Not found: {src}")
            return
        dst = self.root_dir / "checkpoints" / "beq_best.pt"
        shutil.copy2(src, dst)
        messagebox.showinfo("Checkpoints", f"Promoted → {dst}")
        self._refresh_checkpoints()

    # Generate
    def _run_generate(self) -> None:
        ckpt = self.gen_ckpt.get().strip()
        if not ckpt or not Path(ckpt).exists():
            messagebox.showerror("Generate", "Select a valid .pt checkpoint.")
            return
        gen_script = self.root_dir / "generate.py"
        if not gen_script.exists():
            messagebox.showerror("Generate", "generate.py missing.")
            return
        cmd = [
            self.python_executable.get().strip() or sys.executable,
            str(gen_script),
            "--checkpoint", ckpt,
            "--prompt", self.gen_prompt.get(),
            "--max_tokens", self.gen_tokens.get(),
            "--temperature", self.gen_temp.get(),
        ]
        self.gen_output.delete("1.0", "end")
        self.gen_output.insert("end", "Running…\n")

        def worker():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                out = (proc.stdout or "") + (proc.stderr or "")
            except Exception as exc:
                out = f"Error: {exc}"
            self.output_queue.put(("gen", out))

        threading.Thread(target=worker, daemon=True).start()

    # Tools
    def _open_path(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open", str(exc))

    def _run_pip_install(self) -> None:
        req = self.root_dir / "requirements.txt"
        if not req.exists():
            messagebox.showerror("pip", "requirements.txt missing")
            return
        cmd = [self.python_executable.get().strip() or sys.executable, "-m", "pip", "install", "-r", str(req)]
        self._append_log("$ " + " ".join(shlex.quote(c) for c in cmd))

        def worker():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                self.output_queue.put(("log", (proc.stdout or "") + (proc.stderr or "")))
            except Exception as exc:
                self.output_queue.put(("log", f"pip error: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    # Training control
    def _validate(self) -> None:
        errors = []
        if not self.train_script.exists():
            errors.append(f"Missing train script: {self.train_script}")
        data = Path(self.input_path.get())
        if not data.exists():
            errors.append(f"Data file not found: {data}")
        for key in self.settings:
            try:
                float(self.settings[key].get())
            except ValueError:
                errors.append(f"Invalid number for {key}")
        if errors:
            messagebox.showerror("Validate", "\n".join(errors))
        else:
            messagebox.showinfo("Validate", "OK — ready to train.")

    def _start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Training", "Already running.")
            return
        self._validate()
        cmd = self._build_command()
        self._append_log("$ " + " ".join(shlex.quote(c) for c in cmd))
        self.stop_requested = False
        self.start_time = time.time()
        self.status_text.set("Training…")
        self._set_running_state(True)
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.root_dir),
            )
        except Exception as exc:
            self._append_log(f"Failed to start: {exc}")
            self._set_running_state(False)
            self.status_text.set("Failed")
            return

        def reader():
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(("log", line.rstrip("\n")))
            code = self.process.wait()
            self.output_queue.put(("done", code))

        threading.Thread(target=reader, daemon=True).start()

    def _stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stop_requested = True
        try:
            self.process.terminate()
        except OSError:
            pass
        self._append_log("Stop requested.")

    def _poll_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    # crude progress from "step 123"
                    m = re.search(r"step\s+(\d+)", str(payload), re.I)
                    if m:
                        try:
                            step = int(m.group(1))
                            max_steps = int(self.settings["max_steps"].get() or 1)
                            self.progress["value"] = min(100, 100.0 * step / max(1, max_steps))
                            self.progress_text.set(f"step {step}/{max_steps}")
                            self.current_action.set("Training")
                        except Exception:
                            pass
                elif kind == "gen":
                    self.gen_output.delete("1.0", "end")
                    self.gen_output.insert("1.0", str(payload))
                elif kind == "done":
                    code = payload
                    self._set_running_state(False)
                    if code == 0 and not self.stop_requested:
                        self.status_text.set("Done")
                        self._append_log("Training completed successfully.")
                    else:
                        self.status_text.set(f"Stopped (code {code})")
                    self._refresh_checkpoints()
        except queue.Empty:
            pass
        self.after(200, self._poll_output)

    def _tick_elapsed(self) -> None:
        if self.start_time is not None and self.process is not None and self.process.poll() is None:
            elapsed = int(time.time() - self.start_time)
            mm, ss = divmod(elapsed, 60)
            self.elapsed_text.set(f"Elapsed: {mm:02d}:{ss:02d}")
        self.after(500, self._tick_elapsed)

    def _set_running_state(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Training running", "Stop and close?"):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        self.destroy()


if __name__ == "__main__":
    app = BeqTrainerApp()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()
