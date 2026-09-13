#!/usr/bin/env python3
"""Beq Trainer – Tkinter UI for training Beq and managing configs.

Tabs: Train | Config & name | Knowledge (.sbe) | Identity
"""

from __future__ import annotations

import json
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
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
    "Tiny (fast / free tier)": {
        "d_model": "128", "n_layers": "4", "n_heads": "4", "block_size": "64",
        "batch_size": "8", "learning_rate": "0.0003", "max_steps": "500",
        "eval_interval": "100", "save_interval": "250",
    },
    "Default": dict(DEFAULTS),
    "Stronger (more RAM)": {
        "d_model": "384", "n_layers": "8", "n_heads": "8", "block_size": "256",
        "batch_size": "16", "learning_rate": "0.0002", "max_steps": "10000",
        "eval_interval": "250", "save_interval": "1000",
    },
}


class BeqTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Beq Trainer")
        self.geometry("1180x900")
        self.minsize(960, 720)
        self.project_root = Path(__file__).resolve().parent
        self.train_script = self.project_root / "train" / "train.py"
        self.configs_dir = self.project_root / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        default_input = self.project_root / "data" / "input.txt"
        default_output = self.project_root / "checkpoints"
        self.run_name = tk.StringVar(value="beq-default")
        self.model_display_name = tk.StringVar(value="Beq")
        self.input_path = tk.StringVar(value=str(default_input) if default_input.exists() else "")
        self.output_dir = tk.StringVar(value=str(default_output))
        self.python_executable = tk.StringVar(value=sys.executable)
        self.config_path = tk.StringVar(value=str(self.configs_dir / "default.yaml"))
        self.resume_path = tk.StringVar(value="")
        self.preset_name = tk.StringVar(value="Default")
        self.status_text = tk.StringVar(value="Ready")
        self.current_action = tk.StringVar(value="Idle")
        self.progress_text = tk.StringVar(value="No training run is active")
        self.elapsed_text = tk.StringVar(value="Elapsed: 00:00")
        self.settings = {k: tk.StringVar(value=v) for k, v in DEFAULTS.items()}
        for variable in list(self.settings.values()) + [
            self.input_path, self.output_dir, self.python_executable,
            self.run_name, self.model_display_name, self.resume_path,
        ]:
            variable.trace_add("write", self._settings_changed)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_requested = False
        self.current_step = 0
        self.current_loss = "n/a"
        self.start_time: float | None = None
        self.elapsed_seconds = 0.0
        self._sbe_path: Path | None = None
        self._build_ui()
        self._refresh_command_preview()
        self.after(200, self._poll_queue)
        self.after(500, self._tick_elapsed)
        self._load_yaml_if_exists(self.config_path.get())
        self._load_sbe_list()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#555")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="Beq Trainer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Train Beq, manage YAML configs, knowledge (.sbe), and identity.", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w")
        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew")
        tab_train = ttk.Frame(notebook, padding=8)
        tab_config = ttk.Frame(notebook, padding=8)
        tab_knowledge = ttk.Frame(notebook, padding=8)
        tab_identity = ttk.Frame(notebook, padding=8)
        notebook.add(tab_train, text="  Train  ")
        notebook.add(tab_config, text="  Config & name  ")
        notebook.add(tab_knowledge, text="  Knowledge (.sbe)  ")
        notebook.add(tab_identity, text="  Identity  ")
        self._build_train_tab(tab_train)
        self._build_config_tab(tab_config)
        self._build_knowledge_tab(tab_knowledge)
        self._build_identity_tab(tab_identity)
        status_bar = ttk.Frame(root)
        status_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        status_bar.columnconfigure(0, weight=1)
        status_bar.columnconfigure(1, weight=2)
        status_bar.columnconfigure(2, weight=1)
        status_bar.columnconfigure(3, weight=2)
        ttk.Label(status_bar, textvariable=self.status_text, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_bar, textvariable=self.current_action).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(status_bar, textvariable=self.elapsed_text).grid(row=0, column=2, sticky="w")
        ttk.Label(status_bar, textvariable=self.progress_text).grid(row=0, column=3, sticky="w")
        self.progress = ttk.Progressbar(status_bar, mode="determinate", maximum=1)
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)
        identity = ttk.LabelFrame(parent, text="Run identity", padding=10)
        identity.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        identity.columnconfigure(1, weight=1)
        identity.columnconfigure(3, weight=1)
        ttk.Label(identity, text="Run name").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(identity, textvariable=self.run_name).grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(identity, text="Model display name").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Entry(identity, textvariable=self.model_display_name).grid(row=0, column=3, sticky="ew")
        ttk.Label(identity, text="Run name → checkpoint subfolder. Display name → identity.sbe when you sync.", style="Subtitle.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        paths = ttk.LabelFrame(parent, text="Training files", padding=10)
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        paths.columnconfigure(1, weight=1)
        self._add_path_row(paths, 0, "Training text file", self.input_path, "Plain-text corpus (e.g. data/input.txt).", self._browse_input)
        self._add_path_row(paths, 1, "Checkpoint folder", self.output_dir, "Checkpoints + tokenizer.json.", self._browse_output)
        self._add_path_row(paths, 2, "Resume checkpoint (optional)", self.resume_path, "beq_best.pt to continue (copy over if train.py has no --resume).", self._browse_resume)
        model_frame = ttk.LabelFrame(parent, text="Model configuration", padding=10)
        model_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for c in range(4):
            model_frame.columnconfigure(c, weight=1)
        self._add_setting(model_frame, 0, 0, "Embedding size (d_model)", "d_model")
        self._add_setting(model_frame, 0, 2, "Transformer layers", "n_layers")
        self._add_setting(model_frame, 1, 0, "Attention heads", "n_heads")
        self._add_setting(model_frame, 1, 2, "Context length (block size)", "block_size")
        training_frame = ttk.LabelFrame(parent, text="Training configuration", padding=10)
        training_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        for c in range(4):
            training_frame.columnconfigure(c, weight=1)
        self._add_setting(training_frame, 0, 0, "Batch size", "batch_size")
        self._add_setting(training_frame, 0, 2, "Learning rate", "learning_rate")
        self._add_setting(training_frame, 1, 0, "Maximum steps", "max_steps")
        self._add_setting(training_frame, 1, 2, "Evaluation interval", "eval_interval")
        self._add_setting(training_frame, 2, 0, "Checkpoint interval", "save_interval")
        runtime = ttk.LabelFrame(parent, text="Runtime & presets", padding=10)
        runtime.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        runtime.columnconfigure(1, weight=1)
        ttk.Label(runtime, text="Python executable").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(runtime, textvariable=self.python_executable).grid(row=0, column=1, sticky="ew")
        ttk.Button(runtime, text="Browse...", command=self._browse_python).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(runtime, text="Preset").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        preset = ttk.Combobox(runtime, textvariable=self.preset_name, values=list(PRESETS.keys()), state="readonly")
        preset.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        preset.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        ttk.Button(runtime, text="Apply preset", command=self._apply_preset).grid(row=1, column=2, padx=(6, 0), pady=(6, 0))
        bottom = ttk.PanedWindow(parent, orient="vertical")
        bottom.grid(row=5, column=0, sticky="nsew")
        preview_frame = ttk.LabelFrame(bottom, text="Command preview", padding=8)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.command_preview = tk.Text(preview_frame, height=4, wrap="word", font=("Consolas", 10))
        self.command_preview.grid(row=0, column=0, sticky="nsew")
        self.command_preview.configure(state="disabled")
        bottom.add(preview_frame, weight=1)
        log_frame = ttk.LabelFrame(bottom, text="Training output", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = ScrolledText(log_frame, height=12, font=("Consolas", 10), state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        bottom.add(log_frame, weight=3)
        actions = ttk.Frame(parent)
        actions.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Load settings JSON", command=self._load_settings).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Save settings JSON", command=self._save_settings).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Reset defaults", command=self._reset_defaults).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Clear log", command=self._clear_log).pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop_training, state="disabled")
        self.stop_button.pack(side="right", padx=(6, 0))
        self.start_button = ttk.Button(actions, text="Start training", command=self._start_training)
        self.start_button.pack(side="right")

    def _build_config_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        top = ttk.LabelFrame(parent, text="YAML config file", padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Config path").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(top, textvariable=self.config_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse...", command=self._browse_config).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(top, text="Load YAML → form", command=self._load_yaml_from_path).grid(row=1, column=0, pady=(8, 0), sticky="w")
        ttk.Button(top, text="Save form → YAML", command=self._save_yaml_from_form).grid(row=1, column=1, pady=(8, 0), sticky="w", padx=(6, 0))
        ttk.Button(top, text="New config…", command=self._new_config_dialog).grid(row=1, column=2, pady=(8, 0), sticky="e")
        info = ttk.LabelFrame(parent, text="Name sync", padding=10)
        info.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(info, text="Write model display name into configs/identity.sbe.", style="Subtitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(info, text="Sync name → identity.sbe", command=self._sync_name_to_identity).pack(anchor="w", pady=(6, 0))
        editor_frame = ttk.LabelFrame(parent, text="YAML editor (raw)", padding=8)
        editor_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)
        self.yaml_editor = ScrolledText(editor_frame, font=("Consolas", 10), height=18)
        self.yaml_editor.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Frame(editor_frame)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Reload file into editor", command=self._reload_yaml_editor).pack(side="left")
        ttk.Button(bar, text="Save editor to file", command=self._save_yaml_editor).pack(side="left", padx=6)
        ttk.Button(bar, text="Apply editor → form fields", command=self._yaml_editor_to_form).pack(side="left")

    def _build_knowledge_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(parent, text="configs/*.sbe files", padding=8)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        self.sbe_list = tk.Listbox(left, height=20, exportselection=False, width=28)
        self.sbe_list.pack(fill="both", expand=True)
        self.sbe_list.bind("<<ListboxSelect>>", self._on_sbe_select)
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Refresh", command=self._load_sbe_list).pack(side="left")
        ttk.Button(btns, text="New…", command=self._new_sbe_file).pack(side="left", padx=4)
        right = ttk.LabelFrame(parent, text="Edit knowledge (Q: / A: pairs)", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.sbe_editor = ScrolledText(right, font=("Consolas", 10))
        self.sbe_editor.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Frame(right)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Save .sbe", command=self._save_sbe_editor).pack(side="left")
        ttk.Button(bar, text="Insert Q/A template", command=self._insert_qa_template).pack(side="left", padx=6)

    def _build_identity_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        form = ttk.LabelFrame(parent, text="Identity fields", padding=10)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        self.id_name = tk.StringVar(value="Beq")
        self.id_role = tk.StringVar(value="an open-source pure-PyTorch language model")
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(form, textvariable=self.id_name).grid(row=0, column=1, sticky="ew")
        ttk.Label(form, text="Role").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        ttk.Entry(form, textvariable=self.id_role).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(form, text="About").grid(row=2, column=0, sticky="nw", padx=(0, 6), pady=(6, 0))
        self.id_about_box = ScrolledText(form, height=4, font=("Segoe UI", 10))
        self.id_about_box.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self.id_about_box.insert("1.0", "I am Beq, your own open-source AI.")
        actions = ttk.Frame(form)
        actions.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(actions, text="Load identity.sbe", command=self._load_identity_form).pack(side="left")
        ttk.Button(actions, text="Save identity.sbe", command=self._save_identity_form).pack(side="left", padx=6)
        ttk.Button(actions, text="Use as model display name", command=self._identity_to_display_name).pack(side="left")
        help_box = ttk.LabelFrame(parent, text="How knowledge answers work", padding=10)
        help_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(help_box, text="Website/API order: 1) configs/*.sbe  2) math calculator  3) neural model.\nEdit Knowledge tab so common questions are correct without relying on the small model.", justify="left").pack(anchor="w")
        self._load_identity_form()

    def _add_path_row(self, parent, row, label, variable, hint, browse_command) -> None:
        ttk.Label(parent, text=label).grid(row=row * 2, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row * 2, column=1, sticky="ew")
        ttk.Button(parent, text="Browse...", command=browse_command).grid(row=row * 2, column=2, padx=(6, 0))
        ttk.Label(parent, text=hint, style="Subtitle.TLabel").grid(row=row * 2 + 1, column=0, columnspan=3, sticky="w", pady=(0, 6))

    def _add_setting(self, parent, row, col, label, key) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(parent, textvariable=self.settings[key], width=14).grid(row=row, column=col + 1, sticky="w", pady=3)

    def _settings_changed(self, *_a) -> None:
        self._refresh_command_preview()

    def _browse_input(self) -> None:
        s = filedialog.askopenfilename(title="Training text", filetypes=[("Text", "*.txt"), ("All", "*.*")], initialdir=str(self.project_root / "data"))
        if s:
            self.input_path.set(s)

    def _browse_output(self) -> None:
        s = filedialog.askdirectory(title="Checkpoint folder", initialdir=str(self.project_root / "checkpoints"))
        if s:
            self.output_dir.set(s)

    def _browse_resume(self) -> None:
        s = filedialog.askopenfilename(title="Resume checkpoint", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")], initialdir=str(self.project_root / "checkpoints"))
        if s:
            self.resume_path.set(s)

    def _browse_python(self) -> None:
        s = filedialog.askopenfilename(title="Python executable")
        if s:
            self.python_executable.set(s)

    def _browse_config(self) -> None:
        s = filedialog.askopenfilename(title="YAML config", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")], initialdir=str(self.configs_dir))
        if s:
            self.config_path.set(s)
            self._load_yaml_if_exists(s)
            self._reload_yaml_editor()

    def _apply_preset(self) -> None:
        preset = PRESETS.get(self.preset_name.get())
        if not preset:
            return
        for k, v in preset.items():
            if k in self.settings:
                self.settings[k].set(str(v))
        self.status_text.set(f"Preset: {self.preset_name.get()}")

    def _effective_out_dir(self) -> Path:
        base = Path(self.output_dir.get().strip() or "checkpoints")
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.run_name.get().strip()) or "beq"
        if name and name not in ("beq", "beq-default"):
            return base / name
        return base

    def _build_command(self) -> list[str]:
        return [
            self.python_executable.get().strip() or sys.executable,
            str(self.train_script),
            "--data", self.input_path.get().strip(),
            "--out_dir", str(self._effective_out_dir()),
            "--d_model", self.settings["d_model"].get().strip(),
            "--n_layers", self.settings["n_layers"].get().strip(),
            "--n_heads", self.settings["n_heads"].get().strip(),
            "--block_size", self.settings["block_size"].get().strip(),
            "--batch_size", self.settings["batch_size"].get().strip(),
            "--lr", self.settings["learning_rate"].get().strip(),
            "--max_steps", self.settings["max_steps"].get().strip(),
            "--eval_interval", self.settings["eval_interval"].get().strip(),
            "--save_interval", self.settings["save_interval"].get().strip(),
        ]

    def _refresh_command_preview(self) -> None:
        try:
            text = " ".join(shlex.quote(p) for p in self._build_command())
        except Exception as e:
            text = f"(invalid: {e})"
        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", "end")
        self.command_preview.insert("1.0", text)
        self.command_preview.configure(state="disabled")

    def _form_to_yaml_dict(self) -> dict:
        return {
            "name": self.run_name.get().strip() or "beq",
            "display_name": self.model_display_name.get().strip() or "Beq",
            "model": {
                "d_model": int(float(self.settings["d_model"].get())),
                "n_layers": int(float(self.settings["n_layers"].get())),
                "n_heads": int(float(self.settings["n_heads"].get())),
                "block_size": int(float(self.settings["block_size"].get())),
            },
            "training": {
                "data_path": self.input_path.get().strip(),
                "out_dir": str(self._effective_out_dir()),
                "batch_size": int(float(self.settings["batch_size"].get())),
                "learning_rate": float(self.settings["learning_rate"].get()),
                "max_steps": int(float(self.settings["max_steps"].get())),
                "eval_interval": int(float(self.settings["eval_interval"].get())),
                "save_interval": int(float(self.settings["save_interval"].get())),
            },
        }

    def _yaml_dict_to_form(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        if data.get("name"):
            self.run_name.set(str(data["name"]))
        if data.get("display_name"):
            self.model_display_name.set(str(data["display_name"]))
        model, train = data.get("model") or {}, data.get("training") or {}
        mapping = {
            "d_model": model.get("d_model"), "n_layers": model.get("n_layers"),
            "n_heads": model.get("n_heads"), "block_size": model.get("block_size"),
            "batch_size": train.get("batch_size"), "learning_rate": train.get("learning_rate"),
            "max_steps": train.get("max_steps"), "eval_interval": train.get("eval_interval"),
            "save_interval": train.get("save_interval"),
        }
        for k, v in mapping.items():
            if v is not None and k in self.settings:
                self.settings[k].set(str(v))
        if train.get("data_path"):
            self.input_path.set(str(train["data_path"]))
        if train.get("out_dir"):
            self.output_dir.set(str(train["out_dir"]))

    def _load_yaml_if_exists(self, path_str: str) -> None:
        path = Path(path_str)
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
            if yaml is None:
                self.status_text.set("PyYAML missing — use raw editor")
                return
            self._yaml_dict_to_form(yaml.safe_load(text) or {})
            self.status_text.set(f"Loaded {path.name}")
        except Exception as e:
            self.status_text.set(f"Config load failed: {e}")

    def _load_yaml_from_path(self) -> None:
        self._load_yaml_if_exists(self.config_path.get())
        self._reload_yaml_editor()

    def _save_yaml_from_form(self) -> None:
        path = Path(self.config_path.get().strip())
        if not path.name:
            messagebox.showerror("Config", "Choose a config path first.")
            return
        try:
            data = self._form_to_yaml_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(data, sort_keys=False) if yaml else json.dumps(data, indent=2), encoding="utf-8")
            self._reload_yaml_editor()
            messagebox.showinfo("Config", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Config", str(e))

    def _new_config_dialog(self) -> None:
        name = simpledialog.askstring("New config", "File name:", initialvalue="my-model.yaml")
        if not name:
            return
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        self.config_path.set(str(self.configs_dir / name))
        self._save_yaml_from_form()

    def _reload_yaml_editor(self) -> None:
        path = Path(self.config_path.get())
        self.yaml_editor.delete("1.0", "end")
        if path.exists():
            self.yaml_editor.insert("1.0", path.read_text(encoding="utf-8"))
        else:
            try:
                data = self._form_to_yaml_dict()
                self.yaml_editor.insert("1.0", yaml.safe_dump(data, sort_keys=False) if yaml else json.dumps(data, indent=2))
            except Exception:
                pass

    def _save_yaml_editor(self) -> None:
        path = Path(self.config_path.get().strip())
        if not path.name:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.yaml_editor.get("1.0", "end-1c"), encoding="utf-8")
        self.status_text.set(f"Saved editor → {path.name}")

    def _yaml_editor_to_form(self) -> None:
        text = self.yaml_editor.get("1.0", "end-1c")
        try:
            data = yaml.safe_load(text) if yaml else json.loads(text)
            self._yaml_dict_to_form(data or {})
            self.status_text.set("YAML applied to form")
        except Exception as e:
            messagebox.showerror("YAML", str(e))

    def _load_sbe_list(self) -> None:
        self.sbe_list.delete(0, "end")
        files = sorted(self.configs_dir.glob("*.sbe"))
        for p in files:
            self.sbe_list.insert("end", p.name)
        if files and self._sbe_path is None:
            self.sbe_list.selection_set(0)
            self._on_sbe_select()

    def _on_sbe_select(self, _e=None) -> None:
        sel = self.sbe_list.curselection()
        if not sel:
            return
        path = self.configs_dir / self.sbe_list.get(sel[0])
        self._sbe_path = path
        self.sbe_editor.delete("1.0", "end")
        if path.exists():
            self.sbe_editor.insert("1.0", path.read_text(encoding="utf-8"))

    def _save_sbe_editor(self) -> None:
        if self._sbe_path is None:
            messagebox.showerror("Knowledge", "Select or create a .sbe file first.")
            return
        self._sbe_path.write_text(self.sbe_editor.get("1.0", "end-1c"), encoding="utf-8")
        messagebox.showinfo("Knowledge", f"Saved:\n{self._sbe_path}")

    def _new_sbe_file(self) -> None:
        name = simpledialog.askstring("New knowledge", "File name:", initialvalue="extra.sbe")
        if not name:
            return
        if not name.endswith(".sbe"):
            name += ".sbe"
        path = self.configs_dir / name
        if not path.exists():
            path.write_text("# Beq knowledge\n# Q: question?\n# A: answer\n\n", encoding="utf-8")
        self._load_sbe_list()
        for i in range(self.sbe_list.size()):
            if self.sbe_list.get(i) == name:
                self.sbe_list.selection_clear(0, "end")
                self.sbe_list.selection_set(i)
                self._on_sbe_select()
                break

    def _insert_qa_template(self) -> None:
        self.sbe_editor.insert("end", "\nQ: \nA: \n")

    def _identity_path(self) -> Path:
        return self.configs_dir / "identity.sbe"

    def _load_identity_form(self) -> None:
        path = self._identity_path()
        if not path.exists():
            return
        name, role, about = "Beq", "", ""
        section = None
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.lower() == "[identity]":
                section = "identity"
                continue
            if s.startswith("[") and s.endswith("]"):
                section = None
                continue
            if section == "identity" and "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "name":
                    name = v
                elif k == "role":
                    role = v
                elif k == "about":
                    about = v
        self.id_name.set(name)
        self.id_role.set(role)
        self.id_about_box.delete("1.0", "end")
        self.id_about_box.insert("1.0", about)
        self.model_display_name.set(name)

    def _save_identity_form(self) -> None:
        name = self.id_name.get().strip() or "Beq"
        role = self.id_role.get().strip()
        about = self.id_about_box.get("1.0", "end-1c").strip()
        path = self._identity_path()
        extra = ""
        if path.exists():
            keep = []
            in_id = False
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.lower() == "[identity]":
                    in_id = True
                    continue
                if in_id:
                    if s.startswith("Q:") or s.startswith("A:"):
                        in_id = False
                        keep.append(line)
                    elif s.startswith("[") and s.endswith("]"):
                        in_id = False
                        keep.append(line)
                    continue
                keep.append(line)
            extra = "\n".join(keep).strip()
        body = f"# Beq identity\n[identity]\nname={name}\nrole={role}\nabout={about}\n"
        if extra:
            body += "\n" + extra + "\n"
        path.write_text(body, encoding="utf-8")
        self.model_display_name.set(name)
        messagebox.showinfo("Identity", f"Saved:\n{path}")

    def _identity_to_display_name(self) -> None:
        self.model_display_name.set(self.id_name.get().strip() or "Beq")

    def _sync_name_to_identity(self) -> None:
        self.id_name.set(self.model_display_name.get().strip() or "Beq")
        self._save_identity_form()

    def _settings_payload(self) -> dict:
        return {
            "run_name": self.run_name.get(),
            "model_display_name": self.model_display_name.get(),
            "input_path": self.input_path.get(),
            "output_dir": self.output_dir.get(),
            "resume_path": self.resume_path.get(),
            "python_executable": self.python_executable.get(),
            "config_path": self.config_path.get(),
            "settings": {k: v.get() for k, v in self.settings.items()},
        }

    def _load_settings(self) -> None:
        path = filedialog.askopenfilename(title="Load settings JSON", filetypes=[("JSON", "*.json")], initialdir=str(self.project_root))
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.run_name.set(data.get("run_name", self.run_name.get()))
        self.model_display_name.set(data.get("model_display_name", self.model_display_name.get()))
        self.input_path.set(data.get("input_path", self.input_path.get()))
        self.output_dir.set(data.get("output_dir", self.output_dir.get()))
        self.resume_path.set(data.get("resume_path", ""))
        self.python_executable.set(data.get("python_executable", self.python_executable.get()))
        self.config_path.set(data.get("config_path", self.config_path.get()))
        for k, v in (data.get("settings") or {}).items():
            if k in self.settings:
                self.settings[k].set(str(v))

    def _save_settings(self) -> None:
        path = filedialog.asksaveasfilename(title="Save settings JSON", defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir=str(self.project_root), initialfile=f"{self.run_name.get() or 'beq'}-settings.json")
        if not path:
            return
        Path(path).write_text(json.dumps(self._settings_payload(), indent=2), encoding="utf-8")

    def _reset_defaults(self) -> None:
        for k, v in DEFAULTS.items():
            self.settings[k].set(v)
        self.preset_name.set("Default")

    def _validate(self) -> str | None:
        if not self.train_script.exists():
            return f"train/train.py not found at {self.train_script}"
        if not self.input_path.get().strip() or not Path(self.input_path.get()).exists():
            return "Select an existing training text file."
        for k in self.settings:
            try:
                float(self.settings[k].get())
            except ValueError:
                return f"Invalid number for {k}"
        return None

    def _start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Training", "Already running.")
            return
        err = self._validate()
        if err:
            messagebox.showerror("Cannot start", err)
            return
        out = self._effective_out_dir()
        out.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command()
        self._clear_log()
        self._append_log(f"Run: {self.run_name.get()} | Model: {self.model_display_name.get()}")
        self._append_log(" ".join(shlex.quote(c) for c in cmd))
        if self.resume_path.get().strip():
            self._append_log(f"Resume note: {self.resume_path.get()}")
        self.stop_requested = False
        self.current_step = 0
        self.current_loss = "n/a"
        self.start_time = time.monotonic()
        self._set_running_state(True)
        self.status_text.set("Training…")
        try:
            self.process = subprocess.Popen(cmd, cwd=str(self.project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            self._set_running_state(False)
            messagebox.showerror("Start failed", str(e))
            return
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self.process and self.process.stdout
        step_re = re.compile(r"step\s+(\d+).*loss\s+([0-9.]+)", re.I)
        for line in self.process.stdout:
            self.output_queue.put(("log", line.rstrip("\n")))
            m = step_re.search(line)
            if m:
                self.output_queue.put(("progress", (int(m.group(1)), m.group(2))))
        self.output_queue.put(("done", self.process.wait()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    self.current_action.set("Training")
                elif kind == "progress":
                    step, loss = payload  # type: ignore
                    self.current_step, self.current_loss = int(step), str(loss)
                    self.progress_text.set(f"Step {self.current_step} | loss {self.current_loss}")
                    try:
                        mx = max(1, int(float(self.settings["max_steps"].get())))
                        self.progress.configure(maximum=mx, value=min(self.current_step, mx))
                    except ValueError:
                        pass
                elif kind == "done":
                    self._on_process_done(int(payload))  # type: ignore
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _tick_elapsed(self) -> None:
        if self.start_time is not None:
            s = int(time.monotonic() - self.start_time)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            self.elapsed_text.set(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")
        self.after(500, self._tick_elapsed)

    def _on_process_done(self, code: int) -> None:
        self._set_running_state(False)
        self.process = None
        self.start_time = None
        if self.stop_requested:
            self.status_text.set("Stopped")
            self._append_log("Stopped.")
        elif code == 0:
            self.status_text.set("Completed")
            self._append_log("Training completed successfully.")
        else:
            self.status_text.set("Failed")
            self._append_log(f"Exited with code {code}.")

    def _stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stop_requested = True
        try:
            self.process.terminate()
        except OSError as e:
            self._append_log(str(e))

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
