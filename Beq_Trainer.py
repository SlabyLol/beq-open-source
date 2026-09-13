#!/usr/bin/env python3
"""Beq_Trainer.py — complete 9-tab Beq control panel (single real file, no parts)."""
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
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    "d_model": "256", "n_layers": "6", "n_heads": "8", "block_size": "128",
    "batch_size": "32", "learning_rate": "0.0003", "max_steps": "5000",
    "eval_interval": "250", "save_interval": "1000",
}
PRESETS = {
    "Tiny": {"d_model": "128", "n_layers": "4", "n_heads": "4", "block_size": "64",
             "batch_size": "8", "learning_rate": "0.0003", "max_steps": "500",
             "eval_interval": "100", "save_interval": "250"},
    "Default": dict(DEFAULTS),
    "Stronger": {"d_model": "384", "n_layers": "8", "n_heads": "8", "block_size": "256",
                 "batch_size": "16", "learning_rate": "0.0002", "max_steps": "10000",
                 "eval_interval": "250", "save_interval": "1000"},
    "Quick test": {"d_model": "64", "n_layers": "2", "n_heads": "2", "block_size": "32",
                   "batch_size": "4", "learning_rate": "0.001", "max_steps": "50",
                   "eval_interval": "25", "save_interval": "50"},
}


class BeqTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Beq_Trainer — full control panel")
        self.geometry("1200x880")
        self.minsize(960, 700)
        self.root_dir = Path(__file__).resolve().parent
        self.train_script = self.root_dir / "train" / "train.py"
        self.configs_dir = self.root_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
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
        self.data_stats = tk.StringVar(value="")
        self.settings = {k: tk.StringVar(value=v) for k, v in DEFAULTS.items()}
        self.process = None
        self.output_queue: queue.Queue = queue.Queue()
        self.start_time = None
        self._sbe_path = None
        self._data_file = tk.StringVar(value="input.txt")
        self._build_ui()
        self._refresh_cmd()
        self._refresh_data_stats()
        self._refresh_ckpts()
        self.after(200, self._poll)
        self.after(500, self._tick)
        self._load_yaml(self.config_path.get())
        self._load_sbe_list()

    def _build_ui(self) -> None:
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        ttk.Label(root, text="Beq_Trainer", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        nb = ttk.Notebook(root)
        nb.grid(row=1, column=0, sticky="nsew")
        tabs = {}
        for key, title in [("train", "Train"), ("config", "Config"), ("knowledge", "Knowledge"),
                           ("identity", "Identity"), ("data", "Data"), ("ckpt", "Checkpoints"),
                           ("gen", "Generate"), ("tools", "Tools"), ("help", "Help")]:
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text=f"  {title}  ")
            tabs[key] = f
        self._tab_train(tabs["train"])
        self._tab_config(tabs["config"])
        self._tab_knowledge(tabs["knowledge"])
        self._tab_identity(tabs["identity"])
        self._tab_data(tabs["data"])
        self._tab_ckpt(tabs["ckpt"])
        self._tab_gen(tabs["gen"])
        self._tab_tools(tabs["tools"])
        self._tab_help(tabs["help"])
        bar = ttk.Frame(root)
        bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(bar, textvariable=self.status_text).pack(side="left")
        ttk.Label(bar, textvariable=self.elapsed_text).pack(side="left", padx=12)
        ttk.Label(bar, textvariable=self.progress_text).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="determinate")
        self.progress.pack(fill="x", pady=4)

    def _tab_train(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(4, weight=1)
        idf = ttk.LabelFrame(p, text="Run", padding=8)
        idf.grid(row=0, column=0, sticky="ew")
        idf.columnconfigure(1, weight=1)
        ttk.Label(idf, text="Run name").grid(row=0, column=0, sticky="w")
        ttk.Entry(idf, textvariable=self.run_name).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(idf, text="Display name").grid(row=0, column=2, sticky="w")
        ttk.Entry(idf, textvariable=self.model_display_name).grid(row=0, column=3, sticky="ew")
        paths = ttk.LabelFrame(p, text="Paths", padding=8)
        paths.grid(row=1, column=0, sticky="ew", pady=4)
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, "Data", self.input_path, self._browse_input)
        self._path_row(paths, 1, "Out", self.output_dir, self._browse_out)
        self._path_row(paths, 2, "Resume", self.resume_path, self._browse_resume)
        mf = ttk.LabelFrame(p, text="Model / train", padding=8)
        mf.grid(row=2, column=0, sticky="ew")
        for i, k in enumerate(["d_model", "n_layers", "n_heads", "block_size", "batch_size",
                               "learning_rate", "max_steps", "eval_interval", "save_interval"]):
            r, c = divmod(i, 4)
            ttk.Label(mf, text=k).grid(row=r, column=c * 2, sticky="w", padx=2)
            ttk.Entry(mf, textvariable=self.settings[k], width=10).grid(row=r, column=c * 2 + 1, sticky="w")
        rt = ttk.LabelFrame(p, text="Runtime", padding=8)
        rt.grid(row=3, column=0, sticky="ew", pady=4)
        rt.columnconfigure(1, weight=1)
        ttk.Label(rt, text="Python").grid(row=0, column=0)
        ttk.Entry(rt, textvariable=self.python_executable).grid(row=0, column=1, sticky="ew")
        ttk.Button(rt, text="Browse", command=self._browse_py).grid(row=0, column=2)
        ttk.Label(rt, text="Preset").grid(row=1, column=0)
        ttk.Combobox(rt, textvariable=self.preset_name, values=list(PRESETS), state="readonly").grid(row=1, column=1, sticky="ew")
        ttk.Button(rt, text="Apply", command=self._apply_preset).grid(row=1, column=2)
        bottom = ttk.Panedwindow(p, orient="vertical")
        bottom.grid(row=4, column=0, sticky="nsew")
        pf = ttk.LabelFrame(bottom, text="Command", padding=4)
        self.cmd_preview = tk.Text(pf, height=3, font=("Consolas", 10))
        self.cmd_preview.pack(fill="both", expand=True)
        bottom.add(pf)
        lf = ttk.LabelFrame(bottom, text="Log", padding=4)
        self.log = ScrolledText(lf, height=12, font=("Consolas", 10), state="disabled")
        self.log.pack(fill="both", expand=True)
        bottom.add(lf)
        act = ttk.Frame(p)
        act.grid(row=5, column=0, sticky="ew", pady=4)
        ttk.Button(act, text="Validate", command=self._validate).pack(side="left", padx=2)
        ttk.Button(act, text="Clear log", command=self._clear_log).pack(side="left", padx=2)
        self.btn_stop = ttk.Button(act, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="right", padx=2)
        self.btn_start = ttk.Button(act, text="Start training", command=self._start)
        self.btn_start.pack(side="right")

    def _tab_config(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)
        top = ttk.LabelFrame(p, text="YAML", padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Entry(top, textvariable=self.config_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_cfg).grid(row=0, column=1)
        ttk.Button(top, text="Load YAML", command=lambda: self._load_yaml(self.config_path.get())).grid(row=1, column=0, sticky="w")
        ttk.Button(top, text="Save form→YAML", command=self._save_yaml).grid(row=1, column=1)
        ed = ttk.LabelFrame(p, text="Editor", padding=4)
        ed.grid(row=1, column=0, sticky="nsew", pady=4)
        ed.columnconfigure(0, weight=1)
        ed.rowconfigure(0, weight=1)
        self.yaml_ed = ScrolledText(ed, font=("Consolas", 10))
        self.yaml_ed.grid(row=0, column=0, sticky="nsew")

    def _tab_knowledge(self, p):
        p.columnconfigure(1, weight=1)
        p.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(p, text="*.sbe", padding=4)
        left.grid(row=0, column=0, sticky="nsw")
        self.sbe_list = tk.Listbox(left, height=18, width=26, exportselection=False)
        self.sbe_list.pack(fill="both", expand=True)
        self.sbe_list.bind("<<ListboxSelect>>", self._sbe_sel)
        bf = ttk.Frame(left)
        bf.pack(fill="x")
        ttk.Button(bf, text="Refresh", command=self._load_sbe_list).pack(side="left")
        ttk.Button(bf, text="New", command=self._new_sbe).pack(side="left")
        right = ttk.LabelFrame(p, text="Editor", padding=4)
        right.grid(row=0, column=1, sticky="nsew", padx=4)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.sbe_ed = ScrolledText(right, font=("Consolas", 10))
        self.sbe_ed.grid(row=0, column=0, sticky="nsew")
        ttk.Button(right, text="Save .sbe", command=self._save_sbe).grid(row=1, column=0, sticky="w")

    def _tab_identity(self, p):
        p.columnconfigure(1, weight=1)
        self.id_name = tk.StringVar(value="Beq")
        self.id_role = tk.StringVar(value="helpful assistant")
        ttk.Label(p, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(p, textvariable=self.id_name).grid(row=0, column=1, sticky="ew")
        ttk.Label(p, text="Role").grid(row=1, column=0, sticky="w")
        ttk.Entry(p, textvariable=self.id_role).grid(row=1, column=1, sticky="ew")
        ttk.Label(p, text="About").grid(row=2, column=0, sticky="nw")
        self.id_about = ScrolledText(p, height=6)
        self.id_about.grid(row=2, column=1, sticky="ew")
        self.id_about.insert("1.0", "I am Beq, an open-source AI.")
        ttk.Button(p, text="Save identity.sbe", command=self._save_identity).grid(row=3, column=1, sticky="w", pady=6)
        self._load_identity()

    def _tab_data(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)
        top = ttk.Frame(p)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, textvariable=self.data_stats).pack(side="left")
        ttk.Button(top, text="input.txt", command=lambda: self._load_data("input.txt")).pack(side="left", padx=2)
        ttk.Button(top, text="input_extra.txt", command=lambda: self._load_data("input_extra.txt")).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self._save_data).pack(side="left", padx=2)
        self.data_ed = ScrolledText(p, font=("Consolas", 10))
        self.data_ed.grid(row=1, column=0, sticky="nsew", pady=4)

    def _tab_ckpt(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)
        bar = ttk.Frame(p)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Refresh", command=self._refresh_ckpts).pack(side="left")
        ttk.Button(bar, text="Open folder", command=self._open_ckpts).pack(side="left", padx=2)
        ttk.Button(bar, text="Promote→beq_best.pt", command=self._promote).pack(side="left", padx=2)
        self.ckpt_list = tk.Listbox(p, font=("Consolas", 10))
        self.ckpt_list.grid(row=1, column=0, sticky="nsew", pady=4)

    def _tab_gen(self, p):
        p.columnconfigure(1, weight=1)
        p.rowconfigure(5, weight=1)
        ttk.Label(p, text="Checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(p, textvariable=self.gen_ckpt).grid(row=0, column=1, sticky="ew")
        ttk.Button(p, text="Browse", command=self._browse_gen).grid(row=0, column=2)
        ttk.Label(p, text="Prompt").grid(row=1, column=0, sticky="w")
        ttk.Entry(p, textvariable=self.gen_prompt).grid(row=1, column=1, sticky="ew")
        ttk.Label(p, text="Tokens").grid(row=2, column=0)
        ttk.Entry(p, textvariable=self.gen_tokens, width=8).grid(row=2, column=1, sticky="w")
        ttk.Button(p, text="Generate", command=self._generate).grid(row=4, column=1, sticky="w", pady=6)
        self.gen_out = ScrolledText(p, font=("Consolas", 10))
        self.gen_out.grid(row=5, column=0, columnspan=3, sticky="nsew")

    def _tab_tools(self, p):
        for label, cmd in [
            ("Open project", lambda: self._open(self.root_dir)),
            ("Open data/", lambda: self._open(self.data_dir)),
            ("Open configs/", lambda: self._open(self.configs_dir)),
            ("pip install -r requirements.txt", self._pip),
            ("Open beq.onrender.com", lambda: webbrowser.open("https://beq.onrender.com")),
            ("Open GitHub", lambda: webbrowser.open("https://github.com/SlabyLol/beq-open-source")),
        ]:
            ttk.Button(p, text=label, command=cmd).pack(anchor="w", pady=3)

    def _tab_help(self, p):
        t = ScrolledText(p, wrap="word", font=("Segoe UI", 10))
        t.pack(fill="both", expand=True)
        t.insert("1.0", "Beq_Trainer — single file, 9 tabs.\nTrain, Config, Knowledge, Identity, Data, Checkpoints, Generate, Tools, Help.\nSBE optional. Crawler is on website /admin.\n")
        t.configure(state="disabled")

    def _path_row(self, parent, row, label, var, cmd):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="…", command=cmd, width=3).grid(row=row, column=2)

    def _browse_input(self):
        s = filedialog.askopenfilename(initialdir=str(self.data_dir))
        if s:
            self.input_path.set(s)
            self._refresh_cmd()

    def _browse_out(self):
        s = filedialog.askdirectory(initialdir=str(self.root_dir / "checkpoints"))
        if s:
            self.output_dir.set(s)
            self._refresh_cmd()

    def _browse_resume(self):
        s = filedialog.askopenfilename(filetypes=[("pt", "*.pt")])
        if s:
            self.resume_path.set(s)

    def _browse_py(self):
        s = filedialog.askopenfilename()
        if s:
            self.python_executable.set(s)
            self._refresh_cmd()

    def _browse_cfg(self):
        s = filedialog.askopenfilename(initialdir=str(self.configs_dir), filetypes=[("yaml", "*.yaml *.yml")])
        if s:
            self.config_path.set(s)
            self._load_yaml(s)

    def _browse_gen(self):
        s = filedialog.askopenfilename(filetypes=[("pt", "*.pt")], initialdir=str(self.root_dir / "checkpoints"))
        if s:
            self.gen_ckpt.set(s)

    def _apply_preset(self):
        for k, v in (PRESETS.get(self.preset_name.get()) or {}).items():
            if k in self.settings:
                self.settings[k].set(str(v))
        self._refresh_cmd()

    def _out_dir(self) -> Path:
        base = Path(self.output_dir.get() or "checkpoints")
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.run_name.get().strip()) or "beq"
        return base / name if name not in ("beq", "beq-default") else base

    def _cmd(self) -> list:
        return [self.python_executable.get().strip() or sys.executable, str(self.train_script),
                "--data", self.input_path.get().strip(), "--out_dir", str(self._out_dir()),
                "--d_model", self.settings["d_model"].get(), "--n_layers", self.settings["n_layers"].get(),
                "--n_heads", self.settings["n_heads"].get(), "--block_size", self.settings["block_size"].get(),
                "--batch_size", self.settings["batch_size"].get(), "--lr", self.settings["learning_rate"].get(),
                "--max_steps", self.settings["max_steps"].get(), "--eval_interval", self.settings["eval_interval"].get(),
                "--save_interval", self.settings["save_interval"].get()]

    def _refresh_cmd(self):
        try:
            text = " ".join(shlex.quote(x) for x in self._cmd())
        except Exception as e:
            text = str(e)
        self.cmd_preview.delete("1.0", "end")
        self.cmd_preview.insert("1.0", text)

    def _load_yaml(self, path_str: str):
        path = Path(path_str)
        if not path.exists() or yaml is None:
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if data.get("name"):
                self.run_name.set(str(data["name"]))
            if data.get("display_name"):
                self.model_display_name.set(str(data["display_name"]))
            model, train = data.get("model") or {}, data.get("training") or {}
            for k, src in [("d_model", model), ("n_layers", model), ("n_heads", model), ("block_size", model),
                           ("batch_size", train), ("learning_rate", train), ("max_steps", train),
                           ("eval_interval", train), ("save_interval", train)]:
                if src.get(k) is not None and k in self.settings:
                    self.settings[k].set(str(src[k]))
            self.yaml_ed.delete("1.0", "end")
            self.yaml_ed.insert("1.0", path.read_text(encoding="utf-8"))
            self._refresh_cmd()
        except Exception as e:
            self.status_text.set(str(e))

    def _save_yaml(self):
        path = Path(self.config_path.get().strip())
        data = {"name": self.run_name.get(), "display_name": self.model_display_name.get(),
                "model": {k: int(float(self.settings[k].get())) for k in ("d_model", "n_layers", "n_heads", "block_size")},
                "training": {"data_path": self.input_path.get(), "out_dir": str(self._out_dir()),
                             "batch_size": int(float(self.settings["batch_size"].get())),
                             "learning_rate": float(self.settings["learning_rate"].get()),
                             "max_steps": int(float(self.settings["max_steps"].get())),
                             "eval_interval": int(float(self.settings["eval_interval"].get())),
                             "save_interval": int(float(self.settings["save_interval"].get()))}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False) if yaml else json.dumps(data, indent=2), encoding="utf-8")
        messagebox.showinfo("Config", f"Saved {path}")

    def _load_sbe_list(self):
        self.sbe_list.delete(0, "end")
        for p in sorted(self.configs_dir.glob("*.sbe")):
            self.sbe_list.insert("end", p.name)

    def _sbe_sel(self, _e=None):
        sel = self.sbe_list.curselection()
        if not sel:
            return
        path = self.configs_dir / self.sbe_list.get(sel[0])
        self._sbe_path = path
        self.sbe_ed.delete("1.0", "end")
        if path.exists():
            self.sbe_ed.insert("1.0", path.read_text(encoding="utf-8"))

    def _save_sbe(self):
        if not self._sbe_path:
            return messagebox.showerror("SBE", "Select a file")
        self._sbe_path.write_text(self.sbe_ed.get("1.0", "end-1c"), encoding="utf-8")
        messagebox.showinfo("SBE", f"Saved {self._sbe_path}")

    def _new_sbe(self):
        name = simpledialog.askstring("New", "Name:", initialvalue="extra.sbe")
        if not name:
            return
        if not name.endswith(".sbe"):
            name += ".sbe"
        path = self.configs_dir / name
        if not path.exists():
            path.write_text("# Q: \n# A: \n\n", encoding="utf-8")
        self._load_sbe_list()

    def _load_identity(self):
        path = self.configs_dir / "identity.sbe"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "name":
                    self.id_name.set(v)
                elif k == "role":
                    self.id_role.set(v)
                elif k == "about":
                    self.id_about.delete("1.0", "end")
                    self.id_about.insert("1.0", v)

    def _save_identity(self):
        path = self.configs_dir / "identity.sbe"
        path.write_text(f"[ai]\nai=default\nname={self.id_name.get().strip() or 'Beq'}\nrole={self.id_role.get().strip()}\nabout={self.id_about.get('1.0', 'end-1c').strip()}\n", encoding="utf-8")
        self.model_display_name.set(self.id_name.get())
        messagebox.showinfo("Identity", f"Saved {path}")

    def _refresh_data_stats(self):
        parts = []
        for n in ("input.txt", "input_extra.txt", "input_stories.txt"):
            p = self.data_dir / n
            parts.append(f"{n}: {p.stat().st_size if p.exists() else 0} B")
        self.data_stats.set(" | ".join(parts))

    def _load_data(self, name: str):
        self._data_file.set(name)
        path = self.data_dir / name
        self.data_ed.delete("1.0", "end")
        if path.exists():
            self.data_ed.insert("1.0", path.read_text(encoding="utf-8", errors="ignore"))
        self._refresh_data_stats()

    def _save_data(self):
        name = self._data_file.get() or "input.txt"
        path = self.data_dir / name
        path.write_text(self.data_ed.get("1.0", "end-1c"), encoding="utf-8")
        self._refresh_data_stats()
        messagebox.showinfo("Data", f"Saved {path}")

    def _refresh_ckpts(self):
        self.ckpt_list.delete(0, "end")
        root = Path(self.output_dir.get() or self.root_dir / "checkpoints")
        if not root.exists():
            return
        for p in sorted(root.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            self.ckpt_list.insert("end", f"{p}  ({p.stat().st_size} bytes)")

    def _sel_ckpt(self):
        sel = self.ckpt_list.curselection()
        if not sel:
            return None
        path = Path(self.ckpt_list.get(sel[0]).split("  (")[0].strip())
        return path if path.exists() else None

    def _promote(self):
        p = self._sel_ckpt()
        if not p:
            return
        dest = Path(self.output_dir.get()) / "beq_best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        messagebox.showinfo("Checkpoint", f"Copied to {dest}")
        self._refresh_ckpts()

    def _open_ckpts(self):
        Path(self.output_dir.get()).mkdir(parents=True, exist_ok=True)
        self._open(Path(self.output_dir.get()))

    def _open(self, path: Path):
        path = path.resolve()
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _generate(self):
        ckpt = self.gen_ckpt.get().strip()
        if not ckpt or not Path(ckpt).exists():
            return messagebox.showerror("Generate", "Select checkpoint")
        self.gen_out.delete("1.0", "end")
        self.gen_out.insert("end", "Generating…\n")

        def worker():
            try:
                import torch
                sys.path.insert(0, str(self.root_dir))
                from model import BeqTransformer, CharTokenizer
                tok = Path(ckpt).parent / "tokenizer.json"
                if not tok.exists():
                    tok = self.root_dir / "checkpoints" / "tokenizer.json"
                tokenizer = CharTokenizer.load(tok)
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                cfg = blob["config"]
                model = BeqTransformer(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                                       n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], max_seq_len=cfg["max_seq_len"])
                model.load_state_dict(blob["model"])
                model.eval()
                ids = tokenizer.encode(self.gen_prompt.get()) or tokenizer.encode("a")
                with torch.no_grad():
                    out = model.generate(torch.tensor([ids], dtype=torch.long),
                                         max_new_tokens=int(float(self.gen_tokens.get() or 80)),
                                         temperature=float(self.gen_temp.get() or 0.8), top_k=30)
                self.output_queue.put(("gen", tokenizer.decode(out[0].tolist())))
            except Exception as e:
                self.output_queue.put(("gen", f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _pip(self):
        req = self.root_dir / "requirements.txt"
        if not req.exists():
            return messagebox.showerror("pip", "requirements.txt missing")
        try:
            out = subprocess.check_output([self.python_executable.get() or sys.executable, "-m", "pip", "install", "-r", str(req)],
                                          cwd=str(self.root_dir), stderr=subprocess.STDOUT, text=True)
            messagebox.showinfo("pip", out[-1500:] or "OK")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("pip", (e.output or str(e))[-1500:])

    def _validate(self):
        errs = []
        if not self.train_script.exists():
            errs.append("train/train.py missing")
        if not Path(self.input_path.get()).exists():
            errs.append("data file missing")
        try:
            if int(float(self.settings["d_model"].get())) % int(float(self.settings["n_heads"].get())):
                errs.append("d_model must be divisible by n_heads")
        except ValueError:
            errs.append("invalid numbers")
        messagebox.showinfo("Validate", "\n".join(errs) if errs else "OK")

    def _start(self):
        if self.process and self.process.poll() is None:
            return messagebox.showwarning("Train", "Already running")
        if not self.train_script.exists() or not Path(self.input_path.get()).exists():
            return messagebox.showerror("Train", "train script or data missing")
        self._out_dir().mkdir(parents=True, exist_ok=True)
        self._clear_log()
        self._append_log(" ".join(shlex.quote(c) for c in self._cmd()))
        self.start_time = time.monotonic()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_text.set("Training…")
        try:
            self.process = subprocess.Popen(self._cmd(), cwd=str(self.root_dir),
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            return messagebox.showerror("Start", str(e))
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        assert self.process and self.process.stdout
        step_re = re.compile(r"step\s+(\d+).*loss\s+([0-9.]+)", re.I)
        for line in self.process.stdout:
            self.output_queue.put(("log", line.rstrip("\n")))
            m = step_re.search(line)
            if m:
                self.output_queue.put(("prog", (int(m.group(1)), m.group(2))))
        self.output_queue.put(("done", self.process.wait()))

    def _poll(self):
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "prog":
                    step, loss = payload
                    self.progress_text.set(f"step {step} loss {loss}")
                    try:
                        mx = max(1, int(float(self.settings["max_steps"].get())))
                        self.progress.configure(maximum=mx, value=min(step, mx))
                    except ValueError:
                        pass
                elif kind == "done":
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.process = None
                    self.start_time = None
                    self.status_text.set("Done" if int(payload) == 0 else f"Exit {payload}")
                    self._refresh_ckpts()
                elif kind == "gen":
                    self.gen_out.delete("1.0", "end")
                    self.gen_out.insert("end", str(payload))
        except queue.Empty:
            pass
        self.after(200, self._poll)

    def _tick(self):
        if self.start_time is not None:
            s = int(time.monotonic() - self.start_time)
            m, s = divmod(s, 60)
            self.elapsed_text.set(f"Elapsed: {m:02d}:{s:02d}")
        self.after(500, self._tick)

    def _stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Quit", "Stop training and quit?"):
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
