#!/usr/bin/env python3
"""Beq Trainer - an English Tkinter interface for training Beq.

This GUI launches the repository's existing train/train.py script and exposes
all of its command-line training options without duplicating the trainer.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


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


class BeqTrainerApp(tk.Tk):
    """Desktop training control panel for the Beq language model."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Beq Trainer")
        self.geometry("1100x820")
        self.minsize(900, 680)

        self.project_root = Path(__file__).resolve().parent
        self.train_script = self.project_root / "train" / "train.py"
        default_input = self.project_root / "data" / "input.txt"
        default_output = self.project_root / "checkpoints"

        self.input_path = tk.StringVar(
            value=str(default_input) if default_input.exists() else ""
        )
        self.output_dir = tk.StringVar(value=str(default_output))
        self.python_executable = tk.StringVar(value=sys.executable)
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="No training run is active")

        self.settings = {
            key: tk.StringVar(value=value) for key, value in DEFAULTS.items()
        }
        for variable in self.settings.values():
            variable.trace_add("write", self._settings_changed)
        self.input_path.trace_add("write", self._settings_changed)
        self.output_dir.trace_add("write", self._settings_changed)
        self.python_executable.trace_add("write", self._settings_changed)

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_requested = False
        self.current_step = 0

        self._configure_style()
        self._build_interface()
        self._update_command_preview()
        self.after(100, self._poll_output)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Subtitle.TLabel", foreground="#555555")
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Status.TLabel", font=("TkDefaultFont", 10, "bold"))

    def _build_interface(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Beq Trainer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Configure and train your own Beq language model with a simple desktop interface.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        paths = ttk.LabelFrame(root, text="Training files", padding=10)
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        paths.columnconfigure(1, weight=1)
        self._add_path_row(
            paths,
            0,
            "Training text file",
            self.input_path,
            "Select the input.txt or other plain-text file used for training.",
            self._browse_input,
        )
        self._add_path_row(
            paths,
            1,
            "Checkpoint folder",
            self.output_dir,
            "Model checkpoints and tokenizer.json will be saved here.",
            self._browse_output,
        )

        model_frame = ttk.LabelFrame(root, text="Model configuration", padding=10)
        model_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            model_frame.columnconfigure(column, weight=1)
        self._add_setting(model_frame, 0, 0, "Embedding size (d_model)", "d_model")
        self._add_setting(model_frame, 0, 2, "Transformer layers", "n_layers")
        self._add_setting(model_frame, 1, 0, "Attention heads", "n_heads")
        self._add_setting(model_frame, 1, 2, "Context length (block size)", "block_size")

        training_frame = ttk.LabelFrame(root, text="Training configuration", padding=10)
        training_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            training_frame.columnconfigure(column, weight=1)
        self._add_setting(training_frame, 0, 0, "Batch size", "batch_size")
        self._add_setting(training_frame, 0, 2, "Learning rate", "learning_rate")
        self._add_setting(training_frame, 1, 0, "Maximum steps", "max_steps")
        self._add_setting(training_frame, 1, 2, "Evaluation interval", "eval_interval")
        self._add_setting(training_frame, 2, 0, "Checkpoint interval", "save_interval")

        runtime = ttk.LabelFrame(root, text="Runtime", padding=10)
        runtime.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        runtime.columnconfigure(1, weight=1)
        ttk.Label(runtime, text="Python interpreter").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        ttk.Entry(runtime, textvariable=self.python_executable).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(runtime, text="Browse...", command=self._browse_python).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Label(
            runtime,
            text="The trainer automatically selects CUDA, Apple MPS, or CPU.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=1, sticky="w", pady=(6, 0))

        bottom = ttk.PanedWindow(root, orient="vertical")
        bottom.grid(row=5, column=0, sticky="nsew")

        preview_frame = ttk.LabelFrame(bottom, text="Command preview", padding=8)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.command_preview = tk.Text(
            preview_frame,
            height=3,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 9),
            background="#f4f4f4",
        )
        self.command_preview.grid(row=0, column=0, sticky="nsew")
        bottom.add(preview_frame, weight=0)

        log_frame = ttk.LabelFrame(bottom, text="Training output", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = ScrolledText(log_frame, wrap="none", state="disabled", height=12)
        self.log.grid(row=0, column=0, sticky="nsew")
        bottom.add(log_frame, weight=1)

        status_bar = ttk.Frame(root)
        status_bar.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        status_bar.columnconfigure(0, weight=1)
        status_bar.columnconfigure(1, weight=2)
        ttk.Label(status_bar, textvariable=self.status_text, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_bar, textvariable=self.progress_text).grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        self.progress = ttk.Progressbar(status_bar, mode="determinate", maximum=1)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(12, 0))

        actions = ttk.Frame(root)
        actions.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Load Settings", command=self._load_settings).pack(
            side="left"
        )
        ttk.Button(actions, text="Save Settings", command=self._save_settings).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Reset Defaults", command=self._reset_defaults).pack(
            side="left", padx=(8, 0)
        )
        self.stop_button = ttk.Button(
            actions, text="Stop Training", command=self._stop_training, state="disabled"
        )
        self.stop_button.pack(side="right")
        self.start_button = ttk.Button(
            actions, text="Start Training", command=self._start_training
        )
        self.start_button.pack(side="right", padx=(0, 8))

    def _add_path_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        hint: str,
        browse_command,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=(0 if row == 0 else 8, 0)
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=(0 if row == 0 else 8, 0)
        )
        ttk.Button(parent, text="Browse...", command=browse_command).grid(
            row=row, column=2, padx=(8, 0), pady=(0 if row == 0 else 8, 0)
        )
        ttk.Label(parent, text=hint, style="Subtitle.TLabel").grid(
            row=row + 2, column=1, sticky="w"
        ) if row == 1 else None

    def _add_setting(
        self,
        parent: ttk.LabelFrame,
        row: int,
        column: int,
        label: str,
        key: str,
    ) -> None:
        base_column = column
        ttk.Label(parent, text=label).grid(
            row=row, column=base_column, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(parent, textvariable=self.settings[key], width=14).grid(
            row=row, column=base_column + 1, sticky="ew", padx=(0, 18), pady=4
        )

    def _settings_changed(self, *_args) -> None:
        self._update_command_preview()

    def _browse_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select training text file",
            initialdir=str(self.project_root / "data"),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if selected:
            self.input_path.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Select checkpoint folder",
            initialdir=self.output_dir.get() or str(self.project_root),
            mustexist=False,
        )
        if selected:
            self.output_dir.set(selected)

    def _browse_python(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Python interpreter",
            filetypes=[("Python executable", "python*"), ("All files", "*.*")],
        )
        if selected:
            self.python_executable.set(selected)

    def _reset_defaults(self) -> None:
        for key, value in DEFAULTS.items():
            self.settings[key].set(value)
        self.status_text.set("Defaults restored")
        self._append_log("Default training settings restored.")

    def _load_settings(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load Beq trainer settings",
            filetypes=[("JSON settings", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            data = json.loads(Path(selected).read_text(encoding="utf-8"))
            for key, variable in self.settings.items():
                if key in data:
                    variable.set(str(data[key]))
            for key, variable in (
                ("input_path", self.input_path),
                ("output_dir", self.output_dir),
                ("python_executable", self.python_executable),
            ):
                if key in data:
                    variable.set(str(data[key]))
            self.status_text.set("Settings loaded")
            self._append_log(f"Loaded settings from {selected}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            messagebox.showerror("Settings error", f"Could not load settings:\n{error}")

    def _save_settings(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save Beq trainer settings",
            defaultextension=".json",
            filetypes=[("JSON settings", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        data = {key: variable.get() for key, variable in self.settings.items()}
        data.update(
            {
                "input_path": self.input_path.get(),
                "output_dir": self.output_dir.get(),
                "python_executable": self.python_executable.get(),
            }
        )
        try:
            Path(selected).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.status_text.set("Settings saved")
            self._append_log(f"Saved settings to {selected}")
        except OSError as error:
            messagebox.showerror("Settings error", f"Could not save settings:\n{error}")

    def _read_configuration(self) -> dict[str, object]:
        input_file = Path(os.path.expanduser(self.input_path.get().strip())).resolve()
        output_dir = Path(os.path.expanduser(self.output_dir.get().strip())).resolve()
        python_executable = self.python_executable.get().strip()

        if not input_file.is_file():
            raise ValueError("Select an existing plain-text training file.")
        if not self.train_script.is_file():
            raise ValueError(f"Could not find the repository trainer at {self.train_script}")
        if not python_executable:
            raise ValueError("Select a Python interpreter.")

        integer_fields = {
            key: int(self.settings[key].get().strip())
            for key in (
                "d_model",
                "n_layers",
                "n_heads",
                "block_size",
                "batch_size",
                "max_steps",
                "eval_interval",
                "save_interval",
            )
        }
        for key, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{key} must be at least 1.")
        if integer_fields["d_model"] % integer_fields["n_heads"] != 0:
            raise ValueError("Embedding size must be divisible by the number of attention heads.")

        learning_rate = float(self.settings["learning_rate"].get().strip())
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")

        return {
            "input_file": input_file,
            "output_dir": output_dir,
            "python_executable": python_executable,
            **integer_fields,
            "learning_rate": learning_rate,
        }

    def _build_command(self) -> list[str]:
        config = self._read_configuration()
        return [
            str(config["python_executable"]),
            "-u",
            str(self.train_script),
            "--data",
            str(config["input_file"]),
            "--out_dir",
            str(config["output_dir"]),
            "--d_model",
            str(config["d_model"]),
            "--n_layers",
            str(config["n_layers"]),
            "--n_heads",
            str(config["n_heads"]),
            "--block_size",
            str(config["block_size"]),
            "--batch_size",
            str(config["batch_size"]),
            "--lr",
            str(config["learning_rate"]),
            "--max_steps",
            str(config["max_steps"]),
            "--eval_interval",
            str(config["eval_interval"]),
            "--save_interval",
            str(config["save_interval"]),
        ]

    def _update_command_preview(self) -> None:
        try:
            command = shlex.join(self._build_command())
            preview = command
        except (ValueError, OSError):
            preview = "Complete the configuration to preview the training command."
        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", "end")
        self.command_preview.insert("1.0", preview)
        self.command_preview.configure(state="disabled")

    def _start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        try:
            command = self._build_command()
            output_dir = Path(command[command.index("--out_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as error:
            messagebox.showerror("Configuration error", str(error))
            return

        self.stop_requested = False
        self.current_step = 0
        self.progress.configure(value=0, maximum=max(1, int(self.settings["max_steps"].get())))
        self.progress_text.set("Starting training...")
        self.status_text.set("Training is running")
        self._set_running_state(True)
        self._clear_log()
        self._append_log("Starting Beq training...")
        self._append_log(f"Working directory: {self.project_root}")
        self._append_log(f"Command: {shlex.join(command)}")

        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self.process = None
            self._set_running_state(False)
            self.status_text.set("Could not start training")
            messagebox.showerror("Launch error", str(error))
            return

        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.output_queue.put(("line", line.rstrip()))
            return_code = process.wait()
            self.output_queue.put(("done", return_code))
        except OSError as error:
            self.output_queue.put(("error", str(error)))

    def _poll_output(self) -> None:
        try:
            while True:
                event, value = self.output_queue.get_nowait()
                if event == "line":
                    self._handle_output_line(str(value))
                elif event == "done":
                    self._training_finished(int(value))
                elif event == "error":
                    self._append_log(f"Output error: {value}")
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _handle_output_line(self, line: str) -> None:
        self._append_log(line)
        match = re.search(r"\\bstep\\s+(\\d+)\\s+\\|\\s+loss\\s+([0-9.eE+-]+)", line)
        if match:
            self.current_step = int(match.group(1))
            self.progress.configure(value=self.current_step)
            self.progress_text.set(
                f"Step {self.current_step} / {self.settings['max_steps'].get()}   Loss: {match.group(2)}"
            )
        validation = re.search(r"val loss:\\s*([0-9.eE+-]+)", line)
        if validation:
            self.progress_text.set(
                f"Step {self.current_step} / {self.settings['max_steps'].get()}   Validation loss: {validation.group(1)}"
            )

    def _training_finished(self, return_code: int) -> None:
        self.process = None
        self._set_running_state(False)
        if self.stop_requested:
            self.status_text.set("Training stopped")
            self.progress_text.set(f"Stopped at step {self.current_step}")
            self._append_log("Training stopped by the user.")
        elif return_code == 0:
            self.status_text.set("Training completed")
            self.progress_text.set("Final checkpoint saved by train.py")
            self._append_log("Training completed successfully.")
        else:
            self.status_text.set("Training failed")
            self.progress_text.set(f"Process exited with code {return_code}")
            self._append_log(f"Training process exited with code {return_code}.")

    def _stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stop_requested = True
        self.status_text.set("Stopping training...")
        self.progress_text.set("Waiting for the training process to stop")
        self._append_log("Stop requested by the user.")
        try:
            self.process.terminate()
        except OSError as error:
            self._append_log(f"Could not stop process: {error}")

    def _set_running_state(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            should_close = messagebox.askyesno(
                "Training is running",
                "Training is still running. Stop it and close Beq Trainer?",
            )
            if not should_close:
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
