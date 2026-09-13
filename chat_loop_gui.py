"""Tkinter controller that repeatedly runs generate.py and writes chat-loop.log."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "generate.py"
LOG_PATH = ROOT / "chat-loop.log"


class ChatLoopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Beq Chat Loop")
        self.root.geometry("900x650")
        self.root.minsize(720, 500)

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.process_lock = threading.Lock()
        self.closing = False
        self.paused = False
        self.run_number = 0

        self.prompt_var = tk.StringVar(value="Once upon a time")
        self.interval_var = tk.StringVar(value="10")
        self.max_tokens_var = tk.StringVar(value="150")
        self.temperature_var = tk.StringVar(value="0.8")
        self.status_var = tk.StringVar(value="Bereit")
        self.log_path_var = tk.StringVar(value=str(LOG_PATH))

        self._build_ui()
        self._load_existing_log()
        self.root.after(100, self._drain_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(5, weight=1)

        ttk.Label(outer, text="Prompt:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        prompt_entry = ttk.Entry(outer, textvariable=self.prompt_var)
        prompt_entry.grid(row=0, column=1, columnspan=4, sticky="ew", pady=4)

        ttk.Label(outer, text="Intervall (Sekunden):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(outer, textvariable=self.interval_var, width=12).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(outer, text="Max. Tokens:").grid(row=1, column=2, sticky="e", padx=(16, 8), pady=4)
        ttk.Entry(outer, textvariable=self.max_tokens_var, width=12).grid(row=1, column=3, sticky="w", pady=4)
        ttk.Label(outer, text="Temperatur:").grid(row=1, column=4, sticky="e", padx=(16, 8), pady=4)
        ttk.Entry(outer, textvariable=self.temperature_var, width=10).grid(row=1, column=5, sticky="w", pady=4)

        buttons = ttk.Frame(outer)
        buttons.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 4))
        self.start_button = ttk.Button(buttons, text="Start", command=self._start)
        self.start_button.pack(side=tk.LEFT, padx=(0, 6))
        self.pause_button = ttk.Button(buttons, text="Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Log leeren", command=self._clear_log).pack(side=tk.LEFT, padx=(18, 6))
        ttk.Button(buttons, text="Log öffnen", command=self._open_log).pack(side=tk.LEFT, padx=6)

        ttk.Label(outer, textvariable=self.status_var).grid(row=3, column=0, columnspan=6, sticky="w", pady=(4, 2))
        ttk.Label(outer, textvariable=self.log_path_var, foreground="#666666").grid(row=4, column=0, columnspan=6, sticky="w", pady=(0, 6))

        log_frame = ttk.Frame(outer)
        log_frame.grid(row=5, column=0, columnspan=6, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.output = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, undo=False)
        self.output.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

        prompt_entry.focus_set()

    def _load_existing_log(self) -> None:
        if not LOG_PATH.exists():
            return
        try:
            content = LOG_PATH.read_text(encoding="utf-8", errors="replace")
            self._append_text(content[-100_000:])
        except OSError as exc:
            self._append_text(f"Log konnte nicht gelesen werden: {exc}\n")

    def _append_text(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _emit(self, kind: str, value: object) -> None:
        self.ui_queue.put((kind, value))

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "text":
                    self._append_text(str(value))
                elif kind == "status":
                    self.status_var.set(str(value))
                elif kind == "finished":
                    self._loop_finished()
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(100, self._drain_ui_queue)

    def _write_log(self, message: str) -> None:
        lines = message.rstrip("\n").splitlines() or [""]
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            for line in lines:
                stamped = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {line}\n"
                log_file.write(stamped)
                self._emit("text", stamped)

    def _read_settings(self) -> dict[str, object] | None:
        try:
            interval = float(self.interval_var.get().strip())
            max_tokens = int(self.max_tokens_var.get().strip())
            temperature = float(self.temperature_var.get().strip())
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Intervall, Max. Tokens und Temperatur müssen Zahlen sein.")
            return None
        if interval < 0 or max_tokens < 1 or not 0.0 <= temperature <= 2.0:
            messagebox.showerror(
                "Ungültige Eingabe",
                "Intervall muss >= 0, Max. Tokens muss > 0 und Temperatur muss zwischen 0 und 2 liegen.",
            )
            return None
        if not self.prompt_var.get().strip():
            messagebox.showerror("Ungültige Eingabe", "Der Prompt darf nicht leer sein.")
            return None
        if not SCRIPT.exists():
            messagebox.showerror("generate.py fehlt", f"Nicht gefunden:\n{SCRIPT}")
            return None
        return {
            "prompt": self.prompt_var.get(),
            "interval": interval,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        settings = self._read_settings()
        if settings is None:
            return
        self.stop_event.clear()
        self.resume_event.set()
        self.paused = False
        self.run_number = 0
        self.worker = threading.Thread(target=self._run_loop, args=(settings,), daemon=True)
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL, text="Pause")
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Starte Loop …")

    def _toggle_pause(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        if self.paused:
            self.paused = False
            self.resume_event.set()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Loop fortgesetzt")
            self._write_log("Loop fortgesetzt")
        else:
            self.paused = True
            self.resume_event.clear()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Pausiert – der aktuelle Durchlauf darf noch fertig werden")
            self._write_log("Pause angefordert; kein neuer Durchlauf startet")

    def _stop(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        self.stop_event.set()
        self.resume_event.set()
        self.status_var.set("Stoppe …")
        self._write_log("Stop angefordert")
        self._terminate_process()
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.DISABLED)

    def _terminate_process(self) -> None:
        with self.process_lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except OSError:
            pass

    def _run_loop(self, settings: dict[str, object]) -> None:
        interval = float(settings["interval"])
        max_tokens = int(settings["max_tokens"])
        temperature = float(settings["temperature"])
        prompt = str(settings["prompt"])

        try:
            while not self.stop_event.is_set():
                if not self.resume_event.wait(0.2):
                    continue
                if self.stop_event.is_set():
                    break

                self.run_number += 1
                run = self.run_number
                command = [
                    sys.executable,
                    "-u",
                    str(SCRIPT),
                    "--prompt",
                    prompt,
                    "--max_tokens",
                    str(max_tokens),
                    "--temperature",
                    str(temperature),
                ]
                self._write_log(f"===== Durchlauf {run} gestartet =====")
                self._write_log("Befehl: " + subprocess.list2cmdline(command))
                self._emit("status", f"Läuft – Durchlauf {run}")

                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                with self.process_lock:
                    self.process = process

                if process.stdout is not None:
                    for line in process.stdout:
                        self._write_log(line.rstrip("\n"))
                return_code = process.wait()
                with self.process_lock:
                    self.process = None
                self._write_log(f"===== Durchlauf {run} beendet (Exit-Code {return_code}) =====")

                if self.stop_event.is_set():
                    break
                if not self.resume_event.is_set():
                    self._emit("status", "Pausiert")
                    continue
                if interval > 0:
                    self._emit("status", f"Warte {interval:g} Sekunden bis zum nächsten Durchlauf …")
                    end = time.monotonic() + interval
                    while time.monotonic() < end and not self.stop_event.is_set():
                        if not self.resume_event.wait(0.2):
                            self._emit("status", "Pausiert")
                            break
        except Exception:
            error = traceback.format_exc()
            self._write_log("Unerwarteter Fehler:\n" + error)
            self._emit("status", "Fehler – siehe chat-loop.log")
        finally:
            with self.process_lock:
                self.process = None
            self._emit("finished", None)

    def _loop_finished(self) -> None:
        if self.closing:
            return
        self.worker = None
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED, text="Pause")
        self.stop_button.configure(state=tk.DISABLED)
        if self.stop_event.is_set():
            self.status_var.set("Gestoppt")
        else:
            self.status_var.set("Beendet")

    def _clear_log(self) -> None:
        if not messagebox.askyesno("Log leeren", "chat-loop.log wirklich leeren?"):
            return
        try:
            LOG_PATH.write_text("", encoding="utf-8")
            self.output.configure(state=tk.NORMAL)
            self.output.delete("1.0", tk.END)
            self.output.configure(state=tk.DISABLED)
            self.status_var.set("Log geleert")
        except OSError as exc:
            messagebox.showerror("Fehler", f"Log konnte nicht geleert werden:\n{exc}")

    def _open_log(self) -> None:
        LOG_PATH.touch(exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(LOG_PATH)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(LOG_PATH)])
            else:
                subprocess.Popen(["xdg-open", str(LOG_PATH)])
        except OSError as exc:
            messagebox.showerror("Fehler", f"Log konnte nicht geöffnet werden:\n{exc}")

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Beenden", "Loop stoppen und Fenster schließen?"):
                return
            self.closing = True
            self.stop_event.set()
            self.resume_event.set()
            self._terminate_process()
            self._wait_for_worker()
        else:
            self.root.destroy()

    def _wait_for_worker(self) -> None:
        if self.worker and self.worker.is_alive():
            self.root.after(100, self._wait_for_worker)
        else:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ChatLoopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
