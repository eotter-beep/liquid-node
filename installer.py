"""Utility window that helps users install missing dependencies.

The main launcher checks for external requirements such as the Docker CLI or
Python packages. When one or more dependencies are unavailable the
``InstallerWindow`` pops up automatically and guides the user through the
resolution. Python dependencies can be installed directly via ``pip`` while
system dependencies direct the user to the appropriate documentation.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


@dataclass(slots=True)
class MissingDependency:
    """Description for a dependency that has not been satisfied."""

    name: str
    description: str
    install_command: Sequence[str] | None = None
    help_url: str | None = None


class InstallerWindow(tk.Toplevel):
    """Displays missing dependencies and assists with installation."""

    def __init__(
        self,
        parent: tk.Misc,
        missing: Iterable[MissingDependency],
        *,
        on_refresh: Callable[[], Iterable[MissingDependency]],
        brand_background: str,
    ) -> None:
        super().__init__(parent)
        self.title("Liquid Node Installer")
        self.resizable(False, False)
        self.configure(bg=brand_background)

        self._on_refresh = on_refresh
        self._pending_thread: threading.Thread | None = None
        self._missing: List[MissingDependency] = []

        self._status = tk.StringVar(value="Resolve the missing components below.")

        container = ttk.Frame(self, style="Brand.TFrame", padding="24 24 24 24")
        container.pack(fill="both", expand=True)

        title = ttk.Label(
            container,
            text="Dependencies required",
            style="StatusInfo.TLabel",
        )
        title.pack(anchor="w")

        status_label = ttk.Label(
            container,
            textvariable=self._status,
            style="StatusProgress.TLabel",
            wraplength=420,
            justify="left",
        )
        status_label.pack(anchor="w", pady=(4, 16))

        self._dependency_container = ttk.Frame(container, style="Brand.TFrame")
        self._dependency_container.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        self.refresh(missing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, missing: Iterable[MissingDependency]) -> None:
        """Re-render the dependency list with the provided items."""

        self._missing = list(missing)
        for child in self._dependency_container.winfo_children():
            child.destroy()

        if not self._missing:
            self._status.set("All dependencies satisfied. You may close this window.")
            self.after(1500, self._handle_close)
            return

        for dependency in self._missing:
            card = ttk.Frame(
                self._dependency_container,
                style="Surface.TFrame",
                padding="16 16 16 16",
            )
            card.pack(fill="x", pady=(0, 12))

            header = ttk.Label(
                card,
                text=dependency.name,
                style="StatusSuccess.TLabel",
            )
            header.pack(anchor="w")

            description = ttk.Label(
                card,
                text=dependency.description,
                style="OutputInfo.TLabel",
                wraplength=360,
                justify="left",
            )
            description.pack(anchor="w", pady=(6, 8))

            actions = ttk.Frame(card, style="Surface.TFrame")
            actions.pack(anchor="w")

            if dependency.install_command:
                install_button = ttk.Button(
                    actions,
                    text=f"Install {dependency.name}",
                    style="Primary.TButton",
                    command=lambda dep=dependency: self._install_dependency(dep),
                )
                install_button.pack(side="left")

            if dependency.help_url:
                help_button = ttk.Button(
                    actions,
                    text="View instructions",
                    style="Secondary.TButton",
                    command=lambda url=dependency.help_url: webbrowser.open(url),
                )
                help_button.pack(side="left", padx=(12, 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_close(self) -> None:
        if self._pending_thread and self._pending_thread.is_alive():
            messagebox.showwarning(
                "Installer busy",
                "Please wait for the current installation to complete before closing.",
                parent=self,
            )
            return
        try:
            self.grab_release()
        except tk.TclError:  # pragma: no cover - depends on window manager state
            pass
        self.withdraw()

    def _install_dependency(self, dependency: MissingDependency) -> None:
        if self._pending_thread and self._pending_thread.is_alive():
            messagebox.showinfo(
                "Installation running",
                "An installation is already in progress.",
                parent=self,
            )
            return

        if not dependency.install_command:
            return

        self._status.set(f"Installing {dependency.name}…")

        def worker() -> None:
            try:
                subprocess.run(
                    list(dependency.install_command),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:  # pragma: no cover - GUI feedback only
                self.after(
                    0,
                    lambda: self._on_install_failed(
                        dependency, exc.stdout or exc.stderr or "Unknown error"
                    ),
                )
                return

            self.after(0, lambda: self._on_install_finished(dependency))

        self._pending_thread = threading.Thread(target=worker, daemon=True)
        self._pending_thread.start()

    def _on_install_finished(self, dependency: MissingDependency) -> None:
        self._status.set(f"{dependency.name} installed. Re-checking dependencies…")
        self._refresh_missing()

    def _on_install_failed(self, dependency: MissingDependency, log: str) -> None:
        self._status.set(f"Installation failed for {dependency.name}. See terminal for details.")
        messagebox.showerror(
            f"Failed to install {dependency.name}",
            f"The installer was unable to install {dependency.name}.\n\nDetails:\n{log}",
            parent=self,
        )
        self._refresh_missing()

    def _refresh_missing(self) -> None:
        updated_missing = list(self._on_refresh())
        if not updated_missing:
            self._status.set("All dependencies installed. Enjoy Liquid Node!")
            self.after(1200, self._handle_close)
        self.refresh(updated_missing)


def build_python_install_command(package: str) -> Sequence[str]:
    """Return a ``pip install`` command for the active interpreter."""

    return (sys.executable, "-m", "pip", "install", package)

