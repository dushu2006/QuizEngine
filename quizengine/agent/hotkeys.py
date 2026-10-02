"""Windows global hotkeys using the documented RegisterHotKey API.

This only waits for the operator's explicit start/stop keystrokes; it does not
poll or capture the screen in the background. There are no hooks, stealth, input
spoofing, or process-hiding paths.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional


_MODIFIERS = {"ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004, "win": 0x0008, "windows": 0x0008}


@dataclass(frozen=True)
class HotkeySpec:
    source: str
    modifiers: int
    virtual_key: int
    key_name: str

    @classmethod
    def parse(cls, value: str) -> "HotkeySpec":
        parts = [part.strip().lower() for part in (value or "").split("+") if part.strip()]
        if len(parts) < 2:
            raise ValueError("hotkey must include at least one modifier and one key, e.g. win+alt+q")
        key = parts[-1]
        modifier_names = parts[:-1]
        if len(set(modifier_names)) != len(modifier_names):
            raise ValueError(f"hotkey repeats a modifier: {value!r}")
        unknown = [name for name in modifier_names if name not in _MODIFIERS]
        if unknown:
            raise ValueError(f"unsupported hotkey modifier(s): {', '.join(unknown)}")
        if key in _MODIFIERS or len(key) != 1 or not key.isascii() or not key.isalnum():
            raise ValueError("hotkey key must be one ASCII letter or digit")
        modifiers = 0
        for name in modifier_names:
            modifiers |= _MODIFIERS[name]
        return cls(source="+".join(parts), modifiers=modifiers, virtual_key=ord(key.upper()), key_name=key)


class WindowsHotkeyListener:
    """Dedicated Win32 message-loop thread for start/stop global hotkeys."""

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    START_ID = 0x5145
    STOP_ID = 0x5146

    def __init__(self, start_hotkey: str, stop_hotkey: str, on_start: Callable[[], None], on_stop: Callable[[], None]) -> None:
        self.start = HotkeySpec.parse(start_hotkey)
        self.stop = HotkeySpec.parse(stop_hotkey)
        if (self.start.modifiers, self.start.virtual_key) == (self.stop.modifiers, self.stop.virtual_key):
            raise ValueError("start and stop hotkeys must differ")
        self.on_start = on_start
        self.on_stop = on_stop
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread_id: Optional[int] = None

    @staticmethod
    def supported() -> bool:
        return sys.platform == "win32"

    def start_listening(self, *, timeout: float = 3.0) -> None:
        if not self.supported():
            raise RuntimeError("global hotkeys require Windows RegisterHotKey; use the explicit CLI run command here")
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._message_loop, name="quizengine-hotkeys", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("timed out starting the Windows hotkey listener")
        if self._error:
            raise RuntimeError(f"could not register global hotkeys: {self._error}") from self._error

    def stop_listening(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread_id is not None and self.supported():
            try:
                ctypes.windll.user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def _message_loop(self) -> None:  # pragma: no cover - Windows-only integration
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        thread_id = int(kernel32.GetCurrentThreadId())
        self._thread_id = thread_id
        start_ok = bool(user32.RegisterHotKey(None, self.START_ID, self.start.modifiers, self.start.virtual_key))
        stop_ok = bool(user32.RegisterHotKey(None, self.STOP_ID, self.stop.modifiers, self.stop.virtual_key))
        if not start_ok or not stop_ok:
            if start_ok:
                user32.UnregisterHotKey(None, self.START_ID)
            if stop_ok:
                user32.UnregisterHotKey(None, self.STOP_ID)
            self._error = OSError("Windows rejected a hotkey; it may already be registered by another app")
            self._ready.set()
            self._thread_id = None
            return
        self._ready.set()
        message = wintypes.MSG()
        try:
            while not self._stop.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                if message.message == self.WM_HOTKEY:
                    if message.wParam == self.START_ID:
                        self.on_start()
                    elif message.wParam == self.STOP_ID:
                        self.on_stop()
        finally:
            user32.UnregisterHotKey(None, self.START_ID)
            user32.UnregisterHotKey(None, self.STOP_ID)
            self._thread_id = None


__all__ = ["HotkeySpec", "WindowsHotkeyListener"]
