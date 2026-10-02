"""Manual, visibly controlled desktop agent entry points."""

from .controller import AgentController, AgentStatus
from .hotkeys import HotkeySpec, WindowsHotkeyListener

__all__ = ["AgentController", "AgentStatus", "HotkeySpec", "WindowsHotkeyListener"]
