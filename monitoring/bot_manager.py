"""
monitoring/bot_manager.py — Manage the trading bot subprocess.

Provides helpers to start, stop, and check the trading bot from the
Streamlit dashboard.  The bot runs as a detached subprocess whose PID
is persisted to a small file so the dashboard can survive reruns.
"""
import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Project root (one level up from monitoring/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PID_FILE = _PROJECT_ROOT / ".bot.pid"
_LOG_FILE = _PROJECT_ROOT / "reports" / "bot_output.log"


def _python_exe() -> str:
    """Return the path to the current Python interpreter."""
    return sys.executable


def is_running() -> bool:
    """Return True if the trading bot process is still alive."""
    pid = _read_pid()
    if pid is None:
        return False
    try:
        # On Windows, signal 0 doesn't exist, so use os.kill differently
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        # Process no longer exists — clean up stale PID file
        _remove_pid()
        return False


def start() -> tuple[bool, str]:
    """
    Launch ``scripts/run_bot.py`` as a background subprocess.

    Returns:
        (success, message)
    """
    if is_running():
        return False, "Trading bot is already running."

    os.makedirs(_LOG_FILE.parent, exist_ok=True)

    log_fh = open(_LOG_FILE, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [_python_exe(), "-u", str(_PROJECT_ROOT / "scripts" / "run_bot.py")],
            cwd=str(_PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            # On Windows, CREATE_NEW_PROCESS_GROUP lets us kill it later;
            # on Unix we just use a new process group.
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32"
                else 0
            ),
        )
    except Exception as exc:
        log_fh.close()
        return False, f"Failed to start bot: {exc}"

    _write_pid(proc.pid)
    return True, f"Trading bot started (PID {proc.pid})."


def stop() -> tuple[bool, str]:
    """
    Terminate the running trading bot.

    Returns:
        (success, message)
    """
    pid = _read_pid()
    if pid is None:
        return False, "No trading bot is running."

    try:
        if sys.platform == "win32":
            # taskkill forcefully terminates the process tree
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid), "/T"],
                capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            # Give it a moment to shut down gracefully
            time.sleep(2)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except (OSError, ProcessLookupError):
        pass

    _remove_pid()
    return True, "Trading bot stopped."


def get_log(tail: int = 80) -> str:
    """Return the last *tail* lines of bot output."""
    if not _LOG_FILE.exists():
        return ""
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except Exception:
        return ""


# ── Internal helpers ──────────────────────────────────────────────────

def _write_pid(pid: int) -> None:
    _PID_FILE.write_text(str(pid), encoding="utf-8")


def _read_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _remove_pid() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass
