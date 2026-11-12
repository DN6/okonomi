"""Terminal tool for executing shell commands.

Provides local command execution with timeout handling, output truncation,
and working directory support. Designed for use within okonomi's code
execution environment.
"""

import itertools
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from okonomi.config import OKONOMI_HOME
from okonomi.tools._command_guard import check_command
from okonomi.tools._sandbox import wrap_command
from okonomi.utils import read_env_file, redact_secrets, scrub_env


MAX_OUTPUT_CHARS = 50_000
HEAD_RATIO = 0.4
_bg_counter = itertools.count(1)

# Every bash subprocess alive in this process, across all TerminalTools
# instances (the foreground agent, detached tool calls, term-N tasks, and
# sub-agents' own terminals). Each runs in its own process group via setsid,
# so without this the REPL exiting would orphan them.
_live_procs: set[subprocess.Popen] = set()
_live_procs_lock = threading.Lock()


def kill_all_processes() -> int:
    """Shutdown hook: SIGTERM (then SIGKILL) every live bash subprocess.
    Returns how many were killed."""
    with _live_procs_lock:
        procs = list(_live_procs)
    for proc in procs:
        TerminalTools._kill_process(proc)
    return len(procs)


def _find_bash() -> str:
    bash = shutil.which("bash")
    if bash:
        return bash
    for path in ("/usr/bin/bash", "/bin/bash", "/usr/local/bin/bash"):
        if os.path.isfile(path):
            return path
    return "bash"


def _truncate_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    head_len = int(MAX_OUTPUT_CHARS * HEAD_RATIO)
    tail_len = MAX_OUTPUT_CHARS - head_len
    return (
        output[:head_len]
        + f"\n\n... [truncated {len(output) - MAX_OUTPUT_CHARS:,} chars] ...\n\n"
        + output[-tail_len:]
    )


def _drain_pipe(pipe, chunks: list) -> None:
    try:
        while True:
            line = pipe.readline()
            if not line:
                break
            chunks.append(line)
    except Exception:
        pass


_MATCH_RE = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")


class TerminalTools:
    """Tools for executing shell commands on the local machine."""


    def __init__(self):
        self._cwd = os.getcwd()
        self._bash = _find_bash()
        self._interrupt = threading.Event()
        self._has_rg = shutil.which("rg") is not None
        self._on_bg_complete: Any | None = None
        # Hook called when a background task (term-N) starts. Receives
        # (task_id, label, reader) where reader is a zero-arg callable
        # returning the current stdout snapshot. Set by the agent so its
        # /tasks picker can find term-N output.
        self._on_bg_register: Any | None = None
        # Hook called when a background task finishes. Receives
        # (task_id, status) where status is "ok" or "error".
        self._on_bg_finish: Any | None = None
        # Zero-arg callables returning the current session's artifacts and
        # scratch dirs (str), exported to children as $OKONOMI_SESSION_DIR
        # and $OKONOMI_SCRATCH_DIR. Set by the agent.
        self._session_dir_fn: Any | None = None
        self._scratch_dir_fn: Any | None = None

    def _kill_foreground(self) -> None:
        """Kill the foreground command, if one is running (Ctrl+C)."""
        self._interrupt.set()

    def _make_env(self) -> dict:
        env = scrub_env(os.environ)
        env.update(read_env_file())
        env["OKONOMI_HOME"] = str(OKONOMI_HOME)
        for var, fn in (("OKONOMI_SESSION_DIR", self._session_dir_fn), ("OKONOMI_SCRATCH_DIR", self._scratch_dir_fn)):
            if fn is None:
                continue
            try:
                env[var] = fn()
            except Exception:
                pass
        if "/usr/local/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/usr/local/bin:{env.get('PATH', '/usr/bin:/bin')}"
        return env

    def bash(
        self,
        command: str,
        timeout: int = 120,
        workdir: str | None = None,
        background: bool = False,
        _chunks_sink: list | None = None,
        _kill: threading.Event | None = None,
        _approved: bool = False,
    ) -> dict:
        """Execute a shell command and return its output.

        Runs the command in a bash shell with merged stdout/stderr.
        The working directory persists between calls.

        Args:
            command: Shell command to execute.
            timeout: Maximum seconds to wait (default 120). Ignored when
                ``background=True``.
            workdir: Working directory override. If not set, uses the
                directory from the previous call.
            background: When True, run the command in a thread and
                return immediately with ``{"task_id": "term-N", "message":
                ...}``. Output and exit code surface as a system note on
                completion. Use this
                for long downloads (yt-dlp), renders (ffmpeg), or any
                command that would otherwise stall the agent loop.

        Returns:
            Sync mode: dict with 'output' (str), 'exit_code' (int), and 'cwd' (str).
            Background mode: dict with 'task_id' (str) and 'message' (str).
        """
        allowed, reason = check_command(command, approved=_approved)
        if not allowed:
            return {
                "output": reason,
                "exit_code": 126,
                "cwd": self._cwd,
            }

        if background:
            return self._bash_background(command, workdir, _approved)

        if workdir:
            cwd = str(Path(workdir).expanduser().resolve())
        else:
            cwd = self._cwd

        if not os.path.isdir(cwd):
            return {
                "output": f"Directory not found: {cwd}",
                "exit_code": 1,
                "cwd": self._cwd,
            }

        # Only foreground calls honour Ctrl+C; background (term-N) tasks
        # share this instance and must survive a cancelled turn.
        foreground = _chunks_sink is None
        if foreground:
            self._interrupt.clear()

        wrapped = f'{command}\n__exit_code=$?\nprintf "\\n__CWD=%s" "$(pwd)"\nexit $__exit_code'

        try:
            proc = subprocess.Popen(
                wrap_command([self._bash, "-lc", wrapped], cwd=cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env=self._make_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except Exception as e:
            return {
                "output": f"Failed to start process: {e}",
                "exit_code": -1,
                "cwd": self._cwd,
            }

        with _live_procs_lock:
            _live_procs.add(proc)
        chunks: list[str] = _chunks_sink if _chunks_sink is not None else []
        drain = threading.Thread(target=_drain_pipe, args=(proc.stdout, chunks), daemon=True)
        drain.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        interrupted = False

        try:
            while proc.poll() is None:
                if (foreground and self._interrupt.is_set()) or (_kill is not None and _kill.is_set()):
                    interrupted = True
                    self._kill_process(proc)
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    self._kill_process(proc)
                    break
                time.sleep(0.1)
        finally:
            with _live_procs_lock:
                _live_procs.discard(proc)

        drain.join(timeout=2)
        output = "".join(chunks)

        new_cwd = cwd
        if "__CWD=" in output:
            parts = output.rsplit("__CWD=", 1)
            output = parts[0].rstrip()
            new_cwd = parts[1].strip()
            if os.path.isdir(new_cwd):
                self._cwd = new_cwd

        output = _truncate_output(redact_secrets(output))

        if timed_out:
            output += f"\n[Timed out after {timeout}s]"
            exit_code = 124
        elif interrupted:
            output += "\n[Interrupted]"
            exit_code = 130
        else:
            exit_code = proc.returncode or 0

        return {
            "output": output,
            "exit_code": exit_code,
            "cwd": self._cwd,
        }

    def _bash_background(self, command: str, workdir: str | None, approved: bool) -> dict:
        """Spawn a command in a thread and notify on completion.

        Returns immediately. The thread runs the same code path as a
        synchronous ``bash()`` (capturing output, exit code, cwd update)
        and fires ``self._on_bg_complete`` when done (the agent wires it
        to emit a ``TaskFinished`` event; subscribers react from there).
        """
        task_id = f"term-{next(_bg_counter)}"
        # Live chunks list — populated incrementally by the drain thread
        # inside `bash()`. Snapshotting via "".join(chunks) is what makes
        # /tasks shows output while the command is still running.
        chunks: list[str] = []

        def _snapshot() -> str:
            joined = "".join(chunks)
            # Strip the internal cwd marker bash() appends to detect cwd
            # changes — it's not user-visible output.
            if "__CWD=" in joined:
                joined = joined.rsplit("__CWD=", 1)[0]
            return redact_secrets(joined) or "(no output yet)"

        if self._on_bg_register is not None:
            try:
                self._on_bg_register(task_id=task_id, label=command[:60], reader=_snapshot)
            except Exception:
                pass

        def _worker() -> None:
            result = self.bash(command, timeout=3600, workdir=workdir, _chunks_sink=chunks, _approved=approved)
            output = result.get("output", "")
            exit_code = result.get("exit_code", 0)
            error = None if exit_code == 0 else f"exit_code={exit_code}"
            if self._on_bg_finish is not None:
                try:
                    self._on_bg_finish(task_id=task_id, status="error" if error else "ok")
                except Exception:
                    pass
            if self._on_bg_complete is not None:
                try:
                    self._on_bg_complete(
                        bg_id=task_id,
                        error=error,
                        output=output,
                        produced_artifacts=[],
                    )
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True, name=f"okonomi-{task_id}").start()
        return {
            "task_id": task_id,
            "message": (
                f"Started in background as {task_id}. Output and exit code "
                "will arrive in a system note when the command finishes."
            ),
        }

    def _quote(self, s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)
        return path

    def search(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ) -> dict:
        """Search for content inside files using regex patterns.

        Uses ripgrep when available, falls back to grep.

        Args:
            pattern: Regex pattern to search for.
            path: Directory or file to search in (default: current directory).
            file_glob: Filter files by glob pattern (e.g. '*.py', '*.ts').
            limit: Maximum number of results to return (default: 50).
            offset: Skip the first N results for pagination.
            output_mode: 'content' shows matching lines with line numbers,
                'files_only' shows only file paths, 'count' shows match
                counts per file.
            context: Number of context lines around each match.

        Returns:
            Dict with 'matches' or 'files' or 'counts', plus 'total_count'.
        """
        resolved = self._resolve_path(path)
        if not os.path.exists(resolved):
            return {"error": f"Path not found: {resolved}", "total_count": 0}

        if self._has_rg:
            return self._search_rg(pattern, resolved, file_glob, limit, offset, output_mode, context)
        return self._search_grep(pattern, resolved, file_glob, limit, offset, output_mode, context)

    def _search_rg(self, pattern, path, file_glob, limit, offset, output_mode, context):
        parts = ["rg", "--line-number", "--no-heading", "--with-filename"]

        if context > 0:
            parts.extend(["-C", str(context)])
        if file_glob:
            parts.extend(["--glob", self._quote(file_glob)])

        if output_mode == "files_only":
            parts.append("-l")
        elif output_mode == "count":
            parts.append("-c")

        parts.append(self._quote(pattern))
        parts.append(self._quote(path))

        fetch = limit + offset + (200 if context > 0 else 0)
        parts.extend(["|", "head", "-n", str(fetch)])

        result = self.bash(" ".join(parts), timeout=60)
        stdout = result["output"]

        if result["exit_code"] == 2 and not stdout.strip():
            return {"error": "Search failed", "total_count": 0}
        if not stdout.strip():
            return self._empty_result(output_mode)

        return self._parse_search_output(stdout, output_mode, limit, offset)

    def _search_grep(self, pattern, path, file_glob, limit, offset, output_mode, context):
        parts = ["grep", "-rnH", "--exclude-dir='.*'"]

        if context > 0:
            parts.extend(["-C", str(context)])
        if file_glob:
            parts.extend(["--include", self._quote(file_glob)])

        if output_mode == "files_only":
            parts.append("-l")
        elif output_mode == "count":
            parts.append("-c")

        parts.append(self._quote(pattern))
        parts.append(self._quote(path))
        parts.extend(["|", "head", "-n", str(limit + offset)])

        result = self.bash(" ".join(parts), timeout=60)
        stdout = result["output"]

        if not stdout.strip():
            return self._empty_result(output_mode)

        return self._parse_search_output(stdout, output_mode, limit, offset)

    def _parse_search_output(self, stdout, output_mode, limit, offset):
        if output_mode == "files_only":
            all_files = [f for f in stdout.strip().split("\n") if f]
            return {
                "files": all_files[offset : offset + limit],
                "total_count": len(all_files),
            }

        if output_mode == "count":
            counts = {}
            for line in stdout.strip().split("\n"):
                if ":" in line:
                    file_path, _, count_str = line.rpartition(":")
                    try:
                        counts[file_path] = int(count_str)
                    except ValueError:
                        pass
            return {"counts": counts, "total_count": sum(counts.values())}

        matches = []
        for line in stdout.strip().split("\n"):
            if not line or line == "--":
                continue
            m = _MATCH_RE.match(line)
            if m:
                matches.append(
                    {
                        "path": (m.group(1) or "") + m.group(2),
                        "line": int(m.group(3)),
                        "content": m.group(4)[:500],
                    }
                )

        total = len(matches)
        page = matches[offset : offset + limit]
        result = {"matches": page, "total_count": total}
        if total > offset + limit:
            result["truncated"] = True
            result["hint"] = f"Use offset={offset + limit} to see more results."
        return result

    def _empty_result(self, output_mode):
        if output_mode == "files_only":
            return {"files": [], "total_count": 0}
        if output_mode == "count":
            return {"counts": {}, "total_count": 0}
        return {"matches": [], "total_count": 0}

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=2)
