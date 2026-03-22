"""Terminal tool for executing shell commands.

Provides local command execution with timeout handling, output truncation,
and working directory support. Designed for use within okonomi's code
execution environment.
"""

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from okonomi.tools._metadata import ToolMetadataMixin

MAX_OUTPUT_CHARS = 50_000
HEAD_RATIO = 0.4


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
            data = pipe.read(4096)
            if not data:
                break
            chunks.append(data)
    except Exception:
        pass


_MATCH_RE = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")


class TerminalTools(ToolMetadataMixin):
    """Tools for executing shell commands on the local machine."""

    __namespace__ = "terminal"
    __description__ = "Execute shell commands locally with timeout and output capture"

    def __init__(self, config, interrupt_event: Optional[threading.Event] = None):
        self.config = config
        self._cwd = os.getcwd()
        self._bash = _find_bash()
        self._interrupt = interrupt_event or threading.Event()
        self._has_rg = shutil.which("rg") is not None
        self._env_blocklist = {
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "HF_TOKEN",
            "FAL_KEY",
            "TAVILY_API_KEY",
        }

    def _make_env(self) -> dict:
        env = dict(os.environ)
        for key in self._env_blocklist:
            env.pop(key, None)
        if "/usr/local/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/usr/local/bin:{env.get('PATH', '/usr/bin:/bin')}"
        return env

    def run(
        self,
        command: str,
        timeout: int = 120,
        workdir: Optional[str] = None,
    ) -> dict:
        """Execute a shell command and return its output.

        Runs the command in a bash shell with merged stdout/stderr.
        The working directory persists between calls.

        Args:
            command: Shell command to execute.
            timeout: Maximum seconds to wait (default 120).
            workdir: Working directory override. If not set, uses the
                directory from the previous call.

        Returns:
            Dict with 'output' (str), 'exit_code' (int), and 'cwd' (str).
        """
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

        self._interrupt.clear()

        wrapped = f'{command}\n__exit_code=$?\nprintf "\\n__CWD=%s" "$(pwd)"\nexit $__exit_code'

        try:
            proc = subprocess.Popen(
                [self._bash, "-lc", wrapped],
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

        chunks: list[str] = []
        drain = threading.Thread(target=_drain_pipe, args=(proc.stdout, chunks), daemon=True)
        drain.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        interrupted = False

        while proc.poll() is None:
            if self._interrupt.is_set():
                interrupted = True
                self._kill_process(proc)
                break
            if time.monotonic() > deadline:
                timed_out = True
                self._kill_process(proc)
                break
            time.sleep(0.1)

        drain.join(timeout=2)
        output = "".join(chunks)

        new_cwd = cwd
        if "__CWD=" in output:
            parts = output.rsplit("__CWD=", 1)
            output = parts[0].rstrip()
            new_cwd = parts[1].strip()
            if os.path.isdir(new_cwd):
                self._cwd = new_cwd

        output = _truncate_output(output)

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

    def run_background(
        self,
        command: str,
        workdir: Optional[str] = None,
    ) -> dict:
        """Start a command in the background and return immediately.

        Args:
            command: Shell command to execute.
            workdir: Working directory override.

        Returns:
            Dict with 'pid' (int) and 'message' (str).
        """
        cwd = str(Path(workdir).expanduser().resolve()) if workdir else self._cwd

        try:
            proc = subprocess.Popen(
                [self._bash, "-lc", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env=self._make_env(),
                start_new_session=True,
            )
            return {
                "pid": proc.pid,
                "message": f"Started in background (PID {proc.pid})",
            }
        except Exception as e:
            return {
                "pid": -1,
                "message": f"Failed to start: {e}",
            }

    def _quote(self, s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)
        return path

    def search(
        self,
        pattern: str,
        path: str = ".",
        file_glob: Optional[str] = None,
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

        result = self.run(" ".join(parts), timeout=60)
        stdout = result["output"]

        if result["exit_code"] == 2 and not stdout.strip():
            return {"error": "Search failed", "total_count": 0}
        if not stdout.strip():
            return self._empty_result(output_mode)

        return self._parse_search_output(stdout, output_mode, limit, offset, context)

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

        result = self.run(" ".join(parts), timeout=60)
        stdout = result["output"]

        if not stdout.strip():
            return self._empty_result(output_mode)

        return self._parse_search_output(stdout, output_mode, limit, offset, context)

    def _parse_search_output(self, stdout, output_mode, limit, offset, context):
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
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            proc.wait(timeout=2)
