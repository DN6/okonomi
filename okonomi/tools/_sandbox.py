"""macOS Seatbelt wrapper for model-run subprocesses.

``wrap_command`` prepends ``sandbox-exec`` with a write-scoping profile:
reads, network, and exec stay open; file writes are limited to
$OKONOMI_HOME, the temp dirs, and the current working directory.

``sandbox-exec`` is deprecated-but-functional (the kernel mechanism is
load-bearing for macOS itself), so treat it as best-effort hardening:
the first call probes it and any failure degrades to unwrapped
execution with a single logged warning. Non-darwin platforms and
``OKONOMI_NO_SANDBOX=1`` pass through unchanged.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile

from okonomi.config import OKONOMI_HOME


logger = logging.getLogger(__name__)

# None = not probed yet; True/False after the first wrap_command call.
_sandbox_ok: bool | None = None


def _build_profile(cwd: str) -> str:
    # Later rules win in SBPL: allow everything, deny all writes, then
    # re-allow writes under the scoped roots. /private/var/folders is
    # the real location behind tempfile on macOS; /dev for tty/null.
    subpaths = [
        str(OKONOMI_HOME),
        tempfile.gettempdir(),
        cwd,
        "/private/tmp",
        "/private/var/folders",
        "/dev",
    ]
    allow = "\n".join(f'    (subpath "{p}")' for p in subpaths)
    return f"(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write*\n{allow})\n"


def _probe(profile: str) -> bool:
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", profile, "/usr/bin/true"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def wrap_command(argv: list[str], cwd: str | None = None) -> list[str]:
    """Wrap ``argv`` in sandbox-exec when available; pass through otherwise."""
    global _sandbox_ok
    if sys.platform != "darwin" or os.getenv("OKONOMI_NO_SANDBOX"):
        return argv
    if shutil.which("sandbox-exec") is None:
        return argv

    profile = _build_profile(cwd or os.getcwd())
    if _sandbox_ok is None:
        _sandbox_ok = _probe(profile)
        if not _sandbox_ok:
            logger.warning("sandbox-exec probe failed — running commands without a sandbox.")
    if not _sandbox_ok:
        return argv
    return ["sandbox-exec", "-p", profile, *argv]
