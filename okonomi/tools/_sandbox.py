"""OS sandbox for model-run subprocesses: Seatbelt on macOS, bubblewrap on Linux.

``wrap_command`` prepends the platform runner with a write-scoping profile:
reads, network, and exec stay open; file writes are limited to
$OKONOMI_HOME, the temp dirs, and the current working directory.

The first call probes the runner. If it is missing or cannot apply the
profile, or the platform has no runner, ``SandboxUnavailable`` is raised
and the command does not run: there is no unsandboxed fallback.
``bwrap`` is used from the system rather than bundled.
"""

import os
import subprocess
import sys
import tempfile

from okonomi.config import OKONOMI_HOME


class SandboxUnavailable(EnvironmentError):
    pass


_probed = False


def _seatbelt_prefix(cwd: str) -> list[str]:
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
    profile = f"(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write*\n{allow})\n"
    return ["sandbox-exec", "-p", profile]


def _bwrap_prefix(cwd: str) -> list[str]:
    # Read-only view of the whole filesystem, then writable binds over the
    # scoped roots (later binds win). The real /dev is bound so ttys,
    # /dev/null and /dev/shm behave as outside. --bind-try skips roots that
    # do not exist yet, such as a fresh $OKONOMI_HOME.
    args = ["bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev"]
    for path in (str(OKONOMI_HOME), tempfile.gettempdir(), "/tmp", cwd):
        args += ["--bind-try", path, path]
    return [*args, "--"]


_RUNNERS = {
    "darwin": (_seatbelt_prefix, "sandbox-exec ships with macOS; check that /usr/bin/sandbox-exec is present and runnable."),
    "linux": (
        _bwrap_prefix,
        "Install bubblewrap (`sudo apt install bubblewrap` or `sudo dnf install bubblewrap`) and make sure "
        "unprivileged user namespaces are enabled (on Ubuntu 24.04+: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`).",
    ),
}


def _probe(prefix: list[str], hint: str) -> None:
    try:
        result = subprocess.run([*prefix, "true"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        raise SandboxUnavailable(f"Cannot start the sandbox runner {prefix[0]!r}: {e}. {hint}") from e
    if result.returncode != 0:
        raise SandboxUnavailable(
            f"The sandbox runner {prefix[0]!r} failed its probe (exit {result.returncode}): {result.stderr.strip()}. {hint}"
        )


def wrap_command(argv: list[str], cwd: str | None = None) -> list[str]:
    """Wrap ``argv`` in the platform sandbox. Raises ``SandboxUnavailable``
    when there is none, instead of running unsandboxed."""
    global _probed
    runner = _RUNNERS.get(sys.platform)
    if runner is None:
        raise SandboxUnavailable(f"okonomi only sandboxes commands on macOS and Linux, not {sys.platform!r}.")
    prefix_fn, hint = runner
    prefix = prefix_fn(cwd or os.getcwd())
    if not _probed:
        _probe(prefix, hint)
        _probed = True
    return [*prefix, *argv]
