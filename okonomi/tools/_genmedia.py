"""Programmatic wrapper around the `genmedia` CLI.

Single namespace covers all 1000+ fal.ai endpoints — the model writes
synchronous Python (`genmedia.run(model_id, **kwargs)`) and the wrapper
handles subprocess invocation, JSON parsing, and asset auto-tracking.

Async at the okonomi level happens via bg-exec (auto-dispatched when
the agent sees `genmedia.` in an `execute_code` block). Python inside
the bg thread stays sync — we don't expose `submit` / `status` to keep
the surface tight.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TypedDict
from urllib.parse import urlparse

from okonomi.session import Artifact
from okonomi.tools._metadata import ToolMetadataMixin


class GenmediaResult(TypedDict, total=False):
    """Return type for genmedia.run.

    Shape is endpoint-specific — fal's universal asset envelope is
    ``{"url": "https://..."}``, but it's nested differently per model
    (``image``, ``video``, ``audio_file``, ``model_mesh``, ``layers``,
    ``images``). The wrapper unwraps the genmedia CLI envelope so you
    access the inner fal payload directly (`r["images"][0]["url"]`,
    not `r["result"]["images"][...]`).

    Common keys:
      - ``images``: list[{"url": str, ...}]    # most image endpoints
      - ``image``:  {"url": str, ...}          # upscale, segment
      - ``video``:  {"url": str, ...}          # video endpoints
      - ``audio_file``: {"url": str, ...}      # music / audio
      - ``model_mesh``: {"url": str, ...}      # 3d
      - ``error``:  str                        # only on failure
      - ``_status`` / ``_endpoint_id`` / ``_request_id``: envelope metadata

    URLs in the response are auto-tracked.
    """

    error: str
    raw: object  # populated when JSON parsing fails


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
_3D_EXTS = {".glb", ".gltf", ".obj", ".usdz", ".fbx", ".ply"}


def _infer_asset_type(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _3D_EXTS:
        return "3d"
    return "other"


def _walk_urls(payload: Any) -> Iterator[tuple[str, str]]:
    """Recursively yield (url, asset_type) pairs from a fal response.

    fal's universal asset envelope is `{"url": "https://..."}` (often
    nested under `"image"`, `"video"`, `"audio_file"`, `"images": [...]`,
    `"layers": [...]`, etc.). Walks every dict/list and yields each
    distinct URL with its inferred type.
    """
    seen: set[str] = set()

    def _walk(node: Any) -> Iterator[tuple[str, str]]:
        if isinstance(node, dict):
            url = node.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in seen:
                seen.add(url)
                yield url, _infer_asset_type(url)
            for v in node.values():
                yield from _walk(v)
        elif isinstance(node, list):
            for item in node:
                yield from _walk(item)

    yield from _walk(payload)


class GenmediaTools(ToolMetadataMixin):
    """Run any fal.ai endpoint via the genmedia CLI.

    Methods are synchronous from Python's perspective. Long-running
    generations (video, music, slow image models) become async at the
    agent level because okonomi's bg-exec auto-dispatches any
    `execute_code` block containing `genmedia.` — the Python code keeps
    blocking inside the bg thread, the agent stays responsive.

    Generated assets are auto-tracked: every URL in the response is
    registered via `deps.add_artifact(...)`, which triggers eager
    background download into the session directory and surfaces in
    `/assets`. No manual tracking needed.
    """

    __namespace__ = "genmedia"
    __description__ = "Run any fal.ai endpoint via the genmedia CLI. Auto-tracks generated assets in the session."

    def __init__(self):
        # ToolWrapper assigns a fresh deps reference per execute_code
        # call via _set_deps. This stays None when the tool is invoked
        # outside the wrapper (rare — mainly tests).
        self._deps = None

    def _set_deps(self, deps) -> None:
        """Hook called by ToolWrapper at the start of each execute_code
        run. We need deps for auto-tracking; without it, generated URLs
        won't reach the session."""
        self._deps = deps

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run(self, model_id: str, **arguments: Any) -> GenmediaResult:
        """Invoke a fal endpoint synchronously and return its parsed JSON.

        Auto-tracks every URL in the response via the session.

        Args:
            model_id: The fal endpoint id, e.g. ``"fal-ai/flux/dev"``.
                Discover ids with ``genmedia.search(query)`` or via
                fal's catalog at https://fal.ai/models.
            **arguments: Model-specific parameters. Pass anything the
                endpoint's ``--help`` lists. Strings, numbers, and
                booleans are translated to ``--key value`` flags;
                lists become repeated flags.

        Returns:
            The parsed JSON the endpoint emitted. Shape varies per
            model — image endpoints typically return ``{"images":
            [{"url": ...}]}``, video endpoints ``{"video": {"url": ...}}``,
            audio ``{"audio_file": {"url": ...}}``. URLs are already
            registered with the session before this returns.

        Example:
            result = genmedia.run(
                "fal-ai/flux/dev",
                prompt="noir detective in a rainy bar",
                image_size="portrait_16_9",
                num_images=1,
            )
            url = result["images"][0]["url"]
        """
        cmd = ["genmedia", "run", model_id, "--json"]
        for key, value in arguments.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue
            # Lists / tuples / dicts go through as a single JSON-encoded
            # argument. genmedia forwards to fal's JSON API, so structured
            # params (e.g. `loras=[{"path": "...", "scale": 0.8}, ...]`,
            # `image_urls=["...", "..."]`) must arrive as one JSON value,
            # not as repeated flags or Python-repr strings.
            if isinstance(value, (list, tuple, dict)):
                cmd.extend([flag, json.dumps(value)])
                continue
            cmd.extend([flag, str(value)])

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return {"error": completed.stderr.strip() or completed.stdout.strip() or "genmedia run failed"}

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            return {
                "error": f"could not parse genmedia stdout as JSON: {e}",
                "raw": completed.stdout[:2000],
            }

        if not isinstance(payload, dict):
            return {"error": f"unexpected genmedia output type: {type(payload).__name__}", "raw": payload}

        # Unwrap the genmedia CLI envelope.
        #
        # `genmedia run --json` emits something like:
        #   {"status": "completed", "endpoint_id": "...", "request_id": "...",
        #    "result": {"images": [{"url": ...}]}}
        # The model (and the skills) describe the *inner* fal payload shape
        # (`r["images"][0]["url"]`), so without unwrapping every first-attempt
        # access fails with KeyError on the missing `"result"` layer.
        # Detect the envelope conservatively (has `"result"` AND at least one
        # other envelope marker) so a real fal response that happens to have a
        # `"result"` field doesn't get misinterpreted.
        if (
            "result" in payload
            and isinstance(payload["result"], dict)
            and any(k in payload for k in ("status", "endpoint_id", "request_id"))
        ):
            inner = payload["result"]
            # Stash the envelope metadata under underscored fields so it's
            # still recoverable if needed without polluting the natural
            # access pattern.
            for k in ("status", "endpoint_id", "request_id"):
                if k in payload:
                    inner.setdefault(f"_{k}", payload[k])
            payload = inner

        # Auto-track every URL the endpoint returned. Without deps
        # (e.g. unit-test invocation) this is a no-op.
        if self._deps is not None:
            prompt = str(arguments.get("prompt", ""))
            for url, asset_type in _walk_urls(payload):
                self._deps.add_artifact(
                    Artifact(
                        artifact_type=asset_type,
                        url=url,
                        prompt=prompt,
                        tool_name=f"genmedia.{model_id}",
                        model_id=model_id,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        return payload

    def search(self, query: str, limit: int = 10) -> dict:
        """Search fal's catalog for endpoints matching a keyword.

        Args:
            query: Free-text query, e.g. ``"image upscale"``,
                ``"music generation"``, ``"lip sync"``.
            limit: Max number of matches to return. Defaults to 10.

        Returns:
            ``{"models": [{"endpoint_id": str, "name": str, ...}, ...]}``
            or ``{"error": "..."}`` on failure.
        """
        cmd = ["genmedia", "models", query, "--limit", str(limit), "--json"]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            # Some genmedia versions don't accept --json on `models`;
            # fall back to plain text output.
            cmd_plain = ["genmedia", "models", query]
            completed = subprocess.run(cmd_plain, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                return {"error": completed.stderr.strip() or "genmedia models failed"}
            return {"models": completed.stdout.strip().splitlines()[:limit]}

        try:
            return {"models": json.loads(completed.stdout)}
        except json.JSONDecodeError:
            return {"models": completed.stdout.strip().splitlines()[:limit]}

    def schema(self, model_id: str) -> str:
        """Return the parameter schema for a fal endpoint.

        Wraps ``genmedia run <model_id> --help``. Use this when you're
        about to call an endpoint you haven't used before — the schema
        lists every parameter with type and required/optional status.

        Args:
            model_id: The fal endpoint id.

        Returns:
            The raw help-text output. Most useful as ``print()`` input
            so the model can read the schema and pick parameters.
        """
        cmd = ["genmedia", "run", model_id, "--help"]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0 and not completed.stdout:
            return f"genmedia schema lookup failed: {completed.stderr.strip()}"
        return completed.stdout or completed.stderr
