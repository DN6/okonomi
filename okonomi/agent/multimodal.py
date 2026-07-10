"""Image-reference extraction from user messages.

The user can mention image URLs, data: URIs, or local paths in plain text;
these helpers resolve them into pydantic-ai multimodal content parts.
"""

import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote

from pydantic_ai.messages import BinaryContent, ImageUrl

from ..events import SystemNote


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")
_IMAGE_REF_RE = re.compile(
    r"""(?:https?://|file://|data:image/[a-zA-Z]+;base64,)[^\s<>"'`]+""",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpg|jpeg|webp|gif|bmp|tiff)\b", re.IGNORECASE)
_PATH_ANCHOR_RE = re.compile(r"(?:[/~]|\./)")
_SUPPORTED_IMAGE_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _resolve_image(ref: str):
    """Return a pydantic-ai content part for ``ref``, or None for dead
    paths / unsupported formats."""
    if ref.startswith(("http://", "https://", "data:")):
        return ImageUrl(url=ref)
    # Finder drag-and-drop yields file:///foo/bar%20baz.png — decode it.
    path_str = unquote(ref[7:]) if ref.startswith("file://") else ref
    media_type, _ = mimetypes.guess_type(path_str)
    if media_type not in _SUPPORTED_IMAGE_MIME:
        return None
    try:
        data = Path(path_str).expanduser().read_bytes()
    except (OSError, ValueError):
        return None
    return BinaryContent(data=data, media_type=media_type)


def _iter_image_refs(text: str):
    """Yield image refs (URLs, data: URIs, validated local paths) found
    in ``text``, deduplicated. Local paths are robust to spaces,
    backslash-escapes, and surrounding quotes — we walk back from each
    image-extension occurrence to a path-start anchor and validate
    against the filesystem before yielding."""
    seen: set[str] = set()

    for m in _IMAGE_REF_RE.finditer(text):
        ref = m.group(0).rstrip(".,;:)\"'")
        is_image = ref.startswith("data:image/") or ref.split("?", 1)[0].lower().endswith(_IMAGE_EXTS)
        if is_image and ref not in seen:
            seen.add(ref)
            yield ref

    for ext_m in _IMAGE_EXT_RE.finditer(text):
        ext_end = ext_m.end()
        line_start = text.rfind("\n", 0, ext_end) + 1
        for anchor in _PATH_ANCHOR_RE.finditer(text, line_start, ext_end):
            raw = text[anchor.start() : ext_end].strip("\"'").replace("\\ ", " ")
            try:
                p = Path(raw).expanduser()
            except (OSError, ValueError):
                continue
            if p.is_file():
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    yield key
                break


def _parse_user_message(text: str, emitter=None):
    """If ``text`` mentions images, return ``[text, *image_parts]`` so the
    model sees both the pixels and the string references it can pass
    downstream. Returns ``text`` unchanged when no images resolve.

    Dead refs (path missing, unsupported format) surface as SystemNote
    events so the user sees *why* their image wasn't attached.
    """
    parts: list = []
    dead: list[str] = []
    for ref in _iter_image_refs(text):
        part = _resolve_image(ref)
        if part is None:
            dead.append(ref)
        else:
            parts.append(part)

    if emitter is not None:
        for ref in dead:
            emitter.emit(
                SystemNote(
                    f"Couldn't attach image {ref!r} — file missing, unreadable, "
                    "or unsupported format (need png/jpg/gif/webp).",
                    level="warn",
                )
            )

    return [text, *parts] if parts else text
