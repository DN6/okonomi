"""Framework-agnostic Rich rendering helpers shared by both front-ends.

Pure Rich + stdlib — no Textual, no prompt_toolkit — so the streaming REPL
sink can reuse them. ``text_with_path_links`` produces a Rich ``Text`` whose
matched segments carry ``link`` styles; under Textual the app's ``on_click``
reads the target back, while the streaming REPL prints through a real
``rich.console.Console`` which emits genuine OSC 8 hyperlinks.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from rich.text import Text


# One pass matches either an http(s) URL or an on-disk path. URL first so the
# `://` is consumed there; the path branch's negative lookbehind then can't
# re-match the slashes inside a URL.
_LINK_RE = re.compile(
    r"(?P<url>https?://[^\s`\"'<>(){}\[\]]+)"
    r"|(?P<path>(?<![\w:/])(?:~/|/)[^\s`\"'<>(){}\[\]]+)"
)
# Trailing punctuation that's almost never part of a real path/URL — strip
# before resolving so "see /foo/bar.png." or "...png)." resolves.
_PATH_TRAILING_PUNCT = ".,;:!?)"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")


@lru_cache(maxsize=512)
def _path_exists(expanded: str) -> bool:
    return os.path.exists(expanded)


def _is_image_url(url: str) -> bool:
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(_IMAGE_EXTS)


def text_with_path_links(s: str) -> Text:
    """Return a Rich ``Text`` with on-disk paths and http(s) image URLs
    wrapped as ``link`` styles. Only paths that exist on disk, and URLs that
    look like images, are linked — to avoid false positives.
    """
    out = Text()
    last = 0
    for m in _LINK_RE.finditer(s):
        raw = m.group(0)
        trim = 0
        while trim < len(raw) and raw[-1 - trim] in _PATH_TRAILING_PUNCT:
            trim += 1
        candidate = raw[: len(raw) - trim] if trim else raw

        if m.lastgroup == "url":
            if not _is_image_url(candidate):
                continue
            target = candidate
        else:
            expanded = os.path.expanduser(candidate)
            if not _path_exists(expanded):
                continue
            target = f"file://{expanded}"

        out.append(s[last : m.start()])
        out.append(candidate, style=f"link {target}")
        if trim:
            out.append(raw[len(raw) - trim :])
        last = m.end()
    out.append(s[last:])
    return out


# Radial banner gradient: full-block glyphs fade from a bright pink centre out
# to a dim plum edge. The label sits punched into the middle row.
_BANNER_CENTER = "#ffa8cf"
_BANNER_EDGE = "#3a1420"
_BANNER_LABEL = "bold #fff0f6"
_BANNER_HEIGHT = 5
_BANNER_MARGIN = 10  # block cells on each side of the label
_BANNER_GLYPH = "█"


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linear-interpolate between two ``#rrggbb`` colours (t in [0, 1])."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[j] + (b[j] - a[j]) * t):02x}" for j in range(3))


def render_welcome_banner(label: str = "okonomi") -> Text:
    """Build the welcome banner as a Rich ``Text``: a block of full-block
    glyphs whose colour fades outward from a bright centre to a dim edge, with
    the label punched into the middle row. Lines carry no leading whitespace —
    the caller centers them.
    """
    import math

    width = len(label) + _BANNER_MARGIN * 2
    cx, cy = (width - 1) / 2, (_BANNER_HEIGHT - 1) / 2
    norm = math.hypot(cx, cy) or 1.0
    label_start = (width - len(label)) // 2
    mid = _BANNER_HEIGHT // 2

    text = Text()
    for y in range(_BANNER_HEIGHT):
        for x in range(width):
            if y == mid and label_start <= x < label_start + len(label):
                text.append(label[x - label_start], style=_BANNER_LABEL)
                continue
            # ** 0.85 eases the falloff so the fade stays gentle near the centre
            t = (math.hypot(x - cx, y - cy) / norm) ** 0.85
            text.append(_BANNER_GLYPH, style=_lerp_hex(_BANNER_CENTER, _BANNER_EDGE, t))
        text.append("\n")
    return text
