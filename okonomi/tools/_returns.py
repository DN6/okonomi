"""TypedDict return shapes for programmatic tools.

Used as return annotations on the public methods so `inspect.signature(...)`
surfaces a named type instead of opaque `dict`. The model can then introspect
keys via ``Result.__annotations__`` or ``typing.get_type_hints(method)`` to
see the exact dict keys + per-key types — which is the most common cause of
KeyError fumbles.

Each shape uses ``total=False`` so a method can also return an error-only
``{"error": "..."}`` shape from the same TypedDict.

Note: deliberately *not* using ``from __future__ import annotations`` so the
annotation values are real type objects (``list[str]``, ``str``, …) rather
than ``ForwardRef`` strings — makes ``Result.__annotations__`` directly
useful to the model without needing ``typing.get_type_hints``.
"""

from typing import TypedDict


# ---------------------------------------------------------------------------
# image_generation
# ---------------------------------------------------------------------------


class ImageGenResult(TypedDict, total=False):
    """Return type for image_generation.text_to_image / edit_image / reference_image.

    On success: ``images`` is a non-empty list of URL strings.
    On failure: ``error`` is the exception message and no other keys are set.
    """

    images: list[str]  # list of image URL strings, e.g. ["https://fal.media/.../x.png"]
    error: str         # only present on failure


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


class VideoResult(TypedDict, total=False):
    """Return type for video.image_to_video / first_last_frame_to_video.

    Note: the URL key is ``video`` (not ``videos`` or ``url``).
    """

    video: str       # video URL string, e.g. "https://fal.media/.../out.mp4"
    model_id: str    # the fal model id used to generate the video
    error: str       # only present on failure


# ---------------------------------------------------------------------------
# image_processing
# ---------------------------------------------------------------------------


class LoadImagesResult(TypedDict, total=False):
    """Return type for image_processing.load_images. Note: ``images`` is a
    list of PIL.Image.Image objects, NOT URL strings."""

    images: list  # list[PIL.Image.Image]
    count: int
    error: str


class GetImageLayersResult(TypedDict, total=False):
    """Return type for image_processing.get_image_layers. ``layers`` is a
    flat list of URL strings — one per decomposed layer."""

    layers: list[str]
    error: str


class SegmentImageResult(TypedDict, total=False):
    """Return type for image_processing.segment_image. The mask preview is
    returned as a single-element ``images`` list (matches image_generation)."""

    images: list[str]
    error: str


class UpscaleImageResult(TypedDict, total=False):
    """Return type for image_processing.upscale_image.

    The upscaled output URL is in ``images[0]``; matches the rest of the
    image_processing return shapes so callers can use a uniform access
    pattern.
    """

    images: list[str]
    error: str


class LoadVideoFramesResult(TypedDict, total=False):
    """Return type for image_processing.load_video_frames."""

    frames: list  # list[PIL.Image.Image]
    fps: float
    width: int
    height: int
    duration: float
    count: int
    error: str


# ---------------------------------------------------------------------------
# browser
# ---------------------------------------------------------------------------


class ScreenshotResult(TypedDict, total=False):
    """Return type for browser.render. Note: both keys point at the same
    local file path on success."""

    screenshot: str   # local file path to PNG
    output_path: str  # alias of screenshot
    error: str


class CaptureVideoResult(TypedDict, total=False):
    """Return type for browser.capture_video."""

    video: str          # local file path to MP4
    output_path: str    # alias of video
    duration: float
    fps: int
    error: str


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class GeneratePromptResult(TypedDict, total=False):
    """Return type for prompt.create_prompt / .generate_prompt."""

    prompt: str
    error: str


class DescribeImageResult(TypedDict, total=False):
    """Return type for prompt.describe_image."""

    description: str
    error: str


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------


class ResearchResultItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float


class WebSearchResult(TypedDict, total=False):
    """Return type for research.web_search.

    ``query`` echoes the input, ``results`` is a list of
    ``ResearchResultItem`` dicts. ``error`` only set on failure.
    """

    query: str
    results: list[ResearchResultItem]
    error: str
