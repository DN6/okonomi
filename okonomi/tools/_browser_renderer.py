"""Headless browser rendering tools for Okonomi.

Renders arbitrary HTML/CSS/JS in a headless Chromium browser and captures
the output as screenshots or video. Supports WebGL (Three.js), Canvas
(P5.js), CSS animations, SVG, and any web technology.

Playwright's sync API cannot run inside an existing asyncio loop, so all
browser operations are dispatched to a separate thread.
"""

import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.tools._returns import CaptureVideoResult, ScreenshotResult


logger = logging.getLogger(__name__)

_CHROMIUM_ARGS = [
    "--disable-gpu-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--no-sandbox",
]

_thread_pool = ThreadPoolExecutor(max_workers=1)


def _resolve_url(html: str, file_path: str) -> tuple[str, str | None]:
    """Returns (url, tmp_path_or_None)."""
    if html:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        tmp.write(html)
        tmp.close()
        return f"file://{tmp.name}", tmp.name
    resolved = Path(file_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return f"file://{resolved}", None


def _do_screenshot(url, output_path, width, height, wait_ms, full_page, device_scale_factor):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=_CHROMIUM_ARGS)
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(wait_ms)
            page.screenshot(path=output_path, full_page=full_page)
        finally:
            browser.close()


def _do_video(url, output_path, width, height, duration_s, fps, wait_before_record_ms):
    from playwright.sync_api import sync_playwright

    tmp_video_dir = tempfile.mkdtemp()
    webm_path = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=_CHROMIUM_ARGS)
            try:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    record_video_dir=tmp_video_dir,
                    record_video_size={"width": width, "height": height},
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(wait_before_record_ms)
                page.wait_for_timeout(int(duration_s * 1000))
                webm_path = page.video.path()
                context.close()
            finally:
                browser.close()

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(webm_path),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    finally:
        if webm_path:
            Path(webm_path).unlink(missing_ok=True)
        shutil.rmtree(tmp_video_dir, ignore_errors=True)


class BrowserRendererTools(ToolMetadataMixin):
    __namespace__ = "browser"
    __description__ = "Render HTML/JS in a headless browser and capture screenshots or video"

    def __init__(self, config):
        self.config = config

    def render_screenshot(
        self,
        html: str = "",
        file_path: str = "",
        output_path: str | None = None,
        width: int = 1280,
        height: int = 720,
        wait_ms: int = 1000,
        full_page: bool = False,
        device_scale_factor: float = 2.0,
    ) -> ScreenshotResult:
        """Render HTML in a headless browser and capture a screenshot.

        Supports any web technology: Three.js, P5.js, D3.js, CSS animations, SVG, Canvas, WebGL.
        Provide either raw HTML string or a path to an HTML file.

        Args:
            html: Raw HTML string to render. Mutually exclusive with file_path.
            file_path: Path to an HTML file to render. Mutually exclusive with html.
            output_path: Where to save the screenshot. Auto-generated if None.
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            wait_ms: Milliseconds to wait after page load before capturing. Increase for animations that need setup time.
            full_page: If True, capture the full scrollable page instead of just the viewport.
            device_scale_factor: Pixel density multiplier (2.0 = Retina).

        Returns:
            Dict with screenshot path, or error message.
            Example: {"screenshot": "/path/to/screenshot.png", "output_path": "/path/to/screenshot.png"}
        """
        if not html and not file_path:
            return {"error": "Provide either 'html' or 'file_path'"}
        if not output_path:
            output_path = str(Path(tempfile.mkdtemp()) / "screenshot.png")

        tmp_path = None
        try:
            url, tmp_path = _resolve_url(html, file_path)
            future = _thread_pool.submit(
                _do_screenshot, url, output_path, width, height, wait_ms, full_page, device_scale_factor,
            )
            future.result(timeout=60)
            return {"screenshot": output_path, "output_path": output_path}
        except Exception as e:
            return {"error": str(e)}
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def render_video(
        self,
        html: str = "",
        file_path: str = "",
        output_path: str | None = None,
        width: int = 1280,
        height: int = 720,
        duration_s: float = 5.0,
        fps: int = 30,
        wait_before_record_ms: int = 500,
    ) -> CaptureVideoResult:
        """Render HTML in a headless browser and capture video of the page.

        Records the browser viewport for the specified duration, capturing animations,
        WebGL rendering, canvas drawing, etc. Outputs an mp4 file.

        Requires ffmpeg to be installed for webm-to-mp4 conversion.

        Args:
            html: Raw HTML string to render. Mutually exclusive with file_path.
            file_path: Path to an HTML file to render. Mutually exclusive with html.
            output_path: Where to save the mp4 video. Auto-generated if None.
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            duration_s: How many seconds to record.
            fps: Frames per second for the output video.
            wait_before_record_ms: Milliseconds to wait after page load before starting the recording.
                Useful for letting Three.js scenes initialize, assets to load, etc.

        Returns:
            Dict with video path, duration, and fps, or error message.
            Example: {"video": "/path/to/video.mp4", "output_path": "/path/to/video.mp4", "duration_s": 5.0, "fps": 30}
        """
        if not shutil.which("ffmpeg"):
            return {"error": "ffmpeg not found. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"}

        if not html and not file_path:
            return {"error": "Provide either 'html' or 'file_path'"}
        if not output_path:
            output_path = str(Path(tempfile.mkdtemp()) / "render.mp4")

        tmp_path = None
        try:
            url, tmp_path = _resolve_url(html, file_path)
            future = _thread_pool.submit(
                _do_video, url, output_path, width, height, duration_s, fps, wait_before_record_ms,
            )
            future.result(timeout=int(duration_s + 120))
            return {
                "video": output_path,
                "output_path": output_path,
                "duration_s": duration_s,
                "fps": fps,
            }
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
            return {"error": f"ffmpeg conversion failed: {stderr}"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
