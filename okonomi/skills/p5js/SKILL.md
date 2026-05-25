---
name: p5js
description: "Production pipeline for generative visual art and creative coding using p5.js. Default deliverable is a self-contained HTML file the user opens in a browser to see the live, animated, interactive sketch — not a baked image or video. Static PNG / MP4 / GIF / SVG exports are available on request via the headless browser renderer. Covers: 2D/3D rendering, noise and particle systems, flow fields, shaders (GLSL), pixel manipulation, kinetic typography, WebGL scenes, motion graphics. Trigger when users request: p5.js sketches, creative coding, generative art, canvas animations, browser-based visual art, data visualizations, shader effects, or any p5.js project."
metadata:
  tags: [creative-coding, generative-art, p5js, canvas, webgl, shaders, animation]
  related_skills: [genmedia, research]
---

# p5.js

Generative visual art rendered in a browser. The canvas is the medium; the algorithm is the brush. Default deliverable is the HTML file (live, interactive); render to PNG/MP4 only when explicitly asked.

Every recipe in this skill is one ``## H2``. Pull just the section you need with `load_skill_section("p5js", "<slug>")` rather than the whole file. For deep topic dives (noise, shaders, color systems, audio reactivity), use `load_skill_reference("p5js", "<topic>")` against the table at the bottom.

## When to use

- Generative art / creative coding sketches where the deliverable is a live, interactive browser canvas.
- Browser-based animations, kinetic typography, motion graphics.
- WebGL scenes (3D primitives, GLSL shaders, framebuffers).
- Audio-reactive visuals (FFT, amplitude, mic input).
- Data visualization with custom rendering (not "make me a bar chart" — for that, use a charting library).
- Static PNG / MP4 / GIF / SVG export *only* on explicit request — the live HTML is almost always more useful.

**Not for**: AI image / video generation (use `genmedia`), rendering existing media files (use `ffmpeg`), conventional dashboards / charts (use a chart library directly).

## How to use

Five-step pipeline:

1. **Concept** — articulate mood, color world, motion vocabulary, what makes this unique. See `creative-standard`.
2. **Code** — write a single self-contained HTML file. Start from the canonical `html-template` section; save to `get_session_dir()/sketch.html`.
3. **Preview** — capture one screenshot via `browser.render_screenshot(file_path=html_path, wait_ms=2000)` to verify the first frame. See `preview-a-sketch`.
4. **Deliver** — print `f"Sketch ready: file://{html_path}"`. The HTML *is* the deliverable; the user opens it in a browser for the live experience.
5. **Iterate** — edit the HTML, re-capture preview, repeat until the visual matches the concept.

Render to PNG / MP4 / GIF / SVG only when the user explicitly asks. See `deliver-the-html` for the export commands.

Pull specific topics on demand: `load_skill_section("p5js", "<slug>")` for in-skill recipes (`html-template`, `performance-essentials`, `webgl-gotchas`, …) or `load_skill_reference("p5js", "<topic>")` for deep dives (`shaders`, `visual-effects`, `color-systems`, etc.).

### Creative standard

This is art, not a tutorial. Three rules:

- **First-render excellence.** If it looks like a p5.js exercise or generic "AI creative coding," rethink before shipping.
- **Dense, layered, considered.** Every frame rewards viewing. Never flat backgrounds. Always compositional hierarchy. Always intentional color. Always micro-detail that only appears on close inspection.
- **Cohesive aesthetic over feature count.** Shared color temperature, consistent stroke weight vocabulary, harmonious motion speeds. All elements serve a unified visual language.

For every project: a custom 3–7 color palette (never raw `fill(255,0,0)`), a textured/gradient/layered background (never plain `background(0)`), motion variety (primary at 1×, secondary at 0.3×, ambient at 0.1×), and at least one invented element — a custom particle behavior, novel noise application, unique visual mechanic.

### Modes

| Mode | Input | Output | Reference |
|------|-------|--------|-----------|
| Generative art | Seed / params | Procedural composition | `visual-effects` |
| Data visualization | Dataset / API | Custom data displays | `interaction` |
| Interactive experience | None (user drives) | Mouse/keyboard sketch | `interaction` |
| Animation / motion graphics | Timeline / storyboard | Timed sequences, kinetic typography | `animation` |
| 3D scene | Concept | WebGL geometry, lighting, materials | `webgl-and-3d` |
| Image processing | Image file(s) | Pixel manipulation, filters, mosaic | `visual-effects` § Pixel |

### Stack

Single self-contained HTML per project. No build step.

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | p5.js 1.11.3 (CDN) | Canvas rendering, math, transforms |
| 3D | p5.js WebGL mode | 3D geometry, camera, GLSL shaders |
| Audio | p5.sound.js (CDN) | FFT, amplitude, mic input |
| Export | `browser.render_screenshot()` / `browser.render_video()` | Headless PNG / MP4 |
| SVG | p5.js-svg 1.6.0 (optional) | Vector output (p5.js 1.x only) |
| Natural media | p5.brush (optional) | Watercolor, charcoal (p5.js 2.x + WEBGL) |
| Texture | p5.grain (optional) | Film grain, texture overlays |

p5.js 1.x (1.11.3) is the default. p5.js 2.x adds `async setup()`, OKLCH/OKLAB color modes, `splineVertex()`, shader `.modify()`, variable fonts, `textToContours()`. See `core-api` reference § p5.js 2.0.

### Pipeline

```
CONCEPT → DESIGN → CODE → PREVIEW → DELIVER
```

1. **CONCEPT** — articulate creative vision: mood, color world, motion vocabulary, what makes this unique.
2. **DESIGN** — pick mode, canvas size, color system.
3. **CODE** — single HTML file with inline p5.js, save to `get_session_dir()`. Use the template in `html-template`.
4. **PREVIEW** — capture one screenshot with `browser.render_screenshot()` to verify the first frame. See `preview-a-sketch`.
5. **DELIVER** — print the HTML file path; user opens it in a browser. See `deliver-the-html`. Render PNG/MP4 only on explicit request.

### HTML template

Drop this in as the starting point of every sketch:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Name</title>
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
const CONFIG = { seed: 42 };
const PALETTE = { bg: '#0a0a0f', primary: '#e8d5b7' };

function setup() {
  createCanvas(1920, 1080);
  pixelDensity(1);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
}

function draw() {
  // Render frame
}

function windowResized() { resizeCanvas(windowWidth, windowHeight); }
</script>
</body>
</html>
```

Patterns embedded above:
- **Seeded randomness** — `randomSeed()` + `noiseSeed()` for reproducibility.
- **HSB color mode** — intuitive control over hue/saturation/brightness.
- **State separation** — `CONFIG` for parameters, `PALETTE` for colors.
- **FES disabled** — friendly errors are slow in hot loops; turn off before `setup()`.

### Preview a sketch

Write the HTML to the session dir, capture a screenshot to verify the first frame:

```python
session_dir = get_session_dir()
html_path = f"{session_dir}/sketch.html"
with open(html_path, "w") as f:
    f.write(html_content)

result = browser.render_screenshot(
    file_path=html_path,
    output_path=f"{session_dir}/preview.png",
    width=1920, height=1080,
    wait_ms=2000,  # let setup + first draw complete
)
print(result)
```

`wait_ms` matters — too short and you capture a half-rendered frame. 2000ms is safe for most sketches; bump higher if there's heavy first-frame computation.

### Deliver the HTML

The HTML file IS the deliverable. The user opens it in a browser and sees the live, animated, interactive sketch — far better than a baked PNG/MP4. Print the path:

```python
print(f"Sketch ready: file://{html_path}")
```

Render to image/video only when the user explicitly asks (e.g. "give me a still", "export as mp4"):

| Format | Method |
|---|---|
| PNG | `browser.render_screenshot(file_path=html_path, width=3840, height=2160)` |
| MP4 | `browser.render_video(file_path=html_path, duration_s=10, fps=30)` |

For multi-scene video, GIF, or SVG export — see `load_skill_reference("p5js", "export-pipeline")`.

### Performance essentials

```javascript
p5.disableFriendlyErrors = true;     // BEFORE setup() — FES is slow

function setup() {
  pixelDensity(1);                   // prevent 2-4× overdraw on retina
  createCanvas(1920, 1080);
}
```

In hot loops, use `Math.*` instead of p5 wrappers — p5 wraps add per-call overhead:

```javascript
let a = Math.sin(t);                 // not sin(t)
let r = Math.sqrt(dx*dx + dy*dy);    // not dist()
```

Targets to budget against:

| Metric | Target |
|---|---|
| Frame rate (animated) | 30fps minimum |
| Particle count (P2D shapes) | 5,000–10,000 at 60fps |
| Particle count (pixel buffer) | 50,000–100,000 at 60fps |
| Canvas resolution | Up to 3840×2160 export, 1920×1080 interactive |

For deeper profiling and memory-leak debugging, see `load_skill_reference("p5js", "troubleshooting")`.

### Noise — multi-octave, not raw

Raw `noise(x, y)` looks like smooth blobs. Layer octaves:

```javascript
function fbm(x, y, octaves = 4) {
  let val = 0, amp = 1, freq = 1, sum = 0;
  for (let i = 0; i < octaves; i++) {
    val += noise(x * freq, y * freq) * amp;
    sum += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return val / sum;
}
```

For flowing organic forms, layer domain warping on top. See `load_skill_reference("p5js", "visual-effects")`.

### Layered composition with createGraphics()

Offscreen buffers let you composite cleanly — background, trails, foreground each on their own layer:

```javascript
let bgLayer, fgLayer, trailLayer;
function setup() {
  createCanvas(1920, 1080);
  bgLayer = createGraphics(width, height);
  fgLayer = createGraphics(width, height);
  trailLayer = createGraphics(width, height);
}
function draw() {
  renderBackground(bgLayer);
  renderTrails(trailLayer);
  renderForeground(fgLayer);
  image(bgLayer, 0, 0);
  image(trailLayer, 0, 0);
  image(fgLayer, 0, 0);
}
```

Trails are achieved by *not* clearing the trail layer between frames, or by drawing a low-alpha rect to fade older content.

### WebGL gotchas

- `createCanvas(w, h, WEBGL)` — origin is the *center*, not the top-left.
- Y-axis is inverted (positive Y is up).
- `push()` / `pop()` around every transform.
- `texture()` *before* `rect()` / `plane()` — not after.

Deeper WebGL material (camera, lighting, GLSL shaders, framebuffers, post-processing) is in `load_skill_reference("p5js", "webgl-and-3d")`.

## References

Topic-deep material, available via `load_skill_reference("p5js", "<name>")`:

| Reference | Contents |
|---|---|
| `core-api` | Canvas setup, coordinate system, draw loop, transforms, offscreen buffers, composition patterns |
| `shapes-and-geometry` | 2D primitives, `beginShape()`, Bezier/Catmull-Rom curves, `p5.Vector`, signed distance fields |
| `visual-effects` | Perlin/fBM/domain-warp/curl noise, flow fields, particles, pixel manipulation, texture, feedback loops |
| `animation` | Easing, spring physics, state machines, timeline sequencing, transitions |
| `typography` | `loadFont()`, `textToPoints()`, kinetic typography, text masks, responsive text |
| `color-systems` | `colorMode()`, HSB/HSL, `lerpColor()`, procedural palettes, blend modes, palette library |
| `webgl-and-3d` | WEBGL renderer, 3D primitives, camera, lighting, GLSL, framebuffers, post-processing |
| `interaction` | Mouse/keyboard, DOM elements, audio input (FFT/amplitude), scroll-driven animation |
| `export-pipeline` | `render_screenshot()`, `render_video()`, `saveGif()`, SVG, multi-scene, ffmpeg stitching |
| `troubleshooting` | Performance profiling, common mistakes, browser compatibility, WebGL debugging, memory leaks |
