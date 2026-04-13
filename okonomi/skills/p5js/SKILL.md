---
name: p5js
description: "Production pipeline for generative visual art and creative coding using p5.js. Creates browser-based sketches exported as PNG, GIF, MP4, or SVG via okonomi's headless browser renderer. Covers: 2D/3D rendering, noise and particle systems, flow fields, shaders (GLSL), pixel manipulation, kinetic typography, WebGL scenes, and motion graphics. Trigger when users request: p5.js sketches, creative coding, generative art, canvas animations, browser-based visual art, data visualizations, shader effects, or any p5.js project."
metadata:
  tags: [creative-coding, generative-art, p5js, canvas, webgl, shaders, animation]
  related_skills: [video-generation, concept-research]
---

# p5.js Production Pipeline

## Creative Standard

This is visual art rendered in the browser. The canvas is the medium; the algorithm is the brush.

**Before writing a single line of code**, articulate the creative concept. What does this piece communicate? What makes the viewer stop scrolling? The user's prompt is a starting point — interpret it with creative ambition.

**First-render excellence is non-negotiable.** If it looks like a p5.js tutorial exercise or "AI-generated creative coding," it is wrong. Rethink before shipping.

**Dense, layered, considered.** Every frame should reward viewing. Never flat white backgrounds. Always compositional hierarchy. Always intentional color. Always micro-detail that only appears on close inspection.

**Cohesive aesthetic over feature count.** All elements must serve a unified visual language — shared color temperature, consistent stroke weight vocabulary, harmonious motion speeds.

## Modes

| Mode | Input | Output | Reference |
|------|-------|--------|-----------|
| **Generative art** | Seed / parameters | Procedural composition (still or animated) | `references/visual-effects.md` |
| **Data visualization** | Dataset / API | Custom data displays | `references/interaction.md` |
| **Interactive experience** | None (user drives) | Mouse/keyboard-driven sketch | `references/interaction.md` |
| **Animation / motion graphics** | Timeline / storyboard | Timed sequences, kinetic typography | `references/animation.md` |
| **3D scene** | Concept description | WebGL geometry, lighting, materials | `references/webgl-and-3d.md` |
| **Image processing** | Image file(s) | Pixel manipulation, filters, mosaic | `references/visual-effects.md` § Pixel Manipulation |

## Stack

Single self-contained HTML file per project. No build step required.

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | p5.js 1.11.3 (CDN) | Canvas rendering, math, transforms |
| 3D | p5.js WebGL mode | 3D geometry, camera, GLSL shaders |
| Audio | p5.sound.js (CDN) | FFT analysis, amplitude, mic input |
| Export | `browser.render_screenshot()` / `browser.render_video()` | Headless PNG and MP4 capture |
| SVG | p5.js-svg 1.6.0 (optional) | Vector output — requires p5.js 1.x |
| Natural media | p5.brush (optional) | Watercolor, charcoal — requires p5.js 2.x + WEBGL |
| Texture | p5.grain (optional) | Film grain, texture overlays |

### Version Note

**p5.js 1.x** (1.11.3) is the default — stable, broadest compatibility.

**p5.js 2.x** (2.2+) adds: `async setup()`, OKLCH/OKLAB color modes, `splineVertex()`, shader `.modify()`, variable fonts, `textToContours()`. See `references/core-api.md` § p5.js 2.0.

## Pipeline

```
CONCEPT → DESIGN → CODE → PREVIEW → EXPORT → VERIFY
```

1. **CONCEPT** — Articulate the creative vision: mood, color world, motion vocabulary, what makes this unique
2. **DESIGN** — Choose mode, canvas size, color system, export format
3. **CODE** — Write single HTML file with inline p5.js
4. **PREVIEW** — Capture a screenshot with `browser.render_screenshot()` to verify
5. **EXPORT** — Capture final output: screenshot for PNG, `browser.render_video()` for MP4
6. **VERIFY** — Does the output match the concept? Is it visually striking?

## Creative Direction

### Aesthetic Dimensions

| Dimension | Options | Reference |
|-----------|---------|-----------|
| **Color system** | HSB/HSL, named palettes, procedural harmony | `references/color-systems.md` |
| **Noise vocabulary** | Perlin, fractal (fBM), domain warping, curl noise | `references/visual-effects.md` § Noise |
| **Particle systems** | Physics-based, flocking, trail-drawing, attractor-driven | `references/visual-effects.md` § Particles |
| **Shape language** | Geometric primitives, custom vertices, bezier curves | `references/shapes-and-geometry.md` |
| **Motion style** | Eased, spring-based, noise-driven, physics sim | `references/animation.md` |
| **Typography** | Loaded OTF, `textToPoints()` particle text, kinetic | `references/typography.md` |
| **Shader effects** | GLSL fragment/vertex, filter shaders, feedback loops | `references/webgl-and-3d.md` § Shaders |
| **Composition** | Grid, radial, golden ratio, rule of thirds, organic scatter | `references/core-api.md` § Composition |
| **Blend modes** | `ADD`, `MULTIPLY`, `SCREEN`, `DIFFERENCE`, `OVERLAY` | `references/color-systems.md` § Blend Modes |
| **Layering** | `createGraphics()` offscreen buffers, alpha compositing | `references/core-api.md` § Offscreen Buffers |
| **Texture** | Perlin surface, stippling, hatching, halftone, pixel sorting | `references/visual-effects.md` § Texture Generation |

### Per-Project Rules

Never use default configurations. For every project:
- **Custom color palette** — never raw `fill(255, 0, 0)`. Always a designed palette with 3-7 colors
- **Background treatment** — never plain `background(0)`. Always textured, gradient, or layered
- **Motion variety** — different speeds for different elements. Primary at 1x, secondary at 0.3x, ambient at 0.1x
- **At least one invented element** — a custom particle behavior, a novel noise application, a unique visual mechanic

## Workflow

### Step 1: Creative Vision

Before any code:
- **Mood**: What should the viewer feel?
- **Color world**: Warm/cool? Monochrome? Complementary?
- **Shape language**: Organic curves? Sharp geometry? Mixed?
- **Motion vocabulary**: Slow drift? Explosive burst? Breathing pulse?
- **What makes THIS different**: The one thing that makes this sketch unique

### Step 2: Code the Sketch

Single HTML file. Structure:

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

Key patterns:
- **Seeded randomness**: Always `randomSeed()` + `noiseSeed()` for reproducibility
- **Color mode**: Use `colorMode(HSB, 360, 100, 100, 100)` for intuitive color control
- **State separation**: CONFIG for parameters, PALETTE for colors
- **Offscreen buffers**: `createGraphics()` for layered composition and trails

### Step 3: Preview

Write the HTML to the session directory, then capture a screenshot:

```python
session_dir = get_session_dir()
html_path = f"{session_dir}/sketch.html"
with open(html_path, "w") as f:
    f.write(html_content)

result = browser.render_screenshot(
    file_path=html_path,
    output_path=f"{session_dir}/preview.png",
    width=1920, height=1080,
    wait_ms=2000,
)
print(result)
```

### Step 4: Export

| Format | Method |
|--------|--------|
| **PNG** | `browser.render_screenshot(file_path=html_path, width=3840, height=2160)` |
| **MP4** | `browser.render_video(file_path=html_path, duration_s=10, fps=30)` |

See `references/export-pipeline.md` for full export details including multi-scene video.

### Step 5: Iterate

Edit the HTML, re-capture. Adjust parameters until the visual matches the concept.

## Critical Implementation Notes

### Performance — Disable FES First

```javascript
p5.disableFriendlyErrors = true;  // BEFORE setup()

function setup() {
  pixelDensity(1);  // prevent 2x-4x overdraw on retina
  createCanvas(1920, 1080);
}
```

In hot loops, use `Math.*` instead of p5 wrappers:
```javascript
let a = Math.sin(t);          // not sin(t)
let r = Math.sqrt(dx*dx+dy*dy); // not dist()
```

### Noise — Multi-Octave, Not Raw

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

For flowing organic forms, use domain warping. See `references/visual-effects.md`.

### createGraphics() for Layers

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

### WebGL Mode Gotchas

- `createCanvas(w, h, WEBGL)` — origin is center, not top-left
- Y-axis is inverted (positive Y goes up)
- `push()`/`pop()` around every transform
- `texture()` before `rect()`/`plane()` — not after

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame rate (animated) | 30fps minimum |
| Particle count (P2D shapes) | 5,000-10,000 at 60fps |
| Particle count (pixel buffer) | 50,000-100,000 at 60fps |
| Canvas resolution | Up to 3840x2160 (export), 1920x1080 (interactive) |

## References

| File | Contents |
|------|----------|
| `references/core-api.md` | Canvas setup, coordinate system, draw loop, transforms, offscreen buffers, composition patterns |
| `references/shapes-and-geometry.md` | 2D primitives, `beginShape()`, Bezier/Catmull-Rom curves, `p5.Vector`, signed distance fields |
| `references/visual-effects.md` | Noise (Perlin, fBM, domain warp, curl), flow fields, particle systems, pixel manipulation, texture generation, feedback loops |
| `references/animation.md` | Easing functions, spring physics, state machines, timeline sequencing, transition patterns |
| `references/typography.md` | `loadFont()`, `textToPoints()`, kinetic typography, text masks, responsive text |
| `references/color-systems.md` | `colorMode()`, HSB/HSL, `lerpColor()`, procedural palettes, `blendMode()`, curated palette library |
| `references/webgl-and-3d.md` | WEBGL renderer, 3D primitives, camera, lighting, GLSL shaders, framebuffers, post-processing |
| `references/interaction.md` | Mouse/keyboard events, DOM elements, audio input (FFT/amplitude), scroll-driven animation |
| `references/export-pipeline.md` | `browser.render_screenshot()`, `browser.render_video()`, `saveGif()`, SVG export, multi-scene video, ffmpeg stitching |
| `references/troubleshooting.md` | Performance profiling, common mistakes, browser compatibility, WebGL debugging, memory leaks |
| `templates/viewer.html` | Interactive viewer: seed navigation, parameter sliders, download PNG |
