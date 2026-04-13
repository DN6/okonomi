# Export Pipeline

## PNG Export

### In-Sketch (Keyboard Shortcut)

```javascript
function keyPressed() {
  if (key === 's' || key === 'S') {
    saveCanvas('output', 'png');
  }
}
```

### Timed Export (Static Generative)

```javascript
function setup() {
  createCanvas(3840, 2160);
  pixelDensity(1);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  noLoop();
}

function draw() {
  // ... render everything ...
  saveCanvas('output-seed-' + CONFIG.seed, 'png');
}
```

### High-Resolution Export

```javascript
function exportHighRes(scale) {
  let buffer = createGraphics(width * scale, height * scale);
  buffer.scale(scale);
  renderScene(buffer);
  buffer.save('highres-output.png');
}
```

### Headless Screenshot via browser tool

Use okonomi's `browser.render_screenshot()` to capture without opening a browser:

```python
# Write the sketch HTML to the session directory
session_dir = get_session_dir()
html_path = f"{session_dir}/sketch.html"
with open(html_path, "w") as f:
    f.write(html_content)

# Capture at high resolution
result = browser.render_screenshot(
    file_path=html_path,
    output_path=f"{session_dir}/output.png",
    width=1920,
    height=1080,
    wait_ms=2000,        # wait for sketch to render
    device_scale_factor=2.0,  # retina quality
)
print(result)
```

## GIF Export

### saveGif()

```javascript
function keyPressed() {
  if (key === 'g' || key === 'G') {
    saveGif('output', 5);  // 5 seconds
  }
}
```

Limitations:
- GIF is 256 colors max — dithering on gradients
- Use smaller canvas (640x360) for GIF
- Frame rate is approximate

## Video Export

### Headless via browser tool (Recommended)

Use okonomi's `browser.render_video()` for automated video capture:

```python
session_dir = get_session_dir()
html_path = f"{session_dir}/sketch.html"
with open(html_path, "w") as f:
    f.write(html_content)

result = browser.render_video(
    file_path=html_path,
    output_path=f"{session_dir}/output.mp4",
    width=1920,
    height=720,
    duration_s=10.0,
    fps=30,
    wait_before_record_ms=1000,  # let sketch initialize
)
print(result)
```

### Frame Sequence + ffmpeg (Maximum Control)

For frame-perfect capture, use `saveFrames()` or manual frame export:

```javascript
let recording = false;
let frameNum = 0;
const TOTAL_FRAMES = 300;

function keyPressed() {
  if (key === 'r') recording = !recording;
}

function draw() {
  // ... render frame ...
  if (recording) {
    saveCanvas('frame-' + nf(frameNum, 4), 'png');
    frameNum++;
    if (frameNum >= TOTAL_FRAMES) {
      recording = false;
      noLoop();
    }
  }
}
```

Then stitch with ffmpeg:

```
run(command="ffmpeg -framerate 30 -i frame-%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
```

### Multi-Scene Video

Render each scene as a separate HTML + video clip, then stitch:

```python
scenes = ["intro.html", "particles.html", "noise.html", "outro.html"]
clips = []

for scene in scenes:
    clip_path = f"{session_dir}/{scene.replace('.html', '.mp4')}"
    browser.render_video(
        file_path=f"{session_dir}/{scene}",
        output_path=clip_path,
        width=1920, height=1080,
        duration_s=5.0, fps=30,
        wait_before_record_ms=500,
    )
    clips.append(clip_path)

# Write concat file
filelist = f"{session_dir}/filelist.txt"
with open(filelist, "w") as f:
    for clip in clips:
        f.write(f"file '{clip}'\n")
```

Then stitch:
```
run(command="ffmpeg -y -f concat -safe 0 -i {session_dir}/filelist.txt -c copy {session_dir}/final.mp4")
```

## Video Export Gotchas

**YUV420 clips dark values.** H.264 rounds dark RGB values below ~(8,8,8) to pure black. Subtle dark details disappear in encoded video. Fix: ensure minimum brightness of ~10 for visible content.

**Static frames look broken in video.** Add animation even to static content:
- Progressive reveal (expand from center, sweep across)
- Slow parameter drift (rotate color mapping, shift noise offset)
- Camera-like motion (slow zoom, slight pan)
- Overlay animated particles or grain

**Scene transitions.** Hard cuts are jarring. Use fade envelopes:

```javascript
const FADE_FRAMES = 15;
let fade = 1;
if (localFrame < FADE_FRAMES) fade = localFrame / FADE_FRAMES;
if (localFrame > SCENE_FRAMES - FADE_FRAMES) fade = (SCENE_FRAMES - localFrame) / FADE_FRAMES;
fade = fade * fade * (3 - 2 * fade);  // smoothstep
```

## SVG Export

Requires p5.js-svg addon (1.x only):

```html
<script src="https://unpkg.com/p5.js-svg@1.6.0"></script>
```

```javascript
function setup() {
  createCanvas(1080, 1080, SVG);
}

function keyPressed() {
  if (key === 's') save('output.svg');
}
```
