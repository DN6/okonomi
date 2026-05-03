---
name: ffmpeg
description: "Recipe-driven ffmpeg operations for video and audio post-production: format conversion, concatenation, audio muxing, transitions (crossfade/xfade), trimming, looping, speed changes, scaling, and frame/audio extraction. ffmpeg is already installed locally and invoked via the standalone `run` tool. Trigger when users ask to: combine/stitch videos, add or replace an audio track, fade between clips, convert formats (mp4 ↔ webm ↔ gif), trim a clip, slow down or speed up, change aspect ratio, extract audio from video, or extract a frame as PNG."
metadata:
  tags: [ffmpeg, video, audio, post-production, mux, concat, transitions]
  related_skills: [p5js, video-generation]
---

# ffmpeg Recipes

ffmpeg is the universal video/audio toolbox. It's already on the system
(used internally by `browser.render_video` for webm→mp4 conversion) so
this skill is just **how to drive it via `run(command=...)`**.

## Calling Convention

Always invoke through the standalone `run` tool, NOT inside `execute_code`:

```python
result = run(command='ffmpeg -y -i "{input}" -i "{audio}" '
                     '-c:v copy -c:a aac -shortest "{output}"')
```

- `-y` — overwrite output without prompting (always include this)
- `-loglevel error` — suppress info chatter when you don't need it
- Save outputs to `get_session_dir()` so they're tracked in the session
- Always quote filenames — paths can contain spaces, parentheses, etc.

## Recipes

### Mux audio onto a video

Replace or add an audio track. Output ends when the shorter stream ends.

```bash
ffmpeg -y -i video.mp4 -i audio.wav \
  -c:v copy -c:a aac -shortest \
  output.mp4
```

If the audio is longer than the video, this trims it. If you want the
opposite (loop the video to match audio length), see "Loop a video" below.

### Strip / extract audio from a video

Pull just the audio out as a standalone file:

```bash
ffmpeg -y -i input.mp4 -vn -acodec copy output.aac     # keep original codec
ffmpeg -y -i input.mp4 -vn -ac 2 -ar 44100 output.wav  # decode to wav
```

### Concatenate clips (same codec / format)

Fastest path. Build a manifest, then concat without re-encoding:

```bash
# Write the manifest:
cat > list.txt <<EOF
file 'clip1.mp4'
file 'clip2.mp4'
file 'clip3.mp4'
EOF

ffmpeg -y -f concat -safe 0 -i list.txt -c copy out.mp4
```

If clips have different codecs/resolutions, this will fail — use the
filter-based concat instead (see "Concat with re-encode").

### Concat with re-encode (mixed sources)

When clips don't match, normalize through the concat filter. Slower but
robust:

```bash
ffmpeg -y -i a.mp4 -i b.mp4 -i c.mp4 \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" \
  out.mp4
```

`n=3` is the count; `v=1:a=1` says each input contributes one video and
one audio stream. Drop `[N:a]` and `a=1` if your inputs have no audio.

### Crossfade between two clips

`xfade` for video, `acrossfade` for audio. Specify offset (when the
fade *starts* in the first clip's timeline) and duration:

```bash
ffmpeg -y -i a.mp4 -i b.mp4 -filter_complex \
  "[0:v][1:v]xfade=transition=fade:duration=1:offset=4[v]; \
   [0:a][1:a]acrossfade=d=1[a]" \
  -map "[v]" -map "[a]" out.mp4
```

Other transitions: `slideleft`, `slideright`, `wipeleft`, `circleopen`,
`pixelize`, `dissolve`, `radial`. Full list:
`ffmpeg -h filter=xfade`.

### Trim a clip

Cut between two timestamps. Use `-ss` BEFORE `-i` for fast keyframe seek:

```bash
ffmpeg -y -ss 00:00:05 -to 00:00:12 -i input.mp4 -c copy clip.mp4
```

`-c copy` skips re-encoding (instant). Drop `-c copy` if you need
frame-accurate cuts at non-keyframe boundaries.

### Loop a video

Loop the video N times, or until it matches a target duration:

```bash
ffmpeg -y -stream_loop 4 -i input.mp4 -c copy looped.mp4
ffmpeg -y -stream_loop -1 -i input.mp4 -t 30 -c copy looped30s.mp4  # 30s
```

### Speed up or slow down

Video via `setpts` (PTS = presentation timestamp). Audio via `atempo`
(per-step limited to 0.5x–2.0x; chain for larger changes):

```bash
# 2x faster (silent)
ffmpeg -y -i input.mp4 -filter:v "setpts=0.5*PTS" -an out.mp4

# 0.5x slower with audio
ffmpeg -y -i input.mp4 \
  -filter_complex "[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" out.mp4

# 4x faster audio (chain atempo)
ffmpeg -y -i input.wav -filter:a "atempo=2.0,atempo=2.0" out.wav
```

### Convert formats

```bash
# webm → mp4 (h264 + aac, broadly compatible)
ffmpeg -y -i in.webm -c:v libx264 -preset fast -crf 18 -c:a aac out.mp4

# mp4 → webm (vp9)
ffmpeg -y -i in.mp4 -c:v libvpx-vp9 -b:v 2M -c:a libopus out.webm

# mp4 → animated gif (palette pass for quality)
ffmpeg -y -i in.mp4 -vf "fps=15,scale=720:-1:flags=lanczos,palettegen" palette.png
ffmpeg -y -i in.mp4 -i palette.png \
  -filter_complex "fps=15,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif
```

### Scale / pad / crop (aspect ratio fix)

```bash
# Scale to 1080p, preserve aspect (one dim = -2 keeps multiple of 2)
ffmpeg -y -i in.mp4 -vf "scale=1920:-2" out.mp4

# Letterbox to 16:9 1080p (pad black bars)
ffmpeg -y -i in.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease, \
       pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  out.mp4

# Crop center to square
ffmpeg -y -i in.mp4 -vf "crop=ih:ih" out.mp4
```

### Extract a frame as PNG

```bash
# Single frame at timestamp
ffmpeg -y -ss 00:00:03 -i in.mp4 -frames:v 1 frame.png

# Every Nth frame as a sequence
ffmpeg -y -i in.mp4 -vf "select='not(mod(n,30))'" -vsync vfr frame_%03d.png
```

### Add a watermark / image overlay

```bash
ffmpeg -y -i in.mp4 -i logo.png \
  -filter_complex "overlay=W-w-20:H-h-20" \
  out.mp4
```

`W,H` = main video dims, `w,h` = overlay dims; this places the logo
20px from the bottom-right corner.

### Render a still image into a video

Useful when you have a single PNG and need a video clip (e.g. to feed
into a concat or crossfade pipeline):

```bash
ffmpeg -y -loop 1 -i still.png -t 5 -vf "fps=30" \
  -c:v libx264 -pix_fmt yuv420p still.mp4
```

## Quality / Encoding Reference

| Need | Flag |
|------|------|
| Visually lossless H.264 | `-c:v libx264 -preset slow -crf 18` |
| Web-friendly small file | `-c:v libx264 -preset fast -crf 23 -movflags +faststart` |
| Apple-compatible / iMessage | add `-pix_fmt yuv420p` |
| Audio: AAC stereo 44.1kHz | `-c:a aac -ar 44100 -ac 2 -b:a 192k` |
| Strip metadata | `-map_metadata -1` |

`crf` 18 ≈ visually lossless, 23 ≈ default web quality, 28 ≈ small/lossy.
Lower number = bigger file.

## Common Pitfalls

1. **Mixing `-c copy` with filter chains** — filters require re-encoding.
   `-c copy` only works when not transforming.
2. **Concat with mismatched codecs/resolutions** — the demuxer concat
   silently fails or produces a broken file. Use the filter form.
3. **Forgetting `-shortest`** — without it, mux output runs as long as
   the longest input (silent video tail or repeated audio).
4. **Audio sync drift on long clips** — for clips >10 minutes use
   `-async 1` or `-vsync cfr` to enforce constant framerate.
5. **`-ss` after `-i` is slow** — frame-accurate but full decode. Put it
   before `-i` for keyframe seek (10–1000× faster).

## Workflow Pattern

```python
session_dir = get_session_dir()
output = f"{session_dir}/final.mp4"

result = run(command=(
    f'ffmpeg -y -loglevel error '
    f'-i "{video}" -i "{audio}" '
    f'-c:v copy -c:a aac -shortest '
    f'"{output}"'
))

if result.get("exit_code", 0) != 0:
    print("ffmpeg failed:", result.get("stderr", ""))
else:
    print(f"wrote: {output}")
```

`run` returns `{"stdout": str, "stderr": str, "exit_code": int}`. Check
`exit_code` — ffmpeg writes its progress to stderr even on success, so
non-empty stderr alone doesn't mean failure.
