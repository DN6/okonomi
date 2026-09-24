---
name: ffmpeg
description: "Recipe collection for ffmpeg post-production: mux audio, concatenate clips, crossfade transitions, trim, loop, speed change, format conversion (mp4 ↔ webm ↔ gif), scale/pad/crop, extract frames, overlay images, render still-to-video, and encoding presets. ffmpeg is already installed; invoke via the standalone `bash` tool. Trigger when users ask to combine/stitch clips, replace an audio track, fade between scenes, convert formats, trim, slow down or speed up, change aspect ratio, extract audio or frames, or add a watermark."
metadata:
  tags: [ffmpeg, video, audio, post-production, mux, concat, transitions]
---

# ffmpeg

Recipes live as ``### H3`` subsections under ``## How to use``. Pull just the recipe you need with ``load_skill_section("ffmpeg", "<slug>")`` — slugs are flat and unique within the skill, so you don't need to know the parent.

## When to use

- Combining / stitching video clips (concat, crossfade, transitions).
- Adding, replacing, or extracting an audio track.
- Trimming, looping, speeding up or slowing down a clip.
- Converting formats (mp4 ↔ webm ↔ gif).
- Scaling, padding, cropping, or aspect-ratio fixes.
- Extracting one or many frames as PNG.
- Adding watermarks / overlays.
- Rendering a still image into a video clip.

**Not for**: downloading from a URL (use `yt-dlp`), HTTP-seek frame extraction without downloading (use `video-analysis`).

## How to use

Invoke through the `bash` tool:

```
bash(command='ffmpeg -y -loglevel error -i "{input}" -i "{audio}" -c:v copy -c:a aac -shortest "{output}"')
```

Universal flags worth standardizing on:
- `-y` — overwrite output without prompting (always include).
- `-loglevel error` — suppress info chatter; only real errors come through.
- Save outputs into `$OKONOMI_SESSION_DIR` so they're tracked in the session.
- Quote filenames — paths can contain spaces, parentheses, etc.

`bash` returns `{"stdout", "stderr", "exit_code"}`. Check `exit_code` — ffmpeg writes progress to stderr even on success, so non-empty stderr alone does not indicate failure.

For long renders (>30s), pass `background=True` and you'll get a `term-N` task id; the result lands as a system note when finished.

Pick a recipe via `load_skill_section("ffmpeg", "<slug>")` — slugs match the section names below (e.g. `mux-audio-onto-a-video`, `crossfade-between-clips`, `convert-between-formats`).

### Mux audio onto a video

Replace or add an audio track. `-shortest` ends the output at whichever stream finishes first.

```bash
ffmpeg -y -i video.mp4 -i audio.wav \
  -c:v copy -c:a aac -shortest \
  output.mp4
```

To loop the video to match a longer audio track instead of trimming the audio, see `loop-a-video` (use `-stream_loop` on the video, then mux).

### Extract audio from a video

Pull audio out as a standalone file:

```bash
ffmpeg -y -i input.mp4 -vn -acodec copy output.aac     # keep original codec
ffmpeg -y -i input.mp4 -vn -ac 2 -ar 44100 output.wav  # decode to wav
```

`-vn` drops video. `-acodec copy` is instant (no re-encode).

### Concat (same codec and format)

Fastest path. Build a manifest, concat without re-encoding:

```bash
cat > list.txt <<EOF
file 'clip1.mp4'
file 'clip2.mp4'
file 'clip3.mp4'
EOF

ffmpeg -y -f concat -safe 0 -i list.txt -c copy out.mp4
```

If the clips don't share codec / resolution / framerate, this silently produces a broken file — use the filter-based concat instead (see `concat-mixed-sources`).

### Concat (mixed sources — re-encode)

Normalize through the concat filter. Slower but robust to differing codecs / resolutions / framerates:

```bash
ffmpeg -y -i a.mp4 -i b.mp4 -i c.mp4 \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" \
  out.mp4
```

`n=3` is the input count; `v=1:a=1` means each input contributes one video and one audio stream. Drop `[N:a]` and set `a=0` if your inputs have no audio.

### Crossfade between clips

`xfade` for video, `acrossfade` for audio. `offset` is when the fade *starts* in the first clip's timeline; `duration` is the fade length:

```bash
ffmpeg -y -i a.mp4 -i b.mp4 -filter_complex \
  "[0:v][1:v]xfade=transition=fade:duration=1:offset=4[v]; \
   [0:a][1:a]acrossfade=d=1[a]" \
  -map "[v]" -map "[a]" out.mp4
```

Other transitions: `slideleft`, `slideright`, `wipeleft`, `circleopen`, `pixelize`, `dissolve`, `radial`. Full list: `ffmpeg -h filter=xfade`.

### Trim a clip

Cut between two timestamps. Put `-ss` *before* `-i` for fast keyframe seek:

```bash
ffmpeg -y -ss 00:00:05 -to 00:00:12 -i input.mp4 -c copy clip.mp4
```

`-c copy` skips re-encoding (instant). Drop `-c copy` if you need frame-accurate cuts at non-keyframe boundaries (slower).

### Loop a video

Loop N times, or until matching a target duration:

```bash
ffmpeg -y -stream_loop 4 -i input.mp4 -c copy looped.mp4
ffmpeg -y -stream_loop -1 -i input.mp4 -t 30 -c copy looped30s.mp4
```

`-1` = infinite loop, then `-t` clips to the target length.

### Speed up or slow down

Video via `setpts` (PTS = presentation timestamp), audio via `atempo` (per-step limited to 0.5×–2.0×; chain for larger changes):

```bash
# 2x faster, silent
ffmpeg -y -i input.mp4 -filter:v "setpts=0.5*PTS" -an out.mp4

# 0.5x slower with audio
ffmpeg -y -i input.mp4 \
  -filter_complex "[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" out.mp4

# 4x faster audio (chain atempo)
ffmpeg -y -i input.wav -filter:a "atempo=2.0,atempo=2.0" out.wav
```

### Convert between formats

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

### Scale, pad, or crop

```bash
# Scale to 1080p, preserve aspect (-2 keeps a multiple-of-2 dim)
ffmpeg -y -i in.mp4 -vf "scale=1920:-2" out.mp4

# Letterbox to 16:9 1080p with black bars
ffmpeg -y -i in.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease, \
       pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  out.mp4

# Center-crop to a square
ffmpeg -y -i in.mp4 -vf "crop=ih:ih" out.mp4
```

### Extract frames as PNG

```bash
# Single frame at a timestamp
ffmpeg -y -ss 00:00:03 -i in.mp4 -frames:v 1 frame.png

# Every Nth frame as a sequence
ffmpeg -y -i in.mp4 -vf "select='not(mod(n,30))'" -vsync vfr frame_%03d.png
```

For HTTP-seek frame extraction without downloading a full video, see the `video-analysis` skill.

### Image overlay (watermark)

```bash
ffmpeg -y -i in.mp4 -i logo.png \
  -filter_complex "overlay=W-w-20:H-h-20" \
  out.mp4
```

`W,H` are the main video dims; `w,h` are the overlay dims. The example positions the logo 20px from the bottom-right corner. Use `overlay=20:20` for top-left.

### Still image to video

When you need a video clip from a single image — e.g. to feed a concat or crossfade pipeline:

```bash
ffmpeg -y -loop 1 -i still.png -t 5 -vf "fps=30" \
  -c:v libx264 -pix_fmt yuv420p still.mp4
```

`-loop 1` repeats the still indefinitely; `-t 5` truncates to 5 seconds.

### Encoding presets

| Need | Flag |
|---|---|
| Visually lossless H.264 | `-c:v libx264 -preset slow -crf 18` |
| Web-friendly small file | `-c:v libx264 -preset fast -crf 23 -movflags +faststart` |
| Apple / iMessage compatibility | add `-pix_fmt yuv420p` |
| Audio: AAC stereo 44.1kHz | `-c:a aac -ar 44100 -ac 2 -b:a 192k` |
| Strip metadata | `-map_metadata -1` |

`crf` 18 ≈ visually lossless, 23 ≈ default web quality, 28 ≈ small/lossy. Lower = bigger file.

## Pitfalls

1. **Mixing `-c copy` with filter chains** — filters require re-encoding. `-c copy` only works when you're not transforming the stream.
2. **Concat with mismatched codecs/resolutions/framerates** — the demuxer concat silently produces a broken file. Use the filter-based form.
3. **Forgetting `-shortest`** — mux output runs as long as the longest input (silent video tail or repeated audio).
4. **`-ss` after `-i`** — frame-accurate but slow (full decode to the timestamp). Put `-ss` *before* `-i` for keyframe seek (10–1000× faster).
5. **Audio sync drift on long clips** — for clips >10 minutes use `-async 1` or `-vsync cfr` to enforce constant framerate.
