# Post-Production & Stitching

## Video Concatenation with ffmpeg

All stitching uses ffmpeg via the `run` standalone tool.

### Basic Concatenation (Same Format)
When clips share resolution and codec, use the concat demuxer for lossless joining:

```python
import os
session_dir = get_session_dir()

clips = ["scene_1.mp4", "transition_1.mp4", "scene_2.mp4", "transition_2.mp4", "scene_3.mp4"]
filelist = os.path.join(session_dir, "filelist.txt")
with open(filelist, "w") as f:
    for clip in clips:
        f.write(f"file '{os.path.join(session_dir, clip)}'\n")
```

Then stitch:
```
run(command="ffmpeg -y -f concat -safe 0 -i {session_dir}/filelist.txt -c copy {session_dir}/final.mp4")
```

### Re-encoding (Mixed Formats)
When clips have different resolutions or codecs, re-encode to a common format with scaling and padding:

```
run(command="ffmpeg -y -f concat -safe 0 -i {session_dir}/filelist.txt -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -vf 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2' {session_dir}/final.mp4")
```

### Crossfade Between Clips
Smooth blend transition (1-second fade):

```
run(command="ffmpeg -y -i {session_dir}/clip1.mp4 -i {session_dir}/clip2.mp4 -filter_complex 'xfade=transition=fade:duration=1:offset=4' -c:v libx264 -pix_fmt yuv420p {session_dir}/merged.mp4")
```

`offset` = duration of first clip minus crossfade duration.

### Available xfade Transitions
Beyond `fade`, ffmpeg supports: `wipeleft`, `wiperight`, `wipeup`, `wipedown`, `slideleft`, `slideright`, `slideup`, `slidedown`, `circlecrop`, `rectcrop`, `distance`, `dissolve`, `pixelize`, `diagtl`, `diagtr`, `diagbl`, `diagbr`, `hlslice`, `hrslice`, `vuslice`, `vdslice`, `hblur`, `fadegrays`, `squeezev`, `squeezeh`, `zoomin`.

### Chaining Multiple Crossfades
For 3+ clips with crossfades, chain xfade filters:

```
run(command="ffmpeg -y -i clip1.mp4 -i clip2.mp4 -i clip3.mp4 -filter_complex '[0][1]xfade=transition=fade:duration=1:offset=4[v1];[v1][2]xfade=transition=fade:duration=1:offset=8' -c:v libx264 -pix_fmt yuv420p final.mp4")
```

Each subsequent offset accounts for previous durations minus crossfade overlaps.

## Audio

### Adding a Music Track
Layer music or ambient audio, video codec copied (fast):

```
run(command="ffmpeg -y -i {session_dir}/final.mp4 -i {session_dir}/music.mp3 -c:v copy -c:a aac -shortest {session_dir}/final_with_audio.mp4")
```

### Replacing Audio Entirely
Strip existing audio and replace:

```
run(command="ffmpeg -y -i {session_dir}/final.mp4 -i {session_dir}/narration.mp3 -c:v copy -c:a aac -map 0:v -map 1:a -shortest {session_dir}/final_narrated.mp4")
```

### Mixing Audio Tracks
Combine original audio with background music at lower volume:

```
run(command="ffmpeg -y -i {session_dir}/final.mp4 -i {session_dir}/music.mp3 -filter_complex '[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=shortest' -c:v copy {session_dir}/final_mixed.mp4")
```

### Stripping Audio
Remove audio track entirely:

```
run(command="ffmpeg -y -i {session_dir}/input.mp4 -an -c:v copy {session_dir}/silent.mp4")
```

## Color Grading & Filters

### Basic Color Adjustment
```
run(command="ffmpeg -y -i input.mp4 -vf 'eq=brightness=0.05:contrast=1.1:saturation=1.2' -c:a copy output.mp4")
```

### LUT Application
Apply a .cube LUT for cinematic color grading:
```
run(command="ffmpeg -y -i input.mp4 -vf 'lut3d=grade.cube' -c:a copy output.mp4")
```

### Speed Adjustment
Slow motion (0.5x):
```
run(command="ffmpeg -y -i input.mp4 -vf 'setpts=2*PTS' -af 'atempo=0.5' output.mp4")
```

Speed up (2x):
```
run(command="ffmpeg -y -i input.mp4 -vf 'setpts=0.5*PTS' -af 'atempo=2.0' output.mp4")
```

### Adding Text Overlay
Burn-in a title:
```
run(command="ffmpeg -y -i input.mp4 -vf \"drawtext=text='Title Here':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=h-100:shadowcolor=black:shadowx=2:shadowy=2\" -c:a copy output.mp4")
```
