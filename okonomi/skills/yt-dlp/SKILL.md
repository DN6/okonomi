---
name: yt-dlp
description: "Download videos, audio, time-slices, and metadata from YouTube and 1000+ other sites via the `yt-dlp` CLI. Trigger when the user asks to: download / save / rip / fetch a video or audio from a URL (YouTube, Vimeo, TikTok, Twitter/X, Instagram, SoundCloud, Twitch, Bilibili, etc.), extract audio (m4a/mp3) from a video URL, grab a clipped section, or pull metadata. Triggers even if the user doesn't say 'yt-dlp' — any 'download this video' request with a URL applies. Source: https://github.com/yt-dlp/yt-dlp"
metadata:
  tags: [yt-dlp, download, video, audio, youtube]
  related_skills: [ffmpeg]
---

# yt-dlp

Recipes for the four things an okonomi flow actually needs: a video, an audio track, a time-slice, or some metadata. Anything beyond these four — playlists, subtitles, thumbnails, archives, custom output templates — exists in the CLI; reach for `yt-dlp --help | grep <keyword>` when you need it.

Recipes live as `### H3` subsections under `## How to use`. Pull just the recipe you need with `load_skill_section("yt-dlp", "<slug>")` — slugs are flat and unique within the skill, so you don't need to know the parent.

## When to use

- Downloading video or audio from any URL: YouTube, Vimeo, Twitter/X, TikTok, Instagram, SoundCloud, Twitch, Bilibili, etc.
- Extracting an audio track (m4a / mp3) from a video URL.
- Time-slicing a section of audio or video without grabbing the whole file.
- Pulling metadata (title, duration, available formats) from a URL.

**Not for**: editing once downloaded (use `ffmpeg`), HTTP-seek frame extraction for analysis (use `video-analysis`).

## How to use

Invoke through the `bash` tool:

```
bash(command='yt-dlp -o "{out_dir}/%(title)s.%(ext)s" "{url}"')
```

- Save outputs into `$OKONOMI_SESSION_DIR` so they're tracked in the session.
- Always quote URLs — they often contain `&`, `?`, `=`.
- Add `--no-playlist` when given a video URL inside a playlist context, so you don't accidentally pull the whole channel.
- Default to 480p mp4 (see `download-a-video-480p-mp4`) unless the user explicitly asks for higher quality.
- For long downloads, pass `background=True` to `bash` and you'll get a `term-N` handle plus a system note on completion.

Pick a recipe via `load_skill_section("yt-dlp", "<slug>")` — recipes are independent and self-contained.

### Download a video (480p, mp4)

Right-sized for analysis, preview, and most generative-AI workflows. Bump the height filter only when the user explicitly asks for higher quality — bandwidth, decode time, and disk usage scale fast above 720p.

```bash
yt-dlp -f "bv*[height<=480]+ba/b[height<=480]" \
  --merge-output-format mp4 -o "%(title)s.%(ext)s" "URL"
```

Drop the `+ba` and `--merge-output-format` if audio is irrelevant (visual analysis, frame extraction).

### Extract audio (m4a)

YouTube ships AAC; `m4a` skips re-encoding and is the fastest, lossless-relative-to-source option.

```bash
yt-dlp -x --audio-format m4a -o "%(title)s.%(ext)s" "URL"
```

Use `--audio-format mp3` only when something downstream specifically requires MP3.

### Time-slice (audio or video)

Two paths — pick by the size of the source.

**Simple — let yt-dlp slice.** Lands on segment boundaries (HLS/DASH typically 2–6s); the final file is just the requested range.

```bash
# 90s of audio starting at 1:30 as m4a
yt-dlp -x --audio-format m4a \
  --download-sections "*00:01:30-00:03:00" \
  -o "%(title)s.%(ext)s" "URL"

# Same for video
yt-dlp -f "bv*[height<=480]+ba/b[height<=480]" \
  --download-sections "*00:01:30-00:02:45" \
  -o "%(title)s.%(ext)s" "URL"
```

Multiple ranges work: `--download-sections "*0:30-1:00" --download-sections "*5:00-5:30"`. Add `--force-keyframes-at-cuts` for frame-accurate boundaries (slower — re-encodes).

**Efficient on long sources** — bypass yt-dlp's downloader. `yt-dlp -g` resolves the direct CDN URL, ffmpeg fetches just the bytes around the requested range via HTTP byte-range requests. Minimal bandwidth even on a 2-hour podcast.

```bash
URL_DIRECT=$(yt-dlp -f "ba" -g "URL")           # audio-only stream
ffmpeg -y -ss 00:30:00 -t 90 -i "$URL_DIRECT" -c copy slice.m4a
```

Critical: `-ss` must come *before* `-i` — that's what makes ffmpeg seek via Range headers instead of decoding from the start. The resolved URL expires within minutes; re-resolve right before each ffmpeg call. For video, use `-f "bv*[height<=480]"` instead of `"ba"`.

### Metadata only (no download)

```bash
yt-dlp --dump-json --no-warnings "URL"           # full JSON, pipe to jq
yt-dlp --print "%(title)s|%(duration)s" "URL"    # custom one-liner
yt-dlp --list-formats "URL"                      # what's actually available
```

## Pitfalls

- **403 / throttling**: extractors break frequently — first try `yt-dlp -U` to update. If that doesn't help, add `--sleep-requests 1 --sleep-interval 5 --max-sleep-interval 30`.
- **Live streams**: `--live-from-start` to capture from the beginning, otherwise yt-dlp starts at the current point.
- **Slow large downloads**: `-N 4` enables 4 parallel fragment downloads (HLS/DASH only).
- **Auth / age-restricted**: `--cookies-from-browser chrome` (or `safari`, `firefox`) — pulls cookies from your browser session, no manual export.
