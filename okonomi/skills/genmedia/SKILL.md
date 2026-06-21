---
name: genmedia
description: "REQUIRED before any image / video / audio / 3D generation. Single namespace (`genmedia.run`, `genmedia.search`, `genmedia.schema`) covers 1000+ fal.ai endpoints; auto-tracks every produced asset into the session. Load FIRST when users request: text-to-image, image edits, image-to-video, frame-to-frame video, music, sound effects, speech, image-to-3d, lip sync, or any other fal endpoint."
metadata:
  tags: [fal, genmedia, image, video, audio, 3d, generation]
  related_skills: [p5js, ffmpeg]
---

# genmedia

`genmedia` is a programmatic tool in `execute_code`'s globals. Three methods, all synchronous from Python's perspective. Long-running
generations (multi-minute video) become async at the agent level because okonomi auto-dispatches any `execute_code` block touching
`genmedia.` to a background thread — your code stays simple, the agent stays responsive.

| Method | Purpose |
|--------|---------|
| `genmedia.run(model_id, **arguments)` | Invoke an endpoint, return parsed JSON, auto-track produced URLs |
| `genmedia.search(query, limit=10)` | Search fal's catalog by keyword |
| `genmedia.schema(model_id)` | Get an endpoint's parameter schema (`--help` text) |

**Auto-tracking is built in.** Every URL `genmedia.run` returns is
already registered with the session — downloaded in the background,
visible in `/assets`, surfaced in bg-exec completion notifications.
No manual tracking needed.

Pull just the section you need with `load_skill_section("genmedia", "<slug>")` rather than the whole file. Deep creative guidance (image generation, image editing, video, camera, style, post-production) lives in `references/` — load on demand via `load_skill_reference("genmedia", "<name>")`.

## When to use

- Any AI image generation request (text-to-image, edit).
- Image-to-video, frame-to-frame video, video-to-video.
- Music, sound effects, TTS, lip-sync.
- Image-to-3D / mesh generation.
- Any other fal.ai endpoint — discover with `genmedia.search("<keyword>")`.

**Not for**: rendering HTML / canvas / WebGL sketches (use `browser.render_*` from p5js skill), local file transformations like resize / crop / format conversion (use `ffmpeg` or PIL inside `execute_code`), downloading from a non-fal URL (use `yt-dlp` or `bash`).

## How to use

Inside `execute_code` only. okonomi auto-detects `genmedia.` calls and dispatches them to a background thread (you'll see a `bg-exec-N` handle and an approval prompt).

```python
result = genmedia.run(
    "fal-ai/flux/dev",
    prompt="a noir detective in a rainy bar, 35mm film",
    image_size="portrait_16_9",
    num_images=1,
)
url = result["images"][0]["url"]
print(url)
```

Three methods:
- `genmedia.run(model_id, **arguments)` — invoke an endpoint, return parsed JSON, auto-track URLs.
- `genmedia.search(query, limit=10)` — search fal's catalog by keyword (use this when you don't know the right model id).
- `genmedia.schema(model_id)` — get an endpoint's parameter schema before calling.

After dispatch: acknowledge in one short sentence and stop. Do *not* narrate expected output — those claims are wrong on failure and unnecessary on success. You'll be re-invoked with a completion note when the bg job finishes.

Reach for deeper guidance via `load_skill_reference("genmedia", "<topic>")` — `image-generation`, `image-editing`, `video-generation`, `camera-and-framing`, `visual-style`, `post-production`.

### One example per asset type

For deeper recipes per category, load the dedicated skill (see "When to
load companion skills" below). The core skill keeps a single canonical
example for each so you have something to reach for immediately.

```python
# Image
r = genmedia.run("fal-ai/flux/dev", prompt=p, image_size="landscape_16_9")
url = r["images"][0]["url"]

# Video (note: singular "video" key, not "videos")
r = genmedia.run("fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
                 image_url=src, prompt=motion_prompt, duration=5)
url = r["video"]["url"]

# Audio (music / sfx / TTS — varied keys)
r = genmedia.run("fal-ai/stable-audio", prompt=p, seconds_total=12)
url = r["audio_file"]["url"]

# 3D
r = genmedia.run("fal-ai/trellis", image_url=src)
url = r["model_mesh"]["url"]
```

### Discovery

For any endpoint not in the dedicated skills, discover at runtime:

```python
matches = genmedia.search("music generation")
print(matches)

# Then peek at parameters before calling
print(genmedia.schema("fal-ai/stable-audio"))
```

### Return object

`genmedia.run(...)` returns a dict — the **fal endpoint's response payload, already unwrapped from the genmedia CLI envelope.** The CLI itself emits `{"status", "endpoint_id", "request_id", "result"}`, but the wrapper strips that outer shell and hands you the inner `result` directly. So `r["images"][0]["url"]` works on the first try; you do *not* need `r["result"]["images"]`.

### Top-level shape (after unwrap)

The keys depend on the endpoint. fal's universal asset envelope is `{"url": "..."}` (sometimes with `width`, `height`, `content_type`, `file_size` siblings) — what varies is *where* that envelope lives.

| Endpoint family | Top-level key | Asset shape |
|---|---|---|
| Most text-to-image, multi-result | `r["images"]` | `list[{"url", "width", "height", ...}]` |
| Single-image result endpoints | `r["image"]` | `{"url", ...}` |
| Image-to-video, frame-to-frame | `r["video"]` | `{"url", "duration", ...}` |
| Music / SFX / TTS (varies by model) | `r["audio_file"]` or `r["audio_url"]` or `r["audio"]` | `{"url", ...}` or a bare URL string |
| Image-to-3D | `r["model_mesh"]` | `{"url", ...}` (glb/gltf) |
| LoRA-aware image endpoints | `r["images"]` + `r["seed"]`, `r["timings"]`, `r["has_nsfw_concepts"]` | as above |

### Envelope metadata (kept on the unwrapped dict)

The wrapper preserves the CLI envelope fields as underscored keys, so you can reach them without re-parsing:

- `r["_status"]` — usually `"completed"` on success
- `r["_endpoint_id"]` — echo of the model id you ran
- `r["_request_id"]` — fal's request id (useful for `genmedia status` if you ever need to poll, or for debugging support requests)

These are added by the wrapper, not by fal — they only exist on `genmedia.run` results.

### Errors

- Subprocess / parse failures: `r == {"error": "<message>", ...}`. No `images`/`video`/etc. keys.
- Endpoint-side errors (auth, rate limit, validation): the CLI usually exits non-zero, surfaced as `{"error": "<stderr>"}`. Same shape.
- Always check for `"error"` *before* indexing into result keys when you're not certain a generation succeeded.

### Inspecting an unfamiliar endpoint

When in doubt, print first. The result variable persists across `execute_code` blocks (don't re-run a successful generation just to re-inspect):

```python
import json
r = genmedia.run("fal-ai/some-new-endpoint", prompt="...")
print(json.dumps(r, indent=2))   # learn the shape once
```

Or peek at the schema before calling at all:

```python
print(genmedia.schema("fal-ai/some-new-endpoint"))
```

## Pitfalls

1. **KeyError on result access** — fal endpoints disagree on key names
   (`images` vs `image`, `video` vs `videos`). If access fails, inspect
   the response with `print(json.dumps(r, indent=2))` in a *new*
   `execute_code` block. The generation already succeeded — its URL is
   tracked, no need to re-run.
2. **Re-running on parse error** — every re-run costs another fal call.
   Always inspect the result variable (it persists across `execute_code`
   blocks) before re-running.
3. **Forgetting `image_size`** — defaults vary per model. Set it
   explicitly when you have aspect-ratio expectations.
4. **Long jobs feel like freezes** — they're not. The bg-exec thread is
   working; the agent will get a completion notification when fal
   returns. Don't spam status checks.

## References

For deeper creative direction, consult these reference files in `references/` (read with `bash(command="cat <path>")` when you need them):

- **`image-generation.md`** — two-tier doctrine (Krea turbo for exploration → Krea large for the hero render → gpt-image-2 refinement), prompt craft layers, LoRA conventions, parameter cheat sheet, common workflows.
- **`image-editing.md`** — instruction-following edits with `gpt-image-2/edit`, structured edit prompts.
- **`video-generation.md`** — two-tier doctrine (ltx-2.3 ideation, seedance-2.0 refinement), production workflow, prompt anatomy, common mistakes.
- **`camera-and-framing.md`** — shot types, camera moves, lens vocabulary.
- **`visual-style.md`** — film stocks, lighting, color grading lookups.
- **`post-production.md`** — ffmpeg recipes for stitching, transitions, audio muxing.

Pull a reference into context only when the current task actually needs it — they're long. The above bullet list is the index; load specifics on demand.
