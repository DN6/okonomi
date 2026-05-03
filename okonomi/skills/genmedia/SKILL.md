---
name: genmedia
description: "Production pipeline for image, video, audio, and 3D generation via fal.ai. Uses the maintained `genmedia` CLI (one binary, every fal endpoint) to discover models, run jobs, poll status, and download results. Trigger when users request: text-to-image, image-to-image, image edits, image upscale (e.g. SeedVR), image segmentation, image-to-video, first-last-frame video, music or sound effects, speech, image-to-3D, lip sync, or any other fal model — including endpoints not previously seen. Replaces the older `image_generation.*` / `video.*` Python wrappers."
metadata:
  tags: [fal, genmedia, image-generation, video-generation, audio-generation, upscale, segmentation, 3d]
  related_skills: [video-generation, p5js, ffmpeg, concept-research]
---

# genmedia: fal.ai via CLI

`genmedia` is a single CLI binary that fronts every fal.ai endpoint.
Auth via `FAL_KEY` (already exported in this environment). Use it from
inside `execute_code` via the standalone `run` tool.

> **Required:** after every successful generation, call `track_output(url, asset_type, prompt=..., model_id=...)` so the asset is downloaded to the session directory and surfaces in `/assets`. Skipping this means the asset is *not* tracked — it will be lost when the URL expires.

## The four CLI verbs

| Verb | Purpose | Example |
|------|---------|---------|
| `genmedia models "<query>"` | Discover endpoints by keyword | `genmedia models "image upscale"` |
| `genmedia run <id> --json …`  | Invoke an endpoint, return JSON | `genmedia run fal-ai/flux/dev --prompt "..." --json` |
| `genmedia status <request-id>` | Poll a long-running async job | `genmedia status req_abc123` |
| `genmedia download <request-id> --output <path>` | Pull artifacts to disk | `genmedia download req_abc123 --output out.mp4` |

`--json` is **mandatory** when invoking from code — without it the
output is human-prose and can't be parsed reliably.

## Calling pattern

Every generation follows the same five-step shape:

```python
import json

# 1. (optional) discover an endpoint
discover = run(command='genmedia models "image generation flux"')
print(discover.get("stdout", ""))

# 2. invoke
result = run(command=(
    'genmedia run fal-ai/flux/dev --json '
    '--prompt "cyberpunk dancer, neon, rain, 35mm" '
    '--num_images 1 '
    '--image_size landscape_16_9'
))

if result.get("exit_code", 0) != 0:
    print("genmedia failed:", result.get("stderr", ""))
else:
    # 3. parse the JSON the CLI emitted
    payload = json.loads(result["stdout"])

    # 4. extract the URL(s). fal endpoints differ — most images return
    #    {"images": [{"url": ...}]}; videos return {"video": {"url": ...}};
    #    audio often returns {"audio": {"url": ...}}.
    url = payload["images"][0]["url"]

    # 5. REQUIRED: register the asset with the session
    track_output(
        url=url,
        asset_type="image",
        prompt="cyberpunk dancer, neon, rain, 35mm",
        model_id="fal-ai/flux/dev",
    )
    print(f"Generated: {url}")
```

That `track_output` call is what makes the asset:
- Show up in `/assets`
- Get downloaded to `{session_dir}/assets/...` in the background
- Survive when the fal CDN URL expires

If you produce N URLs from one call (e.g. `--num_images 4`), call
`track_output` N times — once per URL.

## Canonical recipes

### Text-to-image

```python
result = run(command=(
    'genmedia run fal-ai/flux/dev --json '
    f'--prompt {json.dumps(prompt)} '
    '--image_size landscape_16_9 --num_images 1'
))
payload = json.loads(result["stdout"])
url = payload["images"][0]["url"]
track_output(url, "image", prompt=prompt, model_id="fal-ai/flux/dev")
```

### Image edit (instruction-following)

```python
result = run(command=(
    'genmedia run fal-ai/flux/dev/redux --json '
    f'--image_url {source_url} '
    f'--prompt {json.dumps(edit_instruction)}'
))
payload = json.loads(result["stdout"])
url = payload["images"][0]["url"]
track_output(url, "image", prompt=edit_instruction, model_id="fal-ai/flux/dev/redux")
```

### Reference-driven generation

```python
# Pass multiple reference URLs; supports up to 8 references on most models
refs = " ".join(f'--image_urls {u}' for u in [ref1_url, ref2_url])
result = run(command=(
    f'genmedia run fal-ai/flux-pro/kontext --json '
    f'{refs} --prompt {json.dumps(prompt)}'
))
```

### Image upscale (SeedVR)

```python
result = run(command=(
    'genmedia run fal-ai/seedvr/upscale/image --json '
    f'--image_url {source_url} '
    '--upscale_factor 4'
))
payload = json.loads(result["stdout"])
url = payload["image"]["url"]   # NOTE: singular "image" here, not "images"
track_output(url, "image", model_id="fal-ai/seedvr/upscale/image")
```

### Image-to-video

```python
result = run(command=(
    'genmedia run fal-ai/kling-video/v2.5-turbo/pro/image-to-video --json '
    f'--image_url {source_url} '
    f'--prompt {json.dumps(motion_prompt)} '
    '--duration 5'
))
payload = json.loads(result["stdout"])
url = payload["video"]["url"]   # NOTE: "video" key, not "videos"
track_output(url, "video", prompt=motion_prompt, model_id="fal-ai/kling-video/v2.5-turbo/pro/image-to-video")
```

### First-last-frame video

```python
result = run(command=(
    'genmedia run fal-ai/runway-gen3/turbo/first-and-last-frame --json '
    f'--first_frame_url {first_url} '
    f'--last_frame_url {last_url} '
    f'--prompt {json.dumps(prompt)}'
))
payload = json.loads(result["stdout"])
url = payload["video"]["url"]
track_output(url, "video", prompt=prompt, model_id="fal-ai/runway-gen3/turbo/first-and-last-frame")
```

### Music / audio generation

```python
result = run(command=(
    'genmedia run fal-ai/stable-audio --json '
    f'--prompt {json.dumps(prompt)} '
    '--seconds_total 12'
))
payload = json.loads(result["stdout"])
url = payload["audio_file"]["url"]
track_output(url, "audio", prompt=prompt, model_id="fal-ai/stable-audio")
```

### Speech (TTS)

```python
result = run(command=(
    'genmedia run fal-ai/elevenlabs/tts/turbo-v2.5 --json '
    f'--text {json.dumps(text)} '
    f'--voice {voice}'
))
payload = json.loads(result["stdout"])
url = payload["audio_url"]["url"] if isinstance(payload.get("audio_url"), dict) else payload["audio_url"]
track_output(url, "audio", prompt=text, model_id="fal-ai/elevenlabs/tts/turbo-v2.5")
```

### Image-to-3D

```python
result = run(command=(
    'genmedia run fal-ai/trellis --json '
    f'--image_url {source_url}'
))
payload = json.loads(result["stdout"])
url = payload["model_mesh"]["url"]
track_output(url, "3d", model_id="fal-ai/trellis")
```

## Discovery

Don't have an endpoint id? Search:

```python
result = run(command='genmedia models "lip sync"')
print(result["stdout"])
```

Pick a model from the list, then see its parameter schema:

```python
result = run(command='genmedia run fal-ai/sync-lipsync --help')
print(result["stdout"])
```

The CLI prints each parameter with type, required/optional, and a
description. Use that to build the `--param value` flags — never guess.

## Long-running jobs

For most endpoints `genmedia run` blocks until the job finishes and
returns the JSON result. For very long jobs you can run async and
poll:

```python
# Submit
result = run(command=(
    'genmedia run fal-ai/some-slow-model --json --async --prompt "..."'
))
request_id = json.loads(result["stdout"])["request_id"]

# Poll
import time
while True:
    status = run(command=f'genmedia status {request_id} --json')
    state = json.loads(status["stdout"])
    if state["status"] == "COMPLETED":
        break
    time.sleep(5)
```

In okonomi, prefer the *blocking* form — okonomi's bg-exec auto-dispatch
already runs the whole `execute_code` block in a background thread, so
the agent stays responsive even on a multi-minute video render. You'll
get a `[system] Background task bg-exec-N completed` notification when
it finishes.

## Pitfalls

1. **Forgetting `--json`** — output is unparseable prose. Always
   `--json` on `run` and `status` when invoking from code.
2. **Forgetting `track_output`** — asset shows up in the response but
   not in `/assets`, and the URL expires before you can save it. Call
   it once per URL produced.
3. **Wrong return key** — most image endpoints return `images: [...]`,
   most video endpoints return `video: {url: ...}` (singular), audio
   varies. Either inspect `print(json.dumps(payload, indent=2))` once
   to learn the shape, or use `--help` on the endpoint to see the
   documented response schema.
4. **Quoting prompts** — the prompt is a CLI argument so it must be
   shell-quoted. Use `json.dumps(prompt)` — it produces valid
   double-quoted output and escapes embedded quotes safely.
5. **Don't re-run on parse error** — if `json.loads(result["stdout"])`
   throws KeyError downstream, the generation already succeeded. Do
   `print(json.dumps(payload, indent=2))` in a *new* execute_code block
   to inspect the shape and fix the access pattern. Re-running burns
   another fal call for nothing.

## Workflow with okonomi's bg-exec

`genmedia run` matches the agent's expensive-pattern detection, so any
`execute_code` block containing it is:

1. Approval-gated (the user sees the code, expanded by default, and
   approves with Enter or denies with `n`)
2. Auto-dispatched to a background thread once approved — the model
   gets `Dispatched as bg-exec-N` and moves on
3. When the bg thread completes, a `[system] Background task bg-exec-N
   completed` notification is injected as the next turn, listing every
   asset `track_output` registered

So a complete one-shot generation is just:

```python
result = run(command=(
    'genmedia run fal-ai/flux/dev --json --prompt "noir portrait" '
    '--image_size portrait_16_9'
))
payload = json.loads(result["stdout"])
url = payload["images"][0]["url"]
track_output(url, "image", prompt="noir portrait", model_id="fal-ai/flux/dev")
```

Everything after — download, persistence, UI rendering, completion
notification — is automatic.
