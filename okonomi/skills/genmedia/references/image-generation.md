
# Image Generation Playbook

Mechanics (the `genmedia.*` methods, return shapes, auto-tracking) live
in the **genmedia core skill**. This skill is the *creative* layer above
it: which model for which job, how to write a prompt that actually
delivers.

`genmedia.run(...)` auto-tracks the produced URL into the session.

## Two-tier doctrine

Image generation in okonomi runs through **two model families**: Krea
for **generation** — `medium/turbo` for fast variation sweeps and
mood-finding, then `large` for the high-fidelity render of a take that
lands — and GPT-image-2 for **editing and refinement**
(instruction-following polish on a chosen result). Don't escalate to
the large render until a sweep has produced something the user wants to
keep, and don't escalate to refinement until that render is right.

### Tier 1 — Exploration with Krea

`fal-ai/krea/v2/medium/turbo/text-to-image` — the default for every
first-attempt image task. Trained on Krea's curated aesthetic data,
so it delivers a strong "looks good by default" baseline without
elaborate prompts.

Use for concept sweeps, mood exploration, "show me 4 takes",
thumbnails, and drafts. Run with `num_images=2-4` to compare
directions.

```python
result = genmedia.run(
    "fal-ai/krea/v2/medium/turbo/text-to-image",
    prompt=prompt,
    image_size="portrait_16_9",
    num_images=4,
)
```

**Quality follow-up — `fal-ai/krea/v2/large/text-to-image`.** Once a
sweep produces a take the user wants to keep, re-render *that* image at
full fidelity with the large model. Reuse the same prompt and fix the
`seed` from the chosen turbo result so the composition carries over —
the large model is slower but sharper, so use it for the hero /
deliverable, not the sweep. This is the natural next step after a
successful image, before reaching for any edits.

```python
chosen = result["images"][0]          # the take the user picked
result = genmedia.run(
    "fal-ai/krea/v2/large/text-to-image",
    prompt=prompt,
    image_size="portrait_16_9",
    seed=result["seed"],              # lock the composition from the sweep
)
url = result["images"][0]["url"]
```


### Tier 2 — Editing & refinement with GPT-image-2

`openai/gpt-image-2/edit` — instruction-following edit on top of an
existing image. Best at "make this hero shot pop", "fix the eyes",
"tighten the composition", "soften the shadows", "make it more
cinematic". Pass the chosen exploration URL as `image_url` and the
refinement instructions as `prompt`. Iterate with successive edit
prompts until the user signs off.

### Workflow

1. **EXPLORE** — `fal-ai/krea/v2/medium/turbo/text-to-image`, `num_images=4`, free
   seed. Present the variations and let the user choose.
2. **RENDER** — re-run the chosen take's prompt through
   `fal-ai/krea/v2/large/text-to-image` with its `seed` locked, for the
   full-fidelity version of the successful image.
3. **REFINE** — take the rendered image's URL and run it through
   `openai/gpt-image-2/edit` with refinement prompts. Single image,
   keep iterating with new edit instructions until the user signs off.

Don't reach outside these models for first-attempt image work. If
the user has a special request the pair can't cover (illustration,
anime, vector, niche style), call `genmedia.search("<keyword>")` to
discover an alternative — but document the rationale to the user
before spending on it.

## Prompt craft

A great prompt has five layers, in this order:

1. **Subject** — what is the thing? "a low-rise apartment", "a woman",
   "a still life of fruit"
2. **Action / pose / state** — "crossing a street", "looking down",
   "rotting on a windowsill"
3. **Style** — "35mm film", "cyanotype", "oil painting", "vaporwave",
   "Wes Anderson cinematography"
4. **Composition** — "low-angle close-up", "wide shot, rule of thirds",
   "symmetric centered framing"
5. **Lighting + atmosphere** — "neon under rain", "golden hour rim
   light", "dim tungsten"

Compress these into one fluent sentence rather than a checklist. Models
respond better to "a portrait of a noir detective in a smoke-filled bar,
35mm film, low-key tungsten lighting, melancholy expression" than the
same content in keyword form.

### Negative space

Don't tell the model what to leave out unless you've seen it produce the
unwanted element first. "A cat, no dog" often produces a dog. Just write
the positive description.

### Length

40–80 words is the sweet spot for prompts 


## Parameter cheat sheet

Pass these as keyword arguments to `genmedia.run(...)`.

| Kwarg | When to set it |
|-------|----------------|
| `image_size=...` | `"landscape_16_9"` for cinematic / wallpapers; `"portrait_16_9"` for character / fashion / vertical screens; `"square_hd"` for social tiles; `"landscape_4_3"` for editorial / classic photo |
| `num_images=...` | 1 by default; 2-4 when the user wants to pick from variations |
| `seed=...` | Set explicitly only when iterating on a single composition (so re-renders converge). Otherwise leave unset for variety. |
| `guidance_scale=...` | Default 3.5 for FLUX. Lower (2.5) for "loose / artistic"; higher (5-7) for "obey the prompt literally" |
| `num_inference_steps=...` | Default 28-40. Higher = more detail at the cost of time. Don't push past 50. |

Model-specific params (anything not in the above table) — call
`genmedia.schema("<model-id>")` to see the full schema.

## Common workflows

### Stage 1 — Exploration sweep

```python
prompt = (
    "A noir detective at a rain-soaked bar, smoke curling under a "
    "single tungsten lamp, 35mm film, melancholy, shallow focus"
)
result = genmedia.run(
    "fal-ai/krea/v2/medium/turbo/text-to-image",   # tier 1: Krea for exploration
    prompt=prompt,
    image_size="portrait_16_9",
    num_images=4,                   # sweep — let the user pick
)
for i, img in enumerate(result["images"]):
    print(f"  variation {i+1}: {img['url']}")
```

### Stage 2 — High-fidelity render of the chosen take

Once the user picks a variation, re-render it on the large Krea model —
same prompt, seed locked from the sweep so the composition holds:

```python
result = genmedia.run(
    "fal-ai/krea/v2/large/text-to-image",   # tier 1: Krea large for the hero render
    prompt=prompt,
    image_size="portrait_16_9",
    seed=result["seed"],
)
hero_url = result["images"][0]["url"]
```

### Stage 3 — Refinement on the rendered image

If the hero render still needs tweaks, polish it via edit rather than
re-rolling:

```python
chosen_url = hero_url
refinement = (
    "Tighten the framing, deepen the rim light on the detective's face, "
    "add a faint reflection on the bar surface."
)
result = genmedia.run(
    "openai/gpt-image-2/edit",   # tier 2 — refinement
    image_url=chosen_url,
    prompt=refinement,
)
url = result["images"][0]["url"]
```

Iterate with successive edit prompts until the user signs off.

### Iterating with a fixed seed (within Tier 1)

If you want to refine within exploration rather than jump to Tier 2 —
say the user wants three subtle prompt variants on the same
composition — fix the seed in Krea calls:

```python
seed = 42
result = genmedia.run("fal-ai/krea/v2/medium/turbo/text-to-image", prompt=prompt_v1,
                      seed=seed, image_size="portrait_16_9")
result = genmedia.run("fal-ai/krea/v2/medium/turbo/text-to-image", prompt=prompt_v2,
                      seed=seed, image_size="portrait_16_9")
```

## After refinement

Don't re-generate from scratch when an edit will do — run `gpt-image-2/edit`
on the chosen result rather than re-rolling a new prompt.

## Pitfalls

1. **Skipping exploration** — going straight to Tier 2 refinement before
   the user has chosen a direction means refining a guess. Always
   sweep first, get sign-off, then refine.
2. **Refining by re-generating** — once you have a chosen exploration
   output, run edits on it via `gpt-image-2/edit`. Re-running text-to-
   image with a "better" prompt produces a *different* image, not a
   refined version.
3. **Re-rolling instead of inspecting** — if the first image is "close
   but the lighting's wrong", change the prompt and re-run with the
   same seed; don't shotgun more random seeds hoping one is better.
4. **Ignoring image_size** — generating a 16:9 wallpaper at the default
   `square_hd` and then cropping is a waste. Set the right size up
   front.
5. **Treating fal CDN URLs as permanent** — they expire within hours.
   `genmedia.run` auto-tracks every URL it returns into the session
   (eager background download), so they survive. If you ever need a
   URL that didn't go through `genmedia.run`, save it to disk inside
   `get_session_dir()` yourself and reference the local path.
