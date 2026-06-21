
# Image Editing Playbook

Mechanics (the `genmedia.*` methods, return shapes, auto-tracking) live
in the **genmedia core skill**. This skill is the *creative + decision*
layer: how to approach an edit and which parameters to set.

`genmedia.run(...)` auto-tracks every URL it returns into the session.

## The edit model

All image edits go through **one** model: `openai/gpt-image-2/edit`.
It's the Tier 2 refinement endpoint from the image-generation skill —
strong at compositional polish, color/lighting tweaks, object changes,
and "make this better" prompts on top of an existing image.

| User says… | Approach |
|------------|----------|
| "Refine this", "polish this", "make it pop", "tighten the composition" | Refinement edit |
| "Change the X to Y", "make it blue", "remove the dog" | Instruction-following edit |

Don't run an instruction edit when the user actually wants a fresh
re-render of the same subject — that's a text-to-image job (see the
image-generation skill).

## Instruction-following edit (refinement / polish / object change)

Pass the chosen source image as `image_url` and the change as `prompt`.

```python
edit = "change the wallpaper to forest green william morris print"
result = genmedia.run("openai/gpt-image-2/edit",
                      image_url=source_url, prompt=edit)
url = result["images"][0]["url"]
```

Tips:
- Lead the prompt with the **change**, not the description of what's
  staying. "Change the car to red" beats "a red car in the same scene".
- For multiple edits in one pass, list them: "change the wallpaper to
  forest green and add a brass lamp on the table". Don't exceed ~3
  edits per call — split into multiple passes.
- Edits are not perfectly preserving — fine details may shift.
- Iterate with successive edit prompts until the user signs off.

## Picking parameters

Call `genmedia.schema("openai/gpt-image-2/edit")` for the full schema.

| Kwarg | Notes |
|-------|-------|
| `image_url=...` | Source image (singular). Accepts a URL or data URI. |
| `prompt=...` | What you want changed. |
| `seed=...` | Set when iterating on the same edit. |

## Pitfalls

1. **Editing when you mean re-render** — if the user wants a different
   take on the same subject, that's text-to-image, not an edit.
2. **Editing an unsaved fal URL** — fal CDN URLs expire. If the user
   wants to edit something from an earlier session, look up its
   `local_path` in `/assets` (`genmedia.run` auto-downloaded it) and
   pass the local path directly — gpt-image-2/edit accepts data URIs
   built from local files.
