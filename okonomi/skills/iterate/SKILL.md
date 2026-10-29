---
name: iterate
description: "Review generated media against the brief and iterate toward what the user wants, recording their picks as preferences. Covers: judging outputs against a concept brief or prompt package, naming specific gaps, changing one variable per round, presenting variants for the user to choose, keeping a series consistent, and filing picks/rejections into the mind wiki. Trigger after any generation the user will judge, when the user says 'not quite', 'closer', 'try again', 'more X', compares variants, or continues an existing series."
metadata:
  tags: [iteration, review, critique, variants, preferences, series, consistency]
---

# Iterate

Generation is rarely right the first time. Look at what came out, compare it against what was asked for, change one thing, and let the user's choices accumulate in the mind wiki so the next round starts closer.

The five steps and the pitfalls below are each their own section. Pull the one you need with `load_skill_section("iterate", "<slug>")`.

## When to use

- After a generation the user will judge, especially from a `research` brief or prompt package.
- The user reacts to an output: "not quite", "closer", "more grain", "the second one".
- Comparing variants, or choosing between directions.
- Continuing an existing series, character, or style across sessions.

**Not for**: developing a concept from a vague brief (use `research`), one-off generations the user won't iterate on.

## How to use

1. **Load context** — the brief, the series page, and the user's preferences. See `load-context`.
2. **Look** — read the outputs and name the gaps against the brief. See `look`.
3. **Change one thing** — one variable per round, everything else fixed. See `change-one-thing`.
4. **Ask** — fix objective gaps yourself; take taste calls to the user. See `ask`.
5. **Record** — file picks, rejections, and approved parameters into the mind wiki. See `record`.

### Load context

- The brief: the `research` synthesis page or prompt package for this project, if one exists.
- The series page: `recall(query="<series name>")`. If a page exists, `read_file` it and work from its locked elements — character description, reference image paths, core prompt, palette, model/LoRA, seeds, approved generations. Don't re-derive what's already locked.
- Preferences: `recall(query="preference")` lists pages of the user's past picks. Apply the ones relevant to this medium or style before the first generation, not after the user complains.

`recall` matches every query term, so keep queries to one or two words.

### Look

- Call `open_viewer()` once so the user sees outputs as they land.
- Mention each output path in your reply so the image is attached and you can actually see it. Don't judge an output you haven't looked at.
- Compare against the brief dimension by dimension: subject, composition, palette, lighting, texture, style. Name gaps concretely — "rim light missing, palette too saturated, subject cropped at the knees" — not "close but could be better".
- Also note what's working, so the next round doesn't lose it.

### Change one thing

Pick the single change most likely to close the biggest gap: prompt wording, a parameter, the seed, the model, a reference image, or post-processing.

- Hold everything else fixed — keep the seed when changing the prompt — so the difference is attributable.
- To explore, generate 2–4 variants along one axis (e.g. three grain strengths) and show them side by side.
- Version outputs in `$OKONOMI_SESSION_DIR` as `<name>_v1`, `<name>_v2`, … and never overwrite, so any earlier round can be compared or returned to.
- Keep a short log of what changed per version in your reply; the user is choosing between versions, not prompts.

### Ask

- **Objective gaps** — a missing element, wrong aspect ratio, artifacts, text errors, something the brief explicitly asked for: fix these without asking.
- **Taste calls** — which mood, which palette, which variant: present the options, say what differs between them, and let the user pick. Don't decide for them.
- **Stuck** — after ~3 rounds without real progress, the problem is the direction, not the execution. Say so and go back to `research` rather than tuning further.

### Record

Same rules as `research`: `remember` is the only way pages are written, and you update before you create — `recall` first, merge into an existing page with `update=True`.

1. **Preferences** — when the user picks or rejects something, file it on a page tagged `preference`: what was chosen over what, in which project, and why if they said. One page per preference, not per session (`grain-over-clean`, not `wizard-session-notes`).
2. **Series page** — when the user approves an output in a series, add it with its exact parameters (prompt, model, seed, references, post-processing). When they approve a change to a locked element, update the element. Create the page, tagged `series`, the first time something is approved.

```python
remember(
    title="Lofi Treatment Preference — VHS Still",
    slug="lofi-treatment-preference",
    tags=["preference", "lofi", "post-processing"],
    related=["wizard-series"],
    body="Chose the VHS still over pushed 35mm and super 8 for the wizard series. "
         "The look comes from chroma bleed + scanlines, not blur; keep the subject vibrant.",
    update=True,
)
```

## Pitfalls

- **Judging without looking.** Describing an output from its prompt instead of the image misses what actually came out.
- **Changing several things at once.** The next round improves or regresses and nobody knows why.
- **Overwriting versions.** The user's favourite is often an earlier round.
- **Deciding taste for the user.** Fix what's objectively wrong; offer options for everything else.
- **Tuning a wrong direction.** Endless small tweaks are a sign to go back to `research`.
- **Not recording.** A pick that isn't filed is lost; the next session repeats the same rounds.
- **Recording sessions instead of preferences.** Session notes don't recall well; one page per preference does.
