# Plan: an agent-maintained mind map that grows with the user

Status: PROPOSAL — no code changes yet.

Two inputs: Karpathy's *LLM Wiki* (a persistent, compounding markdown artifact the
LLM maintains, with ingest / query / lint operations) and Nous's *Hermes Agent*
(a background self-improvement review after each turn that quietly files facts and
distills skills, behind a consent gate, with a bounded snapshot injected once per
session).

## What the mind map is *for* here

Okonomi is not a research assistant, so a general-purpose wiki would be a solved
problem badly re-solved. The thing worth compounding in a creative tool is the
**taste graph**: what the user keeps reaching for, what they reject, and the craft
knowledge that reliably hits it. Everything below is shaped by that.

The distinguishing property vs. a plain wiki: **every node is grounded in artifacts
the user actually saw and judged.** A page about an aesthetic carries the
`artifact_id`s that exemplify it, the hearted ones marked. Claims are pinned to
`(session_id, event_id)`. Taste is evidenced, not asserted.

## What already exists (and must not be rebuilt)

| Layer | Status |
|---|---|
| Raw sources — transcripts (`events` + FTS5), artifacts w/ prompts, favourites, `research/*.md` | **done** |
| Retrieval over raw — `recall()` (discover / scroll / read / browse), BM25 + bookends | **done** |
| Procedural memory — filesystem skills, agent-authored (`create_skill`, `patch_skill_file`) | **done** |
| Out-of-band LLM call on turn commit — `_maybe_update_session_title_async` | **done** (the curator is this pattern, bigger) |
| Branch/abandon summarization — `BRANCH_SUMMARY_PREFIX`, `/tree` navigator | **done** (reusable as a negative-signal source and as the graph browser) |
| **Distilled, interlinked knowledge layer** | **missing — this plan** |
| **Automatic write-back without being asked** | **missing — this plan** |
| **Health/lint pass** | **missing — this plan** |

So okonomi has Karpathy's *raw* layer and Hermes's *skills*, and nothing in
between. Today the only durable distillation happens when the user explicitly
asks for a skill or a research report.

## Decisions

1. **Markdown files on disk, not SQLite rows.** The wiki layer is `~/.okonomi/mind/`
   — git-able, Obsidian-viewable (graph view is the mind map UI, for free), editable
   with the existing `read_file` / `write_file` / `edit_file` tools. **No new tool
   surface for the write path.** SQLite stays the raw layer.
2. **`index.md` is generated, not written.** It is rebuilt deterministically by
   walking node frontmatter. An LLM-maintained index rots; a derived one cannot.
3. **The index is injected once per session, bounded (~3 KB).** Hermes's constraint,
   for the same reason: mid-session injection breaks the prefix cache. Deeper pages
   are pulled on demand with `read_file`.
4. **The write path is a single structured call + deterministic apply**, not an agent
   loop. One model call returns a list of typed deltas; plain Python applies them.
   The *lint* path is the only place a full agent loop is warranted.
5. **Facts → mind map, procedures → skills.** No overlap. If the curator finds a
   repeatable technique it calls the existing skill machinery and the mind node just
   links to the skill by name.
6. **Corroboration threshold for writes.** A node is only created on an explicit user
   signal (a heart, a correction, "remember this") or a *second independent*
   observation. This is the single most important knob — an agent that files
   everything produces slop faster than lint can remove it.
7. **Contradictions append, never overwrite.** Taste drifts; drift is the interesting
   signal. Superseded claims stay with a `supersedes:` pointer.

## Layout

```
~/.okonomi/mind/
  SCHEMA.md          # conventions + curator instructions; co-evolved with the user
  index.md           # GENERATED catalog, bounded, injected at session start
  log.md             # append-only: ## [2026-08-06] curate | <session> | <what>
  nodes/
    motif/<slug>.md        # recurring subjects the user returns to
    aesthetic/<slug>.md    # look clusters: palette, light, grain, lens
    recipe/<slug>.md       # model + params + prompt scaffold; links to a skill
    reference/<slug>.md    # external anchors: artist, film, article
    preference/<slug>.md   # durable taste facts, incl. rejections
    project/<slug>.md      # an ongoing body of work
  pending/           # staged deltas awaiting /mind approve (when gated)
```

Six node types, closed set. An open taxonomy degenerates into one node per session.

### Node format

```markdown
---
type: aesthetic
name: sodium-vapour-night
created: 2026-07-31
updated: 2026-08-06
confidence: corroborated        # observed | corroborated | user-stated
links: ["[[grain]]", "[[theyyam]]", "recipe/[[flux-night-push]]"]
evidence:
  - {session: 2026-07-31-a4f2, event: 118}
  - {artifact: 412}
  - {artifact: 455, hearted: true}
supersedes: []
---

Prose the curator maintains. Why this look, what it is made of, where it
came from, what breaks it.

## Open questions
- Does this survive at 4:5, or is it a widescreen-only effect?
```

`links:` is the graph. `evidence:` is what keeps it honest — a claim with no pin is
written as `confidence: observed` and is a lint target.

## Operations

**Ingest** (Karpathy) — three doors, one apply path:
- explicit: `/mind ingest <path|url>` (also: research reports, dropped articles)
- automatic: the curator on turn commit (below)
- signal-driven: hearting an artifact, abandoning a `/tree` branch

**Query** — no new tool. The index is in-prompt; nodes are `read_file`. Answers
worth keeping are filed back as nodes (Karpathy's point that explorations should
compound the same way sources do). Add mind nodes to `recall()` only if the corpus
outgrows the injected index (phase 5).

**Lint** — a real agent loop, `/mind lint`, via the existing `launch_agent`:
contradictions, claims superseded by newer artifacts, orphan nodes, hub concepts
mentioned everywhere with no page, thin nodes with one piece of evidence,
`confidence: observed` nodes that never got corroborated. Output is a proposal
list, not silent edits.

## The growth engine (Hermes)

`_commit_turn` already fires `_maybe_update_session_title_async` in a daemon thread.
The curator is the same shape: fire-and-forget, silent on failure, warm cache.

**Gate** — do not run every turn. Run when the turn (a) registered an artifact,
(b) produced a heart, (c) contains a user correction of a prior output, or
(d) N turns have passed since the last curate. (c) needs the model, so fold it in:
the gate is cheap-checked on (a)/(b)/(d) and the call itself is allowed to return
zero deltas.

**Call** — one structured response over the turn's messages:

```python
class Delta(BaseModel):
    op: Literal["create", "append", "revise", "link"]
    node: str                    # "aesthetic/sodium-vapour-night"
    body: str = ""
    links: list[str] = []
    evidence: list[dict] = []
    confidence: Literal["observed", "corroborated", "user-stated"]
    rationale: str               # why this crosses the write threshold
```

**Apply** — plain Python: write/patch the file, append to `log.md`, regenerate
`index.md`. No tool loop, no chance of the curator wandering off into the session
directory.

**Consent** — config `mind.write_approval`. Default `false`: writes land, a
`SystemNote` says what changed. `true`: deltas go to `pending/` and surface in
`/mind pending` for approve/reject. Hermes's tension, same resolution: the agent
learns from corrections, the user keeps the veto.

## Surfaces

- `/mind` — browse the graph. **Reuse the `/tree` navigator widget** (it already
  renders a DAG with summaries); nodes instead of session branches.
- `/mind lint`, `/mind pending`, `/mind ingest <src>`.
- Obsidian pointed at `~/.okonomi/mind/` for graph view — the actual mind-map
  visual, at zero implementation cost.
- `git init` in `mind/` — version history of the user's taste, free.

## Rollout

1. **Layout + injection.** `okonomi/mind.py`: node read/write, frontmatter parse,
   `build_index()`, bounded `index_snapshot()`. `build_system_prompt(*, skills,
   mind_index)`. Seeding is manual/`/mind ingest` only — no curator yet.
   *Verify:* nodes round-trip through frontmatter; index regenerates identically
   from a fixed node set; injected snapshot respects the char cap; system prompt is
   byte-identical across turns within a session (prefix cache intact).
2. **Curator.** Gate + structured call + deterministic apply + `log.md`. Config for
   model, threshold, approval.
   *Verify:* a session that generates and hearts an image produces exactly one
   `preference`/`aesthetic` node with a correct `artifact_id` pin; a session of
   pure chit-chat produces zero deltas; a curator exception leaves the turn and the
   mind dir untouched.
3. **Signal triggers.** Heart → taste node; abandoned `/tree` branch → rejection
   evidence on the relevant node (reuses the existing branch summary).
   *Verify:* unfavouriting does not retro-edit history, it appends a supersede.
4. **Lint.** `/mind lint` as a background agent producing a proposal list;
   `/mind pending` review UI shared with gated writes.
   *Verify:* a hand-planted contradiction and a hand-planted orphan are both found.
5. **Search, only if needed.** Mirror node text into a `mind_fts` table and add a
   `mind` source to `recall()`. Trigger for doing this: the bounded index can no
   longer represent the corpus (~150+ nodes).

## Explicitly out of scope

- Embeddings / semantic search over nodes. BM25 and the index first, as with
  `search-retrieval-plan.md`.
- Auto-ingesting the open web. Sourcing stays the user's job (Karpathy's division
  of labour: the human curates and asks, the LLM does the bookkeeping).
- GEPA-style prompt optimization (Hermes has it). Prerequisite is a scored
  eval set of prompt → outcome; `favorites` is the seed of one, but not yet enough.
  Deferred until the mind map has produced `recipe` nodes worth optimizing.
- Multi-user / shared graphs.

## Risks, honestly

- **Slop.** The corroboration threshold (decision 6) is the only real defence, and
  it is a prompt-level heuristic. Expect to tune it. If phase 2 produces more than
  ~1 node per 10 turns, the threshold is too loose.
- **Cost.** An extra call on gated turns. Mitigation is the gate, plus Hermes's
  option of a cheap model over a digest instead of the main model over the transcript.
- **Confabulated links.** The curator will want to connect everything. Requiring an
  evidence pin per claim, and marking unpinned ones `observed`, keeps lint able to
  find them later.
- **The index cap is a real ceiling.** At ~150 nodes the snapshot stops being a
  faithful map and becomes a lossy sample. That is what phase 5 exists for; do not
  start it early.
