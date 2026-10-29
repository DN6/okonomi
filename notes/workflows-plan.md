# Plan: workflows — scripted sub-agent orchestration

Status: PROPOSAL — no code changes yet.

Input: Claude Code's *dynamic workflows* (https://code.claude.com/docs/en/workflows):
a script the model writes, a runtime executes in the background, and the user can
save and rerun as a slash command. `agent()` / `parallel()` / `pipeline()` /
`phase()`, structured output via `schema`, replay-based resume, opt-in only.

## What a workflow is *for* here

Okonomi already fans work out. The `research` skill launches one `launch_agent`
per strand, polls `list_agents()`, then reads the strand files back. Every step of
that orchestration runs through the parent model's context, turn by turn, and the
plan evaporates when the turn ends. A workflow moves the plan into a script:

| | `launch_agent` today | Workflow |
|---|---|---|
| Who decides what runs next | The parent model, per turn | The script |
| Sub-agent result | Not returned (artifact count only) | Returned to the script; structured when asked |
| Intermediate results | Parent context / ad-hoc files | Script variables + a ledger on disk |
| Repeatable | The prompt text, by hand | The script, as `/<name>` or `python3 <name>.py` |
| Interruption | Lost | Resumable from the ledger |

The creative-tool angle: the fan-outs that matter here produce **media**, not text.
A variation sweep, a judged shortlist, a "refine until it passes" loop, a
multi-strand research brief. The runtime must make "generate N things, look at them,
pick" a three-line script, with the pixels reaching the judge.

## What already exists (and must not be rebuilt)

| Layer | Status |
|---|---|
| Sub-agent primitive — a fresh `OkonomiAgent` on the parent's `session_id`, own event loop in a thread, artifacts merged back (`AgentTools.launch`, `TuiSession._spawn_bg_task`) | **done** — the runtime's `agent()` is this, plus a return value |
| Shared workspace — `$OKONOMI_SESSION_DIR`, files auto-tracked as assets by the post-`bash` sweep | **done** — sub-agents in the child write here, the parent sweeps |
| Background execution — `bash(background=True)` → `term-N`, seatbelt-wrapped, registered in `JobRegistry` with a live stdout reader, `JobFinished` with output on exit | **done** — *this is the workflow runtime's host* |
| Completion → synthetic turn — `JobFinished` → `_on_job_finished` | **done** — needs one branch for a text result |
| Skills as slash commands, bundled + user dirs, progressive disclosure | **done** — the same discovery shape for workflows |
| Multimodal auto-attach of image paths in a prompt | **done** — how a judge agent sees candidates |
| **A sub-agent that returns its answer** | **missing** |
| **Structured (JSON) sub-agent output** | **missing** |
| **A script API, a ledger, resume** | **missing — this plan** |
| **The authoring skill** | **missing — this plan** |

## Decisions

1. **A workflow is a Python script run through `bash`.** The model writes it with
   `write_file` into `$OKONOMI_HOME/workflows/<name>.py` and runs
   `bash(command="python3 $OKONOMI_HOME/workflows/<name>.py", background=True)`.
   Writing there makes it a `/<name>` command immediately — no save step. Name a
   script only if it's worth rerunning; overwrite throwaways. No new tool, no new
   runtime host, no RPC. The script gets the seatbelt (writes scoped to
   `$OKONOMI_HOME`, temp, cwd), a `term-N` job id, `/jobs`, `/logs term-N` for
   live progress, and a `JobFinished` with its stdout when it exits. The same file
   runs from a terminal with `python3`.
2. **Any library, any control flow.** A script is an ordinary module:
   `from okonomi.workflows import agent, parallel, pipeline, phase, log`, then
   whatever Python it needs — PIL to tile candidates into one contact sheet, ffmpeg
   via `subprocess`, a CLIP scorer, a `while` loop that calls `agent()` until a
   check passes. Claude Code forbids this because its script runtime *is* the
   isolation boundary; here the boundary is the sandbox the script already runs
   in. Deterministic glue between agent calls is the point — an LLM agent should
   not be spent on `Image.resize`.
3. **`agent()` returns the sub-agent's final reply.** That is the single missing
   primitive. `okonomi.workflows.agent` builds an `OkonomiAgent` in the child
   process on the session derived from `$OKONOMI_SESSION_DIR` (the id is the
   directory name), runs it to completion, returns `run_async`'s output. With
   `schema=`, the reply is JSON validated against it (shallow: type + `required`
   keys in v1; retries up to 3 times with the validation error fed back). Failed
   or stopped agents resolve to `None`, never raise — `pipeline()` keeps the
   `None` so the script sees which item died. No `$OKONOMI_SESSION_DIR` → a fresh
   session, so standalone runs still work.
4. **Ledger-keyed replay for resume, all in the child.** Every `agent()` call gets a
   sequence number assigned at *submission* on the script's main thread
   (deterministic even under `parallel`, because submission is serial). The ledger
   is `$OKONOMI_SESSION_DIR/workflows/<name>.ledger.jsonl` — per session, not
   beside the script, so two sessions running the same workflow never share or
   clobber run state. One line per call: `{seq, label, prompt_hash,
   schema_hash, status, result}`. `python3 <name>.py --resume` re-executes the
   script; call *i* returns the ledger result when its hashes match and status is
   `ok`, otherwise runs, and every later call runs. Same semantics as Claude Code.
   Nothing stops `time`/`random` in a prompt; the skill says not to, and a
   nondeterministic prompt simply reruns from that call.
5. **Opt-in only.** Nothing in the system prompt tells the model to reach for a
   workflow. Entry points are the user typing `/workflow <task>`, `/<saved-name>`,
   or asking for one in words. Media fan-outs cost real money; the user pulls the
   trigger.
6. **Progress is stdout.** `phase()` and `log()` print; `agent()` prints one line
   at start and one at finish (`seq`, label, status, elapsed). `/workflows` is the
   structured view of a run (see Surfaces); `/logs term-N` still shows the raw
   stdout. For `/workflows` to show in-flight agents, the ledger gets a `running`
   line at submission, superseded by the final line for the same `seq`.
7. **The result is the last line of stdout.** The script sets a module-level
   `result`; `okonomi.workflows` prints it as one JSON line at exit (an `atexit`
   hook registered on import). `JobFinished.output` carries it to
   `_on_job_finished`, which needs one branch: when the job's output ends in a
   `result` line, the synthetic turn presents it instead of only the asset block.
8. **Workflows live in one place: `$OKONOMI_HOME/workflows/<name>.py`.** No bundled
   workflows — every workflow is one the model or user wrote. Discovered the way
   skills are, exposed as `/<name>`. `meta` is a literal dict parsed with
   `ast.literal_eval` — never executed at discovery time.

## Script API (the skill's contract)

```python
from okonomi.workflows import agent, args, log, phase, pipeline

meta = {
    "name": "explore",
    "description": "Fan out concept directions, generate a candidate each, judge, return top picks",
    "phases": ["Directions", "Generate", "Judge"],
}

phase("Directions")
brief = agent(
    f"Propose 4 distinct visual directions for: {args['text']}. "
    "Each: a name, a two-sentence rationale, and one layered image prompt.",
    schema={"type": "object", "required": ["directions"],
            "properties": {"directions": {"type": "array", "items": {
                "type": "object", "required": ["name", "prompt"]}}}},
)

phase("Generate")
def generate(d):
    return agent(
        f"Generate one image for the direction '{d['name']}' with this prompt: {d['prompt']}. "
        "Save it into $OKONOMI_SESSION_DIR and reply with only the absolute path.",
        label=d["name"],
    )
paths = pipeline(brief["directions"], generate)
paths = [p for p in paths if p]

phase("Judge")
verdict = agent(
    "Rank these candidates against the brief and explain the top two.\n"
    + "\n".join(paths),
    schema={"type": "object", "required": ["ranking"]},
)
result = {"ranking": verdict["ranking"], "paths": paths}
```

A ralph loop is the same API plus a `for`:

```python
from okonomi.workflows import agent, args, log, phase

meta = {"name": "ralph", "description": "Work a task in rounds until a checker agent passes"}

MAX_ROUNDS = 5
verdict = {"passed": False, "notes": ""}
for round_no in range(1, MAX_ROUNDS + 1):
    phase(f"Round {round_no}")
    agent(
        f"Task: {args['text']}\n\nContinue from whatever is in $OKONOMI_SESSION_DIR. "
        f"Previous reviewer notes: {verdict['notes'] or 'none'}",
        label=f"work-{round_no}",
    )
    verdict = agent(
        f"Review the outputs in $OKONOMI_SESSION_DIR against this task: {args['text']}. "
        "Reply with JSON: passed (bool) and notes (what still fails).",
        schema={"type": "object", "required": ["passed", "notes"]},
        label=f"check-{round_no}",
    ) or verdict
    if verdict["passed"]:
        break
    log(f"round {round_no}: {verdict['notes'][:120]}")
result = {"rounds": round_no, **verdict}
```

Both run unchanged as `python3 ralph.py "<task>"` from a terminal, or
`python3 ralph.py --resume "<task>"` to replay the ledger.

| Name | Signature | Behaviour |
|---|---|---|
| `agent` | `agent(prompt, *, label=None, schema=None, model=None) -> str \| dict \| None` | Runs one sub-agent to completion and returns its reply. `schema` → parsed JSON. `None` on failure/stop. |
| `parallel` | `parallel(thunks) -> list` | Runs zero-arg callables concurrently (bounded pool), returns results in input order. |
| `pipeline` | `pipeline(items, *stages) -> list` | For each item, `stages[0](item)` then `stages[1](prev)`…; items run concurrently, stages per item sequentially. `None` short-circuits an item. |
| `phase` | `phase(title) -> None` | Prints a phase header; groups the agent lines that follow in `/logs`. |
| `log` | `log(text) -> None` | Prints one progress line. |
| `args` | module global | `{"text": <argv joined>}` plus `--key value` pairs parsed from argv; `/name <text>` passes `text`. |
| `result` | module global | The run's result, JSON-serializable. Printed as the last stdout line at exit; fed back to the session as the completion turn. |
| anything else | | Full Python: `import`, files, `subprocess`, loops. Writes are sandbox-scoped like every `bash` call. |

Limits, env-overridable: 4 concurrent agents (`OKONOMI_WORKFLOW_MAX_CONCURRENT`),
50 agents per run (`OKONOMI_WORKFLOW_MAX_AGENTS`; the 51st `agent()` raises).
Media generation is the expensive tool here, so the defaults are a quarter of
Claude Code's.

## Layout

```
okonomi/
  workflows/
    __init__.py                # the whole runtime: agent/parallel/pipeline/phase/log, args parsing,
                               # ledger + --resume, result printing, caps. Imports the agent lazily
  skills/workflows/SKILL.md    # the authoring skill (analog of /workflow-authoring)
$OKONOMI_HOME/workflows/<name>.py                 # every workflow script
<session_dir>/workflows/<name>.ledger.jsonl       # per-session run state for resume
```

The `explore` and `ralph` scripts above are examples for the skill, not shipped files.

Discovery of `$OKONOMI_HOME/workflows/` (`meta` via `ast.literal_eval`) is a helper in
`okonomi/workflows/__init__.py`; `tui/commands.py` imports it the way it imports
`_discover_skills` from `tools.skills`. Nothing under `tools/` — there is no tool.

### Surfaces

- `/workflow <task>` — loads the `workflows` skill and forwards "write and run a
  workflow for: <task>" (same `Action.FORWARD` sugar `/skill` uses). The model
  then writes the file and runs it with `bash(background=True)` like any other
  long command.
- `/<name> [text]` — runs a workflow with no model turn:
  `agent.terminal.bash(command=f"python3 {path} {shlex.quote(text)}",
  background=True)`. Same path `/bg` takes, so `/jobs`, `/logs`, and the
  completion turn come for free.
- `/workflows` — lists this session's workflow runs, running first, then finished
  (the `term-N` jobs whose command is a `$OKONOMI_HOME/workflows/` script). Picking
  one opens its progress: phases, and per agent the label, status, and elapsed
  time, read from its ledger, with the result once it exits. Same picker shape as
  `pick_jobs` in `tui/pickers.py`. Saved scripts are found through `/<name>` slash
  completion, not here.
- `/stop <term-id>` — new, needed once a run can schedule 50 agents. Terminates the
  process group; a resume replays what the ledger recorded before the kill.
  `TerminalTools` already has the interrupt event, so this is a thin command.

### The skill

`okonomi/skills/workflows/SKILL.md`, H2 sections so `load_skill_section` can pull one:

- **When to use** — more than ~3 independent agents, or a judge/verify pass, or a
  loop until a check passes. Not for serial reasoning, not for one generation.
- **Script API** — the table above, verbatim, plus the write-then-`bash` recipe.
- **Patterns** — fan-out + judge (media), strand research (the current `research`
  fan-out, rewritten as a script), refine-until-pass (bounded round count),
  find-until-stable (stop when a round adds nothing).
- **Making the judge see** — put candidate paths in the judge's prompt; okonomi
  attaches the pixels. Ask generators to reply with only the path. Tile many
  candidates into one contact sheet with PIL first.
- **Pitfalls** — `None` results, no shared variables between agents (files in
  `$OKONOMI_SESSION_DIR` are the channel), keep `time`/`random` out of prompts or
  resume reruns from there, keep `meta` literal, bound every loop, size the first
  run small.

Nothing in the system prompt references the skill beyond its catalog line, so it
costs one line of prompt until `/workflow` or the user's words load it.

## Runtime notes

- **`agent()` internals**: the `_run` body of `AgentTools.launch`, factored into a
  function that returns `run_async`'s output. `launch_agent` calls the same function
  and discards the result; that is the one refactor of existing code.
- **Session identity in the child**: `session_id = Path(os.environ["OKONOMI_SESSION_DIR"]).name`.
  Sub-agents run on that id, so their `bash` calls export the same
  `$OKONOMI_SESSION_DIR` and their outputs land where the parent's sweep finds them.
- **Nested sandbox**: the script is seatbelt-wrapped, and each sub-agent's `bash`
  wraps again. `sandbox-exec` inside `sandbox-exec` is allowed when the inner
  profile is no looser; ours is identical. Verify on the first run; if it fails,
  the inner `wrap_command` degrades to unwrapped with its existing warning, which
  is acceptable because the outer profile still holds.
- **Structured output v1**: append a fixed suffix ("Reply with only a JSON object
  matching this schema: …"), extract the last `{…}` block, `json.loads`, check
  `type` and `required`. TODO(dn6): switch to pydantic-ai `output_type` with a model
  built from the schema once the prompted version proves too flaky — remove the
  regex extractor then.
- **Ordering under `parallel`**: sequence numbers are taken when the main thread
  *submits* the thunk, before any worker runs. The ledger is therefore stable across
  replays even though completion order is not.
- **Completion turn**: `_on_job_finished` currently assumes the job produced media.
  Add: when the job's output ends with a `result` JSON line, the synthetic message
  carries it and asks the agent to present it; the asset block stays as-is.
- **Usage and transcripts**: sub-agent tokens and messages are not recorded in the
  parent session — parity with `launch_agent` today. The ledger holds prompt +
  result per agent, which is what `/logs` and resume need. Full transcript
  persistence is a separate question.
- **Cache**: every sub-agent shares the parent's system prompt, so the prefix is
  cached after the first agent on Anthropic/OpenRouter without a stagger. Not
  doing Claude Code's 5 s hold in v1.

## Rollout

1. **Runtime.** `okonomi/workflows/__init__.py`, the `launch_agent` factoring, the
   `_on_job_finished` branch.
   *Verify:* a script with two `parallel` agents and a `schema` agent prints a
   `result` line and the session's completion turn quotes it; a failing agent
   yields `None` and the run still completes; the ledger has one line per agent
   with stable `seq`; a script that writes outside `$OKONOMI_HOME` fails on the
   write, not on launch; the same script runs standalone with `python3` and no
   `OKONOMI_SESSION_DIR`; media a sub-agent wrote appears in `/assets` after the
   job finishes; nested `sandbox-exec` works or degrades with the warning.
2. **Skill + `/workflow`.** `skills/workflows/SKILL.md`, the slash sugar.
   *Verify:* `/workflow <task>` produces a script that runs to a `result` line;
   `load_skill_section("workflows", "script-api")` returns the table.
3. **Discovery + `/workflows`.** `/<name>`, and the runs picker.
   *Verify:* a `meta` containing a function call is dropped from completion with a
   note, not executed; built-in commands beat a workflow on a name clash; a script
   the model wrote in one session runs as `/<name>` in the next; two sessions
   running the same workflow write separate ledgers; `/workflows` during a 4-agent
   run shows the running agents under their phase, and after exit shows the result.
4. **Resume + stop.** `--resume`, `/stop`.
   *Verify:* stop a 6-agent run after 3 complete, resume, exactly 3 agents run;
   edit the prompt of agent 2, resume, agents 2–6 run.

## Explicitly out of scope

- A parent-side servicer or `run_workflow` tool. Considered and dropped: `bash`
  already hosts background jobs, and the only things a servicer added were
  per-agent rows in `/jobs` and parent-enforced caps, both of which the child
  can provide (stdout lines, an in-process cap).
- Per-workflow permission rules / "don't ask again" consent. Okonomi has one user
  and one guard (`check_command`); reuse it.
- Cross-session resume. The ledger lives in the session dir, so `--resume` after
  `okonomi --resume` would work, but nothing in v1 promises it.
- Model-per-agent selection beyond the `model=` parameter passing through to
  `OkonomiAgent(model_id=…)`.
- Usage-limit pausing, size guidelines, the `ultracode` auto-planning mode.
