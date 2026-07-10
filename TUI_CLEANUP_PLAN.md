# Okonomi TUI — cleanup & consolidation plan

## Where things stand today

The TUI has migrated twice: raw `print()` CLI → Textual app → prompt_toolkit
"streaming-scrollback REPL". The last commit (`c8bf14d`) deleted the Textual
front-end (`tui/app.py`, `tui/widgets.py`, ~1,700 lines), but the replacement
was never committed. The result is split across two worktrees:

| Location | Branch | State |
|---|---|---|
| `~/.herdr/worktrees/okonomi/tui-update` (this worktree) | `tui-update` | Clean at `c8bf14d`. **Interactive mode is broken**: `tui/session.py` imports `okonomi.tui.repl.app`, and `tui/repl/` exists in no commit on any branch. |
| `~/Documents/Github/okonomi` | `tui-changes` | Same commit + large uncommitted WIP: the untracked `tui/repl/` package (~670 lines), `tui/rich_helpers.py`, a rewritten `events.py` (typed events, `TurnStart`/`TurnEnd`, `TraceWriter`, `StdoutPrinter` deleted), `display.py` slimmed 205 → 25 lines, and `agent.py` split into an `okonomi/agent/` package. |

The WIP is the real TUI and already does much of the consolidation this plan
would otherwise propose — but it is **also currently broken**: `okonomi/agent/`
has no `__init__.py`, and `agent/core.py` still uses single-dot relative
imports (`from .events import …`, `from .context import …`, `from .session
import …`) that resolved when it was `okonomi/agent.py` but now point at
nonexistent `okonomi.agent.*` modules. `import okonomi` fails outright in that
worktree.

## Phase 0 — one source of truth (blocker, do first)

Nothing else is safe to do while the only working TUI code is untracked in a
different worktree.

1. In `tui-changes`: fix the agent-package imports so the tree runs
   (`okonomi/agent/__init__.py` re-exporting `OkonomiAgent`; `.events`/
   `.context`/`.session` → `..events`/`..context`/`..session` in the new
   `agent/` modules).
   → verify: `python -c "import okonomi.cli, okonomi.tui.session"` and a
   one-shot `okonomi --task "say hi"`.
2. Commit the WIP there (repl package, events rewrite, display slimming,
   agent split) — one or a few logical commits, so the front-end finally
   exists in history.
3. Bring `tui-update` up to that state (merge or rebase `tui-changes` into
   this branch) and do the remaining phases here.

**Decision needed:** confirm `tui-changes`'s WIP is the intended direction
(vs. abandoning it) and which branch the cleanup should land on. The rest of
this plan assumes the WIP is kept and describes deltas *on top of it*.

## Phase 1 — delete what's already dead

- **`display.py`**: after the WIP slimming, only `cli.py` uses it (6 ANSI
  constants + `get_task_input`, which is just `(args.task or "").strip()`).
  Inline both into `cli.py` and delete the module. The boxed-output helpers
  (`print_ansi_box`, `print_code_box`, `highlight_code`), the sakura-wave art
  (`wave_formatted_text`, `print_loading_art`, `_render_ft_ansi`), and the
  pygments dependency in this module all go with it.
- **`ToolCall.expanded`** (`events.py`): a Textual leftover — set by
  `agent/core.py` (`expanded=needs_approval`) but read by no renderer; the
  REPL uses the global `/verbose` toggle instead. Remove the field and its
  one write site.
- **Stale-docstring sweep**: `events.py` still says events are "materialized
  as widgets" and subscribers forward "into Textual's message queue";
  `tui/session.py` mentions "Textual hang"; `rich_helpers.py` says "shared by
  both front-ends". Update to describe the single REPL front-end.
  → verify after phase: `grep -rn "Textual\|StdoutPrinter\|print_ansi_box" okonomi/` is empty; imports still pass.

## Phase 2 — flatten the package layout

`tui/repl/` existed to distinguish the new front-end from the Textual
`tui/app.py`. That app is gone; the extra nesting and the `repl` name now
just add a level of indirection (`okonomi/tui/repl/app.py`,
triple-dot imports `from ...events import …`).

Target layout — everything directly under `okonomi/tui/`:

```
okonomi/tui/
  __init__.py
  session.py      # TuiSession controller (worker thread, queue, bg tasks)
  app.py          # ReplApp — prompt_toolkit front-end (from repl/app.py)
  transcript.py   # ReplTranscript + replay_history + link/banner helpers
  pickers.py      # session/asset modals (from repl/pickers.py)
  completion.py   # slash completer (from repl/completion.py)
  style.py        # single palette + prompt_toolkit Style
  commands.py     # slash-command dispatch only
  assets.py       # asset actions + heart/loved logic (merged, see phase 3)
```

- Move `replay_history` out of `events.py` into `tui/transcript.py`: it's
  pure rendering (walks pydantic-ai messages and re-emits chat events) and
  it's only called from `tui/session.py`. `events.py` then carries no
  pydantic-ai coupling and is purely the event/trace layer.
- Fold `rich_helpers.py` (`text_with_path_links`, `render_welcome_banner`)
  into `transcript.py` — its only consumer.
  → verify: import check, one-shot run, and interactive smoke (below).

## Phase 3 — consolidate duplicated logic

Ordered by value; each is independent.

1. **One palette.** The pink/amber/muted colors are currently declared four
   times with "keep in sync" comments: `repl/style.py` (hex), `transcript.py`
   (rich constants `_ACCENT`/`_WARN`/`_ERROR`/`_RULE`), `rich_helpers.py`
   (banner hexes), `display.py`/`cli.py` (raw ANSI). Declare the hex palette
   once in `tui/style.py`; transcript and banner import it. (`cli.py`'s ANSI
   startup constants can stay local — they print before any TUI exists.)

2. **One "heart" implementation.** Hearting exists twice with *different
   behavior*: `commands.py`'s `/heart` favorites in the DB **and** symlinks
   into `~/.okonomi/loved/`; the asset-picker's heart action
   (`asset_actions._heart_paths`) only favorites — no symlink, and it looks
   artifacts up differently. Merge both into `tui/assets.py` with a single
   `heart(...)` used by the command and the picker, and move the ~180 lines
   of heart/loved helpers (`_guess_artifact_type`, `_symlink_into_loved`,
   `_heart_*`, `_list_loved`, filter tables) out of `commands.py` into it.

3. **Typed command results instead of string sentinels.** `commands.handle`
   returns magic strings (`"__exit__"`, `"__verbose__"`,
   `"__session_picker__"`, `"__asset_picker__"`, `"__bg__:<task>"`,
   `"__btw__:<q>"`, or arbitrary text meaning "forward to agent") that
   `TuiSession._on_submit` re-parses with `startswith`. Replace with a small
   result type, e.g.:

   ```python
   @dataclass
   class CommandResult:
       kind: Literal["exit", "verbose", "session_picker", "asset_picker",
                     "bg", "btw", "forward"]
       payload: str = ""
   # handle() -> CommandResult | None
   ```

   `_on_submit` becomes a dict/match dispatch instead of an if-ladder of
   prefix checks.

4. **Registry-based command dispatch.** `commands.handle` is a ~200-line
   `if low == …` chain, and `SLASH_COMMANDS` (names + descriptions for
   completion/help) is maintained separately from the handlers. Consolidate
   into one table: `(name, description, handler)`; `/help` and
   `SlashCompleter` derive from it, and `handle()` shrinks to lookup +
   arg-splitting. New commands then get added in exactly one place.

5. **One elapsed-time formatter.** The `divmod(secs, 60)` → `"1m 23s"`
   snippet appears in `TuiSession._spinner_label` and
   `ReplTranscript.active_tool_elapsed`. One `format_elapsed(seconds)` in
   `utils.py`.

6. **Drop the `app.session` back-ref.** `TuiSession` sets
   `self._app.session = self  # type: ignore[attr-defined]` solely so
   Ctrl+C can check `_agent_running` and call `cancel_current_turn`. Pass an
   `on_interrupt: Callable[[], bool]` callback into `ReplApp` alongside
   `on_submit` — kills the type-ignore and the private-attr reach-through.
   Also tidy: `run_async(**_ignored)` signature, and the duplicated
   `# --- Entry points ---` section marker in `session.py`.

Explicitly **not** doing (judged not worth the churn right now):
- Factoring the shared `Application` scaffolding out of the two pickers —
  they're ~100 lines each and diverge in focus/keybinding behavior.
- Touching the live-vs-replay dual render path for user turns
  (`echo_user_input` vs `UserMessage` events) — subtle, working, and
  documented in-code.
- Any `agent/core.py` restructuring beyond the Phase 0 import fixes — the
  agent split is its own effort.

## Verification (each phase, and final)

1. `uv run python -c "import okonomi.cli, okonomi.tui.session, okonomi.tui.app"`.
2. One-shot: `okonomi --task "reply with the word ok"` renders stream + tool
   lines via `ReplTranscript`.
3. Interactive smoke: launch `okonomi`; check banner, submit a turn, live
   tool status + elapsed, Ctrl+C cancel, `/help`, `/verbose` toggle,
   `/session` picker, `/assets` picker (open/copy/heart), `/heart` +
   `/loved` (symlink appears in `~/.okonomi/loved/`), approval prompt
   (approve / feedback / skip), `/bg` task completion note, Ctrl+D exit,
   session saved.
4. `grep` sweeps: no `__exit__`-style sentinels, no `StdoutPrinter`,
   no `okonomi.display`, no `tui.repl` imports left.

## Expected impact

- `display.py` (−205 lines committed / −25 post-WIP), `rich_helpers.py`
  merged, `asset_actions.py` merged, `repl/` flattened: package goes from
  10 TUI-related modules across 3 levels to 9 flat modules.
- `commands.py` shrinks by roughly half (heart/loved logic moves out,
  dispatch becomes a table).
- One palette, one heart path, one renderer (`ReplTranscript`) for both
  interactive and one-shot modes — behavior differences between the command
  and picker heart paths eliminated.
