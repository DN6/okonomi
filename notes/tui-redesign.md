# okonomi TUI redesign — streaming scrollback + on-demand modals

Replace the Textual TUI with a **prompt_toolkit streaming-scrollback REPL**, keeping the
`EventEmitter` contract and `TuiSession` untouched. Verified against the installed
prompt_toolkit 3.0.51/3.0.52.

## Why migrate (the deciding constraints)

1. Textual is bloated / hard to customize (the original complaint).
2. Native terminal features must work: **copy/selection** and **click-to-open file/URL links**.
3. Go lower-level.

Whoever owns the screen and the mouse owns copy-and-click. Textual captures the mouse, so it
**takes both away** from the terminal — which is exactly why the current code had to hand-roll an
`on_click` link opener (`app.py:918`) and why native drag-select is broken today. A streaming
scrollback REPL gives both back **for free, with zero modifier keys and zero app code**.

> Honest caveat (two investigators raised it): if copy/links were *not* hard requirements, the
> cheaper move would be to stay in Textual and just swap the slash dropdown's `startswith` filter
> for the existing `fuzzy_match` (utils.py:148) — a ~2-line change. We are migrating because copy /
> links / lower-level are explicitly required, not merely to get fuzzy completion.

## Architecture

```
ONE process. ONE asyncio loop (TuiSession's agent thread stays as-is).

  main REPL  =  prompt_toolkit, full_screen=False, mouse_support=False
                running inside `with patch_stdout():`
                ─ agent transcript streams into REAL terminal scrollback
                  → native select-copy + native OSC 8 click work, no app code
                ─ bottom chrome only: status line, bg-tasks row, approval line,
                  multi-line input, fuzzy completion menu

  slash dropdown  =  FuzzyCompleter(WordCompleter(SLASH_COMMANDS, meta=desc))
                     complete_while_typing=True, CompleteStyle.COLUMN
                     → CompletionsMenu Float at the cursor, rendered in
                       reserve_space_for_menu rows. NO full-screen needed.

  pop-up pickers  =  transient Application(full_screen=True) per open
                     await app.run_async();  exit via app.exit(result=...)
                     → alt-screen: takes the terminal for a few seconds,
                       then restores REPL scrollback byte-for-byte.
```

The pickers use the **alternate screen** (`full_screen=True`) only while open. The REPL never
leaves the primary buffer, so copy/links survive the modal completely (verified: renderer enters
alt-screen at render, quits on exit; main buffer untouched).

## How the EventEmitter drives the screen

One `EventSink` subscribes exactly like today's `StdoutPrinter`:
`unsub = agent.emitter.subscribe(sink.handle)`, late backlog replayed via `emitter.history()`
(same contract as `OkonomiApp.on_mount`, app.py:686). `handle(ev)` reconciles **by `event.id`**
with per-id state (emit and update deliver the *same* mutated object — no created-vs-updated
signal). Marshaling replaces Textual's `call_from_thread`: capture the pt loop at startup; if on
the loop thread render via `run_in_terminal`, else `loop.call_soon_threadsafe(...)` — emits fan out
**synchronously on the emitting thread** (agent daemon thread / bg-exec workers / pt thread for
user-submit emits).

| Event | Source | Scrollback rendering |
|---|---|---|
| **Banner** (events.py:34) | session.py:415 | One-shot sakura banner → ANSI → `write_raw`. id-guarded so replay never double-prints. |
| **Divider** (events.py:39) | emitter.divider() | One `'─'*40` dim rule per id. |
| **UserMessage** (events.py:44) | emitter.user() | Pink `'● '` + text as Rich `Text` (no markup parse), link-wrapped. One-shot. |
| **AssistantStream** (events.py:56) | agent.py:1611-1616 | The **only** streaming type; `.text` is **cumulative**. Reuse StdoutPrinter's per-id seen-offset + line buffer (events.py:225-240): on each update emit only **complete** lines, link-wrap, `write_raw`; flush trailing partial on `done`. New id per model node ⇒ sequential blocks. 60Hz reveal machinery **deleted**. |
| **ToolCall** (events.py:61) | created agent.py:1505, finished events.py:176, bg-exec mutates in place agent.py:1073 | Created-then-mutated. First sight prints `▶ tool_name (args) [running]`. Full details (Rich `Syntax` for execute_code) print inline **only** when `ev.expanded` (approval-gated) or on **error**; `ok`-with-output prints a `→ <snippet>` line. Finish prints `✓/✗`. Keep id→ToolCall map for `/show n`. |
| **SystemNote** (events.py:81) | many | One line colored by level (dim/amber/red). The **approval prompt** note is special-cased: not printed to scrollback — it drives the bottom-chrome approval line and vanishes when answered, while staying in `emitter.history()` for other subscribers. |

Out-of-stream state (spinner / usage / bg-tasks) stays poll-driven via the existing
`spinner_label_fn`/`usage_fn`/`bg_tasks_fn` callables (session.py:364-398), rendered in bottom
chrome via `refresh_interval` — not events.

## Requirement 1 — open paths & URLs

Reuse `text_with_path_links()` (widgets.py:47) — it already classifies on-disk paths vs image URLs
and builds `file://` targets — but change the **sink**: let Rich emit **real OSC 8 bytes**
(`ESC]8;;file:///abs/path ST label ESC]8;;ST`) instead of stashing the target for the in-app
`on_click` hack. Keep the raw path/URL as the **visible label** so copy still yields it. Cmd/Ctrl+click
opens via the terminal's own routing.

**Load-bearing constraint:** OSC 8 bytes must reach the terminal **verbatim** via
`app.output.write_raw(text)` inside `run_in_terminal` — **never** through pt's `ANSI()` parser,
which strips `ESC` and leaks the payload as literal text. (hermes hit this and *gave up*, stripping
OSC 8 with a regex; we bypass the parser instead — this is the whole reason links work for us.)
`ANSI()` is used only for bottom-chrome `FormattedText`.

**Fallback:** feature-detect via `TERM_PROGRAM`/`WT_SESSION`/`VTE_VERSION`; when OSC 8 is
unsupported (Terminal.app pre-Sequoia, plain xterm) also emit a numbered ref `[1] /abs/path:42` and
an app-owned `/open n` that calls the existing `_open_paths()` (asset_actions.py:43, `open`/`xdg-open`).

## Requirement 2 — copy text/output

- **Native (primary):** transcript is real scrollback with no mouse capture → click-drag select +
  copy-on-select work with **zero modifier keys**, persist after exit, and can copy **arbitrary
  sub-spans** (half a code block, one path) — which is exactly what breaks under Textual today.
- **App-driven (whole block):** `/copy [n]` copies the last/Nth assistant message (the sink already
  accumulates full `AssistantStream.text` per id) via the **existing** `copy_to_clipboard()`
  (utils.py:115), which already layers pbcopy/xclip/xsel **then OSC 52** (SSH/tmux). Optional
  `c-y` = copy-last.

**Caveat:** OSC 52 has ~74KB caps and tmux needs `set-clipboard on`; but `copy_to_clipboard` prefers
local pbcopy/xclip first, so OSC 52 is only the remote fallback, and native selection works
regardless. Surface a one-line tmux/SSH hint.

## Requirement 3 — slash commands with fuzzy filtering + dropdown

Confirmed: the completion dropdown works in **non-fullscreen** mode. `PromptSession`/inline
`Application` builds a `FloatContainer` whose completion menu is a `Float` at the cursor, painted in
`reserve_space_for_menu` reserved rows — prior output scrolls up normally above the prompt.

- Wrap the source: `FuzzyCompleter(WordCompleter(SLASH_COMMANDS, meta_dict=descs))` →
  type `hrt` → `/heart`, with matched chars bolded. Use `NestedCompleter` later for arg grammar
  (`/skill <name>`, `/logs <id>`). `complete_while_typing=True`, `complete_style=COLUMN`.
- Reuse the **single source of truth**: `SLASH_COMMANDS` (commands.py:22) already feeds both `/help`
  and the dropdown — keep that. Optionally adopt hermes's `CommandDef`-list registry pattern so one
  dataclass entry defines name + aliases + description + arg hint.
- Gotcha: `complete_while_typing` and `enable_history_search` both bind up/down — keep history nav
  only when the menu is closed (replicates the current dropdown-or-history arrow handler).

## Requirement 4 — pop-up asset search & session picker

On-demand `Application(full_screen=True)` modal, awaited from the slash handler:

```python
result = await build_picker(items).run_async()   # blocks REPL coroutine
# control returns with whatever app.exit(result=...) passed; Esc/c-c → None
```

(From a keybinding instead of a slash handler, wrap in `run_in_terminal`.) Never use `app.run()` /
`asyncio.run()` — nested-loop error. The fzf-style picker is tiny: a search `Buffer`
(`on_text_changed` refilters via existing `fuzzy_match`), a `FormattedTextControl` list (re-evaluated
each render — no manual rebuild the Textual `OptionList` needs), up/down move the highlight while
focus **stays on the search box** (eliminates the focus-bounce dance at app.py:164-187/334-341).

Re-host existing data + actions unchanged:
- **Session picker** ← `SessionManager.list_sessions(limit=500)` (db.py:215); select → `agent.load_session` → `replay_history`.
- **Asset picker** ← `agent.state.artifacts` (multi-select; keep `selected: set`, `space` toggle,
  `a`/`x`, and `o`/`c`/`h` verbs); exit with `app.exit(result=("open", paths))` so the existing
  `handle_asset_action(action, paths)` contract (asset_actions.py) is untouched.

Decision: **alt-screen full_screen modal**, not a Float (nothing to float over a patch_stdout REPL)
and not fzf-by-default (line-oriented; fights rich objects + theme + multi-verb accept — keep it as
an optional `shutil.which("fzf")` enhancement only).

## What we keep / what regresses

**Keep (re-hosted, not rewritten):** EventEmitter decoupling; cross-thread marshaling;
**all of TuiSession** (agent worker loop, queue, approval Future + lock, bg-task spawning, slash
sentinel dispatch, save/atexit) — only the `OkonomiApp` collaborator is swapped; the slash sentinel
contract (`__exit__`/`__session_picker__`/`__asset_picker__`/`__bg__:`/`__btw__:`); inline approval
flow; multi-line input + at-boundary history + `~/.okonomi/input_history`; session resume replay;
asset open/copy/heart; status + bg-tasks rows; Ctrl+C three-stage / Ctrl+D/L/O/Esc; inline
(non-alt-screen) transparent feel — *strengthened* by patch_stdout.

**Regressions to decide on (see open questions):**
- **In-place collapsible tool calls** → print-on-error/approval, else `/show n`. Biggest casualty.
- **Per-character 60Hz reveal** → per-completed-line reveal (optional live partial tail in chrome).
- Tool `running→ok/error` shows as a new status line, not an in-place header mutation.

## Module layout

```
okonomi/tui/scrollback/
  app.py      ScrollbackApp  — Application(full_screen=False) + bottom-chrome HSplit + patch_stdout
  sink.py     EventSink      — emitter subscriber; per-id reconcile; loop marshaling
  render.py   event → ANSI (force_terminal truecolor Rich console); REAL OSC 8 via write_raw
  keys.py     KeyBindings    — enter state-machine, c-c/d/l/o, up/down dropdown-or-history
  pickers.py  session + asset full_screen modals (await run_async)
  commands.py reuse existing handle() + sentinel contract; add /copy /open /show
```

Behind an env flag at first; Textual `okonomi/tui/*` stays until scrollback is default. `events.py`,
`agent.py`, the EventEmitter contract, and `TuiSession`'s concurrency code stay **untouched**.

## Migration plan

1. Scaffold `okonomi/tui/scrollback/` behind a flag; both TUIs runnable.
2. `EventSink` with StdoutPrinter's per-id state (stream seen-offset + line buffer, tool status) +
   id→ToolCall map; marshal to pt loop; serialize transcript prints through one loop-bound queue so
   bursty bg-exec emits don't interleave with the prompt repaint.
3. `render.py`: per-event → ANSI; reuse `render_welcome_banner`, `text_with_path_links` (now **real
   OSC 8**), `ToolCallWidget._details_text` Syntax. **Single audited path:** all transcript bytes via
   `write_raw`, never `ANSI()`.
4. `ScrollbackApp`: `Application(full_screen=False, mouse_support=False, erase_when_done=True,
   refresh_interval=0.12)`; bottom HSplit = approval line / bg row / status / input rule / multiline
   TextArea / completions menu; run inside `with patch_stdout(): await app.run_async()`.
5. Keys: enter state-machine (normal → `session._on_submit`; approval → parsed yes/all/no/feedback),
   c-j newline, c-c three-stage, c-d quit, c-l clear (context/history intact), c-o expand-latest,
   up/down dropdown-or-history.
6. Slash: reuse `commands.handle()` + sentinels unchanged; add `/copy [n]`, `/open n`, `/show n`;
   register in `SLASH_COMMANDS`; wrap completer in `FuzzyCompleter`.
7. OSC 8 capability detection + numbered-ref `/open` fallback; always keep raw label visible.
8. Pickers (session/asset) as `full_screen=True` modals; preserve `show_session_picker`/
   `show_asset_picker` signatures + `handle_asset_action` so TuiSession is unchanged.
9. Swap collaborator: `self._app = ScrollbackApp(...)` with same ctor args + methods; provide a
   `call_from_thread` shim = `loop.call_soon_threadsafe` so `_approval_callback` keeps working.
10. Validate must-preserve end-to-end (per-line streaming, tool ok/error/approval-expanded, approval
    across fg+bg, multi-line input + history, resume replay, asset open/copy/heart, native
    select-copy, native + fallback link open, status/bg rows, Ctrl+C/D/L/O/Esc, shutdown/atexit);
    re-test resize (`erase_when_done` chrome leak) + tmux OSC 52/OSC 8 passthrough.
11. Delete Textual deps + dead widgets once scrollback is default.

## Open questions (genuine decisions)

1. **Tool-call collapsibility** — the real casualty, but there's a middle path (validated by how
   Claude Code's renderer works). Three options, pick one:
   - **A. Flat scrollback + `/show n`.** Active tool call prints a compact `▶` line; details print
     inline only on error/approval; `/show n` reprints full details on demand. Simplest; pure
     native scrollback.
   - **B. Live region for the active tool call (recommended).** Render the currently-running /
     most-recent tool call as an **expandable widget in the bottom chrome** (toggle `Ctrl+O`, as
     today). On completion / next turn, flush its chosen collapsed-or-expanded form to immutable
     scrollback via `write_raw`. This is prompt_toolkit's `patch_stdout` analogue of Ink's
     `<Static>` + mutable-live-region split — recovers glance-then-expand for what you're *watching*
     while keeping scrolled-away output natively copyable/clickable.
   - **C. App-owned render (rejected).** Claude Code's fullscreen mode: app owns every cell, expand
     works everywhere, but native scrollback is sacrificed (copy/search become an explicit
     dump-to-scrollback command). This is the Textual side of the trade-off we're leaving — only
     revisit if in-place expand of *arbitrary old* tool calls is non-negotiable.

   > Context: Claude Code faced this exact tension (Ink `<Static>` freezes committed output, which
   > fights re-expansion) and chose C (own the render, lose native scrollback, rebuild copy/search
   > as `Ctrl+O`→`[`). okonomi is choosing native-scrollback-first; option B is how we keep the
   > collapsibility that matters without giving that up.
2. **Streaming feel** — per-completed-line reveal acceptable, or ship the live partial-line tail in
   bottom chrome from day one?
3. **`/btw`** — keep input editable during a turn (current behavior, enables side-questions) rather
   than hermes-style read-only lock? (Assuming yes.)
