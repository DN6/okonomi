# okonomi Path A — Streaming-Scrollback REPL (implementation plan)

Replace the Textual `OkonomiApp` interactive transcript with a prompt_toolkit streaming REPL
that prints the `EventEmitter` stream into **native terminal scrollback**. Native scroll +
native drag-copy + real clickable OSC 8 links all work because the terminal — not an app —
owns the scrollback buffer and the mouse. Supersedes the migration sections of
`notes/tui-redesign.md`. `mouse=False` native-copy is already validated.

## Architecture

```
ONE process. TuiSession (agent thread + asyncio queue + approval Future) — UNCHANGED.

  transcript  =  ReplTranscript (EventSink) → rich Console(file=sys.stdout, force_terminal=True)
                 prints every event into NATIVE scrollback
                 → native wheel/keyboard scroll, native drag-copy, real OSC 8 links

  bottom chrome =  Application(full_screen=False, mouse_support=False, erase_when_done=True)
                   under `with patch_stdout(raw=True):`
                   → multi-line TextArea input, status line, bg-tasks line, fuzzy slash menu

  pickers     =  on-demand full_screen=True Applications (alt-screen), awaited then discarded
```

`patch_stdout(raw=True)` is the coordinator: streamed event-prints (from the agent daemon
thread) erase the live prompt, write above it, redraw — no manual marshaling for output.

## Reused as-is

- **`TuiSession` core (untouched):** worker loop + agent daemon thread, `_input_queue`/`_submit`,
  `cancel_current_turn`, approval Future + `_approval_ui_lock`, `_inject_side_question`,
  `_spawn_bg_task`/`_on_bg_exec_complete`, save/shutdown/atexit, `_resume_session`,
  the `_spinner_label`/`_usage_label`/`_bg_tasks_list` status sources. `_on_submit` unchanged
  except the three `self._app.*` branches keep the same call shape against `ReplApp`.
- **`StdoutPrinter`** (events.py:199) — kept verbatim for the one-shot CLI (cli.py:84). NOT the
  interactive sink.
- **`replay_history`** (events.py:261) + synthetic-prefix skip — unchanged.
- **`SLASH_COMMANDS` + `commands.handle()` + the sentinel protocol** (`__exit__`,
  `__session_picker__`, `__asset_picker__`, `__bg__:`, `__btw__:`) — verbatim.
- **Picker data sources:** `SessionManager().list_sessions(500)`, `agent.state.artifacts`,
  `handle_asset_action(action, paths, agent, emitter)`.
- **Copy/open:** `utils.copy_to_clipboard` (pbcopy/xclip/xsel + OSC 52), `_open_paths`.
- **Pure-Rich helpers** moved out of Textual: `text_with_path_links`, `render_welcome_banner`,
  `_WAVE_COLORS` (zero Textual deps).

## New

- prompt_toolkit input + `patch_stdout(raw=True)`; `FileHistory` at `~/.okonomi/input_history`
  (same file — replaces hand-rolled `_load_history`/`_append_history`).
- **`ReplTranscript`** — a *fork* of `StdoutPrinter` (new `EventSink`, not an overload).
- **Real OSC 8 hyperlinks** — assistant lines + tool output/details routed through
  `text_with_path_links` → `console.print(text)` emits genuine OSC 8 (terminal parses them;
  impossible under Textual which held the mouse).
- **On-demand full-screen modals** for session/asset pickers (`await app.run_async()` →
  `app.exit(result=...)`).
- **`/copy [n]`** — walks `emitter.history()` for `AssistantStream` events and copies `ev.text`
  (no new sink state; the event object already retains full text).
- REPL-owned **inline approval**: bottom prompt swaps to amber `approve> `; next accepted line
  is the decision. Not a scrollback line.
- **Tool-output verbosity + clamp** (hermes pattern): tool calls render **compact** by default
  (summary line + output clamped to first N lines with a `(+M more lines)` marker); a runtime
  toggle **`/verbose`** flips `tool_detail` to **full** (syntax-highlighted code + full output).
  Verbosity is global and affects rendering going forward — **no per-call expand, no `/show`**.
  Keeps long tool output from flooding scrollback. (The full text is always retained in
  `emitter.history()` regardless, so nothing is lost.)

## Deleted (after flag flip + verify)

`okonomi/tui/app.py` (whole Textual app + `SessionPicker`/`AssetPicker` ModalScreens), the
Textual widgets in `widgets.py` (`AssistantStreamWidget` incl. the 60Hz reveal animation,
`ToolCallWidget` + collapsible `_details_text`, `_HistoryScroll`, `_ChatInput`, status/bg/banner
widgets), all CSS, and the `Ctrl+O` expand-latest binding (no analogue in a linear stream).
Textual dropped from deps last. `StdoutPrinter` is **not** deleted.

## Renderer — `ReplTranscript` (okonomi/tui/repl/transcript.py)

Single `on_event(ev)` dispatch through module-level `Console(file=sys.stdout, force_terminal=True)`:

| Event | Rendering |
|---|---|
| **Banner** (gap) | port `render_welcome_banner()` → center each line on `get_terminal_size().columns` (stdout pads manually) |
| **UserMessage** (gap) | pink `● ` (#d787af) + `ev.text` — frames each turn + needed for replay |
| **AssistantStream** | keep per-id `_stream_seen` offset + `_stream_line_buf` newline buffering (per-completed-line reveal); each line → `text_with_path_links` with base `color(110)` → `console.print` (OSC 8 + LLM-gray). 60Hz reveal dropped. |
| **ToolCall** (biggest change) | non-collapsible. **Compact default** (`tool_detail=compact`): running line `▶ tool(args)`, output **clamped** to first N lines + `… (+M more lines)`, code body shown as a one-line summary. **Verbose** (`tool_detail=full`): full `ev.details` (execute_code → `rich.Syntax`; else verbatim) + full output. Both routed through `text_with_path_links`. **No per-call expand / no `/show`** — switch detail globally via `/verbose`. |
| **SystemNote** | dim/yellow/red one-liner by `ev.level` (approval no longer rides on `note()`) |
| **Divider** | dim 40-char rule |

## TuiSession re-homing (touchpoints)

| session.py | Textual now | REPL replacement |
|---|---|---|
| ctor (66-73) | `OkonomiApp(emitter, on_submit, *labels)` + `app.session=self` | `ReplApp(...)` same 4 callables; labels feed `bottom_toolbar`; drop `app.session` back-ref |
| run (436) | `app.run_async(inline=True, mouse=False)` | `with patch_stdout(raw=True): await app.run_async()`; subscribe `ReplTranscript`, replay `emitter.history()` first |
| 175 `app.exit()` | stop Textual loop | `app.exit(None)` → `finally: shutdown()` |
| 179 `show_session_picker` | `push_screen(SessionPicker)` | schedule nested `full_screen=True` App; `on_select(id)` unchanged |
| 183 `show_asset_picker` | `push_screen(AssetPicker)` | nested multi-select App; `on_action(action, paths)` unchanged |
| 303 `call_from_thread(show_approval)` | marshal to Textual loop | `pt_loop.call_soon_threadsafe(app.show_approval, code, _resolve)`; Future/lock/`await fut` unchanged |

## Input + modals (okonomi/tui/repl/)

- **`app.py` `ReplApp`** — `Application(full_screen=False, mouse_support=False, erase_when_done=True)`;
  `FloatContainer(HSplit([status, input, bg]), floats=[CompletionsMenu])`; multiline `TextArea`
  with `FileHistory` + `FuzzyCompleter`; `get_prompt()` = `› ` / amber `approve> `; periodic
  `invalidate()` task for the elapsed timer (**cancel in run_async finally**); approval API.
- **`key_bindings.py`** — Enter state-machine (accept-completion / approval-decision / submit),
  Ctrl+C triple-duty (skip-approval / cancel-turn / clear-draft, never exits), Ctrl+D (exit when
  buffer empty), Ctrl+L (clear scrollback only), Up/Down (completion → history → cursor), Alt+Enter newline.
- **`completion.py`** — `ThreadedCompleter(FuzzyCompleter(SlashCompleter()))`, fires only on
  leading `/`, over `SLASH_COMMANDS`; `CompletionsMenu` float self-hides.
- **`modals/session_picker.py` / `modals/asset_picker.py`** — full_screen Applications reusing
  `list_sessions` / `artifacts` / `handle_asset_action` / `fuzzy_match`; asset picker keeps
  space/a/x/o/c/h and calls `on_action` **after** alt-screen teardown so notes land in scrollback.
- **`style.py`** — okonomi palette via `Style.from_dict`.

## Threading

Capture `pt_loop = get_running_loop()` at start. Output from any thread (agent daemon, bg workers,
pt thread) just calls `ReplTranscript.on_event` → prints via `Console`; `patch_stdout(raw=True)`
makes that cross-thread-safe (`raw=True` so ANSI/OSC escapes pass through). The **only** explicit
marshal is approval prompt-state mutation via `call_soon_threadsafe(app.show_approval, …)`. Do NOT
wrap real output in `run_in_terminal` (double-lock risk). **Exactly one** stdout subscriber or text doubles.

## Ordered steps (behind `OKONOMI_REPL=1`; both front-ends runnable)

1. Extract pure-Rich helpers out of `widgets.py` → `repl/links.py`. Verify Textual app still runs.
2. Build `ReplTranscript` incl. **compact tool rendering** (clamp output to N lines + `(+M more)`; verbose mode = full code+output). Verify standalone via a canned event list (OSC 8 bytes; clamp marker in compact, `rich.Syntax` in verbose; gray base).
3. Build `ReplApp` skeleton + keys + completion + style (no modals/approval). Verify submit→stream, history, fuzzy `/`, Ctrl+D/C/L, native scroll + drag-copy.
4. Add inline approval. Verify approve/auto/skip/feedback, decision reaches agent, Ctrl+C skip, concurrent bg approval serializes.
5. Add session picker modal. Verify `/session` resume + replay, Esc, chrome repaint.
6. Add asset picker modal. Verify multi-select + o/c/h dispatch after teardown.
7. Add `/copy [n]` (via `emitter.history()`) and `/verbose` (toggle `tool_detail` compact↔full). Verify `/copy` / `/copy 2`, and that `/verbose` switches subsequent tool calls between clamped-summary and full code+output.
8. Flip default to REPL (keep `OKONOMI_REPL=0` escape hatch). Full-session verify; one-shot CLI untouched.
9. Delete Textual (`app.py`, Textual widgets, CSS); drop dep; `grep -r textual okonomi/` clean.

## Risks / open questions

- patch_stdout flicker under heavy streaming → per-line buffering mitigates; batch per emit if needed (per-line is the floor for OSC 8 link-wrapping).
- Concurrent approval ordering — verify a second `show_approval` can't flip the prompt mid-first (gated by `_approval_ui_lock`).
- Reserve `run_in_terminal`/`call_soon_threadsafe` strictly for prompt-state; keep all real output on the patch_stdout path.
- Cancel the `invalidate()` timer task in `run_async` finally (shutdown only tears down the worker loop today).
- Ctrl+C vs Ctrl+D semantics in `prompt_async` must exactly match: never kill a running turn with the wrong key.
- OSC 8 scope: keep conservative (on-disk paths + image URLs) for v1; broadening to all URLs is an open call.
- Banner centering on resize (renders once → low risk).
