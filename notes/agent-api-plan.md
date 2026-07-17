# Plan: make the agent API consumable by external TUIs

Status: PROPOSAL — no code changes yet.

## Context

Today `OkonomiAgent` has exactly one consumer — the bundled prompt_toolkit REPL — and the boundary between them shows it. The TUI reaches into six private attributes, installs callbacks by attribute assignment, polls a mutable string for status, and owns a pile of front-end-agnostic mechanics (worker thread, turn queue, cancellation marshaling, per-turn autosave, background-task spawning) that any external TUI — a different framework, a web UI, another process — would have to reimplement from scratch. The goal: a front-end can drive the agent through a stable, documented, event-driven surface without importing its internals.

## The coupling surface today (measured)

What `tui/session.py` + `commands.py` + `asset_actions.py` actually touch:

| Class | Touches |
|---|---|
| **Already clean API** | `run_async`, `save_session`, `load_session`, `reload`, `emitter`, `timeout`, `session.*` (id, title, tokens, set_title, rewind), `state.artifacts` / `message_history` / `context` |
| **Callback installation by assignment** | `approval_callback`, `on_bg_complete`, `terminal._on_bg_complete` (private of a private) |
| **Private reach-ins** | `_bg_execs`, `_bg_logs`, `_sync_session_assets(_get_dependencies())`, `_asset_dl_pool.shutdown`, `skills._load_full` |
| **Polling** | `spinner_text` (status bar reads it every frame) |
| **Mechanics owned by the TUI that belong in core** | worker loop + input queue, turn serialization, cancel marshaling, per-turn `_safe_save_session`, `/bg` full-agent task spawning + registry, atexit save |

## Design: three moves

### 1. Extract `AgentRuntime` — the headless driver (new `okonomi/runtime.py`)

Everything in `TuiSession` that isn't rendering moves into a front-end-agnostic runtime that wraps one agent:

```python
runtime = AgentRuntime(agent)          # owns worker thread + queue
runtime.submit(text)                   # enqueue a turn (thread-safe, any caller thread)
runtime.cancel()                       # cancel the in-flight turn
runtime.resume(session_id) / runtime.reset()
runtime.spawn_background(task_text)    # today's /bg mechanics, incl. artifact merge-back
runtime.status()                       # snapshot: running?, spinner label, elapsed, usage, bg jobs
runtime.jobs() / runtime.job_log(id)   # replaces _bg_execs/_bg_logs reach-ins
runtime.shutdown()                     # queue drain, loop stop, dl-pool shutdown, final save
```

- Per-turn autosave (status completed/interrupted/error) happens inside the runtime — every front-end gets crash-safe persistence for free.
- The bundled `TuiSession` shrinks to: render events, forward input, draw status — a reference client.

### 2. Everything observable becomes an event; events become the wire format

The `EventEmitter` is already the one subscription point — finish the job:

- **Approval flow inverts**: instead of the front-end assigning `agent.approval_callback`, the agent emits `ApprovalRequest(code, request_id)` and suspends; any subscriber answers via `runtime.approve(request_id, decision)` (True / False / revised code). No callback plumbing, works across a process boundary, and the single-slot UI-lock logic moves into the runtime.
- **Status stops being polled**: `spinner_text` assignments become `StatusChanged(label)` events (keep the attribute briefly for back-compat, emit alongside). `TurnStart`/`TurnEnd` already bracket turns.
- **Background jobs get lifecycle events**: `JobStarted(id, kind, label)` / `JobFinished(id, status)` replace the `on_bg_complete` / `terminal._on_bg_complete` callback pair; job output stays pull-based via `runtime.job_log(id)`.
- **Serialization contract**: every event class the agent emits gains `to_payload()` (the persistable trio already has it) and a registry for `from_payload` — one JSON schema shared by in-process subscribers, the DB's event stream, and the wire protocol below. Document it in the module docstring as the compatibility surface.

### 3. Close the remaining private pokes with facade methods

- `agent.sync_session_assets()` — public wrapper for the `/assets` sweep (hides `_get_dependencies`).
- `agent.load_skill_text(name)` — replaces `skills._load_full` for `/skill`.
- Asset-download pool shutdown folds into `runtime.shutdown()` — no front-end touches `_asset_dl_pool`.
- After the port: `grep -rn "agent\._" okonomi/tui/` must return nothing, and stays that way (add to the verification checklist; optionally a ruff rule for `SLF001` scoped to `tui/`).

## Phase 2: out-of-process consumption (`okonomi serve`)

With the runtime + JSON-able events in place, a wire layer is thin:

- A WebSocket/JSON-RPC server wrapping one `AgentRuntime`: requests (`submit`, `cancel`, `approve`, `resume`, `status`, `jobs`, plus the `sessions.recall` surface) and a streamed event channel (`{event_type, payload, id}` frames — exactly `to_payload()` output).
- Prior art to study when we get here: hermes-agent's `gateway/` + `acp_adapter/` (agent-client-protocol) — if ACP's shapes fit, adopting them buys compatibility with existing ACP clients instead of inventing a protocol.
- Explicitly phase 2: the in-process API must land and the bundled TUI must dogfood it first — the wire layer serializes an API that already works, it doesn't define one.

## Out of scope

- Multi-agent / multi-session concurrent serving (one runtime = one live session; parallel work stays `spawn_background`).
- AuthN/Z on the wire layer (localhost, personal tool).
- Non-TUI clients (the API won't preclude them; they're just not the design target).
- Rewriting the bundled REPL's rendering.

## Rollout order

1. `AgentRuntime` extraction (mechanics out of `TuiSession`), bundled TUI ported onto it.
2. ✅ DONE — Approval-as-event (`ApprovalRequest` + `respond` handle, `approvals_enabled` flag), `StatusChanged` on spinner writes, `JobStarted`/`JobFinished` replacing the `on_bg_complete` callback pair, `EVENT_REGISTRY` via `__init_subclass__` + JSON payload round-trip. Callback attributes deleted; bg agents forward approval/job events to the parent emitter.
3. Facade methods for the remaining private pokes; `agent._` grep goes to zero in `tui/`.
4. `to_payload()` across all emitted events + documented event schema.
5. (Phase 2) `okonomi serve` wire layer; evaluate ACP compatibility against hermes's adapter.

## Verification

- Regression: all existing suites (round-trip, events, commands, export/accounting) pass after each step — the runtime extraction must not change turn semantics.
- Headless-client test: a script that drives `AgentRuntime` with no TUI — submit → collect events → answer an `ApprovalRequest` → cancel mid-turn → confirm autosave stamped `interrupted` — proves the API is sufficient without a front-end.
- Coupling gate: `grep -rn "agent\._\|approval_callback\|on_bg_complete\|spinner_text" okonomi/tui/` → empty.
- Event-schema round-trip: every emitted event type `to_payload()` → JSON → `from_payload()` equals itself.
- Manual: bundled REPL smoke (`okonomi`, one turn, `/bg`, approval prompt, Ctrl+C cancel, `/jobs`, quit-resume) behaves identically on the new plumbing.
