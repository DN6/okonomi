# Plan: search & retrieval for conversation data and media assets

Status: PROPOSAL — no code changes yet.

Decisions (user-confirmed):

1. **SQLite is the single source of truth** for session metadata, the conversation, and artifact records, with FTS across all of it. `messages.jsonl` stops being the record; an **export utility** dumps any session to JSONL on demand. No backwards compatibility with the old layout.
2. **The store captures the entire event stream**, not just LLM messages — system notes, tool activity, title changes, interruptions — one table, conversation messages as one event type (Claude Code's transcript model).
3. **Per-message accounting** — token/cost columns on each model response event; session totals become cached aggregates that are always rebuildable by SUM.
4. **DAG structure** — every event carries a parent pointer, so rewind, forks, and sidechains are representable (Claude Code's `parentUuid`, hermes's `active`-flag semantics). Linear conversations are just the degenerate chain.

## Why this shape (evidence from the three systems studied)

- **okonomi today**: transcripts in per-session jsonl, rewritten wholesale every save (O(history) per turn, quadratic per session), invisible to search; usage only as session totals; UI events ephemeral (EventEmitter history dies with the process); `/clear` leaves a stale-jsonl ghost that resurrects on resume.
- **hermes-agent**: messages as SQLite rows + trigger-synced FTS5 — search is one `MATCH` with BM25 and `snippet()`; the index can't drift because it updates in the message's own transaction. Its mistake: indexes raw content verbatim (tool JSON, base64), filtering only at query time. Its win we adopt: lifecycle-flag search visibility and budget-shaped retrieval (snippets, ±windows, bookends).
- **Claude Code**: the transcript is an *event log* (user/assistant messages interleaved with system/attachment/title/mode records), every record enveloped with `uuid`/`parentUuid`/`timestamp` forming a DAG, and per-assistant-message `usage`. It needed that envelope because a flat file was its only store; we get identity and ordering as columns for free.

## Storage design

### `events` table (replaces both `messages.jsonl` and the plan's earlier `messages` table)

```sql
CREATE TABLE events (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_id    INTEGER NOT NULL,           -- insertion order within the session: 0, 1, 2, …
    parent_id   INTEGER,                    -- DAG edge: preceding event on this branch (NULL = root)
    type        TEXT NOT NULL,              -- EventType enum value (closed set, see below)
    payload     TEXT NOT NULL,              -- full JSON record (pydantic-ai message, or the typed event)
    content     TEXT NOT NULL DEFAULT '',   -- extracted text (what FTS indexes) — lossy; payload is the record
    timestamp   TEXT NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1, -- 0 = rewound/abandoned branch (hermes semantics)
    -- per-message accounting (response events; NULL elsewhere)
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,
    reasoning_tokens    INTEGER,
    cost_usd            REAL,
    PRIMARY KEY (session_id, event_id)
);
```

- **`event_id` is per-session insertion order** (position, not identity across sessions); a record is referenced by `(session_id, event_id)`. Contiguous numbering keeps windowed retrieval a plain `BETWEEN`.
- **DAG**: `parent_id` points at the preceding event on the same branch. In the normal case every event's parent is simply the previous one — a chain. Rewind marks the abandoned suffix `active = 0` and the next event's parent points above it; forks/sidechains are additional children of one parent. The **active path** is materialized by the `active` flag (hermes's approach — a flag update on rewind), so resume is `WHERE active = 1 ORDER BY event_id` with no recursive walk; the DAG remains fully reconstructable from `parent_id` for history/undo tooling.
- **`type` is a closed enum** (`EventType`, a `StrEnum` in code) — the event universe is bounded, and every member has a corresponding typed object that defines its payload shape:

  | EventType | Payload object | Notes |
  |---|---|---|
  | `user_message` | pydantic-ai `ModelRequest` (whole message) | request containing a user prompt |
  | `tool_response` | pydantic-ai `ModelRequest` (whole message) | request containing only tool returns |
  | `assistant_message` | pydantic-ai `ModelResponse` (whole message) | text-only response |
  | `tool_call` | pydantic-ai `ModelResponse` (whole message) | response containing tool calls; carries the accounting columns |
  | `system_note` | `events.SystemNote` | warnings, bg-task completions |
  | `title` | small `{title}` record | out-of-band retitles become visible history |
  | `interrupt` | small `{reason}` record | cancellations / timeouts |

  The four conversation types keep the **whole pydantic-ai message as payload** — lossless resume via `ModelMessagesTypeAdapter`, no part-level explosion, no reassembly logic. The type is *derived from the message's part composition at write time* (user prompt present → `user_message`; only tool returns → `tool_response`; any tool call → `tool_call`; else `assistant_message`) — a semantic index over the stream, not a change to what's stored. Non-conversation types serialize their existing typed objects from `okonomi/events.py`, which today die with the process. Adding an event type means adding an enum member + object — nothing is stringly-typed. No double-storing of assistant text — UI transcript replay renders conversation rows plus non-message events.
- **`content`** is the write-time extraction: user-prompt text and assistant `TextPart`s (plus system-note text), capped ~4 KB/row; tool-call args, tool returns, base64/data-URIs, HTML dumps, and the embedded system prompt are stripped. Deliberately lossy — display and round-tripping always go through `payload`. Filtering at write keeps the index clean (hermes's unsolved noise problem).

### `events.py` = agent events only; UI events defined separately

The module split follows **who produces the event** (the agent can't import from the TUI layer, so the split is also the dependency direction):

**`okonomi/events.py` — agent events.** The persistence taxonomy lives here:
- `EventType` StrEnum + `CONVERSATION_TYPES` — the bounded universe.
- Persistable payload objects with a `to_payload()`/`from_payload()` contract: `SystemNote`, plus new `Title` and `Interrupt` dataclasses. (The four conversation types need no new classes — their payloads are pydantic-ai messages.)
- Agent-produced live-rendering events: `TurnStart`/`TurnEnd`, `AssistantStream`, the tool-call display event (renamed `ToolCallDisplay` to avoid colliding with `EventType.tool_call`). These are emitted by the agent but never persisted — their content is already in the conversation rows; persisting them would double-store.
- `EventEmitter` stays here as the shared transport (agent emits, TUI subscribes).

**`okonomi/tui/events.py` (new) — UI chrome.** Events the TUI itself emits for its own scrollback: `Banner`, `Divider`, `UserMessage` (the echo of typed input). Never persisted, never seen by the agent.

Later, `replay_history` inverts: instead of reconstructing display events from message history, it renders from DB rows (conversation rows + persisted notes/titles/interrupts) — full-fidelity scrollback replay from the store.

### Per-message accounting

- Response events carry their own token/cost columns, extracted from pydantic-ai `usage` + OpenRouter `provider_details["cost"]` at insert — the same values `_accumulate_usage` computes per run today, just attributed to the message that incurred them.
- `sessions`' token columns become **cached aggregates** (kept for cheap listing), verifiable and rebuildable via `SUM(...)` over the conversation types (only `tool_call`/`assistant_message` rows carry values; `SUM` ignores NULLs elsewhere). Cost-per-turn, cost-over-time, and "which message burned the tokens" become one GROUP BY.

### `events_fts` (FTS5, trigger-synced)

```sql
CREATE VIRTUAL TABLE events_fts USING fts5(content, tokenize='porter unicode61');
-- INSERT/UPDATE/DELETE triggers on events mirror content, rowid = events.rowid
```

In-transaction sync: zero drift, no reindex job. Search visibility respects lifecycle: default queries filter `active = 1` (rewound branches vanish from search but stay in the DAG).

### Write path

- `Session.save()` **appends**: `persisted = COUNT(*) WHERE session_id = ?`, then INSERTs the new tail of the event stream with `event_id = persisted, persisted+1, …`, parent chaining to the prior row. Message events come from `message_history`; non-message events come from the EventEmitter's history since last save (the emitter already produces typed events — today they die with the process).
- `/clear` becomes a **rewind**: mark the session's events `active = 0` instead of deleting — history preserved in the DAG, invisible to resume and search. (This replaces the old delete-and-reinsert reconciliation and fixes the stale-jsonl ghost bug by construction.)
- `messages.jsonl` is no longer written. `globals.pkl` stays a file (binary, non-searchable — correctly so).

### Reads

- Resume: `SELECT payload WHERE session_id = ? AND type IN (<conversation types>) AND active = 1 ORDER BY event_id` → adapter validation. `CONVERSATION_TYPES = (user_message, tool_response, assistant_message, tool_call)` lives next to the enum as the one canonical constant.
- Windows: `event_id BETWEEN N-K AND N+K` (optionally conversation-filtered); head/tail reads are `LIMIT` clauses.
- Full TUI replay: all active events in order — the scrollback can be reconstructed from the store, not just the model context.
- Deletion: session-row delete cascades to events (and artifacts, as today).

### Export utility

`okonomi export <session-id> [--out path]`: dumps active message events' `payload`s in order as JSONL (byte-compatible with the old format); `--events` variant dumps the full event stream with envelope columns. For backup, sharing, offline grepping.

## Search: one query surface over everything

### Unified `search(query)` — module function + model tool

Fans out over `sessions_fts` (title + task_input), `events_fts` (conversation + system notes), and `artifacts_fts` (+ url/path LIKE fallback), merged by normalized BM25.

Every hit is a low-friction reference: `{source: session|event|artifact, session_id, session_title, score, snippet | prompt, event_id | artifact_id, path | url, timestamp}` — `(session_id, event_id)` pins any conversation moment, `artifact_id` any asset; enough to open, quote, or attach without a second lookup.

**Query sanitization** (from hermes): preserve quoted phrases, strip FTS5 syntax chars, auto-quote dotted/hyphenated tokens; replaces today's try/except-OperationalError fallback.

### Retrieval shaped for LLM budget

- **Hit** → snippet + ±K window via `load_messages(around_event_id=...)`.
- **Bookends**: discovery results include the session's first ~3 and last ~3 user/assistant messages (goal + resolution) — one call reconstructs *goal → matched moment → outcome*.
- Whole-session reads cap at head 20 + tail 10. Search results never carry full payloads — snippets and windows only.

### Tool shape: one tool, behavior inferred from args

| Args given | Behavior |
|---|---|
| `query` | discover: ranked hits across events/sessions/artifacts, snippets + bookends |
| `session_id` + `around_event_id` | scroll: ±window |
| `session_id` | read: capped head+tail + metadata + artifacts |
| none | browse: recent sessions |

Type filtering at query time (default: `user_message` + `assistant_message`) — tool activity, system notes, and any future indexed tool output are opt-in filters.

### Human surface

`/session` picker falls through to `search()` when the fuzzy title filter finds nothing — "that theyyam video" works from the picker. Artifact hits open the asset; event hits resume the session.

## Media assets: make every asset findable

1. **Swept media gets text at registration** — stamp `description` with filename + current `task_input`. Free, no LLM.
2. **Index the full generation prompt** (drop the 500-char truncation from the indexed text).
3. **Stable references** — artifact hits expose the artifact row `id` (favorites already key on it).
4. **Optional, flagged off: out-of-band image captioner** (titler pattern) for visual-content search. Only piece that costs money.

## Explicitly out of scope

- **Embeddings / semantic search** — BM25 over well-extracted text first.
- **CJK tokenization** (hermes needed a whole second trigram FTS table).
- Indexing `execution_globals` or raw tool outputs.
- Migration/import of old messages.jsonl stores (export is one-directional, DB → file).
- Cross-session DAG (session-level forks already have `parent_session_id`; event-level DAG is within-session for now).

## Rollout order

0. Housekeeping: rename `db.connection_for()` → `db.connection()` (the `for` is a fossil from when it took a `storage_dir` argument; it's been zero-arg since the store became a constant). Delete `Session._conn` for the same reason — it closed over `self.storage_dir` once; now it's a zero-information alias, so `Session` methods call `db.connection()` directly like the module-level functions already do.
1. `EventType` enum + the events.py/tui-events module split, then the `events` table + FTS triggers + append-on-save + row-backed resume (drop jsonl writes). Conversation events only at first; `parent_id`/`active` written but rewind UI deferred.
2. Per-message accounting columns + session totals as verified aggregates.
3. Export utility.
4. Query sanitizer + unified `search()` + windows + bookends; single args-inferred tool; picker fall-through.
5. Non-message event types persisted (system notes, titles, interrupts) → full replay from store; `/clear` as rewind.
6. Media text enrichment; (flagged) captioner.

## Verification sketch

- Turn round-trip: run turns → message events appended (active count == len(history)); resume reproduces history through the adapter; per-response token/cost columns populated and `SUM == session totals`.
- Rewind: `/clear` → events flagged inactive; resume yields empty history; the flagged rows still visible to a DAG/history query; search excludes them.
- Search: seed a phrase mid-conversation → hit with snippet + bookends; delete session → cascade removes events and hits.
- Export: reproduces the old jsonl format; re-validates through the adapter.
- Media: matplotlib-written file findable by task words; artifact hit heart-able by id.
- Perf: 200-turn session — save appends the delta, not 200 rows; window reads ~11 rows.
