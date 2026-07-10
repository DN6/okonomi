# Queue-drain / input-queuing — scope

Comparing okonomi's input handling against mistral-vibe's "message-queue drain
engine," and scoping what (if anything) is worth adopting.

## Key finding: okonomi already queues

The submit path:

`on_chat_input_submitted` (`app.py:827`) → `_on_submit` (`session.py:159`) →
`_submit` (`session.py:201`) → `call_soon_threadsafe(self._input_queue.put_nowait)`.

That `_input_queue` is an `asyncio.Queue` consumed one item at a time by
`_worker_loop` (`session.py:82-90`).

So typing while the agent is busy today: the message is appended to the
transcript immediately (`session.py:167-168`), enqueued, and runs as **its own
turn** after the current one finishes. FIFO serialization already exists.

The mistral-vibe "drain engine" is therefore **not a missing primitive** —
okonomi has the queue. The genuine deltas are only two:

1. **No batching/coalescing.** Each queued message = a separate `run_async`
   turn. Three quick thoughts → three sequential turns, not one combined turn.
   mistral-vibe's "drain" pulls *all* pending items and feeds them as one turn.
2. **No queued-state affordance.** A submitted-while-busy message looks
   identical to a normal one — it silently sits with no response until its turn
   comes. No "queued" badge, no pending count.

## The decision to make (the real fork)

**Should multiple messages typed during a turn become one combined turn, or
stay as separate sequential turns?**

- **Batch them (mistral-vibe's choice):** "let me add three things while it
  works" → agent sees all three together, responds once with full context.
  Better when messages refine one ask. Risk: combining unrelated thoughts into
  one turn muddies intent.
- **Keep separate (okonomi today):** each message is its own turn. Better when
  messages are distinct asks. Cost: the agent can't see message 2 while
  answering message 1, so early turns lack later context.

## Scoped options

### Option A — visibility only (small, ~30–50 lines)
Keep one-turn-per-message. When `_on_submit` runs while `self._agent_running`,
mark that user message widget as "queued" (dim + a `⋯ queued` badge) and surface
a pending count in the status line.

Touch points:
- a `queued` flag on the `UserMessage` event / `UserMessageWidget`
  (`events.py`, `widgets.py`)
- a pending counter exposed via the existing `bg_tasks_fn`/spinner callback wiring
- clear the badge when `_worker_loop` picks the item up

No change to turn semantics.

### Option B — drain + batch (medium, ~80–120 lines)
A's visibility, plus change `_worker_loop` to drain the whole queue when a turn
ends and combine into one `run_async` call.

Touch point localized to `_worker_loop` (`session.py:82-90`): after
`_input_queue.get()`, also `get_nowait()` until empty, join with separators, run
once.

Edge cases:
- the `None` shutdown sentinel mixed into a drained batch
- `[Side question]` / `[system]` tagged messages (`session.py:219`, `337`)
  which must **not** be naively concatenated — they have their own semantics

### Option C — manage the queue (large)
B plus editing/removing pending items before they run. Needs the queue to become
inspectable state (it's an opaque `asyncio.Queue` now) plus a UI surface. Skip
unless specifically wanted — most of the cost for the least-used capability.

## Recommendation

**Start with A.** It closes the actual felt gap (you can't tell a message is
queued) for minimal change and zero turn-semantics risk. Whether to then add B's
batching depends on the batch-vs-separate fork above — decide that after living
with A.
