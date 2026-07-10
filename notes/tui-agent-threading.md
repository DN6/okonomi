# TUI ↔ Agent threading model

How `TuiSession` connects the Textual UI to the agent without them stepping on each other.

## The three things called "loop"

| Name | What it is |
|------|------------|
| `self._loop` | An `asyncio` event loop. The runtime that schedules and runs coroutines. |
| `_worker_loop()` | A single coroutine running on `self._loop`. Its body is `while True: msg = await queue.get(); run_turn(msg)`. The "loop" is the `while True`. |
| `loop.run_forever()` | An infinite loop *inside asyncio* that pumps the event loop — looks at callbacks, resumes coroutines, polls I/O, blocks when idle. |

## Thread layout

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│   MAIN THREAD                   │         │   _agent_thread (daemon)         │
│   (Python's actual main)        │         │                                  │
│                                 │         │                                  │
│   Textual UI event loop         │         │   self._loop (asyncio)           │
│   ─────────────────────         │         │   ────────────────────           │
│   • draws frames                │         │   • runs _worker_loop coroutine  │
│   • handles keystrokes          │         │     forever (await queue.get())  │
│   • dispatches widget events    │         │   • runs one _run_agent_turn     │
│   • runs OkonomiApp callbacks   │         │     per dequeued message         │
└─────────────────────────────────┘         └──────────────────────────────────┘
              │      ▲                                    │      ▲
              │      │                                    │      │
              ▼      │                                    ▼      │
       ┌──────────────────┐                       ┌──────────────────┐
       │ Widgets / Input  │                       │ agent.run_async  │
       │ TextArea, Picker │                       │ (LLM, tools…)    │
       └──────────────────┘                       └──────────────────┘
```

The agent doesn't own a loop. It just defines async methods. Whoever calls `await agent.run_async(...)` provides the loop. On `_agent_thread`, the session's bookkeeping coroutines (`_worker_loop`, `_run_agent_turn`) and the agent's work (`run_async`) all run on the same loop — the session *wraps* the agent call, it isn't a peer running concurrently.

Background tasks are different: `_spawn_bg_task` uses `asyncio.run(bg_agent.run_async(...))` which spins up a *fresh* loop on a *fresh* thread per bg task, so a bg task doesn't fight the foreground agent for the same loop.

## Why a separate thread for the agent loop

1. **Textual already owns the main thread's loop.** When you call `app.run_async()`, Textual takes over the main thread's loop for rendering, input, animation timers. We could schedule agent coroutines onto that same loop, but any non-awaited CPU burst inside agent code (pickling session state, tokenizing for context counts, hashing files, building large prompts) would stall UI frames.
2. **Cancellation isolation.** Stopping the agent loop on shutdown doesn't bring down the UI.
3. **(Historical)** The previous design ran a fresh `asyncio.new_event_loop()` per turn on a per-turn thread. Switching to a persistent thread + persistent loop was the recent refactor; keeping the agent loop separate from Textual's kept the patch small.

## What `_agent_thread` actually does

Its `target` is `self._loop.run_forever`. That's its entire job — call `loop.run_forever()` and never return until the loop is stopped.

At any given moment, almost always: **blocked in the OS (`epoll_wait` / `kqueue`) waiting for something to happen on its loop.** When work arrives (a queued message, an HTTP response, a timer firing), it wakes up, runs one slice of work, and goes back to sleep.

### Concrete walkthrough — what happens across a turn

```
t=0    user starts the app
       __init__ creates self._loop (idle, not running)
       _agent_thread.start()  →  thread enters loop.run_forever()
                                  loop is now running, but nothing to do
                                  thread blocks in OS waiting for work

t=1    run_coroutine_threadsafe schedules _worker_loop()
       loop wakes, starts _worker_loop()
       _worker_loop hits `await queue.get()` → suspends
       loop has nothing else to do → thread blocks again

t=2    user submits "hello"  (this happens on Textual thread)
       Textual thread calls call_soon_threadsafe(...) on our loop
       OS wakes _agent_thread
       loop runs the callback: put_nowait("hello")
       queue.get() now has a value → resumes _worker_loop
       _worker_loop calls _run_agent_turn("hello")
       _run_agent_turn calls `await agent.run_async(...)`
       agent code starts running on this loop

t=3    agent does `await httpx.post(...)` to OpenRouter
       coroutine suspends, loop has nothing to do
       _agent_thread blocks in epoll waiting for the socket

t=4    HTTP response arrives
       kernel wakes _agent_thread
       loop resumes agent at the await point
       agent parses response, picks a tool, calls another await...
       (repeat for many cycles)

t=N    agent.run_async returns
       _run_agent_turn returns
       _worker_loop loops back to `await queue.get()` → suspends
       _agent_thread blocks again, waiting for next user message
```

## Kitchen metaphor

If asyncio is a kitchen, the **loop** is the kitchen itself (orders board, ingredients, timers, all state). The **thread** is the single chef standing in it. One chef can only work one kitchen. Without a chef, no orders get cooked even though the kitchen is set up. `_agent_thread` is "the chef for the agent's kitchen." Textual has its own chef working its own kitchen on the main thread.

## `daemon=True` ≠ blocking

These describe different things and are orthogonal.

- **`daemon=True`** is a *shutdown flag*. Tells Python: "when the main thread exits, don't wait for this thread — just kill it." Changes nothing about how the thread runs while the program is alive.
- **Blocking** is what threads spend most of their time doing. A thread can be *running* (executing bytecode, using a CPU core) or *blocked* (suspended inside a system call like `epoll_wait`, consuming no CPU, waiting for the kernel to wake it).

```
Program exits:
    non-daemon thread blocked in epoll  →  Python waits for it to finish
                                            (interpreter hangs until joined)

    daemon thread blocked in epoll      →  Python kills the thread mid-block
                                            (interpreter exits immediately)
```

We set `daemon=True` on `_agent_thread` so that if the user force-quits or an exception escapes shutdown, the program doesn't hang waiting for the thread parked in `epoll_wait`. If `run_forever` *didn't* block, it would be a busy-spin pegging a core at 100% — blocking is fundamental to how event loops save CPU.

## The three cross-thread bridges

The two threads never share mutable state directly. Everything crosses through one of:

- `self._loop.call_soon_threadsafe(...)` — Textual → agent loop
- `self._app.call_from_thread(...)` — agent → Textual loop
- `asyncio.Future` resolved via `call_soon_threadsafe` — agent waits, Textual delivers

### A. User types a message → agent runs it

```
[Textual thread]                              [agent thread]
─────────────────                             ────────────────
user hits Enter in TextArea
   │
   ▼
_on_submit("hello")
   │
   ▼
self._loop.call_soon_threadsafe(            ───►   _worker_loop awakes:
   _input_queue.put_nowait, "hello")                msg = await queue.get()
                                                    │
                                                    ▼
                                                    _run_agent_turn("hello")
                                                    └─► agent.run_async(...)
                                                        emits events ──┐
                                                                       │
[Textual thread]                                                       │
─────────────────                                                      │
_render_event(event)        ◄──── call_from_thread(...) ◄──── _on_event(event)
   │                                                          (fires on agent thread,
   ▼                                                           marshals to UI)
draws widget
```

### B. User hits Ctrl+C → agent turn cancels

```
[Textual thread]                              [agent thread]
─────────────────                             ────────────────
action_interrupt
   │
   ▼
session.cancel_current_turn()
   │
   ▼
self._loop.call_soon_threadsafe(            ───►   _do_cancel() runs on loop:
   _do_cancel)                                       self._current_task.cancel()
                                                     │
                                                     ▼
                                                     CancelledError raised inside
                                                     agent.run_async → caught by
                                                     _run_agent_turn → status="interrupted"
```

### C. Agent asks for approval → user responds

```
[agent thread]                                [Textual thread]
────────────────                              ─────────────────
agent hits a guarded tool
   │
   ▼
await self._approval_callback(code)
   │
   ├─► fut = loop.create_future()
   │
   ├─► self._app.call_from_thread(        ───►  show_approval(code, resolver)
   │      self._app.show_approval,                renders amber-tinted input box
   │      code, _resolve)                         waits for user submission
   │                                              │
   ├─► await fut         ◄─── set_result ◄───    user types "y" / "n" / feedback
   │   (blocks here)                              │
   │                                              ▼
   │                                              resolver(decision)
   │                                                │
   │                            call_soon_threadsafe(fut.set_result, decision)
   │                                                │
   ▼                                                ▼
returns decision      ◄─── future resolves on agent loop
agent proceeds
```

The `EventEmitter` is the one shared object — but its subscribers (the Textual app) are responsible for marshalling onto their own thread before touching widgets. That's what `_on_event` does in `okonomi/tui/app.py`.

## Why `_submit` exists as a helper

```python
def _submit(self, msg: str | None) -> None:
    self._loop.call_soon_threadsafe(self._input_queue.put_nowait, msg)
```

`asyncio.Queue.put_nowait` is **not thread-safe**. From the Textual thread, you can't call it directly — it has to run on the loop that owns the queue. If someone forgets and just does `self._input_queue.put_nowait(msg)` from the Textual thread, it'll *appear* to work until the queue's internal state races and a message is silently dropped. The helper makes the wrong way harder to write.
