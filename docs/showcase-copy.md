# Showcase copy

These drafts deliberately describe the public architecture only. Do not mention a prospective
client, their business, or errors that inspired the project.

## LinkedIn post

Most public LangGraph demos show a single agent calling a tool. I built the piece that's
actually hard to get right: a support agent that knows when it isn't sure, and never takes a
risky action without asking first.

**Reliable LangGraph Support Agent** is a router/supervisor graph — knowledge lookup, order
management, human-approval gate, escalation — with reliability engineering baked into the parts
that call an external system:

- retries with backoff, per-call timeouts, and a circuit breaker around every order-API call,
  composed once and reused everywhere that call happens;
- refunds and cancellations pause the graph via LangGraph's `interrupt()` and wait for an
  explicit human approve/deny before anything mutates — the graph's structure itself proves
  there's no path around that gate, not just a comment saying so;
- the router's confidence score, not just its route guess, decides whether a message reaches an
  agent at all or goes straight to "let me connect you with a human";
- conversation state and pending approvals persist to SQLite, surviving a process restart;
- an evaluation harness runs a labeled scenario set through the real compiled graph and reports
  routing/approval/escalation accuracy — not a demo that only works on the happy path.

Repository: [add GitHub link after publishing]

## Upwork portfolio description

**Reliable LangGraph Support Agent — Router, Tools, Human-in-the-Loop, Reliability Engineering**

Designed and implemented a production-shaped LangGraph support agent, focused on the parts that
separate a demo from something you'd trust with real actions.

Highlights:

- Router/supervisor node with a pluggable classifier (deterministic rule-based by default,
  real LLM-backed alternative behind the same interface)
- Knowledge-retrieval agent and an order-management tool agent
- Human-approval interrupt node gating every risky action (refund, cancel) — LangGraph's
  `interrupt()`/`Command(resume=...)` pattern, with idempotency handled correctly across the
  pause/resume boundary
- Retry-with-backoff, per-call timeout, and a circuit breaker composed once and reused around
  every external call, with business errors (e.g. "not found") deliberately excluded from
  tripping the breaker
- Structured output validation (Pydantic) at every node boundary
- Escalation to a human when router confidence is low, independent of which route it guessed
- Persistent, restart-safe conversation state (SQLite-backed checkpointing)
- Per-node tracing surfaced in the UI, spanning the full approval round trip
- A routing-correctness evaluation harness with a labeled scenario set, run through the real
  production graph
- FastAPI backend, 66 tests (no live API key or network calls required), vanilla-JS chat UI

Technology: Python, LangGraph, FastAPI, Pydantic, SQLite, OpenAI API (optional).

## Screenshot checklist before publishing

1. Chat screen showing a refund request paused for approval, with the Approve/Deny buttons.
2. The same conversation after approval, showing the completed action and the trace timing.
3. The evaluation results table showing all scenarios passing.
4. The architecture diagram from the README.
5. Never show real customer data or API keys.
