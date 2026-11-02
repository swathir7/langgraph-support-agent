# Architecture decision record

## Why a dedicated human-approval node, not an inline check

The order agent could have checked "is this risky?" and called the mutating tool directly with a
confirmation flag. Instead, refund/cancel routes through a separate `human_approval` node reached
by a conditional edge. Two reasons:

1. **The graph topology documents the safety property.** Anyone reading `graph/build.py` or the
   `/graph/diagram` output sees, structurally, that there is no edge from `order_agent` to a
   mutating call without passing through `human_approval` first — it's not a fact you have to
   trust a docstring for.
2. **LangGraph's `interrupt()` is a node-level primitive.** It pauses the *node* it's called from
   and resumes that same node from the top. Keeping the interrupt in its own node keeps the
   "what re-runs on resume" surface area to exactly the approval logic, not the order lookup that
   already happened.

## Why retry → timeout → circuit breaker, composed once

`resilience.py`'s `call_with_resilience()` is the only place these three compose. Order matters:
the circuit breaker is outermost, so a tripped breaker short-circuits before even attempting a
retry loop — retrying against a backend already known to be down just adds latency for nothing.
Both `order_agent.py` and `human_approval.py` call this one function rather than each assembling
retry+timeout+breaker themselves, which is exactly the kind of duplication that drifts out of
sync (one call site retries on `ConnectionError`, the other forgets `TimeoutError`) if written
twice.

## Why the circuit breaker only counts `RetryExhaustedError`

A naive circuit breaker counts every exception as a failure. That's wrong here: `OrderNotFoundError`
(a user typo'd an order id) is a valid business outcome, not a sign the order API is unhealthy —
counting it would let a string of typos trip the breaker and lock out every *other* conversation
using the same shared `MockOrderApiClient`/circuit breaker instance. `call_with_resilience()`
narrows `count_failure_on` to `(RetryExhaustedError,)` specifically: that's the one exception type
that only appears after retries have already been exhausted against a type in `retry_on`, so it's
a real signal, not user error being misread as infrastructure failure.

## Why routing priority is order-id/action-keyword → knowledge-base → weak-keyword, in that order

`RuleBasedClassifier` (`graph/router.py`) doesn't just OR together keyword sets. An earlier version
checked generic keywords ("track", "order", "status") before checking the FAQ, and misrouted "How
do I track my order?" — a literal FAQ question — to the order-lookup agent, which then had nothing
to look up. The fix: an explicit order id or an unambiguous action word (refund/cancel) always
means order intent (nothing to weigh); otherwise, a knowledge-base match is checked *before*
generic order-ish keywords, because those keywords are exactly as likely to appear in a how-to
question as in an order-lookup request. Only when neither an order id/action word nor a
knowledge-base match exists do the generic keywords fall through to "order" — where the order
agent asks for the missing order id rather than guessing.

## Why the eval endpoint builds a fresh graph instead of reusing the live one

`GET /eval/run` doesn't touch `app.state.graph` (the live, SQLite-persisted graph real
conversations use). It builds a scratch graph with its own `MemorySaver` and fresh
`MockOrderApiClient`, using the same `Settings` (so it exercises the same classifier choice the
live app would). This makes an eval run reproducible regardless of what a human tester has
already done through the demo chat UI — refunding ORD-1002 in the live demo doesn't change what
the next eval run reports.
