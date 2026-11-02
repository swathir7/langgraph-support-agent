"""The human-approval gate for risky actions (refund, cancel).

Uses LangGraph's `interrupt()` — the first time this node runs for a given
`pending_action`, `interrupt()` pauses the graph and the payload it's given
becomes part of `.invoke()`'s result under `__interrupt__`. The caller
(`app.py`'s `/chat/resume`) later resumes the *same* thread with
`Command(resume=<decision>)`, which re-runs this node from the top and this
time `interrupt()` returns `<decision>` instead of pausing again.

Because the node reruns from the top on resume, everything before the
`interrupt()` call must be side-effect-free — it is here: reading
`state["pending_action"]` has no side effects. The mutating tool call only
happens after `interrupt()` returns, so it never runs twice.
"""

from __future__ import annotations

import time

from langgraph.types import interrupt

from ..config import Settings
from ..resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryExhaustedError,
    RetryPolicy,
    call_with_resilience,
)
from ..state import AgentState
from ..tools.orders import MockOrderApiClient
from .order_agent import RETRYABLE_ORDER_ERRORS


def _approval_prompt(pending: dict) -> str:
    if pending["action"] == "refund":
        return f"Approve a ${pending['amount']:.2f} refund for order {pending['order_id']} ({pending['item']})?"
    return f"Approve cancelling order {pending['order_id']} ({pending['item']})?"


def make_human_approval_node(client: MockOrderApiClient, settings: Settings, circuit_breaker: CircuitBreaker):
    retry_policy = RetryPolicy(
        max_attempts=settings.order_retry_max_attempts, base_delay_seconds=settings.order_retry_base_delay_seconds
    )

    async def human_approval_node(state: AgentState) -> dict:
        started = time.monotonic()
        pending = state["pending_action"]
        assert pending is not None, "human_approval_node reached with no pending_action set."

        decision = interrupt(
            {
                "type": "approval_required",
                "action": pending["action"],
                "order_id": pending["order_id"],
                "amount": pending["amount"],
                "item": pending["item"],
                "prompt": _approval_prompt(pending),
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

        if not approved:
            reply = f"Understood — I won't {pending['action']} order {pending['order_id']}. No action was taken."
            return _finish(started, reply, tool_result=None)

        try:
            if pending["action"] == "refund":
                result = await call_with_resilience(
                    lambda: client.refund_order(pending["order_id"], pending["amount"]),
                    retry_policy=retry_policy,
                    timeout_seconds=settings.order_call_timeout_seconds,
                    circuit_breaker=circuit_breaker,
                    retry_on=RETRYABLE_ORDER_ERRORS,
                )
                reply = f"Done — refunded ${pending['amount']:.2f} for order {pending['order_id']}."
            else:
                result = await call_with_resilience(
                    lambda: client.cancel_order(pending["order_id"]),
                    retry_policy=retry_policy,
                    timeout_seconds=settings.order_call_timeout_seconds,
                    circuit_breaker=circuit_breaker,
                    retry_on=RETRYABLE_ORDER_ERRORS,
                )
                reply = f"Done — order {pending['order_id']} has been cancelled."
        except (CircuitOpenError, RetryExhaustedError):
            reply = (
                "This was approved, but our order system is unavailable right now, so I couldn't complete "
                "it. Please try again shortly — your approval doesn't need to be repeated."
            )
            return _finish(started, reply, tool_result=None, escalated=True)

        return _finish(started, reply, tool_result={"executed": result.model_dump()})

    def _finish(started: float, reply: str, *, tool_result: dict | None, escalated: bool = False) -> dict:
        duration_ms = (time.monotonic() - started) * 1000
        return {
            "reply": reply,
            "pending_action": None,
            "tool_result": tool_result,
            "escalated": escalated,
            "messages": [{"role": "assistant", "content": reply}],
            "trace": [{"node": "human_approval", "duration_ms": duration_ms}],
        }

    return human_approval_node
