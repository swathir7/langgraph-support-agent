"""The order-tool agent: looks up orders directly, but routes refunds and
cancellations through the human-approval node (graph/human_approval.py)
before ever calling the mutating tool. Every call to the mock order API goes
through retry + timeout + circuit breaker — this is the one node modeling a
real external dependency, so it's the one node that needs to survive one
being flaky.
"""

from __future__ import annotations

import re
import time

from ..config import Settings
from ..resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryExhaustedError,
    RetryPolicy,
    call_with_resilience,
)
from ..schemas import RISKY_ACTIONS, OrderNotFoundError
from ..state import AgentState
from ..tools.orders import MockOrderApiClient

_ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
RETRYABLE_ORDER_ERRORS = (ConnectionError, TimeoutError)
"""`TimeoutError` covers `asyncio.TimeoutError` (an alias since Python 3.11) raised
by `with_timeout` — a slow-but-recoverable call should be retried, not just failed."""


def _detect_action(message: str) -> str:
    lowered = message.lower()
    if "refund" in lowered:
        return "refund"
    if "cancel" in lowered:
        return "cancel"
    return "lookup"


def make_order_agent_node(client: MockOrderApiClient, settings: Settings, circuit_breaker: CircuitBreaker):
    retry_policy = RetryPolicy(
        max_attempts=settings.order_retry_max_attempts, base_delay_seconds=settings.order_retry_base_delay_seconds
    )

    async def _resilient_lookup(order_id: str):
        return await call_with_resilience(
            lambda: client.lookup_order(order_id),
            retry_policy=retry_policy,
            timeout_seconds=settings.order_call_timeout_seconds,
            circuit_breaker=circuit_breaker,
            retry_on=RETRYABLE_ORDER_ERRORS,
        )

    async def order_agent_node(state: AgentState) -> dict:
        started = time.monotonic()
        last_user_message = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")

        order_id_match = _ORDER_ID_PATTERN.search(last_user_message)
        if order_id_match is None:
            reply = "Which order is this about? Please share your order id (e.g. ORD-1001)."
            return _finish(started, reply, pending_action=None)

        order_id = order_id_match.group(0).upper()
        action = _detect_action(last_user_message)

        try:
            lookup = await _resilient_lookup(order_id)
        except OrderNotFoundError:
            reply = f"I couldn't find an order with id {order_id}. Could you double-check it?"
            return _finish(started, reply, pending_action=None)
        except CircuitOpenError:
            reply = "Our order system is temporarily unavailable after repeated failures. Please try again shortly."
            return _finish(started, reply, pending_action=None, escalated=True)
        except RetryExhaustedError:
            reply = "I couldn't reach our order system after several attempts. Please try again shortly."
            return _finish(started, reply, pending_action=None, escalated=True)

        if action in RISKY_ACTIONS:
            pending_action = {
                "action": action,
                "order_id": order_id,
                "amount": lookup.amount if action == "refund" else None,
                "item": lookup.item,
            }
            return _finish(
                started,
                reply=None,
                pending_action=pending_action,
                tool_result={"lookup": lookup.model_dump()},
            )

        reply = f"Order {lookup.order_id} ({lookup.item}) is currently '{lookup.status}'."
        if lookup.eta:
            reply += f" Estimated arrival: {lookup.eta}."
        return _finish(started, reply, pending_action=None, tool_result={"lookup": lookup.model_dump()})

    def _finish(
        started: float,
        reply: str | None,
        *,
        pending_action: dict | None,
        tool_result: dict | None = None,
        escalated: bool = False,
    ) -> dict:
        duration_ms = (time.monotonic() - started) * 1000
        result: dict = {
            "reply": reply,
            "pending_action": pending_action,
            "tool_result": tool_result,
            "escalated": escalated,
            "trace": [{"node": "order_agent", "duration_ms": duration_ms}],
        }
        if reply is not None:
            result["messages"] = [{"role": "assistant", "content": reply}]
        return result

    return order_agent_node
