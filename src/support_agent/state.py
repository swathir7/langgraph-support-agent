"""The graph's shared state schema.

`messages` and `trace` use `operator.add` reducers, meaning every node's
returned partial state is *appended* onto what's already there rather than
overwriting it — including across the pause/resume boundary of a human-
approval interrupt, since LangGraph persists state per `thread_id` via the
checkpointer between separate `.invoke()` calls. That's what makes the trace
in `/chat/resume`'s response show the full round trip, not just the second half.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Message(TypedDict):
    role: str
    content: str


class TraceEntry(TypedDict):
    node: str
    duration_ms: float


class PendingAction(TypedDict):
    action: str
    order_id: str
    amount: float | None
    item: str | None


class AgentState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
    route: str | None
    confidence: float | None
    reasoning: str | None
    pending_action: PendingAction | None
    tool_result: dict[str, Any] | None
    reply: str | None
    escalated: bool
