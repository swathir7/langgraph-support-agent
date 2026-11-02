"""The retrieval agent: answers general policy/how-to questions from the FAQ
knowledge base. No external call, so no resilience wrapping needed here —
that's specifically the order agent's concern (graph/order_agent.py), since
the order API is the one thing here modeling a real external dependency.
"""

from __future__ import annotations

import time

from ..state import AgentState
from ..tools.knowledge import lookup_knowledge

FALLBACK_REPLY = (
    "I wasn't able to find a confident answer to that in our knowledge base. "
    "Could you rephrase, or would you like me to connect you with a human agent?"
)


async def knowledge_agent_node(state: AgentState) -> dict:
    started = time.monotonic()
    last_user_message = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    hit = lookup_knowledge(last_user_message)
    duration_ms = (time.monotonic() - started) * 1000

    reply = hit.answer if hit is not None else FALLBACK_REPLY
    return {
        "reply": reply,
        "escalated": False,
        "tool_result": {"matched_question": hit.question, "relevance": hit.relevance} if hit else None,
        "messages": [{"role": "assistant", "content": reply}],
        "trace": [{"node": "knowledge_agent", "duration_ms": duration_ms}],
    }
