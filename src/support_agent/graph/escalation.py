"""Reached when the router's confidence falls below
`Settings.escalation_confidence_threshold`, regardless of which route it
guessed — an agent acting confidently on a low-confidence guess is worse than
admitting it isn't sure.
"""

from __future__ import annotations

import time

from ..state import AgentState

ESCALATION_REPLY = (
    "I'm not confident I can resolve this on my own — connecting you with a human agent who can help."
)


async def escalation_node(state: AgentState) -> dict:
    started = time.monotonic()
    duration_ms = (time.monotonic() - started) * 1000
    return {
        "reply": ESCALATION_REPLY,
        "escalated": True,
        "messages": [{"role": "assistant", "content": ESCALATION_REPLY}],
        "trace": [{"node": "escalation", "duration_ms": duration_ms}],
    }
