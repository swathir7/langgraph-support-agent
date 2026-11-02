"""The supervisor: classifies an incoming message into knowledge / order /
escalate with a confidence score.

`RuleBasedClassifier` is the default — deterministic, offline, and what every
test in this repo runs against, so CI never needs an API key. `LLMClassifier`
is a real OpenAI-backed alternative behind the same `Classifier` protocol;
swapping is a one-line change in `graph/build.py` when `OPENAI_API_KEY` is set
(see `graph/build.py`'s `build_graph`).
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

import httpx

from ..schemas import RouterDecision
from ..state import AgentState
from ..tools.knowledge import lookup_knowledge

_ORDER_ID_PATTERN = re.compile(r"\bord-\d+\b", re.IGNORECASE)

# Unambiguous: wanting to *do* something to an order, not ask how something works.
_STRONG_ORDER_KEYWORDS = {"refund", "cancel"}

# Ambiguous on their own — "how do I track my order" and "where's my order" both
# hit these, but the first is a how-to FAQ question and the second wants a real
# lookup. Checked only after a knowledge-base match is ruled out, below.
_WEAK_ORDER_KEYWORDS = {"order", "track", "tracking", "shipment", "shipping", "delivery", "status", "package"}


class ClassifierError(Exception):
    pass


class Classifier(Protocol):
    async def classify(self, message: str) -> RouterDecision: ...


class RuleBasedClassifier:
    """Keyword/pattern matching for order intent, FAQ-overlap scoring (reusing
    `lookup_knowledge`) for knowledge intent. Anything matching neither well
    routes to escalation with low confidence, same as a genuinely ambiguous
    message would under the LLM classifier.

    Priority order matters: an order id or an unambiguous action word
    (refund/cancel) always means "order" — nothing to weigh there. Otherwise a
    knowledge-base match is checked *before* generic order-ish keywords like
    "track" or "status", because "how do I track my order" is a how-to FAQ
    question, not a request to look up a specific order. Only when there's no
    knowledge match do those weak keywords fall through to "order", where the
    order agent will ask for the missing order id itself."""

    async def classify(self, message: str) -> RouterDecision:
        lowered = message.lower()
        has_order_id = bool(_ORDER_ID_PATTERN.search(lowered))
        has_strong_keyword = any(keyword in lowered for keyword in _STRONG_ORDER_KEYWORDS)

        if has_order_id or has_strong_keyword:
            confidence = 0.95 if has_order_id else 0.85
            return RouterDecision(
                route="order",
                confidence=confidence,
                reasoning="Matched an order id." if has_order_id else "Matched an explicit refund/cancel request.",
            )

        knowledge_hit = lookup_knowledge(message)
        if knowledge_hit is not None:
            return RouterDecision(
                route="knowledge",
                confidence=knowledge_hit.relevance,
                reasoning=f"Closest FAQ match: '{knowledge_hit.question}' (relevance {knowledge_hit.relevance:.2f}).",
            )

        weak_keyword_hits = sum(1 for keyword in _WEAK_ORDER_KEYWORDS if keyword in lowered)
        if weak_keyword_hits >= 1:
            confidence = min(0.5 + 0.15 * weak_keyword_hits, 0.8)
            return RouterDecision(
                route="order",
                confidence=confidence,
                reasoning=f"Matched {weak_keyword_hits} order-related keyword(s) with no FAQ match.",
            )

        return RouterDecision(
            route="escalate", confidence=0.1, reasoning="No confident match for knowledge or order intent."
        )


class LLMClassifier:
    """OpenAI-backed alternative to `RuleBasedClassifier`. Not exercised by CI
    (no API key there) — covered by `tests/test_router.py`'s mocked-transport
    tests instead, same pattern as the RAG Quality Workbench's provider tests."""

    def __init__(self, api_key: str, *, model: str = "gpt-4.1-mini", client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=30)

    async def classify(self, message: str) -> RouterDecision:
        response = await self._client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify the user's support message. Respond with JSON matching "
                            '{"route": "knowledge"|"order"|"escalate", "confidence": 0-1, "reasoning": str}. '
                            "\"order\" is for anything about an existing order, refund, cancellation, or "
                            "shipping status. \"knowledge\" is a general policy/how-to question. \"escalate\" "
                            "is for anything you're not confident falls into either."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
            },
        )
        if response.status_code != 200:
            raise ClassifierError(f"OpenAI classification failed: {response.status_code} {response.text}")
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
            return RouterDecision.model_validate(json.loads(content))
        except (KeyError, IndexError, ValueError) as error:
            raise ClassifierError(f"Could not parse classifier response: {error}") from error


def make_router_node(classifier: Classifier):
    """Returns the actual LangGraph node function, closing over which classifier
    to use — this is the seam `graph/build.py` uses to pick `RuleBasedClassifier`
    vs `LLMClassifier` without the node function itself branching on config."""

    async def router_node(state: AgentState) -> dict:
        started = time.monotonic()
        last_user_message = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
        decision = await classifier.classify(last_user_message)
        duration_ms = (time.monotonic() - started) * 1000
        return {
            "route": decision.route,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "trace": [{"node": "router", "duration_ms": duration_ms}],
        }

    return router_node
