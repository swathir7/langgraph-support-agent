"""Structured outputs validated at every node boundary in the graph.

Every node returns one of these (or a plain dict of primitives that a Pydantic
model validates before the next node trusts it) rather than passing free-form
dicts around — a router that returns a route the graph doesn't recognise, or
a tool result missing a required field, fails loudly at the boundary instead
of propagating a KeyError three nodes later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["knowledge", "order", "escalate"]


class RouterDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class KnowledgeHit(BaseModel):
    question: str
    answer: str
    relevance: float = Field(ge=0.0, le=1.0)


class OrderLookupResult(BaseModel):
    order_id: str
    status: str
    item: str
    amount: float
    eta: str | None


class RefundResult(BaseModel):
    order_id: str
    refunded_amount: float
    status: Literal["refunded"]


class CancelResult(BaseModel):
    order_id: str
    status: Literal["cancelled"]


class OrderNotFoundError(Exception):
    def __init__(self, order_id: str) -> None:
        super().__init__(f"No order found with id '{order_id}'.")
        self.order_id = order_id


RISKY_ACTIONS = {"refund", "cancel"}
"""Order actions that route through the human-approval node before executing."""
