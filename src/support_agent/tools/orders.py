"""A mock order/CRM backend, standing in for whatever real order-management
API a production deployment would call. `MockOrderApiClient` supports
injected transient failures (`fail_next_n_calls`) specifically so tests can
exercise the retry/circuit-breaker wiring in `graph/order_agent.py` against
real failure/recovery sequences, not just mocked-out success paths.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..schemas import CancelResult, OrderLookupResult, OrderNotFoundError, RefundResult


@dataclass
class _OrderRecord:
    order_id: str
    status: str
    item: str
    total_amount: float
    eta: str | None = None


def _seed_orders() -> dict[str, _OrderRecord]:
    return {
        "ORD-1001": _OrderRecord("ORD-1001", "shipped", "Wireless Headphones", 79.99, eta="2026-08-15"),
        "ORD-1002": _OrderRecord("ORD-1002", "processing", "Standing Desk", 349.00, eta="2026-08-20"),
        "ORD-1003": _OrderRecord("ORD-1003", "delivered", "USB-C Cable (3-pack)", 14.50, eta=None),
    }


class MockOrderApiClient:
    def __init__(self, *, latency_seconds: float = 0.0) -> None:
        self._orders: dict[str, _OrderRecord] = _seed_orders()
        self._latency_seconds = latency_seconds
        self._fail_next_n_calls = 0

    def fail_next_calls(self, count: int) -> None:
        """Test hook: the next `count` calls raise, then calls succeed again —
        simulates a transient outage the retry policy should ride out."""
        self._fail_next_n_calls = count

    async def _simulate_network(self) -> None:
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)
        if self._fail_next_n_calls > 0:
            self._fail_next_n_calls -= 1
            raise ConnectionError("Simulated transient order-API failure.")

    async def lookup_order(self, order_id: str) -> OrderLookupResult:
        await self._simulate_network()
        record = self._orders.get(order_id)
        if record is None:
            raise OrderNotFoundError(order_id)
        return OrderLookupResult(
            order_id=record.order_id, status=record.status, item=record.item, amount=record.total_amount, eta=record.eta
        )

    async def refund_order(self, order_id: str, amount: float) -> RefundResult:
        await self._simulate_network()
        record = self._orders.get(order_id)
        if record is None:
            raise OrderNotFoundError(order_id)
        record.status = "refunded"
        return RefundResult(order_id=order_id, refunded_amount=amount, status="refunded")

    async def cancel_order(self, order_id: str) -> CancelResult:
        await self._simulate_network()
        record = self._orders.get(order_id)
        if record is None:
            raise OrderNotFoundError(order_id)
        record.status = "cancelled"
        return CancelResult(order_id=order_id, status="cancelled")
