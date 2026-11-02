import unittest

from support_agent.schemas import OrderNotFoundError
from support_agent.tools.orders import MockOrderApiClient


class MockOrderApiClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_known_order(self) -> None:
        client = MockOrderApiClient()
        result = await client.lookup_order("ORD-1001")
        self.assertEqual(result.order_id, "ORD-1001")
        self.assertEqual(result.item, "Wireless Headphones")
        self.assertGreater(result.amount, 0)

    async def test_lookup_unknown_order_raises(self) -> None:
        client = MockOrderApiClient()
        with self.assertRaises(OrderNotFoundError):
            await client.lookup_order("ORD-9999")

    async def test_refund_marks_order_refunded(self) -> None:
        client = MockOrderApiClient()
        result = await client.refund_order("ORD-1001", 79.99)
        self.assertEqual(result.status, "refunded")
        follow_up = await client.lookup_order("ORD-1001")
        self.assertEqual(follow_up.status, "refunded")

    async def test_cancel_marks_order_cancelled(self) -> None:
        client = MockOrderApiClient()
        result = await client.cancel_order("ORD-1002")
        self.assertEqual(result.status, "cancelled")
        follow_up = await client.lookup_order("ORD-1002")
        self.assertEqual(follow_up.status, "cancelled")

    async def test_refund_unknown_order_raises(self) -> None:
        client = MockOrderApiClient()
        with self.assertRaises(OrderNotFoundError):
            await client.refund_order("ORD-9999", 10.0)

    async def test_fail_next_calls_injects_transient_failures_then_recovers(self) -> None:
        client = MockOrderApiClient()
        client.fail_next_calls(2)
        with self.assertRaises(ConnectionError):
            await client.lookup_order("ORD-1001")
        with self.assertRaises(ConnectionError):
            await client.lookup_order("ORD-1001")
        result = await client.lookup_order("ORD-1001")  # third call succeeds
        self.assertEqual(result.order_id, "ORD-1001")

    async def test_each_client_instance_has_independent_state(self) -> None:
        client_a = MockOrderApiClient()
        client_b = MockOrderApiClient()
        await client_a.refund_order("ORD-1001", 79.99)
        result = await client_b.lookup_order("ORD-1001")
        self.assertEqual(result.status, "shipped")  # untouched in the second client


if __name__ == "__main__":
    unittest.main()
