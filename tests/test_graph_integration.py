import unittest
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from support_agent.config import Settings
from support_agent.graph.build import build_graph
from support_agent.graph.router import RuleBasedClassifier
from support_agent.tools.orders import MockOrderApiClient


def _thread_config() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


class GraphIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _build(self, *, order_client: MockOrderApiClient | None = None, settings: Settings | None = None):
        return build_graph(
            settings or Settings(),
            order_client=order_client or MockOrderApiClient(),
            classifier=RuleBasedClassifier(),
            checkpointer=MemorySaver(),
        )

    async def test_knowledge_question_answers_directly(self) -> None:
        graph = self._build()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is your return policy?"}]}, _thread_config()
        )
        self.assertEqual(result["route"], "knowledge")
        self.assertIn("30 days", result["reply"])
        self.assertFalse(result["escalated"])
        self.assertNotIn("__interrupt__", result)

    async def test_safe_order_lookup_answers_directly(self) -> None:
        graph = self._build()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is the status of ORD-1001?"}]}, _thread_config()
        )
        self.assertEqual(result["route"], "order")
        self.assertIn("ORD-1001", result["reply"])
        self.assertNotIn("__interrupt__", result)

    async def test_unknown_order_id_replies_without_escalating(self) -> None:
        graph = self._build()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is the status of ORD-9999?"}]}, _thread_config()
        )
        self.assertIn("couldn't find", result["reply"])
        self.assertFalse(result["escalated"])

    async def test_message_with_no_order_id_asks_for_one(self) -> None:
        graph = self._build()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "I want a refund"}]}, _thread_config()
        )
        self.assertIn("order id", result["reply"].lower())
        self.assertNotIn("__interrupt__", result)

    async def test_ambiguous_message_escalates(self) -> None:
        graph = self._build()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "asdkfj random gibberish query"}]}, _thread_config()
        )
        self.assertTrue(result["escalated"])
        self.assertEqual(result["route"], "escalate")

    async def test_refund_pauses_for_approval_then_executes_on_approve(self) -> None:
        graph = self._build()
        config = _thread_config()
        first = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "I want a refund for ORD-1001"}]}, config
        )
        self.assertIn("__interrupt__", first)
        approval = first["__interrupt__"][0].value
        self.assertEqual(approval["action"], "refund")
        self.assertEqual(approval["order_id"], "ORD-1001")

        second = await graph.ainvoke(Command(resume={"approved": True}), config)
        self.assertIn("refunded", second["reply"].lower())
        self.assertNotIn("__interrupt__", second)
        # Trace accumulates across the pause/resume boundary via the checkpointer.
        self.assertEqual([s["node"] for s in second["trace"]], ["router", "order_agent", "human_approval"])

    async def test_refund_denied_takes_no_action(self) -> None:
        order_client = MockOrderApiClient()
        graph = self._build(order_client=order_client)
        config = _thread_config()
        await graph.ainvoke({"messages": [{"role": "user", "content": "I want a refund for ORD-1002"}]}, config)
        result = await graph.ainvoke(Command(resume={"approved": False}), config)

        self.assertIn("won't", result["reply"])
        lookup = await order_client.lookup_order("ORD-1002")
        self.assertEqual(lookup.status, "processing")  # untouched

    async def test_cancel_also_requires_approval(self) -> None:
        graph = self._build()
        config = _thread_config()
        first = await graph.ainvoke({"messages": [{"role": "user", "content": "cancel ORD-1003"}]}, config)
        approval = first["__interrupt__"][0].value
        self.assertEqual(approval["action"], "cancel")

        second = await graph.ainvoke(Command(resume={"approved": True}), config)
        self.assertIn("cancelled", second["reply"])

    async def test_repeated_order_api_failures_exhaust_retries_and_escalate(self) -> None:
        order_client = MockOrderApiClient()
        order_client.fail_next_calls(10)  # more than enough to exhaust retries
        settings = Settings(order_retry_max_attempts=2, order_retry_base_delay_seconds=0.001, order_circuit_failure_threshold=1)
        graph = self._build(order_client=order_client, settings=settings)
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is the status of ORD-1001?"}]}, _thread_config()
        )
        self.assertTrue(result["escalated"])
        self.assertIn("couldn't reach", result["reply"])

    async def test_circuit_breaker_open_short_circuits_a_second_request(self) -> None:
        order_client = MockOrderApiClient()
        order_client.fail_next_calls(10)
        settings = Settings(order_retry_max_attempts=2, order_retry_base_delay_seconds=0.001, order_circuit_failure_threshold=1)
        graph = self._build(order_client=order_client, settings=settings)

        # First request exhausts retries (RetryExhaustedError) and, as a side effect,
        # trips the breaker open (failure_threshold=1).
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is the status of ORD-1001?"}]}, _thread_config()
        )
        # A second, independent request now hits the already-open breaker directly —
        # a different failure mode with its own message, worth distinguishing.
        second = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "What is the status of ORD-1002?"}]}, _thread_config()
        )
        self.assertTrue(second["escalated"])
        self.assertIn("unavailable", second["reply"])

    async def test_conversation_history_persists_across_turns_on_the_same_thread(self) -> None:
        graph = self._build()
        config = _thread_config()
        await graph.ainvoke({"messages": [{"role": "user", "content": "What is your return policy?"}]}, config)
        second = await graph.ainvoke({"messages": [{"role": "user", "content": "Do you ship internationally?"}]}, config)
        roles_and_content = [(m["role"], m["content"][:10]) for m in second["messages"]]
        self.assertEqual(len(roles_and_content), 4)  # user, assistant, user, assistant


if __name__ == "__main__":
    unittest.main()
