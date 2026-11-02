import json
import unittest

import httpx

from support_agent.graph.router import ClassifierError, LLMClassifier, RuleBasedClassifier


class RuleBasedClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.classifier = RuleBasedClassifier()

    async def test_order_id_always_wins(self) -> None:
        decision = await self.classifier.classify("What is the status of ORD-1001?")
        self.assertEqual(decision.route, "order")
        self.assertGreater(decision.confidence, 0.9)

    async def test_faq_question_routes_to_knowledge_even_with_weak_order_keywords(self) -> None:
        # Regression test: "track"/"order" both appear here, but this is a how-to
        # FAQ question, not a request to look up a specific order.
        decision = await self.classifier.classify("How do I track my order?")
        self.assertEqual(decision.route, "knowledge")

    async def test_explicit_refund_request_routes_to_order_without_an_id(self) -> None:
        decision = await self.classifier.classify("I want a refund")
        self.assertEqual(decision.route, "order")
        self.assertGreaterEqual(decision.confidence, 0.8)

    async def test_explicit_cancel_request_routes_to_order(self) -> None:
        decision = await self.classifier.classify("Please cancel my order")
        self.assertEqual(decision.route, "order")

    async def test_general_policy_question_routes_to_knowledge(self) -> None:
        decision = await self.classifier.classify("What is your return policy?")
        self.assertEqual(decision.route, "knowledge")
        self.assertGreater(decision.confidence, 0.9)

    async def test_weak_order_keyword_with_no_faq_match_falls_through_to_order(self) -> None:
        decision = await self.classifier.classify("shipment status package")
        self.assertEqual(decision.route, "order")

    async def test_gibberish_escalates_with_low_confidence(self) -> None:
        decision = await self.classifier.classify("asdkfj random gibberish query")
        self.assertEqual(decision.route, "escalate")
        self.assertLess(decision.confidence, 0.4)


class LLMClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_a_valid_json_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            content = json.dumps({"route": "order", "confidence": 0.9, "reasoning": "mentions a refund"})
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
        classifier = LLMClassifier("key", client=client)
        decision = await classifier.classify("I'd like a refund")
        self.assertEqual(decision.route, "order")
        self.assertEqual(decision.confidence, 0.9)

    async def test_non_200_response_raises_classifier_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
        classifier = LLMClassifier("bad-key", client=client)
        with self.assertRaises(ClassifierError):
            await classifier.classify("anything")

    async def test_malformed_json_content_raises_classifier_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
        classifier = LLMClassifier("key", client=client)
        with self.assertRaises(ClassifierError):
            await classifier.classify("anything")


if __name__ == "__main__":
    unittest.main()
