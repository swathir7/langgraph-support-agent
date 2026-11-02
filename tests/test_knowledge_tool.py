import unittest

from support_agent.tools.knowledge import lookup_knowledge


class LookupKnowledgeTests(unittest.TestCase):
    def test_exact_faq_question_matches_with_high_relevance(self) -> None:
        hit = lookup_knowledge("What is your return policy?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.question, "What is your return policy?")
        self.assertGreater(hit.relevance, 0.9)

    def test_paraphrased_question_still_matches(self) -> None:
        hit = lookup_knowledge("Do you accept PayPal?")
        self.assertIsNotNone(hit)
        self.assertIn("payment", hit.question.lower())

    def test_unrelated_query_returns_none(self) -> None:
        hit = lookup_knowledge("asdkfj random gibberish query")
        self.assertIsNone(hit)

    def test_empty_query_returns_none(self) -> None:
        self.assertIsNone(lookup_knowledge(""))

    def test_relevance_threshold_is_configurable(self) -> None:
        query = "Do you accept PayPal?"  # partial overlap with the payment-methods FAQ, not a full match
        self.assertIsNotNone(lookup_knowledge(query, relevance_threshold=0.05))
        self.assertIsNone(lookup_knowledge(query, relevance_threshold=0.9))


if __name__ == "__main__":
    unittest.main()
