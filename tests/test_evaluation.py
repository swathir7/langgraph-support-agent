import tempfile
import unittest
from pathlib import Path

from support_agent.config import Settings
from support_agent.evaluation import EvalScenario, load_scenarios, run_eval, summarize
from support_agent.graph.router import RuleBasedClassifier


class LoadScenariosTests(unittest.TestCase):
    def test_loads_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text(
                '{"message": "hi", "expected_route": "knowledge", "expect_approval": false, "expect_escalation": false}\n'
            )
            scenarios = load_scenarios(path)
            self.assertEqual(len(scenarios), 1)
            self.assertEqual(scenarios[0].message, "hi")

    def test_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text(
                '{"message": "hi", "expected_route": "knowledge", "expect_approval": false, "expect_escalation": false}\n\n\n'
            )
            self.assertEqual(len(load_scenarios(path)), 1)

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text("")
            with self.assertRaises(ValueError):
                load_scenarios(path)

    def test_invalid_json_raises_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text("{not valid json}\n")
            with self.assertRaises(ValueError) as context:
                load_scenarios(path)
            self.assertIn(":1:", str(context.exception))


class RunEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_matching_scenario_passes(self) -> None:
        scenarios = [
            EvalScenario(
                message="What is your return policy?",
                expected_route="knowledge",
                expect_approval=False,
                expect_escalation=False,
            )
        ]
        results = await run_eval(scenarios, settings=Settings(), classifier=RuleBasedClassifier())
        self.assertTrue(results[0].passed)

    async def test_mismatched_scenario_fails(self) -> None:
        scenarios = [
            EvalScenario(
                message="What is your return policy?",
                expected_route="order",  # deliberately wrong
                expect_approval=False,
                expect_escalation=False,
            )
        ]
        results = await run_eval(scenarios, settings=Settings(), classifier=RuleBasedClassifier())
        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].actual_route, "knowledge")

    async def test_refund_scenario_detects_approval_correctly(self) -> None:
        scenarios = [
            EvalScenario(
                message="I want a refund for ORD-1001",
                expected_route="order",
                expect_approval=True,
                expect_escalation=False,
            )
        ]
        results = await run_eval(scenarios, settings=Settings(), classifier=RuleBasedClassifier())
        self.assertTrue(results[0].passed)
        self.assertTrue(results[0].actual_approval)

    async def test_each_scenario_runs_on_an_isolated_thread(self) -> None:
        # Two scenarios referencing the same order shouldn't see each other's side effects,
        # since each gets its own thread_id — this just confirms no crash / cross-talk.
        scenarios = [
            EvalScenario("I want a refund for ORD-1001", "order", True, False),
            EvalScenario("What is the status of ORD-1001?", "order", False, False),
        ]
        results = await run_eval(scenarios, settings=Settings(), classifier=RuleBasedClassifier())
        self.assertTrue(all(r.passed for r in results))


class SummarizeTests(unittest.TestCase):
    def test_computes_pass_rate(self) -> None:
        from support_agent.evaluation import EvalResult

        results = [
            EvalResult(EvalScenario("a", "knowledge", False, False), "knowledge", False, False),
            EvalResult(EvalScenario("b", "order", False, False), "escalate", False, True),
        ]
        summary = summarize(results)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertAlmostEqual(summary["pass_rate"], 0.5)

    def test_empty_results_has_zero_pass_rate(self) -> None:
        self.assertEqual(summarize([])["pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
