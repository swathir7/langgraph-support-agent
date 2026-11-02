"""Routing-correctness evaluation: does the graph send each labeled scenario
down the route we expect, trigger approval when it should, and escalate when
it should? This is the agent-orchestration analogue of the RAG Quality
Workbench's retrieval evaluation harness — same idea (a labeled scenario set
run through the real production code path, not a parallel reimplementation),
applied to routing/escalation correctness instead of retrieval metrics.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from .config import Settings
from .graph.build import build_graph
from .graph.router import Classifier
from .tools.orders import MockOrderApiClient


@dataclass(frozen=True)
class EvalScenario:
    message: str
    expected_route: str
    expect_approval: bool
    expect_escalation: bool


@dataclass
class EvalResult:
    scenario: EvalScenario
    actual_route: str | None
    actual_approval: bool
    actual_escalation: bool

    @property
    def passed(self) -> bool:
        return (
            self.actual_route == self.scenario.expected_route
            and self.actual_approval == self.scenario.expect_approval
            and self.actual_escalation == self.scenario.expect_escalation
        )

    def as_dict(self) -> dict:
        return {
            "message": self.scenario.message,
            "expected_route": self.scenario.expected_route,
            "actual_route": self.actual_route,
            "expect_approval": self.scenario.expect_approval,
            "actual_approval": self.actual_approval,
            "expect_escalation": self.scenario.expect_escalation,
            "actual_escalation": self.actual_escalation,
            "passed": self.passed,
        }


def load_scenarios(path: str | Path) -> list[EvalScenario]:
    scenarios = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON — {error}") from error
            scenarios.append(
                EvalScenario(
                    message=record["message"],
                    expected_route=record["expected_route"],
                    expect_approval=bool(record["expect_approval"]),
                    expect_escalation=bool(record["expect_escalation"]),
                )
            )
    if not scenarios:
        raise ValueError(f"{path} contains no scenarios.")
    return scenarios


async def run_eval(
    scenarios: list[EvalScenario],
    *,
    settings: Settings | None = None,
    classifier: Classifier | None = None,
    graph: CompiledStateGraph | None = None,
) -> list[EvalResult]:
    """Builds one fresh graph (fresh MemorySaver, fresh mock order data) unless
    one is passed in, then runs every scenario on its own thread so results
    can't leak state between them."""

    if graph is None:
        settings = settings or Settings()
        graph = build_graph(
            settings,
            order_client=MockOrderApiClient(),
            classifier=classifier,
            checkpointer=MemorySaver(),
        )

    results = []
    for scenario in scenarios:
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:12]}"}}
        outcome = await graph.ainvoke({"messages": [{"role": "user", "content": scenario.message}]}, config)
        results.append(
            EvalResult(
                scenario=scenario,
                actual_route=outcome.get("route"),
                actual_approval=bool(outcome.get("__interrupt__")),
                actual_escalation=bool(outcome.get("escalated", False)),
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "results": [r.as_dict() for r in results],
    }
