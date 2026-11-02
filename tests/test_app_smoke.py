"""Smoke-tests the FastAPI routes with the real app object, with the
per-request state dependency overridden to a MemorySaver-backed graph — so
these run without a real SQLite checkpoint file, same spirit as
test_graph_integration.py. Deliberately does NOT use `with TestClient(...)`
(which would run the real `lifespan` and open the real checkpoint file);
every route under test reads state exclusively through the override.
"""

import unittest

from langgraph.checkpoint.memory import MemorySaver
from starlette.testclient import TestClient

from support_agent.app import _app_state, app
from support_agent.config import Settings
from support_agent.graph.build import build_graph
from support_agent.graph.router import RuleBasedClassifier
from support_agent.tools.orders import MockOrderApiClient


class AppSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.graph = build_graph(
            self.settings,
            order_client=MockOrderApiClient(),
            classifier=RuleBasedClassifier(),
            checkpointer=MemorySaver(),
        )

        async def fake_state() -> dict:
            return {"settings": self.settings, "graph": self.graph}

        app.dependency_overrides[_app_state] = fake_state
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_graph_diagram_returns_mermaid(self) -> None:
        response = self.client.get("/graph/diagram")
        self.assertEqual(response.status_code, 200)
        self.assertIn("router", response.json()["mermaid"])

    def test_static_ui_served_at_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Reliable LangGraph Support Agent", response.content)

    def test_chat_knowledge_question(self) -> None:
        response = self.client.post("/chat", json={"thread_id": "t1", "message": "What is your return policy?"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "complete")
        self.assertIn("30 days", body["reply"])

    def test_chat_rejects_empty_message(self) -> None:
        response = self.client.post("/chat", json={"thread_id": "t1", "message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_chat_refund_then_resume_approve(self) -> None:
        first = self.client.post("/chat", json={"thread_id": "t2", "message": "I want a refund for ORD-1001"})
        self.assertEqual(first.json()["status"], "awaiting_approval")
        self.assertEqual(first.json()["approval"]["order_id"], "ORD-1001")

        second = self.client.post("/chat/resume", json={"thread_id": "t2", "approved": True})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["status"], "complete")
        self.assertIn("refunded", second.json()["reply"].lower())

    def test_resume_without_pending_approval_returns_409(self) -> None:
        self.client.post("/chat", json={"thread_id": "t3", "message": "What is your return policy?"})
        response = self.client.post("/chat/resume", json={"thread_id": "t3", "approved": True})
        self.assertEqual(response.status_code, 409)

    def test_resume_on_unknown_thread_returns_409(self) -> None:
        response = self.client.post("/chat/resume", json={"thread_id": "never-started", "approved": True})
        self.assertEqual(response.status_code, 409)

    def test_eval_run_endpoint(self) -> None:
        response = self.client.get("/eval/run")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("total", body)
        self.assertIn("results", body)
        self.assertGreater(body["total"], 0)


if __name__ == "__main__":
    unittest.main()
