"""Assembles the graph: router → (knowledge_agent | order_agent | escalation),
with order_agent conditionally routing into human_approval before any
mutating tool call. See docs/architecture.md for the diagram — or call
`GET /graph/diagram` on a running instance, which renders this exact
compiled graph via LangGraph's own `draw_mermaid()`, so the diagram can never
drift from the real topology.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import Settings
from ..resilience import CircuitBreaker
from ..state import AgentState
from ..tools.orders import MockOrderApiClient
from .escalation import escalation_node
from .human_approval import make_human_approval_node
from .knowledge_agent import knowledge_agent_node
from .order_agent import make_order_agent_node
from .router import Classifier, LLMClassifier, RuleBasedClassifier, make_router_node


def default_classifier(settings: Settings) -> Classifier:
    if settings.openai_api_key:
        return LLMClassifier(settings.openai_api_key, model=settings.classifier_model)
    return RuleBasedClassifier()


def build_graph(
    settings: Settings,
    *,
    order_client: MockOrderApiClient | None = None,
    classifier: Classifier | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    order_client = order_client or MockOrderApiClient()
    classifier = classifier or default_classifier(settings)
    checkpointer = checkpointer or MemorySaver()
    circuit_breaker = CircuitBreaker(
        failure_threshold=settings.order_circuit_failure_threshold,
        reset_timeout_seconds=settings.order_circuit_reset_timeout_seconds,
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(classifier))
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("order_agent", make_order_agent_node(order_client, settings, circuit_breaker))
    graph.add_node("human_approval", make_human_approval_node(order_client, settings, circuit_breaker))
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("router")

    def route_after_router(state: AgentState) -> str:
        confidence = state.get("confidence") or 0.0
        if confidence < settings.escalation_confidence_threshold:
            return "escalation"
        route = state.get("route")
        if route == "knowledge":
            return "knowledge_agent"
        if route == "order":
            return "order_agent"
        return "escalation"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"knowledge_agent": "knowledge_agent", "order_agent": "order_agent", "escalation": "escalation"},
    )

    def route_after_order_agent(state: AgentState) -> str:
        return "human_approval" if state.get("pending_action") else "done"

    graph.add_conditional_edges(
        "order_agent", route_after_order_agent, {"human_approval": "human_approval", "done": END}
    )

    graph.add_edge("knowledge_agent", END)
    graph.add_edge("human_approval", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
