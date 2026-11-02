"""A small in-memory FAQ knowledge base and lookup function.

Deliberately not a vector-search RAG pipeline — that's the companion
`rag-quality-workbench` project's job. This one demonstrates agent
*orchestration and reliability*, so the retrieval step here is intentionally
simple (lexical overlap scoring) and swappable: `lookup_knowledge` is the one
function `graph/knowledge_agent.py` calls, so pointing it at a real vector
store later is a one-function change, not a graph rewrite.
"""

from __future__ import annotations

import re

from ..schemas import KnowledgeHit

_WORD_RE = re.compile(r"[a-z0-9]+")

FAQ: list[tuple[str, str]] = [
    ("What is your return policy?", "Items can be returned within 30 days of delivery for a full refund."),
    ("How long does shipping take?", "Standard shipping takes 5-7 business days; expedited takes 2-3."),
    ("Do you ship internationally?", "Yes, we ship to over 40 countries. International orders take 10-15 business days."),
    ("How do I track my order?", "Use the tracking link in your shipping confirmation email, or look up your order ID here."),
    ("What payment methods do you accept?", "We accept all major credit cards, PayPal, and Apple Pay."),
    ("Can I change my shipping address after ordering?", "Contact support within 1 hour of placing the order; after that, the warehouse may have already processed it."),
]


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def lookup_knowledge(query: str, *, relevance_threshold: float = 0.15) -> KnowledgeHit | None:
    query_terms = _tokenize(query)
    if not query_terms:
        return None

    best: KnowledgeHit | None = None
    for question, answer in FAQ:
        question_terms = _tokenize(question)
        overlap = query_terms & question_terms
        relevance = len(overlap) / len(query_terms | question_terms) if (query_terms | question_terms) else 0.0
        if best is None or relevance > best.relevance:
            best = KnowledgeHit(question=question, answer=answer, relevance=relevance)

    if best is None or best.relevance < relevance_threshold:
        return None
    return best
