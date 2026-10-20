from __future__ import annotations

from typing import Any, TypedDict

from .core import DocumentRecord, ExtractiveGroundedGenerator, FuzzyEvidence, FuzzyIndex


class RAGState(TypedDict, total=False):
    question: str
    k: int
    expanded: bool
    evidence: list[dict[str, Any]]
    confidence: float
    margin: float
    reason: str
    route: str
    answer: str
    citations: list[str]


def _serialize_hits(hits: list[FuzzyEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "id": h.document.id,
            "text": h.document.text,
            "title": h.document.title,
            "source": h.document.source,
            "metadata": dict(h.document.metadata),
            "raw_relevance": h.raw_relevance,
            "fuzzy_score": h.fuzzy_score,
            "authority": h.authority,
            "freshness": h.freshness,
            "reliability": h.reliability,
        }
        for h in hits
    ]


def _deserialize_hits(items: list[dict[str, Any]]) -> list[FuzzyEvidence]:
    hits: list[FuzzyEvidence] = []
    for x in items:
        d = DocumentRecord(x["id"], x["text"], x.get("title", ""), x.get("source", ""), x.get("metadata", {}))
        hits.append(FuzzyEvidence(d, float(x["raw_relevance"]), float(x["fuzzy_score"]), float(x["authority"]), float(x["freshness"]), float(x["reliability"]), {}))
    return hits


class FuzzyRAGGraph:
    """LangGraph orchestration with fuzzy generate/expand/abstain routing."""

    def __init__(self, index: FuzzyIndex, initial_k: int = 4, generator=None):
        self.index = index
        self.initial_k = initial_k
        self.generator = ExtractiveGroundedGenerator()
        self.external_generator = generator
        self.graph = self._build()

    def _build(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("langgraph is required to build FuzzyRAGGraph") from exc

        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess", self._assess)
        builder.add_node("expand", self._expand)
        builder.add_node("generate", self._generate)
        builder.add_node("abstain", self._abstain)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess",
            lambda s: s["route"],
            {"generate": "generate", "expand": "expand", "abstain": "abstain"},
        )
        builder.add_edge("expand", "retrieve")
        builder.add_edge("generate", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        k = int(state.get("k", self.initial_k))
        pool = self.index.config.extended_pool if state.get("expanded", False) else self.index.config.candidate_pool
        hits = self.index.search(state["question"], k=k, candidate_pool=pool)
        return {"evidence": _serialize_hits(hits)}

    def _assess(self, state: RAGState) -> dict[str, Any]:
        hits = _deserialize_hits(state.get("evidence", []))
        d = self.index.decision(hits)
        route = d["route"]
        # Only one retrieval expansion is permitted. If ambiguity survives the
        # wider search, the graph abstains rather than cycling or bluffing.
        if route == "expand" and state.get("expanded", False):
            route = "abstain"
            reason = "ambiguous_after_expansion"
        else:
            reason = d["reason"]
        return {"route": route, "confidence": float(d["confidence"]), "margin": float(d["margin"]), "reason": reason}

    def _expand(self, state: RAGState) -> dict[str, Any]:
        return {"k": min(self.index.config.extended_pool, max(int(state.get("k", self.initial_k)) * 2, self.initial_k + 2)), "expanded": True}

    def _generate(self, state: RAGState) -> dict[str, Any]:
        hits = _deserialize_hits(state.get("evidence", []))
        if self.external_generator is None:
            answer, citations = self.generator.generate(state["question"], hits)
        else:
            answer = str(self.external_generator(state["question"], hits))
            citations = [hits[0].document.id] if hits else []
        return {"answer": answer, "citations": citations}

    def _abstain(self, state: RAGState) -> dict[str, Any]:
        return {"answer": self.generator.abstain_text, "citations": []}

    def invoke(self, question: str) -> dict[str, Any]:
        out = self.graph.invoke({"question": question, "k": self.initial_k, "expanded": False})
        return out
