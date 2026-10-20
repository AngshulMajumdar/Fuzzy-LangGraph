from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FuzzyRetrievalConfig:
    candidate_pool: int = 10
    answer_threshold: float = 0.60
    ambiguity_margin: float = 0.015
    min_raw_relevance: float = 0.14
    expand_threshold: float = 0.48
    extended_pool: int = 16


@dataclass(frozen=True)
class FuzzyEvidence:
    document: DocumentRecord
    raw_relevance: float
    fuzzy_score: float
    authority: float
    freshness: float
    reliability: float
    memberships: dict[str, float]


class FuzzyEvidenceController:
    """Small dependency-free Sugeno fuzzy controller for RAG evidence quality.

    Inputs are normalized to [0, 1]: lexical/semantic relevance, source authority,
    freshness, and source reliability. The rules intentionally make relevance a
    gate: highly trusted but irrelevant evidence must not become high confidence.
    """

    @staticmethod
    def _clip(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    @classmethod
    def _left(cls, x: float, full_until: float, zero_at: float) -> float:
        x = cls._clip(x)
        if x <= full_until:
            return 1.0
        if x >= zero_at:
            return 0.0
        return (zero_at - x) / (zero_at - full_until)

    @classmethod
    def _right(cls, x: float, zero_until: float, full_at: float) -> float:
        x = cls._clip(x)
        if x <= zero_until:
            return 0.0
        if x >= full_at:
            return 1.0
        return (x - zero_until) / (full_at - zero_until)

    @classmethod
    def _triangle(cls, x: float, left: float, peak: float, right: float) -> float:
        x = cls._clip(x)
        if x <= left or x >= right:
            return 0.0
        if x == peak:
            return 1.0
        if x < peak:
            return (x - left) / (peak - left)
        return (right - x) / (right - peak)

    def memberships(self, relevance: float, authority: float, freshness: float, reliability: float) -> dict[str, float]:
        r = self._clip(relevance)
        a = self._clip(authority)
        f = self._clip(freshness)
        q = self._clip(reliability)
        return {
            "rel_low": self._left(r, 0.05, 0.30),
            "rel_med": self._triangle(r, 0.12, 0.42, 0.70),
            "rel_high": self._right(r, 0.42, 0.72),
            "auth_low": self._left(a, 0.0, 0.55),
            "auth_med": self._triangle(a, 0.35, 0.62, 0.84),
            "auth_high": self._right(a, 0.68, 0.90),
            "fresh_low": self._left(f, 0.0, 0.42),
            "fresh_med": self._triangle(f, 0.28, 0.58, 0.82),
            "fresh_high": self._right(f, 0.68, 0.90),
            "reliability_low": self._left(q, 0.0, 0.50),
            "reliability_high": self._right(q, 0.58, 0.88),
        }

    def score(self, relevance: float, authority: float, freshness: float, reliability: float) -> tuple[float, dict[str, float]]:
        m = self.memberships(relevance, authority, freshness, reliability)
        # First-order Sugeno rules. Consequents are confidence-like evidence values.
        rules: list[tuple[float, float]] = [
            (min(m["rel_high"], m["auth_high"], m["fresh_high"], m["reliability_high"]), 0.98),
            (min(m["rel_med"], m["auth_high"], m["fresh_high"], m["reliability_high"]), 0.90),
            (min(m["rel_high"], m["auth_high"]), 0.92),
            (min(m["rel_med"], m["auth_high"]), 0.84),
            (min(m["rel_high"], m["auth_med"], m["fresh_high"]), 0.74),
            (min(m["rel_med"], m["auth_med"]), 0.54),
            (min(m["rel_high"], m["fresh_low"]), 0.28),
            (min(m["rel_high"], m["auth_low"]), 0.24),
            (min(m["rel_med"], m["auth_low"]), 0.20),
            (min(m["rel_high"], m["reliability_low"]), 0.26),
            (m["rel_low"], 0.06),
        ]
        denom = sum(w for w, _ in rules)
        if denom <= 1e-12:
            # No fuzzy rule fired strongly; relevance still gates the fallback.
            base = 0.10 + 0.55 * self._clip(relevance)
        else:
            base = sum(w * z for w, z in rules) / denom
        # Relevance is already a fuzzy antecedent and is also enforced by the
        # hard minimum-relevance guard in the final decision.
        value = self._clip(base)
        return value, m


class FuzzyIndex:
    def __init__(self, documents: Sequence[DocumentRecord] | None = None, config: FuzzyRetrievalConfig | None = None):
        self.config = config or FuzzyRetrievalConfig()
        self.controller = FuzzyEvidenceController()
        self.documents: list[DocumentRecord] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if documents:
            self.fit(documents)

    @staticmethod
    def _trust(doc: DocumentRecord) -> tuple[float, float, float]:
        md = doc.metadata or {}
        authority = float(md.get("authority", 0.50))
        freshness = float(md.get("freshness", 0.50))
        if "reliability" in md:
            reliability = float(md["reliability"])
        elif "reliability_alpha" in md or "reliability_beta" in md:
            a = float(md.get("reliability_alpha", 1.0))
            b = float(md.get("reliability_beta", 1.0))
            reliability = a / max(a + b, 1e-12)
        else:
            reliability = 0.50
        return tuple(max(0.0, min(1.0, x)) for x in (authority, freshness, reliability))

    def fit(self, documents: Sequence[DocumentRecord]) -> "FuzzyIndex":
        self.documents = list(documents)
        corpus = [f"{d.title}. {d.text}".strip() for d in self.documents]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None
        return self

    def add(self, documents: Iterable[DocumentRecord]) -> "FuzzyIndex":
        return self.fit([*self.documents, *list(documents)])

    def search(self, query: str, k: int = 4, candidate_pool: int | None = None) -> list[FuzzyEvidence]:
        if not self.documents or self._vectorizer is None or self._matrix is None:
            return []
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        pool = min(candidate_pool or self.config.candidate_pool, len(self.documents))
        candidate_ids = np.argsort(-sims, kind="stable")[:pool]
        hits: list[FuzzyEvidence] = []
        for i in candidate_ids:
            doc = self.documents[int(i)]
            relevance = float(sims[int(i)])
            authority, freshness, reliability = self._trust(doc)
            fuzzy, memberships = self.controller.score(relevance, authority, freshness, reliability)
            hits.append(FuzzyEvidence(doc, relevance, fuzzy, authority, freshness, reliability, memberships))
        hits.sort(key=lambda h: (h.fuzzy_score, h.raw_relevance, h.authority, h.freshness), reverse=True)
        return hits[: max(1, k)]

    def decision(self, hits: Sequence[FuzzyEvidence]) -> dict[str, Any]:
        if not hits:
            return {"route": "abstain", "confidence": 0.0, "margin": 0.0, "reason": "no_evidence"}
        top = hits[0]
        second = hits[1] if len(hits) > 1 else None
        if second is None:
            margin = top.fuzzy_score
        else:
            # Fuzzy confidence can intentionally plateau for equally trustworthy
            # sources. A small relevance component breaks benign ties while exact
            # equal-authority/equal-relevance contradictions remain ambiguous.
            margin = 0.80 * (top.fuzzy_score - second.fuzzy_score) + 0.20 * (top.raw_relevance - second.raw_relevance)
        if top.raw_relevance < self.config.min_raw_relevance:
            return {"route": "abstain", "confidence": top.fuzzy_score, "margin": margin, "reason": "low_relevance"}
        trusted_enough = top.authority >= 0.60 or top.reliability >= 0.75
        if top.fuzzy_score >= self.config.answer_threshold and margin >= self.config.ambiguity_margin and trusted_enough:
            return {"route": "generate", "confidence": top.fuzzy_score, "margin": margin, "reason": "strong_evidence"}
        if top.fuzzy_score >= self.config.expand_threshold:
            return {"route": "expand", "confidence": top.fuzzy_score, "margin": margin, "reason": "borderline_or_ambiguous"}
        return {"route": "abstain", "confidence": top.fuzzy_score, "margin": margin, "reason": "weak_evidence"}

    def should_abstain(self, hits: Sequence[FuzzyEvidence]) -> bool:
        return self.decision(hits)["route"] != "generate"


class PlainTfidfIndex:
    """Baseline retriever intentionally unaware of source quality or uncertainty."""

    def __init__(self, documents: Sequence[DocumentRecord]):
        self.documents = list(documents)
        corpus = [f"{d.title}. {d.text}".strip() for d in self.documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 4) -> list[FuzzyEvidence]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        ids = np.argsort(-sims, kind="stable")[: min(k, len(self.documents))]
        out = []
        for i in ids:
            d = self.documents[int(i)]
            s = float(sims[int(i)])
            out.append(FuzzyEvidence(d, s, s, 0.5, 0.5, 0.5, {}))
        return out


class ExtractiveGroundedGenerator:
    """Offline answer synthesizer used for reproducible RAG evaluation.

    It selects the query-most-relevant sentence from the accepted evidence and
    returns it with an explicit citation. Production users can replace this with
    any LangChain Runnable/LLM without changing fuzzy retrieval or routing.
    """

    abstain_text = "Insufficient reliable evidence to answer."

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def generate(self, question: str, hits: Sequence[FuzzyEvidence]) -> tuple[str, list[str]]:
        if not hits:
            return self.abstain_text, []
        top = hits[0]
        sentences = self._sentences(top.document.text) or [top.document.text]
        if len(sentences) == 1:
            best = sentences[0]
        else:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            X = vec.fit_transform([question, *sentences])
            scores = cosine_similarity(X[0:1], X[1:])[0]
            best = sentences[int(np.argmax(scores))]
        return f"{best} [{top.document.id}]", [top.document.id]


class FuzzyRAGEngine:
    def __init__(self, documents: Sequence[DocumentRecord] | None = None, config: FuzzyRetrievalConfig | None = None):
        self.index = FuzzyIndex(documents or [], config)
        self.generator = ExtractiveGroundedGenerator()

    def ingest(self, documents: Sequence[DocumentRecord], replace: bool = True) -> int:
        if replace:
            self.index.fit(documents)
        else:
            self.index.add(documents)
        return len(self.index.documents)

    def query(self, question: str, k: int = 4) -> dict[str, Any]:
        hits = self.index.search(question, k=k)
        decision = self.index.decision(hits)
        expanded = False
        if decision["route"] == "expand":
            expanded = True
            hits = self.index.search(question, k=max(k * 2, 6), candidate_pool=self.index.config.extended_pool)
            decision = self.index.decision(hits)
            if decision["route"] == "expand":
                decision = {**decision, "route": "abstain", "reason": "ambiguous_after_expansion"}
        if decision["route"] == "generate":
            answer, citations = self.generator.generate(question, hits)
        else:
            answer, citations = self.generator.abstain_text, []
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence": round(float(decision["confidence"]), 6),
            "margin": round(float(decision["margin"]), 6),
            "decision": decision["route"],
            "reason": decision["reason"],
            "expanded": expanded,
            "evidence": [
                {
                    "id": h.document.id,
                    "title": h.document.title,
                    "source": h.document.source,
                    "raw_relevance": round(h.raw_relevance, 6),
                    "fuzzy_score": round(h.fuzzy_score, 6),
                    "authority": round(h.authority, 6),
                    "freshness": round(h.freshness, 6),
                    "reliability": round(h.reliability, 6),
                }
                for h in hits
            ],
        }
