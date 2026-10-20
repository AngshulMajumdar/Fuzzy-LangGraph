from fuzzy_langgraph.core import DocumentRecord, FuzzyIndex, FuzzyRAGEngine


def docs():
    return [
        DocumentRecord("stale", "The current access code is ALPHA-1.", metadata={"authority":0.4,"freshness":0.1,"reliability":0.5}),
        DocumentRecord("official", "The active access code is BRAVO-9.", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
    ]


def test_fuzzy_trust_can_override_stale_lexical_match():
    idx = FuzzyIndex(docs())
    hits = idx.search("What is the current access code?", k=2)
    assert hits[0].document.id == "official"


def test_missing_evidence_abstains():
    rag = FuzzyRAGEngine(docs())
    out = rag.query("What is the cafeteria breakfast price?")
    assert out["decision"] == "abstain"


def test_equal_high_quality_ambiguity_abstains():
    d = [
        DocumentRecord("a", "The lab is in room B-204.", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
        DocumentRecord("b", "The lab is in room C-101.", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
    ]
    out = FuzzyRAGEngine(d).query("Which room is the lab in?")
    assert out["decision"] == "abstain"
    assert out["reason"] == "ambiguous_after_expansion"
