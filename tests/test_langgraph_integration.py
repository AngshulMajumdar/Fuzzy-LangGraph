import importlib.util
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")


def test_compiled_stategraph_full_path():
    from fuzzy_langgraph import DocumentRecord, FuzzyIndex
    from fuzzy_langgraph.graph import FuzzyRAGGraph

    docs = [
        DocumentRecord("old", "The current portal is OldDesk.", metadata={"authority":0.4,"freshness":0.1,"reliability":0.5}),
        DocumentRecord("new", "The current portal is PeopleHub.", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
    ]
    g = FuzzyRAGGraph(FuzzyIndex(docs))
    out = g.invoke("What is the current portal?")
    assert "PeopleHub" in out["answer"]
    names = set(g.graph.get_graph().nodes)
    assert {"retrieve","assess","expand","generate","abstain"}.issubset(names)
