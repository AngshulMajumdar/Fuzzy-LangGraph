import importlib.util
import pytest
pytestmark=pytest.mark.skipif(importlib.util.find_spec("langgraph") is None,reason="langgraph not installed")
from evaluation.framework_rag_benchmark import run_framework_benchmark

def test_framework_full_rag():
    r=run_framework_benchmark()
    assert r["clean"]["accuracy"]==1.0
    assert r["conflict"]["accuracy"]==1.0
