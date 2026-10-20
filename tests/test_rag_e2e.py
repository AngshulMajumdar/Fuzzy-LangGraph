from evaluation.rag_benchmark import run_benchmark


def test_full_rag_benchmark():
    r = run_benchmark()
    assert r["clean"]["fuzzy"]["end_to_end_accuracy"] == 1.0
    assert r["conflict"]["fuzzy"]["end_to_end_accuracy"] == 1.0
    assert r["conflict"]["fuzzy"]["recall_at_1"] == 1.0
    assert r["conflict"]["fuzzy"]["abstention_accuracy"] == 1.0
    assert r["conflict"]["baseline"]["end_to_end_accuracy"] < 0.5
