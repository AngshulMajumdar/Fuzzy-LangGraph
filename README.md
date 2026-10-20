# Fuzzy LangGraph RAG

**Fuzzy LangGraph RAG** turns uncertain RAG evidence into explicit graph control flow. Retrieval relevance, source authority, freshness, and reliability are represented as fuzzy variables. A Sugeno controller routes the compiled graph to **generate**, **expand retrieval**, or **abstain**.

## Graph

`START -> retrieve -> assess -> {generate | expand -> retrieve | abstain} -> END`

`expand` is an actual loop. Borderline or ambiguous evidence widens retrieval once and returns to assessment. If ambiguity survives, the graph abstains rather than cycling or forcing an answer.

## Why fuzzy graph routing?

Hard predicates such as `score > 0.7` discard useful information. A source can be relevant but stale, current but weakly authoritative, or one of several equally credible contradictory sources. Fuzzy memberships preserve partial evidence and let the graph make a controlled decision instead of pretending the uncertainty has disappeared.

## Full RAG benchmark

The benchmark runs ingestion, retrieval, fuzzy assessment, graph routing, optional expanded retrieval, grounded answer synthesis, citation, and abstention.

| Suite | Fuzzy RAG | Plain TF-IDF RAG |
|---|---:|---:|
| Clean end-to-end accuracy | **1.000** | 1.000 |
| Conflict/stale/missing end-to-end accuracy | **1.000** | 0.278 |
| Conflict Recall@1 | **1.000** | 0.250 |
| Conflict MRR | **1.000** | 0.611 |
| Conflict abstention accuracy | **1.000** | 0.333 |
| Grounding accuracy | **1.000** | 1.000 |

Raw case-level results are in `artifacts/full_rag_results.json`; compiled-graph results are in `artifacts/framework_full_rag_results.json`.

## Quick start

```python
from fuzzy_langgraph import DocumentRecord, FuzzyIndex
from fuzzy_langgraph.graph import FuzzyRAGGraph

docs = [
    DocumentRecord(
        "official",
        "The current travel portal is TripDesk.",
        source="official-handbook",
        metadata={"authority": 0.98, "freshness": 1.0, "reliability": 0.95},
    )
]

graph = FuzzyRAGGraph(FuzzyIndex(docs))
print(graph.invoke("Which portal is used for travel?"))
```

## Reproduce

```bash
pip install -e .
python evaluation/rag_benchmark.py
python evaluation/framework_rag_benchmark.py
pytest -q
```

This GitHub repository is the **full RAG implementation**: source, graph code, benchmarks, tests, examples, raw results, and validation notes. Hugging Face deployment is intentionally split into a separate API-only package.
