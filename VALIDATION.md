# Validation

Validated on Python 3.13.5.

- Repository test suite: **7/7 passed**.
- Core full-RAG benchmark: **8/8 clean + 18/18 conflict/stale/missing cases passed**.
- LangGraph full-RAG benchmark: **8/8 clean + 18/18 conflict/stale/missing cases passed**.
- Graph validated against the supplied **langgraph 1.2.16** and **langchain-core 1.6.11** source archives.
- Executed graph path: `START -> retrieve -> assess -> {generate | expand -> retrieve | abstain} -> END`.
- Wheel build with local build tooling: passed.

The sandbox cannot resolve PyPI. The supplied LangGraph source distribution normally obtains checkpoint/SDK/cache packages as dependencies. For framework-source validation only, minimal **test-only** compatibility modules were provided for those unavailable peripheral packages and LangSmith tracing. They are not included in this repository and do not replace `StateGraph`, graph compilation/execution, fuzzy routing, retrieval, generation, or the RAG benchmark.
