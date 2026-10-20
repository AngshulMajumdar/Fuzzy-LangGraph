from __future__ import annotations
import json
from pathlib import Path

from evaluation.rag_benchmark import CLEAN_DOCS, CLEAN_CASES, CONFLICT_DOCS, CONFLICT_CASES
from fuzzy_langgraph.core import FuzzyIndex, ExtractiveGroundedGenerator
from fuzzy_langgraph.graph import FuzzyRAGGraph


def _run_suite(docs,cases):
    rag=FuzzyRAGGraph(FuzzyIndex(docs),initial_k=4)
    rows=[]
    for c in cases:
        out=rag.invoke(c["q"])
        evidence=out.get("evidence",[])
        top_id=evidence[0].get("id") if evidence else None
        if c["abstain"]:
            correct=out["answer"] == ExtractiveGroundedGenerator.abstain_text
        else:
            correct=str(c["gold"]).lower() in out["answer"].lower()
        rows.append({"id":c["id"],"top_doc":top_id,"answer":out["answer"],"route":out.get("route"),"reason":out.get("reason"),"expanded":out.get("expanded"),"correct":correct,"citations":out.get("citations",[])})
    return {"n":len(rows),"accuracy":sum(r["correct"] for r in rows)/len(rows),"rows":rows}


def run_framework_benchmark(output_path: str|Path|None=None):
    result={
        "framework":"langgraph",
        "pipeline":"START -> retrieve -> assess -> {generate | expand -> retrieve | abstain} -> END",
        "clean":_run_suite(CLEAN_DOCS,CLEAN_CASES),
        "conflict":_run_suite(CONFLICT_DOCS,CONFLICT_CASES),
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__=="__main__": print(json.dumps(run_framework_benchmark(),indent=2))
