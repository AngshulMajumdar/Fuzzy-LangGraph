from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fuzzy_langgraph.core import (
    FuzzyIndex,
    FuzzyRetrievalConfig,
    DocumentRecord,
    PlainTfidfIndex,
    ExtractiveGroundedGenerator,
)

def _md(kind: str) -> dict[str, Any]:
    if kind == "official":
        return {"reliability_alpha": 18, "reliability_beta": 2, "authority": 0.98, "freshness": 1.0, "tier": kind}
    if kind == "stale":
        return {"reliability_alpha": 6, "reliability_beta": 6, "authority": 0.45, "freshness": 0.18, "tier": kind}
    if kind == "community":
        return {"reliability_alpha": 7, "reliability_beta": 5, "authority": 0.55, "freshness": 0.65, "tier": kind}
    if kind == "official_equal":
        return {"reliability_alpha": 18, "reliability_beta": 2, "authority": 0.98, "freshness": 1.0, "tier": "official"}
    return {"reliability_alpha": 1, "reliability_beta": 1, "authority": 0.5, "freshness": 0.5, "tier": kind}


CONFLICT_DOCS = [
    DocumentRecord("LEG-HR-PORTAL", "The current HR portal for leave requests and payslips is WorkSphere.", "Current HR portal login guide", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-HR-PORTAL", "All employee self-service tasks, including leave requests and payslips, are now handled in PeopleHub.", "Employee self-service migration completed", "official-handbook", _md("official")),
    DocumentRecord("COM-HR-PORTAL", "Some teams used WorkSphere during migration; ask HR if your bookmark no longer works.", "HR portal discussion", "community-wiki", _md("community")),
    DocumentRecord("LEG-LEAVE", "Employees receive 15 paid leave days per year under the leave policy.", "Paid leave entitlement FAQ", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-LEAVE", "The 2026 employee benefits policy grants 18 paid leave days per calendar year.", "2026 employee benefits policy", "official-handbook", _md("official")),
    DocumentRecord("COM-LEAVE", "Leave balances vary by category and should be checked in the HR system.", "Leave balance help", "community-wiki", _md("community")),
    DocumentRecord("LEG-REMOTE", "The current remote work policy permits three remote days per week with manager approval.", "Current remote work days", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-REMOTE", "Under the active flexible-work policy, employees may work remotely up to two days per week with manager approval.", "Flexible-work policy revision", "official-handbook", _md("official")),
    DocumentRecord("COM-REMOTE", "Remote work requires manager approval and may vary during special projects.", "Remote work tips", "community-wiki", _md("community")),
    DocumentRecord("LEG-PAYSLIP", "Salary slips are published on the tenth working day of each month.", "Payslip publication date", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-PAYSLIP", "Payroll publishes salary slips by the fifth working day of each month.", "Payroll service standard", "official-handbook", _md("official")),
    DocumentRecord("COM-PAYSLIP", "Payslip publication can be delayed around holidays.", "Payroll discussion", "community-wiki", _md("community")),
    DocumentRecord("LEG-VPN", "The supported corporate VPN protocol is OpenVPN.", "Corporate VPN protocol", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-VPN", "The security baseline now requires WireGuard for the standard corporate VPN profile.", "Network security baseline", "official-it", _md("official")),
    DocumentRecord("COM-VPN", "OpenVPN profiles may still exist on older laptops.", "VPN migration notes", "community-wiki", _md("community")),
    DocumentRecord("LEG-BACKUP", "Production database backups are retained for 14 days.", "Database backup retention policy", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-BACKUP", "Production database backups have a retention period of 30 days.", "Disaster recovery control DR-7", "official-it", _md("official")),
    DocumentRecord("COM-BACKUP", "Teams should not rely on local snapshots as their only backup.", "Backup practices", "community-wiki", _md("community")),
    DocumentRecord("LEG-RATE", "The public API rate limit is 60 requests per minute per token.", "Public API rate limit", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-RATE", "The current gateway quota permits 120 requests per minute per token.", "API gateway operating limits", "official-it", _md("official")),
    DocumentRecord("COM-RATE", "Burst traffic may still be throttled even below the nominal quota.", "API troubleshooting", "community-wiki", _md("community")),
    DocumentRecord("LEG-REGION", "The primary deployment region is us-east-1.", "Primary cloud deployment region", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-REGION", "The production primary region is ap-south-1; disaster recovery remains in a separate region.", "Cloud production topology", "official-it", _md("official")),
    DocumentRecord("COM-REGION", "Developers may use several regions for non-production experiments.", "Cloud developer notes", "community-wiki", _md("community")),
    DocumentRecord("LEG-EXPENSE", "The meal expense cap is INR 3,000 per day on domestic travel.", "Domestic meal expense cap", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-EXPENSE", "For domestic business travel, reimbursable meals are capped at INR 4,500 per day.", "Travel and expense policy 2026", "official-finance", _md("official")),
    DocumentRecord("COM-EXPENSE", "Itemized receipts are recommended for meal reimbursement.", "Expense claim tips", "community-wiki", _md("community")),
    DocumentRecord("LEG-REIMB", "Employees must submit reimbursement claims within 30 days of the expense.", "Reimbursement submission deadline", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-REIMB", "Expense claims must be submitted within 15 days of the transaction date.", "Expense reimbursement standard", "official-finance", _md("official")),
    DocumentRecord("COM-REIMB", "Finance may request additional receipts during reimbursement review.", "Reimbursement FAQ", "community-wiki", _md("community")),
    DocumentRecord("LEG-TRAVEL", "All corporate travel should be booked through TravelPro.", "Corporate travel booking portal", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-TRAVEL", "Approved business travel must be booked through TripDesk.", "Business travel booking procedure", "official-finance", _md("official")),
    DocumentRecord("COM-TRAVEL", "Direct airline booking can cause reimbursement delays.", "Travel booking discussion", "community-wiki", _md("community")),
    DocumentRecord("LEG-PROC", "Purchases above INR 25,000 require procurement review.", "Procurement review threshold", "legacy-cache", _md("stale")),
    DocumentRecord("OFF-PROC", "Procurement review is mandatory for purchases above INR 50,000.", "Procurement control PC-12", "official-finance", _md("official")),
    DocumentRecord("COM-PROC", "Some categories require procurement review regardless of value.", "Procurement FAQ", "community-wiki", _md("community")),
    DocumentRecord("OFF-LAB-A", "The graduate AI lab has moved to room B-204.", "AI lab room notice A", "official-campus", _md("official_equal")),
    DocumentRecord("OFF-LAB-B", "The graduate AI lab has moved to room C-101.", "AI lab room notice B", "official-campus", _md("official_equal")),
    DocumentRecord("OFF-OFFICE-A", "Professor Sen holds office hours on Tuesday at 3 PM.", "Office hours notice A", "official-campus", _md("official_equal")),
    DocumentRecord("OFF-OFFICE-B", "Professor Sen holds office hours on Wednesday at 4 PM.", "Office hours notice B", "official-campus", _md("official_equal")),
]

CONFLICT_CASES = [
    {"id":"q1","q":"What is the current HR portal for leave requests and payslips?","gold":"PeopleHub","doc":"OFF-HR-PORTAL","abstain":False},
    {"id":"q2","q":"How many paid leave days do employees get per year under the current policy?","gold":"18 paid leave days","doc":"OFF-LEAVE","abstain":False},
    {"id":"q3","q":"How many remote work days per week are currently allowed with manager approval?","gold":"two days per week","doc":"OFF-REMOTE","abstain":False},
    {"id":"q4","q":"By which working day are salary slips published each month?","gold":"fifth working day","doc":"OFF-PAYSLIP","abstain":False},
    {"id":"q5","q":"Which VPN protocol is required by the current corporate security baseline?","gold":"WireGuard","doc":"OFF-VPN","abstain":False},
    {"id":"q6","q":"What is the production database backup retention period?","gold":"30 days","doc":"OFF-BACKUP","abstain":False},
    {"id":"q7","q":"What is the current public API rate limit per minute per token?","gold":"120 requests per minute","doc":"OFF-RATE","abstain":False},
    {"id":"q8","q":"Which cloud region is primary for production?","gold":"ap-south-1","doc":"OFF-REGION","abstain":False},
    {"id":"q9","q":"What is the domestic business travel meal expense cap per day?","gold":"INR 4,500 per day","doc":"OFF-EXPENSE","abstain":False},
    {"id":"q10","q":"Within how many days must expense reimbursement claims be submitted?","gold":"15 days","doc":"OFF-REIMB","abstain":False},
    {"id":"q11","q":"Which portal must be used to book approved business travel?","gold":"TripDesk","doc":"OFF-TRAVEL","abstain":False},
    {"id":"q12","q":"Above what amount is procurement review mandatory?","gold":"INR 50,000","doc":"OFF-PROC","abstain":False},
    {"id":"q13","q":"What is the company dental insurance annual limit?","gold":None,"doc":None,"abstain":True},
    {"id":"q14","q":"What is the cafeteria breakfast price?","gold":None,"doc":None,"abstain":True},
    {"id":"q15","q":"What is the emergency satellite phone number?","gold":None,"doc":None,"abstain":True},
    {"id":"q16","q":"What is the annual gym membership reimbursement?","gold":None,"doc":None,"abstain":True},
    {"id":"q17","q":"Which room is the graduate AI lab currently in?","gold":None,"doc":None,"abstain":True},
    {"id":"q18","q":"When are Professor Sen's office hours?","gold":None,"doc":None,"abstain":True},
]

CLEAN_DOCS = [
    DocumentRecord("C1", "The observatory opens to visitors at 9 AM every Saturday.", "Observatory hours", "handbook", _md("official")),
    DocumentRecord("C2", "The chemistry safety briefing is held in Hall 7.", "Chemistry briefing", "handbook", _md("official")),
    DocumentRecord("C3", "Library books may be borrowed for 21 days.", "Library borrowing", "handbook", _md("official")),
    DocumentRecord("C4", "The robotics club meets on Thursday evenings.", "Robotics club", "handbook", _md("official")),
    DocumentRecord("C5", "Visitor parking is located beside Gate 3.", "Visitor parking", "handbook", _md("official")),
    DocumentRecord("C6", "The campus shuttle begins service at 6:30 AM.", "Campus shuttle", "handbook", _md("official")),
    DocumentRecord("C7", "The data science seminar carries two academic credits.", "Data science seminar", "handbook", _md("official")),
    DocumentRecord("C8", "The emergency assembly point is the North Lawn.", "Emergency procedure", "handbook", _md("official")),
]

CLEAN_CASES = [
    {"id":"c1","q":"When does the observatory open on Saturday?","gold":"9 AM","doc":"C1","abstain":False},
    {"id":"c2","q":"Where is the chemistry safety briefing held?","gold":"Hall 7","doc":"C2","abstain":False},
    {"id":"c3","q":"How long may library books be borrowed?","gold":"21 days","doc":"C3","abstain":False},
    {"id":"c4","q":"When does the robotics club meet?","gold":"Thursday evenings","doc":"C4","abstain":False},
    {"id":"c5","q":"Where is visitor parking?","gold":"Gate 3","doc":"C5","abstain":False},
    {"id":"c6","q":"What time does the campus shuttle begin service?","gold":"6:30 AM","doc":"C6","abstain":False},
    {"id":"c7","q":"How many credits is the data science seminar worth?","gold":"two academic credits","doc":"C7","abstain":False},
    {"id":"c8","q":"Where is the emergency assembly point?","gold":"North Lawn","doc":"C8","abstain":False},
]



def _fuzzy_search(searcher: FuzzyIndex, q: str):
    hits = searcher.search(q, k=4)
    decision = searcher.decision(hits)
    expanded = False
    if decision["route"] == "expand":
        expanded = True
        hits = searcher.search(q, k=8, candidate_pool=searcher.config.extended_pool)
        decision = searcher.decision(hits)
        if decision["route"] == "expand":
            decision = {**decision, "route": "abstain", "reason": "ambiguous_after_expansion"}
    return hits, decision, expanded


def _eval(name: str, docs: list[DocumentRecord], cases: list[dict[str, Any]], fuzzy: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gen = ExtractiveGroundedGenerator()
    searcher = FuzzyIndex(docs, FuzzyRetrievalConfig()) if fuzzy else PlainTfidfIndex(docs)
    rows: list[dict[str, Any]] = []
    rr: list[float] = []
    r1: list[float] = []
    r3: list[float] = []
    doc_map = {d.id: d for d in docs}
    for c in cases:
        if fuzzy:
            hits, decision, expanded = _fuzzy_search(searcher, c["q"])
            abstain = decision["route"] != "generate"
        else:
            hits = searcher.search(c["q"], k=4)
            top0 = hits[0] if hits else None
            # Plain retrieval has no source-quality reasoning. It only rejects
            # near-zero lexical matches, deliberately exposing conflict failures.
            abstain = (not hits) or float(top0.raw_relevance) < 0.12
            decision = {"route": "abstain" if abstain else "generate", "confidence": 0.0, "margin": 0.0, "reason": "tfidf_threshold"}
            expanded = False
        top = hits[0] if hits else None
        if abstain:
            answer, citations = gen.abstain_text, []
        else:
            answer, citations = gen.generate(c["q"], hits)
        if c["abstain"]:
            correct = answer == gen.abstain_text
        else:
            correct = str(c["gold"]).lower() in answer.lower()
            ids = [h.document.id for h in hits]
            rank = ids.index(c["doc"]) + 1 if c["doc"] in ids else 0
            r1.append(1.0 if rank == 1 else 0.0)
            r3.append(1.0 if 0 < rank <= 3 else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
        grounded = True
        if citations:
            sentence = answer.rsplit(" [", 1)[0]
            grounded = sentence in doc_map[citations[0]].text
        rows.append({
            "id": c["id"],
            "query": c["q"],
            "expected_abstain": c["abstain"],
            "top_doc": None if top is None else top.document.id,
            "raw_relevance": None if top is None else round(top.raw_relevance, 6),
            "fuzzy_score": None if top is None else round(top.fuzzy_score, 6),
            "decision": decision["route"],
            "reason": decision["reason"],
            "expanded": expanded,
            "answer": answer,
            "citations": citations,
            "correct": correct,
            "grounded": grounded,
        })
    n_answerable = sum(not c["abstain"] for c in cases)
    n_abstain = sum(c["abstain"] for c in cases)
    summary = {
        "system": name,
        "n_cases": len(cases),
        "answerable_cases": n_answerable,
        "abstention_cases": n_abstain,
        "recall_at_1": sum(r1) / len(r1) if r1 else None,
        "recall_at_3": sum(r3) / len(r3) if r3 else None,
        "mrr": sum(rr) / len(rr) if rr else None,
        "end_to_end_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "grounding_accuracy": sum(r["grounded"] for r in rows) / len(rows),
        "abstention_accuracy": (sum(r["correct"] for r, c in zip(rows, cases) if c["abstain"]) / n_abstain) if n_abstain else None,
        "answer_accuracy_on_answerable": (sum(r["correct"] for r, c in zip(rows, cases) if not c["abstain"]) / n_answerable) if n_answerable else None,
    }
    return summary, rows


def run_benchmark(output_path: str | Path | None = None) -> dict[str, Any]:
    clean_f, clean_f_rows = _eval("fuzzy", CLEAN_DOCS, CLEAN_CASES, True)
    clean_t, clean_t_rows = _eval("plain_tfidf", CLEAN_DOCS, CLEAN_CASES, False)
    conflict_f, conflict_f_rows = _eval("fuzzy", CONFLICT_DOCS, CONFLICT_CASES, True)
    conflict_t, conflict_t_rows = _eval("plain_tfidf", CONFLICT_DOCS, CONFLICT_CASES, False)
    result = {
        "benchmark": "controlled_full_rag_fuzzy_v1",
        "methodology": (
            "End-to-end ingest -> retrieve -> fuzzy evidence assessment -> optional expanded retrieval -> "
            "grounded answer synthesis/citation or abstention. CLEAN checks ordinary RAG. CONFLICT adds stale/current "
            "sources, missing evidence, and equal-authority ambiguity. The deterministic extractive generator keeps "
            "the benchmark focused on orchestration instead of external-LLM variance; production APIs accept the same retrieved context path."
        ),
        "clean": {"fuzzy": clean_f, "baseline": clean_t, "fuzzy_cases": clean_f_rows, "baseline_cases": clean_t_rows},
        "conflict": {"fuzzy": conflict_f, "baseline": conflict_t, "fuzzy_cases": conflict_f_rows, "baseline_cases": conflict_t_rows},
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
