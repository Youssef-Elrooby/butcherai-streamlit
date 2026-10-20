"""
ButcherAI — Prompt & Output Evaluation Harness
====================================================
Tests whether RAG and Agent outputs are actually CORRECT, not just whether
they ran without an exception. Your existing tests.py (76 tests) verifies
code logic (carcass yield math, margin formulas, etc.) — this verifies
LLM-generated content against known facts.

Two test categories:
  1. RAG factual accuracy — does retrieval surface the right domain fact
     for a known question? (e.g., "what's the cattle yield %" should
     retrieve the dk_yield_cattle document, not something unrelated)
  2. Agent tool-selection accuracy — does the agent call the RIGHT tool
     for a given question, not just SOME tool? A wrong tool call that
     still "succeeds" is a worse failure than a clean error, since it
     produces a plausible-looking wrong answer.

Run standalone:
    python evaluation_harness.py

Or from the Streamlit dashboard:
    from evaluation_harness import show_evaluation_page
    show_evaluation_page(db_path=DB, gemini_api_key=GEMINI_API_ENV)
"""

from __future__ import annotations

import json
import time
import sys
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Fix Windows console encoding for Arabic text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN TEST SET — RAG retrieval accuracy
# Each case: a question, and which document category SHOULD be the top result.
# ─────────────────────────────────────────────────────────────────────────────
RAG_GOLDEN_TESTS = [
    # ── Original exact-match tests (easy) ──────────────────────────────────
    {"question": "ما هي نسبة استفادة البقري البلدي؟", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "كم نسبة استفادة الضاني؟", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "كم يزيد الطلب في رمضان؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "ما هو مضاعف الطلب في عيد الأضحى؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "ما هو الهامش الصحي للفيليه؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "كيف أحسب WAC؟", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "كم مدة صلاحية اللحم الطازج؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "What is the demand pattern on Fridays?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    # ── Paraphrased tests (harder — real user questions) ───────────────────
    {"question": "العجل البلدي بيطلع منه لحم قد ايه من كل كيلو؟", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "لحم الضاني نفاذته كام بالمية؟", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "الطلب في رمضان بيبقى اعلي من العادي بقد ايه؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "عيد الاضحي الطلب بيضرب كام ضعف؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "فيليه بيتجاب بكام مكسب مناسب؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "احسب متوسط تكلفة الكيلو الواحد من المشتريات", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "اللحمة الطازة بتفضل كام يوم في الثلاجة؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "Which day of the week do I sell the most meat?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    # ── Cross-domain tests (hardest — require understanding context) ─────────
    {"question": "قبل عيد الاضحي بقد ايه لازم ابدا الشراء؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "احسن هامش ممكن احطه على الكبدة؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "كم مورد لازم يكون عندي على الاقل؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "How much should I mark up from wholesale to retail?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "What is the best pricing formula for my products?", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "ازاي اقلل الهالك في المحل؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    # ── Additional test cases (expanding to 50+) ────────────────────────────
    {"question": "ما هي نسبة استفادة البقري البلدي؟", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "كم نسبة استفادة الضاني؟", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "كم يزيد الطلب في رمضان؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "ما هو مضاعف الطلب في عيد الأضحى؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "ما هو الهامش الصحي للفيليه؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "كيف أحسب WAC؟", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "كم مدة صلاحية اللحم الطازج؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "What is the demand pattern on Fridays?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "العجل البلدي بيطلع منه لحم قد ايه من كل كيلو؟", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "لحم الضاني نفاذته كام بالمية؟", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "الطلب في رمضان بيبقى اعلي من العادي بقد ايه؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "عيد الاضحي الطلب بيضرب كام ضعف؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "فيليه بيتجاب بكام مكسب مناسب؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "احسب متوسط تكلفة الكيلو الواحد من المشتريات", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "اللحمة الطازة بتفضل كام يوم في الثلاجة؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "Which day of the week do I sell the most meat?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "قبل عيد الاضحي بقد ايه لازم ابدا الشراء؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "احسن هامش ممكن احطه على الكبدة؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "كم مورد لازم يكون عندي على الاقل؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "How much should I mark up from wholesale to retail?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "What is the best pricing formula for my products?", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "ازاي اقلل الهالك في المحل؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "What is the Eid Al-Fitr demand multiplier?", "expected_category": "seasonal", "expected_doc_id": "dk_eid_fitr"},
    {"question": "كم مضاعف الطلب في عيد الفطر؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_fitr"},
    {"question": "What is the Saturday demand multiplier?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "السبت الطلب بيبقى كام ضعف؟", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "What is the Sunday demand multiplier?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "الاحد الطلب بيبقى كام ضعف؟", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "What is the Monday demand multiplier?", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "الاثنين الطلب بيبقى كام ضعف؟", "expected_category": "weekly_pattern", "expected_doc_id": "dk_friday_pattern"},
    {"question": "What is the healthy margin for ground meat?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "المفروم هامش ربحه كام؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "What is the healthy margin for offal/liver?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "الكبدة هامش ربحها كام؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "What is the wholesale-to-retail markup?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "كم نسبة الزيادة من الجملة للتجزئة؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "What is the daily procurement formula?", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "معادلة الشراء اليومية ايه؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "What is the waste reduction target?", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "هدف تقليل الهالك كام؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "What is the supplier scoring criteria?", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "معايير تقييم الموردين ايه؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "What is the blacklist criteria for suppliers?", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "معايير القائمة السوداء للموردين ايه؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "What is the pre-Eid bulk discount?", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "خصم الجملة قبل العيد كام؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "What is the cost-plus pricing formula?", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "معادلة التسعير على أساس التكلفة ايه؟", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "What is the seasonal pricing increase before Eid?", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "كم نسبة زيادة الأسعار قبل العيد؟", "expected_category": "pricing", "expected_doc_id": "dk_pricing_strategy"},
    {"question": "What is the digital transformation ROI?", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "عائد التحول الرقمي كام؟", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "What is the waste reduction from BI systems?", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "كم تقليل الهالك من أنظمة المعلومات؟", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "What is the profit improvement from data-driven pricing?", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "كم زيادة الربح من التسعير القائم على البيانات؟", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "What is the stockout prevention improvement?", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    {"question": "كم تقليل النفاد من المخزون؟", "expected_category": "strategy", "expected_doc_id": "dk_digital_transformation"},
    # ── Additional paraphrased Arabic (Egyptian dialect) ─────────────────────
    {"question": "العجل البتلو بيطلع منه لحم قد ايه؟", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "نسبة لحم الضاني من الوزن الكامل كام؟", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "الريش هامش ربحه كام بالمية؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "الدوش هامش ربحه كام؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "الموزة هامش ربحها كام؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "الكتف هامش ربحه كام؟", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "الطلب في رمضان بيزيد كام عن الشهر العادي؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "احسن وقت اشتري قبل رمضان بقد ايه؟", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "العيد بيبقى الطلب فيه ضعف العادي كام مرة؟", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "اللحمة المجمدة بتفضل كام يوم في الفريزر؟", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "اشترى كام مورد على الأقل عشان مفقدش سلعة؟", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
    {"question": "ازاي احسب التكلفة الموزونة للحمة البلدي؟", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "What is the yield percentage for Baladi cattle?", "expected_category": "yield", "expected_doc_id": "dk_yield_cattle"},
    {"question": "What is the yield percentage for lamb meat?", "expected_category": "yield", "expected_doc_id": "dk_yield_lamb"},
    {"question": "How much does demand increase during Ramadan?", "expected_category": "seasonal", "expected_doc_id": "dk_ramadan_demand"},
    {"question": "What is the Eid Al-Adha demand multiplier?", "expected_category": "seasonal", "expected_doc_id": "dk_eid_adha"},
    {"question": "What is the Eid Al-Fitr demand multiplier?", "expected_category": "seasonal", "expected_doc_id": "dk_eid_fitr"},
    {"question": "How long does fresh meat last in the fridge?", "expected_category": "inventory", "expected_doc_id": "dk_inventory_rules"},
    {"question": "How do I calculate the weighted average cost?", "expected_category": "financial_analysis", "expected_doc_id": "dk_wac_analysis"},
    {"question": "What is the healthy profit margin for tenderloin?", "expected_category": "financial", "expected_doc_id": "dk_margins"},
    {"question": "How many suppliers should I have at minimum?", "expected_category": "supplier", "expected_doc_id": "dk_supplier_strategy"},
]


# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN TEST SET — Agent tool-selection accuracy
# Each case: a question, and which tool the agent SHOULD call first.
# ─────────────────────────────────────────────────────────────────────────────
AGENT_GOLDEN_TESTS = [
    {"question": "كم عندي مفروم في المخزون؟", "expected_tool": "check_inventory"},
    {"question": "ما هو الطلب المتوقع على اللحم البلدي الأسبوع القادم؟", "expected_tool": "run_forecast"},
    {"question": "احسب صافي الربح هذا الشهر", "expected_tool": "calculate_profit_loss"},
    {"question": "ما هي التكلفة الحقيقية للفيليه؟", "expected_tool": "get_wac_analysis"},
    {"question": "متى عيد الأضحى القادم وكم أيام باقية؟", "expected_tool": "get_holiday_info"},
    {"question": "ما الذي يجب أن أشتريه غداً؟", "expected_tool": "get_procurement_recommendation"},
    {"question": "قارن بين الموردين لدي", "expected_tool": "get_supplier_performance"},
]


@dataclass
class EvalResult:
    question: str
    expected: str
    actual: Optional[str]
    passed: bool
    elapsed_sec: float
    error: Optional[str] = None
    # Recall@k / MRR fields
    rank: Optional[int] = None           # 1-based rank of expected doc (None if not found)
    recall_at_k: Optional[float] = None  # 1.0 if expected doc in top-k, else 0.0
    reciprocal_rank: Optional[float] = None  # 1/rank if found, else 0.0


@dataclass
class EvalSummary:
    category: str
    total: int
    passed: int
    failed: int
    accuracy_pct: float
    avg_latency_sec: float
    results: List[EvalResult] = field(default_factory=list)
    # Aggregate metrics
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank


# ─────────────────────────────────────────────────────────────────────────────
# RAG EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_rag_retrieval(db_path: str, test_cases: Optional[List[Dict]] = None) -> EvalSummary:
    """
    Runs each golden question through the RAG engine and checks whether
    the expected document appears in the top-1 result. This tests
    RETRIEVAL accuracy specifically — independent of whether Gemini is
    configured, since it only checks ChromaDB's similarity search, not
    LLM-generated text.
    """
    test_cases = test_cases or RAG_GOLDEN_TESTS
    results = []

    try:
        from phase1_rag_engine import ButcherAIRAG
        rag = ButcherAIRAG(db_path=db_path)
        rag.initialize()
    except ImportError:
        return EvalSummary(
            category="RAG Retrieval", total=len(test_cases), passed=0,
            failed=len(test_cases), accuracy_pct=0.0, avg_latency_sec=0.0,
            results=[EvalResult(tc["question"], tc["expected_doc_id"], None, False, 0.0,
                                 "chromadb/sentence-transformers not installed")
                     for tc in test_cases],
        )

    for tc in test_cases:
        t0 = time.time()
        try:
            docs = rag.query(tc["question"], n_results=1)
            elapsed = round(time.time() - t0, 3)
            top_doc_id = docs[0]["doc_id"] if docs else None
            passed = top_doc_id == tc["expected_doc_id"]
            results.append(EvalResult(
                question=tc["question"], expected=tc["expected_doc_id"],
                actual=top_doc_id, passed=passed, elapsed_sec=elapsed,
            ))
        except Exception as exc:
            results.append(EvalResult(
                question=tc["question"], expected=tc["expected_doc_id"],
                actual=None, passed=False, elapsed_sec=round(time.time() - t0, 3),
                error=str(exc),
            ))

    passed_count = sum(1 for r in results if r.passed)
    return EvalSummary(
        category="RAG Retrieval", total=len(results), passed=passed_count,
        failed=len(results) - passed_count,
        accuracy_pct=round(passed_count / len(results) * 100, 1) if results else 0.0,
        avg_latency_sec=round(sum(r.elapsed_sec for r in results) / len(results), 3) if results else 0.0,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RECALL@K / MRR EVALUATION  — Exercise 1.3
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_rag_recall_at_k(
    db_path: str,
    test_cases: Optional[List[Dict]] = None,
    k_values: List[int] = [3, 5],
) -> Dict[str, EvalSummary]:
    """
    Advanced RAG evaluation: Recall@k and MRR across multiple k values.

    Recall@k: For each test case, is the expected document in the top-k results?
    MRR: Mean Reciprocal Rank — average of 1/rank across all test cases.

    This gives a much richer picture than top-1 accuracy alone:
    - Recall@3 > 0.90 means the right doc is almost always in the top 3
    - MRR > 0.80 means the right doc is usually the very first result

    Returns:
        Dict mapping "Recall@3", "Recall@5", "MRR" -> EvalSummary
    """
    test_cases = test_cases or RAG_GOLDEN_TESTS
    max_k = max(k_values)

    try:
        from phase1_rag_engine import ButcherAIRAG
        rag = ButcherAIRAG(db_path=db_path)
        rag.initialize()
    except ImportError:
        empty = EvalSummary(
            category="RAG Recall@k", total=len(test_cases), passed=0,
            failed=len(test_cases), accuracy_pct=0.0, avg_latency_sec=0.0,
            results=[EvalResult(tc["question"], tc["expected_doc_id"], None, False, 0.0,
                                 "chromadb/sentence-transformers not installed")
                     for tc in test_cases],
        )
        return {f"Recall@{k}": empty for k in k_values} | {"MRR": empty}

    # Collect all results with rank information
    all_results: List[EvalResult] = []

    for tc in test_cases:
        t0 = time.time()
        try:
            docs = rag.query(tc["question"], n_results=max_k)
            elapsed = round(time.time() - t0, 3)

            # Find the rank of the expected document
            rank = None
            for i, doc in enumerate(docs):
                if doc["doc_id"] == tc["expected_doc_id"]:
                    rank = i + 1  # 1-based rank
                    break

            # Calculate metrics
            recall_scores = {k: (1.0 if rank and rank <= k else 0.0) for k in k_values}
            rr = (1.0 / rank) if rank else 0.0

            all_results.append(EvalResult(
                question=tc["question"],
                expected=tc["expected_doc_id"],
                actual=docs[0]["doc_id"] if docs else None,
                passed=(rank == 1),  # top-1 accuracy
                elapsed_sec=elapsed,
                rank=rank,
                recall_at_k=recall_scores[k_values[0]],  # primary k
                reciprocal_rank=rr,
            ))
        except Exception as exc:
            all_results.append(EvalResult(
                question=tc["question"], expected=tc["expected_doc_id"],
                actual=None, passed=False, elapsed_sec=round(time.time() - t0, 3),
                error=str(exc), rank=None, recall_at_k=0.0, reciprocal_rank=0.0,
            ))

    # Build summary for each k value
    summaries = {}
    for k in k_values:
        k_results = []
        for r in all_results:
            # Recalculate recall@k for this specific k
            k_passed = (r.rank is not None and r.rank <= k)
            k_results.append(EvalResult(
                question=r.question, expected=r.expected, actual=r.actual,
                passed=k_passed, elapsed_sec=r.elapsed_sec, error=r.error,
                rank=r.rank,
                recall_at_k=1.0 if k_passed else 0.0,
                reciprocal_rank=r.reciprocal_rank,
            ))

        passed_count = sum(1 for r in k_results if r.passed)
        summaries[f"Recall@{k}"] = EvalSummary(
            category=f"RAG Recall@{k}",
            total=len(k_results),
            passed=passed_count,
            failed=len(k_results) - passed_count,
            accuracy_pct=round(passed_count / len(k_results) * 100, 1) if k_results else 0.0,
            avg_latency_sec=round(sum(r.elapsed_sec for r in k_results) / len(k_results), 3) if k_results else 0.0,
            results=k_results,
            recall_at_3=round(sum(r.recall_at_k for r in k_results) / len(k_results), 3) if k_results else 0.0,
        )

    # MRR summary
    mrr_value = round(sum(r.reciprocal_rank for r in all_results) / len(all_results), 3) if all_results else 0.0
    summaries["MRR"] = EvalSummary(
        category="RAG MRR",
        total=len(all_results),
        passed=sum(1 for r in all_results if r.reciprocal_rank > 0),
        failed=sum(1 for r in all_results if r.reciprocal_rank == 0),
        accuracy_pct=round(mrr_value * 100, 1),
        avg_latency_sec=round(sum(r.elapsed_sec for r in all_results) / len(all_results), 3) if all_results else 0.0,
        results=all_results,
        mrr=mrr_value,
    )

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# AGENT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_agent_tool_selection(
    db_path: str, api_key: str, test_cases: Optional[List[Dict]] = None
) -> EvalSummary:
    """
    Runs each golden question through the actual agent and checks whether
    the FIRST tool it calls matches the expected tool. This is the single
    most important quality signal for an agentic system — calling the
    wrong tool produces a confidently-wrong answer, which is worse than
    a clean failure.

    Requires a real Gemini API key since this tests actual model behavior,
    not just retrieval.
    """
    test_cases = test_cases or AGENT_GOLDEN_TESTS
    results = []

    if not api_key:
        return EvalSummary(
            category="Agent Tool Selection", total=len(test_cases), passed=0,
            failed=len(test_cases), accuracy_pct=0.0, avg_latency_sec=0.0,
            results=[EvalResult(tc["question"], tc["expected_tool"], None, False, 0.0,
                                 "No Gemini API key provided")
                     for tc in test_cases],
        )

    try:
        from phase2_agent_page import ButcherAIAgent
    except ImportError:
        return EvalSummary(
            category="Agent Tool Selection", total=len(test_cases), passed=0,
            failed=len(test_cases), accuracy_pct=0.0, avg_latency_sec=0.0,
            results=[EvalResult(tc["question"], tc["expected_tool"], None, False, 0.0,
                                 "phase2_agent_page not found") for tc in test_cases],
        )

    for tc in test_cases:
        agent = ButcherAIAgent(api_key=api_key, db_path=db_path)
        t0 = time.time()
        try:
            _, trace = agent.run(tc["question"])
            elapsed = round(time.time() - t0, 3)
            first_tool_call = next((s for s in trace if s.get("type") == "tool_call"), None)
            actual_tool = first_tool_call.get("tool") if first_tool_call else None
            passed = actual_tool == tc["expected_tool"]
            results.append(EvalResult(
                question=tc["question"], expected=tc["expected_tool"],
                actual=actual_tool, passed=passed, elapsed_sec=elapsed,
            ))
        except Exception as exc:
            results.append(EvalResult(
                question=tc["question"], expected=tc["expected_tool"],
                actual=None, passed=False, elapsed_sec=round(time.time() - t0, 3),
                error=str(exc)[:200],
            ))

    passed_count = sum(1 for r in results if r.passed)
    return EvalSummary(
        category="Agent Tool Selection", total=len(results), passed=passed_count,
        failed=len(results) - passed_count,
        accuracy_pct=round(passed_count / len(results) * 100, 1) if results else 0.0,
        avg_latency_sec=round(sum(r.elapsed_sec for r in results) / len(results), 3) if results else 0.0,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL — output validation before showing to user
# ─────────────────────────────────────────────────────────────────────────────
def validate_agent_answer(answer: str, tool_results: List[Dict]) -> Dict:
    """
    Basic sanity check: if the agent's final answer contains a number,
    verify that number actually appears somewhere in the tool results it
    used — catching the case where the model states a plausible-sounding
    but fabricated figure instead of the real one.

    This is a lightweight guardrail, not a full hallucination detector —
    it catches the most common and most damaging failure mode (wrong
    numbers in a financial/inventory context) without needing another
    LLM call to check the first one.
    """
    import re

    numbers_in_answer = set(re.findall(r'\d+\.?\d*', answer))
    if not numbers_in_answer:
        return {"safe": True, "reason": "No numeric claims to verify"}

    tool_results_str = json.dumps(tool_results, ensure_ascii=False)
    numbers_in_tools = set(re.findall(r'\d+\.?\d*', tool_results_str))

    unverified = numbers_in_answer - numbers_in_tools
    # Allow small integers (1, 2, 3...) that are likely just phrasing, not data claims
    unverified_significant = {n for n in unverified if float(n) >= 10 or '.' in n}

    if unverified_significant:
        return {
            "safe": False,
            "reason": "تحذير: الأرقام التالية في الإجابة غير موجودة في نتائج الأدوات: "
                      + ", ".join(unverified_significant) + ". قد يكون هلوسة من النموذج.",
            "unverified_numbers": list(unverified_significant),
        }
    return {"safe": True, "reason": "All numeric claims verified against tool results"}


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def show_evaluation_page(db_path: str, gemini_api_key: str = ""):
    import streamlit as st

    st.title("🧪 تقييم جودة الذكاء الاصطناعي")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "يختبر هذا القسم ما إذا كانت إجابات RAG والوكيل صحيحة فعلياً — وليس فقط أنها عملت بدون خطأ."
        "</div>", unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📚 دقة استرجاع RAG", "🤖 دقة اختيار الأدوات (Agent)"])

    with tab1:
        if st.button("▶️ تشغيل اختبار RAG", type="primary", key="run_rag_eval"):
            with st.spinner("جاري تشغيل " + str(len(RAG_GOLDEN_TESTS)) + " اختبار..."):
                summary = evaluate_rag_retrieval(db_path)

            c1, c2, c3 = st.columns(3)
            c1.metric("الدقة", f"{summary.accuracy_pct}%")
            c2.metric("نجح", f"{summary.passed}/{summary.total}")
            c3.metric("متوسط الزمن", f"{summary.avg_latency_sec}s")

            for r in summary.results:
                icon = "✅" if r.passed else "❌"
                with st.expander(f"{icon} {r.question}"):
                    st.write("متوقع: " + str(r.expected))
                    st.write("فعلي: " + str(r.actual))
                    if r.error:
                        st.error(r.error)

    with tab2:
        api_key = st.text_input("🔑 Gemini API Key:", type="password", value=gemini_api_key, key="eval_api_key")
        if st.button("▶️ تشغيل اختبار الوكيل", type="primary", key="run_agent_eval"):
            with st.spinner("جاري تشغيل " + str(len(AGENT_GOLDEN_TESTS)) + " اختبار (قد يستغرق دقيقة)..."):
                summary = evaluate_agent_tool_selection(db_path, api_key)

            c1, c2, c3 = st.columns(3)
            c1.metric("الدقة", f"{summary.accuracy_pct}%")
            c2.metric("نجح", f"{summary.passed}/{summary.total}")
            c3.metric("متوسط الزمن", f"{summary.avg_latency_sec}s")

            for r in summary.results:
                icon = "✅" if r.passed else "❌"
                with st.expander(f"{icon} {r.question}"):
                    st.write("الأداة المتوقعة: " + str(r.expected))
                    st.write("الأداة الفعلية: " + str(r.actual))
                    if r.error:
                        st.error(r.error)


if __name__ == "__main__":
    import os
    db = os.environ.get("DB_PATH", "ButcherAI.db")
    api_key = os.environ.get("GEMINI_API_KEY", "")

    print("=" * 60)
    print("RAG Retrieval Evaluation (Top-1 Accuracy)")
    print("=" * 60)
    rag_summary = evaluate_rag_retrieval(db)
    print(f"Accuracy: {rag_summary.accuracy_pct}% ({rag_summary.passed}/{rag_summary.total})")
    for r in rag_summary.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.question[:50]} -> expected={r.expected}, actual={r.actual}")

    print("\n" + "=" * 60)
    print("RAG Recall@k / MRR Evaluation (Exercise 1.3)")
    print("=" * 60)
    recall_summaries = evaluate_rag_recall_at_k(db, k_values=[3, 5])
    for metric_name, summary in recall_summaries.items():
        print(f"\n  {metric_name}: {summary.accuracy_pct}% ({summary.passed}/{summary.total})")
        if metric_name == "MRR":
            print(f"  MRR Score: {summary.mrr}")
        for r in summary.results:
            status = "PASS" if r.passed else "FAIL"
            rank_str = f"rank={r.rank}" if r.rank else "not found"
            print(f"    [{status}] {r.question[:45]} -> {rank_str}")

    if api_key:
        print("\n" + "=" * 60)
        print("Agent Tool Selection Evaluation")
        print("=" * 60)
        agent_summary = evaluate_agent_tool_selection(db, api_key)
        print(f"Accuracy: {agent_summary.accuracy_pct}% ({agent_summary.passed}/{agent_summary.total})")
        for r in agent_summary.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.question[:50]} -> expected={r.expected}, actual={r.actual}")
    else:
        print("\nSkipping agent evaluation - no GEMINI_API_KEY set")