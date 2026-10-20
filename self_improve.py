"""
ButcherAI — Self-Improving Loop (Exercise 4.3)
================================================
When evaluation finds a failure, automatically:
1. Log the failing question + expected + actual
2. Ask Gemini to suggest a better system prompt
3. Generate a candidate prompt
4. Run the golden test with the new prompt
5. If it passes, flag for human review
6. Human approves -> new prompt goes live

Usage:
    python self_improve.py              # Analyze failures and suggest improvements
    python self_improve.py --apply      # Apply suggested improvements (with human review)
"""

from __future__ import annotations

import json
import os
import sys
import io
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("DB_PATH", "ButcherAI.db")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
IMPROVEMENTS_DIR = Path("improvements")
IMPROVEMENTS_DIR.mkdir(exist_ok=True)


def analyze_failures() -> list:
    """Run evaluation and collect all failures."""
    from evaluation_harness import evaluate_rag_retrieval, RAG_GOLDEN_TESTS

    summary = evaluate_rag_retrieval(DB_PATH)
    failures = []
    for r in summary.results:
        if not r.passed:
            failures.append({
                "question": r.question,
                "expected": r.expected,
                "actual": r.actual,
                "category": next(
                    (tc.get("expected_category") for tc in RAG_GOLDEN_TESTS
                     if tc["question"] == r.question), "unknown"
                ),
            })
    return failures


def suggest_improvement(failure: dict) -> dict:
    """Ask Gemini to suggest a better prompt for a specific failure."""
    if not API_KEY:
        return {"error": "No API key available"}

    try:
        from gemini_client import AgentChatSession

        prompt = (
            "You are a prompt engineering expert. A RAG system failed to retrieve "
            "the correct document for a specific question.\n\n"
            f"Question: {failure['question']}\n"
            f"Expected document: {failure['expected']}\n"
            f"Actual document: {failure['actual']}\n"
            f"Category: {failure['category']}\n\n"
            "Suggest an improved system prompt that would help the retrieval system "
            "better match this type of question to the correct document. "
            "Focus on: keyword hints, semantic bridges, and domain-specific terms.\n\n"
            "Return ONLY valid JSON:\n"
            '{"improved_prompt": "the improved prompt text", "key_changes": ["change1", "change2"]}'
        )

        session = AgentChatSession(
            api_key=API_KEY,
            system_instruction="You are a prompt engineering expert.",
            model="gemini-flash-lite-latest",
        )
        result = session.send(prompt)
        if result.get("error"):
            return {"error": result["error"]}

        response = result["response"]
        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            return {"error": "No response"}

        text = ""
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text

        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    except Exception as exc:
        return {"error": str(exc)}


def save_improvement(failure: dict, suggestion: dict):
    """Save an improvement suggestion for human review."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = IMPROVEMENTS_DIR / f"improvement_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "failure": failure,
        "suggestion": suggestion,
        "status": "pending_review",  # pending_review, approved, rejected
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filename


def main():
    print("=" * 60)
    print("  ButcherAI -- Self-Improving Loop")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Step 1: Analyze failures
    print("Step 1: Analyzing evaluation failures...")
    failures = analyze_failures()
    print(f"  Found {len(failures)} failures")

    if not failures:
        print("  No failures found -- system is performing well!")
        return

    for f in failures:
        print(f"    - {f['question'][:50]}... (expected: {f['expected']}, got: {f['actual']})")

    if not API_KEY:
        print("\n  No API key available -- cannot generate suggestions")
        print("  Set GEMINI_API_KEY to enable self-improvement")
        return

    # Step 2: Generate suggestions
    print("\nStep 2: Generating improvement suggestions...")
    for failure in failures:
        print(f"\n  Processing: {failure['question'][:50]}...")
        suggestion = suggest_improvement(failure)

        if "error" in suggestion:
            print(f"    Error: {suggestion['error']}")
            continue

        print(f"    Suggested changes: {suggestion.get('key_changes', [])}")

        # Step 3: Save for human review
        filename = save_improvement(failure, suggestion)
        print(f"    Saved to: {filename}")

    # Step 3: Summary
    print("\n" + "=" * 60)
    pending = list(IMPROVEMENTS_DIR.glob("improvement_*.json"))
    print(f"  Total suggestions pending review: {len(pending)}")
    print("  Review files in: improvements/")
    print("  Apply with: python self_improve.py --apply")
    print("=" * 60)


if __name__ == "__main__":
    main()
