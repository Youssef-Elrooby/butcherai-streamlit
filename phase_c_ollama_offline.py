"""
ButcherAI Scenario C — Full Offline Stack (Ollama + Llama 3.2 3B)
====================================================================
For shops with no reliable internet. Replaces Gemini entirely — not a
fallback alongside it, a full swap of the LLM backend.

Model choice rationale (corrected from earlier 7B/1.5B suggestions):
  - Llama 3.2 3B: better tool-calling reliability than 1.5B models,
    small enough to run on CPU at acceptable speed (no GPU dependency,
    since 2GB VRAM cards can't reliably host even a 3B model on GPU
    alongside everything else running).
  - Forced CPU execution via OLLAMA_NUM_GPU=0 — predictable performance
    instead of Ollama guessing whether partial GPU offload helps or hurts.

What this file does NOT do: try to match Gemini's quality. It's a
documented, honest tradeoff — expect slower, less creative, less
reliable tool-calling. Test extensively before trusting it in production.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. ollama pull llama3.2:3b
  3. Set environment variable: OLLAMA_NUM_GPU=0  (forces CPU-only)
  4. pip install requests  (already a transitive dependency, no new install)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = 60  # seconds — CPU inference can be slow, be generous
MAX_AGENT_TURNS_OFFLINE = 8  # lower than the Gemini agent's 12 — smaller
                              # models degrade faster over long reasoning chains


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama_available() -> Tuple[bool, str]:
    """Verify Ollama is running and the required model is pulled."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            return False, (
                f"Ollama is running but {OLLAMA_MODEL} is not pulled. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
        return True, f"✅ Ollama ready with {OLLAMA_MODEL}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Ollama is not running. Start it or run: ollama serve"
    except Exception as exc:
        return False, f"❌ Ollama check failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL GENERATION CALL (CPU-forced)
# ─────────────────────────────────────────────────────────────────────────────
def ollama_generate(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    temperature: float = 0.3,
) -> Dict:
    """
    Call Ollama's /api/generate endpoint with CPU execution forced.

    Returns: {"text": str, "elapsed_sec": float, "error": Optional[str]}
    """
    payload = {
        "model"  : OLLAMA_MODEL,
        "prompt" : prompt,
        "system" : system,
        "stream" : False,
        "options": {
            "num_gpu"    : 0,          # force CPU — predictable on 2GB VRAM cards
            "temperature": temperature,
            "num_ctx"    : 4096,       # context window; 3B models default lower
        },
    }
    if json_mode:
        payload["format"] = "json"

    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text"       : data.get("response", "").strip(),
            "elapsed_sec": round(time.time() - t0, 2),
            "error"      : None,
        }
    except requests.exceptions.Timeout:
        return {"text": "", "elapsed_sec": OLLAMA_TIMEOUT, "error": "Timeout — CPU inference too slow for this query"}
    except Exception as exc:
        return {"text": "", "elapsed_sec": round(time.time() - t0, 2), "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# RAG ANSWER GENERATION (offline replacement for Gemini in phase1_rag_engine.py)
# ─────────────────────────────────────────────────────────────────────────────
def ollama_rag_answer(question: str, context_block: str) -> Dict:
    """
    Offline replacement for the Gemini call in rag_answer().
    RAG retrieval itself is already offline (ChromaDB + local embeddings) —
    this only replaces the answer-generation step.
    """
    system = (
        "You are ButcherAI, a bilingual Arabic/English assistant for an Egyptian "
        "butcher shop. Answer using ONLY the provided context. Be concise and "
        "practical. Answer in the same language as the question."
    )
    prompt = f"{context_block}\n\nQuestion: {question}\n\nAnswer:"

    result = ollama_generate(prompt, system=system, temperature=0.2)
    if result["error"]:
        return {
            "answer": f"⚠️ Local LLM unavailable ({result['error']}). Falling back to raw context.",
            "source": "error_fallback",
            "elapsed_sec": result["elapsed_sec"],
        }
    return {
        "answer": result["text"],
        "source": f"ollama:{OLLAMA_MODEL}",
        "elapsed_sec": result["elapsed_sec"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOL-CALLING  (simplified ReAct — no native function-calling API in
# Ollama's generate endpoint, so this uses a structured-JSON prompting pattern
# instead of Gemini's protos.Tool schema)
# ─────────────────────────────────────────────────────────────────────────────
def _build_tool_selection_prompt(user_message: str, tool_schemas: List[Dict]) -> str:
    """
    Small models need very explicit, constrained instructions to produce
    valid tool calls reliably. This is more verbose than the Gemini prompt
    on purpose — 3B models need the extra scaffolding.
    """
    tools_desc = "\n".join(
        f"- {t['name']}: {t['description']}\n"
        f"  Parameters: {json.dumps(t['parameters'].get('properties', {}), ensure_ascii=False)}"
        for t in tool_schemas
    )
    return f"""You are a tool-selection engine. Given a user question, decide which ONE tool to call next.

Available tools:
{tools_desc}

Rules:
- Respond with ONLY valid JSON, nothing else — no explanation, no markdown fences.
- Format: {{"tool": "tool_name", "args": {{"param1": "value1"}}}}
- If no tool is needed and you can answer directly, use: {{"tool": null, "answer": "your answer here"}}
- Never invent a tool name not in the list above.

User question: {user_message}

JSON response:"""


def _extract_json(text: str) -> Optional[Dict]:
    """Small models often wrap JSON in markdown fences or add stray text — clean it up."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


class OfflineButcherAIAgent:
    """
    Offline agent using Ollama + Llama 3.2 3B with manual tool-call prompting.

    Deliberately more conservative than the Gemini agent:
      - Fewer max turns (8 vs 12) — smaller models degrade over long chains
      - Explicit JSON-only prompting with retry-on-malformed-output
      - Every tool result truncated more aggressively before re-prompting
        (3B models have smaller effective context before losing coherence)
    """

    SYSTEM_PROMPT = (
        "You are ButcherAI, an offline business assistant for an Egyptian butcher shop. "
        "You have tools to check inventory, run forecasts, and analyze finances. "
        "Always base answers on tool results, never invent numbers. "
        "Answer in the same language as the question (Arabic or English)."
    )

    def __init__(self, db_path: str, tool_schemas: List[Dict], execute_tool_fn):
        self.db_path        = db_path
        self.tool_schemas   = tool_schemas
        self.execute_tool_fn = execute_tool_fn
        self.history: List[Dict] = []

    def reset(self):
        self.history = []

    def run(self, user_message: str) -> Tuple[str, List[Dict]]:
        trace: List[Dict] = [{"step": 0, "type": "user_input", "content": user_message}]

        available, msg = check_ollama_available()
        if not available:
            return f"⚠️ {msg}", trace

        context_so_far = f"Original question: {user_message}\n"
        step = 0

        while step < MAX_AGENT_TURNS_OFFLINE:
            step += 1
            prompt = _build_tool_selection_prompt(context_so_far, self.tool_schemas)

            result = ollama_generate(prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.1)
            if result["error"]:
                trace.append({"step": step, "type": "error", "content": result["error"]})
                return f"⚠️ Local model error: {result['error']}", trace

            parsed = _extract_json(result["text"])
            if parsed is None:
                # Retry once with a stricter reminder — small models sometimes
                # need a nudge to produce clean JSON
                retry_prompt = prompt + "\n\nREMINDER: Output ONLY the JSON object, nothing else."
                result = ollama_generate(retry_prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.0)
                parsed = _extract_json(result["text"])

            if parsed is None:
                trace.append({
                    "step": step, "type": "parse_failure",
                    "content": f"Could not parse tool call from: {result['text'][:200]}",
                })
                # Give up gracefully rather than looping forever
                return (
                    "⚠️ الوكيل المحلي لم يتمكن من فهم الطلب بشكل واضح. "
                    "جرب سؤالاً أبسط أو استخدم Gemini إذا توفر الإنترنت.",
                    trace,
                )

            tool_name = parsed.get("tool")

            # No tool needed — direct answer
            if not tool_name:
                answer = parsed.get("answer", "").strip() or "لم أستطع تكوين إجابة."
                trace.append({"step": step, "type": "final_answer", "content": answer})
                return answer, trace

            args = parsed.get("args", {})
            trace.append({
                "step": step, "type": "tool_call", "tool": tool_name, "args": args,
                "content": f"Calling {tool_name}({json.dumps(args, ensure_ascii=False)})",
            })

            tool_result = self.execute_tool_fn(tool_name, args, db_path=self.db_path)
            # Aggressively truncate — 3B models lose coherence with long context
            result_str = json.dumps(tool_result, ensure_ascii=False)[:600]

            trace.append({
                "step": step, "type": "tool_result", "tool": tool_name,
                "result": tool_result, "content": result_str,
            })

            context_so_far += (
                f"\nTool called: {tool_name}({json.dumps(args, ensure_ascii=False)})\n"
                f"Result: {result_str}\n"
            )

        # Max turns reached — force a final synthesis
        synthesis_prompt = (
            f"{context_so_far}\n\n"
            "Based on all the tool results above, give a final, concise answer "
            "to the original question. Answer in the same language as the question."
        )
        final = ollama_generate(synthesis_prompt, system=self.SYSTEM_PROMPT, temperature=0.2)
        answer = final["text"] or "⚠️ لم أتمكن من إكمال التحليل ضمن الحد الأقصى للخطوات."
        trace.append({"step": step + 1, "type": "forced_final_answer", "content": answer})
        return answer, trace


# ─────────────────────────────────────────────────────────────────────────────
# MARKETING CONTENT (offline replacement for gen_mkt's Gemini path)
# ─────────────────────────────────────────────────────────────────────────────
def ollama_generate_marketing(nar: str, nen: str, cat: str, price: float, ctx: str) -> Dict:
    """
    Offline marketing generation. Honest expectation: noticeably less
    creative than Gemini output. If quality matters more than offline
    capability, keep the existing tmpl() template fallback instead —
    it's actually more consistent than a 3B model's creative writing.
    """
    prompt = (
        f"Write a short bilingual (Arabic + English) marketing message for an Egyptian "
        f"butcher shop. Product: {nen} ({nar}), Category: {cat}, "
        f"Price: {price:.0f} EGP/kg, Occasion: {ctx}. "
        f'Output ONLY JSON: {{"arabic_headline":"","arabic_body":"","arabic_cta":"",'
        f'"english_headline":"","english_body":"","english_cta":""}}'
    )
    result = ollama_generate(prompt, json_mode=True, temperature=0.6)
    parsed = _extract_json(result["text"]) if not result["error"] else None

    if parsed and all(k in parsed for k in ["arabic_headline", "english_headline"]):
        parsed["source"] = f"ollama:{OLLAMA_MODEL}"
        return parsed

    # Graceful degradation to template — recommended default for Scenario C
    # since 3B creative writing quality is inconsistent
    return {"source": "template_fallback", "note": "Ollama output invalid; use tmpl() from app.py instead"}


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT INTEGRATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def show_offline_status_widget():
    """Small sidebar widget showing Ollama connection status. Call from app.py sidebar."""
    import streamlit as st

    available, msg = check_ollama_available()
    if available:
        st.success(f"🔌 وضع عدم الاتصال: {msg}")
    else:
        st.warning(f"🔌 {msg}")
    st.caption(f"Model: {OLLAMA_MODEL} | CPU-forced execution")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Scenario C — Offline Stack Self-Test")
    print("=" * 60)

    available, msg = check_ollama_available()
    print(f"\n1. Ollama connection: {msg}")

    if available:
        print("\n2. Testing simple generation...")
        result = ollama_generate("What is 2+2? Answer in one word.")
        print(f"   Response: {result['text']!r} ({result['elapsed_sec']}s)")

        print("\n3. Testing JSON mode...")
        result = ollama_generate(
            'Return JSON: {"answer": "test"}', json_mode=True, temperature=0.0
        )
        parsed = _extract_json(result["text"])
        print(f"   Parsed: {parsed} ({result['elapsed_sec']}s)")

        print("\n4. Testing RAG answer generation...")
        result = ollama_rag_answer(
            "ما هي نسبة استفادة البقري؟",
            "Baladi cattle yield 70% usable meat from live weight.",
        )
        print(f"   Answer: {result['answer'][:150]}")
        print(f"   Elapsed: {result['elapsed_sec']}s")
    else:
        print("\n⚠️  Skipping generation tests — Ollama not available.")
        print(f"   Setup: ollama pull {OLLAMA_MODEL}")

    print("\n" + "=" * 60)
