"""
ButcherAI Scenario D — Low-Spec Hardware Configuration (4GB RAM)
====================================================================
For machines too weak to run the full RAG + OCR + Agent stack simultaneously.

Two changes from the standard build:
  1. Replace local sentence-transformers (510MB RAM, 470MB download) with
     the Gemini Embeddings API — same RAG quality, zero local model footprint.
  2. Feature flags to cleanly disable OCR/MCP/Agent from the sidebar entirely,
     rather than having them silently fail or eat RAM nobody asked to spend.

This is NOT a degraded RAG — retrieval quality is equal or better than the
local model, since Gemini's embedding models are generally stronger than
MiniLM-L12. The tradeoff is purely: this requires internet for RAG now,
where the standard build's RAG retrieval works offline.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS — set these via environment variables or .env
# ─────────────────────────────────────────────────────────────────────────────
LOWSPEC_MODE       = os.environ.get("BUTCHERAI_LOWSPEC", "false").lower() == "true"
ENABLE_RAG         = os.environ.get("ENABLE_RAG", "true").lower() == "true"
ENABLE_AGENT       = os.environ.get("ENABLE_AGENT", "true").lower() == "true"
ENABLE_OCR         = os.environ.get("ENABLE_OCR", "false" if LOWSPEC_MODE else "true").lower() == "true"
ENABLE_MCP         = os.environ.get("ENABLE_MCP", "false" if LOWSPEC_MODE else "true").lower() == "true"
ENABLE_SHAP        = os.environ.get("ENABLE_SHAP", "false" if LOWSPEC_MODE else "true").lower() == "true"


def get_active_feature_set() -> Dict[str, bool]:
    """Returns the current feature flag state — call this from app.py sidebar to display."""
    return {
        "Low-Spec Mode"        : LOWSPEC_MODE,
        "RAG (Knowledge Base)" : ENABLE_RAG,
        "Agentic AI"           : ENABLE_AGENT,
        "OCR Import"           : ENABLE_OCR,
        "MCP Server"           : ENABLE_MCP,
        "SHAP Explanations"    : ENABLE_SHAP,
    }


def estimated_ram_usage_mb() -> int:
    """Rough estimate of current configuration's RAM footprint, for display."""
    base = 1800 + 280 + 320  # OS + Python/Streamlit + Pandas/NumPy/Plotly
    base += 55  # LightGBM always loaded
    if ENABLE_SHAP:
        base += 110
    if ENABLE_RAG:
        base += 140  # ChromaDB client always needed for RAG
        base += 0 if LOWSPEC_MODE else 510  # local embed model skipped in lowspec
    if ENABLE_AGENT:
        base += 75
    if ENABLE_MCP:
        base += 65
    if ENABLE_OCR:
        base += 2325  # PyTorch + OpenCV + ONNX, only if OCR page actually opened
    return base


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EMBEDDINGS — drop-in replacement for sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────
class GeminiEmbeddingFunction:
    """
    ChromaDB-compatible embedding function using Gemini's embedding API
    instead of a local sentence-transformers model.

    This class implements ChromaDB's EmbeddingFunction protocol
    (a __call__ method taking list[str] -> list[list[float]]),
    so it's a true drop-in replacement in phase1_rag_engine.py's _embed() method.
    """

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model   = model
        self._genai  = None

    def _client(self):
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
        return self._genai

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this with a batch of texts, expects a batch of embeddings."""
        genai = self._client()
        embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])
            except Exception as exc:
                # ChromaDB expects a fixed-size vector even on failure —
                # return a zero vector rather than crashing the whole batch
                print(f"[GeminiEmbedding] Failed on one document: {exc}")
                embeddings.append([0.0] * 768)  # text-embedding-004 dimension
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Some ChromaDB versions call embed_query separately for search queries."""
        genai = self._client()
        try:
            result = genai.embed_content(
                model=self.model, content=text, task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as exc:
            print(f"[GeminiEmbedding] Query embed failed: {exc}")
            return [0.0] * 768


def build_lowspec_rag_engine(db_path: str, gemini_api_key: str):
    """
    Constructs a RAG engine identical to phase1_rag_engine.ButcherAIRAG,
    but with GeminiEmbeddingFunction instead of sentence-transformers.

    This is the recommended integration point: monkey-patch the _embed()
    method on the existing ButcherAIRAG class rather than duplicating
    the entire RAG engine.
    """
    from phase1_rag_engine import ButcherAIRAG

    if not gemini_api_key:
        raise ValueError(
            "Low-spec RAG mode requires a Gemini API key — "
            "local embeddings are disabled to save RAM, so there is no offline fallback."
        )

    rag = ButcherAIRAG(db_path=db_path)

    # Monkey-patch the embedding function to skip loading sentence-transformers
    def _lowspec_embed(self=rag):
        if self._embed_fn is None:
            self._embed_fn = GeminiEmbeddingFunction(api_key=gemini_api_key)
        return self._embed_fn

    rag._embed = _lowspec_embed
    return rag


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR INTEGRATION — feature flag display + toggle UI
# ─────────────────────────────────────────────────────────────────────────────
def show_lowspec_sidebar_widget():
    """Call from app.py sidebar to show current resource configuration."""
    import streamlit as st

    with st.expander("⚙️ إعدادات الأداء (Low-Spec Mode)", expanded=LOWSPEC_MODE):
        features = get_active_feature_set()
        ram_est  = estimated_ram_usage_mb()

        st.metric("الذاكرة المقدرة (RAM)", f"{ram_est:,} MB (~{ram_est/1024:.1f} GB)")

        for name, enabled in features.items():
            icon = "🟢" if enabled else "⚪"
            st.markdown(f"{icon} {name}")

        if LOWSPEC_MODE:
            st.info(
                "🔧 وضع الأجهزة الضعيفة مُفعّل: RAG يستخدم Gemini Embeddings API "
                "بدلاً من النموذج المحلي (يوفر 510 MB رام لكن يتطلب إنترنت)."
            )
        else:
            st.caption("لتفعيل وضع الأجهزة الضعيفة: BUTCHERAI_LOWSPEC=true في .env")


def apply_lowspec_page_filter(pages: List[str]) -> List[str]:
    """
    Filters the sidebar page list based on active feature flags.
    Call this in app.py right after building the `pages` list, before st.radio().
    """
    filtered = list(pages)

    if not ENABLE_RAG:
        filtered = [p for p in filtered if "قاعدة المعرفة" not in p]
    if not ENABLE_AGENT:
        filtered = [p for p in filtered if "وكيل الذكاء" not in p]
    if not ENABLE_OCR:
        filtered = [p for p in filtered if "الدفتر الورقي" not in p]

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# .env TEMPLATE FOR SCENARIO D
# ─────────────────────────────────────────────────────────────────────────────
LOWSPEC_ENV_TEMPLATE = """
# ButcherAI Scenario D — Low-Spec Hardware (.env additions)
BUTCHERAI_LOWSPEC=true
ENABLE_RAG=true
ENABLE_AGENT=true
ENABLE_OCR=false
ENABLE_MCP=false
ENABLE_SHAP=false
GEMINI_API_KEY=your_key_here
"""


def write_lowspec_env_template(path: str = ".env.lowspec"):
    """Generate the .env template file for Scenario D deployment."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(LOWSPEC_ENV_TEMPLATE.strip() + "\n")
    print(f"✅ Written: {path}")
    print("   Rename to .env or merge into existing .env to activate low-spec mode.")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Scenario D — Low-Spec Configuration Check")
    print("=" * 60)

    print(f"\nLow-spec mode active: {LOWSPEC_MODE}")
    print(f"\nFeature flags:")
    for name, val in get_active_feature_set().items():
        print(f"  {'✅' if val else '❌'} {name}")

    ram = estimated_ram_usage_mb()
    print(f"\nEstimated RAM usage: {ram:,} MB (~{ram/1024:.2f} GB)")

    if ram > 4096:
        print(f"⚠️  WARNING: Estimated usage exceeds typical 4GB low-spec target.")
        print(f"   Consider setting ENABLE_MCP=false or ENABLE_AGENT=false.")
    else:
        print(f"✅ Fits within a 4GB RAM budget with headroom.")

    print("\nWriting .env.lowspec template...")
    write_lowspec_env_template()
