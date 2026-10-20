"""
ButcherAI — Unified Deployment Scenario Selector
====================================================
Single entry point that wires the correct backend (Gemini / Ollama / Gemini-embed-only)
into app.py based on one environment variable. Avoids scattering if/else scenario
logic across every page — all scenario switching happens here.

Usage in app.py (add near the top, after existing imports):

    from deployment_config import get_active_scenario, get_llm_backend, get_embed_backend
    SCENARIO = get_active_scenario()

Set the scenario in .env:
    BUTCHERAI_SCENARIO=A   # demo/submission (default)
    BUTCHERAI_SCENARIO=B   # production, online
    BUTCHERAI_SCENARIO=C   # production, offline (Ollama)
    BUTCHERAI_SCENARIO=D   # low-spec hardware (4GB RAM)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScenarioConfig:
    name: str
    description: str
    llm_backend: str        # "gemini" | "ollama"
    embed_backend: str      # "sentence-transformers" | "gemini"
    enable_rag: bool
    enable_agent: bool
    enable_ocr: bool
    enable_mcp: bool
    enable_shap: bool
    requires_internet: bool
    min_ram_gb: float


SCENARIOS: Dict[str, ScenarioConfig] = {
    "A": ScenarioConfig(
        name="Demo / Academic Submission",
        description="Committee demo, stable internet, all features showcased",
        llm_backend="gemini", embed_backend="sentence-transformers",
        enable_rag=True, enable_agent=True, enable_ocr=True, enable_mcp=True,
        enable_shap=True, requires_internet=True, min_ram_gb=8.0,
    ),
    "B": ScenarioConfig(
        name="Production — Reliable Internet",
        description="Real shop deployment with stable Wi-Fi/Ethernet",
        llm_backend="gemini", embed_backend="sentence-transformers",
        enable_rag=True, enable_agent=True, enable_ocr=True, enable_mcp=True,
        enable_shap=True, requires_internet=True, min_ram_gb=8.0,
    ),
    "C": ScenarioConfig(
        name="Production — Offline",
        description="No reliable internet — full local LLM stack via Ollama",
        llm_backend="ollama", embed_backend="sentence-transformers",
        enable_rag=True, enable_agent=True, enable_ocr=True, enable_mcp=True,
        enable_shap=True, requires_internet=False, min_ram_gb=16.0,
    ),
    "D": ScenarioConfig(
        name="Low-Spec Hardware",
        description="4GB RAM machines — Gemini embeds instead of local model, OCR/MCP disabled",
        llm_backend="gemini", embed_backend="gemini",
        enable_rag=True, enable_agent=True, enable_ocr=False, enable_mcp=False,
        enable_shap=False, requires_internet=True, min_ram_gb=4.0,
    ),
}


def get_active_scenario() -> ScenarioConfig:
    """Read BUTCHERAI_SCENARIO from environment, default to A (demo)."""
    key = os.environ.get("BUTCHERAI_SCENARIO", "A").strip().upper()
    if key not in SCENARIOS:
        print(f"⚠️  Unknown scenario '{key}', falling back to A (demo).")
        key = "A"
    return SCENARIOS[key]


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_llm_backend(scenario: Optional[ScenarioConfig] = None) -> Callable:
    """
    Returns a unified generate(prompt, system, json_mode) -> dict function,
    regardless of whether the active scenario uses Gemini or Ollama underneath.

    This lets agent/RAG code call one function without knowing which
    scenario is active — the routing happens once, here.
    """
    scenario = scenario or get_active_scenario()

    if scenario.llm_backend == "ollama":
        from phase_c_ollama_offline import ollama_generate

        def _generate(prompt: str, system: str = "", json_mode: bool = False, **kwargs):
            result = ollama_generate(prompt, system=system, json_mode=json_mode)
            return {
                "text": result["text"], "error": result["error"],
                "elapsed_sec": result["elapsed_sec"], "backend": "ollama",
            }
        return _generate

    else:  # gemini
        def _generate(prompt: str, system: str = "", json_mode: bool = False, api_key: str = "", **kwargs):
            import time
            t0 = time.time()
            try:
                import genai_compat as genai
                genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
                model_kwargs = {"system_instruction": system} if system else {}
                model = genai.GenerativeModel("gemini-1.5-flash", **model_kwargs)
                gen_config = genai.GenerationConfig(response_mime_type="application/json") if json_mode else None
                resp = model.generate_content(prompt, generation_config=gen_config)
                return {
                    "text": resp.text.strip(), "error": None,
                    "elapsed_sec": round(time.time() - t0, 2), "backend": "gemini",
                }
            except Exception as exc:
                return {
                    "text": "", "error": str(exc),
                    "elapsed_sec": round(time.time() - t0, 2), "backend": "gemini",
                }
        return _generate


def get_embed_backend(scenario: Optional[ScenarioConfig] = None, gemini_api_key: str = ""):
    """Returns the correct ChromaDB embedding function for the active scenario."""
    scenario = scenario or get_active_scenario()

    if scenario.embed_backend == "gemini":
        from phase_d_lowspec_config import GeminiEmbeddingFunction
        if not gemini_api_key:
            raise ValueError("Scenario D (Gemini embeddings) requires GEMINI_API_KEY")
        return GeminiEmbeddingFunction(api_key=gemini_api_key)
    else:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SIDEBAR WIDGET — shows active scenario + lets admin switch (display only;
# actual switching requires restart since it's an env var)
# ─────────────────────────────────────────────────────────────────────────────
def show_scenario_widget():
    import streamlit as st

    scenario = get_active_scenario()
    key = os.environ.get("BUTCHERAI_SCENARIO", "A").upper()

    with st.expander(f"🔧 السيناريو النشط: {key} — {scenario.name}", expanded=False):
        st.markdown(f"**الوصف:** {scenario.description}")
        c1, c2 = st.columns(2)
        c1.metric("محرك AI", scenario.llm_backend.upper())
        c2.metric("محرك التضمين", "Gemini" if scenario.embed_backend == "gemini" else "محلي")

        c3, c4, c5, c6 = st.columns(4)
        c3.metric("RAG", "✅" if scenario.enable_rag else "❌")
        c4.metric("Agent", "✅" if scenario.enable_agent else "❌")
        c5.metric("OCR", "✅" if scenario.enable_ocr else "❌")
        c6.metric("MCP", "✅" if scenario.enable_mcp else "❌")

        st.caption(
            f"إنترنت مطلوب: {'نعم' if scenario.requires_internet else 'لا'} | "
            f"حد أدنى RAM: {scenario.min_ram_gb} GB"
        )
        st.caption("لتغيير السيناريو: عدّل BUTCHERAI_SCENARIO في ملف .env وأعد تشغيل التطبيق.")


# ─────────────────────────────────────────────────────────────────────────────
# .env TEMPLATES FOR ALL FOUR SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
ENV_TEMPLATES = {
    "A": """
BUTCHERAI_SCENARIO=A
GEMINI_API_KEY=your_key_here
DB_PATH=ButcherAI.db
""",
    "B": """
BUTCHERAI_SCENARIO=B
GEMINI_API_KEY=your_key_here
DB_PATH=D:/ButcherAI_Project/ButcherAI.db
BACKUP_DIR=D:/ButcherAI_Project/backups
""",
    "C": """
BUTCHERAI_SCENARIO=C
DB_PATH=D:/ButcherAI_Project/ButcherAI.db
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_NUM_GPU=0
""",
    "D": """
BUTCHERAI_SCENARIO=D
GEMINI_API_KEY=your_key_here
DB_PATH=ButcherAI.db
BUTCHERAI_LOWSPEC=true
ENABLE_OCR=false
ENABLE_MCP=false
ENABLE_SHAP=false
""",
}


def write_env_template(scenario_key: str, output_path: Optional[str] = None):
    """Generate the .env file for a specific scenario."""
    scenario_key = scenario_key.upper()
    if scenario_key not in ENV_TEMPLATES:
        raise ValueError(f"Unknown scenario: {scenario_key}. Choose from A, B, C, D.")

    output_path = output_path or f".env.scenario_{scenario_key.lower()}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ENV_TEMPLATES[scenario_key].strip() + "\n")
    print(f"✅ Written: {output_path}")
    print(f"   Rename to .env to activate Scenario {scenario_key}.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — generate all templates at once
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ButcherAI Unified Deployment Config")
    print("=" * 60)

    for key, cfg in SCENARIOS.items():
        print(f"\n[{key}] {cfg.name}")
        print(f"    {cfg.description}")
        print(f"    LLM: {cfg.llm_backend} | Embed: {cfg.embed_backend} | "
              f"Internet: {'required' if cfg.requires_internet else 'not required'} | "
              f"Min RAM: {cfg.min_ram_gb} GB")

    print("\n" + "=" * 60)
    print("Writing .env templates for all 4 scenarios...")
    for key in SCENARIOS:
        write_env_template(key)

    print("\nActive scenario (from current environment):")
    active = get_active_scenario()
    print(f"  {active.name} — {active.description}")
