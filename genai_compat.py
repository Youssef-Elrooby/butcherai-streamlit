"""
genai_compat.py — Compatibility Shim
=======================================
Provides the old `google.generativeai`-style API on top of the modern
`google-genai` SDK (imported as `google.genai`).

This lets existing code written for the deprecated SDK keep working:

    import genai_compat as genai          # instead of: import genai_compat as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    resp  = model.generate_content(prompt, generation_config=genai.GenerationConfig(...))
    text  = resp.text
"""

from __future__ import annotations

from google import genai as _genai
from google.genai import types as _types


# ─────────────────────────────────────────────────────────────────────────────
# Global API key (set via genai.configure)
# ─────────────────────────────────────────────────────────────────────────────
_api_key: str | None = None


def configure(api_key: str = None, **kwargs) -> None:
    """Set the global API key (mirrors the old SDK's genai.configure)."""
    global _api_key
    if api_key:
        _api_key = api_key


# ─────────────────────────────────────────────────────────────────────────────
# GenerationConfig — alias of the modern GenerateContentConfig
# ─────────────────────────────────────────────────────────────────────────────
class GenerationConfig(_types.GenerateContentConfig):
    """Drop-in replacement for genai.GenerationConfig."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# GenerativeModel — wraps client.models.generate_content
# ─────────────────────────────────────────────────────────────────────────────
class GenerativeModel:
    """Drop-in replacement for genai.GenerativeModel."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _genai.Client(api_key=_api_key)
        return self._client

    def generate_content(self, contents, generation_config=None, **kwargs):
        """Generate content (mirrors model.generate_content)."""
        return self._get_client().models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generation_config,
            **kwargs,
        )

    def count_tokens(self, contents):
        """Best-effort token counting (mirrors model.count_tokens)."""
        try:
            return self._get_client().models.count_tokens(
                model=self.model_name, contents=contents
            )
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Export the same names the old SDK exposed
# ─────────────────────────────────────────────────────────────────────────────
__all__ = ["configure", "GenerationConfig", "GenerativeModel"]
