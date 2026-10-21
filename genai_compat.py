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

import base64

from google import genai as _genai
from google.genai import types as _types


def _convert_contents(contents):
    """
    Convert old-SDK style contents (dicts with mime_type/data) into
    the modern SDK's types.Part objects.

    Old:  {"mime_type": "image/jpeg", "data": "<base64>"}
    New:  types.Part.from_bytes(data=b"...", mime_type="image/jpeg")
    """
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        contents = [contents]
    if not isinstance(contents, list):
        return contents

    parts = []
    for item in contents:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            mime = item.get("mime_type", "image/jpeg")
            data = item.get("data", "")
            if isinstance(data, str):
                # base64-encoded image data
                try:
                    raw = base64.b64decode(data)
                    parts.append(_types.Part.from_bytes(data=raw, mime_type=mime))
                except Exception:
                    parts.append(_types.Part.from_bytes(data=data.encode(), mime_type=mime))
            elif isinstance(data, bytes):
                parts.append(_types.Part.from_bytes(data=data, mime_type=mime))
            else:
                parts.append(item)
        else:
            parts.append(item)
    return parts


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
            contents=_convert_contents(contents),
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
