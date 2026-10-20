"""
ButcherAI — Unified Gemini Client (google.genai migration)
=================================================================
Replaces the deprecated google-generativeai package across the entire
project with the current google.genai SDK, plus adds automatic model
fallback so a future Google model retirement (confirmed to happen mid-
project once already) doesn't require manually hunting for a new model
name again.

Two call patterns, matching what the new SDK actually recommends:
  - generate_text() / generate_json() — one-shot calls (RAG answers,
    marketing content). Uses client.models.generate_content() directly.
  - AgentChatSession — multi-turn tool-calling (the Agent). Uses
    client.chats.create() + send_message(), which is the SDK's own
    recommended pattern for function-calling and correctly handles
    thought_signature internally — replacing the manual history-patching
    workaround used with the old deprecated SDK.

Integration — replace every occurrence of:
    import genai_compat as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

with:
    from gemini_client import generate_text
    result = generate_text(prompt, api_key=api_key)
    resp_text = result["text"]
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Callable, Union

# Ranked by preference: try the first, fall back to the next on failure.
# Confirmed working: gemini-flash-lite-latest, gemini-flash-latest.
# Confirmed NOT working on free tier: gemini-1.5-flash (404, retired),
# gemini-2.0-flash / gemini-2.0-flash-lite (quota=0), gemini-2.5-pro (quota=0).
MODEL_FALLBACK_CHAIN = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite-001",
]

VISION_MODEL_FALLBACK_CHAIN = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]


def _get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _try_models(fn: Callable[[str], Dict], chain: List[str]) -> Dict:
    """
    Tries each model in the fallback chain in order, returning the first
    successful result. If all fail, returns the last error with a full
    trace of what was attempted — so a failure is diagnosable, not silent.
    """
    attempts = []
    for model_name in chain:
        result = fn(model_name)
        attempts.append({"model": model_name, "error": result.get("error")})
        if result.get("error") is None:
            result["model_used"] = model_name
            result["fallback_attempts"] = attempts
            return result
    return {
        "text": "", "error": "All models in fallback chain failed",
        "fallback_attempts": attempts, "model_used": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER  — Exercise 5.1
# ─────────────────────────────────────────────────────────────────────────────
class CircuitBreaker:
    """
    Prevents cascading failures when the Gemini API is down.

    States:
      CLOSED    — normal operation, requests pass through
      OPEN      — failing, requests rejected immediately
      HALF_OPEN — testing if service recovered

    After N consecutive failures, the circuit opens for a cooldown period.
    After the cooldown, one trial call is allowed (HALF_OPEN).
    If the trial succeeds, the circuit closes. If it fails, it re-opens.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_sec: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.half_open_max_calls = half_open_max_calls

        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Current circuit state (with automatic transition from OPEN to HALF_OPEN)."""
        if self._state == "OPEN" and self._opened_at:
            if time.time() - self._opened_at >= self.cooldown_sec:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if a request is allowed through the circuit."""
        state = self.state
        if state == "CLOSED":
            return True
        elif state == "OPEN":
            return False
        elif state == "HALF_OPEN":
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False

    def record_success(self):
        """Record a successful call — resets failures and closes the circuit."""
        self._consecutive_failures = 0
        self._state = "CLOSED"
        self._half_open_calls = 0

    def record_failure(self):
        """Record a failed call — may open the circuit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.time()

    def get_status(self) -> Dict:
        """Get current circuit status for monitoring."""
        return {
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "cooldown_sec": self.cooldown_sec,
            "opened_at": self._opened_at,
        }


# Global circuit breaker instance (shared across all API calls)
_circuit_breaker = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    return _circuit_breaker


# ─────────────────────────────────────────────────────────────────────────────
# ONE-SHOT TEXT GENERATION (RAG answers, marketing content)
# ─────────────────────────────────────────────────────────────────────────────
def generate_text(
    prompt: str,
    api_key: str,
    system_instruction: str = "",
    temperature: float = 0.3,
    model: Optional[str] = None,
    db_path: Optional[str] = None,
    feature: str = "general",
) -> Dict:
    if not api_key:
        return {"text": "", "error": "No API key provided", "model_used": None, "elapsed_sec": 0.0}

    # Circuit breaker check
    cb = get_circuit_breaker()
    if not cb.allow_request():
        return {
            "text": "",
            "error": f"Circuit breaker is OPEN. Service temporarily unavailable. "
                     f"Try again in {cb.cooldown_sec}s.",
            "model_used": None,
            "elapsed_sec": 0.0,
            "circuit_breaker": cb.get_status(),
        }

    from google.genai import types
    chain = [model] if model else MODEL_FALLBACK_CHAIN

    def _attempt(model_name: str) -> Dict:
        t0 = time.time()
        try:
            client = _get_client(api_key)
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
            )
            resp = client.models.generate_content(
                model=model_name, contents=prompt, config=config,
            )
            elapsed = round(time.time() - t0, 2)
            cb.record_success()
            if db_path:
                _log_observability(feature, model_name, db_path, resp, elapsed, success=True)
            return {"text": resp.text.strip(), "error": None, "elapsed_sec": elapsed}
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            cb.record_failure()
            if db_path:
                _log_observability(feature, model_name, db_path, None, elapsed, success=False, error=str(exc))
            return {"text": "", "error": str(exc), "elapsed_sec": elapsed}

    return _try_models(_attempt, chain)


def generate_json(
    prompt: str,
    api_key: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    model: Optional[str] = None,
    db_path: Optional[str] = None,
    feature: str = "general",
) -> Dict:
    """Same as generate_text but requests structured JSON output."""
    if not api_key:
        return {"text": "", "error": "No API key provided", "model_used": None, "elapsed_sec": 0.0}

    from google.genai import types
    chain = [model] if model else MODEL_FALLBACK_CHAIN

    def _attempt(model_name: str) -> Dict:
        t0 = time.time()
        try:
            client = _get_client(api_key)
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
            )
            resp = client.models.generate_content(
                model=model_name, contents=prompt, config=config,
            )
            elapsed = round(time.time() - t0, 2)
            if db_path:
                _log_observability(feature, model_name, db_path, resp, elapsed, success=True)
            return {"text": resp.text.strip(), "error": None, "elapsed_sec": elapsed}
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            if db_path:
                _log_observability(feature, model_name, db_path, None, elapsed, success=False, error=str(exc))
            return {"text": "", "error": str(exc), "elapsed_sec": elapsed}

    return _try_models(_attempt, chain)


# ─────────────────────────────────────────────────────────────────────────────
# VISION (OCR — image + prompt)
# ─────────────────────────────────────────────────────────────────────────────
def generate_vision_json(
    image_bytes: bytes,
    prompt: str,
    api_key: str,
    mime_type: str = "image/jpeg",
    model: Optional[str] = None,
    db_path: Optional[str] = None,
    feature: str = "ocr",
) -> Dict:
    if not api_key:
        return {"text": "", "error": "No API key provided", "model_used": None, "elapsed_sec": 0.0}

    from google.genai import types
    chain = [model] if model else VISION_MODEL_FALLBACK_CHAIN

    def _attempt(model_name: str) -> Dict:
        t0 = time.time()
        try:
            client = _get_client(api_key)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            config = types.GenerateContentConfig(response_mime_type="application/json")
            resp = client.models.generate_content(
                model=model_name, contents=[image_part, prompt], config=config,
            )
            elapsed = round(time.time() - t0, 2)
            if db_path:
                _log_observability(feature, model_name, db_path, resp, elapsed, success=True)
            return {"text": resp.text.strip(), "error": None, "elapsed_sec": elapsed}
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            if db_path:
                _log_observability(feature, model_name, db_path, None, elapsed, success=False, error=str(exc))
            return {"text": "", "error": str(exc), "elapsed_sec": elapsed}

    return _try_models(_attempt, chain)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TURN CHAT WITH TOOL CALLING (the Agent)
# ─────────────────────────────────────────────────────────────────────────────
class AgentChatSession:
    """
    Wraps client.chats.create() — the SDK's own recommended pattern for
    multi-turn conversations with function calling. Replaces the manual
    history-list management and thought_signature workaround needed with
    the deprecated SDK — the Chat object manages that state internally.
    """

    def __init__(self, api_key: str, system_instruction: str, tools: List, model: Optional[str] = None):
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.tools = tools
        self.model = model or MODEL_FALLBACK_CHAIN[0]
        self._client = None
        self._chat = None
        self._last_working_model = None

    def _ensure_chat(self, model_name: str):
        from google.genai import types
        if self._client is None:
            self._client = _get_client(self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=self.tools,
        )
        self._chat = self._client.chats.create(model=model_name, config=config)
        self._last_working_model = model_name

    def send(self, message: Union[str, List], db_path: Optional[str] = None) -> Dict:
        chain = [self.model] + [m for m in MODEL_FALLBACK_CHAIN if m != self.model]
        last_error = None
        max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "2"))
        retry_delay = float(os.environ.get("GEMINI_RETRY_DELAY", "1"))

        for model_name in chain:
            for attempt in range(max_retries):
                t0 = time.time()
                try:
                    if self._chat is None or self._last_working_model != model_name:
                        self._ensure_chat(model_name)
                    resp = self._chat.send_message(message)
                    elapsed = round(time.time() - t0, 2)
                    if db_path:
                        _log_observability("agent", model_name, db_path, resp, elapsed, success=True)
                    return {"response": resp, "error": None, "model_used": model_name, "elapsed_sec": elapsed}
                except Exception as exc:
                    elapsed = round(time.time() - t0, 2)
                    error_str = str(exc)

                    # Check if it's an SSL/network error that we should retry
                    is_ssl_error = any(kw in error_str.lower() for kw in [
                        "ssl", "eof", "connection", "timeout", "network", "unexpected"
                    ])

                    if db_path:
                        _log_observability("agent", model_name, db_path, None, elapsed, success=False, error=error_str)

                    self._chat = None
                    last_error = error_str

                    # Only retry SSL/network errors, not other errors
                    if is_ssl_error and attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        break  # Move to next model

        return {"response": None, "error": last_error, "model_used": None, "elapsed_sec": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVABILITY INTEGRATION (reuses llm_observability.py if present)
# ─────────────────────────────────────────────────────────────────────────────
def _log_observability(feature, model_name, db_path, resp, elapsed, success, error=None):
    try:
        from llm_observability import ensure_observability_table
        import sqlite3
        from datetime import datetime

        ensure_observability_table(db_path)
        input_tokens = output_tokens = None
        if resp is not None and hasattr(resp, "usage_metadata") and resp.usage_metadata:
            input_tokens = getattr(resp.usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(resp.usage_metadata, "candidates_token_count", None)

        conn = sqlite3.connect(db_path)
        conn.execute(
            """INSERT INTO llm_calls_log
               (timestamp, feature, model, input_tokens, output_tokens,
                elapsed_sec, estimated_cost_usd, success, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(datetime.now()), feature, model_name, input_tokens, output_tokens,
             elapsed, 0.0, 1 if success else 0, (error or "")[:500]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    import os
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run a live test.")
    else:
        print("Testing generate_text with automatic fallback...")
        result = generate_text("Reply with exactly one word: OK", api_key=api_key)
        print(f"  Model used: {result.get('model_used')}")
        print(f"  Response: {result['text']!r}")
        print(f"  Fallback attempts: {result.get('fallback_attempts')}")
