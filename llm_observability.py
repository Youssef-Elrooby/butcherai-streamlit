"""
ButcherAI — LLM Observability Layer
=========================================
Wraps every Gemini API call across the app (RAG, Agent, Marketing) with
automatic logging of: latency, token counts, estimated cost, success/failure.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Rough cost table (USD per 1M tokens)
# ─────────────────────────────────────────────────────────────────────────────
COST_PER_1M_TOKENS = {
    "gemini-flash-lite-latest": {"input": 0.0375, "output": 0.15},
    "gemini-flash-latest":      {"input": 0.075,  "output": 0.30},
    "default":                  {"input": 0.05,   "output": 0.20},
}


def ensure_observability_table(db_path: str) -> None:
    """Create the llm_calls_log table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            feature         TEXT NOT NULL,
            model           TEXT NOT NULL,
            input_tokens    INTEGER,
            output_tokens   INTEGER,
            elapsed_sec     REAL,
            estimated_cost_usd REAL,
            success         INTEGER NOT NULL,
            error_message   TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_log_ts ON llm_calls_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_log_feature ON llm_calls_log(feature)")
    conn.commit()
    conn.close()


class LLMCallTracker:
    """Context manager returned by track_llm_call()."""

    def __init__(self, feature: str, model: str, db_path: str):
        self.feature = feature
        self.model = model
        self.db_path = db_path
        self.start_time = 0.0
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.success = True
        self.error_message: Optional[str] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def record_response(self, gemini_response) -> None:
        try:
            meta = gemini_response.usage_metadata
            self.input_tokens = meta.prompt_token_count
            self.output_tokens = meta.candidates_token_count
        except AttributeError:
            pass 

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.time() - self.start_time, 3)

        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)[:500]

        cost_table = COST_PER_1M_TOKENS.get(self.model, COST_PER_1M_TOKENS["default"])
        est_cost = 0.0
        if self.input_tokens and self.output_tokens:
            est_cost = round(
                (self.input_tokens / 1_000_000 * cost_table["input"]) +
                (self.output_tokens / 1_000_000 * cost_table["output"]),
                6,
            )

        try:
            ensure_observability_table(self.db_path)
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO llm_calls_log
                   (timestamp, feature, model, input_tokens, output_tokens,
                    elapsed_sec, estimated_cost_usd, success, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(datetime.now()), self.feature, self.model,
                 self.input_tokens, self.output_tokens, elapsed,
                 est_cost, 1 if self.success else 0, self.error_message),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass 

        return False 


@contextmanager
def track_llm_call(feature: str, model: str, db_path: str):
    tracker = LLMCallTracker(feature, model, db_path)
    tracker.__enter__()
    try:
        yield tracker
    except Exception as exc:
        tracker.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        tracker.__exit__(None, None, None)


def show_llm_observability_page(db_path: str):
    import streamlit as st
    import pandas as pd

    st.title("📊 مراقبة استخدام الذكاء الاصطناعي")
    ensure_observability_table(db_path)

    days = st.slider("الفترة (يوم):", 1, 90, 7)
    conn = sqlite3.connect(db_path)

    summary = pd.read_sql_query(
        f"""SELECT feature, COUNT(*) calls,
                   ROUND(AVG(elapsed_sec), 2) avg_latency_sec,
                   SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) errors,
                   ROUND(SUM(COALESCE(estimated_cost_usd,0)), 4) total_cost_usd,
                   SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0)) total_tokens
            FROM llm_calls_log
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY feature ORDER BY calls DESC""",
        conn,
    )

    if summary.empty:
        st.info("لا توجد استدعاءات مسجلة بعد في هذه الفترة.")
        conn.close()
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الاستدعاءات", int(summary.calls.sum()))
    c2.metric("متوسط زمن الاستجابة", f"{summary.avg_latency_sec.mean():.2f}s")
    c3.metric("إجمالي الأخطاء", int(summary.errors.sum()))
    c4.metric("التكلفة التقديرية (لو مدفوعة)", f"${summary.total_cost_usd.sum():.4f}")

    st.subheader("📋 حسب الميزة")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    daily = pd.read_sql_query(
        f"""SELECT date(timestamp) day, COUNT(*) calls,
                   ROUND(AVG(elapsed_sec), 2) avg_latency
            FROM llm_calls_log
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY day ORDER BY day""",
        conn,
    )
    conn.close()

    if not daily.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily.day, y=daily.calls, name="عدد الاستدعاءات"))
        fig.update_layout(height=300, title="الاستدعاءات اليومية")
        st.plotly_chart(fig, use_container_width=True)