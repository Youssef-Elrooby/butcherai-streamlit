"""
ButcherAI — Monitoring Dashboard (Exercise 4.4)
=================================================
Real-time monitoring of AI system performance:
- Agent success rate over time
- Average latency per feature
- Cost per day/week/month
- Evaluation pass rate trend
- Most common failure modes

Integration into app.py:
  elif page == "📊 مراقبة الأداء":
      from monitoring_dashboard import show_monitoring_page
      show_monitoring_page(db_path=DB)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List

DB_PATH = os.environ.get("DB_PATH", "ButcherAI.db")


def get_llm_stats(db_path: str = DB_PATH) -> Dict:
    """Get LLM call statistics from the observability log."""
    try:
        conn = sqlite3.connect(db_path)

        # Total calls
        total = conn.execute("SELECT COUNT(*) FROM llm_calls_log").fetchone()[0]

        # Success rate
        success = conn.execute(
            "SELECT COUNT(*) FROM llm_calls_log WHERE success = 1"
        ).fetchone()[0]

        # Average latency
        avg_latency = conn.execute(
            "SELECT AVG(elapsed_sec) FROM llm_calls_log"
        ).fetchone()[0] or 0

        # Total cost
        total_cost = conn.execute(
            "SELECT SUM(estimated_cost_usd) FROM llm_calls_log"
        ).fetchone()[0] or 0

        # Calls by feature
        by_feature = conn.execute(
            "SELECT feature, COUNT(*) as cnt, AVG(elapsed_sec) as avg_lat, "
            "SUM(estimated_cost_usd) as cost "
            "FROM llm_calls_log GROUP BY feature ORDER BY cnt DESC"
        ).fetchall()

        # Calls by model
        by_model = conn.execute(
            "SELECT model, COUNT(*) as cnt, AVG(elapsed_sec) as avg_lat "
            "FROM llm_calls_log GROUP BY model ORDER BY cnt DESC"
        ).fetchall()

        # Recent failures
        failures = conn.execute(
            "SELECT timestamp, feature, model, error_message "
            "FROM llm_calls_log WHERE success = 0 "
            "ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()

        # Daily trend (last 7 days)
        daily = conn.execute(
            "SELECT date(timestamp) as day, COUNT(*) as cnt, "
            "AVG(elapsed_sec) as avg_lat, SUM(estimated_cost_usd) as cost "
            "FROM llm_calls_log "
            "WHERE timestamp >= date('now', '-7 days') "
            "GROUP BY day ORDER BY day"
        ).fetchall()

        conn.close()

        return {
            "total_calls": total,
            "success_rate": round(success / max(total, 1) * 100, 1),
            "avg_latency_sec": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 4),
            "by_feature": [
                {"feature": r[0], "count": r[1], "avg_latency": round(r[2] or 0, 2), "cost": round(r[3] or 0, 4)}
                for r in by_feature
            ],
            "by_model": [
                {"model": r[0], "count": r[1], "avg_latency": round(r[2] or 0, 2)}
                for r in by_model
            ],
            "recent_failures": [
                {"timestamp": r[0], "feature": r[1], "model": r[2], "error": r[3]}
                for r in failures
            ],
            "daily_trend": [
                {"day": r[0], "count": r[1], "avg_latency": round(r[2] or 0, 2), "cost": round(r[3] or 0, 4)}
                for r in daily
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_evaluation_trend(db_path: str = DB_PATH) -> List[Dict]:
    """Get evaluation results from test reports."""
    from pathlib import Path
    reports_dir = Path("test_reports")
    if not reports_dir.exists():
        return []

    trend = []
    for f in sorted(reports_dir.glob("test_report_*.json"))[-10:]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            trend.append({
                "timestamp": data.get("timestamp", ""),
                "total": len(data.get("tests", {})),
                "passed": sum(1 for t in data.get("tests", {}).values() if t.get("status") == "PASS"),
                "failed": sum(1 for t in data.get("tests", {}).values() if t.get("status") == "FAIL"),
            })
        except Exception:
            continue
    return trend


def show_monitoring_page(db_path: str = DB_PATH):
    """Streamlit page for monitoring dashboard."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("📊 مراقبة أداء الذكاء الاصطناعي")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "مراقبة في الوقت الحقيقي: معدل النجاح، الزمن، التكلفة، والأخطاء."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── LLM Stats ─────────────────────────────────────────────────────────────
    stats = get_llm_stats(db_path)

    if "error" in stats:
        st.error(f"Error loading stats: {stats['error']}")
        return

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الاستدعاءات", stats["total_calls"])
    c2.metric("معدل النجاح", f"{stats['success_rate']}%")
    c3.metric("متوسط الزمن", f"{stats['avg_latency_sec']}s")
    c4.metric("التكلفة الإجمالية", f"${stats['total_cost_usd']}")

    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 الاستدعاءات حسب الميزة", "📊 النموذج", "❌ الأخطاء"])

    with tab1:
        if stats["by_feature"]:
            import pandas as pd
            df = pd.DataFrame(stats["by_feature"])
            st.bar_chart(df.set_index("feature")["count"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("لا توجد بيانات بعد")

    with tab2:
        if stats["by_model"]:
            import pandas as pd
            df = pd.DataFrame(stats["by_model"])
            st.bar_chart(df.set_index("model")["count"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("لا توجد بيانات بعد")

    with tab3:
        if stats["recent_failures"]:
            for f in stats["recent_failures"]:
                with st.expander(f"❌ {f['timestamp'][:19]} — {f['feature']}"):
                    st.write(f"النموذج: {f['model']}")
                    st.write(f"الخطأ: {f['error']}")
        else:
            st.success("لا توجد أخطاء مسجلة")

    # ── Daily trend ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 الاتجاه اليومي (آخر 7 أيام)")
    if stats["daily_trend"]:
        import pandas as pd
        df = pd.DataFrame(stats["daily_trend"])
        st.line_chart(df.set_index("day")["count"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("لا توجد بيانات يومية بعد")

    # ── Evaluation trend ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧪 نتائج التقييم")
    eval_trend = get_evaluation_trend(db_path)
    if eval_trend:
        import pandas as pd
        df = pd.DataFrame(eval_trend)
        st.line_chart(df.set_index("timestamp")[["passed", "failed"]])
    else:
        st.info("لا توجد تقارير تقييم بعد. شغل: python run_tests.py")

    # ── Refresh button ───────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 تحديث البيانات"):
        st.rerun()
