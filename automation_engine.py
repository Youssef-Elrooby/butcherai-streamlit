"""
ButcherAI — Central Automation Engine
=======================================
Runs all scheduled automations and wires them into one place.

Automations included:
  1. auto_backup              — Database backup + cleanup
  2. auto_daily_summary       — Daily sales/profit summary
  3. auto_low_stock_alerts    — Low-stock detection
  4. auto_holiday_prep        — Holiday procurement preparation
  5. auto_wac_monitor         — Margin squeeze detection
  6. auto_weekly_report       — Weekly business report
  7. auto_marketing_post      — Daily bilingual marketing post
  8. auto_failure_log         — Log agent/RAG failures to queue
  9. auto_prompt_optimize     — Suggest better prompts from failures
 10. auto_test_run            — Run golden tests

Usage:
    python automation_engine.py --task all          # Run all tasks
    python automation_engine.py --task backup       # Run one task
    python automation_engine.py --list              # List tasks
    python automation_engine.py --setup-scheduler   # Create Windows scheduled tasks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

DB_PATH = os.environ.get("DB_PATH", "ButcherAI.db")
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "backups"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "automation_logs"))

# Fix Windows console encoding for Arabic text
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def ensure_automation_tables(db_path: str = DB_PATH):
    """Create automation tables if they don't exist."""
    conn = _connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS automation_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task        TEXT NOT NULL,
            status      TEXT NOT NULL,
            message     TEXT,
            details     TEXT,
            timestamp   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failure_queue (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            feature     TEXT NOT NULL,
            question    TEXT,
            expected    TEXT,
            actual      TEXT,
            error       TEXT,
            count       INTEGER DEFAULT 1,
            status      TEXT DEFAULT 'open',
            timestamp   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS automation_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT NOT NULL,
            title       TEXT NOT NULL,
            message     TEXT NOT NULL,
            severity    TEXT DEFAULT 'info',
            is_read     INTEGER DEFAULT 0,
            timestamp   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_task(task: str, status: str, message: str = "", details: dict = None, db_path: str = DB_PATH):
    """Log an automation task run."""
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT INTO automation_log (task, status, message, details, timestamp) VALUES (?, ?, ?, ?, ?)",
            (task, status, message, json.dumps(details, ensure_ascii=False) if details else "", datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def create_alert(alert_type: str, title: str, message: str, severity: str = "info", db_path: str = DB_PATH):
    """Create an in-app alert."""
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT INTO automation_alerts (type, title, message, severity, is_read, timestamp) VALUES (?, ?, ?, ?, 0, ?)",
            (alert_type, title, message, severity, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: AUTO BACKUP
# ─────────────────────────────────────────────────────────────────────────────
def auto_backup(db_path: str = DB_PATH) -> Dict:
    """Backup database and remove old backups."""
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"ButcherAI_auto_{ts}.db"
    shutil.copy2(db_path, dst)

    cutoff = datetime.now() - timedelta(days=30)
    removed = 0
    for f in BACKUP_DIR.glob("ButcherAI_*.db"):
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            removed += 1

    log_task("backup", "success", f"Backup: {dst.name}, removed {removed} old")
    return {"success": True, "backup": dst.name, "removed": removed}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: AUTO DAILY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def auto_daily_summary(db_path: str = DB_PATH) -> Dict:
    """Generate a daily sales/profit summary."""
    conn = _connect(db_path)
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    today_rev = conn.execute("SELECT COALESCE(SUM(total_price),0) FROM sales WHERE sale_date=?", (today,)).fetchone()[0]
    today_pft = conn.execute("SELECT COALESCE(SUM(gross_profit),0) FROM sales WHERE sale_date=?", (today,)).fetchone()[0]
    today_kg = conn.execute("SELECT COALESCE(SUM(weight_kg),0) FROM sales WHERE sale_date=?", (today,)).fetchone()[0]
    today_tx = conn.execute("SELECT COUNT(*) FROM sales WHERE sale_date=?", (today,)).fetchone()[0]

    yest_rev = conn.execute("SELECT COALESCE(SUM(total_price),0) FROM sales WHERE sale_date=?", (yesterday,)).fetchone()[0]
    change = ((today_rev - yest_rev) / max(yest_rev, 1)) * 100

    top = conn.execute(
        "SELECT p.name_ar, SUM(s.total_price) rev FROM sales s JOIN products p ON s.product_id=p.id "
        "WHERE s.sale_date=? GROUP BY s.product_id ORDER BY rev DESC LIMIT 1", (today,)
    ).fetchone()
    conn.close()

    summary = (
        f"Daily Summary - {today}\n"
        f"----------------------\n"
        f"Revenue: {today_rev:,.0f} EGP ({change:+.1f}% vs yesterday)\n"
        f"Profit:  {today_pft:,.0f} EGP\n"
        f"Weight:  {today_kg:,.1f} kg\n"
        f"Transactions: {today_tx}\n"
        f"Top product: {top[0] if top else 'None'}"
    )

    log_task("daily_summary", "success", summary)
    create_alert("daily_summary", f"Daily Summary {today}", summary, "info")
    return {"success": True, "summary": summary, "revenue": today_rev, "profit": today_pft}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: AUTO LOW STOCK ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def auto_low_stock_alerts(db_path: str = DB_PATH) -> Dict:
    """Detect products below reorder point."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT p.name_ar, COALESCE(i.current_stock_kg,0) stock, p.reorder_point "
        "FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
        "WHERE COALESCE(i.current_stock_kg,0) < p.reorder_point "
        "ORDER BY COALESCE(i.current_stock_kg,0) ASC"
    ).fetchall()
    conn.close()

    if not rows:
        log_task("low_stock", "success", "No low stock items")
        return {"success": True, "count": 0, "items": []}

    items = [{"product": r[0], "stock": round(r[1], 1), "reorder_point": r[2]} for r in rows]
    critical = [i for i in items if i["stock"] == 0]

    message = f"{len(items)} products below reorder point ({len(critical)} empty)"
    log_task("low_stock", "success", message, {"items": items})

    severity = "critical" if critical else "warning"
    create_alert("low_stock", f"Low Stock Alert - {len(items)} items", message, severity)
    return {"success": True, "count": len(items), "critical": len(critical), "items": items}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: AUTO HOLIDAY PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def auto_holiday_prep(db_path: str = DB_PATH) -> Dict:
    """Detect upcoming holidays and prepare procurement plan."""
    from price_forecast import UPCOMING_HOLIDAYS

    today = date.today()
    upcoming = []
    for h in UPCOMING_HOLIDAYS:
        days_away = (h["date"] - today).days
        if 0 <= days_away <= 21:
            upcoming.append({
                "name": h["name_ar"],
                "date": h["date"].isoformat(),
                "days_away": days_away,
                "multiplier": h["multiplier"],
            })

    if not upcoming:
        log_task("holiday_prep", "success", "No holidays within 3 weeks")
        return {"success": True, "upcoming": []}

    for h in upcoming:
        if h["days_away"] <= 14:
            message = (
                f"{h['name']} in {h['days_away']} days. "
                f"Demand x{h['multiplier']}. Increase orders by "
                f"{int((h['multiplier']-1)*100)}% starting now."
            )
            create_alert("holiday_prep", h["name"], message, "critical")
        else:
            create_alert("holiday_prep", h["name"], f"{h['name']} in {h['days_away']} days", "info")

    log_task("holiday_prep", "success", f"{len(upcoming)} upcoming holidays", {"upcoming": upcoming})
    return {"success": True, "upcoming": upcoming}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: AUTO WAC MARGIN MONITOR
# ─────────────────────────────────────────────────────────────────────────────
def auto_wac_monitor(db_path: str = DB_PATH) -> Dict:
    """Detect margin squeeze (WAC rising faster than selling price)."""
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT p.name_ar,
                  ROUND(SUM(s.total_cost)/NULLIF(SUM(s.weight_kg),0),2) recent_wac,
                  p.avg_price
           FROM purchases s JOIN products p ON s.product_id=p.id
           WHERE s.purchase_date >= date('now','-14 days')
           GROUP BY s.product_id HAVING recent_wac > 0"""
    ).fetchall()
    conn.close()

    squeezed = []
    for name, wac, price in rows:
        if price and price > 0:
            margin = (price - wac) / price * 100
            if margin < 15:
                squeezed.append({"product": name, "wac": wac, "price": price, "margin": round(margin, 1)})

    if squeezed:
        message = f"{len(squeezed)} products with margin < 15% (margin squeeze risk)"
        create_alert("wac_monitor", "Margin Squeeze Alert", message, "critical")
        log_task("wac_monitor", "alert", message, {"squeezed": squeezed})
    else:
        log_task("wac_monitor", "success", "No margin squeeze detected")

    return {"success": True, "squeezed": squeezed}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: AUTO WEEKLY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def auto_weekly_report(db_path: str = DB_PATH) -> Dict:
    """Generate a weekly business report."""
    conn = _connect(db_path)
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    rev = conn.execute("SELECT COALESCE(SUM(total_price),0) FROM sales WHERE sale_date>=?", (week_ago,)).fetchone()[0]
    pft = conn.execute("SELECT COALESCE(SUM(gross_profit),0) FROM sales WHERE sale_date>=?", (week_ago,)).fetchone()[0]
    kg = conn.execute("SELECT COALESCE(SUM(weight_kg),0) FROM sales WHERE sale_date>=?", (week_ago,)).fetchone()[0]
    exp = conn.execute("SELECT COALESCE(SUM(amount_egp),0) FROM expenses WHERE expense_date>=?", (week_ago,)).fetchone()[0]

    top = conn.execute(
        "SELECT p.name_ar, SUM(s.total_price) rev FROM sales s JOIN products p ON s.product_id=p.id "
        "WHERE s.sale_date>=? GROUP BY s.product_id ORDER BY rev DESC LIMIT 5", (week_ago,)
    ).fetchall()
    conn.close()

    net = pft - exp
    report = {
        "period": f"Last 7 days (since {week_ago})",
        "revenue": round(rev, 0),
        "gross_profit": round(pft, 0),
        "expenses": round(exp, 0),
        "net_profit": round(net, 0),
        "weight_kg": round(kg, 1),
        "top_products": [{"product": t[0], "revenue": round(t[1], 0)} for t in top],
    }

    log_task("weekly_report", "success", f"Weekly revenue: {rev:,.0f} EGP")
    create_alert("weekly_report", "Weekly Business Report", f"Revenue: {rev:,.0f} | Net: {net:,.0f} EGP", "info")
    return {"success": True, "report": report}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7: AUTO MARKETING POST
# ─────────────────────────────────────────────────────────────────────────────
def auto_marketing_post(db_path: str = DB_PATH, api_key: str = "") -> Dict:
    """Generate a daily bilingual marketing post."""
    conn = _connect(db_path)
    prod = conn.execute(
        "SELECT p.name_ar, p.name_en, p.avg_price, COALESCE(i.current_stock_kg,0) stock "
        "FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
        "WHERE COALESCE(i.current_stock_kg,0) > 20 ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    conn.close()

    if not prod:
        log_task("marketing", "skipped", "No products with stock")
        return {"success": False, "reason": "no products"}

    name_ar, name_en, price, stock = prod

    post_ar = f"عروض اليوم - {name_ar} طازج بسعر {price:.0f} ج/كجم. الكمية محدودة!"
    post_en = f"Today's offer - Fresh {name_en} at {price:.0f} EGP/kg. Limited quantity!"

    log_task("marketing", "success", f"Post for {name_ar}")
    return {"success": True, "arabic": post_ar, "english": post_en, "product": name_ar}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8: AUTO FAILURE LOG
# ─────────────────────────────────────────────────────────────────────────────
def auto_failure_log(feature: str, question: str, expected: str, actual: str, error: str = "", db_path: str = DB_PATH):
    """Log a failure to the failure queue (called by other modules)."""
    conn = _connect(db_path)
    existing = conn.execute(
        "SELECT id, count FROM failure_queue WHERE feature=? AND question=? AND status='open'",
        (feature, question),
    ).fetchone()

    if existing:
        conn.execute("UPDATE failure_queue SET count=count+1, actual=?, timestamp=? WHERE id=?",
                     (actual, datetime.now().isoformat(), existing[0]))
    else:
        conn.execute(
            "INSERT INTO failure_queue (feature, question, expected, actual, error, count, status, timestamp) "
            "VALUES (?, ?, ?, ?, ?, 1, 'open', ?)",
            (feature, question, expected, actual, error, datetime.now().isoformat()),
        )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9: AUTO PROMPT OPTIMIZE
# ─────────────────────────────────────────────────────────────────────────────
def auto_prompt_optimize(db_path: str = DB_PATH, api_key: str = "") -> Dict:
    """Suggest better prompts for frequent failures."""
    conn = _connect(db_path)
    failures = conn.execute(
        "SELECT feature, question, expected, actual, count FROM failure_queue "
        "WHERE status='open' AND count>=2 ORDER BY count DESC LIMIT 5"
    ).fetchall()
    conn.close()

    if not failures:
        log_task("prompt_optimize", "success", "No frequent failures")
        return {"success": True, "suggestions": []}

    if not api_key:
        log_task("prompt_optimize", "skipped", "No API key for suggestions")
        return {"success": False, "reason": "no api key", "failures": len(failures)}

    suggestions = []
    for feature, question, expected, actual, count in failures:
        try:
            from gemini_client import generate_text
            result = generate_text(
                prompt=(
                    f"A {feature} system failed. Question: {question}\n"
                    f"Expected: {expected}, Got: {actual}\n"
                    f"Failed {count} times. Suggest an improved system prompt or retrieval hint."
                ),
                api_key=api_key,
                feature="prompt_optimize",
            )
            if not result.get("error"):
                suggestions.append({
                    "question": question,
                    "count": count,
                    "suggestion": result["text"],
                })
        except Exception:
            continue

    log_task("prompt_optimize", "success", f"{len(suggestions)} suggestions", {"suggestions": suggestions})
    return {"success": True, "suggestions": suggestions}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 10: AUTO TEST RUN
# ─────────────────────────────────────────────────────────────────────────────
def auto_test_run(db_path: str = DB_PATH) -> Dict:
    """Run the golden test suite."""
    try:
        from evaluation_harness import evaluate_rag_retrieval
        summary = evaluate_rag_retrieval(db_path)
        passed = summary.passed
        total = summary.total
        accuracy = summary.accuracy_pct

        status = "success" if accuracy >= 40 else "alert"
        log_task("test_run", status, f"RAG accuracy: {accuracy}% ({passed}/{total})")

        if accuracy < 40:
            create_alert("test_run", "RAG Accuracy Low", f"Accuracy dropped to {accuracy}%", "critical")
        return {"success": True, "accuracy": accuracy, "passed": passed, "total": total}
    except Exception as exc:
        log_task("test_run", "error", str(exc))
        return {"success": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, Callable] = {
    "backup": auto_backup,
    "daily_summary": auto_daily_summary,
    "low_stock": auto_low_stock_alerts,
    "holiday_prep": auto_holiday_prep,
    "wac_monitor": auto_wac_monitor,
    "weekly_report": auto_weekly_report,
    "marketing": auto_marketing_post,
    "test_run": auto_test_run,
    "prompt_optimize": auto_prompt_optimize,
}


def run_task(task_name: str, db_path: str = DB_PATH, api_key: str = "") -> Dict:
    """Run a single automation task."""
    if task_name not in TASKS:
        return {"success": False, "error": f"Unknown task: {task_name}"}

    ensure_automation_tables(db_path)
    start = time.time()
    try:
        result = TASKS[task_name](db_path)
        result["task"] = task_name
        result["elapsed_sec"] = round(time.time() - start, 2)
        return result
    except Exception as exc:
        log_task(task_name, "error", str(exc))
        return {"success": False, "task": task_name, "error": str(exc)}


def run_all(db_path: str = DB_PATH, api_key: str = "") -> List[Dict]:
    """Run all automation tasks."""
    ensure_automation_tables(db_path)
    results = []
    for task_name in TASKS:
        results.append(run_task(task_name, db_path, api_key))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_automation_page(db_path: str = DB_PATH):
    """Streamlit page for the automation engine."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("Automation Engine")
    ensure_automation_tables(db_path)

    # ── Alerts ────────────────────────────────────────────────────────────────
    st.subheader("Active Alerts")
    conn = _connect(db_path)
    alerts = conn.execute(
        "SELECT type, title, message, severity, timestamp FROM automation_alerts "
        "WHERE is_read=0 ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()
    conn.close()

    if alerts:
        for a_type, title, message, severity, ts in alerts:
            color = {"critical": "red", "warning": "orange", "info": "blue"}.get(severity, "blue")
            st.markdown(f"**:{color}[{title}]** — {message}")
    else:
        st.info("No active alerts")

    # ── Run Tasks ─────────────────────────────────────────────────────────────
    st.subheader("Run Automation Tasks")
    c1, c2, c3 = st.columns(3)
    if c1.button("Run All Tasks", type="primary"):
        with st.spinner("Running all automation tasks..."):
            results = run_all(db_path)
        for r in results:
            status = "OK" if r.get("success") else "FAIL"
            st.write(f"{status}: {r.get('task')} ({r.get('elapsed_sec', 0)}s)")

    if c2.button("Backup Now"):
        r = run_task("backup", db_path)
        st.success(f"Backup created: {r.get('backup')}")

    if c3.button("Run Tests"):
        r = run_task("test_run", db_path)
        st.write(f"RAG Accuracy: {r.get('accuracy')}%")

    # ── Task List ─────────────────────────────────────────────────────────────
    st.subheader("Available Tasks")
    st.write(", ".join(TASKS.keys()))

    # ── Automation Log ────────────────────────────────────────────────────────
    st.subheader("Recent Automation Runs")
    conn = _connect(db_path)
    logs = conn.execute(
        "SELECT task, status, message, timestamp FROM automation_log "
        "ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()
    conn.close()

    for task, status, message, ts in logs:
        st.caption(f"{ts[:19]} — **{task}** [{status}]: {message}")

    # ── Failure Queue ─────────────────────────────────────────────────────────
    st.subheader("Failure Queue")
    conn = _connect(db_path)
    failures = conn.execute(
        "SELECT feature, question, expected, actual, count, status FROM failure_queue "
        "WHERE status='open' ORDER BY count DESC LIMIT 20"
    ).fetchall()
    conn.close()

    if failures:
        import pandas as pd
        df = pd.DataFrame(failures, columns=["Feature", "Question", "Expected", "Actual", "Count", "Status"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No open failures")

    # ── Setup Scheduler ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Schedule Automation (Windows)")
    if st.button("Create Windows Scheduled Tasks"):
        import automation_scheduler
        automation_scheduler.setup_all(db_path)
        st.success("Scheduled tasks created! Check Task Scheduler.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ButcherAI Automation Engine")
    parser.add_argument("--task", default="all", help="Task name or 'all'")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    parser.add_argument("--setup-scheduler", action="store_true", help="Create Windows scheduled tasks")

    args = parser.parse_args()

    if args.list:
        print("Available tasks:")
        for name in TASKS:
            print(f"  - {name}")
        return

    if args.setup_scheduler:
        import automation_scheduler
        automation_scheduler.setup_all(args.db)
        return

    ensure_automation_tables(args.db)

    if args.task == "all":
        results = run_all(args.db, os.environ.get("GEMINI_API_KEY", ""))
        for r in results:
            status = "OK" if r.get("success") else "FAIL"
            print(f"[{status}] {r.get('task')}: {r.get('message', r.get('error', ''))}")
    else:
        result = run_task(args.task, args.db, os.environ.get("GEMINI_API_KEY", ""))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
