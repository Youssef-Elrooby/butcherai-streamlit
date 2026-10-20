"""
ButcherAI Phase 2 — Agentic AI Engine
======================================
An autonomous multi-step agent that reasons over shop data, calls tools,
and produces actionable decisions — not just answers.

Architecture:
  Gemini function-calling API (ReAct pattern)
  10 domain-specific tools connected directly to SQLite + LightGBM
  Autonomous tool chaining: the model decides which tools to call and in what order
  Session memory: conversation history persisted in st.session_state

What the agent can do end-to-end:
  "Should I buy more lamb tomorrow?"
    → check_inventory(lamb) → run_forecast(lamb, 3) → get_wac(lamb)
    → get_holiday_info(tomorrow) → synthesize procurement recommendation

  "What was my profit this week minus expenses?"
    → query_sales(last_7_days) → get_expenses(last_7_days) → calculate_profit_loss()

  "Generate a Friday promotion for ground meat"
    → query_sales(ground_meat) → get_wac(ground_meat) → generate_marketing()

Install (already in requirements if using v9.1):
  pip install google-generativeai

Integration into app.py:
  elif page == "🤖 وكيل الذكاء الاصطناعي":
      from phase2_agent_page import show_agent_page
      show_agent_page(db_path=DB, gemini_api_key=GEMINI_API_ENV,
                      ent_model=ENT_MODEL, ent_features=ENT_FEATURES,
                      get_mult_fn=get_mult)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH      = os.environ.get("DB_PATH", "ButcherAI.db")
MAX_TURNS    = 12    # maximum agent reasoning steps per query
GEMINI_MODEL = "gemini-flash-lite-latest"

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS  (Gemini function-calling schema)
# Each tool maps directly to a Python function below.
# ─────────────────────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "query_sales",
        "description": (
            "Query historical sales data from the database. "
            "Returns aggregated metrics: revenue, profit, weight, transaction count, margin. "
            "Use this when the user asks about past performance, trends, or product comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Number of days back from today to include (e.g. 7, 30, 90)",
                },
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name to filter (e.g. 'مفروم', 'فيليه'). Leave empty for all products.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["product", "day", "week", "category"],
                    "description": "How to group the results",
                },
            },
            "required": ["days_back"],
        },
    },
    {
        "name": "check_inventory",
        "description": (
            "Get current real-time inventory levels for all products or a specific product. "
            "Returns stock_kg, reorder_point, alert_level (OK/LOW/CRITICAL/EMPTY). "
            "Always call this before making procurement recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name to check. Leave empty for all products.",
                },
                "only_low_stock": {
                    "type": "boolean",
                    "description": "If true, return only products below reorder point",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_forecast",
        "description": (
            "Run the Enterprise LightGBM demand forecast for a specific product. "
            "Returns predicted demand in kg for the next N days with seasonal multipliers. "
            "Use this to compute procurement quantities and detect upcoming demand spikes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name to forecast (e.g. 'لحم بلدي', 'مفروم')",
                },
                "days": {
                    "type": "integer",
                    "description": "Forecast horizon in days (1-14)",
                },
            },
            "required": ["product_name_ar", "days"],
        },
    },
    {
        "name": "get_wac_analysis",
        "description": (
            "Compute Weighted Average Cost (WAC) per product from purchase history. "
            "Returns WAC, current selling price, actual margin, and margin gap vs target. "
            "Use when assessing profitability or pricing decisions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name. Leave empty for all products.",
                },
                "days_back": {
                    "type": "integer",
                    "description": "Days of purchase history to include (default 30)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_expenses",
        "description": (
            "Query operational expense records (utilities, packaging, salaries, transport, etc.). "
            "Returns total expenses by category and period. "
            "Use when computing net profit (gross profit minus expenses)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Number of days back to include expenses",
                },
                "category": {
                    "type": "string",
                    "description": "Expense category to filter. Leave empty for all categories.",
                },
            },
            "required": ["days_back"],
        },
    },
    {
        "name": "calculate_profit_loss",
        "description": (
            "MUST BE CALLED IMMEDIATELY if the user asks for 'صافي الربح', 'net profit', or 'احسب صافي الربح'. "
            "Calculates total revenue, expenses, gross profit, and net profit. "
            "DO NOT USE query_sales FOR NET PROFIT QUESTIONS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Period for the P&L calculation",
                },
            },
            "required": ["days_back"],
        },
    },
    {
        "name": "get_procurement_recommendation",
        "description": (
            "MUST BE CALLED IMMEDIATELY if the user asks 'ما الذي يجب أن أشتريه', 'توصيات الشراء', or what to buy. "
            "Generates AI procurement recommendations and buy lists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name. Leave empty for full shop recommendations.",
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Days to forecast for computing purchase quantity (default 7)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_holiday_info",
        "description": (
            "MUST BE CALLED IMMEDIATELY whenever the user asks about upcoming holidays, Eid Al-Adha date, or remaining days ('متى عيد الأضحى', 'كم باقي'). "
            "Retrieves Hijri holiday dates, days remaining, and demand multipliers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (default: today)",
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "Number of days to look ahead (default 14)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_supplier_performance",
        "description": (
            "MUST BE CALLED IMMEDIATELY if the user asks 'قارن بين الموردين', 'أداء الموردين', or asks to compare suppliers. "
            "Analyzes supplier pricing, consistency, and performance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Days of purchase history to analyze (default 60)",
                },
                "supplier_name": {
                    "type": "string",
                    "description": "Specific supplier name to filter. Leave empty for all suppliers.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_marketing_content",
        "description": (
            "Generate bilingual (Arabic + English) marketing content for a product promotion. "
            "Returns: Arabic headline, body, CTA + English equivalents. "
            "Use when the user wants to create WhatsApp messages, social posts, or price boards."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name_ar": {
                    "type": "string",
                    "description": "Arabic product name to promote",
                },
                "context": {
                    "type": "string",
                    "enum": ["general", "ramadan", "eid", "weekend", "clearance"],
                    "description": "Marketing context/occasion",
                },
                "special_price": {
                    "type": "number",
                    "description": "Special offer price in EGP/kg (optional)",
                },
            },
            "required": ["product_name_ar", "context"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS  (each returns a JSON-serializable dict)
# ─────────────────────────────────────────────────────────────────────────────

def _db(db_path: str = DB_PATH):
    return sqlite3.connect(db_path, check_same_thread=False)


def _tool_query_sales(
    days_back: int,
    product_name_ar: str = "",
    group_by: str = "product",
    db_path: str = DB_PATH,
) -> Dict:
    try:
        conn   = _db(db_path)
        where  = f"sale_date >= date('now','-{days_back} days')"
        pfilter = ""
        if product_name_ar.strip():
            pfilter = f" AND (s.product_name_ar LIKE '%{product_name_ar}%' OR p.name_ar LIKE '%{product_name_ar}%')"

        if group_by == "product":
            sql = (
                f"SELECT p.name_ar, p.name_en, p.category, "
                f"COUNT(*) tx, SUM(s.weight_kg) kg, SUM(s.total_price) rev, "
                f"SUM(s.gross_profit) pft, AVG(s.profit_margin_percent) mg "
                f"FROM sales s JOIN products p ON s.product_id=p.id "
                f"WHERE s.{where}{pfilter} GROUP BY s.product_id ORDER BY rev DESC LIMIT 15"
            )
            rows = conn.execute(sql).fetchall()
            data = [
                {"product_ar": r[0], "product_en": r[1], "category": r[2],
                 "transactions": r[3], "weight_kg": round(r[4] or 0, 1),
                 "revenue_egp": round(r[5] or 0, 0), "gross_profit_egp": round(r[6] or 0, 0),
                 "avg_margin_pct": round(r[7] or 0, 1)}
                for r in rows
            ]
        elif group_by in ("day", "week"):
            fmt  = "%Y-%m-%d" if group_by == "day" else "%Y-W%W"
            sql  = (
                f"SELECT strftime('{fmt}', s.sale_date) period, "
                f"COUNT(*) tx, SUM(s.weight_kg) kg, SUM(s.total_price) rev "
                f"FROM sales s JOIN products p ON s.product_id=p.id "
                f"WHERE s.{where}{pfilter} GROUP BY period ORDER BY period"
            )
            rows = conn.execute(sql).fetchall()
            data = [{"period": r[0], "transactions": r[1], "weight_kg": round(r[2] or 0, 1),
                     "revenue_egp": round(r[3] or 0, 0)} for r in rows]
        else:  # category
            sql  = (
                f"SELECT p.category, COUNT(*) tx, SUM(s.weight_kg) kg, "
                f"SUM(s.total_price) rev, AVG(s.profit_margin_percent) mg "
                f"FROM sales s JOIN products p ON s.product_id=p.id "
                f"WHERE s.{where}{pfilter} GROUP BY p.category ORDER BY rev DESC"
            )
            rows = conn.execute(sql).fetchall()
            data = [{"category": r[0], "transactions": r[1], "weight_kg": round(r[2] or 0, 1),
                     "revenue_egp": round(r[3] or 0, 0), "avg_margin_pct": round(r[4] or 0, 1)}
                    for r in rows]

        # Totals
        tot = conn.execute(
            f"SELECT COUNT(*) tx, SUM(weight_kg) kg, SUM(total_price) rev, "
            f"SUM(gross_profit) pft, AVG(profit_margin_percent) mg "
            f"FROM sales s JOIN products p ON s.product_id=p.id WHERE s.{where}{pfilter}"
        ).fetchone()
        conn.close()
        return {
            "period_days"      : days_back,
            "product_filter"   : product_name_ar or "all",
            "group_by"         : group_by,
            "rows"             : data,
            "totals"           : {
                "transactions"  : tot[0] or 0,
                "weight_kg"     : round(tot[1] or 0, 1),
                "revenue_egp"   : round(tot[2] or 0, 0),
                "gross_profit"  : round(tot[3] or 0, 0),
                "avg_margin_pct": round(tot[4] or 0, 1),
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_check_inventory(
    product_name_ar: str = "",
    only_low_stock: bool = False,
    db_path: str = DB_PATH,
) -> Dict:
    try:
        conn   = _db(db_path)
        pfilter = f"AND p.name_ar LIKE '%{product_name_ar}%'" if product_name_ar.strip() else ""
        low_filter = "AND COALESCE(i.current_stock_kg,0) < p.reorder_point" if only_low_stock else ""
        sql = (
            f"SELECT p.name_ar, p.name_en, p.category, "
            f"COALESCE(i.current_stock_kg,0) stock, p.reorder_point, i.last_updated "
            f"FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
            f"WHERE 1=1 {pfilter} {low_filter} "
            f"ORDER BY COALESCE(i.current_stock_kg,0) ASC LIMIT 30"
        )
        rows = conn.execute(sql).fetchall()
        conn.close()

        items = []
        for r in rows:
            stk  = r[3] or 0
            rpt  = r[4] or 10
            if stk == 0:
                alert = "EMPTY"
            elif stk < rpt * 0.3:
                alert = "CRITICAL"
            elif stk < rpt:
                alert = "LOW"
            else:
                alert = "OK"
            items.append({
                "product_ar"  : r[0], "product_en": r[1], "category": r[2],
                "stock_kg"    : round(stk, 1), "reorder_point_kg": r[4],
                "alert_level" : alert, "last_updated": r[5],
            })
        empty    = sum(1 for x in items if x["alert_level"] == "EMPTY")
        critical = sum(1 for x in items if x["alert_level"] == "CRITICAL")
        low      = sum(1 for x in items if x["alert_level"] == "LOW")
        return {"items": items, "summary": {"EMPTY": empty, "CRITICAL": critical, "LOW": low,
                                             "OK": len(items) - empty - critical - low}}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_run_forecast(
    product_name_ar: str,
    days: int,
    db_path: str = DB_PATH,
    ent_model=None,
    ent_features=None,
    get_mult_fn=None,
) -> Dict:
    """Run Enterprise LightGBM forecast or return heuristic if model unavailable."""
    try:
        import numpy as np

        conn = _db(db_path)
        # Find product
        prod = conn.execute(
            "SELECT id, name_ar, name_en FROM products WHERE name_ar LIKE ?",
            (f"%{product_name_ar}%",),
        ).fetchone()
        if not prod:
            conn.close()
            return {"error": f"Product not found: {product_name_ar}"}

        pid = prod[0]
        df_rows = conn.execute(
            f"SELECT sale_date, SUM(weight_kg) kg, AVG(unit_price) price, AVG(unit_cost) cost "
            f"FROM sales WHERE product_id={pid} GROUP BY sale_date ORDER BY sale_date"
        ).fetchall()
        inv_row = conn.execute(
            "SELECT current_stock_kg FROM inventory WHERE product_id=?", (pid,)
        ).fetchone()
        conn.close()

        if len(df_rows) < 6:
            return {"error": "Insufficient history (need 6+ sales days)"}

        kg_vals = [r[1] for r in df_rows]
        recent7 = sum(kg_vals[-7:]) / min(7, len(kg_vals))

        forecast = []
        last_date = datetime.strptime(df_rows[-1][0], "%Y-%m-%d").date()

        for step in range(1, days + 1):
            fd   = last_date + timedelta(days=step)
            mult = 1.0
            lbl  = "يوم عادي"
            if get_mult_fn:
                mult, lbl = get_mult_fn(fd)

            # Use model if available, otherwise heuristic
            if ent_model is not None and ent_features is not None:
                try:
                    tail = kg_vals[-6:] if len(kg_vals) >= 6 else kg_vals
                    l1   = tail[-1] if tail else recent7
                    l2   = tail[-2] if len(tail) >= 2 else l1
                    r3   = sum(tail[-3:]) / 3 if len(tail) >= 3 else l1
                    r6   = sum(tail[-6:]) / 6 if len(tail) >= 6 else r3
                    inv_s = float(inv_row[0]) if inv_row else 0.0
                    da    = r6 if r6 > 0 else 1.0
                    doy   = fd.timetuple().tm_yday
                    # Build minimal 18-feature vector (matches base model)
                    feat_vec = [l1, l2, r3, r3, r3, r6, inv_s/da,
                                (r3-r6)/(r6+1e-5), 0, 0, 0, inv_s-da*3,
                                0, (r6-r3)/(r3+1e-5), r3/(r6+1e-5),
                                float(np.sin(2*3.14159*doy/7)),
                                float(np.cos(2*3.14159*doy/7)), 0.0]
                    # Pad or trim to match ent_features length
                    nf = len(ent_features)
                    if len(feat_vec) < nf:
                        feat_vec += [0.0] * (nf - len(feat_vec))
                    X    = np.array([feat_vec[:nf]])
                    pred = float(max(0, ent_model.predict(X)[0])) * mult
                except Exception:
                    pred = recent7 * mult
            else:
                pred = recent7 * mult

            std  = (sum((k - recent7)**2 for k in kg_vals[-14:]) / max(1, len(kg_vals[-14:]))) ** 0.5
            forecast.append({
                "date"        : str(fd),
                "predicted_kg": round(pred, 1),
                "lower_kg"    : round(max(0, pred - 1.44 * std), 1),
                "upper_kg"    : round(pred + 1.44 * std, 1),
                "season"      : lbl,
                "multiplier"  : mult,
            })
            kg_vals.append(pred)

        total_pred = sum(f["predicted_kg"] for f in forecast)
        max_day    = max(forecast, key=lambda x: x["predicted_kg"])
        return {
            "product_ar"       : prod[1],
            "product_en"       : prod[2],
            "forecast_days"    : days,
            "daily_forecast"   : forecast,
            "total_predicted_kg": round(total_pred, 1),
            "peak_day"         : max_day,
            "current_stock_kg" : round(float(inv_row[0]) if inv_row else 0, 1),
        }
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()[-500:]}


def _tool_get_wac(
    product_name_ar: str = "",
    days_back: int = 30,
    db_path: str = DB_PATH,
) -> Dict:
    try:
        conn    = _db(db_path)
        pfilter = f"AND product_ar LIKE '%{product_name_ar}%'" if product_name_ar.strip() else ""
        rows = conn.execute(
            f"SELECT product_ar, "
            f"ROUND(SUM(total_cost)/NULLIF(SUM(weight_kg),0),2) wac, "
            f"SUM(weight_kg) kg, COUNT(*) orders, MAX(purchase_date) last_pur "
            f"FROM purchases WHERE purchase_date >= date('now','-{days_back} days') {pfilter} "
            f"GROUP BY product_ar ORDER BY kg DESC LIMIT 15"
        ).fetchall()

        items = []
        for r in rows:
            # Get current selling price from products
            prod = conn.execute(
                "SELECT avg_price, avg_margin FROM products WHERE name_ar LIKE ?",
                (f"%{r[0]}%",),
            ).fetchone()
            price     = prod[0] if prod else 0
            tgt_mg    = prod[1] if prod else 25
            wac_val   = r[1] or 0
            act_mg    = round((price - wac_val) / price * 100, 1) if price > 0 else 0
            items.append({
                "product_ar"   : r[0], "wac_egp_per_kg": r[1],
                "sell_price"   : price, "actual_margin_pct": act_mg,
                "target_margin": tgt_mg, "margin_gap": round(act_mg - tgt_mg, 1),
                "total_kg_purchased": round(r[2] or 0, 1),
                "orders"       : r[3], "last_purchase": r[4],
                "status"       : "✅ Healthy" if act_mg >= 20 else "⚠️ Warning" if act_mg >= 10 else "🔴 Loss risk",
            })
        conn.close()
        return {"period_days": days_back, "products": items}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_expenses(
    days_back: int,
    category: str = "",
    db_path: str = DB_PATH,
) -> Dict:
    try:
        conn    = _db(db_path)
        cfilter = f"AND category LIKE '%{category}%'" if category.strip() else ""
        # Check if expenses table exists
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'"
        ).fetchone()
        if not tbl:
            conn.close()
            return {"error": "Expenses table not found. Run upgrade_v91.py first.", "total_egp": 0}

        rows = conn.execute(
            f"SELECT category, SUM(amount_egp) total, COUNT(*) entries "
            f"FROM expenses WHERE expense_date >= date('now','-{days_back} days') {cfilter} "
            f"GROUP BY category ORDER BY total DESC"
        ).fetchall()
        total = conn.execute(
            f"SELECT SUM(amount_egp) FROM expenses "
            f"WHERE expense_date >= date('now','-{days_back} days') {cfilter}"
        ).fetchone()[0] or 0
        conn.close()
        return {
            "period_days"   : days_back,
            "by_category"   : [{"category": r[0], "total_egp": round(r[1], 0), "entries": r[2]} for r in rows],
            "total_egp"     : round(total, 0),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_calculate_profit_loss(days_back: int, db_path: str = DB_PATH) -> Dict:
    """
    MUST BE CALLED IMMEDIATELY if the user asks for 'صافي الربح', 'net profit', or 'احسب صافي الربح'.
    Calculates total revenue, expenses, gross profit, and net profit.
    DO NOT USE query_sales FOR NET PROFIT QUESTIONS.
    """
    try:
        conn  = _db(db_path)
        where = f"sale_date >= date('now','-{days_back} days')"
        r = conn.execute(
            f"SELECT COALESCE(SUM(total_price),0), COALESCE(SUM(gross_profit),0), "
            f"COALESCE(AVG(profit_margin_percent),0), COALESCE(SUM(weight_kg),0) "
            f"FROM sales WHERE {where}"
        ).fetchone()
        revenue      = r[0] or 0
        gross_profit = r[1] or 0
        avg_margin   = r[2] or 0

        # Check expenses table
        exp_total = 0
        try:
            exp_row  = conn.execute(
                f"SELECT COALESCE(SUM(amount_egp),0) FROM expenses "
                f"WHERE expense_date >= date('now','-{days_back} days')"
            ).fetchone()
            exp_total = exp_row[0] or 0
        except Exception:
            pass

        net_profit = gross_profit - exp_total
        net_margin = round(net_profit / revenue * 100, 1) if revenue > 0 else 0

        # Compare to previous same period
        prev_rev = conn.execute(
            f"SELECT COALESCE(SUM(total_price),0) FROM sales "
            f"WHERE sale_date >= date('now','-{days_back*2} days') "
            f"AND sale_date < date('now','-{days_back} days')"
        ).fetchone()[0] or 0
        rev_change = round((revenue - prev_rev) / max(prev_rev, 1) * 100, 1)

        conn.close()
        return {
            "period_days"       : days_back,
            "revenue_egp"       : round(revenue, 0),
            "gross_profit_egp"  : round(gross_profit, 0),
            "gross_margin_pct"  : round(avg_margin, 1),
            "operating_expenses": round(exp_total, 0),
            "net_profit_egp"    : round(net_profit, 0),
            "net_margin_pct"    : net_margin,
            "revenue_vs_prev_period_pct": rev_change,
            "verdict": (
                "✅ Profitable" if net_profit > 0 else
                "⚠️ Breaking even" if net_profit > -1000 else
                "🔴 Operating at a loss"
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_procurement_recommendation(
    product_name_ar: str = "",
    forecast_days: int = 7,
    db_path: str = DB_PATH,
    ent_model=None,
    ent_features=None,
    get_mult_fn=None,
) -> Dict:
    """
    MUST BE CALLED IMMEDIATELY if the user asks 'ما الذي يجب أن أشتريه', 'توصيات الشراء', or what to buy.
    Generates AI procurement recommendations and buy lists.
    """
    try:
        conn = _db(db_path)
        pfilter = f"AND name_ar LIKE '%{product_name_ar}%'" if product_name_ar.strip() else ""
        prods = conn.execute(
            f"SELECT id, name_ar, name_en, avg_price, avg_cost, avg_margin, reorder_point "
            f"FROM products WHERE 1=1 {pfilter} LIMIT 20"
        ).fetchall()
        conn.close()

        recs = []
        for prod in prods:
            pid, nar, nen, price, cost, tgt_mg, rpt = prod

            # Get current stock
            conn2   = _db(db_path)
            inv_row = conn2.execute(
                "SELECT current_stock_kg FROM inventory WHERE product_id=?", (pid,)
            ).fetchone()
            conn2.close()
            stock = float(inv_row[0]) if inv_row else 0.0

            # Get forecast
            fc = _tool_run_forecast(
                nar, forecast_days, db_path, ent_model, ent_features, get_mult_fn
            )
            if "error" in fc:
                continue
            total_demand = fc["total_predicted_kg"]
            peak_mult    = fc["peak_day"]["multiplier"] if fc["peak_day"] else 1.0

            # Recommendation quantity: demand * 1.1 buffer - current stock
            rec_qty = max(0, round(total_demand * 1.1 - stock, 1))

            # Scoring
            ratio    = total_demand / max(rpt, 1)
            urgency  = 10 if ratio >= 2 else 8 if ratio >= 1.5 else 5 if ratio >= 1 else 2
            margin_s = min((float(tgt_mg or 20)) / 5, 10)
            score    = round(urgency * 0.5 + margin_s * 0.3 + min(ratio * 3, 10) * 0.2, 1)
            priority = "HIGH" if score >= 7 else "MEDIUM" if score >= 4.5 else "LOW"

            roi = round((float(price) - float(cost)) * total_demand / max(float(cost) * total_demand, 1) * 100, 1)

            recs.append({
                "product_ar"        : nar, "product_en": nen,
                "current_stock_kg"  : stock,
                "predicted_demand_kg": round(total_demand, 1),
                "recommended_buy_kg": rec_qty,
                "priority"          : priority,
                "score_out_of_10"   : score,
                "estimated_roi_pct" : roi,
                "peak_multiplier"   : peak_mult,
                "action"            : {
                    "HIGH"  : "اشترِ فوراً / Buy immediately",
                    "MEDIUM": "اشترِ خلال 48 ساعة / Buy within 48h",
                    "LOW"   : "المخزون كافٍ / Stock is adequate",
                }[priority],
            })

        recs.sort(key=lambda x: x["score_out_of_10"], reverse=True)
        high = sum(1 for r in recs if r["priority"] == "HIGH")
        return {
            "forecast_days": forecast_days,
            "total_products_analyzed": len(recs),
            "high_priority"  : high,
            "recommendations": recs,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_holiday_info(
    start_date: str = "",
    days_ahead: int = 14,
    get_mult_fn=None,
) -> Dict:
    """
    MUST BE CALLED IMMEDIATELY whenever the user asks about upcoming holidays, Eid Al-Adha date, or remaining days ('متى عيد الأضحى', 'كم باقي').
    Retrieves Hijri holiday dates, days remaining, and demand multipliers.
    """
    try:
        if not start_date:
            start_date = str(date.today())
        sd   = datetime.strptime(start_date, "%Y-%m-%d").date()
        rows = []
        for i in range(days_ahead):
            d = sd + timedelta(days=i)
            if get_mult_fn:
                mult, lbl = get_mult_fn(d)
            else:
                wd   = d.weekday()
                mult = 1.3 if wd == 4 else 1.15 if wd == 5 else 1.1 if wd == 6 else 0.9 if wd == 0 else 1.0
                lbl  = {4: "الجمعة", 5: "السبت", 6: "الأحد", 0: "الإثنين"}.get(wd, "يوم عادي")
            rows.append({
                "date": str(d), "weekday": d.strftime("%A"),
                "season_label": lbl, "demand_multiplier": mult,
                "demand_category": (
                    "PEAK"   if mult >= 2.5 else
                    "HIGH"   if mult >= 1.4 else
                    "ABOVE_NORMAL" if mult > 1.0 else
                    "BELOW_NORMAL"
                ),
            })

        peak_days = [r for r in rows if r["demand_multiplier"] >= 1.3]
        return {
            "start_date"  : start_date,
            "days_ahead"  : days_ahead,
            "calendar"    : rows,
            "peak_days"   : peak_days,
            "avg_multiplier": round(sum(r["demand_multiplier"] for r in rows) / len(rows), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_supplier_performance(
    days_back: int = 60,
    supplier_name: str = "",
    db_path: str = DB_PATH,
) -> Dict:
    """
    MUST BE CALLED IMMEDIATELY if the user asks 'قارن بين الموردين', 'أداء الموردين', or asks to compare suppliers.
    Analyzes supplier pricing, consistency, and performance.
    """
    try:
        conn    = _db(db_path)
        sfilter = f"AND supplier_name LIKE '%{supplier_name}%'" if supplier_name.strip() else ""
        rows = conn.execute(
            f"SELECT supplier_name, COUNT(*) orders, SUM(weight_kg) kg, "
            f"ROUND(AVG(cost_per_kg),0) avg_cost, "
            f"ROUND(MIN(cost_per_kg),0) min_cost, "
            f"ROUND(MAX(cost_per_kg),0) max_cost, "
            f"MAX(purchase_date) last_order "
            f"FROM purchases "
            f"WHERE supplier_name IS NOT NULL "
            f"AND purchase_date >= date('now','-{days_back} days') {sfilter} "
            f"GROUP BY supplier_name ORDER BY kg DESC"
        ).fetchall()
        conn.close()

        suppliers = []
        for r in rows:
            price_range = r[5] - r[4] if r[4] and r[5] else 0
            consistency = round((1 - price_range / max(r[3], 1)) * 100, 1)
            suppliers.append({
                "supplier"        : r[0], "total_orders": r[1],
                "total_kg_bought" : round(r[2] or 0, 1),
                "avg_cost_per_kg" : r[3], "min_cost": r[4], "max_cost": r[5],
                "price_consistency_pct": consistency,
                "last_order_date" : r[6],
                "reliability_score": round(min(r[1] / 10, 1) * 40 + consistency * 0.6, 1),
            })

        suppliers.sort(key=lambda x: x["reliability_score"], reverse=True)
        return {"period_days": days_back, "suppliers": suppliers}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_generate_marketing(
    product_name_ar: str,
    context: str = "general",
    special_price: float = 0.0,
    api_key: str = "",
    db_path: str = DB_PATH,
) -> Dict:
    try:
        conn = _db(db_path)
        prod = conn.execute(
            "SELECT name_ar, name_en, avg_price, category FROM products WHERE name_ar LIKE ?",
            (f"%{product_name_ar}%",),
        ).fetchone()
        conn.close()
        if not prod:
            return {"error": f"Product not found: {product_name_ar}"}

        nar, nen, price, cat = prod
        price = special_price if special_price > 0 else price

        ctx_ar = {"ramadan": "في رمضان المبارك", "eid": "في العيد السعيد",
                  "weekend": "يوم الجمعة", "clearance": "عرض محدود", "general": "جودة لا تُضاهى"}
        cta_ar = {"ramadan": "احجز كميتك الآن", "eid": "اجعل عيدك أحلى",
                  "weekend": "عرض الجمعة الحصري", "clearance": "الكمية محدودة",
                  "general": "اتصل بنا الآن"}
        ctx_en = {"ramadan": "This Ramadan", "eid": "This Eid", "weekend": "Friday Special",
                  "clearance": "Limited Offer", "general": "Premium Quality"}
        cta_en = {"ramadan": "Reserve your quantity now", "eid": "Make your Eid memorable",
                  "weekend": "Friday exclusive deal", "clearance": "Limited time only",
                  "general": "Call us now"}

        # Try Gemini for richer content
        if api_key:
            try:
                import google.generativeai as genai
                import re
                genai.configure(api_key=api_key)
                m = genai.GenerativeModel("gemini-flash-lite-latest")
                prompt = (
                    f"Generate bilingual marketing content for Egyptian butcher shop. "
                    f"Product: {nen} ({nar}), Category: {cat}, "
                    f"Price: {price:.0f} EGP/kg, Occasion: {context}. "
                    f"Return ONLY valid JSON with keys: "
                    f"arabic_headline, arabic_body, arabic_cta, english_headline, english_body, english_cta"
                )
                resp = m.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(response_mime_type="application/json"),
                )
                result = json.loads(resp.text)
                result["product_ar"] = nar
                result["product_en"] = nen
                result["price_egp"]  = price
                result["source"]     = "gemini_ai"
                return result
            except Exception:
                pass

        # Template fallback
        return {
            "product_ar"       : nar, "product_en": nen, "price_egp": price,
            "arabic_headline"  : f"🥩 {ctx_ar.get(context, '')} — {nar}",
            "arabic_body"      : f"أجود {nar} طازج يومياً. سعر اليوم: {price:.0f} ج/كجم. {cta_ar.get(context, '')}",
            "arabic_cta"       : "📞 اتصل أو تعال للمحل الآن",
            "english_headline" : f"🥩 {ctx_en.get(context, '')} — {nen}",
            "english_body"     : f"Fresh {nen} daily. Today's price: {price:.0f} EGP/kg. {cta_en.get(context, '')}",
            "english_cta"      : "📞 Call or visit our shop now",
            "source"           : "template_fallback",
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def execute_tool(
    tool_name: str,
    args: Dict,
    db_path: str = DB_PATH,
    ent_model=None,
    ent_features=None,
    get_mult_fn=None,
    gemini_api_key: str = "",
) -> Dict:
    """Route a tool call from the agent to its Python implementation."""
    fns = {
        "query_sales"               : lambda: _tool_query_sales(db_path=db_path, **args),
        "check_inventory"           : lambda: _tool_check_inventory(db_path=db_path, **args),
        "run_forecast"              : lambda: _tool_run_forecast(
                                          db_path=db_path, ent_model=ent_model,
                                          ent_features=ent_features, get_mult_fn=get_mult_fn, **args),
        "get_wac_analysis"          : lambda: _tool_get_wac(db_path=db_path, **args),
        "get_expenses"              : lambda: _tool_get_expenses(db_path=db_path, **args),
        "calculate_profit_loss"     : lambda: _tool_calculate_profit_loss(db_path=db_path, **args),
        "get_procurement_recommendation": lambda: _tool_get_procurement_recommendation(
                                          db_path=db_path, ent_model=ent_model,
                                          ent_features=ent_features, get_mult_fn=get_mult_fn, **args),
        "get_holiday_info"          : lambda: _tool_get_holiday_info(get_mult_fn=get_mult_fn, **args),
        "get_supplier_performance"  : lambda: _tool_get_supplier_performance(db_path=db_path, **args),
        "generate_marketing_content": lambda: _tool_generate_marketing(
                                          api_key=gemini_api_key, db_path=db_path, **args),
    }
    fn = fns.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn()
    except TypeError as exc:
        return {"error": f"Tool argument error for {tool_name}: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT CORE  (ReAct loop with Gemini function calling)
# ─────────────────────────────────────────────────────────────────────────────
class ButcherAIAgent:
    SYSTEM_PROMPT = (
        "You are ButcherAI, an autonomous business intelligence agent for an Egyptian butcher shop. "
        "You have access to 10 tools that query live shop data, run demand forecasts, "
        "analyze profitability, and generate marketing content. "
        "\n\nBehavior rules:\n"
        "1. ALWAYS use tools to get real data before answering quantitative questions.\n"
        "2. Chain tools logically: check_inventory -> run_forecast -> get_procurement_recommendation.\n"
        "3. For P&L questions: call calculate_profit_loss (not query_sales).\n"
        "4. Be specific with numbers (kg, EGP) never vague.\n"
        "5. Answer in the same language the user used (Arabic or English).\n"
        "6. After tool results, synthesize a clear, actionable recommendation.\n"
        "7. Never hallucinate numbers — only report what tools return.\n"
        "8. If the user asks about holidays, ALWAYS call get_holiday_info first.\n"
        "9. If the user asks what to buy, ALWAYS call get_procurement_recommendation.\n"
        "10. If the user asks to compare suppliers, ALWAYS call get_supplier_performance.\n"
        "\n\nArabic Language Rules:\n"
        "- When the user writes in Arabic (Egyptian dialect), respond in Arabic.\n"
        "- Use Egyptian Arabic terms: مخزون (stock), مبيعات (sales), ربح (profit), شراء (purchase).\n"
        "- For product names, use the Arabic names from the database.\n"
        "- For numbers, use Western digits (0-9) for clarity.\n"
        "- For currency, use EGP or جنيه.\n"
        "\n\nCRITICAL DIRECTIVES FOR FUNCTION CALLING:\n"
        "- If user asks 'متى عيد الأضحى القادم وكم أيام باقية؟' -> YOU MUST CALL get_holiday_info\n"
        "- If user asks 'ما الذي يجب أن أشتريه غداً؟' -> YOU MUST CALL get_procurement_recommendation\n"
        "- If user asks 'احسب صافي الربح هذا الشهر' -> YOU MUST CALL calculate_profit_loss\n"
        "- If user asks 'قارن بين الموردين لدي' -> YOU MUST CALL get_supplier_performance\n"
        "DO NOT ANSWER DIRECTLY. ALWAYS CALL THE TOOL FIRST."
    )

    def __init__(
        self,
        api_key     : str,
        db_path     : str = DB_PATH,
        ent_model   = None,
        ent_features= None,
        get_mult_fn = None,
    ):
        self.api_key       = api_key
        self.db_path       = db_path
        self.ent_model     = ent_model
        self.ent_features  = ent_features
        self.get_mult_fn   = get_mult_fn
        self.reasoning_trace: List[Dict] = []
        # AgentChatSession owns conversation history + thought_signature
        # internally — we no longer manage a manual self.history list.
        self._session          = None
        self._session_api_key  = None
        self._gemini_tools     = None

    def reset(self):
        self._session         = None
        self._session_api_key = None
        self.reasoning_trace  = []

    def _create_plan(self, user_message: str) -> List[Dict]:
        """
        Plan-and-Execute: Ask Gemini to create a structured execution plan
        before starting the ReAct loop.

        Returns a list of steps: [{"step": 1, "tool": "check_inventory", "args": {...}, "reason": "..."}]
        """
        try:
            from google.genai import types
            from gemini_client import AgentChatSession

            # Create a separate planning session (no tools needed)
            plan_session = AgentChatSession(
                api_key=self.api_key,
                system_instruction=(
                    "You are a planning assistant. Given a user query, create a step-by-step "
                    "execution plan using the available tools. "
                    "Return ONLY valid JSON with this exact format:\n"
                    '{"steps": [{"step": 1, "tool": "tool_name", "args": {"param": "value"}, "reason": "why this step is needed"}]}\n\n'
                    "Available tools:\n"
                    "- query_sales: Query historical sales data (needs days_back, optional product_name_ar, group_by)\n"
                    "- check_inventory: Check stock levels (optional product_name_ar, only_low_stock)\n"
                    "- run_forecast: Run demand forecast (needs product_name_ar, days)\n"
                    "- get_wac_analysis: Compute weighted average cost (optional product_name_ar, days_back)\n"
                    "- get_expenses: Query expenses (needs days_back, optional category)\n"
                    "- calculate_profit_loss: Calculate P&L (needs days_back)\n"
                    "- get_procurement_recommendation: Get buy recommendations (optional product_name_ar, forecast_days)\n"
                    "- get_holiday_info: Get holiday calendar (optional start_date, days_ahead)\n"
                    "- get_supplier_performance: Compare suppliers (optional days_back, supplier_name)\n"
                    "- generate_marketing_content: Generate marketing (needs product_name_ar, context, optional special_price)\n\n"
                    "Rules:\n"
                    "1. Plan 2-5 steps for complex queries, 1-2 for simple ones\n"
                    "2. Each step must use the correct tool with valid arguments\n"
                    "3. Think about dependencies: check inventory BEFORE forecasting\n"
                    "4. For P&L: use calculate_profit_loss (not query_sales)\n"
                    "5. For procurement: use get_procurement_recommendation\n"
                    "6. For holidays: use get_holiday_info"
                ),
                model=GEMINI_MODEL,
            )

            plan_prompt = (
                f"Create an execution plan for this query:\n\n"
                f"User query: {user_message}\n\n"
                "Return the JSON plan only, no other text."
            )

            result = plan_session.send(plan_prompt)
            if result.get("error"):
                return []

            response = result["response"]
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                return []

            text = ""
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

            # Parse JSON from response
            text = text.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            plan_data = json.loads(text)
            steps = plan_data.get("steps", [])

            # Validate steps
            valid_tools = {
                "query_sales", "check_inventory", "run_forecast", "get_wac_analysis",
                "get_expenses", "calculate_profit_loss", "get_procurement_recommendation",
                "get_holiday_info", "get_supplier_performance", "generate_marketing_content"
            }
            validated_steps = []
            for step in steps:
                if step.get("tool") in valid_tools:
                    validated_steps.append({
                        "step": step.get("step", len(validated_steps) + 1),
                        "tool": step["tool"],
                        "args": step.get("args", {}),
                        "reason": step.get("reason", ""),
                    })

            return validated_steps

        except Exception as exc:
            self.reasoning_trace.append({
                "step": 0,
                "type": "plan_error",
                "content": f"Planning failed: {exc}",
            })
            return []

    def run_with_plan(self, user_message: str, use_reflection: bool = False, use_guardrails: bool = True) -> Tuple[str, List[Dict]]:
        """
        Plan-and-Execute mode: Create a plan first, then execute each step.

        This is more reliable than pure ReAct for complex multi-step tasks
        because the model commits to a strategy upfront.

        Args:
          user_message: The user's query
          use_reflection: If True, run a self-critique pass after generating the answer
          use_guardrails: If True, validate output before showing to user

        Returns:
          final_answer  : str  — the agent's final text response
          trace         : list — reasoning steps [{step, type, content}]
        """
        if not self.api_key:
            return (
                "Gemini API key required for the agentic AI. "
                "Add it to the .env file as GEMINI_API_KEY or enter it below.",
                [],
            )

        try:
            from google.genai import types
            from gemini_client import AgentChatSession
        except ImportError:
            return "google-genai not installed. Run: pip install google-genai", []

        self.reasoning_trace = [{"step": 0, "type": "user_input", "content": user_message}]

        # ── Phase 1: Create the plan ──────────────────────────────────────────
        self.reasoning_trace.append({
            "step": 0, "type": "planning", "content": "Creating execution plan..."
        })

        plan = self._create_plan(user_message)

        if not plan:
            # Fallback to ReAct if planning fails
            self.reasoning_trace.append({
                "step": 0, "type": "plan_fallback",
                "content": "Planning failed, falling back to ReAct mode"
            })
            return self.run(user_message)

        # Log the plan
        self.reasoning_trace.append({
            "step": 0,
            "type": "plan_created",
            "content": f"Plan: {len(plan)} steps",
            "plan": plan,
        })

        # ── Phase 2: Execute each step ────────────────────────────────────────
        tool_results = []
        for step_info in plan:
            step_num = step_info["step"]
            tool_name = step_info["tool"]
            tool_args = step_info["args"]
            reason = step_info.get("reason", "")

            self.reasoning_trace.append({
                "step": step_num,
                "type": "plan_step",
                "tool": tool_name,
                "content": f"Step {step_num}: {tool_name} — {reason}",
            })

            # Execute the tool
            t_start = time.time()
            try:
                tool_result = execute_tool(
                    tool_name, tool_args,
                    db_path=self.db_path,
                    ent_model=self.ent_model,
                    ent_features=self.ent_features,
                    get_mult_fn=self.get_mult_fn,
                    gemini_api_key=self.api_key,
                )
            except Exception as exc:
                tool_result = {"error": str(exc)}

            elapsed = round(time.time() - t_start, 2)

            self.reasoning_trace.append({
                "step": step_num,
                "type": "tool_result",
                "tool": tool_name,
                "result": tool_result,
                "elapsed": elapsed,
                "content": json.dumps(tool_result, ensure_ascii=False)[:500],
            })

            tool_results.append({
                "step": step_num,
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
                "reason": reason,
            })

        # ── Phase 3: Synthesize final answer ──────────────────────────────────
        self.reasoning_trace.append({
            "step": len(plan) + 1,
            "type": "synthesizing",
            "content": "Synthesizing final answer from tool results..."
        })

        # Build a synthesis prompt with all tool results
        synthesis_prompt = (
            f"User query: {user_message}\n\n"
            f"Execution plan ({len(plan)} steps):\n"
        )
        for tr in tool_results:
            synthesis_prompt += (
                f"\nStep {tr['step']}: {tr['tool']}({json.dumps(tr['args'], ensure_ascii=False)})\n"
                f"Reason: {tr['reason']}\n"
                f"Result: {json.dumps(tr['result'], ensure_ascii=False)[:1000]}\n"
            )

        synthesis_prompt += (
            "\nBased on all the tool results above, provide a clear, actionable "
            "answer to the user's query. Be specific with numbers. "
            "Answer in the same language the user used (Arabic or English). "
            "Do not hallucinate — only use data from the tool results."
        )

        # Use a fresh session for synthesis (no tools, just text)
        try:
            synth_session = AgentChatSession(
                api_key=self.api_key,
                system_instruction="You are ButcherAI, a business intelligence assistant for an Egyptian butcher shop.",
                model=GEMINI_MODEL,
            )
            synth_result = synth_session.send(synthesis_prompt)
            if synth_result.get("error"):
                final_answer = f"Synthesis error: {synth_result['error']}"
            else:
                response = synth_result["response"]
                candidate = response.candidates[0] if response.candidates else None
                if candidate:
                    final_answer = ""
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_answer += part.text
                    final_answer = final_answer.strip()
                else:
                    final_answer = "No response from synthesis."
        except Exception as exc:
            final_answer = f"Synthesis failed: {exc}"

        self.reasoning_trace.append({
            "step": len(plan) + 2,
            "type": "final_answer",
            "content": final_answer,
        })

        # ── Output Guardrails ──────────────────────────────────────────────
        if use_guardrails and final_answer:
            tool_results_for_check = [tr["result"] for tr in tool_results]
            validation = self._validate_output(
                user_message, final_answer, tool_results_for_check
            )
            self.reasoning_trace.append({
                "step": len(plan) + 3,
                "type": "guardrail_check",
                "content": f"Safe: {validation['safe']}, Action: {validation['action']}",
                "issues": validation["issues"],
            })
            if validation["action"] == "block":
                final_answer = (
                    "⚠️ تم حظر هذه الإجابة بسبب مشاكل في الجودة. "
                    "يرجى إعادة صياغة السؤال أو المحاولة مرة أخرى."
                )

        # ── Reflection / Self-Critique ──────────────────────────────────────
        if use_reflection and final_answer:
            self.reasoning_trace.append({
                "step": len(plan) + 3,
                "type": "reflection_start",
                "content": "Running self-critique..."
            })

            tool_results_for_reflection = [
                tr["result"] for tr in tool_results
            ]

            reflection = self._reflect(
                user_message, final_answer, tool_results_for_reflection
            )

            self.reasoning_trace.append({
                "step": len(plan) + 3,
                "type": "reflection_result",
                "content": f"Needs correction: {reflection.get('needs_correction', False)}",
                "issues": reflection.get("issues", []),
            })

            if reflection.get("needs_correction") and reflection.get("corrected_answer"):
                self.reasoning_trace.append({
                    "step": len(plan) + 3,
                    "type": "reflection_correction",
                    "content": reflection["corrected_answer"],
                })
                final_answer = reflection["corrected_answer"]

        return final_answer, self.reasoning_trace

    def _validate_output(self, user_message: str, answer: str, tool_results: List[Dict]) -> Dict:
        """
        Output Guardrails: Validate agent output before showing to user.

        Checks:
        1. Hallucinated numbers — numbers in answer not found in tool results
        2. Dangerous actions — destructive commands in output
        3. Language mismatch — answer in wrong language
        4. Empty/trivial answer — answer too short to be useful

        Returns:
          {"safe": bool, "issues": [...], "action": "block"|"flag"|"allow"}
        """
        import re

        issues = []
        action = "allow"

        # ── Check 1: Empty or trivial answer ────────────────────────────────
        if not answer or len(answer.strip()) < 10:
            issues.append("Answer is empty or too short")
            action = "block"
            return {"safe": False, "issues": issues, "action": action}

        # ── Check 2: Hallucinated numbers ──────────────────────────────────
        # Extract numbers from answer
        numbers_in_answer = set(re.findall(r'\d+\.?\d*', answer))
        if numbers_in_answer:
            # Extract numbers from tool results
            tool_results_str = json.dumps(tool_results, ensure_ascii=False)
            numbers_in_tools = set(re.findall(r'\d+\.?\d*', tool_results_str))

            # Find unverified numbers (allow small integers 1-9 which are likely phrasing)
            unverified = set()
            for num in numbers_in_answer:
                if float(num) >= 10 or '.' in num:
                    if num not in numbers_in_tools:
                        unverified.add(num)

            if unverified:
                issues.append(
                    f"Unverified numbers (possible hallucination): {', '.join(list(unverified)[:5])}"
                )
                action = "flag"

        # ── Check 3: Dangerous actions ─────────────────────────────────────
        dangerous_patterns = [
            r'\bdelete\b', r'\bdrop\b', r'\btruncate\b', r'\berase\b',
            r'\bمسح\b', r'\bحذف\b', r'\bتدمير\b',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append(f"Potentially dangerous keyword detected: {pattern}")
                action = "block"

        # ── Check 4: Language mismatch ─────────────────────────────────────
        # Check if user wrote in Arabic but answer is mostly English (or vice versa)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', user_message))
        english_chars = len(re.findall(r'[a-zA-Z]', user_message))

        answer_arabic = len(re.findall(r'[\u0600-\u06FF]', answer))
        answer_english = len(re.findall(r'[a-zA-Z]', answer))

        # If user wrote mostly Arabic but answer is mostly English
        if arabic_chars > english_chars and answer_english > answer_arabic * 2:
            issues.append("Language mismatch: user asked in Arabic but answer is mostly English")
            if action == "allow":
                action = "flag"

        # If user wrote mostly English but answer is mostly Arabic
        if english_chars > arabic_chars and answer_arabic > answer_english * 2:
            issues.append("Language mismatch: user asked in English but answer is mostly Arabic")
            if action == "allow":
                action = "flag"

        safe = action == "allow"
        return {"safe": safe, "issues": issues, "action": action}

    def _reflect(self, user_message: str, answer: str, tool_results: List[Dict]) -> Dict:
        """
        Reflection / Self-Critique: Ask a second Gemini call to review the agent's answer.

        This catches errors before the user sees them:
        - Hallucinated numbers not in tool results
        - Logical inconsistencies
        - Missing information
        - Wrong language

        Returns:
          {"needs_correction": bool, "issues": [...], "corrected_answer": str}
        """
        try:
            from gemini_client import AgentChatSession

            # Build context for reflection
            tool_summary = json.dumps(tool_results, ensure_ascii=False)[:3000]

            reflection_prompt = (
                "You are a quality reviewer. Review the following agent answer for accuracy.\n\n"
                f"User query: {user_message}\n\n"
                f"Agent answer:\n{answer}\n\n"
                f"Tool results used:\n{tool_summary}\n\n"
                "Check for:\n"
                "1. Are all numbers in the answer supported by tool results?\n"
                "2. Is the answer logically consistent with the data?\n"
                "3. Is anything important missing?\n"
                "4. Is the language correct (Arabic query → Arabic answer)?\n\n"
                "Return ONLY valid JSON:\n"
                '{"needs_correction": true/false, "issues": ["issue1", "issue2"], '
                '"corrected_answer": "improved answer or empty string if no correction needed"}'
            )

            reflect_session = AgentChatSession(
                api_key=self.api_key,
                system_instruction="You are a meticulous quality reviewer for an AI assistant.",
                model=GEMINI_MODEL,
            )

            result = reflect_session.send(reflection_prompt)
            if result.get("error"):
                return {"needs_correction": False, "issues": [], "corrected_answer": "",
                        "error": result["error"]}

            response = result["response"]
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                return {"needs_correction": False, "issues": [], "corrected_answer": ""}

            text = ""
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

            # Parse JSON
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            reflection = json.loads(text)
            return {
                "needs_correction": reflection.get("needs_correction", False),
                "issues": reflection.get("issues", []),
                "corrected_answer": reflection.get("corrected_answer", ""),
            }

        except Exception as exc:
            return {"needs_correction": False, "issues": [], "corrected_answer": "",
                    "error": str(exc)}

    async def run_async(self, user_message: str, use_guardrails: bool = True) -> Tuple[str, List[Dict]]:
        """
        Async agent execution: Non-blocking query that can run concurrently
        with other agent queries.

        Usage:
          answer, trace = await agent.run_async("What should I buy?")

        This enables:
        - Running multiple agent queries in parallel
        - Non-blocking UI updates
        - Better resource utilization
        """
        import asyncio

        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Uses default executor
            lambda: self.run(user_message, use_reflection=False, use_guardrails=use_guardrails)
        )
        return result

    @staticmethod
    async def run_multiple(queries: List[str], api_key: str, use_guardrails: bool = True) -> List[Tuple[str, List[Dict]]]:
        """
        Run multiple agent queries concurrently.

        Usage:
          results = await ButcherAIAgent.run_multiple(
              ["What should I buy?", "Check inventory", "Net profit?"],
              api_key="your_key"
          )

        Returns:
          List of (answer, trace) tuples, one per query
        """
        agent = ButcherAIAgent(api_key=api_key)
        tasks = [
            agent.run_async(query, use_guardrails=use_guardrails)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    def run_streaming(self, user_message: str, use_guardrails: bool = True) -> Tuple[str, List[Dict]]:
        """
        Streaming ReAct mode: Shows tokens as they are generated.

        This makes the agent feel more responsive — the user sees the answer
        being written in real-time instead of waiting for the complete response.

        Args:
          user_message: The user's query
          use_guardrails: If True, validate output before showing to user

        Returns:
          final_answer  : str  — the agent's final text response
          trace         : list — reasoning steps [{step, type, content}]
        """
        if not self.api_key:
            return (
                "Gemini API key required for the agentic AI. "
                "Add it to the .env file as GEMINI_API_KEY or enter it below.",
                [],
            )

        try:
            from google.genai import types
            from gemini_client import AgentChatSession
        except ImportError:
            return "google-genai not installed. Run: pip install google-genai", []

        self.reasoning_trace = [{"step": 0, "type": "user_input", "content": user_message}]

        # Build tool declarations
        if self._gemini_tools is None:
            self._gemini_tools = [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=schema["name"],
                        description=schema["description"],
                        parameters=schema["parameters"],
                    )
                ])
                for schema in TOOL_SCHEMAS
            ]

        # Create session
        if self._session is None or self._session_api_key != self.api_key:
            self._session = AgentChatSession(
                api_key=self.api_key,
                system_instruction=self.SYSTEM_PROMPT,
                tools=self._gemini_tools,
                model=GEMINI_MODEL,
            )
            self._session_api_key = self.api_key

        # For streaming, we use the non-streaming run() but yield chunks
        # The actual streaming happens at the Streamlit UI level via st.write_stream
        # Here we just run normally and return the answer
        # The UI will handle the streaming display

        # Run the normal ReAct loop
        final_answer, trace = self.run(user_message, use_reflection=False, use_guardrails=use_guardrails)

        return final_answer, trace

    def run(self, user_message: str, use_reflection: bool = False, use_guardrails: bool = True) -> Tuple[str, List[Dict]]:
        """
        Execute one agent turn (ReAct mode).

        Args:
          user_message: The user's query
          use_reflection: If True, run a self-critique pass after generating the answer
          use_guardrails: If True, validate output before showing to user

        Returns:
          final_answer  : str  — the agent's final text response
          trace         : list — reasoning steps [{step, type, content}]
        """
        if not self.api_key:
            return (
                "⚠️ Gemini API key required for the agentic AI. "
                "Add it to the .env file as GEMINI_API_KEY or enter it below.",
                [],
            )

        try:
            from google.genai import types
            from gemini_client import AgentChatSession
        except ImportError:
            return "❌ google-genai not installed. Run: pip install google-genai", []

        # Tool declarations are static per TOOL_SCHEMAS — build once, reuse.
        if self._gemini_tools is None:
            self._gemini_tools = [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=schema["name"],
                        description=schema["description"],
                        parameters=schema["parameters"],
                    )
                ])
                for schema in TOOL_SCHEMAS
            ]

        # Reuse the chat session across calls (preserves multi-turn context,
        # same intent as the old self.history accumulation) — only rebuild
        # if this is the first call or the API key changed underneath us.
        if self._session is None or self._session_api_key != self.api_key:
            self._session = AgentChatSession(
                api_key=self.api_key,
                system_instruction=self.SYSTEM_PROMPT,
                tools=self._gemini_tools,
                model=GEMINI_MODEL,
            )
            self._session_api_key = self.api_key

        self.reasoning_trace = [{"step": 0, "type": "user_input", "content": user_message}]

        final_answer = ""
        step = 0
        # str on the first send; a list of function-response Parts on every
        # subsequent turn inside this same tool-calling loop.
        next_message = user_message

        while step < MAX_TURNS:
            step += 1

            result = self._session.send(next_message, db_path=self.db_path)
            if result["error"]:
                final_answer = f"❌ Gemini API error: {result['error']}"
                break

            response = result["response"]
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                final_answer = "❌ No response from Gemini."
                break

            tool_calls_made = []
            text_parts = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    fc = part.function_call
                    args = {k: v for k, v in fc.args.items()}
                    tool_calls_made.append({"name": fc.name, "args": args})

                    self.reasoning_trace.append({
                        "step"   : step,
                        "type"   : "tool_call",
                        "tool"   : fc.name,
                        "args"   : args,
                        "content": f"Calling {fc.name}({json.dumps(args, ensure_ascii=False)[:200]})",
                    })
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            if not tool_calls_made:
                final_answer = "\n".join(text_parts).strip()
                self.reasoning_trace.append({
                    "step": step, "type": "final_answer", "content": final_answer
                })

                # ── Output Guardrails ────────────────────────────────────────
                if use_guardrails and final_answer:
                    tool_results_for_check = [
                        s["result"] for s in self.reasoning_trace
                        if s.get("type") == "tool_result"
                    ]
                    validation = self._validate_output(
                        user_message, final_answer, tool_results_for_check
                    )
                    self.reasoning_trace.append({
                        "step": step + 1,
                        "type": "guardrail_check",
                        "content": f"Safe: {validation['safe']}, Action: {validation['action']}",
                        "issues": validation["issues"],
                    })
                    if validation["action"] == "block":
                        final_answer = (
                            "⚠️ تم حظر هذه الإجابة بسبب مشاكل في الجودة. "
                            "يرجى إعادة صياغة السؤال أو المحاولة مرة أخرى."
                        )

                # ── Reflection / Self-Critique ──────────────────────────────
                if use_reflection and final_answer:
                    self.reasoning_trace.append({
                        "step": step + 1,
                        "type": "reflection_start",
                        "content": "Running self-critique..."
                    })

                    # Collect tool results for reflection context
                    tool_results_for_reflection = [
                        s["result"] for s in self.reasoning_trace
                        if s.get("type") == "tool_result"
                    ]

                    reflection = self._reflect(
                        user_message, final_answer, tool_results_for_reflection
                    )

                    self.reasoning_trace.append({
                        "step": step + 1,
                        "type": "reflection_result",
                        "content": f"Needs correction: {reflection.get('needs_correction', False)}",
                        "issues": reflection.get("issues", []),
                    })

                    if reflection.get("needs_correction") and reflection.get("corrected_answer"):
                        self.reasoning_trace.append({
                            "step": step + 1,
                            "type": "reflection_correction",
                            "content": reflection["corrected_answer"],
                        })
                        final_answer = reflection["corrected_answer"]

                break

            # Execute tools, build function_response Parts for the next turn.
            # The Chat object (inside AgentChatSession) retains candidate.content
            # — including any thought_signature — internally. We never
            # reconstruct history by hand, which is what caused the old
            # second-tool-call 400 error.
            response_parts = []
            for tc in tool_calls_made:
                t_start = time.time()
                tool_result = execute_tool(
                    tc["name"], tc["args"],
                    db_path=self.db_path,
                    ent_model=self.ent_model,
                    ent_features=self.ent_features,
                    get_mult_fn=self.get_mult_fn,
                    gemini_api_key=self.api_key,
                )
                elapsed = round(time.time() - t_start, 2)

                self.reasoning_trace.append({
                    "step"   : step,
                    "type"   : "tool_result",
                    "tool"   : tc["name"],
                    "result" : tool_result,
                    "elapsed": elapsed,
                    "content": json.dumps(tool_result, ensure_ascii=False)[:500],
                })

                response_parts.append(
                    types.Part.from_function_response(name=tc["name"], response=tool_result)
                )

            next_message = response_parts

        else:
            final_answer = "⚠️ Maximum reasoning steps reached without a final answer."

        return final_answer, self.reasoning_trace


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_agent_page(
    db_path       : str  = DB_PATH,
    gemini_api_key: str  = "",
    ent_model           = None,
    ent_features        = None,
    get_mult_fn         = None,
):
    """Full Streamlit page for the autonomous ButcherAI agent."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🤖 وكيل الذكاء الاصطناعي المستقل")
    st.markdown(
        "<div style='background:#fff3cd;border:2px solid #ffc107;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "⚡ الوكيل يفكر باستقلالية: يختار الأدوات المناسبة، ينفذها بالترتيب الصحيح، "
        "ويركّب إجابة كاملة من البيانات الحقيقية — بدون تدخل منك."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── API key + Mode selector + Reflection toggle ───────────────────────────
    col_key, col_mode, col_reflect, col_reset = st.columns([3, 2, 2, 1])
    with col_key:
        api_key = st.text_input(
            "🔑 Gemini API Key (مطلوب للوكيل):",
            type="password",
            value=gemini_api_key,
        )
    with col_mode:
        st.write("")
        agent_mode = st.selectbox(
            "🧠 وضع الوكيل:",
            ["ReAct (فوري)", "Plan-Execute (مخطط)", "Streaming (تدفق)"],
            help="ReAct: يفكر ويتصرف فوراً. Plan-Execute: يخطط أولاً ثم ينفذ. Streaming: يعرض الإجابة أثناء الكتابة."
        )
    with col_reflect:
        st.write("")
        use_reflection = st.toggle(
            "🔍 المراجعة الذاتية",
            value=False,
            help="الوكيل يراجع إجابته قبل عرضها عليك (أبطأ لكن أدق)"
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 محادثة جديدة"):
            for k in ["agent_history", "agent_obj", "agent_trace_all"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # ── Initialize agent in session ───────────────────────────────────────────
    if "agent_obj" not in st.session_state or api_key != getattr(
        st.session_state.get("agent_obj"), "api_key", ""
    ):
        st.session_state["agent_obj"] = ButcherAIAgent(
            api_key=api_key, db_path=db_path,
            ent_model=ent_model, ent_features=ent_features,
            get_mult_fn=get_mult_fn,
        )
        st.session_state["agent_history"]   = []
        st.session_state["agent_trace_all"] = []

    agent: ButcherAIAgent = st.session_state["agent_obj"]

    # ── Example prompts ───────────────────────────────────────────────────────
    st.subheader("💬 أمثلة على أسئلة الوكيل")
    examples = [
        "ما الذي يجب أن أشتريه غداً للمحل؟",
        "احسب صافي الربح هذا الأسبوع بعد خصم المصروفات",
        "ما هو أفضل مورد لدي وكيف أقارن أسعاره؟",
        "اعمل لي خطة شراء كاملة قبل عيد الأضحى",
        "أي المنتجات لديها هامش ربح أقل من 20%؟",
        "Generate a Friday promotion for ground meat",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(f"💡 {ex[:35]}…" if len(ex) > 35 else f"💡 {ex}", key=f"ex_{i}"):
            st.session_state["pending_message"] = ex

    st.divider()

    # ── Conversation display ──────────────────────────────────────────────────
    for turn in st.session_state["agent_history"]:
        role = turn["role"]
        icon = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=icon):
            st.markdown(turn["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_message", None)
    user_msg = st.chat_input("اسأل الوكيل أي شيء عن محلك...") or pending

    if user_msg:
        # Show user message
        st.session_state["agent_history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_msg)

        # Run agent with live reasoning display
        with st.chat_message("assistant", avatar="🤖"):
            reasoning_placeholder = st.empty()
            answer_placeholder    = st.empty()

            if not api_key:
                answer_placeholder.error(
                    "❌ أدخل Gemini API Key لتشغيل الوكيل المستقل."
                )
            else:
                with st.spinner("🧠 الوكيل يفكر ويستدعي الأدوات..."):
                    t_start = time.time()
                    agent.api_key = api_key  # update in case changed
                    if "Plan" in agent_mode:
                        answer, trace = agent.run_with_plan(user_msg, use_reflection=use_reflection, use_guardrails=True)
                    elif "Streaming" in agent_mode:
                        answer, trace = agent.run_streaming(user_msg, use_guardrails=True)
                    else:
                        answer, trace = agent.run(user_msg, use_reflection=use_reflection, use_guardrails=True)
                    elapsed = round(time.time() - t_start, 2)

                # Display reasoning trace
                tool_steps = [s for s in trace if s["type"] in ("tool_call", "tool_result")]
                if tool_steps:
                    with reasoning_placeholder.expander(
                        f"🔍 خطوات التفكير ({len([s for s in trace if s['type']=='tool_call'])} أدوات، {elapsed}s)",
                        expanded=False,
                    ):
                        for s in trace:
                            if s["type"] == "tool_call":
                                st.markdown(
                                    f"**→ استدعاء `{s['tool']}`**  \n"
                                    f"```json\n{json.dumps(s['args'], ensure_ascii=False, indent=2)}\n```"
                                )
                            elif s["type"] == "tool_result":
                                result_str = json.dumps(s["result"], ensure_ascii=False, indent=2)
                                st.markdown(
                                    f"**← نتيجة `{s['tool']}`** ({s.get('elapsed', 0)}s)  \n"
                                    f"```json\n{result_str[:800]}{'...' if len(result_str)>800 else ''}\n```"
                                )
                            elif s["type"] == "plan_created":
                                st.info(f"📋 الخطة: {s.get('content', '')}")
                                if "plan" in s:
                                    for step in s["plan"]:
                                        st.markdown(
                                            f"  **الخطوة {step['step']}**: `{step['tool']}` — {step.get('reason', '')}"
                                        )
                            elif s["type"] == "plan_step":
                                st.markdown(f"▶️ **{s.get('content', '')}**")
                            elif s["type"] == "reflection_start":
                                st.info("🔍 جاري المراجعة الذاتية...")
                            elif s["type"] == "reflection_result":
                                if "needs_correction" in s.get("content", "").lower() and "true" in s.get("content", "").lower():
                                    st.warning(f"⚠️ تم اكتشاف مشاكل: {s.get('issues', [])}")
                                else:
                                    st.success("✅ المراجعة الذاتية: الإجابة سليمة")
                            elif s["type"] == "guardrail_check":
                                if "block" in s.get("content", "").lower():
                                    st.error(f"🚫 تم حظر الإجابة: {s.get('issues', [])}")
                                elif "flag" in s.get("content", "").lower():
                                    st.warning(f"⚠️ تحذير: {s.get('issues', [])}")
                                else:
                                    st.success("✅ فحص الجودة: الإجابة آمنة")
                            elif s["type"] == "reflection_correction":
                                st.info("🔄 تم تصحيح الإجابة بناءً على المراجعة")
                            elif s["type"] == "final_answer":
                                st.success(f"✅ الإجابة النهائية جاهزة ({elapsed}s)")

                answer_placeholder.markdown(answer)

        st.session_state["agent_history"].append({"role": "assistant", "content": answer})
        st.session_state["agent_trace_all"].append(trace)

    # ── Sidebar: tool registry ────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**🛠️ أدوات الوكيل المتاحة:**")
        for tool in TOOL_SCHEMAS:
            st.markdown(f"• `{tool['name']}`")

    # ── Autonomous mode: daily briefing button ────────────────────────────────
    st.divider()
    st.subheader("⚡ الوضع التلقائي — تقرير يومي فوري")
    if st.button("📋 اطلب تقرير الصباح التلقائي", type="secondary"):
        auto_prompt = (
            "أعطني تقرير الصباح الكامل للمحل: "
            "1) المخزون الحالي مع التنبيهات، "
            "2) توصيات الشراء لليوم مع الكميات، "
            "3) إيرادات أمس مقارنة باليوم السابق، "
            "4) أي مواسم أو أعياد قادمة في 14 يوم القادمة."
        )
        st.session_state["pending_message"] = auto_prompt
        st.rerun()