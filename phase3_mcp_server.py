"""
ButcherAI Phase 3 — MCP (Model Context Protocol) Server
=========================================================
Exposes all ButcherAI shop data and capabilities as an MCP server.
Any MCP client (Claude Desktop, Cursor, custom clients) can connect
and interact with the butcher shop's data autonomously.

What this enables:
  - Claude Desktop → "Check my shop inventory" → queries live SQLite
  - Claude Desktop → "Plan my Eid procurement" → runs forecast + recommendation
  - Any MCP client → full access to 12 tools + 3 resources + 4 prompts

Install:
  pip install mcp

Run the server (standalone):
  python phase3_mcp_server.py

For Claude Desktop — add to claude_desktop_config.json:
  {
    "mcpServers": {
      "butcherai": {
        "command": "python",
        "args": ["/path/to/phase3_mcp_server.py"],
        "env": {
          "DB_PATH": "/path/to/ButcherAI.db",
          "GEMINI_API_KEY": "your_key_here"
        }
      }
    }
  }

Architecture:
  MCP stdio transport (works with Claude Desktop)
  12 tools   — cover all shop operations
   3 resources — schema, product catalog, knowledge base
   4 prompts  — daily briefing, Eid prep, weekly review, procurement
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (from environment variables for Claude Desktop compatibility)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH        = os.environ.get("DB_PATH",        "ButcherAI.db")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SERVER_NAME    = "butcherai"
SERVER_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# MCP IMPORT GUARD
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print(
        "MCP package not found. Install with: pip install mcp",
        file=sys.stderr,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Set DB_PATH environment variable to the correct path."
        )
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _q(sql: str, params: tuple = ()) -> List[Dict]:
    """Execute a SELECT and return rows as list of dicts."""
    conn = _db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _scalar(sql: str, params: tuple = ()) -> Any:
    conn = _db()
    row  = conn.execute(sql, params).fetchone()
    conn.close()
    return row[0] if row else None


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# HOLIDAY MULTIPLIER  (self-contained, no app.py dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _get_mult(d: date) -> tuple[float, str]:
    try:
        from hijri_converter import Gregorian
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        if h.month == 9:                         return 1.55, f"رمضان {h.year}"
        if h.month == 10 and 1 <= h.day <= 3:    return 2.20, f"عيد الفطر {h.year}"
        if h.month == 12 and 10 <= h.day <= 13:  return 3.00, f"عيد الأضحى {h.year}"
    except Exception:
        pass
    wd = d.weekday()
    if wd == 4: return 1.30, "الجمعة"
    if wd == 5: return 1.15, "السبت"
    if wd == 6: return 1.10, "الأحد"
    if wd == 0: return 0.90, "الإثنين"
    return 1.00, "يوم عادي"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS  (each returns a formatted string for MCP text content)
# ─────────────────────────────────────────────────────────────────────────────

def tool_get_inventory(product_name: str = "", low_stock_only: bool = False) -> str:
    """Real-time inventory levels from ButcherAI.db."""
    pf  = f"AND p.name_ar LIKE '%{product_name}%'" if product_name.strip() else ""
    lf  = "AND COALESCE(i.current_stock_kg,0) < p.reorder_point" if low_stock_only else ""
    rows = _q(
        f"SELECT p.name_ar, p.name_en, p.category, "
        f"COALESCE(i.current_stock_kg,0) stock, p.reorder_point, i.last_updated "
        f"FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
        f"WHERE 1=1 {pf} {lf} ORDER BY COALESCE(i.current_stock_kg,0) ASC"
    )
    for r in rows:
        stk = r["stock"]
        rpt = r["reorder_point"] or 10
        r["alert"] = (
            "EMPTY"    if stk == 0      else
            "CRITICAL" if stk < rpt*0.3 else
            "LOW"      if stk < rpt     else "OK"
        )
    summary = {
        "EMPTY"   : sum(1 for r in rows if r["alert"] == "EMPTY"),
        "CRITICAL": sum(1 for r in rows if r["alert"] == "CRITICAL"),
        "LOW"     : sum(1 for r in rows if r["alert"] == "LOW"),
        "OK"      : sum(1 for r in rows if r["alert"] == "OK"),
    }
    return _json({"inventory": rows, "summary": summary})


def tool_query_sales(days_back: int = 30, product_name: str = "",
                     group_by: str = "product") -> str:
    """Query sales history with flexible grouping."""
    pf  = f"AND (s.product_name_ar LIKE '%{product_name}%')" if product_name.strip() else ""
    if group_by == "product":
        rows = _q(
            f"SELECT p.name_ar, p.name_en, p.category, "
            f"COUNT(*) tx, ROUND(SUM(s.weight_kg),1) kg, "
            f"ROUND(SUM(s.total_price),0) rev, ROUND(SUM(s.gross_profit),0) pft, "
            f"ROUND(AVG(s.profit_margin_percent),1) mg "
            f"FROM sales s JOIN products p ON s.product_id=p.id "
            f"WHERE s.sale_date >= date('now','-{days_back} days') {pf} "
            f"GROUP BY s.product_id ORDER BY rev DESC LIMIT 20"
        )
    elif group_by == "day":
        rows = _q(
            f"SELECT s.sale_date, COUNT(*) tx, ROUND(SUM(s.weight_kg),1) kg, "
            f"ROUND(SUM(s.total_price),0) rev "
            f"FROM sales s JOIN products p ON s.product_id=p.id "
            f"WHERE s.sale_date >= date('now','-{days_back} days') {pf} "
            f"GROUP BY s.sale_date ORDER BY s.sale_date"
        )
    else:  # category
        rows = _q(
            f"SELECT p.category, COUNT(*) tx, ROUND(SUM(s.weight_kg),1) kg, "
            f"ROUND(SUM(s.total_price),0) rev "
            f"FROM sales s JOIN products p ON s.product_id=p.id "
            f"WHERE s.sale_date >= date('now','-{days_back} days') {pf} "
            f"GROUP BY p.category ORDER BY rev DESC"
        )
    totals = {
        "transactions": _scalar(f"SELECT COUNT(*) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0,
        "revenue_egp" : _scalar(f"SELECT ROUND(SUM(total_price),0) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0,
        "profit_egp"  : _scalar(f"SELECT ROUND(SUM(gross_profit),0) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0,
        "weight_kg"   : _scalar(f"SELECT ROUND(SUM(weight_kg),1) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0,
    }
    return _json({"period_days": days_back, "group_by": group_by, "data": rows, "totals": totals})


def tool_get_forecast(product_name: str, days: int = 7) -> str:
    """Heuristic demand forecast (full LightGBM requires app.py context)."""
    prod = _q(
        "SELECT id, name_ar, name_en FROM products WHERE name_ar LIKE ?",
        (f"%{product_name}%",),
    )
    if not prod:
        return _json({"error": f"Product not found: {product_name}"})
    pid  = prod[0]["id"]
    hist = _q(
        f"SELECT sale_date, SUM(weight_kg) kg FROM sales "
        f"WHERE product_id={pid} GROUP BY sale_date ORDER BY sale_date DESC LIMIT 30"
    )
    if len(hist) < 3:
        return _json({"error": "Insufficient history (need 3+ days)"})

    kg_vals   = [r["kg"] for r in reversed(hist)]
    recent7   = sum(kg_vals[-7:]) / min(7, len(kg_vals))
    last_date = datetime.strptime(hist[0]["sale_date"], "%Y-%m-%d").date()
    inv_row   = _q(f"SELECT current_stock_kg FROM inventory WHERE product_id={pid}")
    stock     = float(inv_row[0]["current_stock_kg"]) if inv_row else 0.0

    forecast = []
    for step in range(1, days + 1):
        fd   = last_date + timedelta(days=step)
        mult, lbl = _get_mult(fd)
        pred = round(recent7 * mult, 1)
        forecast.append({
            "date": str(fd), "predicted_kg": pred,
            "season": lbl, "multiplier": mult,
        })

    total = sum(f["predicted_kg"] for f in forecast)
    rec   = round(max(0, total * 1.1 - stock), 1)
    return _json({
        "product"         : prod[0]["name_ar"],
        "forecast"        : forecast,
        "total_demand_kg" : round(total, 1),
        "current_stock_kg": stock,
        "recommended_buy_kg": rec,
    })


def tool_get_wac(product_name: str = "", days_back: int = 30) -> str:
    """Weighted Average Cost analysis per product."""
    pf   = f"AND product_ar LIKE '%{product_name}%'" if product_name.strip() else ""
    rows = _q(
        f"SELECT product_ar, "
        f"ROUND(SUM(total_cost)/NULLIF(SUM(weight_kg),0),0) wac, "
        f"ROUND(SUM(weight_kg),1) kg_bought, COUNT(*) orders, MAX(purchase_date) last_pur "
        f"FROM purchases WHERE purchase_date >= date('now','-{days_back} days') {pf} "
        f"GROUP BY product_ar ORDER BY kg_bought DESC"
    )
    for r in rows:
        prod = _q(
            "SELECT avg_price, avg_margin FROM products WHERE name_ar LIKE ?",
            (f"%{r['product_ar']}%",),
        )
        price    = prod[0]["avg_price"] if prod else 0
        tgt      = prod[0]["avg_margin"] if prod else 25
        wac      = r["wac"] or 0
        act_mg   = round((price - wac) / price * 100, 1) if price > 0 else 0
        r.update({
            "sell_price_egp"  : price,
            "actual_margin_pct": act_mg,
            "target_margin_pct": tgt,
            "margin_gap"       : round(act_mg - tgt, 1),
            "status"           : "✅" if act_mg >= 20 else "⚠️" if act_mg >= 10 else "🔴",
        })
    return _json({"period_days": days_back, "wac_analysis": rows})


def tool_get_expenses(days_back: int = 30, category: str = "") -> str:
    """Operational expense summary."""
    try:
        cf   = f"AND category LIKE '%{category}%'" if category.strip() else ""
        rows = _q(
            f"SELECT category, ROUND(SUM(amount_egp),0) total, COUNT(*) entries "
            f"FROM expenses WHERE expense_date >= date('now','-{days_back} days') {cf} "
            f"GROUP BY category ORDER BY total DESC"
        )
        total = _scalar(
            f"SELECT ROUND(SUM(amount_egp),0) FROM expenses "
            f"WHERE expense_date >= date('now','-{days_back} days') {cf}"
        ) or 0
        return _json({"period_days": days_back, "by_category": rows, "total_egp": total})
    except Exception as exc:
        return _json({"error": str(exc), "note": "Run upgrade_v91.py to create expenses table"})


def tool_calculate_pnl(days_back: int = 30) -> str:
    """Full P&L: revenue, gross profit, expenses, net profit."""
    rev   = _scalar(f"SELECT COALESCE(ROUND(SUM(total_price),0),0) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0
    gp    = _scalar(f"SELECT COALESCE(ROUND(SUM(gross_profit),0),0) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0
    mg    = _scalar(f"SELECT COALESCE(ROUND(AVG(profit_margin_percent),1),0) FROM sales WHERE sale_date >= date('now','-{days_back} days')") or 0
    exp   = 0
    try:
        exp = _scalar(f"SELECT COALESCE(ROUND(SUM(amount_egp),0),0) FROM expenses WHERE expense_date >= date('now','-{days_back} days')") or 0
    except Exception:
        pass
    net   = gp - exp
    net_m = round(net / rev * 100, 1) if rev > 0 else 0
    return _json({
        "period_days"      : days_back,
        "revenue_egp"      : rev,
        "gross_profit_egp" : gp,
        "gross_margin_pct" : mg,
        "expenses_egp"     : exp,
        "net_profit_egp"   : net,
        "net_margin_pct"   : net_m,
        "status"           : "✅ Profitable" if net > 0 else "⚠️ Break-even" if net > -500 else "🔴 Loss",
    })


def tool_get_procurement_plan(product_name: str = "", days: int = 7) -> str:
    """AI procurement plan: inventory + forecast → buy recommendations."""
    prods = _q(
        "SELECT id, name_ar, name_en, avg_price, avg_cost, avg_margin, reorder_point FROM products"
        + (f" WHERE name_ar LIKE '%{product_name}%'" if product_name.strip() else "")
        + " LIMIT 20"
    )
    recs = []
    for p in prods:
        pid      = p["id"]
        inv_row  = _q(f"SELECT current_stock_kg FROM inventory WHERE product_id={pid}")
        stock    = float(inv_row[0]["current_stock_kg"]) if inv_row else 0.0
        hist     = _q(
            f"SELECT SUM(weight_kg)/COUNT(DISTINCT sale_date) daily_avg FROM sales "
            f"WHERE product_id={pid} AND sale_date >= date('now','-14 days')"
        )
        daily = float(hist[0]["daily_avg"] or 0) if hist else 0
        if daily == 0:
            continue

        # Compute demand for next N days with holiday multipliers
        total_demand = 0
        peak_mult    = 1.0
        last_sale    = _scalar(f"SELECT MAX(sale_date) FROM sales WHERE product_id={pid}")
        if not last_sale:
            continue
        last_d = datetime.strptime(last_sale, "%Y-%m-%d").date()
        for step in range(1, days + 1):
            m, _ = _get_mult(last_d + timedelta(days=step))
            total_demand += daily * m
            peak_mult = max(peak_mult, m)

        rec_qty  = round(max(0, total_demand * 1.1 - stock), 1)
        est_cost = round(rec_qty * float(p["avg_cost"] or 0), 0)
        margin_s = min(float(p["avg_margin"] or 20) / 5, 10)
        urgency  = 10 if stock < daily * 1 else 7 if stock < daily * 2 else 4
        score    = round(urgency * 0.5 + margin_s * 0.3 + min(peak_mult * 2, 5) * 0.2, 1)
        priority = "HIGH" if score >= 7 else "MEDIUM" if score >= 4.5 else "LOW"

        recs.append({
            "product_ar"        : p["name_ar"], "product_en": p["name_en"],
            "current_stock_kg"  : round(stock, 1),
            "daily_avg_kg"      : round(daily, 1),
            "forecast_demand_kg": round(total_demand, 1),
            "recommended_buy_kg": rec_qty,
            "estimated_cost_egp": est_cost,
            "priority"          : priority,
            "score"             : score,
            "peak_multiplier"   : round(peak_mult, 2),
        })

    recs.sort(key=lambda x: x["score"], reverse=True)
    high = [r for r in recs if r["priority"] == "HIGH"]
    total_cost = sum(r["estimated_cost_egp"] for r in high)
    return _json({
        "forecast_days"  : days,
        "products_count" : len(recs),
        "high_priority"  : len(high),
        "total_urgent_spend_egp": total_cost,
        "recommendations": recs,
    })


def tool_get_holiday_calendar(days_ahead: int = 14) -> str:
    """Demand multiplier calendar for the next N days."""
    today = date.today()
    rows  = []
    for i in range(days_ahead):
        d    = today + timedelta(days=i)
        mult, lbl = _get_mult(d)
        rows.append({
            "date"      : str(d),
            "day_ar"    : ["الإثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"][d.weekday()],
            "season"    : lbl,
            "multiplier": mult,
            "category"  : "PEAK" if mult >= 2.5 else "HIGH" if mult >= 1.4 else "ABOVE" if mult > 1.0 else "NORMAL" if mult == 1.0 else "SLOW",
        })
    return _json({"calendar": rows, "peak_days": [r for r in rows if r["multiplier"] >= 1.3]})


def tool_get_supplier_performance(days_back: int = 60, supplier_name: str = "") -> str:
    """Supplier performance analysis from purchase history."""
    sf   = f"AND supplier_name LIKE '%{supplier_name}%'" if supplier_name.strip() else ""
    rows = _q(
        f"SELECT supplier_name, COUNT(*) orders, ROUND(SUM(weight_kg),1) kg, "
        f"ROUND(AVG(cost_per_kg),0) avg_cost, "
        f"ROUND(MIN(cost_per_kg),0) min_cost, ROUND(MAX(cost_per_kg),0) max_cost, "
        f"MAX(purchase_date) last_order "
        f"FROM purchases WHERE purchase_date >= date('now','-{days_back} days') "
        f"AND supplier_name IS NOT NULL {sf} "
        f"GROUP BY supplier_name ORDER BY kg DESC"
    )
    for r in rows:
        rng = (r["max_cost"] or 0) - (r["min_cost"] or 0)
        avg = r["avg_cost"] or 1
        r["price_consistency_pct"] = round((1 - rng / avg) * 100, 1)
    return _json({"period_days": days_back, "suppliers": rows})


def tool_get_top_products(days_back: int = 30, limit: int = 10) -> str:
    """Top products by revenue with margin analysis."""
    rows = _q(
        f"SELECT p.name_ar, p.name_en, p.category, "
        f"ROUND(SUM(s.total_price),0) revenue, ROUND(SUM(s.gross_profit),0) profit, "
        f"ROUND(AVG(s.profit_margin_percent),1) margin, ROUND(SUM(s.weight_kg),1) kg "
        f"FROM sales s JOIN products p ON s.product_id=p.id "
        f"WHERE s.sale_date >= date('now','-{days_back} days') "
        f"GROUP BY s.product_id ORDER BY revenue DESC LIMIT {limit}"
    )
    return _json({"top_products": rows})


def tool_generate_alert_report() -> str:
    """Generate a prioritized alert report: low stock + upcoming demand spikes."""
    # Low stock alerts
    low_stock = _q(
        "SELECT p.name_ar, COALESCE(i.current_stock_kg,0) stock, p.reorder_point rp "
        "FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
        "WHERE COALESCE(i.current_stock_kg,0) < p.reorder_point "
        "ORDER BY COALESCE(i.current_stock_kg,0)/NULLIF(p.reorder_point,1) ASC LIMIT 10"
    )
    for r in low_stock:
        r["coverage_pct"] = round(r["stock"] / max(r["rp"], 1) * 100, 0)

    # Upcoming peak days (next 7 days)
    peak_days = []
    for i in range(7):
        d    = date.today() + timedelta(days=i)
        mult, lbl = _get_mult(d)
        if mult >= 1.3:
            peak_days.append({"date": str(d), "season": lbl, "multiplier": mult})

    # Yesterday vs today revenue
    today_rev  = _scalar(f"SELECT COALESCE(SUM(total_price),0) FROM sales WHERE sale_date=date('now')") or 0
    yest_rev   = _scalar(f"SELECT COALESCE(SUM(total_price),0) FROM sales WHERE sale_date=date('now','-1 day')") or 0
    rev_delta  = round((today_rev - yest_rev) / max(yest_rev, 1) * 100, 1)

    return _json({
        "generated_at"  : str(datetime.now())[:19],
        "low_stock_items": low_stock,
        "upcoming_peaks" : peak_days,
        "today_revenue" : {"today_egp": today_rev, "yesterday_egp": yest_rev, "change_pct": rev_delta},
        "action_needed" : len(low_stock) > 0 or len(peak_days) > 0,
    })


def tool_add_expense(
    expense_date: str,
    category: str,
    amount_egp: float,
    description: str,
    entered_by: str = "mcp_client",
) -> str:
    """Add a new expense record to the database."""
    try:
        conn = _db()
        conn.execute(
            "INSERT INTO expenses (expense_date, category, amount_egp, description, entered_by, is_live) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (expense_date, category, amount_egp, description, entered_by),
        )
        conn.commit()
        conn.close()
        return _json({"status": "success", "message": f"Expense recorded: {category} {amount_egp:.0f} EGP on {expense_date}"})
    except Exception as exc:
        return _json({"status": "error", "message": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def resource_database_schema() -> str:
    """Return the ButcherAI database schema as readable text."""
    try:
        conn   = _db()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        schema = {}
        for (tbl,) in tables:
            cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            schema[tbl] = [{"col": c[1], "type": c[2], "notnull": bool(c[3]), "pk": bool(c[5])} for c in cols]
        conn.close()
        return _json({"database": DB_PATH, "tables": schema})
    except Exception as exc:
        return _json({"error": str(exc)})


def resource_product_catalog() -> str:
    """Return the full product catalog with pricing and inventory."""
    rows = _q(
        "SELECT p.id, p.name_ar, p.name_en, p.category, "
        "p.avg_price, p.avg_cost, p.avg_margin, p.reorder_point, "
        "COALESCE(i.current_stock_kg, 0) current_stock "
        "FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
        "ORDER BY p.category, p.name_ar"
    )
    return _json({
        "total_products": len(rows),
        "categories": list(set(r["category"] for r in rows)),
        "products": rows,
    })


def resource_butchery_knowledge() -> str:
    """Return Egyptian butchery domain knowledge as a reference document."""
    return json.dumps({
        "title": "ButcherAI Egyptian Butchery Knowledge Base",
        "sections": {
            "carcass_yields": {
                "cattle_baladi": "70% yield from dressed weight",
                "lamb"         : "60% yield from dressed weight",
                "cut_distribution_cattle": {
                    "chuck/دوش": "20%", "ribs/ريش": "15%", "tenderloin/فيليه": "8%",
                    "round/موزة": "18%", "shoulder/كتف": "14%", "ground/مفروم": "15%",
                    "offal/كبدة": "5%", "processed/مشتقات": "5%",
                },
                "cut_distribution_lamb": {
                    "leg/فخدة": "35%", "shoulder/كتف": "25%", "rack/ريش": "20%",
                    "ground/مفروم": "12%", "offal/مشتقات": "8%",
                },
            },
            "demand_multipliers": {
                "eid_al_adha": "3.0x — highest peak of the year",
                "eid_al_fitr": "2.2x — 3 days",
                "ramadan"    : "1.55x — 30 days sustained",
                "friday"     : "1.30x — family meal tradition",
                "saturday"   : "1.15x",
                "sunday"     : "1.10x",
                "monday"     : "0.90x — slowest weekday",
                "other_days" : "1.00x baseline",
            },
            "healthy_margins": {
                "premium_cuts" : "35-45% (tenderloin, ribeye)",
                "standard_cuts": "20-30% (round, shoulder, chuck)",
                "ground_meat"  : "15-22% (high volume, low margin)",
                "offal"        : "40-55% (often overlooked)",
                "shop_average" : "22-28% target",
                "warning_below": "15%",
                "loss_below"   : "10%",
            },
            "inventory_rules": {
                "fresh_shelf_life"     : "2-3 days at 0-4°C",
                "frozen_shelf_life"    : "3 months at -18°C",
                "procurement_formula"  : "max(0, predicted_demand × 1.1 - current_stock)",
                "waste_target"         : "< 3% of revenue",
                "clearance_rule"       : "Mark down 15-20% at 4pm for same-day sale",
            },
            "procurement_timing": {
                "eid_al_adha"  : "Order 3 weeks in advance — suppliers run out 10 days before",
                "eid_al_fitr"  : "Order 1 week in advance",
                "ramadan"      : "Start 2 weeks before, increase daily orders 50%",
                "regular_days" : "Order daily for fresh, weekly for frozen",
            },
        },
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_MORNING_BRIEFING = {
    "name"       : "morning_briefing",
    "description": "Generate a comprehensive morning briefing for the butcher shop owner",
    "arguments"  : [],
    "template"   : (
        "You are ButcherAI. Generate a morning briefing for the Egyptian butcher shop owner. "
        "Use the available tools to:\n"
        "1. Check inventory and flag EMPTY/CRITICAL items\n"
        "2. Get today's demand multiplier (holiday/weekday)\n"
        "3. Compare yesterday's revenue to the previous day\n"
        "4. List the top 3 procurement priorities for today\n"
        "5. Check for any upcoming holidays in the next 7 days\n\n"
        "Format the briefing in Arabic with clear sections and actionable bullet points."
    ),
}

PROMPT_EID_PREPARATION = {
    "name"       : "eid_preparation",
    "description": "Generate a complete Eid procurement and operations plan",
    "arguments"  : [{"name": "eid_type", "description": "أضحى or فطر", "required": True}],
    "template"   : (
        "You are ButcherAI. Generate a complete Eid preparation plan for {eid_type}. "
        "Use tools to:\n"
        "1. Get current inventory levels for all products\n"
        "2. Get demand forecast for the 14 days including Eid\n"
        "3. Calculate procurement recommendations for each product\n"
        "4. Estimate total procurement cost\n"
        "5. Get WAC for cost benchmarking\n\n"
        "Output: week-by-week action plan in Arabic, with specific kg quantities and EGP amounts."
    ),
}

PROMPT_WEEKLY_REVIEW = {
    "name"       : "weekly_review",
    "description": "Generate a comprehensive weekly performance review",
    "arguments"  : [],
    "template"   : (
        "You are ButcherAI. Generate a complete weekly performance review. "
        "Use tools to:\n"
        "1. Query sales for last 7 days grouped by product\n"
        "2. Calculate P&L for last 7 days (gross profit minus expenses)\n"
        "3. Get WAC analysis to identify margin squeeze\n"
        "4. Rank suppliers by volume and cost consistency\n"
        "5. Compare this week's revenue to previous week\n\n"
        "Output: executive summary in Arabic with tables, key insights, and 3 action items for next week."
    ),
}

PROMPT_PROCUREMENT_OPTIMIZATION = {
    "name"       : "procurement_optimization",
    "description": "Optimize procurement based on current inventory and forecast",
    "arguments"  : [
        {"name": "budget_egp", "description": "Available procurement budget in EGP", "required": False}
    ],
    "template"   : (
        "You are ButcherAI. Optimize procurement within {'budget_egp'} EGP budget. "
        "Use tools to:\n"
        "1. Check all inventory levels\n"
        "2. Get 7-day procurement recommendations\n"
        "3. Prioritize by: urgency × margin × forecast demand\n"
        "4. If budget limited: allocate to highest-score products first\n"
        "5. Check holiday calendar for demand spikes\n\n"
        "Output: prioritized shopping list in Arabic with quantities, estimated costs, and supplier suggestions."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_server() -> "Server":
    """Build and configure the MCP server with all tools, resources, and prompts."""
    server = Server(SERVER_NAME)

    # ── List tools ────────────────────────────────────────────────────────────
    @server.list_tools()
    async def list_tools() -> List[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="get_inventory",
                description="Get real-time inventory levels for all products or a specific product. Returns stock_kg, alert levels (EMPTY/CRITICAL/LOW/OK).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string", "description": "Arabic product name filter (optional)"},
                        "low_stock_only": {"type": "boolean", "description": "Return only products below reorder point"},
                    },
                },
            ),
            mcp_types.Tool(
                name="query_sales",
                description="Query sales data with flexible grouping. Returns revenue, profit, weight, margin.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days_back": {"type": "integer", "description": "Days back from today (default 30)"},
                        "product_name": {"type": "string", "description": "Filter by Arabic product name"},
                        "group_by": {"type": "string", "enum": ["product", "day", "category"]},
                    },
                },
            ),
            mcp_types.Tool(
                name="get_demand_forecast",
                description="Forecast demand for a product for the next N days using historical data + seasonal multipliers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string", "description": "Arabic product name"},
                        "days": {"type": "integer", "description": "Forecast horizon (1-14 days)"},
                    },
                    "required": ["product_name"],
                },
            ),
            mcp_types.Tool(
                name="get_wac_analysis",
                description="Compute Weighted Average Cost (WAC) per product. Returns WAC, actual margin, margin gap.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "days_back": {"type": "integer"},
                    },
                },
            ),
            mcp_types.Tool(
                name="get_expenses",
                description="Query operational expenses by category and period.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days_back": {"type": "integer"},
                        "category": {"type": "string"},
                    },
                    "required": ["days_back"],
                },
            ),
            mcp_types.Tool(
                name="calculate_pnl",
                description="Calculate full P&L: revenue, gross profit, expenses, net profit.",
                inputSchema={
                    "type": "object",
                    "properties": {"days_back": {"type": "integer"}},
                    "required": ["days_back"],
                },
            ),
            mcp_types.Tool(
                name="get_procurement_plan",
                description="AI procurement plan combining inventory + forecast + holiday calendar. Returns prioritized buy recommendations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "days": {"type": "integer", "description": "Planning horizon (default 7)"},
                    },
                },
            ),
            mcp_types.Tool(
                name="get_holiday_calendar",
                description="Get demand multipliers for upcoming days. Identifies Eid, Ramadan, and high-demand weekdays.",
                inputSchema={
                    "type": "object",
                    "properties": {"days_ahead": {"type": "integer", "description": "Days to look ahead (default 14)"}},
                },
            ),
            mcp_types.Tool(
                name="get_supplier_performance",
                description="Analyze supplier performance: volume, average cost, price consistency.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days_back": {"type": "integer"},
                        "supplier_name": {"type": "string"},
                    },
                },
            ),
            mcp_types.Tool(
                name="get_top_products",
                description="Top products by revenue with margin analysis.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days_back": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                },
            ),
            mcp_types.Tool(
                name="generate_alert_report",
                description="Generate prioritized alert report: low stock + upcoming demand peaks + daily revenue summary.",
                inputSchema={"type": "object", "properties": {}},
            ),
            mcp_types.Tool(
                name="add_expense",
                description="Add an operational expense record to the database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expense_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "category"    : {"type": "string"},
                        "amount_egp"  : {"type": "number"},
                        "description" : {"type": "string"},
                    },
                    "required": ["expense_date", "category", "amount_egp"],
                },
            ),
        ]

    # ── Call tool ─────────────────────────────────────────────────────────────
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict) -> List[mcp_types.TextContent]:
        dispatch = {
            "get_inventory"       : lambda: tool_get_inventory(**arguments),
            "query_sales"         : lambda: tool_query_sales(**arguments),
            "get_demand_forecast" : lambda: tool_get_forecast(
                                        arguments.get("product_name", ""),
                                        arguments.get("days", 7)),
            "get_wac_analysis"    : lambda: tool_get_wac(**arguments),
            "get_expenses"        : lambda: tool_get_expenses(**arguments),
            "calculate_pnl"       : lambda: tool_calculate_pnl(**arguments),
            "get_procurement_plan": lambda: tool_get_procurement_plan(**arguments),
            "get_holiday_calendar": lambda: tool_get_holiday_calendar(**arguments),
            "get_supplier_performance": lambda: tool_get_supplier_performance(**arguments),
            "get_top_products"    : lambda: tool_get_top_products(**arguments),
            "generate_alert_report": lambda: tool_generate_alert_report(),
            "add_expense"         : lambda: tool_add_expense(**arguments),
        }
        fn = dispatch.get(name)
        if fn is None:
            result = _json({"error": f"Unknown tool: {name}"})
        else:
            try:
                result = fn()
            except Exception as exc:
                result = _json({"error": str(exc)})

        return [mcp_types.TextContent(type="text", text=result)]

    # ── List resources ────────────────────────────────────────────────────────
    @server.list_resources()
    async def list_resources() -> List[mcp_types.Resource]:
        return [
            mcp_types.Resource(
                uri=f"butcherai://schema",
                name="Database Schema",
                description="Complete ButcherAI SQLite database schema (6 tables)",
                mimeType="application/json",
            ),
            mcp_types.Resource(
                uri=f"butcherai://catalog",
                name="Product Catalog",
                description="Full product catalog with current inventory and pricing",
                mimeType="application/json",
            ),
            mcp_types.Resource(
                uri=f"butcherai://knowledge",
                name="Butchery Knowledge Base",
                description="Egyptian butchery domain knowledge: yields, margins, seasonal patterns",
                mimeType="application/json",
            ),
        ]

    # ── Read resource ─────────────────────────────────────────────────────────
    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "butcherai://schema":
            return resource_database_schema()
        if uri == "butcherai://catalog":
            return resource_product_catalog()
        if uri == "butcherai://knowledge":
            return resource_butchery_knowledge()
        return _json({"error": f"Unknown resource: {uri}"})

    # ── List prompts ──────────────────────────────────────────────────────────
    @server.list_prompts()
    async def list_prompts() -> List[mcp_types.Prompt]:
        templates = [
            PROMPT_MORNING_BRIEFING,
            PROMPT_EID_PREPARATION,
            PROMPT_WEEKLY_REVIEW,
            PROMPT_PROCUREMENT_OPTIMIZATION,
        ]
        prompts = []
        for t in templates:
            args = [
                mcp_types.PromptArgument(
                    name=a["name"],
                    description=a["description"],
                    required=a.get("required", False),
                )
                for a in t.get("arguments", [])
            ]
            prompts.append(
                mcp_types.Prompt(
                    name=t["name"],
                    description=t["description"],
                    arguments=args,
                )
            )
        return prompts

    # ── Get prompt ────────────────────────────────────────────────────────────
    @server.get_prompt()
    async def get_prompt(name: str, arguments: Optional[Dict] = None) -> mcp_types.GetPromptResult:
        templates = {
            t["name"]: t
            for t in [
                PROMPT_MORNING_BRIEFING, PROMPT_EID_PREPARATION,
                PROMPT_WEEKLY_REVIEW, PROMPT_PROCUREMENT_OPTIMIZATION,
            ]
        }
        if name not in templates:
            raise ValueError(f"Unknown prompt: {name}")

        tmpl = templates[name]
        text = tmpl["template"]
        if arguments:
            for k, v in arguments.items():
                text = text.replace(f"{{{k}}}", str(v))

        return mcp_types.GetPromptResult(
            description=tmpl["description"],
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=text),
                )
            ],
        )

    return server


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
async def _main():
    if not MCP_AVAILABLE:
        print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(
            f"ERROR: Database not found at '{DB_PATH}'. "
            f"Set DB_PATH environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    server = build_server()
    print(f"[ButcherAI MCP] Starting server v{SERVER_VERSION}", file=sys.stderr)
    print(f"[ButcherAI MCP] Database: {DB_PATH}", file=sys.stderr)
    print(f"[ButcherAI MCP] Tools: 12 | Resources: 3 | Prompts: 4", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        from mcp.server.models import InitializationOptions
        from mcp import ServerCapabilities
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=ServerCapabilities(
                    tools={"listChanged": False},
                    resources={"subscribe": False, "listChanged": False},
                    prompts={"listChanged": False},
                ),
            ),
        )


def main():
    import asyncio
    asyncio.run(_main())


if __name__ == "__main__":
    main()
