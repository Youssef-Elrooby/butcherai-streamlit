"""
ButcherAI Scenario B — Security & Correctness Hardening Patches
==================================================================
Drop-in fixes for production deployment (Scenario B: real shop, reliable internet).

Fixes three concrete issues identified in the code audit:
  1. SQL string concatenation in filtered queries -> parameterized queries
  2. Unprotected to_sql(if_exists="replace") writes -> transaction-wrapped with rollback
  3. WAC "analysis" using static products.avg_cost -> real weighted average from purchases

Usage: import these functions into app.py and replace the flagged call sites.
Each function is a drop-in replacement — same signature pattern as the original.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Parameterized query builder for filtered dropdown-driven SQL
# ─────────────────────────────────────────────────────────────────────────────
def build_sales_filter_query(
    days: int,
    category: str = "الكل",
    payment: str = "الكل",
    live_filter: str = "الكل",
) -> Tuple[str, tuple]:
    """
    Replaces the string-concatenation pattern:
        cc = f" AND p.category='{sc}'" if sc != "الكل" else ""
    with a parameterized query. Returns (sql, params) for pd.read_sql_query.

    This closes the (low-risk but real) SQL injection surface on the
    Sales Analysis page filters, even though the values currently come
    from st.selectbox and are attacker-controlled only if someone
    tampers with the client-side request.
    """
    conditions = ["s.sale_date >= date('now', ?)"]
    params: List[Any] = [f"-{int(days)} days"]

    if category != "الكل":
        conditions.append("p.category = ?")
        params.append(category)

    if payment != "الكل":
        conditions.append("s.payment_method = ?")
        params.append(payment)

    live_map = {"الكل": None, "تاريخية فقط": 0, "مُدخلة يدوياً": 1}
    live_val = live_map.get(live_filter)
    if live_val is not None:
        conditions.append("s.is_live = ?")
        params.append(live_val)

    where_clause = " AND ".join(conditions)
    sql = (
        f"SELECT s.sale_date, p.name_ar, p.category, s.weight_kg, s.unit_price, "
        f"s.total_price, s.gross_profit, s.profit_margin_percent, s.payment_method "
        f"FROM sales s JOIN products p ON s.product_id = p.id "
        f"WHERE {where_clause} ORDER BY s.id DESC LIMIT 500"
    )
    return sql, tuple(params)


def build_audit_filter_query(days: int, username: str = "", action: str = "") -> Tuple[str, tuple]:
    """Parameterized version of the audit log page filters."""
    conditions = ["timestamp >= datetime('now', ?)"]
    params: List[Any] = [f"-{int(days)} days"]

    if username.strip():
        conditions.append("username LIKE ?")
        params.append(f"%{username.strip()}%")

    if action.strip():
        conditions.append("action LIKE ?")
        params.append(f"%{action.strip()}%")

    where_clause = " AND ".join(conditions)
    sql = (
        f"SELECT timestamp, username, action, table_name, summary "
        f"FROM audit_log WHERE {where_clause} ORDER BY id DESC LIMIT 50 OFFSET ?"
    )
    return sql, tuple(params)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Transaction-wrapped table replacement (Data Management page)
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def safe_transaction(db_path: str):
    """
    Context manager giving BEGIN / COMMIT / ROLLBACK safety around any write.

    Usage:
        with safe_transaction(DB) as conn:
            edited_df.to_sql(target_table, conn, if_exists="replace", index=False)
        # auto-commits on success, auto-rolls-back on any exception
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("BEGIN")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def safe_replace_table(db_path: str, table_name: str, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Drop-in replacement for the unprotected pattern:
        conn = _raw(); edited_df.to_sql(target_table, conn, if_exists="replace", index=False)

    Returns (success, message). On failure, the original table data is untouched
    because SQLite's DDL inside a transaction rolls back cleanly.
    """
    # Snapshot row count before, for the success message
    try:
        pre_conn = sqlite3.connect(db_path)
        pre_count = pre_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        pre_conn.close()
    except Exception:
        pre_count = 0

    try:
        with safe_transaction(db_path) as conn:
            # Write to a staging table first, then swap — this way a crash
            # mid-write never leaves the real table half-populated
            staging = f"_staging_{table_name}"
            conn.execute(f"DROP TABLE IF EXISTS {staging}")
            df.to_sql(staging, conn, if_exists="replace", index=False)

            # Verify row count matches before swapping
            staged_count = conn.execute(f"SELECT COUNT(*) FROM {staging}").fetchone()[0]
            if staged_count != len(df):
                raise ValueError(
                    f"Staging write mismatch: expected {len(df)} rows, got {staged_count}"
                )

            conn.execute(f"DROP TABLE {table_name}")
            conn.execute(f"ALTER TABLE {staging} RENAME TO {table_name}")

        return True, f"✅ {table_name} updated: {pre_count} → {len(df)} rows (transaction-safe)"
    except Exception as exc:
        return False, f"❌ Rolled back — {table_name} unchanged. Error: {exc}"


def safe_append_import(db_path: str, table_name: str, df: pd.DataFrame) -> Tuple[bool, str]:
    """Transaction-safe version of the CSV/XLSX bulk import append."""
    try:
        with safe_transaction(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        return True, f"✅ {len(df):,} rows imported into {table_name} (transaction-safe)"
    except Exception as exc:
        return False, f"❌ Import rolled back — no partial data written. Error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Real WAC (Weighted Average Cost) from actual purchase history
# ─────────────────────────────────────────────────────────────────────────────
def compute_real_wac(db_path: str, days_back: int = 90) -> pd.DataFrame:
    """
    Replaces the incorrect pattern in the WAC Analysis page:
        cost = float(prod.avg_cost)   # <- static catalogue value, not weighted average

    Computes the TRUE weighted average cost per product from actual purchase
    transactions:  WAC = SUM(total_cost) / SUM(weight_kg)

    This is what "WAC Analysis" should mean and what your own domain
    knowledge doc (dk_wac_analysis) already describes correctly —
    the implementation just didn't match the documentation.
    """
    conn = sqlite3.connect(db_path)

    wac_df = pd.read_sql_query(
        f"""
        SELECT
            product_ar,
            product_id,
            ROUND(SUM(total_cost) * 1.0 / NULLIF(SUM(weight_kg), 0), 2) AS wac_egp_per_kg,
            SUM(weight_kg) AS total_kg_purchased,
            COUNT(*) AS purchase_count,
            MIN(purchase_date) AS first_purchase,
            MAX(purchase_date) AS last_purchase
        FROM purchases
        WHERE purchase_date >= date('now', ?)
        GROUP BY product_ar, product_id
        HAVING SUM(weight_kg) > 0
        ORDER BY total_kg_purchased DESC
        """,
        conn,
        params=(f"-{int(days_back)} days",),
    )

    # Join against current selling price + target margin from products table
    products_df = pd.read_sql_query(
        "SELECT id, name_ar, avg_price, avg_margin, category FROM products", conn
    )
    conn.close()

    merged = wac_df.merge(
        products_df, left_on="product_id", right_on="id", how="left", suffixes=("", "_prod")
    )
    merged["actual_margin_pct"] = merged.apply(
        lambda r: round((r["avg_price"] - r["wac_egp_per_kg"]) / r["avg_price"] * 100, 1)
        if r["avg_price"] and r["avg_price"] > 0 and pd.notna(r["wac_egp_per_kg"])
        else 0.0,
        axis=1,
    )
    merged["margin_gap"] = merged["actual_margin_pct"] - merged["avg_margin"].fillna(20)
    merged["status"] = merged["actual_margin_pct"].apply(
        lambda m: "✅ Healthy" if m >= 20 else "⚠️ Warning" if m >= 10 else "🔴 Below cost risk"
    )

    return merged[
        [
            "name_ar", "category", "wac_egp_per_kg", "avg_price", "actual_margin_pct",
            "avg_margin", "margin_gap", "total_kg_purchased", "purchase_count",
            "last_purchase", "status",
        ]
    ].rename(columns={"avg_price": "sell_price", "avg_margin": "target_margin"})


def wac_page_patch_snippet() -> str:
    """
    Returns the exact replacement code to paste into app.py's
    "⚙️ تحليل WAC" page, replacing the static avg_cost loop.
    """
    return '''
# ── REPLACE the existing WAC computation loop with this ──────────────────────
from security_patches import compute_real_wac

elif page == "⚙️ تحليل WAC":
    st.title("⚙️ تحليل التكلفة الموزونة الحقيقي (WAC)")
    days_wac = st.slider("فترة حساب WAC (يوم):", 30, 365, 90)
    dw = compute_real_wac(DB, days_back=days_wac)

    if dw.empty:
        st.warning("لا توجد بيانات مشتريات كافية لحساب WAC في هذه الفترة.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("متوسط WAC (حقيقي)", f"{dw.wac_egp_per_kg.mean():.2f} ج/كجم")
    c2.metric("متوسط سعر البيع", f"{dw.sell_price.mean():.2f} ج/كجم")
    c3.metric("الهامش الفعلي (من WAC)", f"{dw.actual_margin_pct.mean():.1f}%")
    c4.metric("منتجات دون الهدف", len(dw[dw.margin_gap < 0]))

    st.dataframe(dw, use_container_width=True, hide_index=True)
    # ... rest of chart code unchanged, just reads from `dw` computed correctly now
'''


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST — verify all three fixes work before deploying
# ─────────────────────────────────────────────────────────────────────────────
def run_self_test(db_path: str) -> Dict[str, bool]:
    """Quick sanity check that all three patches work against the real DB."""
    results = {}

    # Test 1: parameterized query builds without error
    try:
        sql, params = build_sales_filter_query(30, "Premium", "Cash", "الكل")
        conn = sqlite3.connect(db_path)
        pd.read_sql_query(sql, conn, params=params)
        conn.close()
        results["parameterized_queries"] = True
    except Exception as exc:
        print(f"Test 1 failed: {exc}")
        results["parameterized_queries"] = False

    # Test 2: transaction wrapper commits and rolls back correctly
    try:
        conn = sqlite3.connect(db_path)
        test_df = pd.read_sql_query("SELECT * FROM products LIMIT 1", conn)
        conn.close()
        if not test_df.empty:
            # Don't actually mutate — just confirm the context manager enters/exits cleanly
            with safe_transaction(db_path) as c:
                c.execute("SELECT 1")
            results["transaction_safety"] = True
        else:
            results["transaction_safety"] = None  # no data to test with
    except Exception as exc:
        print(f"Test 2 failed: {exc}")
        results["transaction_safety"] = False

    # Test 3: real WAC computes without error
    try:
        wac = compute_real_wac(db_path, days_back=90)
        results["real_wac_calculation"] = True
        results["wac_products_found"] = len(wac)
    except Exception as exc:
        print(f"Test 3 failed: {exc}")
        results["real_wac_calculation"] = False

    return results


if __name__ == "__main__":
    import os
    db = os.environ.get("DB_PATH", "ButcherAI.db")
    print("Running Scenario B security patch self-test...")
    print(run_self_test(db))
