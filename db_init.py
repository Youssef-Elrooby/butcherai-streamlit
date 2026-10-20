"""
ButcherAI — Database Initialization
=====================================
Creates all tables and seeds default data on first run.
Safe to call multiple times (idempotent).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime


def _hash(pw: str) -> str:
    """Hash a password using PBKDF2 (matches app.py's _verify)."""
    s = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), s, 260_000)
    return s.hex() + ":" + dk.hex()


def init_database(db_path: str = "ButcherAI.db") -> dict:
    """
    Initialize all database tables and seed default data.

    Returns:
        {"success": bool, "tables_created": list, "admin_created": bool, "error": str}
    """
    result = {"success": False, "tables_created": [], "admin_created": False, "error": ""}

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.executescript("""
            -- 1. Products
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name_ar TEXT, name_en TEXT, category TEXT,
                reorder_point REAL, avg_price REAL, avg_cost REAL, avg_margin REAL
            );

            -- 2. Sales
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER, product_name_ar TEXT, product_name_en TEXT,
                category TEXT, sale_date TEXT, weight_kg REAL, unit_price REAL,
                total_price REAL, unit_cost REAL, total_cost REAL,
                gross_profit REAL, profit_margin_percent REAL,
                payment_method TEXT, entered_by TEXT, is_live INTEGER
            );

            -- 3. Purchases
            CREATE TABLE IF NOT EXISTS purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purchase_date TEXT, supplier_name TEXT, product_ar TEXT,
                product_id INTEGER, weight_kg REAL, cost_per_kg REAL,
                total_cost REAL, notes TEXT, entered_by TEXT, is_live INTEGER
            );

            -- 4. Inventory
            CREATE TABLE IF NOT EXISTS inventory (
                product_id INTEGER PRIMARY KEY, current_stock_kg REAL, last_updated TEXT
            );

            -- 5. Daily Balance
            CREATE TABLE IF NOT EXISTS daily_balance (
                balance_date TEXT PRIMARY KEY, purchases_net_kg REAL, sales_kg REAL,
                daily_surplus_deficit REAL, cumulative_balance REAL
            );

            -- 6. Expenses
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, expense_date TEXT NOT NULL,
                category TEXT NOT NULL, amount_egp REAL NOT NULL,
                description TEXT, entered_by TEXT, is_live INTEGER DEFAULT 1
            );

            -- 7. Suppliers
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name_ar TEXT NOT NULL,
                name_en TEXT, phone TEXT, area TEXT, payment_terms TEXT,
                notes TEXT, active INTEGER DEFAULT 1
            );

            -- 8. Audit Log
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                username TEXT, action TEXT NOT NULL, table_name TEXT,
                record_id INTEGER, summary TEXT
            );

            -- 9. Login Attempts
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL,
                attempt_time TEXT NOT NULL, success INTEGER NOT NULL
            );

            -- 10. RAG Query Log
            CREATE TABLE IF NOT EXISTS rag_query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                username TEXT, question TEXT NOT NULL, top_doc_id TEXT,
                top_category TEXT, relevance REAL, n_docs_used INTEGER DEFAULT 0,
                had_api_key INTEGER DEFAULT 0, answer_len INTEGER DEFAULT 0
            );

            -- 11. Agent Sessions
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')), username TEXT,
                user_message TEXT NOT NULL, agent_answer TEXT NOT NULL,
                tools_used TEXT, turn_count INTEGER DEFAULT 0, elapsed_sec REAL DEFAULT 0
            );

            -- 12. Agent Tool Calls
            CREATE TABLE IF NOT EXISTS agent_tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')), tool_name TEXT NOT NULL,
                tool_args TEXT, tool_result TEXT, elapsed_sec REAL DEFAULT 0, had_error INTEGER DEFAULT 0
            );

            -- 13. MCP Access Log
            CREATE TABLE IF NOT EXISTS mcp_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                client_id TEXT, tool_name TEXT NOT NULL, arguments TEXT,
                result_size INTEGER DEFAULT 0, elapsed_ms INTEGER DEFAULT 0, had_error INTEGER DEFAULT 0
            );

            -- 14. LLM Calls Log
            CREATE TABLE IF NOT EXISTS llm_calls_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                feature TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER,
                output_tokens INTEGER, elapsed_sec REAL, estimated_cost_usd REAL,
                success INTEGER NOT NULL, error_message TEXT
            );

            -- 15. Users
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL, full_name TEXT, hashed_pw TEXT NOT NULL,
                role TEXT DEFAULT 'staff', active INTEGER DEFAULT 1,
                created_at TEXT, last_login TEXT, first_login INTEGER DEFAULT 0
            );

            -- 16. Automation tables
            CREATE TABLE IF NOT EXISTS automation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL, status TEXT NOT NULL,
                message TEXT, details TEXT, timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failure_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature TEXT NOT NULL, question TEXT, expected TEXT, actual TEXT,
                error TEXT, count INTEGER DEFAULT 1, status TEXT DEFAULT 'open',
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS automation_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL, title TEXT NOT NULL, message TEXT NOT NULL,
                severity TEXT DEFAULT 'info', is_read INTEGER DEFAULT 0, timestamp TEXT NOT NULL
            );
        """)

        # Indexes
        for name, table, col in [
            ("idx_sales_date", "sales", "sale_date"),
            ("idx_sales_product", "sales", "product_id"),
            ("idx_purchases_date", "purchases", "purchase_date"),
            ("idx_purchases_product", "purchases", "product_id"),
            ("idx_expenses_date", "expenses", "expense_date"),
            ("idx_login_attempts_user_time", "login_attempts", "username, attempt_time"),
        ]:
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({col})")
            except Exception:
                pass

        conn.commit()

        # Seed basic products if empty
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            basic_products = [
                ("لحم مفروم", "Ground Beef", "Ground", 10.0, 350.0, 280.0, 20.0),
                ("ستيك", "Steak", "Premium", 10.0, 450.0, 350.0, 22.0),
                ("كباب حلة", "Kebab", "Premium", 10.0, 400.0, 320.0, 20.0),
                ("لحم ضاني", "Lamb", "Standard", 15.0, 420.0, 340.0, 19.0),
                ("كبدة بلدي", "Liver", "Offal", 5.0, 380.0, 290.0, 23.0),
                ("لحم بلدي", "Baladi Cattle", "Carcass", 50.0, 300.0, 250.0, 16.0),
                ("دوش", "Chuck", "Standard", 10.0, 360.0, 290.0, 19.0),
                ("ريش", "Ribs", "Standard", 10.0, 380.0, 300.0, 21.0),
                ("فيليه", "Fillet", "Premium", 5.0, 460.0, 360.0, 21.0),
                ("موزة", "Round", "Standard", 10.0, 390.0, 310.0, 20.0),
                ("كتف", "Shoulder", "Standard", 10.0, 370.0, 300.0, 19.0),
                ("مشتقات", "Processed", "Processed", 15.0, 250.0, 200.0, 20.0),
            ]
            for p in basic_products:
                conn.execute(
                    "INSERT INTO products (name_ar, name_en, category, reorder_point, avg_price, avg_cost, avg_margin) "
                    "VALUES (?,?,?,?,?,?,?)", p
                )
            conn.commit()

        # Create admin user if none exists
        admin_exists = conn.execute(
            "SELECT COUNT(*) FROM users WHERE username='admin'"
        ).fetchone()[0]
        if admin_exists == 0:
            admin_pw = _hash("admin123")
            conn.execute(
                "INSERT INTO users (username, full_name, hashed_pw, role, active, created_at, first_login) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("admin", "System Administrator", admin_pw, "admin", 1, str(datetime.now()), 1),
            )
            conn.commit()
            result["admin_created"] = True

        # Get list of tables
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        result["tables_created"] = tables
        result["success"] = True
        conn.close()

    except Exception as exc:
        result["error"] = str(exc)

    return result


if __name__ == "__main__":
    res = init_database()
    print(res)
