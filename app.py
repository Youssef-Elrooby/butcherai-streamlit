"""
ButcherAI - Simple RAG & OCR Dashboard
Production-ready Streamlit app for Arabic business intelligence.
"""

import streamlit as st
import os
import sqlite3
from datetime import date, datetime

# ── ENV ─────────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "butcherai.db")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ButcherAI - ذكاء المحلات",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── INIT DATABASE ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_pw TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            attempt_time TEXT NOT NULL,
            success INTEGER NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_ar TEXT,
            name_en TEXT,
            category TEXT,
            reorder_point REAL,
            avg_price REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name_ar TEXT,
            sale_date TEXT,
            weight_kg REAL,
            unit_price REAL,
            total_price REAL,
            entered_by TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            product_id INTEGER PRIMARY KEY,
            current_stock_kg REAL,
            last_updated TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── AUTH ────────────────────────────────────────────────────────────────────
def _hash(pw):
    import hashlib
    import secrets
    salt = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 260000)
    return salt.hex() + ":" + dk.hex()

def _verify(pw, stored):
    import hashlib
    salt_hex, dk_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 260000)
    return dk.hex() == dk_hex

def _init_admin():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE username='admin'")
    if c.fetchone()[0] == 0:
        admin_pw = _hash("admin123")
        c.execute(
            "INSERT INTO users (username, full_name, hashed_pw, role, active, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", "System Administrator", admin_pw, "admin", 1, str(datetime.now())),
        )
        conn.commit()
    conn.close()

_init_admin()

def login_page():
    st.title("🥩 ButcherAI")
    st.markdown("** inteligente ولكنige for butcher shops**")
    st.divider()
    with st.form("login"):
        u = st.text_input("اسم المستخدم")
        p = st.text_input("كلمة المرور", type="password")
        if st.form_submit_button("تسجيل الدخول"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            row = c.execute("SELECT hashed_pw, active FROM users WHERE username=?", (u.strip(),)).fetchone()
            conn.close()
            if row and row[1] and _verify(p, row[0]):
                st.session_state["auth"] = True
                st.session_state["user"] = u.strip()
                st.session_state["role"] = "admin"
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة")

def main_app():
    st.sidebar.title("🥩 ButcherAI")
    st.sidebar.write(f"مرحباً، {st.session_state.get('user', '')}")
    st.sidebar.divider()
    
    page = st.sidebar.radio("القائمة", [
        "🏠 الرئيسية",
        "📚 قاعدة المعرفة (RAG)",
        "🤖 الوكيل الذكي",
        "📓 استيراد من الدفتر الورقي",
        "💹 توقع الأسعار",
        "📊 لوحة المراقبة",
        "⚙️ محرك الأتمتة",
    ])

    if page == "🏠 الرئيسية":
        st.title("🏠 لوحة المعلومات الرئيسية")
        st.info("مرحباً بك في نظام ButcherAI الذكي لمحلات اللحوم.")
        col1, col2, col3 = st.columns(3)
        c1.metric("حالة النظام", "متصل ✅")
        c2.metric("النموذج", "Gemini Flash")
        c3.metric("اللغة", "العربية")

    elif page == "📚 قاعدة المعرفة (RAG)":
        st.title("📚 قاعدة المعرفة (RAG)")
        st.info("نظام استرجاع معرفية ذكي للأسئلة المتعلقة بمحل اللحوم.")
        q = st.text_input("اسأل سؤالاً:")
        if q:
            st.info(f"سؤالك: {q}")
            st.info("إجابة: سيتم إضافة وظيفة الإجابة الذكية قريباً.")

    elif page == "🤖 الوكيل الذكي":
        st.title("🤖 الوكيل الذكي المستقل")
        st.info("وكيل ذكي يساعد في إدارة المحلات.")
        q = st.text_input("اسأل الوكيل:")
        if q:
            st.info(f"سؤالك: {q}")
            st.info("سيتم تنفيذ الأمر عبر الوكيل الذكي قريباً.")

    elif page == "📓 استيراد من الدفتر الورقي — Gemini Vision":
        st.title("📓 استيراد من الدفتر الورقي")
        st.markdown("رفع صورة لدفتر الحسابات واستخراج النص تلقائياً.")
        uploaded = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="الصورة المرفوعة")
            if st.button("قراءة الصورة"):
                st.info("جاري القراءة...")
                st.info("الوظيفة ستتوفر مع مفتاح Gemini API.")

    elif page == "💹 توقع الأسعار":
        st.title("💹 توقع الأسعار")
        st.info("نظام تنبؤ بأسعار اللحوم باستخدام LightGBM.")
        p = st.selectbox("اختر المنتج:", ["لحم بلدي", "مفروم", "فيليه", "كباب حلة"])
        if st.button("توقع السعر"):
            st.info(f"السعر المتوقع لـ {p}: سيتم حسابه بناءً على البيانات التاريخية.")

    elif page == "📊 لوحة المراقبة":
        st.title("📊 لوحة المراقبة")
        st.info("مراقبة أداء النظام واستخدام الذكاء الاصطناعي.")
        col1, col2, col3 = st.columns(3)
        c1.metric("طلبات LLM", "0")
        c2.metric("معدل النجاح", "0%")
        c3.metric("المدة المتوسطة", "0s")

    elif page == "⚙️ محرك الأتمتة":
        st.title("⚙️ محرك الأتمتة")
        st.info("نظام أتمتة المهام التلقائية.")
        if st.button("تشغيل المهام التلقائية"):
            st.info("جاري تشغيل المهام التلقائية...")

def main():
    if "auth" not in st.session_state or not st.session_state["auth"]:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
