"""
ButcherAI v9.1 — Enhanced & Secured Streamlit Dashboard
========================================================
Phase 0 Security Fixes applied:
  • API key read from .env / UI input — never hardcoded
  • CSV import: column whitelist + type validation + row cap
  • First-login password change enforced
  • Gemini API rate limiting via session cache

Phase 1 Core Completions added:
  • FR-08: Full expense tracking (table, page, P&L integration)
  • DB-04: Automated rolling backup with restore UI
  • Audit logging for every write operation
  • SHAP feature-contribution panel on Forecasting page
  • Pagination on Sales Analysis (100 rows/page, unlimited depth)
  • Python rotating log file (butcherai.log)
  • Supplier management page
  • Input validation (whitelist, types, row limits) on import
  • PDF report generation (English bilingual)

New pages (19 total, up from 15):
  💸 إدارة المصروفات    — Expense entry + monthly chart (FR-08)
  🚚 إدارة الموردين     — Supplier performance dashboard
  💾 النسخ الاحتياطي   — DB backup / restore (Admin only)
  📜 سجل التدقيق        — Audit trail viewer (Admin only)
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib
import secrets
import os
import sys
import json
import shutil
import warnings
import time
import logging
import joblib
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
# ── Phase 1: RAG Engine ─────────────────────────────────────────────────────
try:
    from phase1_rag_engine import get_rag, show_rag_page, rag_answer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Phase 2: Agentic AI ──────────────────────────────────────────────────────
try:
    from phase2_agent_page import show_agent_page, ButcherAIAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ── Phase 4: Multi-Agent ─────────────────────────────────────────────────────
try:
    from phase4_multi_agent import show_multi_agent_page
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

# ── Improved Arabic OCR ──────────────────────────────────────────────────────
try:
    from arabic_ocr_improved import ArabicOCRProcessor
    IMPROVED_OCR_AVAILABLE = True
except ImportError:
    IMPROVED_OCR_AVAILABLE = False

# ── Monitoring Dashboard ──────────────────────────────────────────────────────
try:
    from monitoring_dashboard import show_monitoring_page
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# ── Price Forecasting ─────────────────────────────────────────────────────────
try:
    from price_forecast import show_price_forecast_page
    PRICE_FORECAST_AVAILABLE = True
except ImportError:
    PRICE_FORECAST_AVAILABLE = False

# ── Automation Engine ────────────────────────────────────────────────────────
try:
    from automation_engine import show_automation_page
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

# ── Unified Deployment Scenario (A/B/C/D) ────────────────────────────────────
try:
    from deployment_config import get_active_scenario, get_llm_backend, get_embed_backend, show_scenario_widget
    ACTIVE_SCENARIO = get_active_scenario()
except ImportError:
    ACTIVE_SCENARIO = None

# ── Scenario B: Security & correctness patches ───────────────────────────────
try:
    from security_patches import build_sales_filter_query, safe_replace_table, safe_append_import, compute_real_wac
    SECURITY_PATCHES_AVAILABLE = True
except ImportError:
    SECURITY_PATCHES_AVAILABLE = False

# ── Scenario C: Offline Ollama stack ──────────────────────────────────────────
try:
    from phase_c_ollama_offline import check_ollama_available, OfflineButcherAIAgent, ollama_rag_answer, show_offline_status_widget
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ── Scenario D: Low-spec feature flags ────────────────────────────────────────
try:
    from phase_d_lowspec_config import apply_lowspec_page_filter, show_lowspec_sidebar_widget, get_active_feature_set
    LOWSPEC_CONFIG_AVAILABLE = True
except ImportError:
    LOWSPEC_CONFIG_AVAILABLE = False



# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

# ── Hijri calendar ────────────────────────────────────────────────────────────
try:
    from hijri_converter import Gregorian
    HAS_HIJRI = True
except ImportError:
    HAS_HIJRI = False

# ── Google Gemini ─────────────────────────────────────────────────────────────
try:
    import genai_compat as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP  (Phase 1 — persistent rotating log)
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = os.environ.get("LOG_FILE", "butcherai.log")
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

logger = logging.getLogger("ButcherAI")
if not logger.handlers:
    logger.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh  = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
import os
from pathlib import Path

# Get the absolute path to the directory where this file (app.py) lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import os
from pathlib import Path
import streamlit as st

# Get the absolute path to the directory where this file (app.py) lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "ButcherAI.db")

# Force the app to use the local file, ignoring any hidden .env settings
DB = DEFAULT_DB_PATH

# Print it to the terminal so we can see exactly where it is trying to save
print(f"🟢 DEBUG - SQLite is trying to open: {DB}")

# It's safest to do the same for backups so they don't get lost
DEFAULT_BACKUP_PATH = os.path.join(BASE_DIR, "backups")
BACKUP_DIR     = Path(os.environ.get("BACKUP_DIR", DEFAULT_BACKUP_PATH))

BACKUP_KEEP    = int(os.environ.get("BACKUP_KEEP_DAYS", "30"))
GEMINI_API_ENV = os.environ.get("GEMINI_API_KEY", "")  # Phase 0: from .env only
GEMINI_MODEL   = "gemini-flash-lite-latest"            # single source of truth

st.set_page_config(
    page_title="ButcherAI v9.1",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
html,body,[class*=css]{font-family:'Cairo',sans-serif;}
.main .block-container{padding-top:0.5rem;}
.auth-wrap{max-width:440px;margin:60px auto;}
.auth-card{background:#fff;border-radius:16px;padding:36px;
           box-shadow:0 8px 32px rgba(139,26,26,0.12);}
[data-testid="metric-container"]{
    background:linear-gradient(135deg,#8B1A1A,#C0392B);
    border-radius:12px;padding:16px;box-shadow:0 4px 14px rgba(0,0,0,0.18);}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricDelta"]{color:#fff!important;}
.mkt-box{background:#fafafa;border-right:5px solid #8B1A1A;border-radius:10px;
         padding:16px 20px;margin:8px 0;direction:rtl;text-align:right;
         white-space:pre-line;line-height:1.8;}
.mkt-en{background:#f0f8ff;border-left:5px solid #1A5276;border-radius:10px;
        padding:16px 20px;margin:8px 0;direction:ltr;text-align:left;
        white-space:pre-line;line-height:1.8;}
.rec-box{background:#fff8f8;border-right:5px solid #8B1A1A;border-radius:10px;
         padding:14px 18px;margin:6px 0;direction:rtl;text-align:right;
         white-space:pre-line;line-height:1.7;}
.ok{background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:12px 16px;
    direction:rtl;text-align:right;font-weight:600;color:#155724;}
.warn-box{background:#fff3cd;border:2px solid #ffc107;border-radius:8px;
          padding:10px 14px;margin:6px 0;font-size:14px;}
.info-box{background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;
          padding:10px 14px;margin:6px 0;font-size:14px;}
.ent-card{background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);
          border-radius:12px;padding:18px 22px;margin:6px 0;color:#fff;}
.exp-box{background:#fff8e1;border:2px solid #ffc107;border-radius:10px;
         padding:14px 18px;margin:6px 0;}
.audit-row{background:#f8f9fa;border-radius:8px;padding:8px 14px;margin:3px 0;
           font-size:13px;border-left:3px solid #8B1A1A;}
h1,h2,h3{color:#8B1A1A!important;}
.version-badge{background:#8B1A1A;color:#fff;font-size:10px;padding:2px 8px;
               border-radius:10px;vertical-align:middle;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN WHITELIST  (Phase 0 — CSV import validation)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_WHITELIST = {
    "sales": {
        "product_id","product_name_ar","product_name_en","category",
        "sale_date","weight_kg","unit_price","total_price","unit_cost",
        "total_cost","gross_profit","profit_margin_percent",
        "payment_method","entered_by","is_live"
    },
    "purchases": {
        "purchase_date","supplier_name","product_ar","product_id",
        "weight_kg","cost_per_kg","total_cost","notes","entered_by","is_live"
    },
    "daily_balance": {
        "balance_date","purchases_net_kg","sales_kg",
        "daily_surplus_deficit","cumulative_balance"
    },
    "inventory": {
        "product_id","current_stock_kg","last_updated"
    },
    "expenses": {
        "expense_date","category","amount_egp","description","entered_by","is_live"
    },
}

EXPENSE_CATS = ["مواد تشغيل", "تعبئة وتغليف", "نقل وشحن", "رواتب",
                "إيجار", "كهرباء وماء", "صيانة", "تسويق", "أخرى"]

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC HOLIDAY CALENDAR  (Hijri-converter based — self-updating)
# ─────────────────────────────────────────────────────────────────────────────
def get_mult(d):
    if isinstance(d, str):   d = datetime.strptime(d, "%Y-%m-%d").date()
    elif hasattr(d, "date"): d = d.date()
    if HAS_HIJRI:
        try:
            h = Gregorian(d.year, d.month, d.day).to_hijri()
            if h.month == 9:                               return 1.55, f"رمضان {h.year}"
            if h.month == 10 and 1 <= h.day <= 3:         return 2.20, f"عيد الفطر {h.year}"
            if h.month == 12 and 10 <= h.day <= 13:       return 3.00, f"عيد الأضحى {h.year}"
        except Exception:
            pass
    wd = d.weekday()
    if wd == 4: return 1.30, "الجمعة"
    if wd == 5: return 1.15, "السبت"
    if wd == 6: return 1.10, "الأحد"
    if wd == 0: return 0.90, "الإثنين"
    return 1.00, "يوم عادي"

def is_ramadan(d: date) -> int:
    if not HAS_HIJRI: return 0
    try:
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        return 1 if h.month == 9 else 0
    except Exception:
        return 0

# ─────────────────────────────────────────────────────────────────────────────
# CARCASS YIELD & CUT MAPPING  (DR-01, DR-03)
# ─────────────────────────────────────────────────────────────────────────────
CARCASS_YIELD = {
    "لحم بلدي": 0.70,
    "لحم ضاني": 0.60,
}
CARCASS_CUT_MAP = {
    "لحم بلدي": [
        ("دوش",    0.20), ("ريش",   0.15), ("فيليه",     0.08),
        ("موزة",   0.18), ("كتف",   0.14), ("مفروم",     0.15),
        ("كبدة",   0.05), ("مشتقات",0.05),
    ],
    "لحم ضاني": [
        ("فخدة ضاني",0.35), ("كتف ضاني", 0.25), ("ريش ضاني",  0.20),
        ("مفروم ضاني",0.12), ("مشتقات",   0.08),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTERPRISE MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_enterprise_model():
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        mdl  = joblib.load(os.path.join(cur, "enterprise_lightgbm_model.pkl"))
        feat = joblib.load(os.path.join(cur, "enterprise_model_features.pkl"))
        logger.info(f"Model loaded: {len(feat)} features")
        return mdl, feat
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None, None

ENT_MODEL, ENT_FEATURES = load_enterprise_model()

ENT_FEAT_DESC = {
    "lag_1":"مبيعات اليوم السابق","lag_2":"مبيعات قبل يومين",
    "lag_3":"مبيعات قبل 3 أيام","lag_4":"مبيعات قبل 4 أيام",
    "lag_7":"مبيعات قبل 7 أيام (أسبوع)","lag_14":"مبيعات قبل 14 يوماً",
    "rolling_3":"متوسط آخر 3 أيام","rolling_6":"متوسط آخر 6 أيام",
    "rolling_14":"متوسط آخر 14 يوماً","rolling_std_7":"تذبذب آخر 7 أيام",
    "inventory_pressure":"ضغط المخزون","inventory_trend":"اتجاه المخزون",
    "price_volatility":"تقلب الأسعار","price_trend":"اتجاه الأسعار",
    "price_level":"مستوى السعر النسبي","procurement_cost_trend":"اتجاه تكلفة الشراء",
    "cost_margin_ratio":"نسبة هامش التكلفة","supply_gap":"فجوة العرض والطلب",
    "demand_elasticity":"مرونة الطلب","stock_cover_days":"أيام تغطية المخزون",
    "trend_strength":"قوة الاتجاه","trend_ratio":"نسبة الاتجاه قصير/طويل",
    "week_sin":"مكوّن sin الأسبوعي","week_cos":"مكوّن cos الأسبوعي",
    "month_sin":"مكوّن sin الشهري","month_cos":"مكوّن cos الشهري",
    "is_ramadan":"مؤشر رمضان (0/1)","is_weekend":"مؤشر عطلة نهاية الأسبوع",
    "days_to_next_holiday":"أيام حتى العيد القادم",
}

def _compute_features(kg_vals, price_vals, cost_vals, inv_stock, step_date):
    """Compute all up-to-29 features for a single forecast step."""
    tail   = kg_vals[-max(14, 1):]
    l1  = tail[-1]  if len(tail)>=1  else 0.0
    l2  = tail[-2]  if len(tail)>=2  else l1
    l3  = tail[-3]  if len(tail)>=3  else l2
    l4  = tail[-4]  if len(tail)>=4  else l3
    l7  = tail[-7]  if len(tail)>=7  else l4
    l14 = tail[-14] if len(tail)>=14 else l7
    r3  = float(np.mean(tail[-3:]))  if len(tail)>=3  else l1
    r6  = float(np.mean(tail[-6:]))  if len(tail)>=6  else r3
    r14 = float(np.mean(tail[-14:])) if len(tail)>=14 else r6
    std7= float(np.std(tail[-7:]))   if len(tail)>=7  else 0.0
    da  = r6 if r6>0 else 1.0
    inv_p  = inv_stock/da
    inv_t  = (r3-r6)/(r6+1e-5)
    p_now  = price_vals[-1] if price_vals else 100.0
    p_wk   = float(np.mean(price_vals[-7:]))  if len(price_vals)>=7  else p_now
    p_m30  = float(np.mean(price_vals[-30:])) if len(price_vals)>=30 else p_now
    p_vol  = float(np.std(price_vals[-7:]))   if len(price_vals)>=7  else 0.0
    p_tr   = (p_now-p_wk)/(p_wk+1e-5)
    p_lv   = p_now/(p_m30+1e-5)
    c_now  = cost_vals[-1] if cost_vals else 80.0
    c_wk   = float(np.mean(cost_vals[-7:])) if len(cost_vals)>=7 else c_now
    c_tr   = (c_now-c_wk)/(c_wk+1e-5)
    margin = (p_now-c_now)/(p_now+1e-5)
    sgap   = inv_stock - da*3
    elast  = -(p_tr/(inv_t+1e-5)) if abs(inv_t)>1e-4 else 0.0
    scov   = inv_stock/da
    la     = float(np.mean(tail[-30:])) if len(tail)>=30 else r6
    ts     = (r6-la)/(la+1e-5)
    tr     = r3/(r6+1e-5)
    doy    = step_date.timetuple().tm_yday
    ws     = float(np.sin(2*np.pi*doy/7))
    wc     = float(np.cos(2*np.pi*doy/7))
    ms     = float(np.sin(2*np.pi*step_date.month/12))
    mc     = float(np.cos(2*np.pi*step_date.month/12))
    ram    = float(is_ramadan(step_date))
    iswkd  = float(1 if step_date.weekday() in {4,5,6} else 0)
    # Approximate days to next holiday
    try:
        if HAS_HIJRI:
            h = Gregorian(step_date.year, step_date.month, step_date.day).to_hijri()
            if h.month == 9:   d2h = float(max(0, 30 - h.day))
            elif h.month < 12: d2h = float((12-h.month)*30 + max(0, 10-h.day))
            else:              d2h = 365.0
        else: d2h = 30.0
    except Exception: d2h = 30.0

    full_29 = [l1,l2,l3,l4,l7,l14, r3,r6,r14,std7,
               inv_p,inv_t, p_vol,p_tr,p_lv, c_tr,margin,
               sgap,elast,scov, ts,tr, ws,wc,ms,mc, ram,iswkd,d2h]

    # Select only the features the loaded model expects
    if ENT_FEATURES:
        feat_map = dict(zip(
            ["lag_1","lag_2","lag_3","lag_4","lag_7","lag_14",
             "rolling_3","rolling_6","rolling_14","rolling_std_7",
             "inventory_pressure","inventory_trend",
             "price_volatility","price_trend","price_level",
             "procurement_cost_trend","cost_margin_ratio",
             "supply_gap","demand_elasticity","stock_cover_days",
             "trend_strength","trend_ratio",
             "week_sin","week_cos","month_sin","month_cos",
             "is_ramadan","is_weekend","days_to_next_holiday"],
            full_29))
        return [feat_map.get(f, 0.0) for f in ENT_FEATURES]
    return full_29[:18]  # fallback to 18 if no feature list

def enterprise_forecast(pid: int, days: int) -> list:
    if ENT_MODEL is None: return []
    cx = sqlite3.connect(DB)
    df = pd.read_sql_query(
        f"SELECT sale_date,SUM(weight_kg) kg,AVG(unit_price) price,AVG(unit_cost) cost "
        f"FROM sales WHERE product_id={pid} GROUP BY sale_date ORDER BY sale_date", cx)
    inv_row = cx.execute("SELECT current_stock_kg FROM inventory WHERE product_id=?", (pid,)).fetchone()
    cx.close()
    if df.empty or len(df) < 6: return []
    df["sale_date"] = pd.to_datetime(df["sale_date"],format="mixed",errors="coerce").dt.date
    full = pd.date_range(df.sale_date.min(), df.sale_date.max())
    df   = df.set_index("sale_date").reindex([d.date() for d in full])
    df["kg"]    = df["kg"].fillna(0.0)
    df["price"] = df["price"].ffill().bfill().fillna(100.0)
    df["cost"]  = df["cost"].ffill().bfill().fillna(80.0)
    df = df.reset_index(); df.columns = ["sale_date","kg","price","cost"]
    kg_vals=df["kg"].values.tolist(); price_vals=df["price"].values.tolist()
    cost_vals=df["cost"].values.tolist(); inv_stock=float(inv_row[0]) if inv_row else 0.0
    last_date=df["sale_date"].iloc[-1]; out=[]
    for step in range(1, days+1):
        fd   = last_date + timedelta(days=step)
        row  = _compute_features(kg_vals, price_vals, cost_vals, inv_stock, fd)
        try:
            pred = float(max(0.0, ENT_MODEL.predict(np.array([row]))[0]))
        except Exception:
            pred = float(np.mean(kg_vals[-3:])) if kg_vals else 0.0
        mult, slbl = get_mult(fd)
        pred_adj   = pred * mult
        std_est    = float(np.std(kg_vals[-14:])) if len(kg_vals)>=14 else pred*0.15
        out.append({"date":str(fd),"predicted":round(pred_adj,2),
                    "lower":round(max(0,pred_adj-1.44*std_est),2),
                    "upper":round(pred_adj+1.44*std_est,2),
                    "season":slbl,"mult":mult})
        kg_vals.append(pred_adj)
        inv_stock = max(0.0, inv_stock - pred_adj)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    s  = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), s, 260_000)
    return s.hex() + ":" + dk.hex()

def _verify(pw: str, stored: str) -> bool:
    try:
        s, d = stored.split(":")
        return hashlib.pbkdf2_hmac("sha256",pw.encode(),bytes.fromhex(s),260_000).hex()==d
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _raw():  return sqlite3.connect(DB, check_same_thread=False)
@st.cache_resource
def _conn(): return sqlite3.connect(DB, check_same_thread=False)
def qr(sql, p=()):  return pd.read_sql_query(sql, _conn(), params=p)
def scalar(sql):
    r = _conn().execute(sql).fetchone()
    return r[0] if r else None
def _bust():
    st.cache_resource.clear()
    st.cache_data.clear()
@st.cache_data(ttl=60)
def prods(): return qr("SELECT * FROM products ORDER BY name_ar")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGING  (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
def log_audit(user: str, action: str, table: str = "", record_id=None, summary: str = ""):
    try:
        cx = _raw()
        cx.execute(
            "INSERT INTO audit_log (timestamp,username,action,table_name,record_id,summary) "
            "VALUES (?,?,?,?,?,?)",
            (str(datetime.now()), user, action, table, record_id, summary[:500]))
        cx.commit(); cx.close()
        logger.info(f"AUDIT | {user} | {action} | {table} | {summary[:80]}")
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA ENTRY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def ins_sale(pid, nar, nen, cat, wkg, up, uc, pay, sd, user):
    tp=round(wkg*up,2); tc=round(wkg*uc,2)
    gp=round(tp-tc,2); mp=round(gp/tp*100,2) if tp>0 else 0
    cx=_raw()
    cx.execute(
        "INSERT INTO sales (product_id,product_name_ar,product_name_en,category,"
        "sale_date,weight_kg,unit_price,total_price,unit_cost,total_cost,"
        "gross_profit,profit_margin_percent,payment_method,entered_by,is_live) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
        (pid,nar,nen,cat,str(sd),wkg,up,tp,uc,tc,gp,mp,pay,user))
    cx.execute(
        "INSERT INTO inventory (product_id,current_stock_kg,last_updated) VALUES (?,0.0,?) "
        "ON CONFLICT(product_id) DO UPDATE SET "
        "current_stock_kg=MAX(0.0,current_stock_kg-?),last_updated=?",
        (pid,str(datetime.now()),wkg,str(datetime.now())))
    cx.commit(); cx.close()
    log_audit(user,"INSERT","sales",None,f"{nar} {wkg:.1f}kg @ {up:.0f} EGP = {tp:.2f} EGP")
    _bust()
    return {"tp":tp,"gp":gp,"mp":mp}

def ins_pur(sup, par, pid, wkg, cpkg, pd_date, notes, user):
    tc=round(wkg*cpkg,2); now_str=str(datetime.now()); cx=_raw()
    cx.execute(
        "INSERT INTO purchases (purchase_date,supplier_name,product_ar,product_id,"
        "weight_kg,cost_per_kg,total_cost,notes,entered_by,is_live) VALUES (?,?,?,?,?,?,?,?,?,1)",
        (str(pd_date),sup,par,pid,wkg,cpkg,tc,notes,user))
    if pid:
        if par in CARCASS_YIELD:
            yp=CARCASS_YIELD[par]; sal=round(wkg*yp,2); cm=CARCASS_CUT_MAP.get(par,[])
            if cm:
                for cn,sh in cm:
                    ckg=round(sal*sh,3)
                    if ckg<=0: continue
                    cr=cx.execute("SELECT id FROM products WHERE name_ar=?",(cn,)).fetchone()
                    if cr:
                        cx.execute(
                            "INSERT INTO inventory (product_id,current_stock_kg,last_updated) "
                            "VALUES (?,?,?) ON CONFLICT(product_id) DO UPDATE SET "
                            "current_stock_kg=current_stock_kg+?,last_updated=?",
                            (cr[0],ckg,now_str,ckg,now_str))
            else:
                cx.execute(
                    "INSERT INTO inventory (product_id,current_stock_kg,last_updated) "
                    "VALUES (?,?,?) ON CONFLICT(product_id) DO UPDATE SET "
                    "current_stock_kg=current_stock_kg+?,last_updated=?",
                    (pid,sal,now_str,sal,now_str))
        else:
            cx.execute(
                "INSERT INTO inventory (product_id,current_stock_kg,last_updated) VALUES (?,?,?) "
                "ON CONFLICT(product_id) DO UPDATE SET "
                "current_stock_kg=current_stock_kg+?,last_updated=?",
                (pid,wkg,now_str,wkg,now_str))
    cx.commit(); cx.close()
    log_audit(user,"INSERT","purchases",None,f"{par} {wkg:.1f}kg from {sup} = {tc:.2f} EGP")
    _bust()
    return tc

def ins_expense(exp_date, category, amount, description, user):
    """FR-08 — Record an operational expense."""
    cx=_raw()
    cx.execute(
        "INSERT INTO expenses (expense_date,category,amount_egp,description,entered_by,is_live) "
        "VALUES (?,?,?,?,?,1)",
        (str(exp_date),category,amount,description,user))
    cx.commit(); cx.close()
    log_audit(user,"INSERT","expenses",None,f"{category}: {amount:.2f} EGP — {description[:50]}")
    _bust()
    logger.info(f"Expense recorded: {category} {amount:.2f} EGP by {user}")

# ─────────────────────────────────────────────────────────────────────────────
# BACKUP SYSTEM  (Phase 1 — DB-04)
# ─────────────────────────────────────────────────────────────────────────────
def backup_db() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"ButcherAI_{ts}.db"
    shutil.copy2(DB, dst)
    # Clean old backups beyond rolling window
    cutoff = datetime.now() - timedelta(days=BACKUP_KEEP)
    removed = 0
    for f in BACKUP_DIR.glob("ButcherAI_*.db"):
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink(); removed += 1
    logger.info(f"Backup created: {dst} (removed {removed} old backups)")
    return dst

def list_backups() -> list:
    if not BACKUP_DIR.exists(): return []
    files = sorted(BACKUP_DIR.glob("ButcherAI_*.db"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"name": f.name, "path": str(f),
             "size_kb": round(f.stat().st_size/1024, 1),
             "created": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
            for f in files]

def restore_db(backup_path: str, user: str):
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    emergency = DB.replace(".db", f"_pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
    shutil.copy2(DB, emergency)
    shutil.copy2(backup_path, DB)
    log_audit(user, "RESTORE", "database", None, f"Restored from {backup_path}")
    logger.warning(f"DB RESTORED from {backup_path} by {user}. Pre-restore copy: {emergency}")

# ─────────────────────────────────────────────────────────────────────────────
# CSV IMPORT VALIDATION  (Phase 0 — security hardening)
# ─────────────────────────────────────────────────────────────────────────────
def validate_import_df(df: pd.DataFrame, target_table: str) -> tuple:
    """Returns (is_valid: bool, message: str)"""
    allowed = COLUMN_WHITELIST.get(target_table, set())
    bad_cols = set(df.columns) - allowed
    if bad_cols:
        return False, f"Unexpected columns detected: {bad_cols}. Only allowed: {allowed}"
    if len(df) > 10_000:
        return False, f"Too many rows ({len(df):,}). Max allowed: 10,000 per import."
    if target_table == "sales":
        if "weight_kg" in df.columns and (df["weight_kg"] <= 0).any():
            return False, "weight_kg must be > 0 for all rows."
        if "unit_price" in df.columns and (df["unit_price"] <= 0).any():
            return False, "unit_price must be > 0 for all rows."
        if "sale_date" in df.columns:
            try:
                dates = pd.to_datetime(df["sale_date"], format="mixed", errors="coerce")
                if dates.isna().any():
                    return False, "Some sale_date values could not be parsed."
                future = (dates.dt.date > date.today()).sum()
                if future > 0:
                    return False, f"{future} rows have future sale_date values. Sales must be ≤ today."
            except Exception as e:
                return False, f"Date parsing error: {e}"
    if target_table == "purchases":
        if "weight_kg" in df.columns and (df["weight_kg"] <= 0).any():
            return False, "weight_kg must be > 0 for all rows."
        if "cost_per_kg" in df.columns and (df["cost_per_kg"] <= 0).any():
            return False, "cost_per_kg must be > 0 for all rows."
    return True, "OK"

# ─────────────────────────────────────────────────────────────────────────────
# MARKETING (Phase 0: rate limiting; API key from parameter only)
# ─────────────────────────────────────────────────────────────────────────────
_CTX_AR={"ramadan":"في شهر رمضان المبارك","eid":"في أيام العيد السعيدة",
         "weekend":"في يوم الجمعة المبارك","clearance":"عروض نهاية الأسبوع","general":"جودة لا تُضاهى"}
_CTX_EN={"ramadan":"This Ramadan","eid":"Celebrate Eid with",
         "weekend":"Your Friday favourite","clearance":"Weekend value","general":"Quality you can taste"}
_CTA_AR={"ramadan":"احجز كميتك لمائدة الإفطار.","eid":"اجعل عيدك لا يُنسى.",
         "weekend":"يوم الجمعة فرصتك.","clearance":"عرض لفترة محدودة.","general":"احجز الآن."}
_CTA_EN={"ramadan":"Reserve your Iftar quantity now.","eid":"Make your Eid unforgettable.",
         "weekend":"Friday special.","clearance":"Limited time.","general":"Reserve now."}
_QAR={"Premium":"من أجود المقاطع الفاخرة","Standard":"طازج يومياً",
      "Economy":"اقتصادي بلا تنازل","Ground":"مفروم أمام عينيك",
      "Processed":"مُحضَّر بعناية","Offal":"تراث مصري أصيل"}
_QEN={"Premium":"our finest premium cut","Standard":"fresh daily",
      "Economy":"great value","Ground":"freshly ground",
      "Processed":"carefully prepared","Offal":"authentic delicacy"}

def tmpl(nar,nen,cat,price,ctx):
    return {
        "arabic_headline": f"🥩 {_CTX_AR.get(ctx,'جودة')} — {nar}",
        "arabic_body":     f"{nar} {_QAR.get(cat,'طازج')}. سعر اليوم: {price:.0f} ج/كجم. {_CTA_AR.get(ctx,'')}",
        "arabic_cta":      "📞 اتصل الآن أو تعال لمتجرنا",
        "english_headline":f"🥩 {_CTX_EN.get(ctx,'Quality')} — {nen}",
        "english_body":    f"{nen} — {_QEN.get(cat,'fresh')}. Today: {price:.0f} EGP/kg. {_CTA_EN.get(ctx,'')}",
        "english_cta":     "📞 Call now or visit our shop",
    }

def gen_mkt(nar, nen, cat, price, margin, ctx, api_key=""):
    # Phase 0: rate limiting — cache by (product, context, price) for 5s
    cache_key = f"mkt_{nar}_{ctx}_{price:.0f}"
    cached    = st.session_state.get(cache_key)
    last_time = st.session_state.get(f"{cache_key}_time", 0)
    if cached and (time.time() - last_time) < 5:
        return cached

    # Phase 0: use parameter key only — NEVER a hardcoded value
    effective_key = api_key.strip() or GEMINI_API_ENV
    if effective_key:
        from gemini_client import generate_json
        prompt = (
            f"You are a bilingual marketing specialist for an Egyptian butcher shop. "
            f"Write catchy content for: {nen} (Arabic: {nar}), Category: {cat}, "
            f"Price: {price:.0f} EGP/kg, Context: {ctx}. "
            f"Output ONLY valid JSON with exactly 6 keys: "
            f"arabic_headline, arabic_body, arabic_cta, english_headline, english_body, english_cta.")
        result = generate_json(prompt, api_key=effective_key, db_path=DB, feature="marketing")
        if result["error"] is None:
            parsed = json.loads(result["text"])
            st.session_state[cache_key]          = parsed
            st.session_state[f"{cache_key}_time"] = time.time()
            logger.info(f"Gemini marketing generated for {nar}/{ctx} via {result['model_used']}")
            return parsed
        else:
            logger.warning(f"Gemini API error (all models failed): {result['error']}")
    return tmpl(nar, nen, cat, price, ctx)

# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT  (Phase 1 — bilingual)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(days: int = 30) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        import io

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        RED    = colors.HexColor("#8B1A1A")
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     textColor=RED, fontSize=18, spaceAfter=6)
        sub_style   = ParagraphStyle("sub",   parent=styles["Heading2"],
                                     textColor=RED, fontSize=12, spaceAfter=4)
        body_style  = styles["BodyText"]

        story.append(Paragraph("ButcherAI v9.1 — Performance Report", title_style))
        story.append(Paragraph(
            f"Period: Last {days} days &nbsp;|&nbsp; Generated: {datetime.now():%Y-%m-%d %H:%M}",
            body_style))
        story.append(Spacer(1, 14))

        # KPI Summary
        story.append(Paragraph("Key Performance Indicators", sub_style))
        rev = scalar(f"SELECT SUM(total_price)  FROM sales WHERE sale_date>=date('now','-{days} days')") or 0
        pft = scalar(f"SELECT SUM(gross_profit) FROM sales WHERE sale_date>=date('now','-{days} days')") or 0
        mg  = scalar(f"SELECT AVG(profit_margin_percent) FROM sales WHERE sale_date>=date('now','-{days} days')") or 0
        wt  = scalar(f"SELECT SUM(weight_kg)    FROM sales WHERE sale_date>=date('now','-{days} days')") or 0
        tx  = scalar(f"SELECT COUNT(*)           FROM sales WHERE sale_date>=date('now','-{days} days')") or 0
        exp_total = scalar(f"SELECT SUM(amount_egp) FROM expenses WHERE expense_date>=date('now','-{days} days')") or 0

        kpi_data = [
            ["Metric", "Value"],
            ["Revenue (EGP)",       f"{rev:,.0f}"],
            ["Gross Profit (EGP)",  f"{pft:,.0f}"],
            ["Expenses (EGP)",      f"{exp_total:,.0f}"],
            ["Net Profit (EGP)",    f"{pft - exp_total:,.0f}"],
            ["Avg Margin",          f"{mg:.1f}%"],
            ["Total Weight (kg)",   f"{wt:,.1f}"],
            ["Transactions",        f"{int(tx):,}"],
        ]
        tbl = Table(kpi_data, colWidths=[200, 150])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), RED),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#fff5f5")]),
            ("GRID",(0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("FONTSIZE",(0,0),(-1,-1), 10),
            ("PADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 14))

        # Top Products
        story.append(Paragraph("Top 10 Products by Revenue", sub_style))
        top = qr(f"SELECT p.name_en, SUM(s.total_price) rev, SUM(s.weight_kg) kg "
                 f"FROM sales s JOIN products p ON s.product_id=p.id "
                 f"WHERE s.sale_date>=date('now','-{days} days') "
                 f"GROUP BY s.product_id ORDER BY rev DESC LIMIT 10")
        if not top.empty:
            tp_data = [["Product","Revenue (EGP)","Weight (kg)"]]
            for _, row in top.iterrows():
                tp_data.append([str(row[0]), f"{row[1]:,.0f}", f"{row[2]:,.1f}"])
            tbl2 = Table(tp_data, colWidths=[200,130,110])
            tbl2.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), RED),
                ("TEXTCOLOR",(0,0),(-1,0), colors.white),
                ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#fff5f5")]),
                ("GRID",(0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("FONTSIZE",(0,0),(-1,-1), 9),
                ("PADDING",(0,0),(-1,-1), 5),
            ]))
            story.append(tbl2)

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "Generated by ButcherAI v9.1 — AI-Powered Business Intelligence for Egyptian Butcher Shops",
            ParagraphStyle("footer", parent=body_style, textColor=colors.grey, fontSize=8)))

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return b""

# ─────────────────────────────────────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────────────────────────────────────
for _k,_v in [("auth",False),("user",""),("dname",""),("role",""),
               ("ap","login"),("first_login",False)]:
    if _k not in st.session_state: st.session_state[_k]=_v

# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def show_first_login_change():
    """Phase 0: Force password change on first login."""
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'>"
                "<h2 style='color:#8B1A1A'>🔐 تغيير كلمة المرور الأولى</h2>"
                "<p style='color:#888;font-size:13px'>يجب تغيير كلمة المرور قبل استخدام النظام</p>"
                "</div>", unsafe_allow_html=True)
    with st.form("first_pw"):
        p1 = st.text_input("كلمة المرور الجديدة *", type="password", placeholder="8 أحرف على الأقل")
        p2 = st.text_input("تأكيد كلمة المرور *",   type="password")
        if st.form_submit_button("تأكيد وحفظ ✓", type="primary", use_container_width=True):
            if len(p1) < 8:   st.error("⚠️ كلمة المرور يجب أن تكون 8 أحرف على الأقل")
            elif p1 != p2:    st.error("⚠️ كلمتا المرور غير متطابقتين")
            else:
                cx = _raw()
                cx.execute("UPDATE users SET hashed_pw=?,first_login=0 WHERE username=?",
                           (_hash(p1), st.session_state.user))
                cx.commit(); cx.close()
                st.session_state.first_login = False
                log_audit(st.session_state.user, "CHANGE_PASSWORD", "users", None, "First-login password change")
                st.success("✅ تم تغيير كلمة المرور. أهلاً بك!")
                st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

def show_login():
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'>"
                "<h1 style='color:#8B1A1A;margin:0'>🥩 ButcherAI</h1>"
                "<p style='color:#888;font-size:11px;margin:4px 0 20px'>"
                "v9.1 · Enterprise LightGBM · BI Analytics · Arabic RTL"
                "<span class='version-badge' style='margin-left:6px'>v9.1</span></p></div>",
                unsafe_allow_html=True)
    with st.form("lf"):
        u = st.text_input("👤 اسم المستخدم")
        p = st.text_input("🔒 كلمة المرور", type="password")
        c1,c2 = st.columns(2)
        with c1: sub = st.form_submit_button("تسجيل الدخول ←", use_container_width=True)
        with c2: reg = st.form_submit_button("إنشاء حساب",      use_container_width=True)
        if sub:
            if not u or not p:
                st.error("⚠️ أدخل بيانات الدخول")
            else:
                uname_clean = u.strip()
                cx_rl = _raw()
                recent_fails = cx_rl.execute(
                    "SELECT COUNT(*) FROM login_attempts WHERE username=? "
                    "AND success=0 AND attempt_time >= datetime('now', '-5 minutes')",
                    (uname_clean,)
                ).fetchone()[0]

                if recent_fails >= 5:
                    st.error("🔒 تم قفل الحساب مؤقتاً بسبب محاولات دخول فاشلة متكررة. حاول بعد 5 دقائق.")
                    cx_rl.close()
                else:
                    row = cx_rl.execute(
                        "SELECT hashed_pw,full_name,role,active,COALESCE(first_login,0) "
                        "FROM users WHERE username=?", (uname_clean,)).fetchone()

                    login_ok = row and row[3] and _verify(p, row[0])

                    cx_rl.execute(
                        "INSERT INTO login_attempts (username, attempt_time, success) VALUES (?, ?, ?)",
                        (uname_clean, str(datetime.now()), 1 if login_ok else 0)
                    )
                    cx_rl.commit()

                    if not row:
                        st.error("❌ اسم المستخدم غير موجود")
                    elif not row[3]:
                        st.error("❌ الحساب موقوف")
                    elif not _verify(p, row[0]):
                        remaining = max(0, 5 - recent_fails - 1)
                        st.error("❌ كلمة المرور خاطئة" + ((" — محاولة أخيرة قبل القفل المؤقت" if remaining == 0 else "")))
                    else:
                        cx_rl.execute("UPDATE users SET last_login=? WHERE username=?",
                                    (str(datetime.now()), uname_clean))
                        cx_rl.commit()
                        st.session_state.update({
                            "auth":True,"user":uname_clean,"dname":row[1],
                            "role":row[2],"first_login":bool(row[4])})
                        logger.info("Login: " + uname_clean)
                        cx_rl.close()
                        st.rerun()
                    cx_rl.close()
        if reg: st.session_state.ap="register"; st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

def show_register():
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><h1 style='color:#8B1A1A;margin:0'>🥩 إنشاء حساب</h1></div>",
                unsafe_allow_html=True)
    with st.form("rf", clear_on_submit=True):
        u  = st.text_input("👤 اسم المستخدم *", placeholder="بالإنجليزية بدون مسافات")
        dn = st.text_input("📝 الاسم الظاهر *")
        p1 = st.text_input("🔒 كلمة المرور *",  type="password", placeholder="8 أحرف على الأقل")
        p2 = st.text_input("🔒 تأكيد",           type="password")
        rl = st.selectbox("الدور", ["staff","viewer"],
                          format_func=lambda x:{"staff":"موظف — إدخال + تحليل",
                                                 "viewer":"مشاهد — قراءة فقط"}[x])
        c1,c2 = st.columns(2)
        with c1: sub=st.form_submit_button("إنشاء ✓",     use_container_width=True)
        with c2: bk =st.form_submit_button("رجوع للدخول", use_container_width=True)
        if sub:
            errs=[(not u or " " in u,"اسم المستخدم: بدون مسافات"),
                  (not dn,"الاسم الظاهر مطلوب"),
                  (len(p1)<8,"كلمة المرور 8 أحرف على الأقل"),
                  (p1!=p2,"كلمتا المرور غير متطابقتين")]
            for cond,msg in errs:
                if cond: st.error(f"⚠️ {msg}")
            if not any(c for c,_ in errs):
                cx=_raw()
                if cx.execute("SELECT id FROM users WHERE username=?",(u,)).fetchone():
                    st.error("❌ اسم المستخدم موجود"); cx.close()
                else:
                    cx.execute(
                        "INSERT INTO users (username,full_name,hashed_pw,role,created_at,first_login) "
                        "VALUES (?,?,?,?,?,1)",
                        (u,dn,_hash(p1),rl,str(datetime.now())))
                    cx.commit(); cx.close()
                    st.success("✅ تم إنشاء الحساب!")
                    st.session_state.ap="login"; st.rerun()
        if bk: st.session_state.ap="login"; st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

# Route auth
if not st.session_state.auth:
    if   st.session_state.ap=="register": show_register()
    else:                                  show_login()
    st.stop()
if st.session_state.first_login:
    show_first_login_change()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
role  = st.session_state.role
uname = st.session_state.user
dname = st.session_state.dname
RCLR  = {"admin":"#8B1A1A","staff":"#1A5276","viewer":"#27AE60"}
RLBL  = {"admin":"مدير النظام","staff":"موظف","viewer":"مشاهد"}

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:8px'>"
        "<h1 style='color:#8B1A1A;font-size:18px;margin:0'>🥩 ButcherAI v9.1</h1>"
        "<p style='color:#888;font-size:9px;margin:0'>Enterprise LightGBM · 19 Pages · Phase 1</p>"
        "</div>", unsafe_allow_html=True)
    rc=RCLR.get(role,"#888"); rl_lbl=RLBL.get(role,role)
    st.markdown(
        f"<div style='background:#f8f0f0;border-radius:8px;padding:8px 12px;margin:6px 0'>"
        f"<b style='color:#8B1A1A'>👤 {dname}</b><br>"
        f"<span style='background:{rc};color:#fff;font-size:10px;"
        f"padding:2px 8px;border-radius:4px'>{rl_lbl}</span></div>",
        unsafe_allow_html=True)
    st.divider()

    base_pages = [
        "🏠 الرئيسية",
        "➕ إدخال مبيعات",
        "📦 إدخال مشتريات",
        "📦 إدارة المخزون",
        "💸 إدارة المصروفات",
        "📊 تحليل المبيعات",
        "⚡ التحليل الفوري",
        "🔮 التنبؤ بالطلب",
        "🤖 المساعد الذكي",
        "📓 استيراد من الدفتر الورقي",
        "💡 توصيات الشراء",
        "📢 التسويق الذكي",
        "⚙️ تحليل WAC",
        "📈 BI Analytics",
        "📅 رصيد المخزون اليومي",
        "🏭 Enterprise Forecast",
        "💹 توقع الأسعار",
        "⚙️ محرك الأتمتة",
        "📚 قاعدة المعرفة (RAG)",
        "🤖 وكيل الذكاء الاصطناعي",
        "👥 الوكلاء المتعددين",
        "🚚 إدارة الموردين",
        "📊 مراقبة الذكاء الاصطناعي",
        "📊 لوحة المراقبة",
        "🧪 تقييم الجودة",
    ]
    admin_pages = ["💾 النسخ الاحتياطي","📜 سجل التدقيق",
                   "👥 إدارة المستخدمين","⚙️ إدارة البيانات"]
    viewer_hidden = {"➕ إدخال مبيعات","📦 إدخال مشتريات",
                     "📦 إدارة المخزون","💸 إدارة المصروفات","🚚 إدارة الموردين",
                     "📓 استيراد من الدفتر الورقي"}

    pages = base_pages.copy()
    if role == "admin":   pages += admin_pages
    if role == "viewer":  pages = [p for p in pages if p not in viewer_hidden]
    page = st.radio("القائمة", pages)
    st.divider()

    try:
        ts=scalar("SELECT COUNT(*) FROM sales") or 0
        ls=scalar("SELECT COUNT(*) FROM sales WHERE is_live=1") or 0
        ic=scalar("SELECT COUNT(*) FROM inventory WHERE current_stock_kg>0") or 0
        ex_today=scalar(f"SELECT SUM(amount_egp) FROM expenses WHERE expense_date=date('now')") or 0
        st.success("✅ قاعدة البيانات متصلة")
        c1,c2=st.columns(2)
        c1.metric("مبيعات",f"{ts:,}"); c2.metric("🟢 جديدة",f"{ls}")
        st.metric("منتجات في المخزون",f"{ic}")
        st.metric("مصروفات اليوم",f"{ex_today:,.0f} ج")
        if ENT_MODEL:
            nc=len(ENT_FEATURES) if ENT_FEATURES else "?"
            st.info(f"🏭 Enterprise Model ✅ ({nc} features)")
        else: st.warning("⚠️ Model PKL missing")
    except Exception as e:
        st.error(f"⚠️ خطأ: {e}")
    st.divider()
    if st.button("🚪 تسجيل الخروج", use_container_width=True):
        logger.info(f"Logout: {uname}")
        for k in ["auth","user","dname","role","first_login"]:
            st.session_state[k] = "" if k!="auth" else False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: الرئيسية — Main Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 الرئيسية":
    st.title("🏠 لوحة المعلومات الرئيسية")
    ls_cnt=scalar("SELECT COUNT(*) FROM sales WHERE is_live=1") or 0
    if ls_cnt>0:
        st.markdown(f"<div class='info-box'>🟢 يشمل التحليل {ls_cnt} عملية بيع مُدخلة يدوياً</div>",
                    unsafe_allow_html=True)
    ft=st.radio("نطاق الأرقام:",["اليوم فقط","هذا الأسبوع","فترة مخصصة"],horizontal=True)
    if ft=="اليوم فقط":     days=0
    elif ft=="هذا الأسبوع": days=7
    else:                    days=st.slider("آخر (يوم):",30,700,365)
    where=f"sale_date>=date('now','-{days} days')"
    rev=scalar(f"SELECT SUM(total_price)         FROM sales WHERE {where}") or 0
    pft=scalar(f"SELECT SUM(gross_profit)         FROM sales WHERE {where}") or 0
    mg =scalar(f"SELECT AVG(profit_margin_percent) FROM sales WHERE {where}") or 0
    wt =scalar(f"SELECT SUM(weight_kg)            FROM sales WHERE {where}") or 0
    tx =scalar(f"SELECT COUNT(*)                   FROM sales WHERE {where}") or 0
    exp_sum=scalar(f"SELECT SUM(amount_egp) FROM expenses WHERE expense_date>=date('now','-{days} days')") or 0
    net_pft=pft-exp_sum
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("💰 الإيرادات",f"{rev:,.0f} ج")
    c2.metric("📈 الربح الإجمالي",f"{pft:,.0f} ج")
    c3.metric("💸 المصروفات",f"{exp_sum:,.0f} ج")
    c4.metric("🎯 صافي الربح",f"{net_pft:,.0f} ج")
    c5.metric("⚖️ الوزن",f"{wt:,.0f} كجم")
    c6.metric("🛒 المعاملات",f"{tx:,}")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        st.subheader("🏆 أعلى 10 منتجات إيراداً")
        top=qr(f"SELECT p.name_ar,SUM(s.total_price) rev FROM sales s "
               f"JOIN products p ON s.product_id=p.id WHERE s.{where} "
               f"GROUP BY s.product_id ORDER BY rev DESC LIMIT 10")
        if not top.empty:
            fig=px.bar(top,x="rev",y="name_ar",orientation="h",color="rev",
                       color_continuous_scale="Reds",text="rev",
                       labels={"rev":"الإيرادات","name_ar":"المنتج"})
            fig.update_traces(texttemplate="%{text:,.0f}",textposition="outside")
            fig.update_layout(showlegend=False,height=370,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.subheader("📊 توزيع حسب الفئة")
        CAT={"Premium":"فاخر","Standard":"عادي","Economy":"اقتصادي",
             "Ground":"مفروم","Processed":"مصنع","Offal":"مخلفات"}
        bc=qr(f"SELECT p.category,SUM(s.total_price) rev FROM sales s "
              f"JOIN products p ON s.product_id=p.id WHERE s.{where} GROUP BY p.category")
        if not bc.empty:
            bc["category"]=bc.category.map(CAT).fillna(bc.category)
            fig=px.pie(bc,values="rev",names="category",hole=0.4,
                       color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition="inside",textinfo="percent+label")
            fig.update_layout(height=370)
            st.plotly_chart(fig,use_container_width=True)
    st.subheader("📅 الاتجاه اليومي")
    daily=qr(f"SELECT sale_date,SUM(total_price) rev,SUM(gross_profit) pft,MAX(is_live) hl "
             f"FROM sales WHERE {where} GROUP BY sale_date ORDER BY sale_date")
    if not daily.empty:
        daily["sale_date"]=pd.to_datetime(daily.sale_date,format="mixed",errors="coerce")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=daily.sale_date,y=daily.rev,name="الإيرادات",
                                 fill="tozeroy",line=dict(color="#8B1A1A",width=2)))
        fig.add_trace(go.Scatter(x=daily.sale_date,y=daily.pft,name="الأرباح",
                                 fill="tozeroy",line=dict(color="#28a745",width=2)))
        hl=daily[daily.hl==1]
        if not hl.empty:
            fig.add_trace(go.Scatter(x=hl.sale_date,y=hl.rev,mode="markers",name="🟢 مُدخل يدوياً",
                                     marker=dict(color="#28a745",size=10,symbol="star")))
        fig.update_layout(hovermode="x unified",height=300)
        st.plotly_chart(fig,use_container_width=True)
    # PDF download button
    st.divider()
    if st.button("📄 تحميل تقرير PDF"):
        pdf_bytes=generate_pdf_report(days if days>0 else 30)
        if pdf_bytes:
            st.download_button("⬇️ تحميل PDF",pdf_bytes,
                               file_name=f"ButcherAI_Report_{date.today()}.pdf",
                               mime="application/pdf")
        else:
            st.warning("⚠️ reportlab غير مثبت. شغل: pip install reportlab")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: إدخال مبيعات
# ══════════════════════════════════════════════════════════════════════════════
elif page == "➕ إدخال مبيعات":
    st.title("➕ إدخال عملية بيع جديدة")
    p=prods(); mn,ln=get_mult(date.today())
    if mn>=1.2:
        st.markdown(f"<div class='warn-box'>📅 موسم {ln} — المضاعف ×{mn:.2f}</div>",unsafe_allow_html=True)
    with st.form("sf",clear_on_submit=True):
        c1,c2=st.columns(2)
        with c1:
            sel=st.selectbox("المنتج *",p.name_ar.tolist())
            pr=p[p.name_ar==sel].iloc[0]; pid=int(pr.id)
            sp=round(float(pr.avg_price),0); sc=round(float(pr.avg_cost),0)
            wkg=st.number_input("الوزن (كجم) *",0.1,2000.0,1.0,0.1,format="%.2f")
            pay=st.selectbox("طريقة الدفع *",["Cash","Card","Credit"],
                             format_func=lambda x:{"Cash":"نقدي","Card":"بطاقة","Credit":"آجل"}[x])
        with c2:
            up=st.number_input("سعر البيع (ج/كجم) *",1.0,5000.0,float(sp),5.0)
            uc=st.number_input("تكلفة الشراء (ج/كجم) *",1.0,5000.0,float(sc),5.0)
            sd=st.date_input("تاريخ البيع *",date.today(),max_value=date.today())
        cur=_conn().execute("SELECT current_stock_kg FROM inventory WHERE product_id=?",(pid,)).fetchone()
        cur_kg=float(cur[0]) if cur else 0.0
        if cur_kg<wkg:
            st.warning(f"⚠️ المخزون الحالي: {cur_kg:.1f} كجم — أقل من الكمية المطلوبة")
        else:
            st.info(f"📦 المخزون: {cur_kg:.1f} كجم → سيصبح {max(0,cur_kg-wkg):.1f} كجم")
        if wkg>0 and up>0:
            tp=round(wkg*up,2); gp=round(wkg*(up-uc),2); mp=round((up-uc)/up*100,1) if up>0 else 0
            st.info(f"💰 الإجمالي **{tp:,.2f} ج** | ربح **{gp:,.2f} ج** | هامش **{mp:.1f}%** | ×{get_mult(sd)[0]:.2f}")
        sub=st.form_submit_button("✅ حفظ",type="primary",use_container_width=True)
        if sub:
            if wkg<=0 or up<=0: st.error("⚠️ الوزن والسعر يجب > 0")
            else:
                r=ins_sale(pid,str(pr.name_ar),str(pr.name_en),str(pr.category),wkg,up,uc,pay,sd,uname)
                st.markdown(
                    f"<div class='ok'>✅ تم حفظ البيع! {pr.name_ar} | "
                    f"{wkg:.2f} كجم × {up:.0f} ج = {r['tp']:,.2f} ج | "
                    f"ربح {r['gp']:,.2f} ج | هامش {r['mp']:.1f}% "
                    f"<span style='background:#28a745;color:#fff;padding:2px 8px;"
                    f"border-radius:4px;font-size:11px'>🟢 is_live=1</span></div>",
                    unsafe_allow_html=True)
    st.divider()
    st.subheader("📋 آخر 20 عملية بيع مُدخلة")
    rec=qr("SELECT sale_date,product_name_ar,weight_kg,unit_price,total_price,"
           "gross_profit,profit_margin_percent,payment_method FROM sales "
           "WHERE is_live=1 ORDER BY id DESC LIMIT 20")
    if not rec.empty:
        rec.columns=["التاريخ","المنتج","الوزن","سعر/كجم","الإجمالي","الربح","الهامش%","الدفع"]
        st.dataframe(rec,use_container_width=True,hide_index=True,
                     column_config={"الهامش%":st.column_config.ProgressColumn(
                         format="%.1f%%",min_value=0,max_value=60)})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: إدخال مشتريات
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 إدخال مشتريات":
    st.title("📦 إدخال عملية شراء جديدة")
    p=prods()
    SUPS=["أحمد","محمد","مجدي","أنور","غير محدد","مورد جديد..."]
    carcass_names=list(CARCASS_YIELD.keys())
    carcass_df=p[p["name_ar"].isin(carcass_names)]; cuts_df=p[~p["name_ar"].isin(carcass_names)]
    purchase_type=st.radio("نوع الشراء:",
        ["🐂 ذبيحة كاملة (مع تطبيق نسبة الاستفادة 70%)","🔪 قطعة جاهزة (وزن كامل)"],horizontal=True)
    is_carcass=purchase_type.startswith("🐂")
    fp=carcass_df if is_carcass and not carcass_df.empty else (cuts_df if not is_carcass else p)
    with st.form("pf",clear_on_submit=True):
        c1,c2=st.columns(2)
        with c1:
            so=st.selectbox("المورد *",SUPS)
            sup=st.text_input("اسم المورد الجديد") if so=="مورد جديد..." else so
            par=st.selectbox("المنتج *",fp["name_ar"].tolist())
            pr2=p[p.name_ar==par].iloc[0]; pid2=int(pr2.id)
            wkg=st.number_input("الوزن (كجم) *",0.1,10000.0,100.0,0.5,format="%.1f")
        with c2:
            rp=_conn().execute("SELECT AVG(cost_per_kg) FROM purchases WHERE product_id=?",(pid2,)).fetchone()[0] or float(pr2.avg_cost)
            cpkg=st.number_input("السعر (ج/كجم) *",1.0,5000.0,round(float(rp),0),5.0)
            pd_date=st.date_input("تاريخ الشراء *",date.today(),max_value=date.today())
            notes=st.text_input("ملاحظات (اختياري)")
        if is_carcass and par in CARCASS_YIELD:
            yp=CARCASS_YIELD[par]; sal=round(wkg*yp,2); waste=round(wkg-sal,2)
            st.markdown(
                f"<div class='warn-box'>🐂 <b>ذبيحة كاملة — نسبة الاستفادة: {yp*100:.0f}%</b><br>"
                f"⚖️ الوزن المشترى: <b>{wkg:.1f} كجم</b> | ✅ اللحم الصافي: <b>{sal:.1f} كجم</b> | "
                f"🗑️ الهالك: <b>{waste:.1f} كجم</b> | 💵 الإجمالي: <b>{round(wkg*cpkg,2):,.2f} ج</b></div>",
                unsafe_allow_html=True)
            cm=CARCASS_CUT_MAP.get(par,[])
            if cm:
                st.markdown("**📋 توزيع القطع المتوقع:**")
                st.dataframe(pd.DataFrame([{"القطعة":cn,"النسبة %":f"{sh*100:.0f}%",
                                            "الكمية كجم":f"{sal*sh:.1f}"}
                                           for cn,sh in cm]),use_container_width=True,hide_index=True)
        else:
            st.info(f"💵 معاينة: {wkg:.1f} كجم × {cpkg:.0f} ج = **{round(wkg*cpkg,2):,.2f} ج**")
        sub=st.form_submit_button("✅ حفظ",type="primary",use_container_width=True)
        if sub:
            tc=ins_pur(sup.strip(),par,pid2,wkg,cpkg,pd_date,notes,uname)
            if is_carcass and par in CARCASS_YIELD:
                yp=CARCASS_YIELD[par]; sal=round(wkg*yp,2)
                st.markdown(f"<div class='ok'>✅ {sup} | {par} | {wkg:.1f} كجم خام → {sal:.1f} كجم صافٍ ({yp*100:.0f}%) = {tc:,.2f} ج</div>",unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ok'>✅ {sup} | {par} | {wkg:.1f} كجم × {cpkg:.0f} ج = {tc:,.2f} ج</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: إدارة المخزون
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 إدارة المخزون":
    st.title("📦 إدارة المخزون الحي")
    inv=qr("SELECT p.name_ar,p.name_en,p.category,p.reorder_point,"
           "COALESCE(i.current_stock_kg,0) stock,i.last_updated "
           "FROM products p LEFT JOIN inventory i ON p.id=i.product_id ORDER BY COALESCE(i.current_stock_kg,0) ASC")
    inv.columns=["المنتج (ع)","المنتج (إ)","الفئة","نقطة الطلب","المخزون كجم","آخر تحديث"]
    def _status(row):
        s=row["المخزون كجم"] or 0; r=row["نقطة الطلب"] or 10
        if s==0:     return "❌ نفد"
        if s<r*0.3:  return "🔴 حرج"
        if s<r:      return "🟡 منخفض"
        return "🟢 كافٍ"
    inv["الحالة"]=inv.apply(_status,axis=1)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("❌ نفد",   len(inv[inv["الحالة"]=="❌ نفد"]))
    c2.metric("🔴 حرج",  len(inv[inv["الحالة"]=="🔴 حرج"]))
    c3.metric("🟡 منخفض",len(inv[inv["الحالة"]=="🟡 منخفض"]))
    c4.metric("🟢 كافٍ", len(inv[inv["الحالة"]=="🟢 كافٍ"]))
    st.dataframe(inv[["المنتج (ع)","الفئة","المخزون كجم","نقطة الطلب","الحالة","آخر تحديث"]],
                 use_container_width=True,hide_index=True,
                 column_config={"المخزون كجم":st.column_config.NumberColumn(format="%.1f كجم")})
    st.divider()
    st.subheader("✏️ تعديل يدوي")
    p=prods(); sel=st.selectbox("المنتج:",p.name_ar.tolist(),key="inv_sel")
    pr3=p[p.name_ar==sel].iloc[0]; pid3=int(pr3.id)
    new_stock=st.number_input("الكمية الجديدة (كجم):",0.0,10000.0,0.0,0.5,format="%.1f")
    if st.button("تحديث المخزون"):
        cx=_raw()
        cx.execute(
            "INSERT INTO inventory (product_id,current_stock_kg,last_updated) VALUES (?,?,?) "
            "ON CONFLICT(product_id) DO UPDATE SET current_stock_kg=?,last_updated=?",
            (pid3,new_stock,str(datetime.now()),new_stock,str(datetime.now())))
        cx.commit(); cx.close()
        log_audit(uname,"UPDATE","inventory",pid3,f"{sel} → {new_stock:.1f} kg")
        _bust(); st.success(f"✅ تم تحديث مخزون {sel} إلى {new_stock:.1f} كجم"); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: إدارة المصروفات  (FR-08 — FULL IMPLEMENTATION)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💸 إدارة المصروفات":
    st.title("💸 إدارة المصروفات التشغيلية")
    st.markdown("<div class='info-box'>📋 سجّل المصروفات اليومية لحساب صافي الربح الحقيقي = الربح الإجمالي − المصروفات</div>",
                unsafe_allow_html=True)

    # ── Entry form ───────────────────────────────────────────────────────────
    with st.form("ef", clear_on_submit=True):
        c1,c2=st.columns(2)
        with c1:
            exp_cat=st.selectbox("الفئة *", EXPENSE_CATS)
            exp_amt=st.number_input("المبلغ (جنيه) *",0.01,500_000.0,100.0,50.0,format="%.2f")
        with c2:
            exp_date=st.date_input("التاريخ *",date.today(),max_value=date.today())
            exp_desc=st.text_input("وصف (اختياري)",placeholder="مثال: فاتورة كهرباء يوليو")
        if st.form_submit_button("✅ تسجيل المصروف",type="primary",use_container_width=True):
            ins_expense(exp_date,exp_cat,exp_amt,exp_desc,uname)
            st.markdown(f"<div class='ok'>✅ تم تسجيل: {exp_cat} — {exp_amt:,.2f} ج</div>",unsafe_allow_html=True)

    st.divider()
    # ── KPIs ─────────────────────────────────────────────────────────────────
    days_exp=st.slider("الفترة (أيام):",7,365,30,key="exp_slider")
    exp_where=f"expense_date>=date('now','-{days_exp} days')"
    tot_exp = scalar(f"SELECT SUM(amount_egp) FROM expenses WHERE {exp_where}") or 0
    tot_rev = scalar(f"SELECT SUM(total_price) FROM sales WHERE sale_date>=date('now','-{days_exp} days')") or 0
    grs_pft = scalar(f"SELECT SUM(gross_profit) FROM sales WHERE sale_date>=date('now','-{days_exp} days')") or 0
    net_pft = grs_pft - tot_exp
    c1,c2,c3,c4=st.columns(4)
    c1.metric("💸 إجمالي المصروفات",f"{tot_exp:,.0f} ج")
    c2.metric("📈 الربح الإجمالي",f"{grs_pft:,.0f} ج")
    c3.metric("🎯 صافي الربح",f"{net_pft:,.0f} ج",delta=f"{net_pft/tot_rev*100:.1f}%" if tot_rev>0 else "—")
    c4.metric("📊 نسبة المصروفات للإيرادات",f"{tot_exp/tot_rev*100:.1f}%" if tot_rev>0 else "—")

    # ── By category chart ────────────────────────────────────────────────────
    st.subheader("📊 المصروفات حسب الفئة")
    by_cat=qr(f"SELECT category,SUM(amount_egp) total FROM expenses WHERE {exp_where} GROUP BY category ORDER BY total DESC")
    if not by_cat.empty:
        c1,c2=st.columns(2)
        with c1:
            fig=px.pie(by_cat,values="total",names="category",hole=0.4,title="توزيع المصروفات",
                       color_discrete_sequence=px.colors.sequential.Oranges)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2=px.bar(by_cat,x="total",y="category",orientation="h",
                        title="المصروفات بالقيمة",color="total",color_continuous_scale="Oranges")
            st.plotly_chart(fig2,use_container_width=True)

    # ── Monthly P&L chart ────────────────────────────────────────────────────
    st.subheader("📅 الربح والمصروفات الشهرية")
    monthly_rev=qr("SELECT strftime('%Y-%m',sale_date) mo, SUM(gross_profit) gp FROM sales "
                   "GROUP BY mo ORDER BY mo")
    monthly_exp=qr("SELECT strftime('%Y-%m',expense_date) mo, SUM(amount_egp) exp FROM expenses "
                   "GROUP BY mo ORDER BY mo")
    if not monthly_rev.empty:
        merged=monthly_rev.merge(monthly_exp,on="mo",how="left").fillna(0)
        merged["net"]=merged.gp-merged.exp
        fig3=go.Figure()
        fig3.add_trace(go.Bar(x=merged.mo,y=merged.gp,name="الربح الإجمالي",marker_color="#28a745",opacity=0.7))
        fig3.add_trace(go.Bar(x=merged.mo,y=merged.exp,name="المصروفات",marker_color="#ffc107",opacity=0.7))
        fig3.add_trace(go.Scatter(x=merged.mo,y=merged.net,name="صافي الربح",
                                  line=dict(color="#8B1A1A",width=3),mode="lines+markers"))
        fig3.update_layout(barmode="group",height=320,hovermode="x unified")
        st.plotly_chart(fig3,use_container_width=True)

    # ── Expense table with pagination ────────────────────────────────────────
    st.subheader("📋 سجل المصروفات")
    total_exp_rows=scalar(f"SELECT COUNT(*) FROM expenses WHERE {exp_where}") or 0
    page_size=50; total_pages=max(1,(total_exp_rows+page_size-1)//page_size)
    p_num=st.number_input("الصفحة:",1,total_pages,1,key="exp_page")-1
    exp_df=qr(f"SELECT expense_date,category,amount_egp,description,entered_by "
              f"FROM expenses WHERE {exp_where} ORDER BY id DESC "
              f"LIMIT {page_size} OFFSET {p_num*page_size}")
    if not exp_df.empty:
        exp_df.columns=["التاريخ","الفئة","المبلغ ج","الوصف","مُدخل بواسطة"]
        st.dataframe(exp_df,use_container_width=True,hide_index=True)
        st.caption(f"عرض {len(exp_df)} من {total_exp_rows:,} سجل")
    csv_exp=exp_df.to_csv(index=False).encode("utf-8") if not exp_df.empty else b""
    if csv_exp:
        st.download_button("⬇️ تحميل CSV",csv_exp,f"expenses_{date.today()}.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: تحليل المبيعات  (with pagination)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 تحليل المبيعات":
    st.title("📊 تحليل المبيعات التفصيلي")
    c1,c2,c3,c4=st.columns(4)
    with c1: days=st.slider("الفترة:",7,700,90)
    with c2:
        cats=["الكل"]+qr("SELECT DISTINCT category FROM products").category.tolist()
        sc=st.selectbox("الفئة:",cats)
    with c3:
        sp=st.selectbox("الدفع:",["الكل","Cash","Card","Credit"],
                        format_func=lambda x:{"الكل":"الكل","Cash":"نقدي","Card":"بطاقة","Credit":"آجل"}.get(x,x))
    with c4:
        ef=st.selectbox("البيانات:",["الكل","تاريخية فقط","مُدخلة يدوياً"])
    cc=f" AND p.category='{sc}'" if sc!="الكل" else ""
    pc=f" AND s.payment_method='{sp}'" if sp!="الكل" else ""
    lv={"الكل":"","تاريخية فقط":" AND s.is_live=0","مُدخلة يدوياً":" AND s.is_live=1"}[ef]
    base_where=f"s.sale_date>=date('now','-{days} days'){cc}{pc}{lv}"
    total_cnt=scalar(f"SELECT COUNT(*) FROM sales s JOIN products p ON s.product_id=p.id WHERE {base_where}") or 0
    page_size=100; total_pages=max(1,(total_cnt+page_size-1)//page_size)
    p_num=st.number_input("الصفحة:",1,total_pages,1,key="sal_page")-1
    sl=qr(f"SELECT s.sale_date,p.name_ar,p.category,s.weight_kg,s.unit_price,"
          f"s.total_price,s.gross_profit,s.profit_margin_percent,s.payment_method "
          f"FROM sales s JOIN products p ON s.product_id=p.id WHERE {base_where} "
          f"ORDER BY s.id DESC LIMIT {page_size} OFFSET {p_num*page_size}")
    tot_rev=scalar(f"SELECT SUM(s.total_price) FROM sales s JOIN products p ON s.product_id=p.id WHERE {base_where}") or 0
    tot_pft=scalar(f"SELECT SUM(s.gross_profit) FROM sales s JOIN products p ON s.product_id=p.id WHERE {base_where}") or 0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("السجلات",f"{total_cnt:,}"); c2.metric("الإيرادات",f"{tot_rev:,.0f} ج")
    c3.metric("الأرباح",f"{tot_pft:,.0f} ج")
    if not sl.empty:
        c4.metric("الهامش",f"{sl.profit_margin_percent.mean():.1f}%")
        sl.columns=["التاريخ","المنتج","الفئة","الوزن","سعر/كجم","الإجمالي","الربح","الهامش%","الدفع"]
        st.dataframe(sl,use_container_width=True,hide_index=True,
                     column_config={"الهامش%":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=60)})
        st.caption(f"صفحة {p_num+1} من {total_pages} | عرض {len(sl)} من {total_cnt:,} سجل")
    csv_sl=sl.to_csv(index=False).encode("utf-8") if not sl.empty else b""
    if csv_sl: st.download_button("⬇️ تحميل CSV",csv_sl,f"sales_{date.today()}.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: التحليل الفوري
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ التحليل الفوري":
    st.title("⚡ التحليل الفوري للمبيعات")
    if st.button("🔄 تحديث",type="primary"): _bust(); st.rerun()
    ts_s=str(date.today()); ys_s=str(date.today()-timedelta(days=1))
    mn,ln=get_mult(date.today())
    td=qr(f"SELECT SUM(total_price) rev,SUM(gross_profit) pft,SUM(weight_kg) kg,COUNT(*) tx FROM sales WHERE sale_date='{ts_s}'").iloc[0]
    yd=qr(f"SELECT SUM(total_price) rev FROM sales WHERE sale_date='{ys_s}'").iloc[0]
    tr=float(td.rev or 0); yr=float(yd.rev or 0)
    rd=round(((tr-yr)/yr*100) if yr>0 else 0,1)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("إيرادات اليوم",f"{tr:,.0f} ج",f"{rd:+.1f}% vs أمس")
    c2.metric("أرباح اليوم",f"{float(td.pft or 0):,.0f} ج")
    c3.metric("وزن اليوم",f"{float(td.kg or 0):,.1f} كجم")
    c4.metric("معاملات",f"{int(td.tx or 0)}")
    if mn>=2.0: st.markdown(f"<div class='warn-box'>🎉 موسم {ln} — الطلب المتوقع ×{mn:.1f}</div>",unsafe_allow_html=True)
    st.subheader("📈 EMA — المتوسط الأسي للإيرادات (14 يوم)")
    w14=qr("SELECT sale_date,SUM(total_price) rev FROM sales WHERE sale_date>=date('now','-30 days') GROUP BY sale_date ORDER BY sale_date")
    if not w14.empty:
        w14["sale_date"]=pd.to_datetime(w14.sale_date,format="mixed",errors="coerce")
        w14["EMA7"]=w14.rev.ewm(span=7,adjust=False).mean()
        w14["EMA14"]=w14.rev.ewm(span=14,adjust=False).mean()
        fig=go.Figure()
        fig.add_trace(go.Bar(x=w14.sale_date,y=w14.rev,name="الإيرادات",marker_color="#C0392B",opacity=0.5))
        fig.add_trace(go.Scatter(x=w14.sale_date,y=w14.EMA7,name="EMA-7",line=dict(color="#8B1A1A",width=2)))
        fig.add_trace(go.Scatter(x=w14.sale_date,y=w14.EMA14,name="EMA-14",line=dict(color="#27AE60",width=2,dash="dash")))
        fig.update_layout(height=260,hovermode="x unified",margin=dict(t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("🏆 أعلى منتجات اليوم")
    tp_today=qr(f"SELECT product_name_ar,SUM(weight_kg) kg,SUM(total_price) rev FROM sales WHERE sale_date='{ts_s}' GROUP BY product_id ORDER BY rev DESC LIMIT 5")
    if not tp_today.empty:
        tp_today.columns=["المنتج","الوزن كجم","الإيرادات"]; st.dataframe(tp_today,use_container_width=True,hide_index=True)
    else: st.info("لا مبيعات مسجلة اليوم بعد.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: التنبؤ بالطلب  (with SHAP panel)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 التنبؤ بالطلب":
    st.title("🔮 التنبؤ بالطلب — Enterprise LightGBM")
    if not ENT_MODEL: st.error("⚠️ النموذج غير متاح"); st.stop()
    p=prods()
    view_mode=st.radio("مستوى التحليل:",["🥩 القطعيات الفردية","🐂 احتياج الذبائح"],horizontal=True)
    st.divider()
    if view_mode.startswith("🥩"):
        c1,c2=st.columns(2)
        with c1: sel=st.selectbox("المنتج:",p.name_ar.tolist())
        with c2: fcd=st.slider("أيام التنبؤ:",3,14,7)
        pid=int(p[p.name_ar==sel].id.values[0])
        if st.button("🚀 توليد التنبؤ",type="primary"):
            with st.spinner("Enterprise LightGBM..."):
                t0=time.time(); fc=enterprise_forecast(pid,fcd); el=round(time.time()-t0,2)
            if not fc: st.error("بيانات غير كافية (يلزم 6 أيام على الأقل)")
            else:
                df_fc=pd.DataFrame(fc)
                c1,c2,c3,c4=st.columns(4)
                c1.metric(f"الطلب الكلي ({fcd}د)",f"{df_fc.predicted.sum():.1f} كجم")
                c2.metric("يومي متوسط",f"{df_fc.predicted.mean():.1f} كجم")
                c3.metric("المضاعف الموسمي",f"×{df_fc.mult.mean():.2f}")
                c4.metric("وقت التنبؤ",f"{el}s")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df_fc.date,y=df_fc.upper,mode="lines",line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=df_fc.date,y=df_fc.lower,fill="tonexty",
                                         fillcolor="rgba(139,26,26,0.12)",line=dict(width=0),name="نطاق الثقة 85%"))
                fig.add_trace(go.Scatter(x=df_fc.date,y=df_fc.predicted,mode="lines+markers",
                                         name="تنبؤ Enterprise LightGBM",line=dict(color="#8B1A1A",width=3),marker=dict(size=8)))
                fig.update_layout(title=f"تنبؤ — {sel}",height=380,hovermode="x unified")
                st.plotly_chart(fig,use_container_width=True)
                st.dataframe(df_fc.rename(columns={"date":"التاريخ","predicted":"المتوقع كجم",
                                                     "lower":"الحد الأدنى","upper":"الحد الأعلى",
                                                     "season":"الموسم","mult":"المضاعف"}),
                             use_container_width=True,hide_index=True)
                # ── SHAP Explanation Panel (Phase 1 — FR-15) ─────────────────────
                with st.expander("🔬 تفسير التنبؤ — SHAP Feature Contributions"):
                    try:
                        import shap
                        # Rebuild features for first forecast day
                        cx2=sqlite3.connect(DB)
                        df_hist=pd.read_sql_query(
                            f"SELECT sale_date,SUM(weight_kg) kg,AVG(unit_price) price,"
                            f"AVG(unit_cost) cost FROM sales WHERE product_id={pid} "
                            f"GROUP BY sale_date ORDER BY sale_date",cx2)
                        cx2.close()
                        if not df_hist.empty and len(df_hist)>=14:
                            df_hist["sale_date"]=pd.to_datetime(df_hist["sale_date"],format="mixed",errors="coerce").dt.date
                            kg_v=df_hist["kg"].values.tolist()
                            pr_v=df_hist["price"].values.tolist()
                            co_v=df_hist["cost"].values.tolist()
                            fd1=(df_hist["sale_date"].iloc[-1]+timedelta(days=1))
                            inv_r=_conn().execute("SELECT current_stock_kg FROM inventory WHERE product_id=?",(pid,)).fetchone()
                            inv_s=float(inv_r[0]) if inv_r else 0.0
                            feat_vals=_compute_features(kg_v,pr_v,co_v,inv_s,fd1)
                            feat_names=ENT_FEATURES if ENT_FEATURES else [f"f{i}" for i in range(len(feat_vals))]
                            X_shap=np.array([feat_vals])
                            explainer=shap.TreeExplainer(ENT_MODEL)
                            shap_vals=explainer.shap_values(X_shap)
                            if isinstance(shap_vals,list): sv=shap_vals[0][0]
                            else: sv=shap_vals[0]
                            shap_df=pd.DataFrame({
                                "الميزة":[ENT_FEAT_DESC.get(f,f) for f in feat_names],
                                "التأثير على التنبؤ":sv.tolist()
                            }).sort_values("التأثير على التنبؤ",key=abs,ascending=False).head(15)
                            fig_sh=px.bar(shap_df,x="التأثير على التنبؤ",y="الميزة",orientation="h",
                                         color="التأثير على التنبؤ",color_continuous_scale="RdBu",
                                         title="تأثير كل ميزة على تنبؤ اليوم الأول")
                            fig_sh.update_layout(height=420)
                            st.plotly_chart(fig_sh,use_container_width=True)
                            st.caption("القيم الموجبة (أزرق) ترفع التنبؤ. القيم السالبة (أحمر) تخفضه.")
                        else: st.info("يلزم 14 يوم من البيانات التاريخية لتفسير SHAP.")
                    except ImportError:
                        st.info("💡 شغل `pip install shap` لتفعيل تفسير SHAP")
                    except Exception as e:
                        st.warning(f"SHAP غير متاح: {e}")
    else:
        c1,c2=st.columns(2)
        with c1: sel_c=st.selectbox("نوع الذبيحة:",list(CARCASS_YIELD.keys()))
        with c2: fcd2=st.slider("أيام التنبؤ:",3,14,7,key="carcass_slider")
        if st.button("🚀 حساب احتياج الذبائح",type="primary"):
            with st.spinner("جاري تجميع التوقعات..."):
                yp=CARCASS_YIELD.get(sel_c,0.70); cm=CARCASS_CUT_MAP.get(sel_c,[])
                total_clean=0.0; agg=None
                for cn,_ in cm:
                    ci=prods()[prods().name_ar==cn]
                    if not ci.empty:
                        fc2=enterprise_forecast(int(ci.id.values[0]),fcd2)
                        if fc2:
                            df2=pd.DataFrame(fc2)
                            total_clean+=df2.predicted.sum()
                            agg=df2[["date","predicted","season","mult"]].copy() if agg is None else agg.copy()
                            if agg is not None: agg["predicted"]+=df2["predicted"].values[:len(agg)]
            if total_clean>0:
                bone_in=total_clean/yp; avg_w=350.0 if "بلدي" in sel_c else 50.0
                c1,c2,c3=st.columns(3)
                c1.metric("اللحم الصافي المطلوب",f"{total_clean:.1f} كجم")
                c2.metric(f"{sel_c} بالعظم",f"{bone_in:.1f} كجم",delta=f"نسبة التشفية {yp*100:.0f}%",delta_color="off")
                c3.metric("عدد الذبائح التقديري",f"~ {bone_in/avg_w:.1f}")
                if agg is not None:
                    agg["predicted_bone_in"]=agg["predicted"]/yp
                    fig4=px.bar(agg,x="date",y="predicted_bone_in",title=f"الاحتياج اليومي من {sel_c} بالعظم",
                                labels={"predicted_bone_in":"الوزن بالعظم كجم","date":"التاريخ"})
                    st.plotly_chart(fig4,use_container_width=True)
            else: st.warning("بيانات غير كافية لقطعيات هذه الذبيحة.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🤖 المساعد الذكي — Natural-language holiday forecasting + XAI + granularity
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 المساعد الذكي":
    from ai_assistant_page import show_ai_assistant
    show_ai_assistant(
        ENT_MODEL=ENT_MODEL,
        ENT_FEATURES=ENT_FEATURES,
        ENT_FEAT_DESC=ENT_FEAT_DESC,
        enterprise_forecast_fn=enterprise_forecast,
        get_mult_fn=get_mult,
        compute_features_fn=_compute_features,
        conn_fn=_conn,
        prods_fn=prods,
        current_user=uname,
        gemini_env_key=GEMINI_API_ENV,
        gemini_model=GEMINI_MODEL,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📓 استيراد من الدفتر الورقي — Handwritten notebook OCR import
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡 توصيات الشراء":
    st.title("💡 توصيات الشراء الذكية")
    fcd=st.slider("أيام التنبؤ:",3,14,7)
    if st.button("🧠 توليد التوصيات",type="primary"):
        with st.spinner("AI يحلل..."):
            pds=prods(); recs=[]
            for _,prod in pds.iterrows():
                pid2=int(prod.id); fc2=enterprise_forecast(pid2,fcd) if ENT_MODEL else []
                if not fc2: continue
                pred_kg=sum(f["predicted"] for f in fc2)
                price=float(prod.avg_price); cost=float(prod.avg_cost); margin=float(prod.avg_margin)
                reorder=float(prod.reorder_point); ratio=pred_kg/reorder if reorder>0 else 1
                prof=min(margin/5,10); urg=(10 if ratio>=2 else 8 if ratio>=1.5 else 6 if ratio>=1 else 4 if ratio>=0.5 else 2)
                dem=min(ratio*5,10); ceff=min((price-cost)/price*100/5,10) if price>0 else 0
                score=round(prof*0.35+urg*0.35+dem*0.20+ceff*0.10,2)
                pri="HIGH" if score>=7 else "MEDIUM" if score>=4.5 else "LOW"
                cur=_conn().execute("SELECT current_stock_kg FROM inventory WHERE product_id=?",(pid2,)).fetchone()
                cur_kg=float(cur[0]) if cur else 0.0
                rec_qty=round(max(0,pred_kg*1.1-cur_kg),1)
                recs.append(dict(name_ar=str(prod.name_ar),priority=pri,
                                 urg_ar={"HIGH":"🔴 عاجل","MEDIUM":"🟡 متوسط","LOW":"🟢 منخفض"}[pri],
                                 score=score,pred_kg=round(pred_kg,1),rec_qty=rec_qty,cur_kg=cur_kg))
        recs.sort(key=lambda x:x["score"],reverse=True)
        c1,c2,c3=st.columns(3)
        c1.metric("🔴 عاجل",len([r for r in recs if r["priority"]=="HIGH"]))
        c2.metric("🟡 متوسط",len([r for r in recs if r["priority"]=="MEDIUM"]))
        c3.metric("🟢 منخفض",len([r for r in recs if r["priority"]=="LOW"]))
        for pr,pl in [("HIGH","🔴"),("MEDIUM","🟡"),("LOW","🟢")]:
            grp=[r for r in recs if r["priority"]==pr]
            if not grp: continue
            st.subheader(f"{pl} {pr} ({len(grp)} منتجات)")
            for r in grp:
                with st.expander(f"{r['urg_ar']}  {r['name_ar']}  |  درجة: {r['score']}/10"):
                    cx1,cx2=st.columns(2)
                    cx1.metric("الطلب المتوقع",f"{r['pred_kg']:.1f} كجم")
                    cx2.metric("الكمية المقترحة للشراء",f"{r['rec_qty']:.1f} كجم")
                    action={"HIGH":"يجب الشراء فوراً","MEDIUM":"يُنصح خلال 48 ساعة","LOW":"المخزون كافٍ"}[pr]
                    st.markdown(f"<div class='rec-box'>📦 المخزون: {r['cur_kg']:.1f} كجم | 🎯 {action}</div>",unsafe_allow_html=True)

elif page == "📢 التسويق الذكي":
    from advanced_marketing_page import show_advanced_marketing_page
    show_advanced_marketing_page(db_path=DB, gemini_api_key=GEMINI_API_ENV, get_mult_fn=get_mult)

elif page == "⚙️ تحليل WAC":
    st.title("⚙️ تحليل التكلفة الموزونة (WAC)")
    days_wac = st.slider("فترة حساب WAC (يوم):", 30, 365, 90, key="wac_days_slider")
    dw_real = compute_real_wac(DB, days_back=days_wac)
    if dw_real.empty:
        st.warning("لا توجد بيانات مشتريات كافية لحساب WAC الحقيقي في هذه الفترة.")
        st.stop()
    dw = dw_real.rename(columns={"wac_egp_per_kg":"wac","sell_price":"price","actual_margin_pct":"act_mg","target_margin":"target","margin_gap":"gap"})
    c1,c2,c3,c4=st.columns(4)
    c1.metric("متوسط WAC (حقيقي)",f"{dw.wac.mean():.2f} ج/كجم"); c2.metric("متوسط سعر البيع",f"{dw.price.mean():.2f} ج/كجم")
    c3.metric("الهامش الفعلي (من WAC)",f"{dw.act_mg.mean():.1f}%"); c4.metric("دون الهدف",len(dw[dw.gap<0]))
    st.subheader("📊 WAC مقابل سعر البيع")
    fig_wac=go.Figure()
    fig_wac.add_trace(go.Bar(name="WAC",x=dw.name_ar,y=dw.wac,marker_color="#C0392B"))
    fig_wac.add_trace(go.Bar(name="سعر البيع",x=dw.name_ar,y=dw.price,marker_color="#27AE60"))
    fig_wac.update_layout(barmode="group",height=370,xaxis_tickangle=-45)
    st.plotly_chart(fig_wac,use_container_width=True)
    st.subheader("📉 فجوة الهامش")
    dws=dw.sort_values("gap")
    fig_g=go.Figure(go.Bar(x=dws.name_ar,y=dws.gap,
                            marker_color=["#dc3545" if v<0 else "#28a745" for v in dws.gap],
                            text=[f"{v:+.1f}%" for v in dws.gap],textposition="outside"))
    fig_g.add_hline(y=0,line_color="black",line_width=1)
    fig_g.update_layout(height=300,xaxis_tickangle=-45)
    st.plotly_chart(fig_g,use_container_width=True)
    st.subheader("🔵 تحليل الموردين")
    sup_df=qr("SELECT supplier_name,SUM(weight_kg) total_kg,AVG(cost_per_kg) avg_cost,COUNT(*) orders "
              "FROM purchases WHERE supplier_name IS NOT NULL GROUP BY supplier_name ORDER BY total_kg DESC")
    if not sup_df.empty:
        fig_sc=px.scatter(sup_df,x="avg_cost",y="total_kg",size="orders",text="supplier_name",
                          color="orders",color_continuous_scale="Reds",
                          labels={"avg_cost":"متوسط التكلفة","total_kg":"إجمالي الكميات","orders":"عدد الطلبات"})
        fig_sc.update_traces(textposition="top center"); fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc,use_container_width=True)
        sup_df.columns=["المورد","إجمالي كجم","متوسط التكلفة","الطلبات"]
        st.dataframe(sup_df,use_container_width=True,hide_index=True)

elif page == "📈 BI Analytics":
    st.title("📈 BI Analytics — ذكاء الأعمال المتقدم")
    d=st.slider("الفترة:",30,700,365)
    dy=qr(f"SELECT sale_date,SUM(total_price) rev,SUM(gross_profit) pft FROM sales WHERE sale_date>=date('now','-{d} days') GROUP BY sale_date ORDER BY sale_date")
    if dy.empty: st.info("لا بيانات"); st.stop()
    dy["sale_date"]=pd.to_datetime(dy.sale_date,format="mixed",errors="coerce")
    dy["ma7"]=dy.rev.rolling(7,min_periods=1).mean(); dy["ma30"]=dy.rev.rolling(30,min_periods=1).mean()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("متوسط يومي",f"{dy.rev.mean():,.0f} ج"); c2.metric("أعلى يوم",f"{dy.rev.max():,.0f} ج")
    c3.metric("الإجمالي",f"{dy.rev.sum():,.0f} ج"); c4.metric("أيام النشاط",f"{(dy.rev>0).sum()}")
    fig_m=go.Figure()
    fig_m.add_trace(go.Bar(x=dy.sale_date,y=dy.rev,name="الإيرادات",marker_color="#E74C3C",opacity=0.55))
    fig_m.add_trace(go.Scatter(x=dy.sale_date,y=dy.ma7,name="MA-7",line=dict(color="#8B1A1A",width=2)))
    fig_m.add_trace(go.Scatter(x=dy.sale_date,y=dy.ma30,name="MA-30",line=dict(color="#27AE60",width=2,dash="dash")))
    fig_m.update_layout(height=370,hovermode="x unified"); st.plotly_chart(fig_m,use_container_width=True)
    dy["month"]=dy.sale_date.dt.to_period("M").astype(str)
    monthly=dy.groupby("month").agg(rev=("rev","sum"),pft=("pft","sum")).reset_index()
    st.subheader("📅 الأداء الشهري")
    fig_mo=go.Figure()
    fig_mo.add_trace(go.Bar(x=monthly.month,y=monthly.rev,name="الإيرادات",marker_color="#C0392B",opacity=0.8))
    fig_mo.add_trace(go.Scatter(x=monthly.month,y=monthly.pft,name="الأرباح",line=dict(color="#27AE60",width=2),yaxis="y2"))
    fig_mo.update_layout(height=320,yaxis2=dict(overlaying="y",side="right"),hovermode="x unified")
    st.plotly_chart(fig_mo,use_container_width=True)
    st.subheader("🗓️ خريطة الحرارة — اليوم × الشهر")
    dy["dow_ar"]=dy.sale_date.dt.dayofweek.map({0:"الإثنين",1:"الثلاثاء",2:"الأربعاء",3:"الخميس",4:"الجمعة",5:"السبت",6:"الأحد"})
    piv=dy.pivot_table(values="rev",index="dow_ar",columns="month",aggfunc="mean")
    if not piv.empty:
        fig_hm=px.imshow(piv,color_continuous_scale="Reds",aspect="auto",title="متوسط الإيرادات حسب اليوم والشهر")
        fig_hm.update_layout(height=320,xaxis_tickangle=-45); st.plotly_chart(fig_hm,use_container_width=True)
    st.subheader("📦 خريطة حرارة المنتجات الشهرية")
    pm=qr(f"SELECT p.name_ar,strftime('%Y-%m',s.sale_date) mo,SUM(s.total_price) rev "
          f"FROM sales s JOIN products p ON s.product_id=p.id WHERE s.sale_date>=date('now','-{d} days') "
          f"GROUP BY p.id,mo ORDER BY mo")
    if not pm.empty:
        piv2=pm.pivot_table(values="rev",index="name_ar",columns="mo",aggfunc="sum").fillna(0)
        fig_ph=px.imshow(piv2,color_continuous_scale="RdYlGn",aspect="auto",title="إيرادات كل منتج شهرياً")
        fig_ph.update_layout(height=500,xaxis_tickangle=-45); st.plotly_chart(fig_ph,use_container_width=True)

elif page == "📅 رصيد المخزون اليومي":
    st.title("📅 رصيد المخزون اليومي")
    st.markdown("<div class='info-box'>📊 بيانات من <b>Daily_Inventory_Balance.xlsx</b> — مخزنة في قاعدة البيانات</div>",unsafe_allow_html=True)
    bal=qr("SELECT * FROM daily_balance ORDER BY balance_date")
    if bal.empty: st.warning("لا توجد بيانات رصيد يومي"); st.stop()
    bal["balance_date"]=pd.to_datetime(bal["balance_date"],format="mixed",errors="coerce")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("أيام البيانات",f"{len(bal):,}"); c2.metric("إجمالي المشتريات",f"{bal.purchases_net_kg.sum():,.1f} كجم")
    c3.metric("إجمالي المبيعات",f"{bal.sales_kg.sum():,.1f} كجم"); c4.metric("صافي الفائض/العجز",f"{bal.daily_surplus_deficit.sum():+,.1f} كجم")
    dr=st.slider("نطاق البيانات:",min_value=bal.balance_date.min().date(),max_value=bal.balance_date.max().date(),
                 value=(bal.balance_date.min().date(),bal.balance_date.max().date()),format="YYYY-MM-DD")
    mask=((bal.balance_date.dt.date>=dr[0])&(bal.balance_date.dt.date<=dr[1])); bf=bal[mask].copy()
    fig1=go.Figure()
    fig1.add_trace(go.Bar(x=bf.balance_date,y=bf.purchases_net_kg,name="المشتريات",marker_color="#1A5276",opacity=0.75))
    fig1.add_trace(go.Bar(x=bf.balance_date,y=bf.sales_kg,name="المبيعات",marker_color="#C0392B",opacity=0.75))
    fig1.update_layout(barmode="group",height=340,hovermode="x unified"); st.plotly_chart(fig1,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=bf.balance_date,y=bf.cumulative_balance,fill="tozeroy",
                               fillcolor="rgba(139,26,26,0.15)",line=dict(color="#8B1A1A",width=2),name="الرصيد التراكمي"))
    fig2.add_hline(y=0,line_dash="dash",line_color="red"); fig2.update_layout(height=300,hovermode="x unified")
    st.plotly_chart(fig2,use_container_width=True)
    csv_out=bf[["balance_date","purchases_net_kg","sales_kg","daily_surplus_deficit","cumulative_balance"]].copy()
    csv_out["balance_date"]=csv_out["balance_date"].astype(str)
    st.download_button("⬇️ تحميل CSV",csv_out.to_csv(index=False).encode("utf-8"),
                       file_name="daily_balance_export.csv",mime="text/csv")

elif page == "💹 توقع الأسعار":
    if not PRICE_FORECAST_AVAILABLE:
        st.error("❌ توقع الأسعار غير متاح. تأكد من وجود price_forecast.py")
    else:
        show_price_forecast_page(db_path=DB)

elif page == "⚙️ محرك الأتمتة":
    if not AUTOMATION_AVAILABLE:
        st.error("❌ محرك الأتمتة غير متاح. تأكد من وجود automation_engine.py")
    else:
        show_automation_page(db_path=DB)

elif page == "🏭 Enterprise Forecast":
    st.title("🏭 Enterprise Forecast — نظام التنبؤ المؤسسي")
    nc=len(ENT_FEATURES) if ENT_FEATURES else "?"
    if ENT_MODEL:
        st.markdown(f"<div class='ent-card'>🏭 <b>Enterprise LightGBM — محمّل ✅</b> ({nc} features)<br>"
                    "Lag · Rolling · Inventory Pressure · Price Dynamics · Seasonality · Ramadan</div>",
                    unsafe_allow_html=True)
    else: st.warning("⚠️ enterprise_lightgbm_model.pkl غير موجود")
    st.subheader("📊 التنبؤ المؤسسي — فترة عيد الأضحى 2026")
    try:
        fc_df=pd.read_csv("final_enterprise_forecast.csv")
        fc_df["forecast_date"]=pd.to_datetime(fc_df["forecast_date"],format="mixed",errors="coerce")
        c1,c2,c3=st.columns(3)
        c1.metric("أسابيع التنبؤ",f"{len(fc_df)}"); c2.metric("إجمالي الطلب المتوقع",f"{fc_df.expected_demand_kg.sum():,.1f} كجم")
        c3.metric("مستوى التنبيه الأعلى",fc_df.alert_level.iloc[-1] if len(fc_df) else "—")
        fig_ent=go.Figure()
        fig_ent.add_trace(go.Scatter(x=fc_df.forecast_date,y=fc_df.upper_bound,mode="lines",line=dict(width=0),showlegend=False))
        fig_ent.add_trace(go.Scatter(x=fc_df.forecast_date,y=fc_df.lower_bound,fill="tonexty",
                                      fillcolor="rgba(15,110,86,0.15)",line=dict(width=0),name="نطاق الثقة 85%"))
        fig_ent.add_trace(go.Scatter(x=fc_df.forecast_date,y=fc_df.expected_demand_kg,mode="lines+markers+text",
                                      text=[f"{v:.0f}كجم" for v in fc_df.expected_demand_kg],textposition="top center",
                                      line=dict(color="#0F6E56",width=3),marker=dict(size=10),name="الطلب المتوقع"))
        fig_ent.update_layout(height=400,hovermode="x unified"); st.plotly_chart(fig_ent,use_container_width=True)
        alert_clr={"HIGH":"#dc3545","CRITICAL":"#6f0000","MEDIUM":"#fd7e14","LOW":"#28a745"}
        for _,row in fc_df.iterrows():
            clr=alert_clr.get(row.alert_level,"#888")
            st.markdown(f"<div style='background:#f9f9f9;border-left:5px solid {clr};border-radius:8px;padding:12px 18px;margin:6px 0'>"
                        f"<b style='color:{clr}'>🔔 {row.alert_level}</b> — {row.holiday_name}<br>"
                        f"📅 {str(row.forecast_date)[:10]} | 📦 {row.expected_demand_kg:,.1f} كجم ({row.demand_change_percent:+.1f}%)</div>",
                        unsafe_allow_html=True)
    except FileNotFoundError: st.error("❌ final_enterprise_forecast.csv غير موجود")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 15: إدارة الموردين  (NEW — Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 إدارة الموردين":
    st.title("🚚 إدارة الموردين")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("➕ إضافة مورد جديد")
        with st.form("sup_form",clear_on_submit=True):
            s_name=st.text_input("اسم المورد (عربي) *")
            s_name_en=st.text_input("اسم المورد (إنجليزي)")
            s_phone=st.text_input("رقم الهاتف")
            s_area=st.text_input("المنطقة / العنوان")
            s_terms=st.selectbox("شروط الدفع",["نقدي","آجل 7 أيام","آجل 30 يوم","مقايضة"])
            s_notes=st.text_area("ملاحظات",height=60)
            if st.form_submit_button("حفظ المورد ✓",type="primary",use_container_width=True):
                if not s_name: st.error("اسم المورد مطلوب")
                else:
                    cx=_raw()
                    cx.execute("INSERT INTO suppliers (name_ar,name_en,phone,area,payment_terms,notes) VALUES (?,?,?,?,?,?)",
                               (s_name,s_name_en,s_phone,s_area,s_terms,s_notes))
                    cx.commit(); cx.close()
                    log_audit(uname,"INSERT","suppliers",None,f"Added supplier: {s_name}")
                    _bust(); st.success(f"✅ تمت إضافة {s_name}"); st.rerun()
    with c2:
        st.subheader("📋 قائمة الموردين")
        sups=qr("SELECT name_ar,name_en,phone,area,payment_terms,active FROM suppliers ORDER BY id DESC")
        if not sups.empty:
            sups["active"]=sups["active"].map({1:"✅ نشط",0:"❌ موقوف"})
            sups.columns=["المورد","الاسم بالإنجليزية","الهاتف","المنطقة","شروط الدفع","الحالة"]
            st.dataframe(sups,use_container_width=True,hide_index=True)
        else: st.info("لا موردين مسجلين بعد.")
    st.divider()
    st.subheader("📊 أداء الموردين من المشتريات")
    sup_perf=qr("SELECT supplier_name,COUNT(*) orders,SUM(weight_kg) kg,AVG(cost_per_kg) avg_cost,"
                "MIN(cost_per_kg) min_cost,MAX(cost_per_kg) max_cost "
                "FROM purchases WHERE supplier_name IS NOT NULL GROUP BY supplier_name ORDER BY kg DESC")
    if not sup_perf.empty:
        sup_perf["consistency"]=round((1-(sup_perf["max_cost"]-sup_perf["min_cost"])/(sup_perf["avg_cost"]+0.01))*100,1)
        fig_s=px.scatter(sup_perf,x="avg_cost",y="kg",size="orders",text="supplier_name",
                         color="consistency",color_continuous_scale="RdYlGn",
                         title="الموردون: التكلفة مقابل الكمية (اللون = ثبات الأسعار)",
                         labels={"avg_cost":"متوسط التكلفة","kg":"إجمالي الكميات كجم"})
        fig_s.update_traces(textposition="top center"); fig_s.update_layout(height=400)
        st.plotly_chart(fig_s,use_container_width=True)
        sup_perf.columns=["المورد","الطلبات","إجمالي كجم","متوسط التكلفة","أقل سعر","أعلى سعر","ثبات الأسعار %"]
        st.dataframe(sup_perf.round(2),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 16: النسخ الاحتياطي  (NEW Admin — Phase 1 DB-04)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💾 النسخ الاحتياطي":
    if role!="admin": st.error("للمدراء فقط"); st.stop()
    st.title("💾 النسخ الاحتياطي واستعادة البيانات")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("🔒 إنشاء نسخة احتياطية الآن")
        db_size=os.path.getsize(DB)/1024/1024 if os.path.exists(DB) else 0
        st.metric("حجم قاعدة البيانات",f"{db_size:.2f} MB")
        if st.button("💾 إنشاء نسخة احتياطية",type="primary",use_container_width=True):
            dst=backup_db()
            log_audit(uname,"BACKUP","database",None,f"Backup created: {dst.name}")
            st.success(f"✅ تم إنشاء النسخة: {dst.name}")
    with c2:
        st.subheader("📋 النسخ المتاحة")
        backups=list_backups()
        if backups:
            st.metric("عدد النسخ المتاحة",f"{len(backups)} (آخر {BACKUP_KEEP} يوم)")
        else:
            st.info("لا توجد نسخ احتياطية بعد.")
    st.divider()
    st.subheader("📦 قائمة النسخ الاحتياطية")
    backups=list_backups()
    if backups:
        for b in backups:
            c1,c2,c3=st.columns([3,1,1])
            c1.write(f"📁 {b['name']} — {b['size_kb']} KB — {b['created']}")
            with open(b["path"],"rb") as f:
                c2.download_button("⬇️",f.read(),file_name=b["name"],mime="application/octet-stream",key=f"dl_{b['name']}")
            if c3.button("♻️ استعادة",key=f"rst_{b['name']}"):
                st.warning(f"⚠️ سيتم استبدال قاعدة البيانات الحالية بالنسخة {b['name']}. هذا الإجراء لا يمكن التراجع عنه!")
                if st.button("✅ تأكيد الاستعادة",key=f"confirm_{b['name']}",type="primary"):
                    restore_db(b["path"],uname); _bust()
                    st.success("✅ تمت الاستعادة. سيتم إعادة تحميل النظام."); st.rerun()
    else: st.info("لا توجد نسخ احتياطية. أنشئ نسخة أولاً باستخدام الزر أعلاه.")
    st.divider()
    st.subheader("🔄 مزامنة المخزون بعد الاستيراد")
    if st.button("🔄 إعادة حساب المخزون بالكامل",type="primary"):
        cx=_raw()
        cx.execute("DELETE FROM inventory")
        cx.execute("""INSERT INTO inventory (product_id,current_stock_kg,last_updated)
            SELECT p.id,MAX(0.0,COALESCE((SELECT SUM(weight_kg) FROM purchases WHERE product_id=p.id),0)-
                              COALESCE((SELECT SUM(weight_kg) FROM sales WHERE product_id=p.id),0)),
                   datetime('now') FROM products p""")
        cx.commit(); cx.close()
        log_audit(uname,"RESYNC","inventory",None,"Full inventory recalculation")
        _bust(); st.success("✅ تمت مزامنة المخزون!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 17: سجل التدقيق  (NEW Admin — Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜 سجل التدقيق":
    if role!="admin": st.error("للمدراء فقط"); st.stop()
    st.title("📜 سجل التدقيق — Audit Trail")
    c1,c2,c3=st.columns(3)
    with c1: days_a=st.slider("الفترة:",1,90,30,key="audit_days")
    with c2: user_f=st.text_input("فلترة بالمستخدم:")
    with c3: action_f=st.text_input("فلترة بالإجراء:")
    where_parts = ["timestamp >= datetime('now', ?)"]
    params_a = ["-" + str(days_a) + " days"]
    if user_f.strip():
        where_parts.append("username LIKE ?")
        params_a.append("%" + user_f.strip() + "%")
    if action_f.strip():
        where_parts.append("action LIKE ?")
        params_a.append("%" + action_f.strip() + "%")
    where_a_sql = " AND ".join(where_parts)
    total_a = _conn().execute("SELECT COUNT(*) FROM audit_log WHERE " + where_a_sql, tuple(params_a)).fetchone()[0] or 0
    ps=50; tp=max(1,(total_a+ps-1)//ps); pn=st.number_input("الصفحة:",1,tp,1,key="audit_page")-1
    audit = pd.read_sql_query(
        "SELECT timestamp,username,action,table_name,summary FROM audit_log WHERE " + where_a_sql + " ORDER BY id DESC LIMIT ? OFFSET ?",
        _conn(), params=tuple(params_a) + (ps, pn*ps)
    )
    c1,c2,c3=st.columns(3)
    c1.metric("إجمالي السجلات",f"{total_a:,}"); c2.metric("الصفحة الحالية",f"{pn+1}/{tp}")
    c3.metric("المستخدمون النشطون", _conn().execute("SELECT COUNT(DISTINCT username) FROM audit_log WHERE " + where_a_sql, tuple(params_a)).fetchone()[0] or 0)
    st.divider()
    if not audit.empty:
        for _,row in audit.iterrows():
            act_color={"INSERT":"#28a745","UPDATE":"#ffc107","DELETE":"#dc3545",
                       "BACKUP":"#17a2b8","RESTORE":"#6f0000","LOGIN":"#1A5276",
                       "CHANGE_PASSWORD":"#6f42c1"}.get(str(row.action),"#6c757d")
            st.markdown(
                f"<div class='audit-row'>"
                f"<b style='color:{act_color}'>{row.action}</b> — "
                f"<b>{row.username}</b> | {str(row.timestamp)[:19]} | "
                f"<span style='color:#888'>{row.table_name or ''}</span> | {row.summary or ''}"
                f"</div>", unsafe_allow_html=True)
        csv_audit=audit.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ تحميل سجل التدقيق CSV",csv_audit,f"audit_{date.today()}.csv","text/csv")
    else: st.info("لا سجلات في الفترة المحددة.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 18: إدارة المستخدمين
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 إدارة المستخدمين":
    if role!="admin": st.error("للمدراء فقط"); st.stop()
    st.title("👥 إدارة المستخدمين")
    ud=qr("SELECT id,username,full_name,role,active,created_at,last_login,COALESCE(first_login,0) FROM users ORDER BY id")
    ud.columns=["ID","اسم المستخدم","الاسم الظاهر","الدور","نشط","تاريخ الإنشاء","آخر دخول","أول دخول؟"]
    ud["نشط"]=ud["نشط"].map({1:"نعم",0:"لا"})
    ud["الدور"]=ud["الدور"].map({"admin":"مدير","staff":"موظف","viewer":"مشاهد"})
    ud["أول دخول؟"]=ud["أول دخول؟"].map({1:"⚠️ لم يغير كلمة المرور",0:"✅ آمن"})
    st.dataframe(ud,use_container_width=True,hide_index=True)
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        st.subheader("تغيير كلمة مرور")
        with st.form("cpf"):
            all_users=qr("SELECT username FROM users").username.tolist()
            tu=st.selectbox("المستخدم:",all_users)
            np1=st.text_input("الجديدة",type="password"); np2=st.text_input("تأكيد",type="password")
            if st.form_submit_button("تحديث"):
                if len(np1)<8: st.error("8 أحرف على الأقل")
                elif np1!=np2: st.error("غير متطابق")
                else:
                    cx=_raw(); cx.execute("UPDATE users SET hashed_pw=?,first_login=0 WHERE username=?",(_hash(np1),tu))
                    cx.commit(); cx.close()
                    log_audit(uname,"CHANGE_PASSWORD","users",None,f"Changed password for {tu}")
                    st.success(f"تم تحديث {tu}")
    with c2:
        st.subheader("تغيير الدور")
        with st.form("crf"):
            others=qr(f"SELECT username FROM users WHERE username!='{uname}'").username.tolist()
            if others:
                tu2=st.selectbox("المستخدم:",others)
                nr=st.selectbox("الدور:",["admin","staff","viewer"],
                                format_func=lambda x:{"admin":"مدير","staff":"موظف","viewer":"مشاهد"}[x])
                ia=st.checkbox("نشط",True)
                if st.form_submit_button("حفظ"):
                    cx=_raw(); cx.execute("UPDATE users SET role=?,active=? WHERE username=?",(nr,1 if ia else 0,tu2))
                    cx.commit(); cx.close()
                    log_audit(uname,"UPDATE","users",None,f"{tu2} → role={nr}, active={ia}")
                    st.success(f"تم تحديث {tu2}"); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 19: إدارة البيانات  (with validation)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ إدارة البيانات":
    if role!="admin": st.error("للمدراء فقط"); st.stop()
    st.title("⚙️ إدارة وتعديل البيانات")
    st.markdown("⚠️ **التعديلات تؤثر مباشرة على قاعدة البيانات.**")
    st.subheader("📝 تعديل سجلات قاعدة البيانات")
    target_table=st.selectbox("اختر الجدول:",["sales","purchases","products","inventory","expenses","daily_balance"])
    df_to_edit=qr(f"SELECT * FROM {target_table}")
    if not df_to_edit.empty:
        st.info(f"جدول: {target_table} — {len(df_to_edit):,} سجل")
        edited_df=st.data_editor(df_to_edit,use_container_width=True,num_rows="dynamic")
        if st.button("💾 حفظ التعديلات",type="primary"):
            success, msg = safe_replace_table(DB, target_table, edited_df)
            if success:
                log_audit(uname,"UPDATE",target_table,None,f"Transaction-safe save: {len(edited_df)} rows")
                _bust(); st.success(msg); time.sleep(1); st.rerun()
            else:
                st.error(msg)
    else: st.warning("الجدول فارغ.")
    st.divider()
    st.subheader("📂 رفع ملفات البيانات (CSV / XLSX) — مع التحقق التلقائي")
    uploaded_file=st.file_uploader("اختر ملف البيانات",type=["csv","xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):  df_upload=pd.read_csv(uploaded_file)
            else:                                     df_upload=pd.read_excel(uploaded_file)
            st.success(f"✅ تم استلام: {uploaded_file.name} — {len(df_upload):,} صف")
            st.dataframe(df_upload.head(5),use_container_width=True)
            # Detect table
            if "sale_date" in df_upload.columns:         tgt="sales"
            elif "purchase_date" in df_upload.columns:   tgt="purchases"
            elif "balance_date" in df_upload.columns:    tgt="daily_balance"
            elif "current_stock_kg" in df_upload.columns:tgt="inventory"
            elif "expense_date" in df_upload.columns:    tgt="expenses"
            else: st.error("⚠️ لم يتم التعرف على نوع الملف."); st.stop()
            st.info(f"📋 سيتم الاستيراد إلى جدول: **{tgt}**")
            # Phase 0: Validate before import
            is_valid,msg=validate_import_df(df_upload,tgt)
            if not is_valid:
                st.error(f"❌ فشل التحقق من البيانات: {msg}")
            else:
                st.success("✅ التحقق من البيانات ناجح. اضغط لتأكيد الاستيراد.")
                if st.button("🚀 استيراد الملف",type="primary"):
                    with st.spinner("جاري الاستيراد..."):
                        success, msg = safe_append_import(DB, tgt, df_upload)
                        if success:
                            log_audit(uname,"IMPORT",tgt,None,f"{len(df_upload)} rows from {uploaded_file.name}")
                            _bust(); st.success(msg)
                            time.sleep(1); st.rerun()
                        else:
                            st.error(msg)
        except Exception as e: st.error(f"❌ خطأ في قراءة الملف: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📚 قاعدة المعرفة (RAG) — Phase 1
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 قاعدة المعرفة (RAG)":
    if not RAG_AVAILABLE:
        st.error("❌ RAG غير متاح. تأكد من تثبيت chromadb و sentence-transformers")
    else:
        show_rag_page(db_path=DB, gemini_api_key=GEMINI_API_ENV)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🤖 وكيل الذكاء الاصطناعي — Phase 2
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 وكيل الذكاء الاصطناعي":
    if not AGENT_AVAILABLE:
        st.error("❌ الوكيل غير متاح. تأكد من وجود phase2_agent_page.py")
    else:
        show_agent_page(
            db_path        = DB,
            gemini_api_key = GEMINI_API_ENV,
            ent_model      = ENT_MODEL,
            ent_features   = ENT_FEATURES,
            get_mult_fn    = get_mult,
        )


elif page == "👥 الوكلاء المتعددين":
    if not MULTI_AGENT_AVAILABLE:
        st.error("❌ الوكلاء المتعددين غير متاح. تأكد من وجود phase4_multi_agent.py")
    else:
        show_multi_agent_page(
            db_path=DB,
            gemini_api_key=GEMINI_API_ENV,
            ent_model=ENT_MODEL,
            ent_features=ENT_FEATURES,
            get_mult_fn=get_mult,
        )

elif page == "📊 مراقبة الذكاء الاصطناعي":
    from llm_observability import show_llm_observability_page
    show_llm_observability_page(db_path=DB)

elif page == "📊 لوحة المراقبة":
    if not MONITORING_AVAILABLE:
        st.error("❌ لوحة المراقبة غير متاحة. تأكد من وجود monitoring_dashboard.py")
    else:
        show_monitoring_page(db_path=DB)

elif page == "🧪 تقييم الجودة":
    from evaluation_harness import show_evaluation_page
    show_evaluation_page(db_path=DB, gemini_api_key=GEMINI_API_ENV)


elif page == "📓 استيراد من الدفتر الورقي":
    from gemini_vision_ocr import show_gemini_ocr_page
    show_gemini_ocr_page(
        prods_fn=prods, ins_sale_fn=ins_sale, ins_pur_fn=ins_pur,
        log_audit_fn=log_audit, current_user=uname, gemini_api_key=GEMINI_API_ENV,
    )
