"""
ai_assistant_page.py — ButcherAI v9.1 AI Query Engine
=======================================================
Implements three capabilities:

1. Natural-language future holiday forecasting
   - "كم سيكون الطلب في عيد الأضحى 2027؟"
   - Hijri converter computes the exact Gregorian window automatically
   - Seasonal anchor method for long-horizon (300+ day) estimates

2. Explainable AI narrative on every response
   - SHAP TreeExplainer for feature contributions
   - Gemini converts SHAP values into Arabic/English plain text
   - Template fallback if Gemini unavailable

3. Daily / Weekly / Monthly granularity forecasting
   - Runs daily enterprise_forecast(), then resamples
   - Error bars re-computed correctly per granularity
   - Holiday detection at each aggregation level

HOW TO USE:
  Copy this file into your project folder, then add
  one elif block to your app.py sidebar page router:

  elif page == "🤖 المساعد الذكي":
      from ai_assistant_page import show_ai_assistant
      show_ai_assistant(ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                        enterprise_forecast, get_mult, _compute_features,
                        _conn, prods, uname,
                        GEMINI_API_ENV, GEMINI_MODEL)

  Also add "🤖 المساعد الذكي" to the pages list in the sidebar.
"""

from __future__ import annotations
import json
import re
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import date, timedelta, datetime

try:
    from hijri_converter import Gregorian, Hijri
    HAS_HIJRI = True
except ImportError:
    HAS_HIJRI = False

try:
    import genai_compat as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. FUTURE HOLIDAY COMPUTATION  (Q1)
# ─────────────────────────────────────────────────────────────────────────────
HIJRI_HOLIDAYS = {
    "رمضان":         (9,  1,  30),   # month, start_day, duration
    "عيد الفطر":     (10, 1,   3),
    "عيد الأضحى":    (12, 10,  4),
}
HOLIDAY_MULT = {
    "رمضان":     1.55,
    "عيد الفطر": 2.20,
    "عيد الأضحى": 3.00,
}

def compute_holiday_window(holiday_ar: str, hijri_year: int) -> tuple[date, date, float]:
    """
    Compute the Gregorian start/end dates for an Islamic holiday in a given
    Hijri year.  Works for any future year — no hardcoding needed.

    End date is computed by adding duration as a timedelta on the Gregorian
    side — this avoids Hijri month-length edge cases (some months are 29 days).

    Returns (start_date, end_date, demand_multiplier) or raises ValueError.
    """
    if not HAS_HIJRI:
        raise ValueError("hijri-converter not installed — run: pip install hijri-converter")

    info = HIJRI_HOLIDAYS.get(holiday_ar)
    if not info:
        raise ValueError(f"Unknown holiday: {holiday_ar}. Known: {list(HIJRI_HOLIDAYS)}")

    month, start_day, duration = info
    try:
        start_greg = Hijri(hijri_year, month, start_day).to_gregorian()
        start_date = date(start_greg.year, start_greg.month, start_greg.day)
        # Add duration on the Gregorian side to avoid Hijri month-length edge cases
        end_date   = start_date + timedelta(days=duration - 1)
    except Exception as e:
        raise ValueError(f"Hijri conversion failed for year {hijri_year}: {e}")

    mult = HOLIDAY_MULT.get(holiday_ar, 1.0)
    return start_date, end_date, mult

def gregorian_year_to_hijri(greg_year: int) -> int:
    """Approximate Hijri year for a given Gregorian year."""
    if HAS_HIJRI:
        h = Gregorian(greg_year, 6, 1).to_hijri()
        return h.year
    return greg_year - 579  # rough approximation


def get_upcoming_holidays(years_ahead: int = 3) -> list[dict]:
    """Return all major Islamic holidays from today up to years_ahead years out."""
    if not HAS_HIJRI:
        return []
    today = date.today()
    results = []
    h_today = Gregorian(today.year, today.month, today.day).to_hijri()
    for yr_offset in range(0, years_ahead + 1):
        h_year = h_today.year + yr_offset
        for hol_ar, (month, start_day, duration) in HIJRI_HOLIDAYS.items():
            try:
                start, end, mult = compute_holiday_window(hol_ar, h_year)
                if start >= today:
                    days_away = (start - today).days
                    results.append({
                        "holiday": hol_ar,
                        "hijri_year": h_year,
                        "start": start,
                        "end":   end,
                        "mult":  mult,
                        "days_away": days_away,
                        "greg_range": f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}",
                    })
            except Exception:
                pass
    results.sort(key=lambda x: x["start"])
    return results[:12]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-GRANULARITY FORECAST  (Q3)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_forecast(daily_preds: list, granularity: str) -> pd.DataFrame:
    """
    Aggregate daily forecast predictions to weekly or monthly granularity.

    Confidence intervals are re-computed as sqrt(sum of daily variances)
    — not simply summed — which gives statistically correct wider bands
    reflecting the true aggregated uncertainty.
    """
    if not daily_preds:
        return pd.DataFrame()

    df = pd.DataFrame(daily_preds)
    df["date"] = pd.to_datetime(df["date"])

    if granularity == "daily":
        df["period"]     = df["date"]
        df["period_str"] = df["date"].dt.strftime("%Y-%m-%d")
    elif granularity == "weekly":
        df["period"]     = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        df["period_str"] = df["date"].dt.to_period("W").astype(str)
    else:  # monthly
        df["period"]     = df["date"].dt.to_period("M").apply(lambda p: p.start_time)
        df["period_str"] = df["date"].dt.to_period("M").astype(str)

    # Sum predicted; re-compute CI correctly (sqrt of sum of variances)
    df["variance"] = ((df["upper"] - df["lower"]) / (2 * 1.44)) ** 2

    agg = df.groupby("period_str").agg(
        date=("period", "first"),
        predicted=("predicted", "sum"),
        variance=("variance", "sum"),
        season=("season", lambda x: x.mode()[0] if len(x) > 0 else ""),
        mult=("mult", "mean"),
        days=("predicted", "count"),
    ).reset_index(drop=True)

    agg["std"]   = np.sqrt(agg["variance"])
    agg["lower"] = np.maximum(0, agg["predicted"] - 1.44 * agg["std"])
    agg["upper"] = agg["predicted"] + 1.44 * agg["std"]
    agg["mae_est"] = agg["std"] * 0.8  # approximate MAE from std

    # % error estimate (lower is better)
    daily_vol_est = agg["predicted"] / agg["days"]
    agg["pct_err"] = (agg["mae_est"] / (agg["predicted"] + 1e-5) * 100).round(1)

    return agg[["date", "predicted", "lower", "upper", "season", "mult",
                "days", "pct_err"]].copy()


def build_forecast_chart(agg: pd.DataFrame, product_name: str,
                          granularity: str, holiday_info: dict = None) -> go.Figure:
    """Build a Plotly figure for a forecast at the specified granularity."""
    x = agg["date"].dt.strftime("%Y-%m-%d" if granularity == "daily" else
                                 "%Y-W%W"   if granularity == "weekly"  else "%Y-%m")

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=x, y=agg["upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=agg["lower"], fill="tonexty",
        fillcolor="rgba(139,26,26,0.10)", line=dict(width=0),
        name="نطاق الثقة 85%"))

    # Main bars (more readable than lines for weekly/monthly)
    if granularity == "daily":
        fig.add_trace(go.Scatter(
            x=x, y=agg["predicted"],
            mode="lines+markers",
            name=f"تنبؤ {granularity}",
            line=dict(color="#8B1A1A", width=2.5),
            marker=dict(size=6),
            hovertemplate="%{x}<br>المتوقع: %{y:.1f} كجم<extra></extra>"))
    else:
        fig.add_trace(go.Bar(
            x=x, y=agg["predicted"],
            name=f"تنبؤ {granularity}",
            marker_color="#8B1A1A",
            opacity=0.8,
            hovertemplate="%{x}<br>المتوقع: %{y:.1f} كجم<extra></extra>"))

    # Holiday marker
    if holiday_info:
        target_x = holiday_info.get("start", "").strftime("%Y-%m-%d")
        fig.add_vline(x=target_x, line_dash="dash", line_color="#D97706",
                      annotation_text=f"🎯 {holiday_info.get('holiday', '')}",
                      annotation_position="top right")

    # % error annotation on first bar
    if not agg.empty and "pct_err" in agg.columns:
        first_row = agg.iloc[0]
        fig.add_annotation(
            x=x.iloc[0] if len(x) > 0 else 0,
            y=float(first_row["predicted"]),
            text=f"±{first_row['pct_err']:.0f}%",
            showarrow=False,
            yshift=14,
            font=dict(size=10, color="#666"),
        )

    gran_ar = {"daily": "يومي", "weekly": "أسبوعي", "monthly": "شهري"}
    fig.update_layout(
        title=f"التنبؤ بالطلب — {product_name} ({gran_ar.get(granularity, granularity)})",
        height=380, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPLAINABLE AI NARRATIVE  (Q2)
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap_for_day(model, features_list: list, feat_vals: list) -> dict | None:
    """
    Compute SHAP values for a single prediction vector.
    Returns {feature_name: shap_value_in_kg} sorted by abs magnitude, or None.
    """
    if not HAS_SHAP or model is None:
        return None
    try:
        X = np.array([feat_vals])
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        result = {f: float(v) for f, v in zip(features_list, sv)}
        # Sort by absolute magnitude
        return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
    except Exception:
        return None


def build_shap_chart(shap_dict: dict, feat_desc: dict, top_n: int = 10) -> go.Figure:
    """Plotly waterfall / horizontal bar chart from SHAP values."""
    items = list(shap_dict.items())[:top_n]
    labels = [feat_desc.get(f, f) for f, _ in items]
    values = [v for _, v in items]
    colors = ["#1A5276" if v > 0 else "#C0392B" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.2f} كجم<extra></extra>"))
    fig.update_layout(
        title="تفسير التنبؤ — تأثير كل ميزة (كجم)",
        height=max(280, top_n * 32),
        xaxis_title="التأثير على التنبؤ (كجم)",
        xaxis=dict(zeroline=True, zerolinecolor="#ccc", zerolinewidth=1.5))
    return fig


def generate_xai_narrative(shap_dict: dict, feat_desc: dict, prediction: float,
                             season_label: str, product_ar: str,
                             api_key: str = "", model_name: str = "gemini-flash-lite-latest",
                             lang: str = "ar") -> str:
    """
    Convert SHAP values into a plain-language explanation.
    Uses Gemini if an API key is available; falls back to a template.
    """
    top3 = list(shap_dict.items())[:3]
    top3_str = "\n".join([
        f"- {feat_desc.get(f, f)}: {v:+.1f} كجم" for f, v in top3])

    # Template fallback (always works, no API needed)
    pos = [(feat_desc.get(f, f), v) for f, v in top3 if v > 0]
    neg = [(feat_desc.get(f, f), v) for f, v in top3 if v < 0]
    pos_str = " و ".join([f"**{n}** (+{v:.1f} كجم)" for n, v in pos[:2]])
    neg_str = " و ".join([f"**{n}** ({v:.1f} كجم)" for n, v in neg[:1]])
    template_ar = (
        f"🔍 **تفسير التنبؤ:**\n\n"
        f"النموذج يتوقع **{prediction:.1f} كجم** من {product_ar} خلال {season_label}.\n\n"
        f"**العوامل الرافعة للطلب:** {pos_str if pos_str else 'لا يوجد'}\n\n"
        f"{'**العوامل الخافضة:** ' + neg_str if neg_str else ''}\n\n"
        f"هذا التنبؤ يساعدك على تحديد كمية الشراء المثلى مسبقاً.")

    if not (api_key or "") and not HAS_GEMINI:
        return template_ar

    effective_key = (api_key or "").strip()
    if not effective_key or not HAS_GEMINI:
        return template_ar

    try:
        genai.configure(api_key=effective_key)
        m = genai.GenerativeModel(model_name)
        lang_inst = "in Arabic" if lang == "ar" else "in English"
        prompt = f"""You are an AI assistant for an Egyptian butcher shop owner.

The demand forecast model predicted {prediction:.1f} kg of {product_ar} during {season_label}.

Top contributing features (SHAP values):
{top3_str}

Write a clear, practical 3-sentence explanation {lang_inst} for a non-technical shop owner.
- Mention the 2 most important factors and what they mean practically
- End with a concrete procurement recommendation
- Keep it under 60 words
- Do NOT use technical ML terms"""
        resp = m.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return template_ar


# ─────────────────────────────────────────────────────────────────────────────
# 4. NATURAL LANGUAGE INTENT PARSER  (Q1 + Q2 combined)
# ─────────────────────────────────────────────────────────────────────────────
def parse_query_intent(question: str, api_key: str = "",
                        model_name: str = "gemini-flash-lite-latest") -> dict:
    """
    Use Gemini to extract structured intent from a natural-language question.
    Returns a dict with keys: product_ar, holiday, year, granularity, query_type.
    Falls back to empty values if parsing fails.
    """
    template = {
        "product_ar":   "",
        "holiday":      "",
        "year":         date.today().year + 1,
        "granularity":  "daily",
        "query_type":   "forecast",
    }

    effective_key = (api_key or "").strip()
    if not effective_key or not HAS_GEMINI:
        # Simple rule-based fallback
        q = question.lower()
        if "رمضان" in question:   template["holiday"] = "رمضان"
        if "أضحى" in question:    template["holiday"] = "عيد الأضحى"
        if "فطر"  in question:    template["holiday"] = "عيد الفطر"
        if "أسبوعي" in question or "weekly" in q:  template["granularity"] = "weekly"
        if "شهري"   in question or "monthly" in q: template["granularity"] = "monthly"
        for yr in range(2025, 2035):
            if str(yr) in question: template["year"] = yr; break
        return template

    try:
        genai.configure(api_key=effective_key)
        m = genai.GenerativeModel(model_name)
        prompt = f"""Extract the intent from this butcher shop demand question.
Question: {question}

Return ONLY valid JSON with these exact keys:
- product_ar: Arabic product name (e.g. "مفروم", "لحم بلدي") — empty string if not mentioned
- holiday: one of ["رمضان", "عيد الأضحى", "عيد الفطر", ""] — empty if not a holiday question
- year: Gregorian year as integer (e.g. 2027) — use {date.today().year + 1} if not specified
- granularity: one of ["daily", "weekly", "monthly"] — "daily" by default
- query_type: one of ["forecast", "comparison", "explanation"] — "forecast" by default"""
        resp = m.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        parsed = json.loads(resp.text)
        template.update({k: v for k, v in parsed.items() if k in template})
    except Exception:
        pass
    return template


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEASONAL ANCHOR FORECAST  (for 300+ day horizons)
# ─────────────────────────────────────────────────────────────────────────────
def seasonal_anchor_forecast(conn, pid: int, holiday_ar: str, hijri_year: int,
                              get_mult_fn) -> dict:
    """
    For long-horizon (300+ day) holiday forecasting where recursive error
    accumulation makes step-by-step prediction unreliable.

    Formula:
      expected_daily = recent_7d_avg × seasonal_multiplier × (1 + trend_factor)

    Returns a dict with:
      predicted_daily, predicted_total, multiplier, holiday_window, confidence_note
    """
    try:
        start, end, mult = compute_holiday_window(holiday_ar, hijri_year)
    except ValueError as e:
        return {"error": str(e)}

    df = pd.read_sql_query(
        f"SELECT sale_date, SUM(weight_kg) kg FROM sales "
        f"WHERE product_id={pid} GROUP BY sale_date ORDER BY sale_date DESC LIMIT 60",
        conn)

    if df.empty or len(df) < 7:
        return {"error": "Insufficient historical data (need at least 7 days)"}

    df["kg"] = df["kg"].fillna(0)
    recent_7   = float(df["kg"].head(7).mean())
    recent_30  = float(df["kg"].head(30).mean()) if len(df) >= 30 else recent_7
    trend      = (recent_7 - recent_30) / (recent_30 + 1e-5) if recent_30 > 0 else 0.0

    # Clamp trend to ±30% for long-horizon stability
    trend = max(-0.30, min(0.30, trend))

    expected_daily = recent_7 * mult * (1 + trend)
    duration_days  = (end - start).days + 1
    expected_total = expected_daily * duration_days

    return {
        "holiday":         holiday_ar,
        "hijri_year":      hijri_year,
        "start":           start,
        "end":             end,
        "mult":            mult,
        "recent_7d_avg":   round(recent_7, 1),
        "trend_factor":    round(trend * 100, 1),
        "predicted_daily": round(expected_daily, 1),
        "predicted_total": round(expected_total, 1),
        "duration_days":   duration_days,
        "confidence":      "high" if trend == 0 else ("medium" if abs(trend) < 0.15 else "low"),
        "note": (f"Based on your recent {recent_7:.0f} kg/day average × "
                 f"{mult}× {holiday_ar} multiplier "
                 f"{'with ' + ('+' if trend>=0 else '') + f'{trend*100:.0f}% trend adjustment' if abs(trend)>0.01 else ''}")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def show_ai_assistant(ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                       enterprise_forecast_fn, get_mult_fn, compute_features_fn,
                       conn_fn, prods_fn, current_user: str,
                       gemini_env_key: str = "", gemini_model: str = "gemini-flash-lite-latest"):

    st.title("🤖 المساعد الذكي")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "💬 اسألني عن الطلب المستقبلي لأي عيد، أو احصل على توقع يومي/أسبوعي/شهري "
        "مع تفسير كامل لكل نتيجة."
        "</div>", unsafe_allow_html=True)

    api_key = st.text_input(
        "🔑 Gemini API Key (للأسئلة بالعربية الطبيعية):",
        type="password",
        value=gemini_env_key,
        help="اتركه فارغاً لاستخدام القالب الافتراضي")

    tabs = st.tabs(["🎙️ سؤال حر", "📅 توقع عيد محدد", "📊 تغيير التفاصيل الزمنية"])

    # ── TAB 1: Natural language question ─────────────────────────────────────
    with tabs[0]:
        st.subheader("اسأل بالعربية أو الإنجليزية")
        examples = [
            "كم سيكون الطلب على المفروم في رمضان 2027؟",
            "What will demand be for lamb during Eid Al-Adha 2027?",
            "أعطني توقعاً أسبوعياً للفيليه للشهر القادم",
            "How will Eid Al-Fitr 2027 affect beef demand?",
        ]
        col_ex1, col_ex2 = st.columns(2)
        for i, ex in enumerate(examples):
            (col_ex1 if i % 2 == 0 else col_ex2).caption(f"💡 *{ex}*")

        question = st.text_area("سؤالك:", height=80, placeholder=examples[0])
        p = prods_fn()

        if st.button("🚀 إرسال السؤال", type="primary") and question.strip():
            with st.spinner("جاري تحليل السؤال..."):
                intent = parse_query_intent(question, api_key, gemini_model)

            st.info(f"🎯 فهمت: منتج = **{intent['product_ar'] or 'غير محدد'}** | "
                    f"مناسبة = **{intent['holiday'] or 'غير محدد'}** | "
                    f"سنة = **{intent['year']}** | "
                    f"الدقة = **{intent['granularity']}**")

            # Find matching product
            selected_pid = None
            if intent["product_ar"]:
                match = p[p.name_ar.str.contains(intent["product_ar"], na=False)]
                if not match.empty:
                    selected_pid = int(match.iloc[0].id)
                    product_ar = str(match.iloc[0].name_ar)
                else:
                    st.warning(f"لم أجد منتجاً يطابق '{intent['product_ar']}'. اختر منتجاً:")
                    sel = st.selectbox("المنتج:", p.name_ar.tolist(), key="nl_prod")
                    selected_pid = int(p[p.name_ar == sel].id.values[0])
                    product_ar = sel
            else:
                sel = st.selectbox("حدد المنتج:", p.name_ar.tolist(), key="nl_prod2")
                selected_pid = int(p[p.name_ar == sel].id.values[0])
                product_ar = sel

            if selected_pid is None:
                st.stop()

            # Decide: short-horizon recursive vs long-horizon anchor
            if intent["holiday"]:
                try:
                    h_year = gregorian_year_to_hijri(intent["year"])
                    start_dt, _, _ = compute_holiday_window(intent["holiday"], h_year)
                    days_away = (start_dt - date.today()).days
                except Exception:
                    days_away = 999

                if days_away > 180:
                    # Use anchor method for long horizon
                    with st.spinner(f"جاري حساب توقع {intent['holiday']} {intent['year']}..."):
                        conn = conn_fn()
                        result = seasonal_anchor_forecast(
                            conn, selected_pid, intent["holiday"], h_year, get_mult_fn)
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📅 النطاق الزمني",
                                    f"{result['start'].strftime('%d %b')} – {result['end'].strftime('%d %b %Y')}")
                        col2.metric(f"المتوقع يومياً خلال {intent['holiday']}",
                                    f"{result['predicted_daily']:.0f} كجم/يوم",
                                    delta=f"×{result['mult']} مقارنة بالمعتاد")
                        col3.metric("إجمالي الموسم",
                                    f"{result['predicted_total']:.0f} كجم",
                                    delta=f"مدة {result['duration_days']} أيام")

                        conf_color = {"high": "🟢", "medium": "🟡", "low": "🟠"}
                        st.info(f"{conf_color.get(result['confidence'], '🔵')} **{result['note']}**")

                        # XAI narrative for anchor method
                        narrative = (
                            f"**توقع {intent['holiday']} {intent['year']}:**\n\n"
                            f"بناءً على متوسط طلبك الأخير ({result['recent_7d_avg']:.0f} كجم/يوم) "
                            f"مضروباً في معامل موسم {intent['holiday']} (×{result['mult']:.2f})"
                            f"{', مع تعديل اتجاه ' + ('+' if result['trend_factor']>=0 else '') + str(result['trend_factor']) + '%' if abs(result['trend_factor']) > 1 else ''}, "
                            f"يُقدَّر الطلب بـ **{result['predicted_daily']:.0f} كجم/يوم**.\n\n"
                            f"**التوصية:** احجز {round(result['predicted_total'] * 1.1):.0f} كجم من موردك "
                            f"على الأقل أسبوعين قبل {result['start'].strftime('%d %b %Y')}.")
                        st.markdown(narrative)
                else:
                    # Short horizon — use recursive forecast
                    target_days = max(days_away + 10, 14)
                    with st.spinner("Enterprise LightGBM..."):
                        fc = enterprise_forecast_fn(selected_pid, min(target_days, 60))
                    if not fc:
                        st.error("بيانات غير كافية للتنبؤ.")
                        st.stop()
                    agg = aggregate_forecast(fc, intent["granularity"])
                    hol_info = {"holiday": intent["holiday"],
                                "start": start_dt} if intent["holiday"] else None
                    fig = build_forecast_chart(agg, product_ar, intent["granularity"], hol_info)
                    st.plotly_chart(fig, use_container_width=True)

                    # SHAP for first predicted day
                    _run_shap_explanation(
                        fc, ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                        product_ar, api_key, gemini_model,
                        conn_fn, selected_pid)
            else:
                # General forecast — no specific holiday
                with st.spinner("Enterprise LightGBM..."):
                    fc = enterprise_forecast_fn(selected_pid, 14)
                if not fc:
                    st.error("بيانات غير كافية للتنبؤ.")
                    st.stop()
                agg = aggregate_forecast(fc, intent["granularity"])
                fig = build_forecast_chart(agg, product_ar, intent["granularity"])
                st.plotly_chart(fig, use_container_width=True)
                _run_shap_explanation(
                    fc, ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                    product_ar, api_key, gemini_model,
                    conn_fn, selected_pid)

    # ── TAB 2: Specific holiday forecast ─────────────────────────────────────
    with tabs[1]:
        st.subheader("📅 توقع طلب مناسبة محددة")
        upcoming = get_upcoming_holidays(3)

        col1, col2 = st.columns(2)
        with col1:
            prod_sel = st.selectbox("المنتج:", p.name_ar.tolist(), key="hol_prod")
            pid_hol  = int(p[p.name_ar == prod_sel].id.values[0])
        with col2:
            if upcoming:
                hol_options = [
                    f"{h['holiday']} {h['hijri_year']} ({h['greg_range']}) — بعد {h['days_away']} يوم"
                    for h in upcoming]
                hol_sel_idx = st.selectbox("اختر المناسبة:", range(len(hol_options)),
                                           format_func=lambda i: hol_options[i], key="hol_sel")
                selected_hol = upcoming[hol_sel_idx]
            else:
                st.warning("hijri-converter غير مثبت. شغل: pip install hijri-converter")
                selected_hol = None

        if selected_hol and st.button("🎯 احسب توقع المناسبة", type="primary", key="hol_btn"):
            days_away = selected_hol["days_away"]

            if days_away > 180:
                with st.spinner("Seasonal anchor method..."):
                    result = seasonal_anchor_forecast(
                        conn_fn(), pid_hol,
                        selected_hol["holiday"], selected_hol["hijri_year"], get_mult_fn)
                if "error" not in result:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("التاريخ المتوقع", selected_hol.get("greg_range") or (str(result["start"]) + " - " + str(result["end"])))
                    c2.metric("الطلب اليومي المتوقع", f"{result['predicted_daily']:.0f} كجم/يوم",
                              delta=f"×{result['mult']} الأيام العادية")
                    c3.metric("إجمالي الموسم", f"{result['predicted_total']:.0f} كجم")
                    st.info(result["note"])
            else:
                target = min(days_away + 14, 60)
                with st.spinner("Enterprise LightGBM..."):
                    fc = enterprise_forecast_fn(pid_hol, target)
                if fc:
                    agg = aggregate_forecast(fc, "daily")
                    fig = build_forecast_chart(
                        agg, prod_sel, "daily",
                        {"holiday": selected_hol["holiday"], "start": selected_hol["start"]})
                    st.plotly_chart(fig, use_container_width=True)
                    _run_shap_explanation(
                        fc, ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                        prod_sel, api_key, gemini_model, conn_fn, pid_hol)

    # ── TAB 3: Granularity selector ───────────────────────────────────────────
    with tabs[2]:
        st.subheader("📊 اختر الدقة الزمنية للتوقع")
        col1, col2, col3 = st.columns(3)
        with col1:
            gran_prod = st.selectbox("المنتج:", p.name_ar.tolist(), key="gran_prod")
            gran_pid  = int(p[p.name_ar == gran_prod].id.values[0])
        with col2:
            granularity = st.radio(
                "الدقة:", ["daily", "weekly", "monthly"],
                format_func=lambda x: {"daily": "يومي (أعلى دقة)", "weekly": "أسبوعي", "monthly": "شهري"}[x],
                key="gran_radio")
        with col3:
            forecast_days = st.slider("أيام التوقع:", 7, 90,
                                       {"daily": 14, "weekly": 42, "monthly": 90}[granularity],
                                       key="gran_days")

        # Accuracy preview
        daily_mae = 16.28
        if granularity == "weekly":
            est_mae = daily_mae * (7 ** 0.5)
            est_vol = daily_mae * 7 * 8  # rough typical
            note = f"الخطأ المتوقع أسبوعياً: ≈{est_mae:.0f} كجم/أسبوع (أقل نسبةً من اليومي)"
        elif granularity == "monthly":
            est_mae = daily_mae * (30 ** 0.5)
            note = f"الخطأ المتوقع شهرياً: ≈{est_mae:.0f} كجم/شهر (أفضل لتخطيط الميزانية)"
        else:
            note = f"الخطأ اليومي: ≈{daily_mae:.1f} كجم (أفضل للقرارات التشغيلية اليومية)"
        st.info(f"📏 {note}")

        if st.button("🚀 توليد التوقع", type="primary", key="gran_btn"):
            with st.spinner("Enterprise LightGBM..."):
                fc = enterprise_forecast_fn(gran_pid, forecast_days)
            if not fc:
                st.error("بيانات غير كافية.")
            else:
                agg = aggregate_forecast(fc, granularity)
                fig = build_forecast_chart(agg, gran_prod, granularity)
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                if not agg.empty:
                    st.subheader("📋 ملخص التوقعات")
                    display = agg[["date", "predicted", "lower", "upper", "pct_err"]].copy()
                    display["date"] = display["date"].dt.strftime(
                        "%Y-%m-%d" if granularity == "daily" else
                        "%Y-W%W"   if granularity == "weekly" else "%b %Y")
                    display.columns = ["الفترة", "المتوقع كجم", "الحد الأدنى", "الحد الأعلى", "الخطأ %"]
                    st.dataframe(display.round(1), use_container_width=True, hide_index=True)

                    total_pred = agg["predicted"].sum()
                    avg_err    = agg["pct_err"].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"إجمالي التوقع ({len(agg)} {'أيام' if granularity=='daily' else 'أسابيع' if granularity=='weekly' else 'أشهر'})",
                              f"{total_pred:,.0f} كجم")
                    c2.metric("متوسط الخطأ %", f"{avg_err:.1f}%")
                    c3.metric("الدقة النسبية", f"{100 - avg_err:.1f}%")

                _run_shap_explanation(
                    fc, ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                    gran_prod, api_key, gemini_model, conn_fn, gran_pid)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SHAP + narrative block (reused across all tabs)
# ─────────────────────────────────────────────────────────────────────────────
def _run_shap_explanation(fc, ENT_MODEL, ENT_FEATURES, ENT_FEAT_DESC,
                           product_ar, api_key, gemini_model, conn_fn, pid):
    """Run SHAP explanation for the first forecast day and display it."""
    if not fc or ENT_MODEL is None or ENT_FEATURES is None:
        return

    first = fc[0]
    try:
        import sqlite3
        conn = conn_fn()
        df_h = pd.read_sql_query(
            f"SELECT sale_date, SUM(weight_kg) kg, AVG(unit_price) price, "
            f"AVG(unit_cost) cost FROM sales WHERE product_id={pid} "
            f"GROUP BY sale_date ORDER BY sale_date", conn)
        if df_h.empty or len(df_h) < 14:
            return
        df_h["sale_date"] = pd.to_datetime(df_h["sale_date"], format="mixed", errors="coerce").dt.date
        kg_v = df_h["kg"].values.tolist()
        pr_v = df_h["price"].values.tolist()
        co_v = df_h["cost"].values.tolist()
        inv_r = conn.execute(
            "SELECT current_stock_kg FROM inventory WHERE product_id=?", (pid,)).fetchone()
        inv_s = float(inv_r[0]) if inv_r else 0.0
        fd1 = df_h["sale_date"].iloc[-1] + timedelta(days=1)

        # Compute features
        feat_vals = _compute_features_inline(kg_v, pr_v, co_v, inv_s, fd1, ENT_FEATURES)
        shap_d = compute_shap_for_day(ENT_MODEL, ENT_FEATURES, feat_vals)

        with st.expander("🔬 تفسير التنبؤ — Explainable AI"):
            col_sh, col_txt = st.columns([1, 1])
            with col_sh:
                if shap_d:
                    fig_sh = build_shap_chart(shap_d, ENT_FEAT_DESC)
                    st.plotly_chart(fig_sh, use_container_width=True)
                else:
                    st.info("💡 شغل: pip install shap لعرض المخطط")
            with col_txt:
                narrative = generate_xai_narrative(
                    shap_d or {}, ENT_FEAT_DESC,
                    first["predicted"], first["season"], product_ar,
                    api_key, gemini_model, lang="ar")
                st.markdown(narrative)
    except Exception as e:
        st.caption(f"XAI غير متاح: {e}")


def _compute_features_inline(kg_vals, price_vals, cost_vals, inv_stock, step_date, ENT_FEATURES):
    """Inline version of _compute_features to avoid circular import."""
    tail  = kg_vals[-max(14, 1):]
    l1  = float(tail[-1])  if len(tail)>=1  else 0.0
    l2  = float(tail[-2])  if len(tail)>=2  else l1
    l3  = float(tail[-3])  if len(tail)>=3  else l2
    l4  = float(tail[-4])  if len(tail)>=4  else l3
    l7  = float(tail[-7])  if len(tail)>=7  else l4
    l14 = float(tail[-14]) if len(tail)>=14 else l7
    r3  = float(np.mean(tail[-3:]))  if len(tail)>=3  else l1
    r6  = float(np.mean(tail[-6:]))  if len(tail)>=6  else r3
    r14 = float(np.mean(tail[-14:])) if len(tail)>=14 else r6
    std7= float(np.std(tail[-7:]))   if len(tail)>=7  else 0.0
    da  = r6 if r6>0 else 1.0
    p_now  = price_vals[-1] if price_vals else 100.0
    p_wk   = float(np.mean(price_vals[-7:])) if len(price_vals)>=7 else p_now
    p_m30  = float(np.mean(price_vals[-30:])) if len(price_vals)>=30 else p_now
    c_now  = cost_vals[-1] if cost_vals else 80.0
    c_wk   = float(np.mean(cost_vals[-7:])) if len(cost_vals)>=7 else c_now
    la     = float(np.mean(tail[-30:])) if len(tail)>=30 else r6
    doy    = step_date.timetuple().tm_yday
    inv_t  = (r3-r6)/(r6+1e-5)

    full_map = {
        "lag_1": l1, "lag_2": l2, "lag_3": l3, "lag_4": l4, "lag_7": l7, "lag_14": l14,
        "rolling_3": r3, "rolling_6": r6, "rolling_14": r14, "rolling_std_7": std7,
        "inventory_pressure": inv_stock/da, "inventory_trend": inv_t,
        "price_volatility": float(np.std(price_vals[-7:])) if len(price_vals)>=7 else 0.0,
        "price_trend": (p_now-p_wk)/(p_wk+1e-5),
        "price_level": p_now/(p_m30+1e-5),
        "procurement_cost_trend": (c_now-c_wk)/(c_wk+1e-5),
        "cost_margin_ratio": (p_now-c_now)/(p_now+1e-5),
        "supply_gap": inv_stock - da*3,
        "demand_elasticity": 0.0,
        "stock_cover_days": inv_stock/da,
        "trend_strength": (r6-la)/(la+1e-5),
        "trend_ratio": r3/(r6+1e-5),
        "week_sin": float(np.sin(2*np.pi*doy/7)),
        "week_cos": float(np.cos(2*np.pi*doy/7)),
        "month_sin": float(np.sin(2*np.pi*step_date.month/12)),
        "month_cos": float(np.cos(2*np.pi*step_date.month/12)),
        "is_ramadan": 0.0, "is_weekend": float(1 if step_date.weekday() in {4,5,6} else 0),
        "days_to_next_holiday": 30.0,
    }
    return [full_map.get(f, 0.0) for f in ENT_FEATURES]
