"""
ButcherAI — Price Forecasting Dashboard (Enhanced)
====================================================
Features:
- Dynamic dashboard with per-period analysis
- Alert levels (NORMAL / HIGH / CRITICAL) per period
- Explainable AI: Gemini-powered reasoning for each period (with heuristic fallback)
- Detailed analytics: price change %, trend, season, reliability, recommendation
- Interactive Plotly chart with period markers and annotations
- Holiday-aware projections for Ramadan / Eid (any year)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HOLIDAY CALENDAR (2026-2030)
# ─────────────────────────────────────────────────────────────────────────────
ISLAMIC_HOLIDAYS = {
    (9, 1): ("رمضان", "Ramadan", 1.55),
    (10, 1): ("عيد الفطر", "Eid Al-Fitr", 2.20),
    (12, 10): ("عيد الأضحى", "Eid Al-Adha", 3.00),
}

UPCOMING_HOLIDAYS = [
    {"name_ar": "رمضان 2026", "name_en": "Ramadan 2026", "date": date(2026, 2, 18), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2026", "name_en": "Eid Al-Fitr 2026", "date": date(2026, 3, 20), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2026", "name_en": "Eid Al-Adha 2026", "date": date(2026, 5, 27), "multiplier": 3.00},
    {"name_ar": "رمضان 2027", "name_en": "Ramadan 2027", "date": date(2027, 2, 7), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2027", "name_en": "Eid Al-Fitr 2027", "date": date(2027, 3, 10), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2027", "name_en": "Eid Al-Adha 2027", "date": date(2027, 5, 16), "multiplier": 3.00},
    {"name_ar": "رمضان 2028", "name_en": "Ramadan 2028", "date": date(2028, 1, 26), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2028", "name_en": "Eid Al-Fitr 2028", "date": date(2028, 2, 15), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2028", "name_en": "Eid Al-Adha 2028", "date": date(2028, 5, 5), "multiplier": 3.00},
    {"name_ar": "رمضان 2029", "name_en": "Ramadan 2029", "date": date(2029, 1, 15), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2029", "name_en": "Eid Al-Fitr 2029", "date": date(2029, 2, 4), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2029", "name_en": "Eid Al-Adha 2029", "date": date(2029, 4, 24), "multiplier": 3.00},
    {"name_ar": "رمضان 2030", "name_en": "Ramadan 2030", "date": date(2030, 1, 4), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2030", "name_en": "Eid Al-Fitr 2030", "date": date(2030, 1, 24), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2030", "name_en": "Eid Al-Adha 2030", "date": date(2030, 4, 13), "multiplier": 3.00},
]

WEEKDAY_MULTIPLIERS = {
    4: (1.30, "الجمعة"),
    5: (1.15, "السبت"),
    6: (1.10, "الأحد"),
    0: (0.90, "الإثنين"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PeriodAnalysis:
    label: str
    date: date
    horizon_days: int
    projected_price: float
    adjusted_price: float
    price_change_pct: float
    alert: str          # NORMAL / HIGH / CRITICAL
    alert_reason: str
    trend: str          # rising / falling / stable
    season: str
    reliability: str
    recommendation: str
    risk: str


@dataclass
class PriceForecast:
    product_id: int
    product_ar: str
    product_en: str
    category: str
    current_price: float
    trend_per_day: float
    r_squared: float
    data_points: int
    last_date: date
    last_price: float
    periods: List[PeriodAnalysis] = field(default_factory=list)
    overall_alert: str = "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# CORE FORECASTING + ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def _classify_alert(price_change_pct: float, season: str, reliability: str) -> Tuple[str, str]:
    """Classify a period into NORMAL / HIGH / CRITICAL."""
    if season in ("رمضان", "عيد الفطر", "عيد الأضحى"):
        if price_change_pct > 10:
            return "CRITICAL", f"Major holiday ({season}) + sharp price rise"
        return "HIGH", f"Major holiday demand surge ({season})"
    if abs(price_change_pct) > 15:
        return "CRITICAL", f"Price volatility {price_change_pct:+.1f}%"
    if abs(price_change_pct) > 5:
        return "HIGH", f"Price movement {price_change_pct:+.1f}%"
    if reliability == "low":
        return "HIGH", "Low forecast confidence — verify before acting"
    return "NORMAL", "Stable price expectation"


def _season_label(d: date) -> str:
    """Return the season/period label for a date."""
    for h in UPCOMING_HOLIDAYS:
        if abs((d - h["date"]).days) <= 3:
            return h["name_ar"]
    return WEEKDAY_MULTIPLIERS.get(d.weekday(), (1.0, "يوم عادي"))[1]


def _recommendation(alert: str, trend: str, price_change_pct: float, season: str) -> str:
    """Actionable recommendation for the period."""
    if alert == "CRITICAL":
        if price_change_pct > 0:
            return "Lock in supply now — prices expected to spike. Buy ahead."
        return "Hold purchases — prices may drop. Wait and monitor."
    if alert == "HIGH":
        if price_change_pct > 0:
            return "Consider forward buying 5-10% to hedge against rise."
        return "Reduce order size — prices trending down."
    if trend == "rising":
        return "Gradual price rise — plan purchases early in the period."
    if trend == "falling":
        return "Prices stable-to-falling — buy only what you need."
    return "Stable outlook — follow your normal procurement rhythm."


def _risk_note(alert: str, r_squared: float) -> str:
    if alert == "CRITICAL":
        return "HIGH RISK: Decision-sensitive. Verify with supplier quotes."
    if r_squared < 0.4:
        return "MEDIUM RISK: Weak trend fit — treat projection as indicative only."
    return "LOW RISK: Stable trend supports confident planning."


def forecast_price_for_product(
    db_path: str,
    product_id: int,
    target_dates: List[date] = None,
    lookback_days: int = 90,
    start_date: date = None,
) -> Optional[PriceForecast]:
    """Fit a linear price trend and analyse each target period."""
    conn = sqlite3.connect(db_path)

    prod_row = conn.execute(
        "SELECT name_ar, name_en, category, avg_price FROM products WHERE id=?", (product_id,)
    ).fetchone()
    if not prod_row:
        conn.close()
        return None

    if start_date:
        date_filter = f"sale_date >= '{start_date}' AND sale_date <= date('now')"
    else:
        date_filter = f"sale_date >= date('now', '-{lookback_days} days')"

    price_history = pd.read_sql_query(
        f"""SELECT sale_date, AVG(unit_price) avg_price FROM sales
            WHERE product_id={product_id} AND {date_filter}
            GROUP BY sale_date ORDER BY sale_date""",
        conn,
    )
    conn.close()

    if len(price_history) < 3:
        return None

    prices = price_history["avg_price"].values
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices, 1)
    predicted = intercept + slope * x
    ss_res = np.sum((prices - predicted) ** 2)
    ss_tot = np.sum((prices - prices.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    last_date = datetime.strptime(price_history["sale_date"].iloc[-1], "%Y-%m-%d").date()
    last_price = float(prices[-1])
    last_x = len(prices) - 1

    if not target_dates:
        target_dates = [last_date + timedelta(days=h) for h in [7, 14, 30, 60, 90]]

    periods: List[PeriodAnalysis] = []
    for target_date in sorted(set(target_dates)):
        horizon_days = (target_date - last_date).days
        if horizon_days <= 0:
            continue

        projected = max(0.0, intercept + slope * (last_x + horizon_days))

        # Holiday adjustment
        season = _season_label(target_date)
        holiday_mult = 1.0
        for h in UPCOMING_HOLIDAYS:
            if h["name_ar"] == season:
                holiday_mult = h["multiplier"]
                break
        adjusted = projected * (1 + (holiday_mult - 1) * 0.3) if holiday_mult > 1 else projected

        price_change_pct = (adjusted - last_price) / last_price * 100 if last_price else 0
        alert, alert_reason = _classify_alert(price_change_pct, season, _reliability(r_squared, horizon_days))

        if slope > 0.05:
            trend = "rising"
        elif slope < -0.05:
            trend = "falling"
        else:
            trend = "stable"

        periods.append(PeriodAnalysis(
            label=f"{season} ({target_date.strftime('%Y-%m-%d')})" if season != "يوم عادي" else target_date.strftime("%Y-%m-%d"),
            date=target_date,
            horizon_days=horizon_days,
            projected_price=round(projected, 1),
            adjusted_price=round(adjusted, 1),
            price_change_pct=round(price_change_pct, 1),
            alert=alert,
            alert_reason=alert_reason,
            trend=trend,
            season=season,
            reliability=_reliability(r_squared, horizon_days),
            recommendation=_recommendation(alert, trend, price_change_pct, season),
            risk=_risk_note(alert, r_squared),
        ))

    overall = "NORMAL"
    if any(p.alert == "CRITICAL" for p in periods):
        overall = "CRITICAL"
    elif any(p.alert == "HIGH" for p in periods):
        overall = "HIGH"

    return PriceForecast(
        product_id=product_id,
        product_ar=prod_row[0], product_en=prod_row[1], category=prod_row[2],
        current_price=round(last_price, 1),
        trend_per_day=round(float(slope), 4),
        r_squared=round(float(r_squared), 3),
        data_points=len(prices),
        last_date=last_date,
        last_price=round(last_price, 1),
        periods=periods,
        overall_alert=overall,
    )


def _reliability(r_squared: float, horizon_days: int) -> str:
    if horizon_days > 60 or r_squared < 0.15:
        return "low"
    if horizon_days > 30 or r_squared < 0.4:
        return "medium"
    return "high"


def generate_ai_explanation(forecast: PriceForecast, period: PeriodAnalysis, api_key: str = "") -> str:
    """Generate an AI explanation for a specific period using Gemini."""
    if not api_key:
        return _heuristic_explanation(forecast, period)

    try:
        from gemini_client import generate_text
        result = generate_text(
            prompt=(
                f"You are a pricing analyst for an Egyptian butcher shop. Explain this price forecast "
                f"for the product '{forecast.product_ar}' ({forecast.product_en}) in 3-4 sentences, "
                f"in Arabic, in a clear and practical way for a shop owner.\n\n"
                f"- Current price: {forecast.current_price} EGP/kg (as of {forecast.last_date})\n"
                f"- Period: {period.label}\n"
                f"- Projected price: {period.adjusted_price} EGP/kg\n"
                f"- Price change: {period.price_change_pct:+.1f}%\n"
                f"- Alert level: {period.alert}\n"
                f"- Trend: {period.trend} ({forecast.trend_per_day:+.4f} EGP/day)\n"
                f"- Season: {period.season}\n"
                f"- Forecast reliability: {period.reliability} (R2={forecast.r_squared})\n"
                f"- Recommendation: {period.recommendation}\n\n"
                f"Explain WHY the price is expected to move this way (seasonality, trend, market), "
                f"and what the shop owner should DO."
            ),
            api_key=api_key,
            feature="price_explain",
            temperature=0.3,
        )
        if result.get("error"):
            return _heuristic_explanation(forecast, period)
        return result["text"]
    except Exception:
        return _heuristic_explanation(forecast, period)


def _heuristic_explanation(forecast: PriceForecast, period: PeriodAnalysis) -> str:
    """Rule-based explanation when no API key is available."""
    parts = []
    parts.append(
        f"السعر الحالي للمنتج '{forecast.product_ar}' هو {forecast.current_price} ج/كجم "
        f"(آخر بيع في {forecast.last_date})."
    )
    if period.trend == "rising":
        parts.append(f"الاتجاه العام صاعد بمقدار {forecast.trend_per_day:+.4f} ج/كجم يومياً، أي أن السعر يرتفع تدريجياً.")
    elif period.trend == "falling":
        parts.append(f"الاتجاه العام هابط بمقدار {abs(forecast.trend_per_day):.4f} ج/كجم يومياً.")
    else:
        parts.append("الاتجاه العام مستقر نسبياً.")

    if period.season not in ("يوم عادي",):
        parts.append(f"الفترة تشمل '{period.season}' وهو موسم طلب مرتفع، مما يرفع السعر المتوقع.")
    else:
        parts.append("الفترة لا تشمل موسماً رئيسياً، فالطلب ضمن المعدل الطبيعي.")

    parts.append(
        f"السعر المتوقع في هذه الفترة هو {period.adjusted_price} ج/كجم "
        f"({period.price_change_pct:+.1f}%)، ومستوى التنبيه {period.alert}."
    )
    parts.append(f"التوصية: {period.recommendation}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER QUERIES
# ─────────────────────────────────────────────────────────────────────────────
def get_all_categories(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_all_products(db_path: str, category: str = None) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    if category:
        rows = conn.execute(
            "SELECT id, name_ar, name_en, category, avg_price FROM products WHERE category=? ORDER BY name_ar",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name_ar, name_en, category, avg_price FROM products ORDER BY name_ar"
        ).fetchall()
    conn.close()
    return [{"id": r[0], "name_ar": r[1], "name_en": r[2], "category": r[3], "avg_price": r[4]} for r in rows]


def forecast_all_product_prices(db_path, target_dates=None, lookback_days=90, start_date=None, category_filter=None):
    conn = sqlite3.connect(db_path)
    if category_filter:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM products WHERE category=? ORDER BY name_ar", (category_filter,)).fetchall()]
    else:
        ids = [r[0] for r in conn.execute("SELECT id FROM products ORDER BY name_ar").fetchall()]
    conn.close()
    results = []
    for pid in ids:
        fc = forecast_price_for_product(db_path, pid, target_dates, lookback_days, start_date)
        if fc is not None:
            results.append(fc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
ALERT_COLORS = {"NORMAL": "#2ecc71", "HIGH": "#f1c40f", "CRITICAL": "#e74c3c"}


def show_price_forecast_page(db_path: str, gemini_api_key: str = ""):
    import streamlit as st
    import plotly.graph_objects as go

    st.title("Price Forecast Dashboard — توقع الأسعار")

    # ── Product / Category ────────────────────────────────────────────────────
    cats = get_all_categories(db_path)
    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox("الفئة:", ["الكل"] + cats)
    with c2:
        prods = get_all_products(db_path, None if cat == "الكل" else cat)
        names = [f"{p['name_ar']} ({p['name_en']})" for p in prods]
        sel_name = st.selectbox("المنتج:", names) if names else ""
        prod = next((p for p in prods if f"{p['name_ar']} ({p['name_en']})" == sel_name), None)

    if not prod:
        st.info("اختر منتجاً لعرض التوقعات.")
        return

    # ── Period range ──────────────────────────────────────────────────────────
    lookback = st.slider("فترة البيانات التاريخية (يوم):", 30, 365, 90)
    weeks_ahead = st.slider("عدد الأسابيع للتوقع:", 2, 26, 8)

    # ── Build target dates ────────────────────────────────────────────────────
    today = date.today()
    target_dates = []
    for w in range(1, int(weeks_ahead) + 1):
        target_dates.append(today + timedelta(weeks=w))

    if st.button("عرض لوحة التوقعات", type="primary"):
        with st.spinner("جاري تحليل الأسعار..."):
            fc = forecast_price_for_product(db_path, prod["id"], target_dates=target_dates, lookback_days=lookback)

        if fc is None:
            st.warning("لا توجد بيانات مبيعات كافية لهذا المنتج لحساب الاتجاه.")
            return

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1: DASHBOARD METRICS
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("المؤشرات الرئيسية")
        max_p = max(fc.periods, key=lambda p: p.adjusted_price) if fc.periods else None
        min_p = min(fc.periods, key=lambda p: p.adjusted_price) if fc.periods else None
        avg_change = np.mean([p.price_change_pct for p in fc.periods]) if fc.periods else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("السعر الحالي", f"{fc.current_price:,.0f} ج")
        m2.metric("الاتجاه اليومي", f"{fc.trend_per_day:+.3f} ج")
        m3.metric("أعلى سعر متوقع", f"{max_p.adjusted_price:,.0f} ج" if max_p else "-")
        m4.metric("أقل سعر متوقع", f"{min_p.adjusted_price:,.0f} ج" if min_p else "-")
        m5.metric("متوسط التغير", f"{avg_change:+.1f}%")

        # Overall alert banner
        alert_color = ALERT_COLORS.get(fc.overall_alert, "#95a5a6")
        st.markdown(
            f"<div style='background:{alert_color};color:#fff;border-radius:8px;padding:14px;"
            f"font-weight:bold;font-size:16px;text-align:center;margin:10px 0'>"
            f"مستوى التنبيه الإجمالي: {fc.overall_alert}</div>",
            unsafe_allow_html=True,
        )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2: PERIOD ALERTS
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("تنبيهات الفترات")
        for p in fc.periods:
            color = ALERT_COLORS.get(p.alert, "#95a5a6")
            icon = {"NORMAL": "NORMAL", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}.get(p.alert, "")
            st.markdown(
                f"<div style='background:#fafafa;border-right:5px solid {color};border-radius:6px;"
                f"padding:10px 14px;margin:6px 0'>"
                f"<b>{icon} — {p.season}</b> | {p.date.strftime('%Y-%m-%d')} | "
                f"{p.adjusted_price:,.0f} ج ({p.price_change_pct:+.1f}%)<br>"
                f"<span style='color:#666;font-size:13px'>{p.alert_reason}</span></div>",
                unsafe_allow_html=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3: INTERACTIVE CHART
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("الرسم البياني التفاعلي")
        fig = go.Figure()

        # Historical
        conn = sqlite3.connect(db_path)
        hist = pd.read_sql_query(
            f"""SELECT sale_date, AVG(unit_price) avg_price FROM sales
                WHERE product_id={fc.product_id} AND sale_date >= date('now','-{lookback} days')
                GROUP BY sale_date ORDER BY sale_date""",
            conn,
        )
        conn.close()
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["sale_date"], y=hist["avg_price"],
                mode="lines+markers", name="الأسعار التاريخية",
                line=dict(color="#1f77b4", width=2),
            ))

        # Projected
        proj_x = [fc.last_date.strftime("%Y-%m-%d")]
        proj_y = [fc.last_price]
        for p in sorted(fc.periods, key=lambda x: x.date):
            proj_x.append(p.date.strftime("%Y-%m-%d"))
            proj_y.append(p.adjusted_price)
        fig.add_trace(go.Scatter(
            x=proj_x, y=proj_y, mode="lines+markers", name="السعر المتوقع",
            line=dict(color="#8B1A1A", width=3, dash="dash"),
            marker=dict(size=8),
        ))

        # Annotations for high/critical periods
        for p in fc.periods:
            if p.alert in ("HIGH", "CRITICAL"):
                fig.add_annotation(
                    x=p.date.strftime("%Y-%m-%d"), y=p.adjusted_price,
                    text=p.season, showarrow=True, arrowhead=2,
                    font=dict(size=10, color=ALERT_COLORS[p.alert]),
                )

        fig.update_layout(
            title=f"توقع سعر {fc.product_ar}",
            xaxis_title="التاريخ", yaxis_title="السعر (ج/كجم)",
            height=420, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4: DETAILED PERIOD ANALYTICS
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("التحليل التفصيلي لكل فترة")
        rows = []
        for p in sorted(fc.periods, key=lambda x: x.date):
            rows.append({
                "الفترة": p.season,
                "التاريخ": p.date.strftime("%Y-%m-%d"),
                "السعر المتوقع (ج)": p.adjusted_price,
                "التغير %": f"{p.price_change_pct:+.1f}%",
                "الاتجاه": p.trend,
                "التنبيه": p.alert,
                "الموثوقية": p.reliability,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5: AI EXPLANATION PER PERIOD
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("التفسير الذكي لكل فترة")
        period_labels = [f"{p.season} ({p.date.strftime('%Y-%m-%d')})" for p in sorted(fc.periods, key=lambda x: x.date)]
        sel_period = st.selectbox("اختر الفترة لعرض التحليل:", period_labels)
        sel_p = next((p for p in fc.periods if f"{p.season} ({p.date.strftime('%Y-%m-%d')})" == sel_period), None)

        if sel_p:
            st.markdown(
                f"<div style='background:#f8f9fa;border:2px solid #8B1A1A;border-radius:8px;"
                f"padding:14px 18px;margin:8px 0'>"
                f"<b>الفترة:</b> {sel_p.season} — {sel_p.date.strftime('%Y-%m-%d')}<br>"
                f"<b>السعر المتوقع:</b> {sel_p.adjusted_price:,.0f} ج/كجم "
                f"({sel_p.price_change_pct:+.1f}%)<br>"
                f"<b>التنبيه:</b> {sel_p.alert} — {sel_p.alert_reason}<br>"
                f"<b>المخاطر:</b> {sel_p.risk}<br>"
                f"<b>التوصية:</b> {sel_p.recommendation}</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("جاري توليد التفسير الذكي..."):
                explanation = generate_ai_explanation(fc, sel_p, gemini_api_key)
            st.markdown(
                f"<div style='background:#e8f4fd;border-right:5px solid #1A5276;border-radius:8px;"
                f"padding:14px 18px;margin:8px 0;line-height:1.8'>{explanation}</div>",
                unsafe_allow_html=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 6: ALL PRODUCTS COMPARISON
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        with st.expander("مقارنة كل المنتجات"):
            all_fc = forecast_all_product_prices(
                db_path, target_dates=target_dates, lookback_days=lookback,
                category_filter=None if cat == "الكل" else cat,
            )
            comp_rows = []
            for f in all_fc:
                worst = max(f.periods, key=lambda p: p.price_change_pct) if f.periods else None
                comp_rows.append({
                    "المنتج": f.product_ar,
                    "السعر الحالي": f.current_price,
                    "أعلى تغير %": f"{worst.price_change_pct:+.1f}%" if worst else "-",
                    "التنبيه الأعلى": f.overall_alert,
                })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            else:
                st.info("لا توجد بيانات كافية لأي منتج.")
