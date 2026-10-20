"""
ButcherAI — Price Forecasting (Enhanced)
==========================================
Features:
- Forecast to specific dates (Ramadan, Eid Fitr, Eid Adha for any year)
- Select any week's data for analysis
- Select from all products with all categories/types
- Linear trend projection with reliability labels
- Holiday-aware demand multipliers
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HOLIDAR CALENDAR — Upcoming Islamic holidays (2026-2030)
# ─────────────────────────────────────────────────────────────────────────────
ISLAMIC_HOLIDAYS = {
    # (hijri_month, hijri_day): (name_ar, name_en, demand_multiplier)
    (9, 1): ("رمضان", "Ramadan", 1.55),
    (10, 1): ("عيد الفطر", "Eid Al-Fitr", 2.20),
    (12, 10): ("عيد الأضحى", "Eid Al-Adha", 3.00),
}

# Approximate Gregorian dates for upcoming years (can be refined with hijri_converter)
UPCOMING_HOLIDAYS = [
    # 2026
    {"name_ar": "رمضان 2026", "name_en": "Ramadan 2026", "date": date(2026, 2, 18), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2026", "name_en": "Eid Al-Fitr 2026", "date": date(2026, 3, 20), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2026", "name_en": "Eid Al-Adha 2026", "date": date(2026, 5, 27), "multiplier": 3.00},
    # 2027
    {"name_ar": "رمضان 2027", "name_en": "Ramadan 2027", "date": date(2027, 2, 7), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2027", "name_en": "Eid Al-Fitr 2027", "date": date(2027, 3, 10), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2027", "name_en": "Eid Al-Adha 2027", "date": date(2027, 5, 16), "multiplier": 3.00},
    # 2028
    {"name_ar": "رمضان 2028", "name_en": "Ramadan 2028", "date": date(2028, 1, 26), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2028", "name_en": "Eid Al-Fitr 2028", "date": date(2028, 2, 15), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2028", "name_en": "Eid Al-Adha 2028", "date": date(2028, 5, 5), "multiplier": 3.00},
    # 2029
    {"name_ar": "رمضان 2029", "name_en": "Ramadan 2029", "date": date(2029, 1, 15), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2029", "name_en": "Eid Al-Fitr 2029", "date": date(2029, 2, 4), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2029", "name_en": "Eid Al-Adha 2029", "date": date(2029, 4, 24), "multiplier": 3.00},
    # 2030
    {"name_ar": "رمضان 2030", "name_en": "Ramadan 2030", "date": date(2030, 1, 4), "multiplier": 1.55},
    {"name_ar": "عيد الفطر 2030", "name_en": "Eid Al-Fitr 2030", "date": date(2030, 1, 24), "multiplier": 2.20},
    {"name_ar": "عيد الأضحى 2030", "name_en": "Eid Al-Adha 2030", "date": date(2030, 4, 13), "multiplier": 3.00},
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
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
    projections: Dict[str, float] = field(default_factory=dict)  # {label: projected_price}
    reliability_by_horizon: Dict[str, str] = field(default_factory=dict)
    reliability_reasons: Dict[str, str] = field(default_factory=dict)
    holiday_adjustments: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY LABEL
# ─────────────────────────────────────────────────────────────────────────────
def _reliability_label(r_squared: float, data_points: int, horizon_days: int) -> tuple:
    """Honest reliability assessment based on R-squared, data points, and horizon."""
    if data_points < 14:
        return "low", f"Only {data_points} days of price history"
    if horizon_days > 60:
        return "low", "Beyond ~60-day horizon"
    if r_squared < 0.15:
        return "low", f"Too noisy/flat (R2={r_squared:.2f})"
    if horizon_days > 30 or r_squared < 0.4:
        return "medium", f"Moderate (R2={r_squared:.2f}, {horizon_days}-day)"
    return "high", f"Stable trend (R2={r_squared:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# CORE FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
def forecast_price_for_product(
    db_path: str,
    product_id: int,
    target_dates: List[date] = None,
    lookback_days: int = 90,
    start_date: date = None,
) -> Optional[PriceForecast]:
    """
    Fits a linear price trend and projects to specific target dates.

    Args:
        db_path: Path to SQLite database
        product_id: Product ID to forecast
        target_dates: List of dates to project prices for
        lookback_days: How many days of history to use
        start_date: Start date for lookback (default: today - lookback_days)
    """
    conn = sqlite3.connect(db_path)

    prod_row = conn.execute(
        "SELECT name_ar, name_en, category, avg_price FROM products WHERE id=?", (product_id,)
    ).fetchone()
    if not prod_row:
        conn.close()
        return None

    # Build date filter
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

    current_price = float(prices[-1])
    last_date = datetime.strptime(price_history["sale_date"].iloc[-1], "%Y-%m-%d").date()
    last_x = len(prices) - 1

    # Default target dates if none provided
    if not target_dates:
        target_dates = [last_date + timedelta(days=h) for h in [7, 14, 30, 60, 90]]

    projections = {}
    reliability_by_horizon = {}
    reliability_reasons = {}
    holiday_adjustments = {}

    for target_date in target_dates:
        horizon_days = (target_date - last_date).days
        if horizon_days <= 0:
            continue

        projected = intercept + slope * (last_x + horizon_days)
        projected = max(0.0, projected)

        # Check if target date is a holiday
        holiday_name = None
        for holiday in UPCOMING_HOLIDAYS:
            if abs((target_date - holiday["date"]).days) <= 3:
                holiday_name = holiday["name_ar"]
                # Apply demand multiplier to price (higher demand = higher price)
                adjusted_price = projected * (1 + (holiday["multiplier"] - 1) * 0.3)
                holiday_adjustments[holiday_name] = round(adjusted_price, 1)
                break

        label = target_date.strftime("%Y-%m-%d")
        if holiday_name:
            label = f"{holiday_name} ({target_date.strftime('%Y-%m-%d')})"

        projections[label] = round(projected, 1)
        rel, reason = _reliability_label(r_squared, len(prices), horizon_days)
        reliability_by_horizon[label] = rel
        reliability_reasons[label] = reason

    return PriceForecast(
        product_id=product_id,
        product_ar=prod_row[0],
        product_en=prod_row[1],
        category=prod_row[2],
        current_price=round(current_price, 1),
        trend_per_day=round(float(slope), 4),
        r_squared=round(float(r_squared), 3),
        data_points=len(prices),
        projections=projections,
        reliability_by_horizon=reliability_by_horizon,
        reliability_reasons=reliability_reasons,
        holiday_adjustments=holiday_adjustments,
    )


def forecast_all_product_prices(
    db_path: str,
    target_dates: List[date] = None,
    lookback_days: int = 90,
    start_date: date = None,
    category_filter: str = None,
) -> List[PriceForecast]:
    """Forecast prices for all products (optionally filtered by category)."""
    conn = sqlite3.connect(db_path)
    if category_filter:
        product_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM products WHERE category=? ORDER BY name_ar", (category_filter,)
            ).fetchall()
        ]
    else:
        product_ids = [
            r[0] for r in conn.execute("SELECT id FROM products ORDER BY name_ar").fetchall()
        ]
    conn.close()

    results = []
    for pid in product_ids:
        fc = forecast_price_for_product(db_path, pid, target_dates, lookback_days, start_date)
        if fc is not None:
            results.append(fc)

    return results


def get_all_categories(db_path: str) -> List[str]:
    """Get all product categories from the database."""
    conn = sqlite3.connect(db_path)
    categories = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()
    ]
    conn.close()
    return categories


def get_all_products(db_path: str, category: str = None) -> List[Dict]:
    """Get all products with their details."""
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

    return [
        {"id": r[0], "name_ar": r[1], "name_en": r[2], "category": r[3], "avg_price": r[4]}
        for r in rows
    ]


def to_dataframe(forecasts: List[PriceForecast]) -> pd.DataFrame:
    """Flatten forecasts into a review-friendly table."""
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    rows = []
    for fc in forecasts:
        row = {
            "product_ar": fc.product_ar,
            "product_en": fc.product_en,
            "category": fc.category,
            "current_price": fc.current_price,
            "trend_per_day": fc.trend_per_day,
        }
        for label, price in fc.projections.items():
            rel = fc.reliability_by_horizon.get(label, "low")
            row[label] = f"{icon.get(rel, '')} {price}"
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_price_forecast_page(db_path: str):
    import streamlit as st
    import plotly.graph_objects as go

    st.title("Price Forecasting — All Products")
    st.markdown(
        "<div style='background:#fff3cd;border:2px solid #ffc107;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "Linear price trend projection. Reliability decreases with longer horizons."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Product Selection ─────────────────────────────────────────────────────
    st.subheader("Product Selection")
    categories = get_all_categories(db_path)
    selected_category = st.selectbox(
        "Category (optional):",
        ["All Categories"] + categories,
    )

    if selected_category == "All Categories":
        products = get_all_products(db_path)
    else:
        products = get_all_products(db_path, selected_category)

    product_names = [f"{p['name_ar']} ({p['name_en']})" for p in products]
    selected_product = st.selectbox("Select Product:", product_names)
    selected_product_idx = product_names.index(selected_product)
    selected_product_id = products[selected_product_idx]["id"]

    # ── Date Range Selection ──────────────────────────────────────────────────
    st.subheader("Data Range")
    c1, c2 = st.columns(2)
    with c1:
        lookback_days = st.slider("Historical Data (days):", 30, 365, 90)
    with c2:
        use_custom_start = st.checkbox("Custom Start Date")
        if use_custom_start:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
        else:
            start_date = None

    # ── Target Date Selection ─────────────────────────────────────────────────
    st.subheader("Target Dates for Forecasting")

    # Holiday selection
    st.markdown("**Select Islamic Holidays:**")
    selected_holidays = []
    cols = st.columns(3)
    for i, holiday in enumerate(UPCOMING_HOLIDAYS):
        with cols[i % 3]:
            if st.checkbox(holiday["name_ar"], key=f"holiday_{i}"):
                selected_holidays.append(holiday)

    # Custom date selection
    st.markdown("**Or Add Custom Dates:**")
    custom_dates = []
    num_custom = st.number_input("Number of custom dates:", 0, 10, 0)
    for i in range(int(num_custom)):
        d = st.date_input(f"Custom Date {i+1}", value=date.today() + timedelta(days=30*(i+1)), key=f"custom_{i}")
        custom_dates.append(d)

    # Build target dates list
    target_dates = [h["date"] for h in selected_holidays] + custom_dates
    target_dates = sorted(set(target_dates))

    # ── Forecast Button ───────────────────────────────────────────────────────
    if st.button("Forecast Prices", type="primary"):
        if not target_dates:
            st.warning("Please select at least one target date.")
            return

        with st.spinner("Analyzing price history..."):
            forecast = forecast_price_for_product(
                db_path,
                product_id=selected_product_id,
                target_dates=target_dates,
                lookback_days=lookback_days,
                start_date=start_date,
            )

        if forecast is None:
            st.warning("Not enough sales data for this product.")
            return

        # ── Display Results ───────────────────────────────────────────────────
        st.success(f"Forecast for {forecast.product_ar}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"{forecast.current_price} EGP")
        c2.metric("Trend/Day", f"{forecast.trend_per_day:+.4f} EGP")
        c3.metric("R-squared", f"{forecast.r_squared:.3f}")
        c4.metric("Data Points", forecast.data_points)

        # Projections table
        st.subheader("Price Projections")
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        for label, price in forecast.projections.items():
            rel = forecast.reliability_by_horizon.get(label, "low")
            reason = forecast.reliability_reasons.get(label, "")
            st.markdown(f"**{icon.get(rel, '')} {label}:** {price} EGP — {reason}")

        # Holiday adjustments
        if forecast.holiday_adjustments:
            st.subheader("Holiday Price Adjustments")
            for holiday, adj_price in forecast.holiday_adjustments.items():
                st.info(f"{holiday}: Adjusted price = {adj_price} EGP")

        # Chart
        st.subheader("Price Trend Chart")
        fig = go.Figure()

        # Historical data
        conn = sqlite3.connect(db_path)
        hist = pd.read_sql_query(
            f"""SELECT sale_date, AVG(unit_price) avg_price FROM sales
                WHERE product_id={forecast.product_id}
                AND sale_date >= date('now', '-{lookback_days} days')
                GROUP BY sale_date ORDER BY sale_date""",
            conn,
        )
        conn.close()

        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["sale_date"],
                y=hist["avg_price"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="#1f77b4"),
            ))

        # Projections
        proj_dates = []
        proj_prices = []
        for label, price in forecast.projections.items():
            # Extract date from label
            for td in target_dates:
                if td.strftime("%Y-%m-%d") in label:
                    proj_dates.append(td.strftime("%Y-%m-%d"))
                    proj_prices.append(price)
                    break

        if proj_dates:
            fig.add_trace(go.Scatter(
                x=proj_dates,
                y=proj_prices,
                mode="lines+markers",
                name="Projected",
                line=dict(color="#8B1A1A", dash="dash"),
            ))

        fig.update_layout(
            title=f"Price Forecast — {forecast.product_ar}",
            xaxis_title="Date",
            yaxis_title="Price (EGP/kg)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── All Products Comparison ───────────────────────────────────────────────
    st.divider()
    st.subheader("All Products Comparison")
    if st.button("Forecast All Products"):
        with st.spinner("Analyzing all products..."):
            forecasts = forecast_all_product_prices(
                db_path,
                target_dates=target_dates,
                lookback_days=lookback_days,
                start_date=start_date,
                category_filter=None if selected_category == "All Categories" else selected_category,
            )

        if not forecasts:
            st.warning("Not enough data for any product.")
            return

        df = to_dataframe(forecasts)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Products Forecasted", len(forecasts))
        c2.metric("High Confidence", sum(1 for f in forecasts if any(r == "high" for r in f.reliability_by_horizon.values())))
        c3.metric("Low Confidence", sum(1 for f in forecasts if all(r == "low" for r in f.reliability_by_horizon.values())))
