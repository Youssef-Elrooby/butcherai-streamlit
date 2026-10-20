"""
ButcherAI — Advanced Intelligent Marketing
==============================================
Upgrades the basic single-product/single-format marketing page into a
data-grounded, multi-channel content engine.

What this adds over the existing "📢 التسويق الذكي" page:
  1. Price safety check — validates special price against REAL WAC
     (not the static products.avg_cost), warns before you publish a loss-leader
     by accident.
  2. Auto-context detection — reads today's actual date via get_mult() and
     pre-selects Ramadan/Eid/Friday/regular instead of manual guessing.
  3. RAG-grounded generation — injects real domain knowledge (margins,
     seasonal patterns) into the Gemini prompt so content isn't generic.
  4. Multi-channel output — WhatsApp, Facebook post, and price-board sign
     text generated together, each with correct length/tone for that channel.
  5. A/B headline variants — 2 options per generation, not 1.
  6. Smart product suggestions — surfaces slow-movers and high-margin
     products worth promoting right now, using real sales data.

Integration into app.py — replace the existing "📢 التسويق الذكي" elif block with:

    elif page == "📢 التسويق الذكي":
        from advanced_marketing_page import show_advanced_marketing_page
        show_advanced_marketing_page(
            db_path=DB, gemini_api_key=GEMINI_API_ENV, get_mult_fn=get_mult
        )
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRICE SAFETY CHECK — real WAC validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_special_price(db_path: str, product_name_ar: str, proposed_price: float,
                            days_back: int = 90) -> Dict:
    """
    Checks a proposed marketing price against the REAL weighted average cost
    computed from actual purchase history — not the static catalogue value.

    Returns: {safe: bool, wac: float, margin_pct: float, warning: str or None}
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            """
            SELECT ROUND(SUM(total_cost) * 1.0 / NULLIF(SUM(weight_kg), 0), 2) AS wac
            FROM purchases
            WHERE product_ar LIKE ? AND purchase_date >= date('now', ?)
            """,
            (f"%{product_name_ar}%", f"-{int(days_back)} days"),
        ).fetchone()
        conn.close()

        wac = row[0] if row and row[0] else None
        if wac is None:
            return {
                "safe": True, "wac": None, "margin_pct": None,
                "warning": "⚠️ لا توجد بيانات شراء كافية للتحقق من التكلفة الحقيقية — احذر عند التسعير يدوياً.",
            }

        margin_pct = round((proposed_price - wac) / proposed_price * 100, 1) if proposed_price > 0 else 0

        if proposed_price < wac:
            return {
                "safe": False, "wac": wac, "margin_pct": margin_pct,
                "warning": f"🔴 خطر: السعر المقترح ({proposed_price:.0f} ج) أقل من التكلفة الحقيقية (WAC: {wac:.0f} ج/كجم)! هذا سيُسبب خسارة فعلية على كل عملية بيع.",
            }
        elif margin_pct < 10:
            return {
                "safe": True, "wac": wac, "margin_pct": margin_pct,
                "warning": f"⚠️ هامش منخفض جداً ({margin_pct:.1f}%) — التكلفة الحقيقية {wac:.0f} ج/كجم. فكر في رفع السعر قليلاً.",
            }
        else:
            return {"safe": True, "wac": wac, "margin_pct": margin_pct, "warning": None}

    except Exception as exc:
        return {"safe": True, "wac": None, "margin_pct": None, "warning": f"تعذر التحقق: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTO-CONTEXT DETECTION — read today's actual demand season
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_marketing_context(get_mult_fn, target_date: Optional[date] = None) -> Dict:
    """
    Uses the real holiday/weekday multiplier function to determine what
    marketing context to pre-select, instead of the user guessing manually.

    Returns: {context: str, label_ar: str, multiplier: float, days_to_next_peak: int}
    """
    target_date = target_date or date.today()
    mult, label_ar = get_mult_fn(target_date)

    if mult >= 2.5:
        context = "eid"
    elif mult >= 1.4:
        context = "ramadan"
    elif mult >= 1.2:
        context = "weekend"
    else:
        context = "general"

    # Look ahead 21 days for the next real demand spike, to suggest pre-marketing timing
    days_to_peak = None
    for i in range(1, 22):
        future_date = target_date + timedelta(days=i)
        fm, flabel = get_mult_fn(future_date)
        if fm >= 1.4:
            days_to_peak = i
            break

    return {
        "context": context,
        "label_ar": label_ar,
        "multiplier": mult,
        "days_to_next_peak": days_to_peak,
        "detected_today": target_date == date.today(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMART PRODUCT SUGGESTIONS — what's actually worth promoting right now
# ─────────────────────────────────────────────────────────────────────────────
def get_marketing_suggestions(db_path: str, days_back: int = 14) -> Dict[str, List[Dict]]:
    """
    Surfaces two categories of products worth marketing right now:
      - slow_movers: high stock, low recent sales (need a push to avoid waste)
      - high_margin: good margin but under-promoted (could sell more at current price)

    Uses actual sales + inventory data, not guesswork.
    """
    try:
        conn = sqlite3.connect(db_path)

        slow_movers = pd.read_sql_query(
            f"""
            SELECT p.name_ar, p.name_en, p.category, p.avg_price,
                   COALESCE(i.current_stock_kg, 0) AS stock_kg,
                   COALESCE(SUM(s.weight_kg), 0) AS sold_last_period
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN sales s ON p.id = s.product_id AND s.sale_date >= date('now', '-{days_back} days')
            GROUP BY p.id
            HAVING stock_kg > 5 AND sold_last_period < stock_kg * 0.3
            ORDER BY stock_kg DESC
            LIMIT 5
            """,
            conn,
        )

        high_margin = pd.read_sql_query(
            f"""
            SELECT p.name_ar, p.name_en, p.category, p.avg_price, p.avg_margin,
                   COALESCE(SUM(s.weight_kg), 0) AS sold_last_period
            FROM products p
            LEFT JOIN sales s ON p.id = s.product_id AND s.sale_date >= date('now', '-{days_back} days')
            WHERE p.avg_margin >= 30
            GROUP BY p.id
            ORDER BY p.avg_margin DESC
            LIMIT 5
            """,
            conn,
        )
        conn.close()

        return {
            "slow_movers": slow_movers.to_dict("records"),
            "high_margin": high_margin.to_dict("records"),
        }
    except Exception as exc:
        return {"slow_movers": [], "high_margin": [], "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. RAG-GROUNDED, MULTI-CHANNEL CONTENT GENERATION
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_SPECS = {
    "whatsapp": {
        "label_ar": "واتساب",
        "max_words": 40,
        "tone": "friendly, personal, uses 1-2 emoji, ends with a clear call to visit or order",
    },
    "facebook": {
        "label_ar": "فيسبوك",
        "max_words": 70,
        "tone": "engaging, storytelling opener, 2-3 emoji, includes a question to drive comments",
    },
    "price_board": {
        "label_ar": "لوحة الأسعار",
        "max_words": 15,
        "tone": "very short, bold, price-focused, no emoji, shop-sign style",
    },
}


def _build_rag_context(product_ar: str, category: str, api_key: str, db_path: str) -> str:
    """Pull relevant domain knowledge from the RAG engine to ground the marketing copy."""
    try:
        from phase1_rag_engine import get_rag
        rag = get_rag(db_path=db_path)
        query = f"{product_ar} {category} margin pricing seasonal demand"
        context = rag.build_context_block(query, n_results=3)
        return context
    except Exception:
        return ""  # RAG unavailable — generation still works, just less grounded


def generate_advanced_marketing(
    db_path: str,
    product_ar: str,
    product_en: str,
    category: str,
    price: float,
    context_season: str,
    api_key: str = "",
    channels: Optional[List[str]] = None,
) -> Dict:
    """
    Main generation function. Produces:
      - 2 Arabic headline variants (A/B)
      - Full content for each requested channel (default: all 3)
      - Grounded in RAG domain knowledge when available

    Returns a dict with keys: variants, channels, rag_used, source
    """
    channels = channels or list(CHANNEL_SPECS.keys())
    rag_context = _build_rag_context(product_ar, category, api_key, db_path)

    if not api_key:
        return _template_fallback(product_ar, product_en, category, price, context_season, channels)

    try:
        import genai_compat as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-lite-latest")

        channel_instructions = "\n".join(
            f"- {ch}: max {CHANNEL_SPECS[ch]['max_words']} words, tone: {CHANNEL_SPECS[ch]['tone']}"
            for ch in channels
        )

        prompt = f"""{rag_context}
You are a marketing specialist for an Egyptian butcher shop. Using the domain
knowledge above (if provided), write marketing content for:

Product: {product_en} ({product_ar})
Category: {category}
Price: {price:.0f} EGP/kg
Season/Context: {context_season}

Generate content for these channels with these exact constraints:
{channel_instructions}

Also generate 2 DIFFERENT Arabic headline variants (for A/B testing) —
one emphasizing quality/premium, one emphasizing value/price.

Output ONLY valid JSON with this exact structure:
{{
  "headline_variant_a": "...",
  "headline_variant_b": "...",
  "whatsapp": {{"arabic": "...", "english": "..."}},
  "facebook": {{"arabic": "...", "english": "..."}},
  "price_board": {{"arabic": "...", "english": "..."}}
}}
Only include the channel keys that were requested above."""

        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(resp.text)
        parsed["rag_used"] = bool(rag_context)
        parsed["source"] = "gemini_ai"
        return parsed

    except Exception as exc:
        fallback = _template_fallback(product_ar, product_en, category, price, context_season, channels)
        fallback["error"] = str(exc)
        return fallback


def _template_fallback(product_ar: str, product_en: str, category: str, price: float,
                        context_season: str, channels: List[str]) -> Dict:
    """Offline/no-API-key fallback — still produces multi-channel, A/B content."""
    ctx_ar = {"ramadan": "في رمضان", "eid": "في العيد", "weekend": "يوم الجمعة", "general": ""}
    result = {
        "headline_variant_a": f"🥩 {product_ar} فاخر {ctx_ar.get(context_season, '')} — جودة لا تُضاهى",
        "headline_variant_b": f"🥩 {product_ar} بسعر مميز {ctx_ar.get(context_season, '')} — {price:.0f} ج/كجم",
        "rag_used": False,
        "source": "template_fallback",
    }
    if "whatsapp" in channels:
        result["whatsapp"] = {
            "arabic": f"🥩 {product_ar} طازج اليوم! السعر: {price:.0f} ج/كجم. اطلب الآن 📞",
            "english": f"🥩 Fresh {product_en} today! {price:.0f} EGP/kg. Order now 📞",
        }
    if "facebook" in channels:
        result["facebook"] = {
            "arabic": f"هل جربت {product_ar} الطازج عندنا؟ 🥩 جودة عالية بسعر {price:.0f} ج/كجم. زورونا اليوم!",
            "english": f"Have you tried our fresh {product_en}? 🥩 Premium quality at {price:.0f} EGP/kg. Visit us today!",
        }
    if "price_board" in channels:
        result["price_board"] = {
            "arabic": f"{product_ar}: {price:.0f} ج/كجم",
            "english": f"{product_en}: {price:.0f} EGP/kg",
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_advanced_marketing_page(db_path: str, gemini_api_key: str = "", get_mult_fn=None):
    import streamlit as st

    st.title("📢 التسويق الذكي المتقدم")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "🧠 يستخدم بيانات محلك الحقيقية: WAC الفعلي، التقويم الموسمي، وقاعدة المعرفة "
        "لتوليد محتوى مبني على بيانات — عبر 3 قنوات مختلفة بضغطة واحدة."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Auto-context detection ────────────────────────────────────────────────
    if get_mult_fn:
        auto_ctx = auto_detect_marketing_context(get_mult_fn)
        c1, c2, c3 = st.columns(3)
        c1.metric("السياق المكتشف اليوم", auto_ctx["label_ar"])
        c2.metric("مضاعف الطلب", f"×{auto_ctx['multiplier']:.2f}")
        if auto_ctx["days_to_next_peak"]:
            c3.metric("أقرب موسم ذروة", f"بعد {auto_ctx['days_to_next_peak']} يوم")
        else:
            c3.metric("أقرب موسم ذروة", "لا يوجد خلال 21 يوم")
        default_context = auto_ctx["context"]
    else:
        default_context = "general"

    st.divider()

    # ── Smart product suggestions ──────────────────────────────────────────────
    with st.expander("💡 اقتراحات ذكية — منتجات تستحق الترويج الآن", expanded=False):
        suggestions = get_marketing_suggestions(db_path)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🐢 حركة بطيئة (تحتاج دفعة تسويقية)**")
            if suggestions["slow_movers"]:
                for item in suggestions["slow_movers"]:
                    st.markdown(f"- {item['name_ar']}: {item['stock_kg']:.0f} كجم مخزون، "
                                f"{item['sold_last_period']:.0f} كجم مباع فقط")
            else:
                st.caption("لا توجد منتجات بحركة بطيئة حالياً")
        with sc2:
            st.markdown("**💰 هامش مرتفع (يستحق ترويج أكبر)**")
            if suggestions["high_margin"]:
                for item in suggestions["high_margin"]:
                    st.markdown(f"- {item['name_ar']}: هامش {item.get('avg_margin', 0):.0f}%، "
                                f"{item['sold_last_period']:.0f} كجم مباع مؤخراً")
            else:
                st.caption("لا توجد بيانات كافية")

    st.divider()

    # ── Main form ────────────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    products_df = pd.read_sql_query("SELECT * FROM products ORDER BY name_ar", conn)
    conn.close()

    c1, c2 = st.columns(2)
    with c1:
        sel = st.selectbox("المنتج:", products_df.name_ar.tolist())
        row = products_df[products_df.name_ar == sel].iloc[0]
        price = st.number_input(
            "💰 سعر العرض (ج/كجم):", min_value=0.0, value=float(row.avg_price), step=5.0
        )
    with c2:
        context_options = ["general", "ramadan", "eid", "weekend", "clearance"]
        context_labels = {"general": "يوم عادي", "ramadan": "رمضان", "eid": "عيد",
                          "weekend": "نهاية الأسبوع", "clearance": "تصفية"}
        default_idx = context_options.index(default_context) if default_context in context_options else 0
        ctx = st.selectbox(
            "السياق:", context_options, index=default_idx,
            format_func=lambda x: context_labels[x] + (" (مكتشف تلقائياً)" if x == default_context else ""),
        )
        channels_selected = st.multiselect(
            "القنوات المطلوبة:",
            ["whatsapp", "facebook", "price_board"],
            default=["whatsapp", "facebook", "price_board"],
            format_func=lambda x: CHANNEL_SPECS[x]["label_ar"],
        )

    api_key = st.text_input("🔑 Gemini API Key (اختياري):", type="password", value=gemini_api_key)

    # ── Price safety check — always runs, before generation ───────────────────
    validation = validate_special_price(db_path, sel, price)
    if validation["warning"]:
        if not validation["safe"]:
            st.error(validation["warning"])
        else:
            st.warning(validation["warning"])
    elif validation["wac"] is not None:
        st.success(f"✅ السعر آمن — التكلفة الحقيقية (WAC): {validation['wac']:.0f} ج/كجم، "
                   f"الهامش: {validation['margin_pct']:.1f}%")

    # ── Generate button ─────────────────────────────────────────────────────
    if st.button("✨ توليد محتوى متقدم", type="primary", use_container_width=True):
        if not validation["safe"]:
            st.error("⚠️ لن يتم التوليد — السعر أقل من التكلفة الحقيقية. عدّل السعر أولاً.")
        else:
            with st.spinner("جاري التوليد باستخدام قاعدة المعرفة والبيانات الحقيقية..."):
                result = generate_advanced_marketing(
                    db_path=db_path, product_ar=str(row.name_ar), product_en=str(row.name_en),
                    category=str(row.category), price=price, context_season=ctx,
                    api_key=api_key, channels=channels_selected,
                )

            if result.get("rag_used"):
                st.info("🧠 تم إثراء المحتوى من قاعدة المعرفة (RAG)")

            st.subheader("🅰️🅱️ عناوين بديلة (اختبار A/B)")
            vc1, vc2 = st.columns(2)
            vc1.markdown(f"<div style='background:#fff8f8;border-right:4px solid #8B1A1A;"
                         f"border-radius:8px;padding:12px;direction:rtl'><b>A (فاخر):</b><br>"
                         f"{result.get('headline_variant_a','')}</div>", unsafe_allow_html=True)
            vc2.markdown(f"<div style='background:#f0f8ff;border-right:4px solid #1A5276;"
                         f"border-radius:8px;padding:12px;direction:rtl'><b>B (قيمة):</b><br>"
                         f"{result.get('headline_variant_b','')}</div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("📱 المحتوى حسب القناة")
            for ch in channels_selected:
                if ch in result:
                    with st.expander(f"{CHANNEL_SPECS[ch]['label_ar']}", expanded=True):
                        cc1, cc2 = st.columns(2)
                        cc1.markdown(f"**🇸🇦 عربي**\n\n{result[ch].get('arabic','')}")
                        cc2.markdown(f"**🇬🇧 English**\n\n{result[ch].get('english','')}")

            st.caption(f"المصدر: {result.get('source', 'unknown')}")
            if "error" in result:
                st.caption(f"⚠️ Gemini fallback reason: {result['error'][:150]}")
