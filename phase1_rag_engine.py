"""
ButcherAI Phase 1 — RAG (Retrieval-Augmented Generation) Engine
================================================================
Grounds every AI response in verified Egyptian butchery domain knowledge
AND live business data extracted automatically from the SQLite database.

Architecture:
  ChromaDB (local persistent vector store, zero subscription, works offline)
  + sentence-transformers paraphrase-multilingual-MiniLM-L12-v2
    (supports Arabic + English natively, ~120MB, runs on CPU)
  + 12 curated domain knowledge documents
  + 5 auto-generated live DB summary documents (refreshed every hour)

Install:
  pip install chromadb sentence-transformers

Integration into app.py (add to sidebar and router):
  elif page == "📚 قاعدة المعرفة (RAG)":
      from phase1_rag_engine import show_rag_page
      show_rag_page(db_path=DB, gemini_api_key=GEMINI_API_ENV)
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import math
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from llm_observability import track_llm_call

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PATH     = os.environ.get("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "butcherai_rag_v1"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"  # Arabic + English
DB_PATH         = os.environ.get("DB_PATH", "ButcherAI.db")
SUMMARY_TTL_SEC = 3600   # regenerate live DB summaries every 60 minutes
MIN_RELEVANCE   = 0.25   # discard docs with cosine similarity < this

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE CORPUS
# 12 curated Egyptian butchery documents (bilingual AR+EN)
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_DOCS: List[Dict] = [
    {
        "id": "dk_yield_cattle",
        "text": (
            "ما هي نسبة استفادة البقري البلدي؟ نسبة الاستفادة 70%. "
            "Baladi cattle (لحم بلدي) yield exactly 70% usable meat from live/dressed weight in Egypt. "
            "Cut distribution of the 70 kg saleable portion per 100 kg purchased: "
            "Chuck/دوش 20% (14 kg), Ribs/ريش 15% (10.5 kg), Tenderloin/فيليه 8% (5.6 kg), "
            "Round-Leg/موزة 18% (12.6 kg), Shoulder/كتف 14% (9.8 kg), "
            "Ground/مفروم 15% (10.5 kg), Liver-Offal/كبدة 5% (3.5 kg), Processed/مشتقات 5% (3.5 kg). "
            "نسبة استفادة البقري البلدي 70%. الدوش 20%، الريش 15%، الفيليه 8%، الموزة 18%، "
            "الكتف 14%، المفروم 15%، الكبدة 5%، المشتقات 5%."
        ),
        "metadata": {"category": "yield", "animal": "cattle", "source": "domain"},
    },
    {
        "id": "dk_yield_lamb",
        "text": (
            "كم نسبة استفادة الضاني؟ نسبة استفادة اللحم الضاني 60%. "
            "Egyptian lamb (لحم ضاني) yields 60% usable meat from dressed weight. "
            "Cut distribution: Leg/فخدة 35% (21 kg per 100 kg), Shoulder/كتف 25% (15 kg), "
            "Rack-Ribs/ريش 20% (12 kg), Ground/مفروم 12% (7.2 kg), Offal/مشتقات 8% (4.8 kg). "
            "Average Egyptian lamb weighs 35-50 kg live, yielding 21-30 kg usable meat per animal. "
            "نسبة استفادة الضاني 60%. الفخدة 35%، الكتف 25%، الريش 20%، المفروم 12%، المشتقات 8%."
        ),
        "metadata": {"category": "yield", "animal": "lamb", "source": "domain"},
    },
    {
        "id": "dk_ramadan_demand",
        "text": (
            "كم يزيد الطلب في رمضان؟ الطلب يرتفع 55% (مضاعف 1.55). "
            "Ramadan demand pattern: 1.55× multiplier (55% above baseline) sustained for 30 days. "
            "Peak demand: Iftar preparation window 2pm-7pm drives 65% of daily sales. "
            "Top Ramadan products: ground meat (kofta), thin-sliced beef for grilling, whole chicken (non-butcher). "
            "Procurement rule: start ordering 2 weeks before Ramadan; increase daily orders by 50%. "
            "Egyptian consumer behavior: families buy in larger batches (2-3 kg per purchase vs 1 kg normally). "
            "رمضان: الطلب يرتفع 55% طوال الشهر. ذروة الشراء 2-7 مساء للإفطار. "
            "المفروم والشرائح الرفيعة الأكثر طلباً. ابدأ الشراء قبل أسبوعين. زد المخزون 50%."
        ),
        "metadata": {"category": "seasonal", "holiday": "ramadan", "source": "domain"},
    },
    {
        "id": "dk_eid_adha",
        "text": (
            "ما هو مضاعف الطلب في عيد الأضحى؟ مضاعف الطلب في عيد الأضحى 3.0× (200-300%). "
            "Eid Al-Adha: highest demand period of the year at 3.0× multiplier (200-300% above normal). "
            "Duration: 4-7 days of elevated demand starting day before Eid. "
            "Whole carcass demand: families buy entire lambs (Udhiya/sacrifice), driving carcass sales not cuts. "
            "Revenue impact: a single Eid week can equal 5-7 normal weeks combined. "
            "Procurement must start 3 weeks before: suppliers run out 10 days before Eid. "
            "Alert threshold: trigger CRITICAL procurement alert 21 days before calculated Eid date. "
            "عيد الأضحى: مضاعف ×3. الطلب أعلى موسم في العام. "
            "الذبائح الكاملة تُباع بدل القطعيات. إيرادات أسبوع العيد = 5-7 أسابيع عادية. "
            "المشتريات تبدأ قبل 3 أسابيع."
        ),
        "metadata": {"category": "seasonal", "holiday": "eid_adha", "source": "domain"},
    },
    {
        "id": "dk_eid_fitr",
        "text": (
            "مضاعف الطلب في عيد الفطر 2.2× لمدة 3 أيام. "
            "Eid Al-Fitr: 2.2× demand multiplier for 3 days. "
            "Consumer preference shifts to premium whole cuts (leg of lamb, ribeye) not ground meat. "
            "Families celebrate with high-quality pieces for Eid feasts. "
            "Stock up 7-10 days before with emphasis on premium cuts at full price (no discounting). "
            "عيد الفطر: مضاعف ×2.2 لمدة 3 أيام. القطعيات الكاملة الفاخرة مطلوبة أكثر. "
            "الفخدة والريش والفيليه الأكثر طلباً. الشراء قبل 7-10 أيام."
        ),
        "metadata": {"category": "seasonal", "holiday": "eid_fitr", "source": "domain"},
    },
    {
        "id": "dk_friday_pattern",
        "text": (
            "What is the demand pattern on Fridays? Friday weekly pattern: 1.30× multiplier. "
            "Friday weekly pattern: 1.30× multiplier — highest demand weekday in Egypt. "
            "Deep Egyptian cultural tradition: family meat meal on Fridays (غداء الجمعة). "
            "Peak hours: 9am-2pm (85% of Friday sales). Stock must be fresh and fully replenished by 9am. "
            "Saturday: 1.15×, Sunday: 1.10×, Monday: 0.90× (post-weekend slowdown). "
            "Operational implication: buy fresh stock on Thursday evening for Friday opening. "
            "الجمعة: مضاعف 1.30. تقليد مصري بوجبة لحم يوم الجمعة. "
            "الذروة 9 صباحاً - 2 ظهراً. اشتر الخميس مساءً للفتح الجمعة."
        ),
        "metadata": {"category": "weekly_pattern", "day": "friday", "source": "domain"},
    },
    {
        "id": "dk_margins",
        "text": (
            "ما هو الهامش الصحي للفيليه؟ الهامش الصحي للفيليه 35-45%. "
            "Egyptian butcher profit margins by category: "
            "Premium cuts (tenderloin/فيليه, ribeye/ريش): 35-45% margin. "
            "Standard cuts (round/موزة, shoulder/كتف, chuck/دوش): 20-30% margin. "
            "Ground meat (مفروم): 15-22% margin (low margin but highest volume, drives traffic). "
            "Offal/Liver (كبدة): 40-55% margin (often overlooked high-margin product). "
            "Wholesale-to-retail markup: 25-40%. "
            "Warning: any product below 15% is danger zone. Below 10% is loss. "
            "Healthy overall shop margin: 22-28%. "
            "هوامش الربح: الفيليه والريش 35-45%، العادي 20-30%، المفروم 15-22%، الكبدة 40-55%. "
            "تحذير عند أقل من 15%. صحي يكون 22-28% للمحل ككل."
        ),
        "metadata": {"category": "financial", "source": "domain"},
    },
    {
        "id": "dk_inventory_rules",
        "text": (
            "كم مدة صلاحية اللحم الطازج؟ مدة الصلاحية 2-3 أيام. "
            "Egyptian butcher inventory rules: "
            "Fresh meat at 0-4°C: 2-3 days maximum shelf life. "
            "Frozen at -18°C: up to 3 months. "
            "Never hold more than 2 days average demand as fresh stock. "
            "Daily procurement formula: max(0, predicted_demand × 1.10 - current_stock). "
            "The 10% buffer prevents stockouts without excess. "
            "Waste reduction: mark down slow items by 15-20% at 4pm. Convert unsold to ground meat. "
            "Waste target: below 3% of total revenue. "
            "قواعد المخزون: لحم طازج 2-3 أيام فقط (0-4 درجة). مجمد 3 أشهر. "
            "لا تشترِ أكثر من يومين طلب. معادلة الشراء: الطلب × 1.1 - المخزون الحالي. "
            "هدف الهالك: أقل من 3% من الإيرادات."
        ),
        "metadata": {"category": "inventory", "source": "domain"},
    },
    {
        "id": "dk_wac_analysis",
        "text": (
            "كيف أحسب WAC؟ WAC = Σ(quantity_purchased × cost_per_kg) / Σ(quantity_purchased). "
            "WAC (Weighted Average Cost) formula: WAC = Σ(quantity_purchased × cost_per_kg) / Σ(quantity_purchased). "
            "In a butcher shop context, WAC must be computed separately per product, not across all products. "
            "Compare WAC to actual selling price for real margin. "
            "WAC rising faster than selling price = margin squeeze = raise prices or find cheaper supplier. "
            "Track WAC weekly to detect wholesale inflation 2-3 weeks before it hits customers. "
            "A supplier with consistent WAC is more valuable than one with lower average but high variability. "
            "تحليل WAC: المتوسط المرجح للتكلفة = مجموع (الكمية × السعر) ÷ الكمية الكلية. "
            "WAC يرتفع أسرع من البيع = ضغط على الهامش. تتبع أسبوعياً لاكتشاف التضخم مبكراً."
        ),
        "metadata": {"category": "financial_analysis", "source": "domain"},
    },
    {
        "id": "dk_supplier_strategy",
        "text": (
            "Supplier management best practices for Egyptian butchers: "
            "Maintain 2-3 active suppliers minimum to avoid single-supplier dependency. "
            "Pre-Eid bulk discount negotiation: contact suppliers 4 weeks before Eid, "
            "request 10-15% discount for pre-committed large orders. "
            "Supplier scoring: price consistency (40%) + quality score (35%) + delivery reliability (25%). "
            "Blacklist criteria: 3+ late deliveries in 30 days OR quality complaint on >5% of orders. "
            "استراتيجية الموردين: 2-3 موردين. خصومات جملة قبل العيد 10-15%. "
            "تقييم: ثبات السعر 40%، الجودة 35%، الالتزام بالمواعيد 25%."
        ),
        "metadata": {"category": "supplier", "source": "domain"},
    },
    {
        "id": "dk_digital_transformation",
        "text": (
            "Digital transformation ROI for Egyptian butcher shops using BI systems: "
            "Waste reduction: 15-25% reduction with demand forecasting vs intuition-based buying. "
            "Profit improvement: 10-20% increase from data-driven pricing and margin tracking. "
            "Time savings: 30-40% of daily management time freed by automation. "
            "Stockout prevention: AI forecast with holiday intelligence reduces stockouts by 60-70%. "
            "Case evidence: shops using weekly demand analysis average 18% higher monthly profit "
            "than those relying on experience alone (Egyptian retail SME studies, 2023-2024). "
            "عائد التحول الرقمي: تقليل الهالك 15-25%، زيادة الربح 10-20%، "
            "توفير 30-40% وقت الإدارة، تقليل النفاد من المخزون 60-70%."
        ),
        "metadata": {"category": "strategy", "source": "domain"},
    },
    {
        "id": "dk_pricing_strategy",
        "text": (
            "Pricing strategy for Egyptian butcher shops: "
            "Cost-plus pricing: price = WAC / (1 - target_margin). "
            "Seasonal pricing: raise prices 8-15% in the week before Eid (demand is inelastic). "
            "Competitor benchmarking: check 2-3 nearby competitors weekly for key products (ground meat, leg of lamb). "
            "Price anchoring: display premium cuts prominently at high prices to make standard cuts look like value. "
            "Avoid race to the bottom: compete on freshness and quality, not price alone. "
            "استراتيجية التسعير: سعر = WAC ÷ (1 - الهامش المستهدف). "
            "ارفع الأسعار 8-15% قبل العيد. لا تنافس على السعر فقط، نافس على الطازجية والجودة."
        ),
        "metadata": {"category": "pricing", "source": "domain"},
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BM25 RETRIEVER  — Exercise 1.1: Hybrid Search
# ─────────────────────────────────────────────────────────────────────────────
class BM25Retriever:
    """
    Lightweight BM25 keyword retriever for hybrid search.
    Complements vector search by catching exact word matches that
    embeddings might miss (especially with paraphrased Arabic).

    BM25 formula: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    """

    def __init__(self, documents: List[Dict]):
        """
        documents: list of {id, text, metadata} dicts
        """
        self.documents = documents
        self.k1 = 1.5  # term frequency saturation
        self.b = 0.75  # length normalization
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, keep Arabic."""
        # Keep Arabic letters, Latin letters, and numbers
        tokens = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def _build_index(self):
        """Build the BM25 index from documents."""
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_len: Dict[str, int] = {}
        self.term_freq: Dict[str, Counter] = {}
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}

        total_len = 0
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            self.doc_tokens[doc["id"]] = tokens
            self.doc_len[doc["id"]] = len(tokens)
            self.term_freq[doc["id"]] = Counter(tokens)
            total_len += len(tokens)

        self.avgdl = total_len / max(len(self.documents), 1)

        # Calculate IDF for each term
        N = len(self.documents)
        all_terms = set()
        for tf in self.term_freq.values():
            all_terms.update(tf.keys())

        for term in all_terms:
            df = sum(1 for tf in self.term_freq.values() if term in tf)
            self.idf[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the top-k most relevant documents.
        Returns: list of (doc_id, score) tuples, sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}

        for doc_id, tf in self.term_freq.items():
            score = 0.0
            dl = self.doc_len[doc_id]

            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores[doc_id] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def update_documents(self, documents: List[Dict]):
        """Update the index with new documents (e.g., after live summary refresh)."""
        self.documents = documents
        self._build_index()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RAG CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ButcherAIRAG:
    """
    Local RAG engine using ChromaDB + multilingual sentence-transformers.

    Usage:
        rag = ButcherAIRAG()
        status = rag.initialize()
        results = rag.query("كم يجب أن أشتري قبل عيد الأضحى؟")
        prompt  = rag.build_augmented_prompt("what margin should I target?")
    """

    def __init__(self, db_path: str = DB_PATH, chroma_path: str = CHROMA_PATH):
        self.db_path     = db_path
        self.chroma_path = chroma_path
        self._client     = None
        self._collection = None
        self._embed_fn   = None
        self._last_db_refresh: Optional[datetime] = None
        self.initialized = False
        # Hybrid search: BM25 retriever
        self._bm25: Optional[BM25Retriever] = None
        self._hybrid_weight = 0.8  # weight for vector score (1-weight for BM25)
        # Higher weight = trust semantics more; lower = trust keywords more

    # ── Internal lazy-loaders ─────────────────────────────────────────────────
    def _embed(self):
        if self._embed_fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=EMBED_MODEL
                )
            except ImportError:
                raise ImportError(
                    "Run: pip install chromadb sentence-transformers"
                )
        return self._embed_fn

    def _client_obj(self):
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.chroma_path)
            except ImportError:
                raise ImportError("Run: pip install chromadb")
        return self._client

    def _col(self):
        if self._collection is None:
            self._collection = self._client_obj().get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embed(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Initialization ────────────────────────────────────────────────────────
    def initialize(self) -> str:
        """Seed domain knowledge and update live DB summaries. Returns status string."""
        col = self._col()

        # Check what domain docs are already present
        present = set(col.get(ids=[d["id"] for d in DOMAIN_DOCS])["ids"])
        new_docs = [d for d in DOMAIN_DOCS if d["id"] not in present]

        if new_docs:
            col.add(
                documents=[d["text"]     for d in new_docs],
                ids      =[d["id"]       for d in new_docs],
                metadatas=[d["metadata"] for d in new_docs],
            )

        self._refresh_live_summaries()
        self._build_bm25_index()
        self.initialized = True
        return (
            f"RAG ready: {col.count()} docs total "
            f"({len(new_docs)} new domain docs seeded)"
        )

    def _build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB."""
        try:
            col = self._col()
            all_docs = col.get(include=["documents", "metadatas"])
            if all_docs["ids"]:
                doc_list = [
                    {
                        "id": doc_id,
                        "text": all_docs["documents"][i],
                        "metadata": all_docs["metadatas"][i] or {},
                    }
                    for i, doc_id in enumerate(all_docs["ids"])
                ]
                self._bm25 = BM25Retriever(doc_list)
        except Exception as exc:
            print(f"[RAG] BM25 index build failed: {exc}")
            self._bm25 = None

    # ── Live DB summaries (auto-updated) ──────────────────────────────────────
    def _refresh_live_summaries(self):
        """Pull fresh summaries from SQLite and upsert into ChromaDB."""
        # Throttle: don't refresh more than once per SUMMARY_TTL_SEC
        if (
            self._last_db_refresh
            and (datetime.now() - self._last_db_refresh).seconds < SUMMARY_TTL_SEC
        ):
            return
        if not os.path.exists(self.db_path):
            return

        docs: List[Dict] = []
        try:
            conn = sqlite3.connect(self.db_path)

            # ── 1. Overall 30-day KPIs ────────────────────────────────────────
            r = conn.execute(
                "SELECT COUNT(*) tx, COALESCE(SUM(total_price),0) rev, "
                "COALESCE(SUM(gross_profit),0) pft, "
                "COALESCE(AVG(profit_margin_percent),0) mg, "
                "COALESCE(SUM(weight_kg),0) kg "
                "FROM sales WHERE sale_date >= date('now','-30 days')"
            ).fetchone()
            if r and r[0]:
                docs.append({
                    "id": "live_kpi_30d",
                    "text": (
                        f"Shop performance last 30 days: {r[0]:,} transactions, "
                        f"{r[1]:,.0f} EGP revenue, {r[2]:,.0f} EGP gross profit, "
                        f"{r[3]:.1f}% average margin, {r[4]:,.0f} kg total sold. "
                        f"أداء آخر 30 يوم: {r[0]:,} معاملة، إيرادات {r[1]:,.0f} جنيه، "
                        f"ربح {r[2]:,.0f} جنيه، هامش {r[3]:.1f}%، وزن {r[4]:,.0f} كجم."
                    ),
                    "metadata": {"category": "live_kpi", "period": "30d", "source": "db"},
                })

            # ── 2. Top 5 selling products ─────────────────────────────────────
            rows = conn.execute(
                "SELECT p.name_ar, p.name_en, SUM(s.total_price) rev, "
                "SUM(s.weight_kg) kg, AVG(s.profit_margin_percent) mg "
                "FROM sales s JOIN products p ON s.product_id=p.id "
                "WHERE s.sale_date >= date('now','-30 days') "
                "GROUP BY s.product_id ORDER BY rev DESC LIMIT 5"
            ).fetchall()
            if rows:
                parts = [
                    f"{r[0]} ({r[1]}): {r[2]:,.0f} EGP, {r[3]:.0f} kg, {r[4]:.0f}% margin"
                    for r in rows
                ]
                docs.append({
                    "id": "live_top_products",
                    "text": (
                        "Top 5 products by revenue last 30 days: " + " | ".join(parts) +
                        ". أعلى 5 منتجات مبيعاً: " + " | ".join(parts)
                    ),
                    "metadata": {"category": "live_products", "source": "db"},
                })

            # ── 3. Low-stock alerts ───────────────────────────────────────────
            low = conn.execute(
                "SELECT p.name_ar, p.name_en, COALESCE(i.current_stock_kg,0) s, "
                "p.reorder_point rp "
                "FROM products p LEFT JOIN inventory i ON p.id=i.product_id "
                "WHERE COALESCE(i.current_stock_kg,0) < p.reorder_point "
                "ORDER BY COALESCE(i.current_stock_kg,0) ASC LIMIT 8"
            ).fetchall()
            if low:
                parts = [f"{r[0]}: {r[2]:.1f}kg (reorder at {r[3]:.0f}kg)" for r in low]
                docs.append({
                    "id": "live_low_stock",
                    "text": (
                        "Products currently below reorder point (need procurement): "
                        + " | ".join(parts) +
                        ". منتجات دون نقطة إعادة الطلب: " + " | ".join(parts)
                    ),
                    "metadata": {"category": "live_inventory", "source": "db"},
                })

            # ── 4. WAC per product (recent purchases) ─────────────────────────
            wac = conn.execute(
                "SELECT product_ar, "
                "ROUND(SUM(total_cost)/NULLIF(SUM(weight_kg),0), 0) wac, "
                "SUM(weight_kg) kg, MAX(purchase_date) last_pur "
                "FROM purchases WHERE purchase_date >= date('now','-30 days') "
                "GROUP BY product_ar ORDER BY kg DESC LIMIT 6"
            ).fetchall()
            if wac:
                parts = [
                    f"{r[0]}: WAC={r[1]:.0f} EGP/kg, {r[2]:.0f}kg bought, last on {r[3]}"
                    for r in wac
                ]
                docs.append({
                    "id": "live_wac_current",
                    "text": (
                        "Current WAC (Weighted Average Cost) by product last 30 days: "
                        + " | ".join(parts)
                    ),
                    "metadata": {"category": "live_wac", "source": "db"},
                })

            # ── 5. Revenue trend (4-week comparison) ─────────────────────────
            wks = conn.execute(
                "SELECT strftime('%W',sale_date) wk, COALESCE(SUM(total_price),0) rev "
                "FROM sales WHERE sale_date >= date('now','-28 days') "
                "GROUP BY wk ORDER BY wk"
            ).fetchall()
            if len(wks) >= 2:
                last_r = wks[-1][1] or 0
                prev_r = wks[-2][1] or 1
                pct    = (last_r - prev_r) / max(prev_r, 1) * 100
                docs.append({
                    "id": "live_revenue_trend",
                    "text": (
                        f"Weekly revenue trend: last week {last_r:,.0f} EGP vs "
                        f"previous week {prev_r:,.0f} EGP ({pct:+.1f}% change). "
                        f"{'Revenue is growing ↑' if pct>0 else 'Revenue is declining ↓'} "
                        f"اتجاه الإيرادات الأسبوعية: {last_r:,.0f} جنيه مقابل {prev_r:,.0f} ({pct:+.1f}%)."
                    ),
                    "metadata": {"category": "live_trend", "source": "db"},
                })

            conn.close()

            # Upsert all live docs
            if docs:
                self._col().upsert(
                    documents=[d["text"]     for d in docs],
                    ids      =[d["id"]       for d in docs],
                    metadatas=[d["metadata"] for d in docs],
                )

            self._last_db_refresh = datetime.now()

        except Exception as exc:
            print(f"[RAG] Live summary refresh error: {exc}")

    # ── Public query API ──────────────────────────────────────────────────────
    def _rerank(self, question: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank candidates using a cross-encoder model.
        Cross-encoders see (query, document) together → much more accurate scoring.

        Optimized: limits to top 5 candidates for speed on CPU.
        """
        if not candidates:
            return candidates

        # Limit candidates for CPU performance
        candidates = candidates[:5]

        try:
            from sentence_transformers import CrossEncoder
            # Try multilingual cross-encoder first (supports Arabic)
            try:
                cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLM-L-6-v2")
            except Exception:
                # Fallback to English (works but may hurt Arabic)
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Score each (query, document) pair
            pairs = [(question, doc["text"]) for doc in candidates]
            scores = cross_encoder.predict(pairs, batch_size=5, show_progress_bar=False)

            # Attach scores and re-sort
            for i, doc in enumerate(candidates):
                doc["rerank_score"] = round(float(scores[i]), 4)

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:top_k]
        except Exception as exc:
            print(f"[RAG] Reranker failed (using hybrid scores): {exc}")
            return candidates[:top_k]

    def _generate_hypothetical_answer(self, question: str, api_key: str = "") -> str:
        """
        HyDE: Generate a hypothetical answer to the question, then use THAT
        for embedding instead of the raw query.

        Why? User questions are short and informal. Documents are longer and
        formal. A hypothetical answer bridges this gap — it "sounds like" a
        document, so its embedding is closer to the real documents.
        """
        if not api_key:
            return question  # No API key — fall back to raw query

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-flash-lite-latest")

            prompt = (
                "You are a knowledge base document writer. "
                "Write a short factual paragraph (2-3 sentences) that answers "
                "the following question. Write it in the same style as a "
                "reference document — formal, informative, and direct.\n\n"
                f"Question: {question}\n\n"
                "Answer paragraph:"
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as exc:
            print(f"[RAG] HyDE generation failed: {exc}")
            return question

    def _expand_arabic_query(self, question: str) -> str:
        """
        Arabic-specific query expansion: adds synonyms and related terms
        to improve retrieval for paraphrased Arabic questions.

        Example: "العجل البلدي بيطلع منه لحم قد ايه" ->
                 "العجل البلدي بيطلع منه لحم قد ايه نسبة استفادة البقري البلدي"
        """
        # Arabic synonym mappings for common butcher shop terms
        arabic_expansions = {
            # Yield terms
            "بيطلع منه لحم": "نسبة استفادة",
            "نفاذته": "نسبة استفادة",
            "قد ايه من كل كيلو": "نسبة استفادة",
            # Margin terms
            "مكسب": "هامش ربح",
            "هامش": "هامش ربح",
            # Demand terms
            "بيبقى اعلي": "مضاعف طلب",
            "بيضرب كام ضعف": "مضاعف طلب",
            # Inventory terms
            "صلاحية": "مدة صلاحية",
            "كام يوم": "مدة صلاحية",
            # Pricing terms
            "سعر": "تسعير",
            "تسعير": "استراتيجية تسعير",
            # Supplier terms
            "مورد": "مورد",
            "موردين": "إدارة موردين",
            # Waste terms
            "هالك": "تقليل هالك",
            "تقليل": "تقليل هالك",
        }

        expanded = question
        for term, expansion in arabic_expansions.items():
            if term in question and expansion not in expanded:
                expanded += " " + expansion

        return expanded

    def _semantic_cache_lookup(self, question: str, threshold: float = 0.92) -> Optional[Dict]:
        """
        Semantic caching: Check if a similar question was already answered.
        Uses ChromaDB to find similar cached queries.

        Returns cached result if similarity >= threshold, else None.
        """
        try:
            col = self._col()
            # Check if cache collection exists
            cache_col = self._client_obj().get_or_create_collection(
                name="butcherai_semantic_cache",
                embedding_function=self._embed(),
                metadata={"hnsw:space": "cosine"},
            )

            if cache_col.count() == 0:
                return None

            results = cache_col.query(
                query_texts=[question],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

            if results["documents"] and results["documents"][0]:
                similarity = 1 - results["distances"][0][0]
                if similarity >= threshold:
                    return {
                        "cached": True,
                        "similarity": round(similarity, 4),
                        "answer": results["documents"][0][0],
                        "metadata": results["metadatas"][0][0] if results["metadatas"][0] else {},
                    }
            return None
        except Exception:
            return None

    def _semantic_cache_store(self, question: str, answer: str):
        """Store a Q&A pair in the semantic cache."""
        try:
            cache_col = self._client_obj().get_or_create_collection(
                name="butcherai_semantic_cache",
                embedding_function=self._embed(),
                metadata={"hnsw:space": "cosine"},
            )
            cache_col.upsert(
                documents=[answer],
                ids=[f"cache_{hash(question) & 0xFFFFFFFF}"],
                metadatas=[{"query": question, "timestamp": str(datetime.now())}],
            )
        except Exception:
            pass

    def query(
        self,
        question: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        refresh_db: bool = True,
        use_hybrid: bool = True,
        use_reranker: bool = False,  # Disabled by default — English cross-encoder hurts Arabic
        api_key: str = "",  # For HyDE query expansion
        use_hyde: bool = False,  # Disabled by default — requires API key
        use_semantic_cache: bool = True,  # Check cache before retrieval
    ) -> List[Dict]:
        """
        Retrieve top-n relevant documents for a question.

        Hybrid search (default): combines vector similarity + BM25 keyword scores.
        Set use_hybrid=False to use vector search only (original behavior).

        Returns list of dicts:
          {text, doc_id, category, relevance_score, source}
        sorted by relevance descending.
        """
        if not self.initialized:
            self.initialize()
        if refresh_db:
            self._refresh_live_summaries()

        # ── Semantic Cache: Check if similar question was already answered ──
        if use_semantic_cache:
            cached = self._semantic_cache_lookup(question)
            if cached:
                return [{
                    "text": cached["answer"],
                    "doc_id": "semantic_cache",
                    "category": "cache",
                    "relevance": cached["similarity"],
                    "source": "cache",
                }]

        # ── HyDE: Generate hypothetical answer for better embedding ──────────
        query_for_embedding = question
        if use_hyde and api_key:
            hypothetical = self._generate_hypothetical_answer(question, api_key)
            if hypothetical and hypothetical != question:
                query_for_embedding = hypothetical

        col = self._col()
        n   = min(n_results, max(1, col.count()))

        kwargs: Dict = {
            "query_texts": [query_for_embedding],
            "n_results"  : n,
            "include"    : ["documents", "metadatas", "distances"],
        }
        if category_filter:
            kwargs["where"] = {"category": category_filter}

        try:
            raw = col.query(**kwargs)
        except Exception as exc:
            print(f"[RAG] Query error: {exc}")
            return []

        # ── Vector search results ──────────────────────────────────────────────
        vector_results = []
        for i in range(len(raw["documents"][0])):
            relevance = round(1 - raw["distances"][0][i], 4)
            if relevance < MIN_RELEVANCE:
                continue
            meta = raw["metadatas"][0][i] or {}
            vector_results.append({
                "text"      : raw["documents"][0][i],
                "doc_id"    : raw["ids"][0][i],
                "category"  : meta.get("category", "general"),
                "relevance" : relevance,
                "source"    : meta.get("source", "unknown"),
            })

        # ── Hybrid fusion: combine vector + BM25 ──────────────────────────────
        if use_hybrid and self._bm25 is not None:
            # Get BM25 scores for more candidates (top 3n to allow fusion + reranking)
            # Use the HyDE-enhanced query for BM25 too (better keyword matching)
            bm25_results = self._bm25.search(query_for_embedding, top_k=n * 3)

            # Normalize BM25 scores to [0, 1]
            if bm25_results:
                max_bm25 = max(score for _, score in bm25_results)
                if max_bm25 > 0:
                    bm25_normalized = {
                        doc_id: score / max_bm25
                        for doc_id, score in bm25_results
                    }
                else:
                    bm25_normalized = {}
            else:
                bm25_normalized = {}

            # Build a lookup for vector results
            vector_lookup = {r["doc_id"]: r for r in vector_results}

            # Combine scores using weighted fusion
            all_doc_ids = set(vector_lookup.keys()) | set(bm25_normalized.keys())
            fused_results = []
            for doc_id in all_doc_ids:
                vec_score = vector_lookup[doc_id]["relevance"] if doc_id in vector_lookup else 0.0
                bm25_score = bm25_normalized.get(doc_id, 0.0)
                # Weighted combination
                fused_score = self._hybrid_weight * vec_score + (1 - self._hybrid_weight) * bm25_score

                if doc_id in vector_lookup:
                    doc_data = vector_lookup[doc_id]
                else:
                    # Doc from BM25 only — fetch from ChromaDB
                    try:
                        doc_raw = col.get(ids=[doc_id], include=["documents", "metadatas"])
                        if doc_raw["documents"]:
                            meta = doc_raw["metadatas"][0] or {}
                            doc_data = {
                                "text": doc_raw["documents"][0],
                                "doc_id": doc_id,
                                "category": meta.get("category", "general"),
                                "relevance": 0.0,
                                "source": meta.get("source", "unknown"),
                            }
                        else:
                            continue
                    except Exception:
                        continue

                doc_data["relevance"] = round(fused_score, 4)
                fused_results.append(doc_data)

            fused_results.sort(key=lambda x: x["relevance"], reverse=True)

            # ── Rerank top candidates with cross-encoder ──────────────────────
            if use_reranker:
                # Rerank top 10, return top n
                rerank_candidates = fused_results[:10]
                reranked = self._rerank(question, rerank_candidates, top_k=n_results)
                return reranked

            return fused_results[:n_results]

        # Vector-only (original behavior)
        vector_results.sort(key=lambda x: x["relevance"], reverse=True)

        # Apply reranker even in vector-only mode
        if use_reranker and vector_results:
            rerank_candidates = vector_results[:10]
            reranked = self._rerank(question, rerank_candidates, top_k=n_results)
            return reranked

        return vector_results

    def build_context_block(self, question: str, n_results: int = 5) -> str:
        """Format retrieved documents into a context block for LLM injection."""
        docs = self.query(question, n_results=n_results)
        if not docs:
            return ""
        parts = []
        for doc in docs:
            parts.append(
                f"[Source: {doc['category']} | Relevance: {doc['relevance']:.2f}]\n"
                f"{doc['text']}"
            )
        return (
            "=== RETRIEVED KNOWLEDGE BASE CONTEXT ===\n"
            + "\n\n".join(parts)
            + "\n=== END CONTEXT ===\n\n"
        )

    def build_augmented_prompt(
        self,
        question: str,
        system_prefix: str = "",
        n_context: int = 5,
    ) -> str:
        """Return a complete prompt: system + RAG context + question."""
        context = self.build_context_block(question, n_results=n_context)
        system  = system_prefix or (
            "You are ButcherAI, a bilingual (Arabic/English) business intelligence "
            "assistant specializing in Egyptian butcher shops. "
            "Always ground your answers in the retrieved context below. "
            "Be specific, practical, and quantitative where possible."
        )
        return f"{system}\n\n{context}User question: {question}\n\nAnswer:"

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> Dict:
        """Return a summary of the collection for display in Streamlit."""
        try:
            count = self._col().count()
            return {
                "status"      : "✅ Ready" if self.initialized else "⚠️ Not initialized",
                "total_docs"  : count,
                "domain_docs" : len(DOMAIN_DOCS),
                "live_docs"   : max(0, count - len(DOMAIN_DOCS)),
                "last_refresh": str(self._last_db_refresh)[:19] if self._last_db_refresh else "Never",
                "embed_model" : EMBED_MODEL,
                "chroma_path" : self.chroma_path,
            }
        except Exception as exc:
            return {"status": f"❌ Error: {exc}", "total_docs": 0}

    def add_custom_doc(self, doc_id: str, text: str, category: str = "custom") -> bool:
        """Let users add their own knowledge to the RAG store."""
        try:
            self._col().upsert(
                documents=[text],
                ids      =[f"user_{doc_id}"],
                metadatas=[{"category": category, "source": "user",
                            "added": str(date.today())}],
            )
            return True
        except Exception:
            return False

    def reset(self) -> str:
        """Wipe and reseed the collection."""
        try:
            client = self._client_obj()
            client.delete_collection(COLLECTION_NAME)
            self._collection      = None
            self._last_db_refresh = None
            self.initialized      = False
            return self.initialize()
        except Exception as exc:
            return f"Reset failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON (import and reuse)
# ─────────────────────────────────────────────────────────────────────────────
_rag: Optional[ButcherAIRAG] = None


def get_rag(db_path: str = DB_PATH) -> ButcherAIRAG:
    """Return (or create) the global singleton RAG engine."""
    global _rag
    if _rag is None:
        _rag = ButcherAIRAG(db_path=db_path)
        _rag.initialize()
    return _rag


def rag_answer(
    question : str,
    api_key  : str = "",
    model    : str = "gemini-flash-lite-latest",
    n_context: int = 5,
    db_path  : str = DB_PATH,
) -> Dict:
    """
    One-call pipeline: retrieve → augment → generate.

    Returns:
      answer       : str   — LLM answer (or best-match document text)
      context_used : list  — documents retrieved
      sources      : list  — [{doc_id, category, relevance}]
    """
    rag     = get_rag(db_path)
    context = rag.query(question, n_results=n_context)
    prompt  = rag.build_augmented_prompt(question, n_context=n_context)

    answer = ""
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            m    = genai.GenerativeModel(model)
            with track_llm_call(feature="rag_api", model=model, db_path=db_path) as tracker:
                resp = m.generate_content(prompt)
                tracker.record_response(resp)
            answer = resp.text.strip()
        except Exception as exc:
            answer = f"[Gemini error: {exc}]"
    else:
        # Fallback: surface the top-ranked document
        if context:
            best   = context[0]
            answer = (
                f"**Domain knowledge [{best['category']}] "
                f"(relevance {best['relevance']:.0%}):**\n\n{best['text']}\n\n"
                "*(Add a Gemini API key for AI-generated answers)*"
            )
        else:
            answer = "No relevant documents found in the knowledge base for this question."

    return {
        "answer"      : answer,
        "context_used": context,
        "sources"     : [
            {"doc_id": d["doc_id"], "category": d["category"], "relevance": d["relevance"]}
            for d in context
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE  — add to app.py sidebar as "📚 قاعدة المعرفة (RAG)"
# ─────────────────────────────────────────────────────────────────────────────
def show_rag_page(db_path: str = DB_PATH, gemini_api_key: str = ""):
    """Complete Streamlit page for RAG knowledge base interaction."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available — run from within the app.")
        return

    st.title("📚 قاعدة المعرفة الذكية (RAG)")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "🔍 النظام يبحث في قاعدة معرفة متخصصة بمحلات اللحوم المصرية + بيانات محلك الحية "
        "قبل توليد أي إجابة — مما يجعل الإجابات دقيقة وموثوقة."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Init ─────────────────────────────────────────────────────────────────
    @st.cache_resource
    def _init_rag():
        engine = ButcherAIRAG(db_path=db_path)
        msg    = engine.initialize()
        return engine, msg

    try:
        rag, init_msg = _init_rag()
    except ImportError as exc:
        st.error(f"❌ Missing dependency: {exc}")
        st.code("pip install chromadb sentence-transformers")
        return

    st.caption(f"🟢 {init_msg}")

    # ── Stats sidebar card ────────────────────────────────────────────────────
    stats = rag.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 إجمالي المستندات",  stats["total_docs"])
    c2.metric("📖 معرفة المجال",       stats["domain_docs"])
    c3.metric("🔴 بيانات حية",         stats["live_docs"])
    c4.metric("🕐 آخر تحديث",         stats.get("last_refresh", "—")[-8:])

    st.divider()
    tabs = st.tabs(["💬 اسأل قاعدة المعرفة", "📋 استعراض المستندات",
                    "➕ إضافة معرفة مخصصة", "⚙️ الإعدادات"])

    # ── Tab 1: Q&A ────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("اسأل أي سؤال عن محلك أو صناعة اللحوم")

        example_qs = [
            "كم يجب أن أشتري من الضاني قبل عيد الأضحى؟",
            "ما هو هامش الربح الصحي للفيليه؟",
            "كيف أحسب التكلفة الموزونة WAC؟",
            "متى يكون أفضل وقت للشراء من الموردين؟",
            "What is the expected demand multiplier for Ramadan?",
        ]
        col_ex = st.columns(len(example_qs))
        for i, q in enumerate(example_qs):
            if col_ex[i].button(f"💡 {q[:30]}…" if len(q) > 30 else f"💡 {q}", key=f"ex_{i}"):
                st.session_state["rag_q"] = q

        question = st.text_area(
            "سؤالك:",
            value=st.session_state.get("rag_q", ""),
            height=80,
            placeholder="اكتب سؤالك بالعربية أو الإنجليزية...",
        )

        col_opts = st.columns([2, 1, 1])
        with col_opts[0]:
            api_key = st.text_input(
                "🔑 Gemini API Key (اختياري):",
                type="password",
                value=gemini_api_key,
                help="بدون مفتاح: يعرض أفضل مستند مطابق. بمفتاح: يولّد إجابة AI.",
            )
        with col_opts[1]:
            n_ctx = st.slider("عدد مستندات السياق:", 1, 8, 4)
        with col_opts[2]:
            cat_filter = st.selectbox(
                "فلتر الفئة:",
                ["الكل", "yield", "seasonal", "financial", "inventory",
                 "supplier", "pricing", "live_kpi", "live_inventory"],
            )

        if st.button("🔍 بحث + إجابة", type="primary") and question.strip():
            with st.spinner("جاري البحث في قاعدة المعرفة..."):
                t0   = time.time()
                filt = None if cat_filter == "الكل" else cat_filter
                # Force live DB refresh
                rag._refresh_live_summaries()
                context_docs = rag.query(question, n_results=n_ctx, category_filter=filt)
                prompt       = rag.build_augmented_prompt(question, n_context=n_ctx)
                elapsed_ret  = round(time.time() - t0, 2)

                answer = ""
                elapsed_gen = 0
                if api_key:
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        m    = genai.GenerativeModel("gemini-flash-lite-latest")
                        t1   = time.time()
                        with track_llm_call(feature="rag_ui", model="gemini-flash-lite-latest", db_path=db_path) as tracker:
                            resp = m.generate_content(prompt)
                            tracker.record_response(resp)
                        answer      = resp.text.strip()
                        elapsed_gen = round(time.time() - t1, 2)
                    except Exception as exc:
                        answer = f"[Gemini error: {exc}]"
                elif context_docs:
                    best   = context_docs[0]
                    answer = (
                        f"**أفضل مستند مطابق** (ملاءمة: {best['relevance']:.0%}):\n\n"
                        f"{best['text']}"
                    )
                else:
                    answer = "لم يتم العثور على معلومات ذات صلة في قاعدة المعرفة."

            # Display answer
            st.subheader("💬 الإجابة")
            st.markdown(answer)

            # Display retrieved context
            st.subheader(f"📄 المستندات المسترجعة ({len(context_docs)})")
            for i, doc in enumerate(context_docs):
                bar_pct = int(doc["relevance"] * 100)
                color   = "#28a745" if doc["relevance"] > 0.7 else \
                          "#ffc107" if doc["relevance"] > 0.4 else "#dc3545"
                with st.expander(
                    f"[{i+1}] {doc['category']} — ملاءمة: {doc['relevance']:.0%} "
                    f"({'domain' if doc['source']=='domain' else '🔴 live'})",
                    expanded=(i == 0),
                ):
                    st.progress(bar_pct)
                    st.write(doc["text"])

            st.caption(
                f"⏱️ Retrieval: {elapsed_ret}s"
                + (f" | Generation: {elapsed_gen}s" if elapsed_gen else "")
            )

    # ── Tab 2: Browse docs ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("📋 استعراض جميع مستندات قاعدة المعرفة")
        st.markdown("**مستندات المجال (ثابتة):**")
        for doc in DOMAIN_DOCS:
            with st.expander(f"📖 {doc['id']} — {doc['metadata']['category']}"):
                st.write(doc["text"])
        st.markdown("**الملخصات الحية من قاعدة البيانات (تتحدث كل ساعة):**")
        try:
            col     = rag._col()
            live_ids = ["live_kpi_30d", "live_top_products", "live_low_stock",
                        "live_wac_current", "live_revenue_trend"]
            existing = col.get(ids=live_ids, include=["documents", "metadatas"])
            for i, doc_id in enumerate(existing["ids"]):
                meta = existing["metadatas"][i] if existing["metadatas"] else {}
                with st.expander(f"🔴 {doc_id} — {meta.get('category', '')}"):
                    st.write(existing["documents"][i])
        except Exception:
            st.info("الملخصات الحية ستظهر بعد أول تهيئة.")

    # ── Tab 3: Add custom doc ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("➕ إضافة معرفة مخصصة لمحلك")
        st.info("أضف معلومات خاصة بمحلك: أسعار مناطق معينة، موردون محليون، قواعد عمل خاصة.")
        with st.form("add_doc_form", clear_on_submit=True):
            doc_id  = st.text_input("معرّف المستند (بالإنجليزية، بدون مسافات):")
            doc_cat = st.selectbox("الفئة:", ["custom", "local_pricing", "local_supplier",
                                              "shop_rules", "market_insight"])
            doc_txt = st.text_area("نص المستند (عربي أو إنجليزي أو كلاهما):", height=150)
            if st.form_submit_button("💾 إضافة للقاعدة", type="primary"):
                if doc_id and doc_txt:
                    ok = rag.add_custom_doc(doc_id, doc_txt, doc_cat)
                    if ok:
                        st.success(f"✅ تمت الإضافة: {doc_id}")
                    else:
                        st.error("❌ فشلت الإضافة.")
                else:
                    st.warning("أدخل معرّف ونص المستند.")

    # ── Tab 4: Settings ───────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("⚙️ إعدادات قاعدة المعرفة")
        st.json(stats)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 تحديث الملخصات الحية الآن"):
                rag._last_db_refresh = None
                rag._refresh_live_summaries()
                st.success("✅ تم تحديث الملخصات الحية.")
        with col2:
            if st.button("♻️ إعادة تهيئة كاملة (مسح وإعادة بناء)", type="secondary"):
                msg = rag.reset()
                st.success(f"✅ {msg}")
