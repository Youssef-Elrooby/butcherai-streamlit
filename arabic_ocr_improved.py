"""
ButcherAI — Improved Arabic OCR
================================
Enhanced OCR pipeline for Arabic handwritten text with:
- Better image preprocessing (denoising, contrast enhancement, deskewing)
- Batch processing for multiple images
- Confidence-based routing (high confidence = auto-accept, low = human review)
- Arabic-specific prompt optimizations
- Caching to avoid redundant API calls

Usage:
    from arabic_ocr_improved import ArabicOCRProcessor
    processor = ArabicOCRProcessor(api_key="your_key")
    result = processor.process_image(image_bytes)
    results = processor.process_batch([image1, image2, image3])
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

DB_PATH = os.environ.get("DB_PATH", "ButcherAI.db")


class ArabicOCRProcessor:
    """
    Improved Arabic OCR with preprocessing, batch processing, and confidence routing.
    """

    def __init__(self, api_key: str, db_path: str = DB_PATH, use_cache: bool = True):
        self.api_key = api_key
        self.db_path = db_path
        self.use_cache = use_cache
        self._cache_table_created = False

    def _ensure_cache_table(self):
        """Create OCR cache table if it doesn't exist."""
        if self._cache_table_created:
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_cache (
                image_hash   TEXT PRIMARY KEY,
                result_json  TEXT NOT NULL,
                timestamp    TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        self._cache_table_created = True

    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Generate a hash for image caching."""
        return hashlib.md5(image_bytes).hexdigest()

    def _check_cache(self, image_hash: str) -> Optional[Dict]:
        """Check if OCR result is cached."""
        if not self.use_cache:
            return None
        try:
            self._ensure_cache_table()
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT result_json FROM ocr_cache WHERE image_hash = ?",
                (image_hash,),
            ).fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
        return None

    def _store_cache(self, image_hash: str, result: Dict):
        """Store OCR result in cache."""
        if not self.use_cache:
            return
        try:
            self._ensure_cache_table()
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO ocr_cache (image_hash, result_json, timestamp) "
                "VALUES (?, ?, ?)",
                (image_hash, json.dumps(result, ensure_ascii=False), datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Enhanced image preprocessing for Arabic handwriting:
        - Noise reduction
        - Contrast enhancement
        - Deskewing
        - Paper region isolation
        """
        try:
            import cv2
            import numpy as np

            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return image_bytes

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Deskewing
            coords = np.column_stack(np.where(enhanced > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:  # Only deskew if significant skew
                (h, w) = enhanced.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(enhanced, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = enhanced

            # Paper region isolation
            _, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                total_area = rotated.shape[0] * rotated.shape[1]

                if 0.05 < area < 0.95 * total_area:
                    x, y, w, h = cv2.boundingRect(largest)
                    pad = 15
                    x, y = max(0, x - pad), max(0, y - pad)
                    w = min(rotated.shape[1] - x, w + 2 * pad)
                    h = min(rotated.shape[0] - y, h + 2 * pad)
                    cropped = rotated[y:y + h, x:x + w]
                else:
                    cropped = rotated
            else:
                cropped = rotated

            # Encode back to JPEG
            success, encoded = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                return encoded.tobytes()
            return image_bytes

        except Exception:
            # Fallback to original image
            return image_bytes

    def extract_notebook_lines(self, image_bytes: bytes) -> Dict:
        """
        Extract handwritten Arabic notebook lines with improved preprocessing.
        """
        if not self.api_key:
            return {"success": False, "lines": [], "error": "No API key provided", "elapsed_sec": 0.0}

        # Check cache
        image_hash = self._get_image_hash(image_bytes)
        cached = self._check_cache(image_hash)
        if cached:
            cached["from_cache"] = True
            return cached

        t0 = time.time()

        try:
            from gemini_client import generate_text

            processed_bytes = self.preprocess_image(image_bytes)
            img_b64 = base64.b64encode(processed_bytes).decode("utf-8")

            prompt = """You are reading a handwritten notebook page from an Egyptian butcher shop.
The page may contain Arabic text (product names), numbers (weights in kg, prices in EGP),
and dates, and may use Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) or Western digits (0123456789).

Extract EVERY line of handwritten content you can identify. For each line:
- raw_text: transcribe exactly what you see, in the original script
- product_guess: your best guess at the product name (Arabic), empty string if unclear
- weight_guess: your best guess at a weight number in kg, as a plain number, null if unclear
- price_guess: your best guess at a price number in EGP, as a plain number, null if unclear
- date_guess: if this line contains a date, extract it as DD/MM/YYYY or DD/MM format
- confidence: "high", "medium", or "low" based on how legible the handwriting is

Some lines may just be numbers with no clear product — include them anyway with product_guess as empty string.

Output ONLY valid JSON in this exact format, nothing else:
{"lines": [{"raw_text": "...", "product_guess": "...", "weight_guess": 0.0, "price_guess": 0.0, "date_guess": "", "confidence": "high"}]}"""

            # Use the improved gemini_client with circuit breaker
            result = generate_text(
                prompt=prompt,
                api_key=self.api_key,
                system_instruction="You are an expert Arabic handwriting OCR system for Egyptian butcher shops.",
                temperature=0.1,
                feature="ocr_notebook",
                db_path=self.db_path,
            )

            if result.get("error"):
                return {
                    "success": False, "lines": [],
                    "error": result["error"],
                    "elapsed_sec": round(time.time() - t0, 2),
                }

            # Parse the response
            text = result["text"].strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            lines = parsed.get("lines", [])

            result_data = {
                "success": True,
                "lines": lines,
                "error": None,
                "elapsed_sec": round(time.time() - t0, 2),
                "from_cache": False,
            }

            # Store in cache
            self._store_cache(image_hash, result_data)

            return result_data

        except json.JSONDecodeError as exc:
            return {
                "success": False, "lines": [],
                "error": f"Malformed JSON: {exc}",
                "elapsed_sec": round(time.time() - t0, 2),
            }
        except Exception as exc:
            return {
                "success": False, "lines": [],
                "error": f"OCR error: {exc}",
                "elapsed_sec": round(time.time() - t0, 2),
            }

    def extract_carcass_slip(self, image_bytes: bytes) -> Dict:
        """
        Extract cattle weighing slip data with improved preprocessing.
        """
        if not self.api_key:
            return {"success": False, "carcasses": [], "error": "No API key provided", "elapsed_sec": 0.0}

        # Check cache
        image_hash = self._get_image_hash(image_bytes)
        cached = self._check_cache(image_hash)
        if cached:
            cached["from_cache"] = True
            return cached

        t0 = time.time()

        try:
            from gemini_client import generate_text

            processed_bytes = self.preprocess_image(image_bytes)
            img_b64 = base64.b64encode(processed_bytes).decode("utf-8")

            prompt = """You are reading a handwritten CATTLE WEIGHING SLIP (قسيمة موازين عجول)
from an Egyptian butcher shop supplier.

Each animal on the page is represented by exactly TWO handwritten numbers
that must be SUMMED to get the total bone-in carcass weight in kg.

- Numbers may use Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) or Western digits.
- A name at the bottom is usually the BUYER, not the supplier.
- There is NO product name and NO price on this document type.
- A phone number (10-11 digits) is NOT a weight — exclude it.

For each pair of weight numbers:
- weight_1: the first number in the pair
- weight_2: the second number in the pair
- confidence: "high", "medium", or "low"

Output ONLY valid JSON:
{"carcasses": [{"weight_1": 126.0, "weight_2": 118.0, "confidence": "high"}], "supplier_name_guess": ""}"""

            result = generate_text(
                prompt=prompt,
                api_key=self.api_key,
                system_instruction="You are an expert Arabic handwriting OCR system for Egyptian butcher shop weighing slips.",
                temperature=0.1,
                feature="ocr_carcass",
                db_path=self.db_path,
            )

            if result.get("error"):
                return {
                    "success": False, "carcasses": [],
                    "error": result["error"],
                    "elapsed_sec": round(time.time() - t0, 2),
                }

            text = result["text"].strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            carcasses_raw = parsed.get("carcasses", [])

            carcasses = []
            for c in carcasses_raw:
                w1 = self._parse_number(c.get("weight_1"))
                w2 = self._parse_number(c.get("weight_2"))
                total = (w1 + w2) if (w1 is not None and w2 is not None) else None
                carcasses.append({
                    "weight_1": w1, "weight_2": w2, "total_weight": total,
                    "confidence": c.get("confidence", "medium"),
                })

            result_data = {
                "success": True,
                "carcasses": carcasses,
                "supplier_name_guess": parsed.get("supplier_name_guess", ""),
                "error": None,
                "elapsed_sec": round(time.time() - t0, 2),
                "from_cache": False,
            }

            self._store_cache(image_hash, result_data)
            return result_data

        except Exception as exc:
            return {
                "success": False, "carcasses": [],
                "error": str(exc),
                "elapsed_sec": round(time.time() - t0, 2),
            }

    def process_batch(self, images: List[bytes], doc_type: str = "notebook") -> List[Dict]:
        """
        Process multiple images in batch.

        Args:
            images: List of image bytes
            doc_type: "notebook" or "carcass"

        Returns:
            List of result dicts
        """
        results = []
        for img in images:
            if doc_type == "notebook":
                result = self.extract_notebook_lines(img)
            else:
                result = self.extract_carcass_slip(img)
            results.append(result)
        return results

    def get_confidence_summary(self, result: Dict) -> Dict:
        """
        Analyze confidence levels and recommend action.

        Returns:
            {"action": "auto_accept"|"review"|"reject", "high": n, "medium": n, "low": n}
        """
        if not result.get("success"):
            return {"action": "reject", "high": 0, "medium": 0, "low": 0}

        lines = result.get("lines", []) or result.get("carcasses", [])
        high = sum(1 for l in lines if l.get("confidence") == "high")
        medium = sum(1 for l in lines if l.get("confidence") == "medium")
        low = sum(1 for l in lines if l.get("confidence") == "low")

        total = high + medium + low
        if total == 0:
            return {"action": "review", "high": 0, "medium": 0, "low": 0}

        # Routing logic
        if low == 0 and medium <= 1:
            action = "auto_accept"
        elif low <= 2:
            action = "review"
        else:
            action = "reject"

        return {"action": action, "high": high, "medium": medium, "low": low}

    @staticmethod
    def _parse_number(value) -> Optional[float]:
        """Safely parse a number, handling Arabic-Indic digits."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            translation = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
            cleaned = str(value).translate(translation)
            cleaned = re.sub(r"[^\d.]", "", cleaned)
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_improved_ocr_page(
    prods_fn, ins_sale_fn, ins_pur_fn, log_audit_fn,
    current_user: str, gemini_api_key: str = "",
):
    """Streamlit page for the improved Arabic OCR."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("استيراد من الدفتر الورقي — محسّن")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "OCR محسّن مع معالجة صورة متقدمة، تخزين مؤقت، وتوجيه بالثقة."
        "</div>",
        unsafe_allow_html=True,
    )

    api_key = st.text_input("Gemini API Key:", type="password", value=gemini_api_key)
    doc_type = st.selectbox("نوع المستند:", ["notebook", "carcass"])
    uploaded = st.file_uploader("ارفع صورة:", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("ارفع صورة لبدء القراءة.")
        return

    st.image(uploaded, caption="الصورة المرفوعة", use_container_width=True)

    if st.button("قراءة الصورة", type="primary"):
        if not api_key:
            st.error("أدخل Gemini API Key أولاً.")
            return

        processor = ArabicOCRProcessor(api_key=api_key)
        image_bytes = uploaded.getvalue()

        with st.spinner("جاري تحليل الصورة..."):
            if doc_type == "notebook":
                result = processor.extract_notebook_lines(image_bytes)
            else:
                result = processor.extract_carcass_slip(image_bytes)

        if not result["success"]:
            st.error(f"فشلت القراءة: {result['error']}")
            return

        # Show confidence summary
        summary = processor.get_confidence_summary(result)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Confidence", summary["high"])
        c2.metric("Medium Confidence", summary["medium"])
        c3.metric("Low Confidence", summary["low"])
        c4.metric("Action", summary["action"])

        if summary["action"] == "auto_accept":
            st.success("جودة عالية — يمكن القبول التلقائي")
        elif summary["action"] == "review":
            st.warning("تحتاج مراجعة — بعض الأسطر منخفضة الثقة")
        else:
            st.error("جودة منخفضة — يُنصح بإعادة التصوير")

        if result.get("from_cache"):
            st.info("النتيجة من التخزين المؤقت")

        st.success(f"تم العثور على {len(result.get('lines', result.get('carcasses', [])))} عنصر ({result['elapsed_sec']}s)")

        # Show extracted data
        if doc_type == "notebook" and result.get("lines"):
            import pandas as pd
            df = pd.DataFrame(result["lines"])
            st.dataframe(df, use_container_width=True)
        elif doc_type == "carcass" and result.get("carcasses"):
            import pandas as pd
            df = pd.DataFrame(result["carcasses"])
            st.dataframe(df, use_container_width=True)
