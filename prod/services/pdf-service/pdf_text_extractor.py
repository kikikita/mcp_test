"""
pdf_text_extractor.py

Извлечение текста по страницам из PDF (цифровые и сканированные).
Возвращает список словарей вида:
  [{"page": 1, "source": "digital"|"scan", "text": "..."}, ...]

Поддерживаемые OCR: PaddleOCR (рекомендуется) и Tesseract.
"""

from __future__ import annotations
import os
import re
from typing import List, Dict, Optional
import json

# Опциональные библиотеки (проверяются при вызове)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None

import numpy as np


# ---------------------- Исключения ----------------------
class DocumentParsingError(Exception):
    pass


class OCRConfigError(Exception):
    pass


# ---------------------- Утилиты ----------------------
def _normalize_whitespace(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace('\xa0', ' ')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    lines = [ln.strip() for ln in s.split('\n')]
    # trim empty head/tail lines
    while lines and lines[0] == '':
        lines.pop(0)
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _is_texty(s: Optional[str], min_chars: int = 400) -> bool:
    """Страница считается цифровой, если после сокращения пробелов длина >= min_chars."""
    if not s:
        return False
    s2 = re.sub(r"\s+", " ", s).strip()
    return len(s2) >= min_chars


# ---------------------- OCR-обёртки ----------------------
def _ocr_with_tesseract(pil_img: 'Image.Image', lang: str = "rus+eng") -> str:
    if pytesseract is None:
        raise OCRConfigError("pytesseract не установлен. Установите pytesseract и tesseract-ocr в систему.")
    # pytesseract принимает PIL.Image
    return pytesseract.image_to_string(pil_img, lang=lang)


def _extract_text_from_paddle_raw(raw) -> str:
    """
    Парсер для вывода PaddleOCR — поддерживает несколько форматов, которые встречаются в разных версиях:
      - [[(box, (text, score)), ...]]  (список, первая вложенность — lines)
      - [[ [box, text, score], ... ], ...]
      - [{ 'rec_texts': [...], 'rec_boxes': [...] }, ...]
    Возвращает объединённый текст (строки через '\n').
    """
    if not raw:
        return ""

    texts: List[str] = []

    try:
        # случай: raw == [{ 'rec_texts': [...], 'rec_boxes': [...] }, ...]
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            for page_block in raw:
                rec_texts = page_block.get('rec_texts') or page_block.get('rec_texts_list') or page_block.get('texts') or []
                if rec_texts:
                    for t in rec_texts:
                        if t:
                            texts.append(str(t))
            return '\n'.join(_normalize_whitespace(t) for t in texts if t)

        # случай: raw == [[(box,(text,score)), ...]] или [[ [box, text, score], ... ]]
        for item in raw:
            if isinstance(item, (list, tuple)):
                for det in item:
                    try:
                        if isinstance(det, (list, tuple)) and len(det) >= 2:
                            second = det[1]
                            if isinstance(second, (list, tuple)):
                                text_candidate = second[0]
                                if text_candidate:
                                    texts.append(str(text_candidate))
                            else:
                                texts.append(str(second))
                        else:
                            if isinstance(det, dict):
                                t = det.get('text') or det.get('rec_text') or det.get('rec_texts')
                                if t:
                                    texts.append(str(t))
                    except Exception:
                        try:
                            s = str(det)
                            texts.append(s)
                        except Exception:
                            continue
            else:
                try:
                    s = str(item)
                    texts.append(s)
                except Exception:
                    continue

    except Exception:
        pass

    cleaned = [_normalize_whitespace(t) for t in texts if t and str(t).strip()]
    return '\n'.join(cleaned)


# ---------------------- Рендеринг страниц ----------------------
def _render_page_with_fitz(path: str, page_index: int, dpi: int = 300) -> 'Image.Image':
    """Рендерит указанную страницу (1-based index) через PyMuPDF и возвращает PIL.Image."""
    if fitz is None:
        raise OCRConfigError("PyMuPDF (fitz) не установлен. Установите pymupdf или используйте другую схему рендеринга.")
    doc = fitz.open(path)
    if page_index < 1 or page_index > len(doc):
        raise DocumentParsingError(f"Неверный индекс страницы: {page_index}")
    page = doc.load_page(page_index - 1)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# ---------------------- Основная функция ----------------------
def extract_pdf_text(
    path: str,
    min_chars_for_digital: int = 400,
    ocr_engine: str = "tesseract",
    ocr_lang: str = "rus+eng",
    dpi: int = 300,
    return_format: str = "dict"  # "dict" или "list"
) -> List:
    """
    Вернуть массив текста по страницам.

    - return_format:
        "dict" -> [{"page": N, "source": "digital"|"scan", "text": "..."}]
        "list" -> ["...", "...", ...]
    """
    if not os.path.exists(path):
        raise DocumentParsingError(f"Файл не найден: {path}")

    if pdfplumber is None:
        raise OCRConfigError("pdfplumber не установлен. Установите pdfplumber через pip.")

    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            if n_pages == 0:
                raise DocumentParsingError("PDF не содержит страниц.")
            extracted_texts = []
            texty_count = 0
            for page in pdf.pages:
                try:
                    raw = page.extract_text() or ""
                except Exception:
                    raw = ""
                extracted_texts.append(raw)
                if _is_texty(raw, min_chars=min_chars_for_digital):
                    texty_count += 1
    except Exception as e:
        raise DocumentParsingError(f"Ошибка при чтении PDF через pdfplumber: {e}")

    texty_ratio = texty_count / max(1, n_pages)
    is_digital = texty_ratio >= 0.6

    results: List[Dict[str, str]] = []

    if is_digital:
        for i, raw in enumerate(extracted_texts, start=1):
            normalized = _normalize_whitespace(raw)
            results.append({"page": i, "source": "digital", "text": normalized})
    else:
        if ocr_engine not in ("paddle", "tesseract"):
            raise OCRConfigError("ocr_engine должен быть 'paddle' или 'tesseract'.")

        if ocr_engine == "paddle" and PaddleOCR is None:
            raise OCRConfigError("PaddleOCR не установлен. Установите paddleocr/paddlepaddle или переключитесь на tesseract.")
        if ocr_engine == "tesseract" and pytesseract is None:
            raise OCRConfigError("pytesseract не установлен. Установите pytesseract и tesseract-ocr в систему.")

        try:
            pil_images = []
            if fitz is not None:
                doc = fitz.open(path)
                for p in range(len(doc)):
                    pil_images.append(_render_page_with_fitz(path, p + 1, dpi=dpi))
                doc.close()
            else:
                raise OCRConfigError("PyMuPDF (fitz) отсутствует. Установите pymupdf чтобы рендерить страницы.")
        except Exception as e:
            raise DocumentParsingError(f"Не удалось рендерить PDF: {e}")

        paddle = None
        if ocr_engine == "paddle":
            paddle = PaddleOCR(lang=ocr_lang, use_angle_cls=True, det_db_unclip_ratio=1.5)

        for idx, pil_img in enumerate(pil_images, start=1):
            page_text = ""
            if ocr_engine == "paddle":
                try:
                    raw = paddle.ocr(np.asarray(pil_img))
                    page_text = _extract_text_from_paddle_raw(raw)
                except Exception:
                    page_text = ""
            else:
                try:
                    page_text = _ocr_with_tesseract(pil_img, lang=ocr_lang)
                except Exception:
                    page_text = ""

            normalized = _normalize_whitespace(page_text)
            results.append({"page": idx, "source": "scan", "text": normalized})

    # Преобразование результата в нужный формат
    if return_format == "list":
        return [item["text"] for item in results]
    return results