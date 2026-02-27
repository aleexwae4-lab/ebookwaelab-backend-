from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Sanitiza nombres de archivo (evita rutas y caracteres raros)."""
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r"\s+", " ", name).strip()

    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9._ -]+", "", normalized)
    normalized = normalized.replace(" ", "_")
    return normalized or "file"


def safe_join(base_dir: str, filename: str) -> str:
    """Previene path traversal, fuerza a que el archivo viva dentro de base_dir."""
    base_dir = os.path.abspath(base_dir)
    full = os.path.abspath(os.path.join(base_dir, filename))
    if not (full == base_dir or full.startswith(base_dir + os.sep)):
        raise ValueError("Ruta insegura detectada")
    return full


def compact_whitespace(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def style_to_css_class(style: str) -> str:
    mapping = {
        "minimalista": "style-minimal",
        "académico": "style-academic",
        "narrativo": "style-narrative",
        "técnico": "style-technical",
    }
    return mapping.get(style, "style-academic")


def guess_is_scanned(extracted_text: str, min_chars: int = 800) -> bool:
    return len((extracted_text or "").strip()) < min_chars


def try_ocr_pdf(pdf_path: str, lang: str = "spa+eng") -> Optional[str]:
    """OCR opcional (no rompe si faltan dependencias del sistema).

    Requiere (cuando se usa):
    - tesseract instalado en el sistema
    - pdf2image + poppler
    """
    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        return None

    try:
        images = convert_from_path(pdf_path, dpi=220)
    except Exception:
        return None

    out = []
    for img in images:
        try:
            out.append(pytesseract.image_to_string(img, lang=lang))
        except Exception:
            continue

    text = "\n\n".join(out).strip()
    return text or None
