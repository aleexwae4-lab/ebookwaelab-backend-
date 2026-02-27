from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PyPDF2 import PdfReader
from ebooklib import epub
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw, ImageFont

import markdown as md

from app.utils import (
    compact_whitespace,
    guess_is_scanned,
    style_to_css_class,
    try_ocr_pdf,
)

# IA: OpenAI + Gemini (opcionales según API keys)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


# -----------------------------
# PDF -> texto limpio (con OCR opcional)
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    pages = len(reader.pages)

    texts: List[str] = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")

    raw = compact_whitespace("\n\n".join(texts))

    if guess_is_scanned(raw):
        ocr = try_ocr_pdf(pdf_path)
        if ocr and len(ocr.strip()) > len(raw.strip()):
            return {"text": compact_whitespace(ocr), "ocr_used": True, "pages": pages}

    return {"text": raw, "ocr_used": False, "pages": pages}


# -----------------------------
# IA: texto limpio -> estructura JSON (Markdown interno)
# -----------------------------

def ai_organize_ebook(
    raw_text: str,
    style: str,
    openai_model: str,
    gemini_model: str,
    title_hint: Optional[str],
    author_hint: Optional[str],
    language: str = "es",
) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return _fallback_structure(style, title_hint, author_hint, language)

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # Sin OpenAI: fallback (pero no rompe)
    if not openai_key or OpenAI is None:
        return _fallback_structure_from_text(raw_text, style, title_hint, author_hint, language)

    base = _openai_structure(raw_text, style, openai_model, title_hint, author_hint, language)

    # Refinado opcional con Gemini
    if gemini_key and genai is not None:
        try:
            refined = _gemini_refine_structure(base, style, gemini_model, language)
            if refined and isinstance(refined, dict) and refined.get("chapters"):
                base = refined
        except Exception:
            pass

    return _normalize_structure(base, style, title_hint, author_hint, language)


def _openai_structure(
    raw_text: str,
    style: str,
    model: str,
    title_hint: Optional[str],
    author_hint: Optional[str],
    language: str,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system = (
        "Eres un editor premium de ebooks para Amazon KDP. "
        "Devuelves SIEMPRE JSON estricto (sin markdown, sin texto extra). "
        "El contenido interno debe ser Markdown (no HTML)."
    )

    prompt = f"""
Idioma: {language}
Estilo: {style}

Objetivo:
1) Limpia el texto (basura OCR, repetición, encabezados inútiles).
2) Organiza en capítulos y secciones con títulos claros.
3) Produce Markdown premium (párrafos, listas, tablas si aplica, bloques de código cuando corresponda).

Salida JSON estricta con este esquema exacto:
{{
  "title": "...",
  "author": "...",
  "language": "{language}",
  "isbn": null,
  "publish_date": null,
  "chapters": [
    {{
      "id": "cap-1",
      "title": "...",
      "sections": [
        {{"title": "...", "markdown": "..."}}
      ]
    }}
  ]
}}

Notas:
- Si no sabes el autor, usa: "EbookWaeLab".
- Si no sabes el título, propón uno profesional según el contenido.
- Mantén el ebook coherente y útil. Tono acorde al estilo.
- NO metas HTML dentro del Markdown.

Hints:
- title_hint: {title_hint}
- author_hint: {author_hint}

TEXTO (recortado si es enorme):
{raw_text[:140000]}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )

    content = resp.choices[0].message.content or ""
    data = _safe_json_loads(content)
    return data or _fallback_structure_from_text(raw_text, style, title_hint, author_hint, language)


def _gemini_refine_structure(structure: Dict[str, Any], style: str, model: str, language: str) -> Dict[str, Any]:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gm = genai.GenerativeModel(model)

    prompt = f"""
Idioma: {language}
Estilo: {style}

Tarea:
- Recibes un JSON de ebook con Markdown interno.
- Refina: claridad, coherencia, mejores títulos, mejor orden.
- No cambies el esquema.
- Devuelve SOLO JSON estricto.

JSON:
{json.dumps(structure, ensure_ascii=False)}
"""

    out = gm.generate_content(prompt)
    txt = getattr(out, "text", "") or ""
    data = _safe_json_loads(txt)
    return data or structure


# -----------------------------
# Preview HTML (Markdown -> HTML + plantilla)
# -----------------------------

def render_preview_html(structure: Dict[str, Any], style: str) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("preview_template.html")

    title = structure.get("title") or "Ebook"
    author = structure.get("author") or "EbookWaeLab"

    chapters = []
    for ch in (structure.get("chapters") or []):
        sections = []
        for s in (ch.get("sections") or []):
            md_txt = s.get("markdown") or ""
            html = markdown_to_html(md_txt)
            sections.append({"title": s.get("title") or "", "html": html})
        chapters.append({"id": ch.get("id"), "title": ch.get("title"), "sections": sections})

    toc = [{"chapter_id": c["id"], "chapter_title": c["title"]} for c in chapters if c.get("id")]

    return template.render(
        title=title,
        author=author,
        style=style,
        css_class=style_to_css_class(style),
        chapters=chapters,
        toc=toc,
    )


def markdown_to_html(markdown_text: str) -> str:
    markdown_text = (markdown_text or "").strip()
    return md.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "codehilite"],
        output_format="html5",
    )


# -----------------------------
# EPUB (KDP): Markdown -> XHTML
# -----------------------------

def generate_epub_from_structure(
    structure: Dict[str, Any],
    style: str,
    output_path: str,
    isbn: Optional[str] = None,
    publish_date: Optional[str] = None,
) -> None:
    book = epub.EpubBook()

    title = (structure.get("title") or "Ebook").strip()
    author = (structure.get("author") or "EbookWaeLab").strip()
    language = (structure.get("language") or "es").strip()

    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language(language)
    book.add_author(author)

    if isbn:
        book.add_metadata("DC", "identifier", isbn, {"id": "isbn"})
    if publish_date:
        book.add_metadata("DC", "date", publish_date)

    # Portada (IA activada por defecto) con fallback seguro
    cover_bytes = generate_cover_image_bytes(title=title, author=author, style=style)
    book.set_cover("cover.jpg", cover_bytes)

    css = _epub_css_for_style(style)
    style_item = epub.EpubItem(uid="style", file_name="style/style.css", media_type="text/css", content=css)
    book.add_item(style_item)

    epub_items: List[epub.EpubHtml] = []

    for idx, ch in enumerate(structure.get("chapters") or [], start=1):
        ch_id = ch.get("id") or f"cap-{idx}"
        ch_title = ch.get("title") or f"Capítulo {idx}"

        body_parts = [f"<h1>{_escape_html(ch_title)}</h1>"]
        for s in (ch.get("sections") or []):
            st = s.get("title") or ""
            if st:
                body_parts.append(f"<h2>{_escape_html(st)}</h2>")
            body_parts.append(markdown_to_html(s.get("markdown") or ""))

        html_doc = f"""
        <html xmlns=\"http://www.w3.org/1999/xhtml\">
          <head>
            <meta charset=\"utf-8\"/>
            <title>{_escape_html(ch_title)}</title>
            <link rel=\"stylesheet\" type=\"text/css\" href=\"../style/style.css\"/>
          </head>
          <body class=\"{style_to_css_class(style)}\">
            {''.join(body_parts)}
          </body>
        </html>
        """

        item = epub.EpubHtml(title=ch_title, file_name=f"text/{ch_id}.xhtml", lang=language)
        item.content = html_doc.encode("utf-8")
        item.add_item(style_item)
        book.add_item(item)
        epub_items.append(item)

    book.toc = tuple(epub_items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_items

    epub.write_epub(output_path, book, {})


# -----------------------------
# Portada IA (por defecto ENABLE_AI_COVER=true)
# -----------------------------

def generate_cover_image_bytes(title: str, author: str, style: str) -> bytes:
    # Activada por defecto si no existe variable
    enable_ai = os.getenv("ENABLE_AI_COVER", "true").lower() == "true"

    bg = None
    if enable_ai:
        bg = _try_ai_cover_background(style=style, title=title)

    if bg is None:
        bg = _pillow_background(style=style)

    img = bg.convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 92)
        author_font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except Exception:
        title_font = ImageFont.load_default()
        author_font = ImageFont.load_default()

    W, H = img.size
    margin = 110

    title_lines = _wrap_text(draw, title, title_font, max_width=W - 2 * margin)
    y = int(H * 0.22)

    # sombra suave
    for dx, dy in [(3, 3), (2, 2)]:
        _draw_multiline(draw, title_lines, (margin + dx, y + dy), title_font, fill=(0, 0, 0))

    _draw_multiline(draw, title_lines, (margin, y), title_font, fill=(245, 245, 245))

    author_y = int(H * 0.80)
    draw.text((margin, author_y), f"por {author}", font=author_font, fill=(235, 235, 235))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


def _try_ai_cover_background(style: str, title: str) -> Optional[Image.Image]:
    """Intenta generar fondo de portada usando GPT (imágenes) y, si hay GEMINI, mejora el prompt."""

    # 1) Construye prompt (Gemini opcional)
    prompt = (
        f"Premium ebook cover background, no text, abstract, modern, high-end, "
        f"style '{style}', subtle gradients, professional 2D design, "
        f"for a Spanish ebook titled '{title}'."
    )

    if os.getenv("GEMINI_API_KEY") and genai is not None:
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gm = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
            refine = gm.generate_content(
                "Crea un prompt breve (1-2 frases) para generar un fondo de portada premium (SIN TEXTO). "
                f"Estilo: {style}. Título (solo referencia, no debe aparecer): {title}."
            )
            txt = (getattr(refine, "text", "") or "").strip()
            if 20 < len(txt) < 500:
                prompt = txt
        except Exception:
            pass

    # 2) Genera imagen con OpenAI (si hay key)
    if not os.getenv("OPENAI_API_KEY") or OpenAI is None:
        return None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # La API de imágenes puede variar por versión; si falla, devolvemos None.
    try:
        res = client.images.generate(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=prompt,
            size="1024x1536",
        )
        b64 = res.data[0].b64_json
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def _pillow_background(style: str) -> Image.Image:
    W, H = 1024, 1536
    img = Image.new("RGB", (W, H), (20, 20, 24))
    draw = ImageDraw.Draw(img)

    palettes = {
        "minimalista": ((245, 245, 245), (210, 210, 210)),
        "académico": ((18, 34, 64), (80, 130, 180)),
        "narrativo": ((70, 24, 20), (190, 120, 70)),
        "técnico": ((10, 20, 24), (0, 170, 160)),
    }
    c1, c2 = palettes.get(style, palettes["académico"])

    for y in range(H):
        t = y / (H - 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    draw.ellipse((W * 0.55, H * 0.05, W * 1.15, H * 0.65), fill=(255, 255, 255, 30))
    draw.ellipse((W * -0.15, H * 0.55, W * 0.55, H * 1.15), fill=(255, 255, 255, 18))
    return img


def _epub_css_for_style(style: str) -> bytes:
    base = """
    body { font-family: serif; line-height: 1.55; padding: 0 0.6rem; }
    h1 { font-size: 1.6em; margin: 1.2em 0 0.4em; }
    h2 { font-size: 1.2em; margin: 1.0em 0 0.2em; }
    p  { margin: 0.6em 0; }
    ul { margin: 0.6em 0 0.6em 1.2em; }
    pre { background: #111; color: #eee; padding: 0.8em; overflow-x: auto; }
    code { font-family: monospace; }
    """

    themed = {
        "minimalista": "body { color:#111; background:#fff; } h1,h2{letter-spacing:0.2px;}",
        "académico": "body { color:#111; } p{ text-align: justify; }",
        "narrativo": "body { color:#1a1a1a; } p{ font-size: 1.02em; }",
        "técnico": "h1,h2{font-family: sans-serif;} code{font-size:0.95em;}",
    }.get(style, "")

    return (base + themed).encode("utf-8")


# -----------------------------
# Fallbacks + helpers
# -----------------------------

def _fallback_structure(style: str, title: Optional[str], author: Optional[str], language: str) -> Dict[str, Any]:
    return {
        "title": (title or "Ebook").strip(),
        "author": (author or "EbookWaeLab").strip(),
        "language": language,
        "isbn": None,
        "publish_date": None,
        "chapters": [
            {
                "id": "cap-1",
                "title": "Capítulo 1",
                "sections": [
                    {"title": "Contenido", "markdown": "No se pudo procesar con IA (faltan API keys)."}
                ],
            }
        ],
    }


def _fallback_structure_from_text(
    raw_text: str, style: str, title: Optional[str], author: Optional[str], language: str
) -> Dict[str, Any]:
    snippet = compact_whitespace(raw_text)[:6000]
    return {
        "title": (title or "Ebook").strip(),
        "author": (author or "EbookWaeLab").strip(),
        "language": language,
        "isbn": None,
        "publish_date": None,
        "chapters": [
            {
                "id": "cap-1",
                "title": "Capítulo 1",
                "sections": [{"title": "Texto base", "markdown": snippet}],
            }
        ],
    }


def _normalize_structure(
    data: Dict[str, Any],
    style: str,
    title_hint: Optional[str],
    author_hint: Optional[str],
    language: str,
) -> Dict[str, Any]:
    data = dict(data or {})

    data["language"] = (data.get("language") or language).strip()
    data["title"] = (data.get("title") or title_hint or "Ebook").strip()
    data["author"] = (data.get("author") or author_hint or "EbookWaeLab").strip()

    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        chapters = _fallback_structure(style, data["title"], data["author"], data["language"])["chapters"]

    for i, ch in enumerate(chapters, start=1):
        if not isinstance(ch, dict):
            continue
        ch.setdefault("id", f"cap-{i}")
        ch.setdefault("title", f"Capítulo {i}")
        ch.setdefault("sections", [])
        if not isinstance(ch["sections"], list):
            ch["sections"] = []
        for s in ch["sections"]:
            if not isinstance(s, dict):
                continue
            s.setdefault("title", "")
            s.setdefault("markdown", "")

    data["chapters"] = chapters
    data.setdefault("isbn", None)
    data.setdefault("publish_date", None)
    return data


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []

    for w in words:
        test = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]

    if cur:
        lines.append(" ".join(cur))

    return lines[:5]


def _draw_multiline(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    xy: Tuple[int, int],
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 10
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=font, fill=fill)
