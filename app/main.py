from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.utils import ensure_dirs, safe_join, sanitize_filename, now_utc_iso
from app.processor import (
    extract_text_from_pdf,
    ai_organize_ebook,
    render_preview_html,
    generate_epub_from_structure,
)

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # dotenv es opcional; en Railway/Railwail suele usarse panel de variables
    pass

APP_NAME = "EbookWaeLab Premium SaaS API"

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

ALLOWED_STYLES = {"minimalista", "académico", "narrativo", "técnico"}


class UploadPayload(BaseModel):
    style: str = Field(..., description="minimalista | académico | narrativo | técnico")
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = "es"


class PreviewRequest(BaseModel):
    text: str = Field(..., min_length=1)
    style: str
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = "es"


class GenerateEpubRequest(PreviewRequest):
    isbn: Optional[str] = None
    publish_date: Optional[str] = None
    file_name: Optional[str] = None


def create_app() -> FastAPI:
    ensure_dirs(UPLOAD_DIR, OUTPUT_DIR)

    app = FastAPI(title=APP_NAME, version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": APP_NAME, "time": now_utc_iso()}

    # 1) Subida PDF (multipart) + payload JSON
    # Requerimiento del usuario: "Recibe JSON: {style, pdf_file}".
    # En HTTP real, archivo + JSON se envía típicamente como multipart/form-data:
    # - payload: string JSON
    # - pdf_file: archivo PDF
    @app.post("/upload_pdf/")
    async def upload_pdf(
        payload: str = Form(..., description="JSON string con {style,title,author,language}"),
        pdf_file: UploadFile = File(...),
    ):
        try:
            payload_dict = json.loads(payload)
            p = UploadPayload(**payload_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"payload JSON inválido: {e}")

        if p.style not in ALLOWED_STYLES:
            raise HTTPException(status_code=400, detail=f"style inválido. Usa: {sorted(ALLOWED_STYLES)}")

        if not (pdf_file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se acepta PDF.")

        safe_name = sanitize_filename(pdf_file.filename or "upload.pdf")
        pdf_path = safe_join(UPLOAD_DIR, safe_name)

        content = await pdf_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío.")

        with open(pdf_path, "wb") as f:
            f.write(content)

        extracted = extract_text_from_pdf(pdf_path)
        clean_text = extracted["text"]

        structure = ai_organize_ebook(
            raw_text=clean_text,
            style=p.style,
            openai_model=OPENAI_MODEL,
            gemini_model=GEMINI_MODEL,
            title_hint=p.title,
            author_hint=p.author,
            language=p.language,
        )

        preview_html = render_preview_html(structure=structure, style=p.style)

        return JSONResponse(
            {
                "file_name": safe_name,
                "detected": {
                    "ocr_used": extracted.get("ocr_used", False),
                    "pages": extracted.get("pages"),
                },
                "clean_text": clean_text,
                "ebook": structure,
                "preview_html": preview_html,
            }
        )

    # 2) Preview dinámico (HTML)
    @app.post("/preview/", response_class=HTMLResponse)
    async def preview(req: PreviewRequest):
        if req.style not in ALLOWED_STYLES:
            raise HTTPException(status_code=400, detail=f"style inválido. Usa: {sorted(ALLOWED_STYLES)}")

        structure = ai_organize_ebook(
            raw_text=req.text,
            style=req.style,
            openai_model=OPENAI_MODEL,
            gemini_model=GEMINI_MODEL,
            title_hint=req.title,
            author_hint=req.author,
            language=req.language,
        )
        html = render_preview_html(structure=structure, style=req.style)
        return HTMLResponse(content=html)

    # 3) Generación de EPUB (KDP)
    @app.post("/generate_epub/")
    async def generate_epub(req: GenerateEpubRequest):
        if req.style not in ALLOWED_STYLES:
            raise HTTPException(status_code=400, detail=f"style inválido. Usa: {sorted(ALLOWED_STYLES)}")

        structure = ai_organize_ebook(
            raw_text=req.text,
            style=req.style,
            openai_model=OPENAI_MODEL,
            gemini_model=GEMINI_MODEL,
            title_hint=req.title,
            author_hint=req.author,
            language=req.language,
        )

        final_name = req.file_name or f"{(structure.get('title') or 'ebook').strip().replace(' ', '_')}.epub"
        final_name = sanitize_filename(final_name)
        if not final_name.lower().endswith(".epub"):
            final_name += ".epub"

        epub_path = safe_join(OUTPUT_DIR, final_name)

        generate_epub_from_structure(
            structure=structure,
            style=req.style,
            output_path=epub_path,
            isbn=req.isbn,
            publish_date=req.publish_date,
        )

        return {
            "file_name": final_name,
            "download_url": f"/download/{final_name}",
            "title": structure.get("title"),
            "author": structure.get("author"),
        }

    # 4) Descarga EPUB
    @app.get("/download/{file_name}")
    async def download(file_name: str):
        safe_name = sanitize_filename(file_name)
        epub_path = safe_join(OUTPUT_DIR, safe_name)
        if not os.path.exists(epub_path):
            raise HTTPException(status_code=404, detail="EPUB no encontrado.")

        return FileResponse(
            path=epub_path,
            media_type="application/epub+zip",
            filename=safe_name,
        )

    return app


app = create_app()
