# EbookWaeLab Premium SaaS — Backend (FastAPI)

Backend modular en Python para tu SaaS **EbookWaeLab Premium**:
- Subida de PDF
- Extracción de texto (PyPDF2 + OCR si es escaneado)
- Procesamiento con **OpenAI (GPT-4\*) + Gemini** para estructura premium
- Preview dinámico (HTML) para Base44
- Generación de **EPUB** (Amazon KDP-ready) con TOC y metadatos
- Portada **IA activada por defecto** (`ENABLE_AI_COVER=true`) con fallback seguro

> PDF de ejemplo compartido en el chat (útil para pruebas):
> - https://gensparkstorageprodwest.blob.core.windows.net/personal/c7730733-ac4d-41df-87bc-41f613b3a734/upload/default/5a54a5be-8d65-4567-8ec5-1cc8e6e7c9f8

---

## 1) Requisitos
- Python 3.10+
- Variables de entorno con tus API keys
- (Opcional OCR) **Tesseract** instalado en el sistema
- (Opcional OCR PDF escaneado) **Poppler** (requerido por `pdf2image`)

---

## 2) Instalación

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

---

## 3) Variables de entorno

Ver `.env.example`.

- `OPENAI_API_KEY` (requerida para IA principal)
- `GEMINI_API_KEY` (opcional pero recomendado)
- `UPLOAD_DIR` (por defecto: `uploads`)
- `OUTPUT_DIR` (por defecto: `outputs`)
- `ENABLE_AI_COVER` (por defecto: `true`)

Modelos (opcionales):
- `OPENAI_MODEL` (por defecto: `gpt-4o-mini`)
- `GEMINI_MODEL` (por defecto: `gemini-1.5-pro`)
- `OPENAI_IMAGE_MODEL` (por defecto: `gpt-image-1`)

---

## 4) Ejecutar en local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Swagger: `GET http://localhost:8000/docs`

---

## 5) Endpoints

### `POST /upload_pdf/`
**Requerimiento:** "Recibe JSON: `{style: ..., pdf_file: <archivo>}`".

En HTTP real, archivo + JSON se envía como `multipart/form-data`:
- `payload`: *string* con JSON (ej. `{"style":"académico"}`)
- `pdf_file`: el archivo PDF

**Respuesta incluye:**
- `clean_text`: texto limpio sin estructura
- `ebook`: JSON estructurado (capítulos/secciones con **Markdown interno**)
- `preview_html`: HTML renderizado listo para Base44

Ejemplo `curl`:
```bash
curl -X POST "http://localhost:8000/upload_pdf/" \
  -F 'payload={"style":"académico","language":"es"}' \
  -F 'pdf_file=@mi_archivo.pdf;type=application/pdf'
```

---

### `POST /preview/`
Recibe JSON y devuelve **HTML** (render en tiempo real):

```json
{
  "text": "...",
  "style": "técnico",
  "title": "Opcional",
  "author": "Opcional",
  "language": "es"
}
```

---

### `POST /generate_epub/`
Genera un EPUB premium (KDP-ready) desde `text + style`.

```json
{
  "text": "...",
  "style": "narrativo",
  "title": "Opcional",
  "author": "Opcional",
  "language": "es",
  "isbn": null,
  "publish_date": null,
  "file_name": "mi_ebook.epub"
}
```

Respuesta:
- `download_url`: ruta de descarga

---

### `GET /download/{file_name}`
Descarga el EPUB generado.

---

## 6) Deployment (Railway / "Raywail")

Comando recomendado:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Configura variables en el panel:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (recomendado)
- `UPLOAD_DIR=uploads`
- `OUTPUT_DIR=outputs`
- `ENABLE_AI_COVER=true`

> Nota: `uploads/` y `outputs/` son carpetas runtime. En plataformas con FS efímero, considera usar storage persistente (S3/Blob) para outputs.
