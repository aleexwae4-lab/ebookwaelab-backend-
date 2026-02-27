import os
import uuid
import fitz  # PyMuPDF para extraer texto
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
from ebooklib import epub

app = FastAPI(title="Wae Production Premium API")

# 1. Habilitar CORS (Crucial para que el Dashboard vea la previsualización)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configuración de IA (Gemini 2.5 Flash)
# RECUERDA: Configura GEMINI_API_KEY en las variables de entorno de Railway
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# Directorios de trabajo
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PreviewRequest(BaseModel):
    style: str
    text: str

class EpubRequest(BaseModel):
    title: str
    author: str
    text: str
    style: str

@app.get("/")
async def health():
    return {"status": "online", "engine": "Gemini 2.5 Flash", "cors": "enabled"}

# ENDPOINT 1: Subir y extraer texto
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Extraer texto usando PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove(file_path) # Limpiar archivo temporal

        return {
            "filename": file.filename,
            "text": text,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ENDPOINT 2: Generar Preview con IA
@app.post("/preview/")
async def generate_preview(req: PreviewRequest):
    try:
        prompt = f"""
        Actúa como un maquetador editorial experto. Toma el siguiente texto y organízalo 
        en capítulos y párrafos con un estilo '{req.style}'. 
        Usa etiquetas HTML (h1, h2, p). No incluyas etiquetas <html> o <body>.
        Texto: {req.text[:4000]}
        """
        response = model.generate_content(prompt)
        
        # CSS inyectado para el Dashboard
        styles = {
            "académico": "font-family: serif; line-height: 1.8; text-align: justify; color: #1e293b;",
            "técnico": "font-family: monospace; background: #f8f9fa; padding: 20px; color: #334155;",
            "minimalista": "font-family: sans-serif; color: #444;",
            "narrativo": "font-family: Georgia, serif; line-height: 1.6; color: #000;"
        }
        css = styles.get(req.style, "font-family: sans-serif;")
        
        return HTMLResponse(content=f"<div style='{css}'>{response.text}</div>")
    except Exception as e:
        return HTMLResponse(content=f"Error en IA: {str(e)}", status_code=500)

# ENDPOINT 3: Generar EPUB Final
@app.post("/generate_epub/")
async def generate_epub(req: EpubRequest):
    try:
        book = epub.EpubBook()
        book.set_title(req.title)
        book.set_language('es')
        book.add_author(req.author)

        # Dividir por capítulos simples
        chapters_text = req.text.split('\n\n')
        for i, content in enumerate(chapters_text[:10]): # Limitamos a 10 para ejemplo
            c = epub.EpubHtml(title=f'Capítulo {i+1}', file_name=f'chap_{i+1}.xhtml')
            c.content = f'<h1>Capítulo {i+1}</h1><p>{content}</p>'
            book.add_item(c)
            book.spine.append(c)

        file_name = f"{uuid.uuid4().hex}.epub"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        epub.write_epub(file_path, book)

        return {"file_name": file_name, "download_url": f"/download/{file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ENDPOINT 4: Descarga
@app.get("/download/{file_name}")
async def download_file(file_name: str):
    path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path, filename="tu_libro_wae.epub")
    return {"error": "Archivo no encontrado"}
