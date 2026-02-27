import os
import uuid
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
from ebooklib import epub

app = FastAPI(title="Wae Production API")

# 1. Configuración de CORS - Vital para la previsualización
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configuración de Gemini (Usando el modelo ESTABLE 1.5 Flash)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    # Cambiamos a 1.5-flash para máxima compatibilidad en Railway
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("❌ ERROR: No se encontró GEMINI_API_KEY en las variables.")

# Carpetas
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
    return {"status": "online", "ia_active": bool(API_KEY), "model": "gemini-1.5-flash"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        os.remove(file_path)
        return {"filename": file.filename, "text": text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error PDF: {str(e)}")

@app.post("/preview/")
async def generate_preview(req: PreviewRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Falta GEMINI_API_KEY en Railway")
    try:
        # Prompt científico
        prompt = f"Actúa como maquetador editorial. Estructura este texto con estilo '{req.style}' usando HTML (h1, p). Solo contenido: {req.text[:4000]}"
        response = model.generate_content(prompt)
        
        # Estilos visuales
        styles = {
            "académico": "font-family: serif; line-height: 1.8; color: #1e293b; text-align: justify;",
            "técnico": "font-family: monospace; background: #f8fafc; padding: 20px; border-left: 5px solid #f97316;",
            "minimalista": "font-family: sans-serif; color: #333;",
            "narrativo": "font-family: Georgia, serif; line-height: 1.7;"
        }
        css = styles.get(req.style, "font-family: sans-serif;")
        return HTMLResponse(content=f"<div style='{css}'>{response.text}</div>")
    except Exception as e:
        # Si falla, devolvemos el error real para verlo en el dashboard
        raise HTTPException(status_code=500, detail=f"Error IA: {str(e)}")

@app.get("/download/{file_name}")
async def download(file_name: str):
    path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path, filename=file_name, media_type='application/epub+zip')
    return {"error": "No encontrado"}
