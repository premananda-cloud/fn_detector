import io
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
import httpx
from newspaper import Article
from pdfminer.high_level import extract_text as pdf_extract_text

from detector import FakeNewsDetector

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
detector: Optional[FakeNewsDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = FakeNewsDetector()
    yield
    detector = None


app = FastAPI(title="Fake News Detector", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_url_text(url: str) -> str:
    """Use newspaper3k to extract article body from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    text = article.text.strip()
    if not text:
        raise ValueError("Could not extract article text from URL.")
    return text


def extract_pdf_text(file_bytes: bytes) -> str:
    text = pdf_extract_text(io.BytesIO(file_bytes))
    text = text.strip()
    if not text:
        raise ValueError("Could not extract text from PDF.")
    return text


def run_analysis(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        raise ValueError("Text is too short to analyse.")
    return detector.analyze(text)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextItem(BaseModel):
    text: str

class UrlItem(BaseModel):
    url: str

class BatchTextRequest(BaseModel):
    items: list[str]

class BatchUrlRequest(BaseModel):
    items: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("ui/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze/text")
async def analyze_text(req: TextItem):
    try:
        return run_analysis(req.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/analyze/url")
async def analyze_url(req: UrlItem):
    try:
        text = await fetch_url_text(str(req.url))
        result = run_analysis(text)
        result['extracted_text_preview'] = text[:300]
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or ""

    try:
        if filename.endswith(".pdf"):
            text = extract_pdf_text(content)
        elif filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore").strip()
        else:
            raise HTTPException(status_code=415, detail="Only .txt and .pdf files are supported.")

        result = run_analysis(text)
        result['filename'] = filename
        result['extracted_text_preview'] = text[:300]
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")


@app.post("/analyze/batch/text")
async def analyze_batch_text(req: BatchTextRequest):
    results = []
    for i, text in enumerate(req.items):
        try:
            results.append({"index": i, "status": "ok", **run_analysis(text)})
        except Exception as e:
            results.append({"index": i, "status": "error", "detail": str(e)})
    return {"results": results}


@app.post("/analyze/batch/url")
async def analyze_batch_url(req: BatchUrlRequest):
    results = []
    for i, url in enumerate(req.items):
        try:
            text = await fetch_url_text(url)
            result = run_analysis(text)
            result['extracted_text_preview'] = text[:300]
            results.append({"index": i, "status": "ok", **result})
        except Exception as e:
            results.append({"index": i, "status": "error", "detail": str(e)})
    return {"results": results}
