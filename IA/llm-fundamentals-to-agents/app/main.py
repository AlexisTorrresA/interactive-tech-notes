from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .labs import EXAMPLE_CODE, cosine_similarity, neuron_step, quantization_estimate, rag_search, softmax

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = PROJECT_DIR / "index.html"

app = FastAPI(
    title="AI Study Lab",
    version="4.0.0",
    description="Guía interactiva PLN → LLM → RAG → agentes con laboratorios JavaScript y Python.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

LAB_SECTION = """
<section id="python-backend-labs" class="backend-lab">
  <p class="eyebrow">Python + FastAPI</p>
  <h2>Laboratorios con backend real</h2>
  <p>Compara el comportamiento en JavaScript con cálculos ejecutados en Python a través de una API FastAPI.</p>
  <div id="backendStatus" class="backend-status" aria-live="polite"></div>
  <div class="backend-tabs" aria-label="Laboratorios Python">
    <button class="backend-tab active" data-backend-tab="neuron" type="button">Neurona</button>
    <button class="backend-tab" data-backend-tab="attention" type="button">Softmax</button>
    <button class="backend-tab" data-backend-tab="rag" type="button">Mini-RAG</button>
    <button class="backend-tab" data-backend-tab="quantization" type="button">Cuantización</button>
  </div>
  <div id="backendLabApp"></div>
</section>
"""


def rendered_index() -> str:
    html = INDEX_FILE.read_text(encoding="utf-8")
    if "app-enhancements.css" not in html:
        html = html.replace("</head>", '<link rel="stylesheet" href="/static/app-enhancements.css">\n</head>')
    if "python-backend-labs" not in html:
        marker = "</main>" if "</main>" in html else "</body>"
        html = html.replace(marker, LAB_SECTION + "\n" + marker, 1)
    if "backend-labs.js" not in html:
        html = html.replace("</body>", '<script src="/static/backend-labs.js"></script>\n</body>')
    return html


class NeuronRequest(BaseModel):
    x1: float = 1.0
    x2: float = 0.5
    w1: float = 0.8
    w2: float = -0.4
    bias: float = 0.1
    target: float = Field(1.0, ge=0, le=1)
    learning_rate: float = Field(0.3, gt=0, le=2)


class AttentionRequest(BaseModel):
    logits: list[float] = Field(default_factory=lambda: [2.4, 1.1, 0.2, -0.6], min_length=2, max_length=32)
    temperature: float = Field(1.0, gt=0.01, le=10)


class CosineRequest(BaseModel):
    a: list[float] = Field(min_length=1, max_length=128)
    b: list[float] = Field(min_length=1, max_length=128)

    @field_validator("b")
    @classmethod
    def validate_b(cls, value: list[float], info):
        a = info.data.get("a")
        if a is not None and len(a) != len(value):
            raise ValueError("Los vectores a y b deben tener la misma dimensión")
        return value


class RagDoc(BaseModel):
    id: str
    title: str
    text: str


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(2, ge=1, le=10)
    documents: list[RagDoc] | None = Field(default=None, max_length=20)


class QuantRequest(BaseModel):
    params_b: float = Field(7, gt=0, le=1000)
    bits: int = Field(4)
    context_k: float = Field(8, gt=0, le=2048)
    layers: int = Field(32, ge=1, le=400)

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, value: int):
        if value not in {2, 3, 4, 5, 6, 8, 16, 32}:
            raise ValueError("bits debe ser 2, 3, 4, 5, 6, 8, 16 o 32")
        return value


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(rendered_index())


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "app": "AI Study Lab", "version": app.version, "python_labs": True}


@app.post("/api/labs/neuron")
def lab_neuron(payload: NeuronRequest):
    return neuron_step(**payload.model_dump())


@app.post("/api/labs/attention")
def lab_attention(payload: AttentionRequest):
    probs = softmax(payload.logits, payload.temperature)
    return {"probabilities": probs, "sum": sum(probs)}


@app.post("/api/labs/cosine")
def lab_cosine(payload: CosineRequest):
    return {"similarity": cosine_similarity(payload.a, payload.b)}


@app.post("/api/labs/rag")
def lab_rag(payload: RagRequest):
    docs = [d.model_dump() for d in payload.documents] if payload.documents else None
    return rag_search(payload.query, docs, payload.top_k)


@app.post("/api/labs/quantization")
def lab_quantization(payload: QuantRequest):
    return quantization_estimate(**payload.model_dump())


@app.get("/api/examples/{lab}")
def example_code(lab: str):
    item = EXAMPLE_CODE.get(lab)
    if not item:
        raise HTTPException(status_code=404, detail="Ejemplo no encontrado")
    return {"lab": lab, **item}
