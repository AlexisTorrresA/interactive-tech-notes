from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def neuron_step(*, x1: float, x2: float, w1: float, w2: float, bias: float, target: float, learning_rate: float = 0.3) -> dict:
    z = x1 * w1 + x2 * w2 + bias
    y = sigmoid(z)
    eps = 1e-12
    loss = -(target * math.log(y + eps) + (1 - target) * math.log(1 - y + eps))
    error = y - target
    return {
        "z": z,
        "prediction": y,
        "loss": loss,
        "gradient": {"w1": error * x1, "w2": error * x2, "bias": error},
        "updated": {
            "w1": w1 - learning_rate * error * x1,
            "w2": w2 - learning_rate * error * x2,
            "bias": bias - learning_rate * error,
        },
    }


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    return dot / (na * nb) if na and nb else 0.0


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    t = max(temperature, 1e-6)
    scaled = [v / t for v in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "y", "o", "un", "una", "para", "por", "con",
    "mi", "que", "como", "cómo", "se", "en", "a", "es", "al", "lo", "su", "sus",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-záéíóúüñ0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _counter_cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


DEFAULT_DOCS = [
    {"id": "rag", "title": "RAG", "text": "RAG recupera evidencia relevante antes de pedir al modelo que genere una respuesta fundamentada."},
    {"id": "agents", "title": "Agentes", "text": "Un agente combina un modelo con herramientas, memoria, políticas y un ciclo de decisión para ejecutar acciones."},
    {"id": "tokens", "title": "Tokenización", "text": "La tokenización transforma texto en unidades discretas que luego se convierten en identificadores y embeddings."},
    {"id": "mcp", "title": "MCP", "text": "MCP estandariza la forma en que aplicaciones de IA descubren y usan herramientas, recursos y prompts expuestos por servidores."},
]


def rag_search(query: str, docs: list[dict] | None = None, top_k: int = 2) -> dict:
    documents = docs or DEFAULT_DOCS
    qv = Counter(_tokens(query))
    scored = []
    for doc in documents:
        dv = Counter(_tokens(f"{doc.get('title', '')} {doc.get('text', '')}"))
        score = _counter_cosine(qv, dv)
        scored.append({**doc, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    chosen = scored[: max(1, min(top_k, len(scored)))]
    grounded = [d for d in chosen if d["score"] > 0]
    answer = (
        f"Evidencia principal: {grounded[0]['text']} [{grounded[0]['id']}]"
        if grounded
        else "No se recuperó evidencia suficiente; el sistema debería abstenerse o reformular la búsqueda."
    )
    return {"results": scored, "selected": chosen, "answer": answer}


def quantization_estimate(*, params_b: float, bits: int, context_k: float = 8, layers: int = 32) -> dict:
    weights_gb = params_b * bits / 8
    overhead_factor = 1.10 if bits < 16 else 1.02
    weights_with_overhead_gb = weights_gb * overhead_factor
    kv_relative_gb = context_k * layers * 0.0015
    return {
        "weights_theoretical_gb": weights_gb,
        "weights_with_overhead_gb": weights_with_overhead_gb,
        "kv_cache_relative_gb": kv_relative_gb,
        "didactic_note": "La KV cache depende además de heads, head_dim, batch, dtype y arquitectura. Esta cifra es pedagógica.",
    }


EXAMPLE_CODE = {
    "neuron": {
        "python": """import math\n\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-z))\n\nz = x1*w1 + x2*w2 + bias\ny = sigmoid(z)\nerror = y - target\nw1 -= lr * error * x1\nw2 -= lr * error * x2\nbias -= lr * error""",
        "javascript": """const sigmoid = z => 1 / (1 + Math.exp(-z));\n\nconst z = x1*w1 + x2*w2 + bias;\nconst y = sigmoid(z);\nconst error = y - target;\nw1 -= lr * error * x1;\nw2 -= lr * error * x2;\nbias -= lr * error;""",
    },
    "attention": {
        "python": """import math\n\ndef softmax(logits, temperature=1.0):\n    scaled = [x / temperature for x in logits]\n    m = max(scaled)\n    exps = [math.exp(x - m) for x in scaled]\n    total = sum(exps)\n    return [x / total for x in exps]""",
        "javascript": """function softmax(logits, temperature = 1) {\n  const scaled = logits.map(x => x / temperature);\n  const m = Math.max(...scaled);\n  const exps = scaled.map(x => Math.exp(x - m));\n  const total = exps.reduce((a, b) => a + b, 0);\n  return exps.map(x => x / total);\n}""",
    },
    "rag": {
        "python": """q = vectorize(query)\nscored = []\nfor doc in documents:\n    score = cosine(q, vectorize(doc[\"text\"]))\n    scored.append((score, doc))\n\ncontext = sorted(scored, reverse=True)[:top_k]""",
        "javascript": """const q = vectorize(query);\nconst scored = documents\n  .map(doc => ({...doc, score: cosine(q, vectorize(doc.text))}))\n  .sort((a, b) => b.score - a.score);\n\nconst context = scored.slice(0, topK);""",
    },
    "quantization": {
        "python": """params_b = 7\nbits = 4\nweights_gb = params_b * bits / 8\nprint(f\"Pesos teóricos: {weights_gb:.2f} GB\")""",
        "javascript": """const paramsB = 7;\nconst bits = 4;\nconst weightsGB = paramsB * bits / 8;\nconsole.log(`Pesos teóricos: ${weightsGB.toFixed(2)} GB`);""",
    },
}
