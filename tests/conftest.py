"""
Shared test setup: a fake Gemini, a temporary Chroma database, tiny PDFs.

Tests never call the real Gemini API, so they're free, fast, and don't
fail when Google is busy.
"""

import hashlib
import math
from types import SimpleNamespace

import chromadb
import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse

import api
import rag_assistant as rag

FAKE_ANSWER = "FAKE ANSWER from the context"


def fake_vector(text: str, dimensions: int = 64) -> list[float]:
    """Bag-of-words vector: texts sharing words point in similar directions."""
    vector = [0.0] * dimensions
    for word in text.lower().split():
        word = word.strip(".,:;!?()'\"")
        if word:
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dimensions] += 1.0
    vector[0] += 0.01  # never an all-zero vector
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


class FakeGenerator:
    """Stands in for client.models.generate_content (the answer model)."""

    def __init__(self):
        self.calls = 0
        self.models = []  # which model each call asked
        self.timeout_ms = None
        self.error = None  # set to an exception to simulate a failing model
        self.failing_models = None  # None = every model fails when error is set

    def __call__(self, model, contents, config=None):
        self.calls += 1
        self.models.append(model)
        self.timeout_ms = config.http_options.timeout if config else None
        if self.error and (self.failing_models is None or model in self.failing_models):
            raise self.error
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=10, total_token_count=110)
        return SimpleNamespace(text=FAKE_ANSWER, usage_metadata=usage)


class FakeModels:
    def __init__(self, generator):
        self.embed_calls = 0
        self.generate_content = generator

    def embed_content(self, model, contents):
        self.embed_calls += 1
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=fake_vector(text)) for text in contents]
        )


@pytest.fixture
def fake_gemini():
    generator = FakeGenerator()
    return SimpleNamespace(models=FakeModels(generator), generator=generator)


@pytest.fixture
def services(tmp_path, fake_gemini, monkeypatch):
    monkeypatch.delenv("APP_ACCESS_KEY", raising=False)
    # backups go to a temp folder, never to real Azure storage
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(rag.time, "sleep", lambda seconds: None)  # no real waiting
    # fake vectors give lower scores than Gemini, so use a test threshold
    monkeypatch.setattr(rag, "MIN_SCORE", 0.4)
    monkeypatch.setattr(rag, "BACKUP_DIR", tmp_path / "index_backup")

    collection = chromadb.PersistentClient(path=str(tmp_path / "db")).get_or_create_collection(
        name="test_docs",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    return api.Services(
        client=fake_gemini,
        langfuse=Langfuse(public_key="test", secret_key="test", tracing_enabled=False),
        collection=collection,
        pdf_dir=tmp_path / "pdfs",
    )


@pytest.fixture
def client(services):
    # `with` runs the app's startup code (lifespan) with our fake services
    with TestClient(api.create_app(services)) as test_client:
        yield test_client


def build_pdf(pages: list[str]) -> bytes:
    """Build a small, valid PDF in memory with one line of text per page."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
        f"/Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return pdf


@pytest.fixture
def make_pdf():
    return build_pdf
