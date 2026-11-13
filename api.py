"""
Web API for the support assistant (FastAPI).

Run locally:
    uvicorn api:app --reload
then open http://127.0.0.1:8000/docs to try every endpoint in the browser.

Endpoints (everything except /api/health needs the X-API-Key header when
APP_ACCESS_KEY is set in .env):
    GET    /api/health              is the server up?
    GET    /api/documents           uploaded PDFs, chunk counts and status
    POST   /api/documents           upload a PDF (indexed in the background)
    DELETE /api/documents/{name}    remove a PDF and its chunks
    POST   /api/ask                 ask a question about the documents
"""

import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import rag_assistant as rag

MAX_UPLOAD_BYTES = 25 * 1024 * 1024    # 25 MB
MAX_QUESTION_CHARS = 1000


# --- Shared state ---

@dataclass
class Services:
    """Everything the endpoints need; tests pass in fakes instead."""
    client: Any
    langfuse: Any
    collection: Any
    pdf_dir: Path
    # file name -> "processing" or "failed: <reason>"; finished files drop out
    jobs: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_services() -> Services:
    rag.PDF_DIR.mkdir(exist_ok=True)
    return Services(
        client=rag.make_gemini_client(),
        langfuse=rag.make_langfuse(),
        collection=rag.get_collection(),
        pdf_dir=rag.PDF_DIR,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


# --- Access key ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_access_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
    expected = os.getenv("APP_ACCESS_KEY")
    if not expected:
        return  # no key configured: local development
    # compare_digest takes the same time for every wrong key, so the key
    # can't be guessed character by character from response times
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or wrong access key.")


# --- Request and response shapes ---

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    status: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    source: str
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    covered: bool
    best_score: float
    sources: list[Source]
    trace_id: str | None
    trace_url: str | None


# --- Helpers ---

def safe_pdf_name(filename: str | None) -> str:
    """Keep only a plain file name, so '../../x.pdf' can't escape the pdfs folder."""
    name = Path((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if name.startswith(".") or not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are accepted.")
    return name


def index_in_background(services: Services, name: str) -> None:
    path = services.pdf_dir / name
    try:
        rag.index_pdf(path, services.collection, services.client)
    except Exception as error:
        # e.g. a scanned PDF with no text, or the daily quota: clean up so
        # a broken file never sits half-indexed, and report it in the list
        rag.delete_document(services.collection, name)
        path.unlink(missing_ok=True)
        services.jobs[name] = f"failed: {error}"
    else:
        services.jobs.pop(name, None)


# --- Endpoints ---

public = APIRouter(prefix="/api")
protected = APIRouter(prefix="/api", dependencies=[Depends(require_access_key)])


@public.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@protected.get("/documents", response_model=list[DocumentInfo])
def list_documents(services: ServicesDep) -> list[DocumentInfo]:
    counts = rag.list_documents(services.collection)
    names = sorted(set(counts) | set(services.jobs))
    return [
        DocumentInfo(name=name, chunks=counts.get(name, 0), status=services.jobs.get(name, "ready"))
        for name in names
    ]


@protected.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentInfo)
def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    services: ServicesDep,
) -> DocumentInfo:
    name = safe_pdf_name(file.filename)

    # read one byte more than allowed to detect files that are too big
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF is larger than 25 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This file is not a PDF.")

    with services.lock:
        if services.jobs.get(name) == "processing":
            raise HTTPException(status.HTTP_409_CONFLICT, f"{name} is still being processed.")
        services.jobs[name] = "processing"

    services.pdf_dir.mkdir(parents=True, exist_ok=True)
    (services.pdf_dir / name).write_bytes(data)
    # reading and embedding takes a while: answer now, index afterwards
    background.add_task(index_in_background, services, name)
    return DocumentInfo(name=name, chunks=0, status="processing")


@protected.delete("/documents/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(name: str, services: ServicesDep) -> None:
    name = safe_pdf_name(name)
    if services.jobs.get(name) == "processing":
        raise HTTPException(status.HTTP_409_CONFLICT, f"{name} is still being processed.")

    path = services.pdf_dir / name
    if name not in rag.list_documents(services.collection) and not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document called {name}.")

    rag.delete_document(services.collection, name)
    path.unlink(missing_ok=True)
    services.jobs.pop(name, None)


@protected.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, services: ServicesDep) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The question is empty.")

    try:
        result = rag.ask(question, services.collection, services.client, services.langfuse)
    except rag.DailyQuotaError as error:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error))
    except Exception as error:
        code = rag.error_status(error)
        if code == 429:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "The AI model's usage limit was reached. Try again later "
                "(free-tier limits reset daily) or set GENERATION_MODEL to another model.",
            )
        if code >= 500 or rag.is_timeout(error):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The AI model is busy right now. Please try again in a minute.",
            )
        raise

    return AskResponse(
        answer=result["answer"],
        covered=result["covered"],
        best_score=result["best_score"],
        sources=[
            Source(source=c["source"], page=c["page"], score=c["score"], text=c["text"])
            for c in result["chunks"]
        ],
        trace_id=result["trace_id"],
        trace_url=result["trace_url"],
    )


# --- App ---

def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # runs once when the server starts; tests pass their own services
        app.state.services = services or create_services()
        yield
        app.state.services.langfuse.flush()

    app = FastAPI(title="Support RAG Assistant", lifespan=lifespan)
    app.include_router(public)
    app.include_router(protected)
    return app


app = create_app()
