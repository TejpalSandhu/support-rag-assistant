"""
Web API for the support assistant (FastAPI).

Run locally:
    uvicorn api:app --reload
then open http://127.0.0.1:8000/docs to try every endpoint in the browser.

Endpoints (everything except health and /api/auth/... needs a signed-in user
when GOOGLE_CLIENT_ID is set; see auth.py):
    GET    /api/health              is the server up?
    GET    /api/auth/config         Google client ID for the sign-in button
    POST   /api/auth/google         sign in with a Google ID token
    POST   /api/auth/logout         sign out
    GET    /api/me                  who is signed in, questions left today
    GET    /api/documents           uploaded PDFs, chunk counts and status
    POST   /api/documents           upload a PDF (indexed in the background)
    DELETE /api/documents/{name}    remove a PDF and its chunks
    POST   /api/ask                 ask a question about the documents
    POST   /api/feedback            👍/👎 on an answer (saved in Langfuse)
    GET    /api/metrics?days=7      usage numbers for the metrics page
"""

import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Query,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import metrics
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
    # feedback and metrics live in Langfuse, so they need its keys
    tracing: bool = False
    # (owner, file name) -> "processing" or "failed: <reason>"; finished files drop out
    jobs: dict[tuple[str, str], str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # (email, UTC date) -> questions asked that day. Kept in memory for now,
    # so it resets when the server restarts; moves to the database later.
    questions_today: dict[tuple[str, str], int] = field(default_factory=dict)


def create_services() -> Services:
    rag.PDF_DIR.mkdir(parents=True, exist_ok=True)
    collection = rag.get_web_collection()
    # a fresh container starts with an empty Chroma: refill it from the backups
    # (files from before accounts existed go to the owner)
    restored = rag.restore_backups(collection, legacy_owner=auth.owner_email())
    if restored:
        print(f"Restored {len(restored)} document(s) from backup: {', '.join(restored)}")
    return Services(
        client=rag.make_gemini_client(),
        langfuse=rag.make_langfuse(),
        collection=collection,
        pdf_dir=rag.PDF_DIR,
        tracing=rag.tracing_configured(),
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


# --- Signed-in user and daily limit ---

UserDep = Annotated[auth.User, Depends(auth.current_user)]


def daily_question_limit() -> int:
    # each user's share of the free Gemini answers (~40/day for everyone)
    return int(os.getenv("DAILY_QUESTION_LIMIT") or 20)


def today_key(user: auth.User) -> tuple[str, str]:
    return (user.email, datetime.now(timezone.utc).date().isoformat())


def questions_left(services: Services, user: auth.User) -> int:
    return max(daily_question_limit() - services.questions_today.get(today_key(user), 0), 0)


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


class Me(BaseModel):
    email: str
    name: str
    picture: str
    login_enabled: bool
    questions_left_today: int
    daily_question_limit: int


class FeedbackRequest(BaseModel):
    # the trace_id from /api/ask: 32 lowercase hex characters
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    helpful: bool


class MetricsResponse(BaseModel):
    days: int
    environment: str
    questions: int
    answered: int
    refused: int
    failed: int
    avg_seconds: float | None
    p95_seconds: float | None
    feedback_votes: int
    helpful_rate: float | None


# --- Helpers ---

def safe_pdf_name(filename: str | None) -> str:
    """Keep only a plain file name, so '../../x.pdf' can't escape the pdfs folder."""
    name = Path((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if name.startswith(".") or not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are accepted.")
    return name


def pdf_path(services: Services, owner: str, name: str) -> Path:
    # each person's PDFs in their own folder, named by a code, not the email
    return services.pdf_dir / rag.owner_key(owner) / name


def index_in_background(services: Services, owner: str, name: str) -> None:
    path = pdf_path(services, owner, name)
    try:
        rag.index_pdf(path, services.collection, services.client, owner)
    except Exception as error:
        # e.g. a scanned PDF with no text, or the daily quota: clean up so
        # a broken file never sits half-indexed, and report it in the list
        rag.delete_document(services.collection, owner, name)
        path.unlink(missing_ok=True)
        services.jobs[(owner, name)] = f"failed: {error}"
    else:
        services.jobs.pop((owner, name), None)


# --- Endpoints ---

public = APIRouter(prefix="/api")
protected = APIRouter(prefix="/api", dependencies=[Depends(auth.current_user)])


@public.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@protected.get("/me", response_model=Me)
def me(user: UserDep, services: ServicesDep) -> Me:
    return Me(
        email=user.email,
        name=user.name,
        picture=user.picture,
        login_enabled=auth.login_enabled(),
        questions_left_today=questions_left(services, user),
        daily_question_limit=daily_question_limit(),
    )


@protected.get("/documents", response_model=list[DocumentInfo])
def list_documents(services: ServicesDep, user: UserDep) -> list[DocumentInfo]:
    counts = rag.list_documents(services.collection, user.email)
    jobs = {name: state for (owner, name), state in services.jobs.items() if owner == user.email}
    return [
        DocumentInfo(name=name, chunks=counts.get(name, 0), status=jobs.get(name, "ready"))
        for name in sorted(set(counts) | set(jobs))
    ]


@protected.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentInfo)
def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    services: ServicesDep,
    user: UserDep,
) -> DocumentInfo:
    name = safe_pdf_name(file.filename)

    # read one byte more than allowed to detect files that are too big
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF is larger than 25 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This file is not a PDF.")

    job = (user.email, name)
    with services.lock:
        if services.jobs.get(job) == "processing":
            raise HTTPException(status.HTTP_409_CONFLICT, f"{name} is still being processed.")
        services.jobs[job] = "processing"

    path = pdf_path(services, user.email, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    # reading and embedding takes a while: answer now, index afterwards
    background.add_task(index_in_background, services, user.email, name)
    return DocumentInfo(name=name, chunks=0, status="processing")


@protected.delete("/documents/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(name: str, services: ServicesDep, user: UserDep) -> None:
    name = safe_pdf_name(name)
    job = (user.email, name)
    if services.jobs.get(job) == "processing":
        raise HTTPException(status.HTTP_409_CONFLICT, f"{name} is still being processed.")

    # only your own files: someone else's file of the same name is a 404 too
    path = pdf_path(services, user.email, name)
    if name not in rag.list_documents(services.collection, user.email) and not path.exists()             and job not in services.jobs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document called {name}.")

    rag.delete_document(services.collection, user.email, name)
    path.unlink(missing_ok=True)
    services.jobs.pop(job, None)


@protected.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, services: ServicesDep, user: UserDep) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The question is empty.")
    if questions_left(services, user) == 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You've used your {daily_question_limit()} questions for today. "
            "The limit resets at midnight UTC.",
        )

    try:
        result = rag.ask(
            question, services.collection, services.client, services.langfuse, owner=user.email,
        )
    except rag.DailyQuotaError as error:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error))
    except Exception as error:
        code = rag.error_status(error)
        if code == 429:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "The AI models' usage limits were reached. Try again later "
                "(free-tier limits reset daily) or set GENERATION_MODEL to another model.",
            )
        if code >= 500 or rag.is_timeout(error):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The AI models are busy right now. Please try again in a minute.",
            )
        raise

    # only questions that got an answer or a refusal count (not model errors)
    with services.lock:
        key = today_key(user)
        services.questions_today[key] = services.questions_today.get(key, 0) + 1

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


def require_tracing(services: Services) -> None:
    if not services.tracing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Feedback and metrics need Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
        )


@protected.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def feedback(body: FeedbackRequest, services: ServicesDep) -> None:
    require_tracing(services)
    services.langfuse.create_score(
        name=metrics.FEEDBACK_SCORE,
        value=1 if body.helpful else 0,
        data_type="BOOLEAN",
        trace_id=body.trace_id,
        # one score ID per answer: changing your vote replaces the old one
        score_id=f"{body.trace_id}-feedback",
    )


@protected.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    services: ServicesDep,
    days: Annotated[int, Query(ge=1, le=30)] = 7,  # the free plan keeps 30 days
) -> MetricsResponse:
    require_tracing(services)
    try:
        return MetricsResponse(**metrics.summary(services.langfuse, days))
    except Exception as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Couldn't load the numbers from Langfuse. Try again in a minute.",
        ) from error


# --- App ---

def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # runs once when the server starts; tests pass their own services
        app.state.services = services or create_services()
        yield
        app.state.services.langfuse.flush()

    app = FastAPI(title="Support RAG Assistant", lifespan=lifespan)
    auth.add_sessions(app)
    app.include_router(public)
    app.include_router(auth.router)
    app.include_router(protected)

    # on Azure there's no nginx: the backend also serves the built website.
    # Added last, so /api/... routes are matched first.
    static_dir = os.getenv("STATIC_DIR")
    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="website")
    return app


app = create_app()
