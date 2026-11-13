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
    POST   /api/ask                 ask a question (saved in a chat)
    GET    /api/chats               your saved chats, newest first
    GET    /api/chats/{id}          one chat's messages
    DELETE /api/chats/{id}          delete a chat
    POST   /api/feedback            👍/👎 on an answer (saved in Langfuse)
    GET    /api/metrics?days=7      your numbers for the metrics page
    GET    /api/metrics?scope=all   everyone's numbers and a per-person table (owner only)
"""

import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

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
import store as chat_store
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
    # saved chats and the daily counter: Neon PostgreSQL when DATABASE_URL is
    # set, otherwise in memory (tests, local runs)
    store: Any = field(default_factory=chat_store.MemoryStore)


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
        store=chat_store.make_store(os.getenv("DATABASE_URL")),
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


# --- Signed-in user and daily limit ---

UserDep = Annotated[auth.User, Depends(auth.current_user)]


def daily_question_limit() -> int:
    # each user's share of the free Gemini answers (~40/day for everyone)
    return int(os.getenv("DAILY_QUESTION_LIMIT") or 20)


def today() -> date:
    return datetime.now(timezone.utc).date()


def questions_left(services: Services, user: auth.User) -> int:
    used = services.store.questions_used(user.email, today())
    return max(daily_question_limit() - used, 0)


# --- Request and response shapes ---

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    status: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # continue this chat; leave out to start a new one
    chat_id: str | None = None


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
    chat_id: str


class ChatSummary(BaseModel):
    id: str
    title: str
    updated_at: datetime


class ChatMessage(BaseModel):
    role: str
    content: str
    # for answers: covered, best_score, sources, trace_id, trace_url
    details: dict[str, Any] | None


class Me(BaseModel):
    email: str
    name: str
    picture: str
    login_enabled: bool
    is_owner: bool
    questions_left_today: int
    daily_question_limit: int


class FeedbackRequest(BaseModel):
    # the trace_id from /api/ask: 32 lowercase hex characters
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    helpful: bool


class PersonUsage(BaseModel):
    email: str
    questions: int
    last_active: date


class MetricsResponse(BaseModel):
    days: int
    scope: Literal["me", "all"]
    environment: str
    questions: int
    answered: int
    refused: int
    failed: int
    avg_seconds: float | None
    p95_seconds: float | None
    feedback_votes: int
    helpful_rate: float | None
    # only for scope=all: questions per person (from the database)
    people: list[PersonUsage] | None = None


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
        is_owner=user.email == auth.owner_email(),
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
    if body.chat_id and not services.store.chat_exists(user.email, body.chat_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found.")
    if questions_left(services, user) == 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You've used your {daily_question_limit()} questions for today. "
            "The limit resets at midnight UTC.",
        )

    try:
        # a new chat gets its ID now, so the trace can be tagged with it;
        # the chat itself is only saved once there is an answer
        chat_id = body.chat_id or str(uuid.uuid4())
        result = rag.ask(
            question, services.collection, services.client, services.langfuse,
            owner=user.email, chat_id=chat_id,
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
    services.store.count_question(user.email, today())

    response = AskResponse(
        answer=result["answer"],
        covered=result["covered"],
        best_score=result["best_score"],
        sources=[
            Source(source=c["source"], page=c["page"], score=c["score"], text=c["text"])
            for c in result["chunks"]
        ],
        trace_id=result["trace_id"],
        trace_url=result["trace_url"],
        chat_id=body.chat_id or services.store.create_chat(
            user.email, chat_store.make_title(question), chat_id=chat_id,
        ),
    )
    # everything the website needs to show the answer again later
    details = response.model_dump(include={"covered", "best_score", "sources", "trace_id", "trace_url"})
    services.store.add_exchange(user.email, response.chat_id, question, response.answer, details)
    return response


@protected.get("/chats", response_model=list[ChatSummary])
def list_chats(services: ServicesDep, user: UserDep) -> list[ChatSummary]:
    return [ChatSummary(**chat) for chat in services.store.list_chats(user.email)]


@protected.get("/chats/{chat_id}", response_model=list[ChatMessage])
def get_chat(chat_id: str, services: ServicesDep, user: UserDep) -> list[ChatMessage]:
    messages = services.store.get_messages(user.email, chat_id)
    if messages is None:  # someone else's chat looks exactly like a missing one
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found.")
    return [ChatMessage(**message) for message in messages]


@protected.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(chat_id: str, services: ServicesDep, user: UserDep) -> None:
    if not services.store.delete_chat(user.email, chat_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found.")


def require_tracing(services: Services) -> None:
    if not services.tracing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Feedback and metrics need Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
        )


@protected.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def feedback(body: FeedbackRequest, services: ServicesDep, user: UserDep) -> None:
    require_tracing(services)
    # only answers in your own chats: a guessed trace ID gets "not found"
    if not services.store.owns_trace(user.email, body.trace_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found.")
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
    user: UserDep,
    days: Annotated[int, Query(ge=1, le=30)] = 7,  # the free plan keeps 30 days
    scope: Literal["me", "all"] = "me",
) -> MetricsResponse:
    require_tracing(services)
    if scope == "all" and user.email != auth.owner_email():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner can see everyone's numbers.")
    try:
        numbers = metrics.summary(services.langfuse, days, user_id=user.email if scope == "me" else None)
    except Exception as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Couldn't load the numbers from Langfuse. Try again in a minute.",
        ) from error
    people = None
    if scope == "all":
        since = today() - timedelta(days=days - 1)
        people = [PersonUsage(**row) for row in services.store.usage_by_person(since)]
    return MetricsResponse(**numbers, scope=scope, people=people)


# --- App ---

def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # runs once when the server starts; tests pass their own services
        app.state.services = services or create_services()
        yield
        app.state.services.langfuse.flush()
        app.state.services.store.close()

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
