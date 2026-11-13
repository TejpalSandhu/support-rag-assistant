# Support RAG Assistant

[![CI](https://github.com/TejpalSandhu/support-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/TejpalSandhu/support-rag-assistant/actions/workflows/ci.yml)

A question-answering assistant for PDFs, built with Retrieval-Augmented
Generation (RAG). Invited users sign in with Google, upload their own PDFs and
chat with them; every answer cites the pages it came from, and questions the
documents don't cover are refused instead of guessed.

It started as a customer-support bot for two 3D-printer manuals and grew into
a deployed web app. The focus throughout is on the parts that matter in
production:

- **Retrieval quality measured and tuned with data**: hit@k and MRR on a labelled question set
- **Refusing what the documents don't cover**: a calibrated score threshold plus a grounded prompt
- **Observability**: every answer traced in Langfuse, with user feedback and a live metrics page
- **Multi-user and private**: Google sign-in, an email allowlist, per-user documents, chats and daily limits
- **Shipped like a real service**: 96 automated tests, CI/CD with GitHub Actions, Docker, and automatic deploys to Azure

## Results at a glance

Measured on the two Prusa printer manuals (113 pages):

| | |
|---|---|
| Right page in the top 5 results | **100%** of 28 test questions |
| Right page ranked first | **86%** (MRR 0.91) |
| Context sent to the model after tuning | **12 → 5 chunks** (~58% less), same hit rate |
| Questions the manuals don't cover that were refused | **8 of 8**: 5 by the score threshold (no model call), 3 by the model |
| Hallucinated answers in the end-to-end check | **0** |

Full numbers and decisions are in [experiments.md](experiments.md).

## Architecture

```mermaid
flowchart LR
    U[Browser<br/>React] -->|HTTPS + Google sign-in| A[Azure Container Apps<br/>FastAPI serving API + website<br/>scales to zero]
    A -->|embeddings, answers| G[Gemini API<br/>main + fallback model]
    A --> P[(Neon PostgreSQL<br/>chats, daily usage)]
    A --> B[(Azure Blob Storage<br/>index backups)]
    A -.->|traces, feedback| L[Langfuse<br/>metrics API]
    H[GitHub Actions] -->|tests pass: build image,<br/>push to GHCR, deploy via OIDC| A
```

**How a question is answered:**

```mermaid
flowchart LR
    A[PDF] -->|pypdf, page by page| B[Chunks<br/>100 words, 20 overlap]
    B -->|Gemini embeddings,<br/>batched, only new chunks| C[(Chroma<br/>vector DB)]
    Q[Question] -->|embed| D[Vector search<br/>top 5, cosine,<br/>only this user's chunks]
    C --> D
    D --> E{Best score<br/>>= MIN_SCORE?}
    E -->|no| F[Not covered.<br/>Model not called]
    E -->|yes| G[Gemini answers from the<br/>chunks only, with page citations]
    D -.-> L[Langfuse trace]
    E -.-> L
    G -.-> L
```

1. **Ingest:** each PDF page is split into overlapping 100-word chunks. A chunk
   never spans two pages, so every answer can cite a page.
2. **Embed and store:** chunks are embedded with `gemini-embedding-001` in
   batches and stored in a persistent Chroma collection. Each chunk ID includes
   a hash of its text, so re-runs embed only new or changed chunks, and every
   batch is saved right away, so a crash mid-build loses nothing.
3. **Retrieve:** the question is embedded and Chroma returns the 5 nearest
   chunks of the signed-in user's documents. `verify_chroma.py` checks that
   Chroma's ranking matches a hand-written cosine similarity over all chunks.
4. **Two refusal layers:**
   - If even the best match scores below `MIN_SCORE`, the assistant says the
     documents don't cover it and skips the model call.
   - Otherwise the prompt tells the model to answer only from the retrieved
     text and to say when it doesn't have enough information.
5. **Answer, with a fallback:** if the main model is busy, over its free-tier
   limit or too slow, a second model is asked straight away (never for errors
   like a bad request, which a second model would hit too).
6. **Trace:** every question becomes a Langfuse trace with the retrieved
   chunks, the relevance check, the prompt, the answer, the model that answered,
   token usage and the time each step took.

## Evaluation

`eval_questions.json` holds 28 customer-style questions, each with the file
and page that answers it, plus 8 questions the manuals can't answer (5 near
misses such as "How much does the XL cost?", and 3 off-topic).

`evaluate.py` measures **hit@k** (is the right page in the top k?) and **MRR**
(how high is the first right result?) for several chunk settings:

| chunk size | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 358 | 79% | 89% | 100% | 0.87 |
| **100** | **20** | **409** | **86%** | **93%** | **100%** | **0.91** |
| 100 | 50 | 557 | 82% | 93% | 93% | 0.88 |
| 200 | 40 | 219 | 86% | 93% | 100% | 0.90 |
| 300 | 60 | 159 | 79% | 96% | 100% | 0.87 |

- **Overlap helps, up to a point.** 0 → 20 words moved 3 questions to rank 1.
  At 50, near-duplicate chunks crowd the top 5.
- **hit@5 = 100%**, so `TOP_K` went from 12 to 5: less noise and a shorter prompt.
- **Main failure mode is product confusion.** The XL's nearly identical
  nozzle-cleaning page outranks the MINI's for a MINI question.

`choose_threshold.py` picks `MIN_SCORE` from the best-match scores of all 36
questions. Near-miss questions can score **higher** than real ones (0.73 vs
0.65), so a threshold alone can't stop them. That's why the prompt is the
second layer.

**The threshold is a per-deployment setting.** 0.631 was tuned on the
manuals, but on the live site a broad question about a 2-page job description
scored 0.599 and was wrongly refused. The live site uses `MIN_SCORE=0.55`: in
the evaluation every answerable question still passes, clearly unrelated
questions (0.46–0.52) are still stopped without a model call, and the prompt
refuses the near misses that get through.

## The web app

### Sign-in and access

1. The website shows Google's sign-in button; Google gives the browser a
   signed ID token for the chosen account.
2. The backend verifies Google's signature, that the token was issued for this
   app, that the email is verified and that it's in `ALLOWED_EMAILS`.
3. It then sets a session cookie for 7 days: signed with `SESSION_SECRET`,
   HttpOnly (page scripts can't read it) and SameSite (other sites can't make
   the browser send it). The allowlist is checked on every request, so removing
   someone locks them out immediately.

The live site runs with `REQUIRE_LOGIN=true`: if the Google settings are ever
missing, it refuses to start rather than run open to everyone.

### Each person's data stays theirs

- **Documents:** every chunk stores its owner, and listing, searching and
  deleting only ever see the signed-in person's chunks, so two people can each
  have a `manual.pdf`. Files and backups live under a short hash of the email
  (`<owner code>/manual.pdf.json`), so addresses don't appear in paths. The
  website uses its own Chroma collection, separate from the evaluation's manuals.
- **Chats** are saved in PostgreSQL ([Neon](https://neon.com), free tier), so
  they survive refreshes and work across devices:

  ```
  chats        id | owner | title | created_at | updated_at
  messages     id | chat_id -> chats | role | content | details (sources, score, trace) | created_at
  daily_usage  owner | day | questions
  ```

  Every query filters by owner, so someone else's chat ID just returns "not
  found". A question and its answer are saved in one transaction, and the daily
  counter is a single `INSERT ... ON CONFLICT DO UPDATE`, so two questions at
  once can't lose a count.
- **Daily limit:** each person gets `DAILY_QUESTION_LIMIT` questions a day
  (default 20), so one user can't use up the shared free Gemini quota. Failed
  answers don't count.
- **Feedback** is only accepted on answers in your own chats.

### Feedback and metrics

Each answer has 👍/👎 buttons; a click is saved as a Langfuse score on that
answer's trace (changing your vote replaces it). The **Metrics** page reads
Langfuse's real-time Metrics API: questions asked, answered, refused before
calling the model, failed, average and p95 response time, and the 👍 rate.
Everyone sees their own numbers; the owner (the first email in
`ALLOWED_EMAILS`) can switch to **Everyone**, which adds a questions-per-person
table from the database.

## Tracing with Langfuse

With Langfuse keys set, each question shows up as a trace:

```
support-question                 question in, final answer out; user + chat (session)
├── embed-question   (embedding)  model, time taken
├── vector-search    (retriever)  the 5 chunks with source, page and score
├── relevance-check  (guardrail)  best score vs MIN_SCORE -> covered or not
└── answer           (generation) full prompt, answer, model used, token usage
```

Traces carry the user's email and the chat ID, so each chat appears as one
Langfuse session, and the live site tags its traces with the environment
`production` so local experiments never mix into the live metrics. Without
keys the assistant runs the same way and tracing is simply off.

## Deployment

Every push to `main` runs the CI/CD pipeline in
[.github/workflows/ci.yml](.github/workflows/ci.yml):

```
backend tests (pytest + a throwaway PostgreSQL) ─┐
website lint, tests, build (ESLint, Vitest)      ├─► Docker image builds ─► deploy to Azure
                                                 ┘   (only for main)
```

The deploy job builds one image (API plus the built website), pushes it to
GitHub Container Registry tagged with the commit ID, logs in to Azure with
OIDC (no stored password; the identity only trusts this repository's `main`
branch) and points the container app at the new image, then checks the live
`/api/health`.

- **One container that scales to zero:** it only runs (and costs) while in
  use; the first visit after a quiet spell takes a few seconds longer.
- **Storage:** Chroma lives on the container's own disk, because its SQLite
  file is unreliable on network drives. After a PDF is indexed, its chunks and
  vectors are backed up as JSON to Azure Blob Storage, and a fresh container
  rebuilds the index from those backups at startup without calling Gemini.
- **Secrets** (Gemini, Langfuse, session secret, database URL) are Azure
  secrets, copied from `.env` by `deploy/set-azure-secrets.ps1`.
- **Region:** Korea Central. The Gemini API rejects requests from Hong Kong
  (the first region tried), and the student subscription only offers
  lightweight "express" environments: one container per app and no mounted
  storage, which shaped the single-container and Blob-backup design.

## Run it locally

Requirements: Python 3.12, a free [Gemini API key](https://aistudio.google.com/apikey),
and [Node.js](https://nodejs.org) 20+ for the website.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env, then add your keys
```

Only `GEMINI_API_KEY` is required. Without `GOOGLE_CLIENT_ID` login is off
(you're "local@localhost"), without `DATABASE_URL` chats are kept in memory,
and without Langfuse keys tracing is off.

| Command | What it does |
|---|---|
| `uvicorn api:app --reload` | The web API; try it at http://127.0.0.1:8000/docs |
| `cd frontend && npm install && npm run dev` | The website at http://localhost:5173 (talks to the API above) |
| `docker compose up --build` | API + website in containers at http://localhost:8080 |
| `pytest` | Backend tests (fake Gemini and Google, no quota used) |
| `cd frontend && npm test && npm run lint` | Website tests and code checks |
| `python rag_assistant.py` | The original terminal assistant over `pdfs/` |
| `python evaluate.py` | Retrieval evaluation across chunk settings → `eval_results.md` |
| `python choose_threshold.py` | Recommends `MIN_SCORE` → `threshold_results.md` |
| `python verify_chroma.py` | Compares Chroma's ranking with brute-force cosine similarity |

The terminal assistant and the evaluation scripts use the Prusa manuals in
`pdfs/` (not redistributed in this repo):

```bash
mkdir pdfs
curl -L -o pdfs/prusa3d_manual_mini_en.pdf https://www.prusa3d.com/downloads/manual/prusa3d_manual_mini_en.pdf
curl -L -o pdfs/prusa3d_manual_xl_en.pdf https://www.prusa3d.com/downloads/manual/prusa3d_manual_xl_en.pdf
```

To run the store tests against a real PostgreSQL as CI does:

```bash
docker run -d --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:18
set TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres   # macOS/Linux: export ...
pytest
```

## Configuration

All settings are environment variables (locally in `.env`; see [.env.example](.env.example)).

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini embeddings and answers (required) |
| `GENERATION_MODEL` / `FALLBACK_MODEL` | Answer model, and the one asked when it's busy (`none` turns the fallback off) |
| `MIN_SCORE` | Relevance threshold (default 0.631; 0.55 on the live site) |
| `GOOGLE_CLIENT_ID` | Turns on Google sign-in |
| `SESSION_SECRET` | Signs session cookies (32+ random characters) |
| `ALLOWED_EMAILS` | Who may sign in, comma-separated; the first is the owner |
| `DAILY_QUESTION_LIMIT` | Questions per person per day (default 20) |
| `DATABASE_URL` | PostgreSQL for chats and daily usage |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Tracing, feedback and metrics |
| `LANGFUSE_TRACING_ENVIRONMENT` | Tags traces (`production` on the live site) |
| `AZURE_STORAGE_CONNECTION_STRING` | Back up the index to Blob Storage instead of a local folder |
| `CHROMA_PATH`, `DATA_DIR`, `STATIC_DIR` | Where the index, data and built website live |
| `COOKIE_SECURE`, `REQUIRE_LOGIN` | `true` on the live site: HTTPS-only cookies, never start without sign-in |

## Web API

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Is the server up? (no login needed) |
| `GET /api/auth/config` | Google client ID for the sign-in button |
| `POST /api/auth/google` | Sign in with a Google ID token; sets the session cookie |
| `POST /api/auth/logout` | Sign out |
| `GET /api/me` | Who is signed in, owner or not, questions left today |
| `GET /api/documents` | Your PDFs with chunk counts and status |
| `POST /api/documents` | Upload a PDF; it's indexed in the background |
| `DELETE /api/documents/{name}` | Remove one of your PDFs and its chunks |
| `POST /api/ask` | `{"question": "...", "chat_id": "..."}` → answer, sources, score, trace link, chat ID |
| `GET /api/chats` | Your saved chats, newest first |
| `GET /api/chats/{id}` | One chat's questions and answers |
| `DELETE /api/chats/{id}` | Delete a chat |
| `POST /api/feedback` | `{"trace_id": "...", "helpful": true}`: 👍/👎 on one of your answers |
| `GET /api/metrics?days=7&scope=me` | Your numbers; `scope=all` (owner only) adds everyone and a per-person table |

With sign-in on, every endpoint except health and `/api/auth/...` needs the
session cookie.

## Project structure

```
rag_assistant.py       the RAG core: ingest, embed, store, retrieve, refuse, answer, trace
api.py                 web API (FastAPI) around rag_assistant.py
auth.py                sign in with Google, email allowlist, session cookie
store.py               saved chats and the daily counter (PostgreSQL, or memory)
metrics.py             numbers for the metrics page, from Langfuse's Metrics API
tests/                 pytest suite with fake Gemini and Google; store tests also run on PostgreSQL
frontend/              React website (Vite, Vitest): sign-in, documents, chats, metrics
Dockerfile             one image: the built website + the API (used on Azure)
frontend/Dockerfile    the website served by nginx (used by docker compose)
docker-compose.yml     website + API locally
.github/workflows/     CI/CD: tests, lint, Docker builds; deploys main to Azure
deploy/                script that copies the keys from .env into Azure secrets
evaluate.py            retrieval evaluation (hit@k, MRR) across chunk settings
choose_threshold.py    picks the "docs don't cover this" threshold
verify_chroma.py       checks Chroma's ranking against hand-written cosine similarity
eval_questions.json    28 answerable + 8 unanswerable test questions with expected pages
experiments.md         experiment log: every change with before/after numbers
```

## Limitations and next steps

- **Page-level evaluation is too generous.** A question counts as found if
  any chunk from the right page is retrieved, even if it's not the chunk with
  the answer. Next: chunk-level labels, or judging the final answers.
- **Fixed-size chunks can split a section** or merge two topics. Next: chunk
  by headings.
- **Vocabulary mismatch** ("take out" vs "unload"). Next: hybrid keyword +
  vector search.
- **One replica only.** Upload progress lives in the server's memory and
  Chroma on its disk, so the app can't run several copies at once, and a cold
  start rebuilds the whole index from backups. Next: move the vectors into
  PostgreSQL with pgvector, which would make the app stateless.
- **Small eval set:** 28 questions, so one question is about 3.6 points.

## What I learned

- **Measure before tuning.** The real win, cutting `TOP_K` from 12 to 5, only
  became visible once retrieval was measured.
- **Read the failures, not just the score.** The misses revealed product
  confusion and the page-level evaluation gap, which a single percentage hides.
- **Saying "I don't know" takes two layers.** A threshold stops off-topic
  questions cheaply; near misses score like real ones and need the prompt rule.
- **Real use breaks assumptions.** A threshold tuned on manuals was too strict
  for other documents, and a deployment region blocked the model API. Both were
  found by using the live site, and fixed with settings rather than hacks.
- **Free tiers fail in their own ways.** Per-text embedding quotas, 20 answers
  a day per model and overloaded models led to quota-aware retries, a fallback
  model and per-user limits.
- **Fail closed.** The live site refuses to start without its login settings,
  and every database query includes the owner, so a mistake shows up as an
  error rather than as someone else's data.

## Tech stack

Python 3.12 · Gemini API (`google-genai`) · Chroma · Langfuse · pypdf ·
FastAPI · PostgreSQL (Neon, psycopg 3) · Google Identity Services ·
React 19 (Vite, Vitest) · Docker · GitHub Actions · Azure Container Apps ·
Azure Blob Storage · GitHub Container Registry
