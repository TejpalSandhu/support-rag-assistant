from types import SimpleNamespace

import rag_assistant as rag
from conftest import FAKE_ANSWER

FILAMENT_PAGE = (
    "Loading filament: cut the filament tip, push the filament into the "
    "extruder until the gears grab it, then confirm on the screen."
)
NOZZLE_PAGE = "Cleaning the nozzle: heat the nozzle and push a thin needle into it."


def upload(client, name, data):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


# --- Health and access key ---

def test_health_needs_no_key(client, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_KEY", "secret-123")
    assert client.get("/api/health").json() == {"status": "ok"}


def test_protected_endpoints_need_the_right_key(client, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_KEY", "secret-123")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/documents", headers={"X-API-Key": "secret-123"}).status_code == 200


# --- Uploading ---

def test_upload_indexes_the_pdf(client, services, make_pdf):
    response = upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE, NOZZLE_PAGE]))
    assert response.status_code == 202
    assert response.json()["status"] == "processing"

    # TestClient runs background tasks before returning, so it's done now
    documents = client.get("/api/documents").json()
    assert documents == [{"name": "manual.pdf", "chunks": 2, "status": "ready"}]
    assert (services.pdf_dir / "manual.pdf").exists()


def test_upload_rejects_files_that_are_not_pdfs(client, make_pdf):
    assert upload(client, "notes.txt", b"hello").status_code == 400
    assert upload(client, "fake.pdf", b"just text, not a PDF").status_code == 400


def test_upload_rejects_files_over_the_size_limit(client, monkeypatch, make_pdf):
    monkeypatch.setattr("api.MAX_UPLOAD_BYTES", 100)
    assert upload(client, "big.pdf", make_pdf([FILAMENT_PAGE])).status_code == 413


def test_upload_cannot_escape_the_pdf_folder(client, services, make_pdf, tmp_path):
    response = upload(client, "../../evil.pdf", make_pdf([NOZZLE_PAGE]))
    assert response.json()["name"] == "evil.pdf"
    assert (services.pdf_dir / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_pdf_without_text_is_reported_and_removed(client, services, make_pdf):
    upload(client, "scanned.pdf", make_pdf([""]))
    [document] = client.get("/api/documents").json()
    assert document["status"].startswith("failed: No extractable text")
    assert not (services.pdf_dir / "scanned.pdf").exists()


def test_reupload_replaces_the_old_version(client, services, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    upload(client, "manual.pdf", make_pdf([NOZZLE_PAGE]))
    stored = services.collection.get(include=["documents"])["documents"]
    assert stored == [NOZZLE_PAGE]


# --- Deleting ---

def test_delete_removes_file_and_chunks(client, services, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    assert client.delete("/api/documents/manual.pdf").status_code == 204
    assert client.get("/api/documents").json() == []
    assert not (services.pdf_dir / "manual.pdf").exists()
    assert client.delete("/api/documents/manual.pdf").status_code == 404


# --- Asking ---

def test_ask_answers_from_the_uploaded_pdf(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE, NOZZLE_PAGE]))
    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})

    body = response.json()
    assert response.status_code == 200
    assert body["covered"] is True
    assert body["answer"] == FAKE_ANSWER
    assert body["sources"][0] == {
        "source": "manual.pdf", "page": 1,
        "score": body["sources"][0]["score"], "text": FILAMENT_PAGE,
    }
    assert fake_gemini.generator.calls == 1


def test_ask_refuses_without_calling_the_model(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    body = client.post("/api/ask", json={"question": "Who won the World Cup in 2022?"}).json()

    assert body["covered"] is False
    assert body["answer"] == rag.NOT_COVERED_MESSAGE
    assert fake_gemini.generator.calls == 0


def test_ask_with_no_documents_refuses(client, fake_gemini):
    body = client.post("/api/ask", json={"question": "How do I load filament?"}).json()
    assert body["covered"] is False
    assert body["sources"] == []
    assert fake_gemini.generator.calls == 0


def test_ask_rejects_empty_questions(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_busy_model_returns_503(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("Busy", (Exception,), {"status_code": 503})()

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 503
    # one quick retry for a temporary server error, then give up
    assert fake_gemini.generator.calls == 2


def test_model_rate_limit_returns_429(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("RateLimitError", (Exception,), {"status_code": 429})()

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 429
    assert "usage limit" in response.json()["detail"]
    assert fake_gemini.generator.calls == 1  # waiting wouldn't help, so no retry


def test_model_timeout_returns_503(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("APITimeoutError", (Exception,), {})()

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 503
    assert fake_gemini.generator.calls == 1  # already waited the full limit
    assert fake_gemini.generator.timeout_ms == rag.GENERATION_TIMEOUT_SECONDS * 1000
