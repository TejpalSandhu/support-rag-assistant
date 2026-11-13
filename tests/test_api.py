from types import SimpleNamespace

import chromadb
import pytest
from fastapi.testclient import TestClient

import api
import rag_assistant as rag
from conftest import FAKE_ANSWER

FILAMENT_PAGE = (
    "Loading filament: cut the filament tip, push the filament into the "
    "extruder until the gears grab it, then confirm on the screen."
)
NOZZLE_PAGE = "Cleaning the nozzle: heat the nozzle and push a thin needle into it."


def upload(client, name, data):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


# --- Health ---

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_backend_can_serve_the_website(services, monkeypatch, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>Support RAG Assistant</h1>")
    monkeypatch.setenv("STATIC_DIR", str(site))

    with TestClient(api.create_app(services)) as client:
        assert "Support RAG Assistant" in client.get("/").text
        assert client.get("/api/health").json() == {"status": "ok"}  # API still wins


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


# --- Backups (so the index survives a fresh container) ---

def test_upload_saves_a_backup_and_delete_removes_it(client, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE, NOZZLE_PAGE]))
    backup = rag.BACKUP_DIR / "manual.pdf.json"
    assert backup.exists()

    client.delete("/api/documents/manual.pdf")
    assert not backup.exists()


def test_restore_rebuilds_an_empty_index_without_calling_gemini(client, fake_gemini, make_pdf, tmp_path):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE, NOZZLE_PAGE]))
    embed_calls = fake_gemini.models.embed_calls

    # a brand-new, empty Chroma, like a container that just started
    fresh = chromadb.PersistentClient(path=str(tmp_path / "fresh_db")).get_or_create_collection(
        name="fresh", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
    )
    assert rag.restore_backups(fresh) == ["manual.pdf"]
    assert rag.list_documents(fresh) == {"manual.pdf": 2}
    assert fake_gemini.models.embed_calls == embed_calls

    # already there: nothing to do the second time
    assert rag.restore_backups(fresh) == []


class FakeBlobContainer:
    """Stands in for azure.storage.blob.ContainerClient, keeping blobs in a dict."""

    def __init__(self):
        self.blobs = {}

    def upload_blob(self, name, data, overwrite=False):
        self.blobs[name] = data

    def get_blob_client(self, name):
        return SimpleNamespace(exists=lambda: name in self.blobs)

    def delete_blob(self, name):
        del self.blobs[name]

    def list_blobs(self):
        return [SimpleNamespace(name=name) for name in self.blobs]

    def download_blob(self, name):
        return SimpleNamespace(readall=lambda: self.blobs[name])


def test_blob_backups_save_load_and_delete():
    backups = rag.BlobBackups(FakeBlobContainer())
    backups.save("b.pdf.json", b"two")
    backups.save("a.pdf.json", b"one")
    assert backups.load_all() == [("a.pdf.json", b"one"), ("b.pdf.json", b"two")]

    backups.delete("a.pdf.json")
    backups.delete("missing.pdf.json")  # no error for a file that isn't there
    assert backups.load_all() == [("b.pdf.json", b"two")]


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
    # the main model, then the fallback once; both busy, so give up
    assert fake_gemini.generator.models == [rag.GENERATION_MODEL, rag.FALLBACK_MODEL]


def test_fallback_model_answers_when_the_main_one_is_busy(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("Busy", (Exception,), {"status_code": 503})()
    fake_gemini.generator.failing_models = {rag.GENERATION_MODEL}

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 200
    assert response.json()["answer"] == FAKE_ANSWER
    assert fake_gemini.generator.models == [rag.GENERATION_MODEL, rag.FALLBACK_MODEL]


def test_no_fallback_for_errors_that_switching_cannot_fix(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    # e.g. 400 bad request: the fallback would fail the same way
    BadRequest = type("BadRequest", (Exception,), {"status_code": 400})
    fake_gemini.generator.error = BadRequest()

    # an unexpected error is raised (a 500 for real users; TestClient re-raises it)
    with pytest.raises(BadRequest):
        client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert fake_gemini.generator.models == [rag.GENERATION_MODEL]


def test_fallback_can_be_turned_off(client, fake_gemini, make_pdf, monkeypatch):
    monkeypatch.setattr(rag, "FALLBACK_MODEL", "none")
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("Busy", (Exception,), {"status_code": 503})()

    assert client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"}).status_code == 503
    assert fake_gemini.generator.models == [rag.GENERATION_MODEL]


def test_model_rate_limit_returns_429(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("RateLimitError", (Exception,), {"status_code": 429})()

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 429
    assert "usage limits" in response.json()["detail"]
    # the fallback has its own daily limit, so it's worth asking
    assert fake_gemini.generator.calls == 2


def test_model_timeout_returns_503(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("APITimeoutError", (Exception,), {})()

    response = client.post("/api/ask", json={"question": "How do I push the filament into the extruder?"})
    assert response.status_code == 503
    assert fake_gemini.generator.calls == 2  # main model, then the fallback
    assert fake_gemini.generator.timeout_ms == rag.GENERATION_TIMEOUT_SECONDS * 1000
