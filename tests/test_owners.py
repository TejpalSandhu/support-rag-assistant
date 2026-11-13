"""Each signed-in person only sees, searches and deletes their own PDFs."""

import json

import chromadb
import pytest
from fastapi.testclient import TestClient

import api
import auth
import rag_assistant as rag
from test_auth import fake_verify

OWNER = "owner@gmail.com"
FRIEND = "friend@gmail.com"
FILAMENT_PAGE = "Loading filament: push the filament into the extruder until the gears grab it."
NOZZLE_PAGE = "Cleaning the nozzle: heat the nozzle and push a thin needle into it."
FILAMENT_QUESTION = {"question": "How do I push the filament into the extruder?"}


@pytest.fixture
def people(services, monkeypatch):
    """Two browsers: the owner and a friend, each signed in."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setenv("SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ALLOWED_EMAILS", f"{OWNER},{FRIEND}")
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    app = api.create_app(services)
    with TestClient(app) as owner, TestClient(app) as friend:
        owner.post("/api/auth/google", json={"credential": f"valid:{OWNER}"})
        friend.post("/api/auth/google", json={"credential": f"valid:{FRIEND}"})
        yield owner, friend


def upload(client, name, pdf):
    return client.post("/api/documents", files={"file": (name, pdf, "application/pdf")})


def names(client):
    return [doc["name"] for doc in client.get("/api/documents").json()]


def test_people_only_see_their_own_documents(people, make_pdf):
    owner, friend = people
    upload(owner, "owner-notes.pdf", make_pdf([FILAMENT_PAGE]))
    upload(friend, "friend-notes.pdf", make_pdf([NOZZLE_PAGE]))

    assert names(owner) == ["owner-notes.pdf"]
    assert names(friend) == ["friend-notes.pdf"]


def test_two_people_can_each_have_a_file_with_the_same_name(people, services, make_pdf):
    owner, friend = people
    upload(owner, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    upload(friend, "manual.pdf", make_pdf([NOZZLE_PAGE]))

    assert rag.list_documents(services.collection, OWNER) == {"manual.pdf": 1}
    assert rag.list_documents(services.collection, FRIEND) == {"manual.pdf": 1}
    # different files on disk and in the backups
    assert api.pdf_path(services, OWNER, "manual.pdf") != api.pdf_path(services, FRIEND, "manual.pdf")
    assert (rag.BACKUP_DIR / rag.backup_name(OWNER, "manual.pdf")).exists()
    assert (rag.BACKUP_DIR / rag.backup_name(FRIEND, "manual.pdf")).exists()


def test_nobody_can_delete_someone_elses_file(people, services, make_pdf):
    owner, friend = people
    upload(owner, "manual.pdf", make_pdf([FILAMENT_PAGE]))

    assert friend.delete("/api/documents/manual.pdf").status_code == 404
    assert names(owner) == ["manual.pdf"]  # still there

    # and deleting your own copy leaves theirs alone
    upload(friend, "manual.pdf", make_pdf([NOZZLE_PAGE]))
    assert friend.delete("/api/documents/manual.pdf").status_code == 204
    assert names(owner) == ["manual.pdf"]
    assert names(friend) == []


def test_questions_only_search_your_own_documents(people, make_pdf):
    owner, friend = people
    upload(owner, "manual.pdf", make_pdf([FILAMENT_PAGE]))

    mine = owner.post("/api/ask", json=FILAMENT_QUESTION).json()
    assert mine["covered"] is True
    assert mine["sources"][0]["text"] == FILAMENT_PAGE

    # the friend has no documents, so the owner's manual must not be found
    theirs = friend.post("/api/ask", json=FILAMENT_QUESTION).json()
    assert theirs["covered"] is False
    assert theirs["sources"] == []


def test_the_owner_is_the_first_allowed_email(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "x")
    monkeypatch.setenv("ALLOWED_EMAILS", "Boss@Gmail.com, friend@gmail.com")
    assert auth.owner_email() == "boss@gmail.com"


def test_backups_from_before_accounts_go_to_the_owner(services, tmp_path):
    # a backup in the old format: no owner, at the top of the backup folder
    old = {
        "ids": ["job.pdf-p1-c1-abcd1234"],
        "embeddings": [[0.1] * 8],
        "documents": ["Role: AI platform engineering intern."],
        "metadatas": [{"source": "job.pdf", "page": 1}],
    }
    rag.BACKUP_DIR.mkdir(parents=True)
    (rag.BACKUP_DIR / "job.pdf.json").write_text(json.dumps(old))

    fresh = chromadb.PersistentClient(path=str(tmp_path / "fresh")).get_or_create_collection(
        name="fresh", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
    )
    rag.restore_backups(fresh, legacy_owner=OWNER)

    assert rag.list_documents(fresh, OWNER) == {"job.pdf": 1}
    # moved to the owner's folder in the new format, old file gone
    assert not (rag.BACKUP_DIR / "job.pdf.json").exists()
    moved = json.loads((rag.BACKUP_DIR / rag.backup_name(OWNER, "job.pdf")).read_text())
    assert moved["metadatas"][0]["owner"] == OWNER
    assert moved["ids"][0].startswith(rag.owner_key(OWNER) + "/")


def test_owner_codes_hide_the_email():
    key = rag.owner_key("friend@gmail.com")
    assert len(key) == 16 and "@" not in key
    assert rag.owner_key("Friend@Gmail.com") == key  # same person, same code
