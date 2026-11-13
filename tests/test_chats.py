"""Saved chats through the API: every answer is stored in a chat of its owner."""

from fastapi.testclient import TestClient

import api
from test_owners import FILAMENT_PAGE, FRIEND, OWNER, people, upload  # noqa: F401 (fixture)

QUESTION = {"question": "How do I push the filament into the extruder?"}


def test_first_question_starts_a_chat_named_after_it(client, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    answer = client.post("/api/ask", json=QUESTION).json()

    [chat] = client.get("/api/chats").json()
    assert chat["id"] == answer["chat_id"]
    assert chat["title"] == QUESTION["question"]

    question, reply = client.get(f"/api/chats/{chat['id']}").json()
    assert question == {"role": "user", "content": QUESTION["question"], "details": None}
    assert reply["role"] == "assistant"
    assert reply["content"] == answer["answer"]
    assert reply["details"]["covered"] is True
    assert reply["details"]["sources"] == answer["sources"]  # shown again on reload


def test_follow_up_questions_go_into_the_same_chat(client, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    chat_id = client.post("/api/ask", json=QUESTION).json()["chat_id"]

    again = client.post("/api/ask", json={"question": "And the extruder gears?", "chat_id": chat_id})
    assert again.json()["chat_id"] == chat_id
    assert len(client.get("/api/chats").json()) == 1
    assert len(client.get(f"/api/chats/{chat_id}").json()) == 4


def test_refused_questions_are_saved_too(client):
    answer = client.post("/api/ask", json={"question": "Who won the World Cup?"}).json()
    assert answer["covered"] is False
    [_, reply] = client.get(f"/api/chats/{answer['chat_id']}").json()
    assert reply["details"]["covered"] is False


def test_failed_answers_are_not_saved(client, fake_gemini, make_pdf):
    upload(client, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    fake_gemini.generator.error = type("Busy", (Exception,), {"status_code": 503})()
    assert client.post("/api/ask", json=QUESTION).status_code == 503
    assert client.get("/api/chats").json() == []


def test_unknown_chat_ids(client):
    assert client.get("/api/chats/not-a-chat").status_code == 404
    assert client.delete("/api/chats/not-a-chat").status_code == 404
    assert client.post("/api/ask", json={**QUESTION, "chat_id": "not-a-chat"}).status_code == 404


def test_delete_a_chat(client):
    chat_id = client.post("/api/ask", json={"question": "Who won the World Cup?"}).json()["chat_id"]
    assert client.delete(f"/api/chats/{chat_id}").status_code == 204
    assert client.get("/api/chats").json() == []


def test_chats_are_private(people, make_pdf):  # noqa: F811 (fixture)
    owner, friend = people
    upload(owner, "manual.pdf", make_pdf([FILAMENT_PAGE]))
    chat_id = owner.post("/api/ask", json=QUESTION).json()["chat_id"]

    assert friend.get("/api/chats").json() == []
    assert friend.get(f"/api/chats/{chat_id}").status_code == 404
    assert friend.delete(f"/api/chats/{chat_id}").status_code == 404
    # can't add questions to someone else's chat either
    assert friend.post("/api/ask", json={**QUESTION, "chat_id": chat_id}).status_code == 404
    assert len(owner.get(f"/api/chats/{chat_id}").json()) == 2


def test_daily_counter_survives_a_restart(services, monkeypatch):
    # the counter lives in the store (the database), not in the server's memory
    monkeypatch.setenv("DAILY_QUESTION_LIMIT", "1")
    with TestClient(api.create_app(services)) as first:
        assert first.post("/api/ask", json={"question": "Who won the World Cup?"}).status_code == 200
    with TestClient(api.create_app(services)) as after_restart:
        assert after_restart.get("/api/me").json()["questions_left_today"] == 0
        assert after_restart.post("/api/ask", json={"question": "Another one?"}).status_code == 429
