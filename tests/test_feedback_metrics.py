import json
from types import SimpleNamespace

from datetime import date

import pytest
from fastapi.testclient import TestClient

import api
import auth
from test_auth import fake_verify

TRACE_ID = "549d3b0af7e2aae54c27028ee260eb9a"
ME = auth.LOCAL_USER_EMAIL


class FakeLangfuse:
    """Records scores and answers Metrics API queries with canned rows."""

    def __init__(self, rows=None, error=None):
        self.scores = []
        self.queries = []
        self.rows = rows or {}
        self.error = error
        self.api = SimpleNamespace(metrics=SimpleNamespace(metrics=self._metrics))

    def create_score(self, **score):
        self.scores.append(score)

    def flush(self):
        pass

    def _metrics(self, query):
        if self.error:
            raise self.error
        query = json.loads(query)
        self.queries.append(query)
        # pick the canned rows by view and whether it groups by level
        key = (query["view"], tuple(d["field"] for d in query["dimensions"]))
        return SimpleNamespace(data=self.rows.get(key, []))


@pytest.fixture
def langfuse(services):
    services.langfuse = FakeLangfuse()
    services.tracing = True
    # an answer with this trace ID in one of my chats, so I may rate it
    chat = services.store.create_chat(ME, "a chat")
    services.store.add_exchange(ME, chat, "q", "a", {"trace_id": TRACE_ID})
    return services.langfuse


USER_FILTER = {"column": "userId", "operator": "=", "value": ME, "type": "string"}


# --- Feedback ---

def test_feedback_is_saved_as_a_score_on_the_trace(client, langfuse):
    response = client.post("/api/feedback", json={"trace_id": TRACE_ID, "helpful": True})
    assert response.status_code == 204
    assert langfuse.scores == [{
        "name": "user-feedback", "value": 1, "data_type": "BOOLEAN",
        "trace_id": TRACE_ID, "score_id": f"{TRACE_ID}-feedback",
    }]


def test_changing_the_vote_reuses_the_same_score_id(client, langfuse):
    client.post("/api/feedback", json={"trace_id": TRACE_ID, "helpful": True})
    client.post("/api/feedback", json={"trace_id": TRACE_ID, "helpful": False})
    first, second = langfuse.scores
    assert second["value"] == 0
    assert first["score_id"] == second["score_id"]  # Langfuse replaces the first


def test_you_can_only_rate_answers_from_your_own_chats(client, langfuse):
    someone_elses = "0123456789abcdef0123456789abcdef"
    response = client.post("/api/feedback", json={"trace_id": someone_elses, "helpful": True})
    assert response.status_code == 404
    assert langfuse.scores == []


def test_feedback_rejects_a_made_up_trace_id(client, langfuse):
    for bad in ["", "abc", "../../etc", TRACE_ID.upper()]:
        response = client.post("/api/feedback", json={"trace_id": bad, "helpful": True})
        assert response.status_code == 422
    assert langfuse.scores == []


def test_feedback_and_metrics_need_langfuse_keys(client):
    # the default test services have tracing off
    assert client.post("/api/feedback", json={"trace_id": TRACE_ID, "helpful": True}).status_code == 503
    assert client.get("/api/metrics").status_code == 503


# --- Metrics ---

def test_metrics_summary(client, langfuse, monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "production")
    langfuse.rows = {
        ("observations", ()): [{"count_count": 10, "avg_latency": 2440, "p95_latency": 6100}],
        ("observations", ("name", "level")): [
            {"name": "support-question", "level": "DEFAULT", "count_count": 9},
            {"name": "support-question", "level": "ERROR", "count_count": 1},
            {"name": "answer", "level": "DEFAULT", "count_count": 6},
            {"name": "answer", "level": "ERROR", "count_count": 1},
        ],
        ("scores-boolean", ()): [{"count_count": 4, "avg_value": 0.75}],
    }

    body = client.get("/api/metrics?days=30").json()
    assert body == {
        "days": 30, "scope": "me", "environment": "production",
        "questions": 10, "answered": 6, "refused": 3, "failed": 1,
        "avg_seconds": 2.4, "p95_seconds": 6.1,
        "feedback_votes": 4, "helpful_rate": 0.75, "people": None,
    }
    # every query is limited to this environment, and to me
    for query in langfuse.queries:
        assert {"column": "environment", "operator": "=", "value": "production",
                "type": "string"} in query["filters"]
        assert USER_FILTER in query["filters"]


def test_owner_sees_everyone_and_a_table_per_person(client, langfuse, services):
    # login is off, so the local user is the owner
    today = api.today()
    services.store.count_question("friend@gmail.com", today)
    services.store.count_question("friend@gmail.com", today)
    services.store.count_question(ME, today)
    services.store.count_question("old@gmail.com", date(2000, 1, 1))  # outside the range

    body = client.get("/api/metrics?days=7&scope=all").json()
    assert body["scope"] == "all"
    assert body["people"] == [
        {"email": "friend@gmail.com", "questions": 2, "last_active": today.isoformat()},
        {"email": ME, "questions": 1, "last_active": today.isoformat()},
    ]
    for query in langfuse.queries:  # no per-person filter: everyone
        assert USER_FILTER not in query["filters"]


def test_only_the_owner_sees_everyone(services, monkeypatch):
    services.langfuse = FakeLangfuse()
    services.tracing = True
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setenv("SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ALLOWED_EMAILS", "owner@gmail.com,friend@gmail.com")
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    with TestClient(api.create_app(services)) as web:
        web.post("/api/auth/google", json={"credential": "valid:friend@gmail.com"})
        assert web.get("/api/me").json()["is_owner"] is False
        assert web.get("/api/metrics?scope=all").status_code == 403
        assert web.get("/api/metrics").status_code == 200

        web.post("/api/auth/google", json={"credential": "valid:owner@gmail.com"})
        assert web.get("/api/me").json()["is_owner"] is True
        assert web.get("/api/metrics?scope=all").status_code == 200


def test_metrics_with_no_data_yet(client, langfuse):
    body = client.get("/api/metrics").json()
    assert body["days"] == 7
    assert (body["questions"], body["answered"], body["refused"], body["failed"]) == (0, 0, 0, 0)
    assert body["avg_seconds"] is None
    assert body["helpful_rate"] is None


def test_metrics_days_must_be_1_to_30(client, langfuse):
    assert client.get("/api/metrics?days=0").status_code == 422
    assert client.get("/api/metrics?days=31").status_code == 422


def test_metrics_when_langfuse_fails(client, langfuse):
    langfuse.error = RuntimeError("Langfuse is down")
    response = client.get("/api/metrics")
    assert response.status_code == 502
    assert "Langfuse" in response.json()["detail"]
