import json
from types import SimpleNamespace

import pytest

TRACE_ID = "549d3b0af7e2aae54c27028ee260eb9a"


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
    return services.langfuse


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
        "days": 30, "environment": "production",
        "questions": 10, "answered": 6, "refused": 3, "failed": 1,
        "avg_seconds": 2.4, "p95_seconds": 6.1,
        "feedback_votes": 4, "helpful_rate": 0.75,
    }
    # every query is limited to this environment
    for query in langfuse.queries:
        assert {"column": "environment", "operator": "=", "value": "production",
                "type": "string"} in query["filters"]


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
