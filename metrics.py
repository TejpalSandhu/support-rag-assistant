"""
Numbers for the website's metrics page, read from Langfuse.

Every question is a "support-question" trace (see rag_assistant.ask). It has
an "answer" step only when the model was called, and a 👍/👎 click is saved as
a true/false "user-feedback" score on that trace. So:

    answered = "answer" steps that didn't fail
    failed   = questions that ended in an error (model busy, quota, ...)
    refused  = the rest: stopped by the relevance check, no model call

Uses Langfuse's Metrics API, which is real-time (new data shows up within
seconds), and only counts traces from this environment (LANGFUSE_TRACING_ENVIRONMENT,
"production" on Azure), so local tests don't mix with the live site.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

FEEDBACK_SCORE = "user-feedback"


def tracing_environment() -> str:
    # the Langfuse SDK tags every trace with this (its default is "default")
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or "default"


def name_in(*names: str) -> dict[str, Any]:
    return {"column": "name", "operator": "any of", "value": list(names), "type": "stringOptions"}


def run_query(
    langfuse: Any,
    view: str,
    measures: list[tuple[str, str]],
    filters: list[dict[str, Any]],
    days: int,
    dimensions: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """One Metrics API query over the last `days` days; returns its rows."""
    now = datetime.now(timezone.utc)
    query = {
        "view": view,
        "dimensions": [{"field": field} for field in dimensions],
        "metrics": [{"measure": measure, "aggregation": how} for measure, how in measures],
        "filters": [
            {"column": "environment", "operator": "=", "value": tracing_environment(), "type": "string"},
            *filters,
        ],
        "fromTimestamp": (now - timedelta(days=days)).isoformat(),
        "toTimestamp": now.isoformat(),
    }
    return langfuse.api.metrics.metrics(query=json.dumps(query)).data


def seconds(milliseconds: float | None) -> float | None:
    return None if milliseconds is None else round(milliseconds / 1000, 1)


def summary(langfuse: Any, days: int) -> dict[str, Any]:
    # 1. how many questions, and how long they took (end to end)
    rows = run_query(
        langfuse, "observations",
        [("count", "count"), ("latency", "avg"), ("latency", "p95")],
        [name_in("support-question")], days,
    )
    timing = rows[0] if rows else {}

    # 2. questions and answer steps, split by level (ERROR = it failed)
    counts = {
        (row["name"], row["level"]): int(row["count_count"])
        for row in run_query(
            langfuse, "observations", [("count", "count")],
            [name_in("support-question", "answer")], days,
            dimensions=("name", "level"),
        )
    }
    questions = sum(n for (name, _), n in counts.items() if name == "support-question")
    failed = counts.get(("support-question", "ERROR"), 0)
    answered = sum(
        n for (name, level), n in counts.items() if name == "answer" and level != "ERROR"
    )

    # 3. feedback: the average of true(1)/false(0) votes is the 👍 rate
    rows = run_query(
        langfuse, "scores-boolean", [("count", "count"), ("value", "avg")],
        [name_in(FEEDBACK_SCORE)], days,
    )
    feedback = rows[0] if rows else {}
    votes = int(feedback.get("count_count") or 0)

    return {
        "days": days,
        "environment": tracing_environment(),
        "questions": questions,
        "answered": answered,
        "refused": max(questions - failed - answered, 0),
        "failed": failed,
        "avg_seconds": seconds(timing.get("avg_latency")) if questions else None,
        "p95_seconds": seconds(timing.get("p95_latency")) if questions else None,
        "feedback_votes": votes,
        "helpful_rate": round(feedback["avg_value"], 3) if votes else None,
    }
