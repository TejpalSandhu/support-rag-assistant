"""
Pick the minimum similarity score below which the assistant says
"the docs don't cover this" instead of calling the model.

Takes the best retrieval score of every question in eval_questions.json
(answerable and unanswerable) for the current chunk setting, shows how
several thresholds split the two groups, and recommends the highest one
that still answers every answerable question. Saves the result to
threshold_results.md.

    python choose_threshold.py

Needs the current collection to exist (run rag_assistant.py once first).
Question vectors come from evaluate.py's cache, so re-runs are free.
"""

import json
from datetime import datetime

from evaluate import EVAL_PATH, embed_questions
from rag_assistant import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DailyQuotaError,
    get_collection,
    make_gemini_client,
    search,
)

RESULTS_PATH = BASE_DIR / "threshold_results.md"


def best_scores(questions: list[dict], vectors: list[list[float]], collection) -> list[float]:
    return [search(collection, vector, top_k=1)[0]["score"] for vector in vectors]


def rates(threshold: float, answerable: list[float], unanswerable: list[float]) -> tuple[float, float]:
    """(share of answerable questions answered, share of unanswerable refused)."""
    answered = sum(score >= threshold for score in answerable) / len(answerable)
    refused = sum(score < threshold for score in unanswerable) / len(unanswerable)
    return answered, refused


def recommend(answerable: list[float], unanswerable: list[float]) -> float:
    """Highest threshold that still answers every answerable question.

    Refusing a real customer question is worse than letting a near-miss
    through, because the prompt already tells the model to refuse when the
    context doesn't support an answer. So the cutoff sits halfway between the
    lowest answerable score and the highest unanswerable score below it.
    """
    lowest_answerable = min(answerable)
    below = [score for score in unanswerable if score < lowest_answerable]
    if not below:
        return round(lowest_answerable - 0.01, 3)
    return round((lowest_answerable + max(below)) / 2, 3)


def main() -> None:
    with EVAL_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    answerable_qs, unanswerable_qs = data["answerable"], data["unanswerable"]

    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("Collection is empty - run rag_assistant.py first.")

    client = make_gemini_client()
    try:
        vectors = embed_questions(
            [q["question"] for q in answerable_qs + unanswerable_qs], client
        )
    except DailyQuotaError as error:
        raise SystemExit(str(error))

    scores = best_scores(answerable_qs + unanswerable_qs, vectors, collection)
    answerable = scores[: len(answerable_qs)]
    unanswerable = scores[len(answerable_qs):]

    print(f"Best retrieval score per question ({CHUNK_SIZE}/{CHUNK_OVERLAP}):\n")
    rows = [(s, "answerable", q["id"]) for s, q in zip(answerable, answerable_qs)]
    rows += [(s, f"NOT covered ({q['kind']})", q["id"]) for s, q in zip(unanswerable, unanswerable_qs)]
    for score, kind, qid in sorted(rows, reverse=True):
        print(f"  {score:.3f}  {kind:<25} {qid}")

    threshold = recommend(answerable, unanswerable)

    table = ["| threshold | answerable answered | unanswerable refused |", "|---:|---:|---:|"]
    for t in sorted({0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, threshold}):
        answered, refused = rates(t, answerable, unanswerable)
        mark = " **(recommended)**" if t == threshold else ""
        table.append(f"| {t:.3f}{mark} | {answered:.0%} | {refused:.0%} |")
    table_text = "\n".join(table)

    print(f"\n{table_text}\n\nRecommended MIN_SCORE = {threshold}")

    RESULTS_PATH.write_text(
        "# Threshold results\n\n"
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | chunk setting {CHUNK_SIZE}/{CHUNK_OVERLAP} "
        f"| {len(answerable)} answerable, {len(unanswerable)} unanswerable questions\n\n"
        "Lowest answerable best score: "
        f"{min(answerable):.3f} | highest unanswerable best score: {max(unanswerable):.3f}\n\n"
        f"{table_text}\n",
        encoding="utf-8",
    )
    print(f"Saved to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
