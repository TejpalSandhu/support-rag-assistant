"""
Retrieval evaluation: how often does the right page show up in the top-k?

For each chunk setting, builds (or reuses) its Chroma collection, searches
for every answerable question in eval_questions.json and records the rank of
the first chunk from the expected file + page. Prints a results table and
saves it to eval_results.md.

    python evaluate.py

First run embeds every new chunk setting (a few minutes with rate limits);
later runs only re-read the PDFs and embed the questions (1 API call).
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from rag_assistant import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    DailyQuotaError,
    chunk_documents,
    embed_with_retry,
    get_collection,
    load_pages,
    make_gemini_client,
    open_collection,
    search,
    sync_collection,
)

EVAL_PATH = BASE_DIR / "eval_questions.json"
RESULTS_PATH = BASE_DIR / "eval_results.md"

# (chunk size, overlap) pairs to compare - each gets its own collection
CHUNK_SETTINGS = [(100, 0), (100, 20), (100, 50), (200, 40), (300, 60)]
TOP_K_VALUES = [1, 3, 5, 12]
MAX_K = max(TOP_K_VALUES)


def embed_questions(texts: list[str], client) -> list[list[float]]:
    """Question vectors, cached in Chroma so each question is embedded only once."""
    cache = open_collection(f"questions_{EMBEDDING_MODEL}")
    ids = [hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] for text in texts]

    stored = cache.get(ids=ids, include=["embeddings"])
    vectors = {
        question_id: list(vector)
        for question_id, vector in zip(stored["ids"], stored["embeddings"])
    }

    missing = [(qid, text) for qid, text in zip(ids, texts) if qid not in vectors]
    if missing:
        new_vectors = embed_with_retry([text for _, text in missing], client)
        cache.upsert(
            ids=[qid for qid, _ in missing],
            embeddings=new_vectors,
            documents=[text for _, text in missing],
        )
        vectors.update(zip([qid for qid, _ in missing], new_vectors))

    print(f"Question vectors: {len(texts) - len(missing)} cached, {len(missing)} embedded")
    return [vectors[qid] for qid in ids]


def first_hit_rank(
    results: list[dict[str, Any]],
    source: str,
    pages: list[int],
) -> int | None:
    """1-based rank of the first result from the expected file + page, or None."""
    for rank, chunk in enumerate(results, start=1):
        if chunk["source"] == source and chunk["page"] in pages:
            return rank
    return None


def score_ranks(ranks: list[int | None]) -> dict[str, float]:
    """hit@k for each k, plus MRR (mean of 1/rank, 0 when not found)."""
    scores = {
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
        for k in TOP_K_VALUES
    }
    scores["mrr"] = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return scores


def evaluate_setting(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    questions: list[dict[str, Any]],
    question_vectors: list[list[float]],
    client,
) -> tuple[dict[str, Any], list[int | None], list[dict[str, Any]]]:
    chunks = chunk_documents(pages, chunk_size=chunk_size, overlap=overlap)
    collection = get_collection(chunk_size, overlap)
    sync_collection(collection, chunks, client)

    ranks: list[int | None] = []
    top_results: list[dict[str, Any]] = []
    for question, vector in zip(questions, question_vectors):
        results = search(collection, vector, top_k=MAX_K)
        ranks.append(first_hit_rank(results, question["source"], question["pages"]))
        top_results.append(results[0])

    row = {"chunk_size": chunk_size, "overlap": overlap, "chunks": len(chunks)}
    row.update(score_ranks(ranks))
    return row, ranks, top_results


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table - readable in the terminal and on GitHub."""
    header = ["chunk size", "overlap", "chunks"] + [f"hit@{k}" for k in TOP_K_VALUES] + ["MRR"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" for _ in header) + "|",
    ]
    for row in rows:
        cells = [str(row["chunk_size"]), str(row["overlap"]), str(row["chunks"])]
        cells += [f"{row[f'hit@{k}']:.0%}" for k in TOP_K_VALUES]
        cells.append(f"{row['mrr']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_misses(
    questions: list[dict[str, Any]],
    ranks: list[int | None],
    top_results: list[dict[str, Any]],
    k: int = 5,
) -> None:
    """Questions whose expected page was not in the top k - the ones to study."""
    misses = [
        (question, rank, top)
        for question, rank, top in zip(questions, ranks, top_results)
        if rank is None or rank > k
    ]
    print(f"\nMisses at top-{k} for the current setting "
          f"({CHUNK_SIZE}/{CHUNK_OVERLAP}): {len(misses)} of {len(questions)}")
    for question, rank, top in misses:
        found = f"rank {rank}" if rank else f"not in top {MAX_K}"
        print(f"- {question['id']}: expected p{question['pages']} -> {found}; "
              f"top result was {top['source']} p{top['page']} ({top['score']:.3f})")


def main() -> None:
    with EVAL_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)["answerable"]

    client = make_gemini_client()
    pages = load_pages()

    # a question's vector doesn't depend on chunk settings: embed once, reuse
    try:
        question_vectors = embed_questions([q["question"] for q in questions], client)
    except DailyQuotaError as error:
        print(f"\n{error}")
        return

    rows = []
    current = None
    for chunk_size, overlap in CHUNK_SETTINGS:
        print(f"\n--- chunk size {chunk_size}, overlap {overlap} ---")
        try:
            row, ranks, top_results = evaluate_setting(
                pages, chunk_size, overlap, questions, question_vectors, client
            )
        except DailyQuotaError as error:
            # keep the settings that finished; the rest resume next run
            print(f"\n{error}\nStopping here - re-run later to finish the rest.")
            break
        rows.append(row)
        if (chunk_size, overlap) == (CHUNK_SIZE, CHUNK_OVERLAP):
            current = (ranks, top_results)

    if not rows:
        return

    table = format_table(rows)
    print(f"\nRetrieval results ({len(questions)} answerable questions):\n")
    print(table)

    if current:
        print_misses(questions, *current)

    run_info = (
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | {len(questions)} answerable questions "
        f"| embedding model: {EMBEDDING_MODEL}"
    )
    if len(rows) < len(CHUNK_SETTINGS):
        run_info += f" | PARTIAL: {len(rows)} of {len(CHUNK_SETTINGS)} settings"
    RESULTS_PATH.write_text(
        f"# Retrieval evaluation\n\n{run_info}\n\n"
        "hit@k = share of questions whose expected page is in the top k results. "
        "MRR = average of 1/rank of the first correct result (1.0 = always first).\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
