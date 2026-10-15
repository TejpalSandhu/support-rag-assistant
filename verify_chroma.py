"""
Sanity check: does Chroma rank chunks the same way as plain cosine similarity?

Scores every stored chunk by hand with cosine_similarity(), asks Chroma for
its top results for the same question, and compares the two lists.

Run after rag_assistant.py has built the collection:
    python verify_chroma.py
    python verify_chroma.py "How do I clean the nozzle?"
"""

import sys

from rag_assistant import (
    TOP_K,
    cosine_similarity,
    get_collection,
    get_embedding,
    make_gemini_client,
)

DEFAULT_QUESTION = "How do I load filament on the MINI?"


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("Collection is empty - run rag_assistant.py first.")

    client = make_gemini_client()
    query_embedding = get_embedding(question, client)

    # brute force: score every stored vector ourselves
    stored = collection.get(include=["embeddings"])
    by_hand = sorted(
        (
            (cosine_similarity(query_embedding, list(vector)), chunk_id)
            for chunk_id, vector in zip(stored["ids"], stored["embeddings"])
        ),
        reverse=True,
    )[:TOP_K]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["distances"],
    )
    by_chroma = list(zip(results["ids"][0], results["distances"][0]))

    print(f"Question: {question}")
    print(f"Compared {len(stored['ids'])} chunks in '{collection.name}'\n")
    print(f"{'rank':<5}{'by hand (cosine)':<52}{'Chroma (1 - distance)':<52}")

    max_difference = 0.0
    for rank, ((hand_score, hand_id), (chroma_id, distance)) in enumerate(
        zip(by_hand, by_chroma), start=1
    ):
        chroma_score = 1 - distance
        if hand_id == chroma_id:
            max_difference = max(max_difference, abs(hand_score - chroma_score))
        mark = "" if hand_id == chroma_id else "  <-- different"
        print(
            f"{rank:<5}{hand_id[:42]:<43}{hand_score:.6f}  "
            f"{chroma_id[:42]:<43}{chroma_score:.6f}{mark}"
        )

    same_order = [chunk_id for _, chunk_id in by_hand] == [
        chunk_id for chunk_id, _ in by_chroma
    ]
    print(f"\nSame top-{TOP_K} order: {same_order}")
    # Chroma stores float32, so tiny differences (~1e-7) are expected
    print(f"Largest score difference: {max_difference:.2e}")


if __name__ == "__main__":
    main()
