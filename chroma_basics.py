"""
Lesson 6.1: Store your chunks in a vector database (Chroma) and search it.

What this script does:
1. Reads the 12 chunks you already embedded (embedded_chunks.json).
   -> No new embedding API calls for the documents.
2. Saves them into a Chroma database stored in a folder called chroma_db.
3. Embeds ONE question with Gemini (one API call).
4. Asks Chroma for the most similar chunks and prints them.

Setup:
- Run this in the same folder as embedded_chunks.json and .env
- Install: python -m pip install chromadb
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai


# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# The file your earlier script created. We reuse its embeddings.
JSON_PATH = BASE_DIR / "embedded_chunks.json"

# Chroma will create this folder and store its database inside it.
CHROMA_PATH = BASE_DIR / "chroma_db"

# A "collection" is like one table / one labelled box inside the database.
COLLECTION_NAME = "dog_documents"

EMBEDDING_MODEL = "gemini-embedding-001"

QUESTION = "Why might a dog stop frequently during a walk?"

# How many matching chunks Chroma should return.
N_RESULTS = 3


# ============================================================
# 3. READ THE CHUNKS WE ALREADY HAVE
# ============================================================

def load_chunks_from_json(json_path: Path) -> list[dict]:
    """Return the list of chunks saved by the multi-PDF script."""
    with json_path.open("r", encoding="utf-8") as file:
        cache_data = json.load(file)

    # The file looks like {"embedding_model": "...", "chunks": [...]}
    return cache_data["chunks"]


# ============================================================
# 4. GIVE EVERY CHUNK A UNIQUE, STABLE ID
# ============================================================

def make_chunk_ids(chunks: list[dict]) -> list[str]:
    """Create IDs like 'my_document.pdf-p1-c2' (file, page 1, chunk 2).

    The same chunk always gets the same ID, so storing it again
    updates it instead of creating a duplicate.
    """
    ids = []
    count_per_page = {}  # remembers how many chunks we've seen on each page

    for chunk in chunks:
        page_key = (chunk["source"], chunk["page"])

        # .get(page_key, 0) means "current count, or 0 if this page is new"
        count_per_page[page_key] = count_per_page.get(page_key, 0) + 1
        chunk_number = count_per_page[page_key]

        ids.append(f"{chunk['source']}-p{chunk['page']}-c{chunk_number}")

    return ids


# ============================================================
# 5. PUT THE CHUNKS INTO CHROMA
# ============================================================

def store_chunks(collection, chunks: list[dict]) -> None:
    """Save every chunk's ID, text, vector and metadata in the collection."""
    collection.upsert(
        ids=make_chunk_ids(chunks),
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in chunks
        ],
    )


# ============================================================
# 6. EMBED THE QUESTION (same function as before)
# ============================================================

def get_embedding(text: str, client) -> list[float]:
    """Convert text into a vector using Gemini."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("The embedding API returned no vector.")

    return response.embeddings[0].values


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    # --- Gemini client (only used to embed the question) ---
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from your .env file")
    gemini_client = genai.Client(api_key=api_key)

    # --- Step 1: read the chunks we embedded earlier ---
    chunks = load_chunks_from_json(JSON_PATH)
    print(f"Loaded {len(chunks)} chunks from {JSON_PATH.name}")

    # --- Step 2: open (or create) the database on disk ---
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # --- Step 3: open (or create) our collection ---
    # "space": "cosine" tells Chroma to compare vectors with cosine,
    # the same measure you wrote by hand in cosine_similarity().
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    # --- Step 4: store the chunks ---
    store_chunks(collection, chunks)
    print(f"Chunks now stored in Chroma: {collection.count()}")

    # --- Step 5: embed the question ---
    question_embedding = get_embedding(QUESTION, gemini_client)

    # --- Step 6: ask Chroma for the closest chunks ---
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    # --- Step 7: print the results ---
    # Chroma can answer many questions at once, so every field is a
    # list of lists. We asked ONE question, so we take item [0].
    ids = results["ids"][0]
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\nQuestion:", QUESTION)
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, texts, metadatas, distances), start=1
    ):
        similarity = 1 - distance  # turn Chroma's distance back into your score
        print(
            f"{rank}. {chunk_id} | {metadata['source']}, page {metadata['page']} "
            f"| distance {distance:.4f} | similarity {similarity:.4f}"
        )
        print(f"   {text[:90]}...")


if __name__ == "__main__":
    main()