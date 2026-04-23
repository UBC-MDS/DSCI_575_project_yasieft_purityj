# ─────────────────────────────────────────────────────────────
# semantic.py
# Purpose: Build, save, load, and query a semantic search index
#          using sentence embeddings and FAISS vector similarity.
# ─────────────────────────────────────────────────────────────

import torch
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from src.utils import load_corpus
except ModuleNotFoundError:
    from utils import load_corpus

REPO_ROOT = Path(__file__).parent.parent
FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
EMBEDDINGS_PATH  = REPO_ROOT / "data" / "processed" / "embeddings.npy"

# Embedding model - small, fast, good quality for sentence similarity
# 384 dimensions, ~22M parameters, no GPU
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Encode ────────────────────────────────────────────────────

def encode_documents(documents, model, batch_size=256, show_progress=True):
    """
    Goal: convert raw document strings into embedding vectors.
    This is the expensive step - runs each document through the
    neural network. Done once, then saved to disk.

    batch_size: how many documents to encode at once
                larger = faster but more RAM
                256 is a good balance for most laptops

    Returns numpy array of shape (n_documents, 384)
    """
    print(f"Encoding {len(documents)} documents with {MODEL_NAME}...")
    print(f"This may take 2-3 minutes for 112K documents...")

    # encode() handles batching internally
    # show_progress_bar gives a tqdm progress bar so you know its working
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,    # return numpy array not tensor
        normalize_embeddings=True # L2 normalize for cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ── Build FAISS index ─────────────────────────────────────────

def build_faiss_index(embeddings):
    """
    Goal: build a FAISS index over the embeddings for fast
    nearest neighbor search at query time.

    We use IndexFlatIP (Inner Product) because our embeddings
    are normalized - inner product equals cosine similarity
    when vectors are unit length.

    IndexFlatIP is exact search - checks all vectors.
    Fast enough for 112K vectors without approximation.

    Returns fitted FAISS index.
    """
    # embedding dimension - 384 for all-MiniLM-L6-v2
    dimension = embeddings.shape[1]
    print(f"Building FAISS index (dimension={dimension})...")

    # IndexFlatIP = exact search using inner product (cosine similarity)
    # for normalized vectors, inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # Add all embeddings to the index
    # FAISS expects float32 - convert if needed
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index built with {index.ntotal} vectors")
    return index

# ── Save ──────────────────────────────────────────────────────

def save_semantic_index(index, embeddings,
                        index_path=FAISS_INDEX_PATH,
                        embeddings_path=EMBEDDINGS_PATH):
    """
    Goal: persist FAISS index and raw embeddings to disk.
    Saves us from re-encoding 112K documents on every app start.

    We save both:
    - faiss index: for fast search
    - embeddings: useful for debugging and future experiments
    """
    index_path = Path(index_path)
    embeddings_path = Path(embeddings_path)

    # faiss has its own write function - not pickle
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, str(index_path))

    # Save raw embeddings as numpy binary file
    # .npy is numpy's native format - fast to save and load
    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)

    print("Saved successfully")


# ── Load ──────────────────────────────────────────────────────

def load_semantic_index(index_path=FAISS_INDEX_PATH):
    """
    Goal: load a previously saved FAISS index from disk.
    Called every time the app starts - fast, no re-encoding needed.

    Returns the FAISS index ready for searching.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            f"Run build_faiss_index() first."
        )

    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index

# ── Search ────────────────────────────────────────────────────

def search_semantic(query, index, metadata, model, top_k=5):
    """
    Goal: given a natural language query, return top_k most
    semantically similar products.

    Steps:
    1. Encode query into a vector using same model as documents
       (critical - must use identical model for meaningful comparison)
    2. FAISS finds the closest document vectors to query vector
    3. Return results with metadata and similarity scores

    Returns list of dicts with product info and similarity score.
    """

    # Step 1: encode query → vector
    # normalize=True so cosine similarity works correctly
    query_vector = model.encode(
        [query],                    # encode expects a list
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Step 2: search FAISS for top_k nearest neighbors
    # returns two arrays:
    # scores: cosine similarity scores (higher = more similar)
    # indices: positions in our corpus matching those scores
    scores, indices = index.search(
        query_vector.astype(np.float32),
        top_k
    )

    # Step 3: build results - scores and indices are 2D arrays
    # [0] because we only have one query
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        result = metadata[idx].copy()
        result["score"] = round(float(score), 4)
        result["rank"] = rank + 1
        results.append(result)

    return results


# ── Main: build and save index ────────────────────────────────

if __name__ == "__main__":

    # Step 1: load corpus from parquet
    print("=== Loading corpus ===")
    documents, metadata = load_corpus()

    # Step 2: load embedding model
    # downloads ~90MB model on first run, cached after that
    print(f"\n=== Loading embedding model: {MODEL_NAME} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Step 3: encode all documents into vectors
    print("\n=== Encoding documents ===")
    embeddings = encode_documents(documents, model)

    # Step 4: build FAISS index
    print("\n=== Building FAISS index ===")
    index = build_faiss_index(embeddings)

    # Step 5: save index and embeddings to disk
    print("\n=== Saving index ===")
    save_semantic_index(index, embeddings)

    # Step 6: test search to verify everything works
    print("\n=== Test search ===")

    # Load from disk to verify save/load cycle works
    index_loaded = load_semantic_index()

    # Load metadata - shared with BM25
    METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
    with open(METADATA_PATH, "rb") as f:
        metadata_loaded = pickle.load(f)

    test_queries = [
        "gentle cleanser for sensitive skin",
        "something to keep my face hydrated all day",
        "CeraVe moisturizer"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_semantic(query, index_loaded, metadata_loaded, model, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] {r['title'][:60]}  (score: {r['score']})")
    