# ─────────────────────────────────────────────────────────────
# bm25.py
# Purpose: Build, save, load and query a BM25 keyword search 
#          index over the All_beauty product corpus
# ─────────────────────────────────────────────────────────────

import pickle
import numpy as np
from pathlib import Path 
from rank_bm25 import BM25Okapi

#from src.utils import load_corpus, tokenize
try:
    from src.utils import load_corpus, tokenize
except ModuleNotFoundError:
    from utils import load_corpus, tokenize
 

# Deafult path to save/load the BM25 index
# Using __fle__ so path is always relative to this file's location
REPO_ROOT = Path(__file__).parent.parent
BM25_INDEX_PATH = REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"

# ── Build ─────────────────────────────────────────────────────

def build_bm25_index(documents):
    """
    Goal: tokenize all documents and fit a BM25 index over them.
    This is the 'training' step - done once, then saved to disk.

    BM25Okapi expects a list of token lists:
    [["gentle", "cleanser"], ["retinol", "serum", "vitamin"], ...]

    Returns the fitted BM25Okapi object.
    """
    print("Tokenizing corpus for BM25...")

    # Tokenize every document - converts raw strings to token lists
    tokenized_corpus = [tokenize(doc) for doc in documents]

    print(f"  Tokenized {len(tokenized_corpus)} documents")
    print(f"  Sample tokens: {tokenized_corpus[4][:10]}")

    # Fit BM25 on the tokenized corpus
    # This computes IDF for every word and stores TF per document
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index built successfully")
    return bm25, tokenized_corpus

# ── Save ──────────────────────────────────────────────────────

def save_bm25_index(bm25, metadata, tokenized_corpus,
                    index_path=BM25_INDEX_PATH,
                    metadata_path=CORPUS_METADATA_PATH):
    """
    Goal: persist the BM25 index and corpus metadata to disk
    so the app can load them instantly without rebuilding.

    We save three things:
    - bm25 object: the fitted index with all IDF/TF scores
    - metadata: product info (title, price, rating) for display
    - tokenized_corpus: needed for some BM25 operations
    """
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Save BM25 index and tokenized corpus together
    # wb = write binary - pickle always uses binary mode
    print(f"Saving BM25 index to {index_path}...")
    with open(index_path, "wb") as f:
        #pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
        pickle.dump({"bm25": bm25}, f)  # tokenized_corpus removed to slim indexes 

    # Save metadata separately - used by both BM25 and semantic search
    print(f"Saving corpus metadata to {metadata_path}...")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("Saved successfully")

# ── Load ──────────────────────────────────────────────────────

def load_bm25_index(index_path=BM25_INDEX_PATH,
                    metadata_path=CORPUS_METADATA_PATH):
    """
    Goal: load a previously saved BM25 index from disk.
    Callled every time the app starts, so it can serve queries instantly - much faster than rebuilding.

    Returns bm25 object, metadata list, tokenized_corpus list.
    """
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {index_path}. "
            f"Run build_bm25_index() first."
        )

    # rb = read binary
    print(f"Loading BM25 index from {index_path}...")
    with open(index_path, "rb") as f:
        saved = pickle.load(f)

    bm25 = saved["bm25"]
    #tokenized_corpus = saved["tokenized_corpus"]

    print(f"Loading corpus metadata from {metadata_path}...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded index with {len(metadata)} documents")
    return bm25, metadata, [] # empty lists keeps callers from breaking

# ── Search ────────────────────────────────────────────────────

def search_bm25(query, bm25, metadata, top_k=5):
    """
    Goal: given a natural language query, return the top_k most 
    relevant products according to BM25 keyword matching.

    Steps:
    1. Tokenize the query the same way we tokenized the documents
    (critical - query and docs must use identical tokenization for BM25 to work).
    2. Get BM25 scores for every document 
    3. Return top_k results with metadata and scores

    Returns a lust of dicts, each containing product info + score.
    """

    # 1. Tokenize query using same function as corpus
    # If tokenized differently, score would be meaningless
    query_tokens = tokenize(query)
    print(f"Query tokens: {query_tokens}")

    # 2. score every document against the query
    # get_scores returns a numpy array of shape (n_documents,)
    # each value us the BM25 score for that document
    scores = bm25.get_scores(query_tokens)

    # diagnostic 
    # print(f"Scores shape: {scores.shape}")
    # print(f"Max score: {scores.max()}")
    # print(f"Non-zero scores: {np.count_nonzero(scores)}")
    # print(f"Top 5 indices: {np.argsort(scores)[::-1][:5]}")
    # print(f"Top 5 scores: {scores[np.argsort(scores)[::-1][:5]]}")

    # 3. get indices of top_k highest scores
    # argsort sorts ascending, so reverse with [::-1] and take top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 4. build results list with metadata and scores
    results = []
    for idx in top_indices:
        result = metadata[idx].copy() # copy so we dontt modify original metadata
        result["score"] = round(float(scores[idx]), 4)
        result["rank"] = len(results) + 1
        results.append(result)
    return results

# ── Main: build and save index ────────────────────────────────

if __name__ == "__main__":
    # Step 1: load corpus from parquet
    print("=== Loading corpus ===")
    documents, metadata = load_corpus()

    # Step 2: build BM25 index
    print("\n=== Building BM25 index ===")
    bm25, tokenized_corpus = build_bm25_index(documents)

    # Step 3: save index and metadata to disk
    print("\n=== Saving index ===")
    save_bm25_index(bm25, metadata, tokenized_corpus)

    # Step 4: quick search test to verify everything works
    print("\n=== Test search ===")
    test_queries = [
        "gentle cleanser for sensitive skin",
        "CeraVe moisturizer",
        "vitamin C serum",
        "something to keep my face hydrated all day"
    ]

    # Load from disk to verify save/load works correctly
    bm25_loaded, metadata_loaded, _ = load_bm25_index()

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_bm25(query, bm25_loaded, metadata_loaded, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] {r['title'][:60]} {r['parent_asin'][:60]} (score: {r['score']})")

