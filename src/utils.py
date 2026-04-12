# ─────────────────────────────────────────────────────────────
# utils.py
# Purpose: Shared utilities for corpus construction and text
#          preprocessing used by both BM25 and semantic search.
# ─────────────────────────────────────────────────────────────

import re
import duckdb
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
META_PARQUET_PATH = REPO_ROOT / "data" / "processed" / "meta_All_Beauty.parquet"

#print(f"Looking for file at {META_PARQUET_PATH}")
#print(f"File exists: {META_PARQUET_PATH.exists()}")


# ── Document Construction ─────────────────────────────────────


def build_document(row):
    """
    Goal: combine all relevant text fields into one searchable string.
    This is what both BM25 and the embedding model will index.
    
    Fields combined: title + features + description + store + details
    - title:       concise product name, always present
    - features:    bullet points describing product attributes
    - description: longer form product context
    - store:       brand name, not always captured in details
    - details:     skin type, material, brand, size etc.
    """
    parts = []

    # Title - most important signal, always include
    if pd.notna(row['title']) and row['title']:
        parts.append(str(row['title']))

    # Features - numpy array from parquet, use len() not isinstance()
    try:
        if row['features'] is not None and len(row['features']) > 0:
            parts.append(" ".join(str(f) for f in row['features']))
    except (TypeError, ValueError):
        pass

    # Description - same type-agnostic approach as features
    try:
        if row['description'] is not None and len(row['description']) > 0:
            parts.append(" ".join(str(d) for d in row['description']))
    except (TypeError, ValueError):
        pass

    # Store - captures brand names missing from details
    if pd.notna(row['store']) and row['store']:
        parts.append(str(row['store']))

    # Details - JSON string with brand, material, skin type etc.
    if pd.notna(row['details']) and row['details']:
        parts.append(str(row['details']))

    return " ".join(parts)


# ── Text Preprocessing ────────────────────────────────────────

def tokenize(text):
    """
    Goal: convert a raw document string into a list of clean tokens
    for BM25 indexing. Better tokenization = better keyword matching.

    Steps:
    1. Lowercase - so "Moisturizer" and "moisturizer" match
    2. Remove punctuation - so "skin-care" doesn't become one token
    3. Split on whitespace
    4. Remove stopwords - "the", "a", "for" add noise not signal
    5. Remove short tokens - single letters add no meaning
    
    Note: we do NOT stem/lemmatize here (converting "moisturizing" 
    to "moistur") - it can hurt precision for product searches where
    exact terms matter.
    """

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "it",
        "this", "that", "are", "was", "be", "has", "have"
    }

    # 1: lowercase
    text = text.lower()

    # 2: remove punctuation and special characters
    # re.sub replaces anything that isn't a letter, number, or space with a space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3: split into tokens on whitespace
    tokens = text.split()

    #  4 & 5: remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return tokens

# ── Corpus Loading ────────────────────────────────────────────

def load_corpus(
    parquet_path=META_PARQUET_PATH,
    limit=None
):
    """
    Goal: load product metadata from parquet and build the document
    corpus used by both BM25 and semantic search.

    Returns two parallel lists:
    - documents: raw text strings (used by semantic search)
    - metadata:  dicts with title, price, rating etc. (used for display)

    Why parallel lists? BM25 and FAISS return integer indices.
    We use those indices to look up the original product info.
    
    limit: int or None - load a subset for testing, None = full dataset
    """
    limit_str = f"LIMIT {limit}" if limit else ""

    # Load only the columns we need - use DuckDB to read these efficiently
    query = f"""
        SELECT 
            parent_asin,
            title,
            features,
            description,
            store,
            details,
            price,
            average_rating,
            rating_number
        FROM read_parquet('{parquet_path}')
        {limit_str}
    """

    df = duckdb.query(query).df()

    # Build document strings and metadata dicts in one pass
    documents = []  # searchable text for each product
    metadata = []   # display info for each product

    for _, row in df.iterrows():
        # Build the searchable document string
        documents.append(build_document(row))

        # Keep metadata for displaying results in the app
        metadata.append({
            "parent_asin":    row['parent_asin'],
            "title":          row['title'],
            "price":          row['price'],
            "average_rating": row['average_rating'],
            "rating_number":  row['rating_number'],
            "store":          row['store']
        })

    print(f"Loaded {len(documents)} documents from corpus")
    return documents, metadata