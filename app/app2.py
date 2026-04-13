import streamlit as st
import pickle
import random
import sys
import os
import re
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# REPO_ROOT is the base directory for relative paths
REPO_ROOT = Path(__file__).parent.parent
FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.npy"

# BM25 index and metadata paths
BM25_INDEX_PATH = REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"

# Tokenization function for BM25
def tokenize(text):
    """
    Tokenize the input text for BM25 indexing.
    """
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "it",
        "this", "that", "are", "was", "be", "has", "have"
    }

    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return tokens

# Load the semantic FAISS index
def load_semantic_index(index_path=FAISS_INDEX_PATH):
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build_faiss_index() first.")
    
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index

# Load BM25 index and metadata
def load_bm25_index(index_path=BM25_INDEX_PATH, metadata_path=CORPUS_METADATA_PATH):
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run build_bm25_index() first.")
    
    print(f"Loading BM25 index from {index_path}...")
    with open(index_path, "rb") as f:
        saved = pickle.load(f)
    
    bm25 = saved["bm25"]
    tokenized_corpus = saved["tokenized_corpus"]
    
    print(f"Loading corpus metadata from {metadata_path}...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {len(metadata)} documents")
    return bm25, metadata, tokenized_corpus

# Search function for semantic queries using FAISS
def search_semantic(query, index, metadata, model, top_k=5):
    query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(query_vector.astype(np.float32), top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        result = metadata[idx].copy()
        result["score"] = round(float(score), 4)
        result["rank"] = rank + 1
        results.append(result)

    return results

# Search function for BM25 queries
def search_bm25(query, bm25, metadata, top_k=5):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = metadata[idx].copy()
        result["score"] = round(float(scores[idx]), 4)
        result["rank"] = len(results) + 1
        results.append(result)

    return results

# Load BM25 index if not already loaded
if 'bm25_loaded' not in st.session_state:
    bm25, tokenized_corpus, _ = load_bm25_index()
    st.session_state.bm25_loaded = True
    st.session_state.bm25 = bm25
    st.session_state.tokenized_corpus = tokenized_corpus

# Load semantic index if not already loaded
if 'semantic_loaded' not in st.session_state:
    faiss_index = load_semantic_index()
    st.session_state.semantic_loaded = True
    st.session_state.faiss_index = faiss_index
    st.session_state.metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the sentence transformer model

# Streamlit interface for search mode selection
search_mode = st.radio("Select Search Mode", ["BM25", "Semantic", "Hybrid"])
query = st.text_input("Enter your query:")

# Display results if the query is not empty
if query:
    if search_mode == "BM25":
        results = search_bm25(query, st.session_state.bm25, st.session_state.tokenized_corpus, top_k=3)
    elif search_mode == "Semantic":
        results = search_semantic(query, st.session_state.faiss_index, st.session_state.metadata, st.session_state.model, top_k=3)
    else:
        results = []  # Handle Hybrid search or other modes if needed

    st.subheader(f"Top 3 results for '{query}' using {search_mode} search:")

    feedback_data = []  # Store feedback here

    # Loop through each result and display it
    for idx, result in enumerate(results):
        with st.expander(f"Result {idx + 1}: {result['title']}"):
            st.write(f"**Product Title:** {result['title']}")
            truncated_review = f"Simulated review for {result['title'][:200]}..."  # Replace with real review
            st.write(f"**Review:** {truncated_review}")
            st.write(f"**Retrieval Score:** {result['score']:.2f}")

            feedback = st.radio(f"Was this result helpful?", options=["👍", "👎"], key=f"feedback_{idx}")
            if feedback:
                feedback_data.append({
                    "product_title": result["title"],
                    "feedback": feedback,
                    "score": result["score"]
                })

#     # Save feedback to CSV
#     if feedback_data:
#         store_feedback(feedback_data)
#         st.success("Feedback saved!")

# # Function to save feedback in CSV
# def store_feedback(feedback_data):
#     import pandas as pd
#     df = pd.DataFrame(feedback_data)
#     df.to_csv("feedback.csv", mode="a", header=False, index=False)