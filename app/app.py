import streamlit as st
import pickle
import duckdb
import random
import sys
import os
import re
import faiss
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# REPO_ROOT is the base directory for relative paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

from src.utils import tokenize
# from src.data_loader import tokenize
from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic

FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.npy"

# BM25 index and metadata paths
BM25_INDEX_PATH = REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
FEEDBACK_PATH = REPO_ROOT / "data" / "processed" / "feedback.csv"


PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
conn = duckdb.connect()
  
def get_review(parent_asin):
    query = f"""
        SELECT text
        FROM read_parquet('{PARQUET_PATH}')
        WHERE asin = ?
        LIMIT 1
    """
    result = conn.execute(query, [parent_asin]).fetchone()

    if result:
        return result[0]
    return "No review found"

def store_feedback(feedback_data):
    if not feedback_data:
        return

    df = pd.DataFrame(feedback_data)

    # Add timestamp (VERY useful for analysis later)
    df["timestamp"] = datetime.now().isoformat()

    # Append to CSV (create if not exists)
    file_exists = FEEDBACK_PATH.exists()

    df.to_csv(
        FEEDBACK_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )


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
search_mode = st.radio("Select Search Mode", ["BM25", "Semantic"]) #, "Hybrid"])
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
            # truncated_review = f"Simulated review for {result['title'][:200]}..."  # Replace with real review
            
            truncated_review = get_review(result["parent_asin"])[:200]
            st.write(f"**Review:** {truncated_review}...")  # Truncate review for display
            
            st.write(f"**Retrieval Score:** {result['score']:.2f}")

            feedback = st.radio(f"Was this result helpful?", 
                                options=["Not selected", "👍", "👎"], 
                                index=0,
                                key=f"feedback_{idx}")
            if feedback in ["👍", "👎"]:
                feedback_data.append({
                    "product_title": result["title"],
                    "feedback": feedback,
                    "score": result["score"]
                })

    # Save feedback to CSV
    if feedback_data:
        store_feedback(feedback_data)
        st.success("Feedback saved!")




