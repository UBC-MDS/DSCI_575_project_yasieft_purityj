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

# ==============================
# PATH SETUP
# ==============================
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic

FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.npy"

BM25_INDEX_PATH = REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
FEEDBACK_PATH = REPO_ROOT / "data" / "processed" / "feedback.csv"

PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
conn = duckdb.connect()

# ==============================
# TOKENIZER
# ==============================
def tokenize(text):
    STOPWORDS = {
        "a","an","the","and","or","but","in","on","at",
        "to","for","of","with","by","from","is","it",
        "this","that","are","was","be","has","have"
    }

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# ==============================
# DATA HELPERS
# ==============================
def get_review(parent_asin):
    query = f"""
        SELECT text
        FROM read_parquet('{PARQUET_PATH}')
        WHERE asin = ?
        LIMIT 1
    """
    result = conn.execute(query, [parent_asin]).fetchone()
    return result[0] if result else "No review found"

def store_feedback(feedback_data):
    if not feedback_data:
        return

    df = pd.DataFrame(feedback_data)
    df["timestamp"] = datetime.now().isoformat()

    file_exists = FEEDBACK_PATH.exists()
    df.to_csv(
        FEEDBACK_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )

# ==============================
# LOAD MODELS (SESSION STATE)
# ==============================
if 'bm25_loaded' not in st.session_state:
    bm25, tokenized_corpus, _ = load_bm25_index()
    st.session_state.bm25 = bm25
    st.session_state.tokenized_corpus = tokenized_corpus
    st.session_state.bm25_loaded = True

if 'semantic_loaded' not in st.session_state:
    faiss_index = load_semantic_index()
    st.session_state.faiss_index = faiss_index
    st.session_state.metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.semantic_loaded = True

# ==============================
# SIMPLE RAG GENERATOR (placeholder)
# ==============================
def generate_rag_answer(query, docs):
    """
    Simple placeholder RAG generation.
    Replace this later with OpenAI / LLM call.
    """
    context = " ".join(docs)

    answer = f"""
    Based on the retrieved documents, here is a helpful answer:

    {context[:500]}
    """

    # Truncate long answers
    MAX_LEN = 400
    if len(answer) > MAX_LEN:
        answer = answer[:MAX_LEN] + "..."

    return answer

# ==============================
# UI TABS
# ==============================
tab1, tab2 = st.tabs(["🔍 Search", "🤖 RAG"])

# ==============================
# SEARCH TAB (UNCHANGED)
# ==============================
with tab1:
    search_mode = st.radio("Select Search Mode", ["BM25", "Semantic"])
    query = st.text_input("Enter your query:", key="search_query")

    if query:
        if search_mode == "BM25":
            results = search_bm25(query, st.session_state.bm25, st.session_state.tokenized_corpus, top_k=3)
        else:
            results = search_semantic(query, st.session_state.faiss_index, st.session_state.metadata, st.session_state.model, top_k=3)

        st.subheader(f"Top 3 results for '{query}' using {search_mode}")

        feedback_data = []

        for idx, result in enumerate(results):
            with st.expander(f"Result {idx + 1}: {result['title']}"):
                st.write(f"**Product Title:** {result['title']}")
                review = get_review(result["parent_asin"])[:200]
                st.write(f"**Review:** {review}...")
                st.write(f"**Score:** {result['score']:.2f}")

                feedback = st.radio(
                    "Helpful?",
                    ["Not selected", "👍", "👎"],
                    key=f"feedback_{idx}"
                )

                if feedback in ["👍", "👎"]:
                    feedback_data.append({
                        "product_title": result["title"],
                        "feedback": feedback,
                        "score": result["score"]
                    })

        if feedback_data:
            store_feedback(feedback_data)
            st.success("Feedback saved!")

# ==============================
# RAG TAB (NEW)
# ==============================
with tab2:
    rag_mode = st.radio("Retrieval Method", ["BM25", "Semantic"], key="rag_mode")
    rag_query = st.text_input("Ask a question:", key="rag_query")

    if rag_query:
        # Retrieve documents
        if rag_mode == "BM25":
            results = search_bm25(rag_query, st.session_state.bm25, st.session_state.tokenized_corpus, top_k=3)
        else:
            results = search_semantic(rag_query, st.session_state.faiss_index, st.session_state.metadata, st.session_state.model, top_k=3)

        # Extract document texts
        docs = [get_review(r["parent_asin"])[:300] for r in results]

        # Generate answer
        answer = generate_rag_answer(rag_query, docs)

        # =========================
        # ANSWER PANEL (PROMINENT)
        # =========================
        st.markdown("## 🧠 Generated Answer")
        st.success(answer)

        # =========================
        # SOURCE ATTRIBUTION
        # =========================
        st.markdown("### 📚 Sources")

        for i, (doc, res) in enumerate(zip(docs, results), 1):
            with st.expander(f"[{i}] {res['title']}"):
                st.write(doc)