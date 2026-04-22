from src.loader import download_index_files
download_index_files()

import streamlit as st
import pickle
import duckdb
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import sys

# ==============================
# PATH SETUP
# ==============================
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
FEEDBACK_PATH = REPO_ROOT / "data" / "processed" / "feedback.csv"

from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic
from src.rag_pipeline import rag_pipeline
from src.hybrid import hybrid_rag_pipeline

conn = duckdb.connect()

# ==============================
# CACHED LOADERS
# ==============================
@st.cache_resource
def load_models():
    # Correct unpacking: bm25, metadata, tokenized_corpus
    bm25, metadata, tokenized_corpus = load_bm25_index()
    faiss_index = load_semantic_index()
    sem_metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return bm25, metadata, faiss_index, sem_metadata, model

@st.cache_data
def get_reviews(asins):
    if not asins:
        return {}
    placeholders = ','.join(['?' for _ in asins])
    query = f"""
        SELECT asin, text
        FROM read_parquet('{PARQUET_PATH}')
        WHERE asin IN ({placeholders})
    """
    return {r[0]: r[1] for r in conn.execute(query, asins).fetchall()}

# ==============================
# FEEDBACK STORAGE
# ==============================
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
# LOAD MODELS INTO SESSION STATE
# ==============================
if 'loaded' not in st.session_state:
    bm25, metadata, faiss_index, sem_metadata, model = load_models()
    st.session_state.bm25 = bm25
    st.session_state.metadata = metadata          # for BM25 search
    st.session_state.faiss_index = faiss_index
    st.session_state.sem_metadata = sem_metadata  # for semantic search
    st.session_state.model = model
    st.session_state.loaded = True

# ==============================
# STREAMLIT UI
# ==============================
st.title("Amazon Beauty Products Assistant")

tab1, tab2 = st.tabs(["🔍 Search", "🧠 RAG"])

# ==============================
# SEARCH TAB
# ==============================
with tab1:
    search_mode = st.radio("Select Search Mode", ["BM25", "Semantic"])
    query = st.text_input("Enter your query:", key="search_query")

    if query:
        if search_mode == "BM25":
            results = search_bm25(
                query,
                st.session_state.bm25,
                st.session_state.metadata,   # correct: metadata not tokenized_corpus
                top_k=3
            )
        else:
            results = search_semantic(
                query,
                st.session_state.faiss_index,
                st.session_state.sem_metadata,
                st.session_state.model,
                top_k=3
            )

        st.subheader(f"Top 3 results for '{query}' using {search_mode}")

        asins = [r['parent_asin'] for r in results]
        reviews = get_reviews(asins)

        feedback_data = []

        for idx, result in enumerate(results):
            with st.expander(f"Result {idx + 1}: {result['title']}"):
                st.write(f"**Product Title:** {result['title']}")
                st.write(f"**Average Rating:** {result.get('average_rating', 'N/A')} / 5")
                st.write(f"**Number of Reviews:** {result.get('rating_number', 'N/A')}")
                st.write(f"**Store:** {result.get('store', 'N/A')}")
                review = reviews.get(result['parent_asin'], "No review found")[:200]
                st.write(f"**Review:** {review}...")
                st.write(f"**Retrieval Score:** {result['score']:.4f}")

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
# RAG TAB
# ==============================
with tab2:
    rag_mode = st.radio(
        "Select RAG Mode",
        ["BM25", "Semantic", "Hybrid"],
        key="rag_mode"
    )
    use_tools = st.checkbox("🔍 Augment with web search (for pricing, availability, certifications, etc.)")
    rag_query = st.text_input("Ask a question:", key="rag_query")

    if rag_query:
        with st.spinner("Generating answer..."):
            if rag_mode == "Hybrid":
                answer, prompt, docs, tool_used = hybrid_rag_pipeline(
                    rag_query,
                    mode="Hybrid",
                    prompt_version="v1",
                    top_k=5,
                    use_tools=use_tools
                )
            else:
                # BM25 or Semantic — use rag_pipeline
                answer, prompt, docs, tool_used = rag_pipeline(
                    rag_query,
                    mode=rag_mode,
                    prompt_version="v1",
                    top_k=5,
                    use_tools=use_tools
                )

        # Display answer directly — no wrapper
        st.markdown("## 🧠 Generated Answer")
        if tool_used:
            st.info("ℹ️ Web search was used to supplement Amazon reviews for this query.")
        st.success(answer)

        # Display sources cleanly
        st.markdown("### 📚 Sources")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"[{i}] {doc['title']}"):
                st.write(f"**ASIN:** {doc.get('parent_asin', 'N/A')}")
                st.write(f"**Average Rating:** {doc.get('average_rating', 'N/A')} / 5")
                st.write(f"**Number of Reviews:** {doc.get('rating_number', 'N/A')}")
                st.write(f"**Store:** {doc.get('store', 'N/A')}")
