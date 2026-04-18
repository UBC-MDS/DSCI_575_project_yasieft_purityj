import streamlit as st
import pickle
import duckdb
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import sys


# ==============================
# PATH SETUP
# ==============================
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
BM25_INDEX_PATH = REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
FEEDBACK_PATH = REPO_ROOT / "data" / "processed" / "feedback.csv"
# Custom imports
from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic
from src.rag_pipeline import rag_pipeline
from src.hybrid import hybrid_rag_pipeline
conn = duckdb.connect()

# ==============================
# UTILITIES
# ==============================
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at",
    "to","for","of","with","by","from","is","it",
    "this","that","are","was","be","has","have"
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return tokens

# ==============================
# CACHED LOADERS
# ==============================
@st.cache_resource
def load_models():
    bm25, tokenized_corpus, _ = load_bm25_index()
    faiss_index = load_semantic_index()
    metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return bm25, tokenized_corpus, faiss_index, metadata, model

@st.cache_data
def get_reviews(asins):
    if not asins:
        return {}
    query = f"""
        SELECT asin, text
        FROM read_parquet('{PARQUET_PATH}')
        WHERE asin IN ({','.join(['?']*len(asins))})
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
    bm25, tokenized_corpus, faiss_index, metadata, model = load_models()
    st.session_state.bm25 = bm25
    st.session_state.tokenized_corpus = tokenized_corpus
    st.session_state.faiss_index = faiss_index
    st.session_state.metadata = metadata
    st.session_state.model = model
    st.session_state.loaded = True

# ==============================
# RAG ANSWER FORMATTER
# ==============================
def format_rag_answer(query, results):
    answer = f"""
    Based on the retrieved documents and reviews, here is a helpful answer:

    {results}
    """
    MAX_LEN = 800
    if len(answer) > MAX_LEN:
        answer = answer[:MAX_LEN] + "..."
    return answer

# ==============================
# STREAMLIT UI
# ==============================
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
                query, st.session_state.bm25, st.session_state.tokenized_corpus, top_k=3
            )
        else:
            results = search_semantic(
                query,
                st.session_state.faiss_index,
                st.session_state.metadata,
                st.session_state.model,
                top_k=3
            )

        st.subheader(f"Top 3 results for '{query}' using {search_mode}")

        # Batch fetch reviews
        asins = [r['parent_asin'] for r in results]
        reviews = get_reviews(asins)

        feedback_data = []

        for idx, result in enumerate(results):
            with st.expander(f"Result {idx + 1}: {result['title']}"):
                st.write(f"**Product Title:** {result['title']}")
                review = reviews.get(result['parent_asin'], "No review found")[:200]
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
# RAG TAB
# ==============================
with tab2:
    rag_mode = st.radio("Select Search Mode", ["BM25", "Semantic", "Hybrid"], key="rag_mode")
    rag_query = st.text_input("Ask a question:", key="rag_query")

    if rag_query:
        with st.spinner("Generating answer..."):
            
            if rag_mode == "Hybrid":
                results, prompt, docs = hybrid_rag_pipeline(rag_query, mode="Hybrid", prompt_version="v1", top_k=5)

            else:
                results, prompt, docs = rag_pipeline(rag_query, mode="BM25")
            # results, prompt, docs = rag_pipeline(rag_query, mode=rag_mode)
            answer = format_rag_answer(rag_query, results)

        st.markdown("## 🧠 Generated Answer")
        st.success(answer)

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"[{i}] {doc['title']}"):
                st.write(doc)