import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

# ==============================
# PAGE CONFIG — must be first st call
# ==============================
st.set_page_config(page_title="BeautyFinder AI", layout="wide")

# ==============================
# SECRETS — must be set before any src imports
# ==============================
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY") or ""
os.environ["HF_TOKEN"] = get_secret("HF_TOKEN") or ""
os.environ["TAVILY_API_KEY"] = get_secret("TAVILY_API_KEY") or ""

# ==============================
# DOWNLOAD INDEX FILES
# ==============================
from src.loader import download_index_files

with st.spinner("⏳ Checking index files..."):
    download_index_files()

# ==============================
# IMPORTS
# ==============================
import re
import pickle
import pandas as pd
from datetime import datetime
# from dotenv import load_dotenv

# load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"
FEEDBACK_PATH = REPO_ROOT / "data" / "processed" / "feedback.csv"

# ==============================
# CACHED RESOURCE LOADERS
# ==============================

@st.cache_resource
def load_faiss_and_model():
    from sentence_transformers import SentenceTransformer
    from src.semantic import load_semantic_index
    faiss_index = load_semantic_index()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return faiss_index, model

@st.cache_resource
def load_metadata():
    with open(CORPUS_METADATA_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_bm25():
    from src.bm25 import load_bm25_index
    bm25, _, _ = load_bm25_index()
    return bm25

@st.cache_resource
def load_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )

@st.cache_resource
def load_review_lookup():
    """
    Load asin→review snippet lookup at startup.
    Replaces per-query parquet scans with a simple dict lookup.
    Only stores first 200 chars per review to keep memory low.
    """
    import duckdb
    conn = duckdb.connect()
    result = conn.execute(f"""
        SELECT asin, LEFT(text, 200) as snippet
        FROM read_parquet('{PARQUET_PATH}')
        WHERE text IS NOT NULL
    """).fetchall()
    conn.close()
    return {row[0]: row[1] for row in result}

# ==============================
# LOAD ALL RESOURCES AT STARTUP
# ==============================
faiss_index, embedding_model = load_faiss_and_model()
metadata = load_metadata()
bm25 = load_bm25()
llm = load_llm()
review_lookup = load_review_lookup()

# ==============================
# SEARCH FUNCTIONS
# ==============================
from src.bm25 import search_bm25
from src.semantic import search_semantic
from src.tools import web_search, should_use_web_search

def do_semantic_search(query, top_k=5):
    return search_semantic(query, faiss_index, metadata, embedding_model, top_k)

def do_bm25_search(query, top_k=5):
    return search_bm25(query, bm25, metadata, top_k)

def do_hybrid_search(query, top_k=5):
    sem_results = do_semantic_search(query, top_k=10)
    bm25_results = do_bm25_search(query, top_k=10)
    rrf_scores = {}
    doc_map = {}
    for rank, doc in enumerate(sem_results):
        asin = doc.get("parent_asin", "")
        rrf_scores[asin] = rrf_scores.get(asin, 0) + 1 / (rank + 1 + 60)
        doc_map[asin] = doc
    for rank, doc in enumerate(bm25_results):
        asin = doc.get("parent_asin", "")
        rrf_scores[asin] = rrf_scores.get(asin, 0) + 1 / (rank + 1 + 60)
        doc_map[asin] = doc
    sorted_asins = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[asin] for asin in sorted_asins[:top_k]]

# ==============================
# RAG PIPELINE
# ==============================

def get_review(parent_asin):
    """Dict lookup — replaces expensive per-query parquet scans."""
    return review_lookup.get(parent_asin, "No review found")

def build_context(docs):
    """Use top 3 docs only to keep prompt size and memory usage down."""
    blocks = []
    for doc in docs[:3]:
        blocks.append(f"""
        Product ASIN: {doc.get('parent_asin', 'N/A')}
        Title: {doc.get('title', 'N/A')}
        Price: {doc.get('price', 'N/A')}
        Average Rating: {doc.get('average_rating', 'N/A')} out of 5
        Number of Reviews: {doc.get('rating_number', 'N/A')}
        Store: {doc.get('store', 'N/A')}
        Review: {get_review(doc['parent_asin'])[:150]}
        """.strip())
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """
You are a helpful Amazon shopping assistant.
Answer using the provided Amazon product information (metadata, descriptions, and customer reviews).
Do NOT make up information. Keep the answer short and clear (2-4 sentences max).
Always cite product ASINs when relevant. Output ONLY the final answer.
"""

def build_prompt(query, context, web_augmented=False):
    source_instruction = (
        "Use BOTH the Amazon product information AND the web search results provided."
        if web_augmented
        else "Answer using the provided Amazon product information."
    )
    return f"{SYSTEM_PROMPT}\n{source_instruction}\n\nProduct Information:\n{context}\n\nQuestion: {query}\n\nFINAL ANSWER:"

def generate_answer(prompt):
    response = llm.invoke(prompt)
    answer = response.content.strip()
    return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

def run_rag(query, mode="Semantic", top_k=5, use_tools=False):
    if mode == "Semantic":
        docs = do_semantic_search(query, top_k)
    elif mode == "BM25":
        docs = do_bm25_search(query, top_k)
    else:
        docs = do_hybrid_search(query, top_k)

    web_context = ""
    tool_used = False
    web_sources = []

    if use_tools and should_use_web_search(query, docs):
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            results = client.search(f"{query} Amazon beauty product", max_results=3)
            web_sources = [
                {"title": r.get("title", "Web result"), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in results.get("results", [])
            ]
            web_context = "\n\n".join(
                f"{s['content']}\nSource: {s['url']}" for s in web_sources
            )
            web_context = f"\n\nWeb Search Results:\n{web_context}"
            tool_used = True

    context = build_context(docs) + web_context
    prompt = build_prompt(query, context, web_augmented=tool_used)
    answer = generate_answer(prompt)
    return answer, docs, tool_used, web_sources

# ==============================
# FEEDBACK
# ==============================
def store_feedback(feedback_data):
    if not feedback_data:
        return
    df = pd.DataFrame(feedback_data)
    df["timestamp"] = datetime.now().isoformat()
    file_exists = FEEDBACK_PATH.exists()
    df.to_csv(FEEDBACK_PATH, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")

@st.cache_data
def get_reviews(asins):
    """Open and close DuckDB connection locally to avoid holding memory."""
    if not asins:
        return {}
    import duckdb
    conn = duckdb.connect()
    placeholders = ','.join(['?' for _ in asins])
    query = f"SELECT asin, text FROM read_parquet('{PARQUET_PATH}') WHERE asin IN ({placeholders})"
    result = {r[0]: r[1] for r in conn.execute(query, list(asins)).fetchall()}
    conn.close()
    return result

# ==============================
# UI
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
        results = do_bm25_search(query, top_k=3) if search_mode == "BM25" else do_semantic_search(query, top_k=3)
        st.subheader(f"Top 3 results for '{query}' using {search_mode}")

        asins = [r['parent_asin'] for r in results]
        reviews = get_reviews(tuple(asins))
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
                feedback = st.radio("Helpful?", ["Not selected", "👍", "👎"], key=f"feedback_{idx}")
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
    rag_mode = st.radio("Select RAG Mode", ["BM25", "Semantic", "Hybrid"], key="rag_mode")
    use_tools = st.checkbox("🔍 Augment with web search (for pricing, availability, certifications, etc.)")
    rag_query = st.text_input("Ask a question:", key="rag_query")

    if rag_query:
        with st.spinner("Generating answer..."):
            #answer, docs, tool_used = run_rag(rag_query, mode=rag_mode, top_k=5, use_tools=use_tools)
            answer, docs, tool_used, web_sources = run_rag(
            rag_query, mode=rag_mode, top_k=5, use_tools=use_tools
        )

        st.markdown("## 🧠 Generated Answer")
        if tool_used:
            st.info("ℹ️ Web search was used to supplement Amazon reviews for this query.")
        st.success(answer)

        # Web sources
        if web_sources:
            st.markdown("### 🌐 Web Sources")
            for s in web_sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

        # Product sources
        st.markdown("### 📚 Product Sources")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"[{i}] {doc['title']}"):
                st.write(f"**ASIN:** {doc.get('parent_asin', 'N/A')}")
                st.write(f"**Average Rating:** {doc.get('average_rating', 'N/A')} / 5")
                st.write(f"**Number of Reviews:** {doc.get('rating_number', 'N/A')}")
                st.write(f"**Store:** {doc.get('store', 'N/A')}")
