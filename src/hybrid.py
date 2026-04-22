import re
import sys
import os
import pickle
import duckdb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic
from src.tools import web_search, should_use_web_search

# -----------------------------
# PATH SETUP
# -----------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"

conn = duckdb.connect()

# -----------------------------
# STEP 1: LLM PIPELINE (Groq)
# -----------------------------
load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

def generate_llm_answer(prompt):
    response = llm.invoke(prompt)
    answer = response.content.strip()
    # Robustly strip any <think>...</think> blocks
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer


# -----------------------------
# LOAD DATA
# -----------------------------
faiss_index = load_semantic_index()
metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fix: unpack correctly — metadata is position 1, not tokenized_corpus
#bm25, _, tokenized_corpus = load_bm25_index()
bm25, _, _ = load_bm25_index()


# -----------------------------
# RETRIEVERS
# -----------------------------
def semantic_retrieve(query, top_k=5):
    return search_semantic(query, faiss_index, metadata, model, top_k)


def bm25_retrieve(query, top_k=5):
    # Fix: pass metadata not tokenized_corpus
    return search_bm25(query, bm25, metadata, top_k)


# -----------------------------
# HYBRID RETRIEVER
# -----------------------------
def hybrid_retrieve(query, top_k=5):
    # Get more candidates from each retriever
    sem_results = semantic_retrieve(query, top_k=10)
    bm25_results = bm25_retrieve(query, top_k=10)

    # Reciprocal Rank Fusion
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

    # Sort by RRF score descending
    sorted_asins = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [doc_map[asin] for asin in sorted_asins[:top_k]]


# -----------------------------
# CONTEXT BUILDER
# -----------------------------
def get_review(parent_asin):
    query = f"""
        SELECT text
        FROM read_parquet('{PARQUET_PATH}')
        WHERE asin = ?
        LIMIT 1
    """
    result = conn.execute(query, [parent_asin]).fetchone()
    return result[0] if result else "No review found"


def build_context(docs):
    context_blocks = []
    for doc in docs:
        block = f"""
        Product ASIN: {doc.get('parent_asin', 'N/A')}
        Title: {doc.get('title', 'N/A')}
        Price: {doc.get('price', 'N/A')}
        Average Rating: {doc.get('average_rating', 'N/A')} out of 5
        Number of Reviews: {doc.get('rating_number', 'N/A')}
        Store: {doc.get('store', 'N/A')}
        Review: {doc.get('review_text', get_review(doc["parent_asin"])[:200])}
        """.strip()
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


# -----------------------------
# PROMPT TEMPLATES
# -----------------------------
SYSTEM_PROMPT_V1 = """
You are a helpful Amazon shopping assistant.

STRICT RULES:
- Answer ONLY using the provided reviews
- Do NOT make up information
- Do NOT repeat the question or instructions
- Do NOT output words like "assistant" or tags like "<think>"
- Do NOT generate numbered lists (1, 2, 3, etc.)
- Keep the answer short and clear (2-4 sentences max)
- Always cite product ASINs when relevant

Output ONLY the final answer. No extra text.
"""

SYSTEM_PROMPT_V2 = """
You are an expert product recommendation assistant.

TASK:
Use the provided reviews to answer the question.

RULES:
- Base your answer ONLY on the reviews
- Do NOT invent information
- Do NOT repeat the prompt or instructions
- Do NOT output "assistant" or "<think>"
- Avoid numbered or bullet lists unless necessary
- Keep the answer concise and readable
- Mention product ASINs when relevant

Output ONLY the final answer. No explanations.
"""


def build_prompt(query, context, version="v1", web_augmented=False):
    system_prompt = SYSTEM_PROMPT_V1 if version == "v1" else SYSTEM_PROMPT_V2

    source_instruction = (
        "Use BOTH the Amazon product information AND the web search results provided."
        if web_augmented
        else "Answer using the provided Amazon product information (metadata, descriptions, and customer reviews)."
    )

    return f"""
    {system_prompt}

    {source_instruction}

    Product Information and Context:
    {context}

    Question is: {query}

    FINAL ANSWER:
    """


# -----------------------------
# HYBRID RAG PIPELINE
# -----------------------------
def hybrid_rag_pipeline(query, mode="Hybrid", prompt_version="v1", top_k=5, use_tools=False):
    # 1. Retrieval
    docs = hybrid_retrieve(query, top_k)

    # 2. Tool augmentation 
    web_context = ""
    tool_used = False
    if use_tools and should_use_web_search(query, docs):
        web_results = web_search.invoke({"query": query})
        web_context = f"\n\nWeb Search Results:\n{web_results}"
        tool_used = True

    # 3. Context
    context = build_context(docs) + web_context

    # 4. Prompt
    prompt = build_prompt(query, context, version=prompt_version, web_augmented=tool_used)

    # 5. LLM
    answer = generate_llm_answer(prompt)

    return answer, prompt, docs, tool_used


# -----------------------------
# EVALUATION LOOP
# -----------------------------
def evaluate_queries(queries):
    results = []
    for q in queries:
        answer, prompt, docs, tool_used = hybrid_rag_pipeline(q)
        print("\n======================")
        print("QUERY:", q)
        print("ANSWER:", answer[:500])
        results.append({"query": q, "answer": answer})
    return results


# -----------------------------
# MAIN TEST
# -----------------------------
if __name__ == "__main__":
    test_queries = [
        "What are the best moisturizers for dry skin?",
        "Something to keep my face hydrated all day",
        "Product for dark spots and uneven skin tone"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"QUERY: {query}")
        answer, prompt, docs, tool_used = hybrid_rag_pipeline(query)
        print(f"ANSWER: {answer}")
        