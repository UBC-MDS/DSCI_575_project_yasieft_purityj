# ─────────────────────────────────────────────────────────────
# rag_pipeline.py
# Purpose: Base RAG pipeline for single retriever modes
#          (BM25 or Semantic). Also owns all shared components
#          (LLM, retrievers, context builder, prompt templates)
#          that hybrid.py imports from here.
# ─────────────────────────────────────────────────────────────
import re
import sys
import os
import pickle
import duckdb
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------------
# PATH SETUP
# -----------------------------
REPO_ROOT = Path(__file__).parent.parent
#sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"
FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"

from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic
from src.tools import web_search, should_use_web_search

conn = duckdb.connect()

# -----------------------------
# LLM SETUP
# -----------------------------
load_dotenv()

def get_llm(provider="groq", local_model="Qwen/Qwen3.5-0.8B"):
    """
    Factory function to get the LLM backend.
    
    provider: "groq" (default, cloud-based, fast, no GPU needed)
              "local" (HuggingFace model, runs on your machine)
    
    local_model: only used when provider="local"
                 any HuggingFace model ID e.g.:
                 - "Qwen/Qwen3.5-0.8B"      (tiny, laptop-friendly)
                 - "Qwen/Qwen3.5-2B"         (better quality)
                 - "meta-llama/Llama-3.2-3B-Instruct"
    """
    if provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
    elif provider == "local":
        return pipeline(
            "text-generation",
            model=local_model,
            max_new_tokens=200
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'groq' or 'local'")


# Default: Groq — change to "local" here if you want to use a local model
LLM_PROVIDER = "groq"
#llm = get_llm(provider=LLM_PROVIDER)
llm = None   #not initialized at import time

def get_llm_instance():
    """Return the LLM instance, initializing it on first call."""
    global _llm
    if _llm is None:
        _llm = get_llm(provider=LLM_PROVIDER)
    return _llm

def generate_llm_answer(prompt):
    """Generate an answer using the configured LLM backend."""
    if LLM_PROVIDER == "groq":
        response = llm.invoke(prompt)
        answer = response.content.strip()
    elif LLM_PROVIDER == "local":
        output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
        answer = output[len(prompt):].strip()
        # Clean up local model artifacts
        if "assistant" in answer:
            answer = answer.split("assistant")[-1].strip()

    # Strip <think> blocks (some models output these)
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer

# -----------------------------
# LOAD INDEXES
# -----------------------------
faiss_index = load_semantic_index()
metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the sentence transformer model

#bm25, tokenized_corpus, _ = load_bm25_index()
bm25, _, _ = load_bm25_index()


# -----------------------------
# STEP 2.1: RETRIEVERS
# -----------------------------
def semantic_retrieve(query, top_k=5):
    return search_semantic(query, faiss_index, metadata, embedding_model, top_k)


def bm25_retrieve(query, top_k=5):
    return search_bm25(query, bm25, metadata, top_k)


# -----------------------------
# STEP 2.2: CONTEXT BUILDER
# -----------------------------
def get_review(parent_asin):
    """Fetch one review text for a product from the reviews parquet file."""
    sql = """
        SELECT text
        FROM read_parquet(?)
        WHERE asin = ?
        LIMIT 1
    """
    result = conn.execute(sql, [str(PARQUET_PATH), parent_asin]).fetchone()
    return result[0] if result else "No review found"


def build_context(docs):
    """Convert retrieved product docs into a structured text block for the LLM."""
    context_blocks = []
    for doc in docs:
        block = f"""
        Product ASIN: {doc.get('parent_asin', 'N/A')}
        Title: {doc.get('title', 'N/A')}
        Price: {doc.get('price', 'N/A')}
        Average Rating: {doc.get('average_rating', 'N/A')} out of 5
        Number of Reviews: {doc.get('rating_number', 'N/A')}
        Store: {doc.get('store', 'N/A')}
        Review: {get_review(doc["parent_asin"])[:200]}
        """.strip()
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


# -----------------------------
# STEP 2.3: PROMPT TEMPLATES
# -----------------------------

# Variant 1 (strict grounding)
SYSTEM_PROMPT_V1 = """
You are a helpful Amazon shopping assistant.

STRICT RULES:
- Answer using the provided Amazon product information (metadata, descriptions, and customer reviews)
- Do NOT make up information
- Do NOT repeat the question or instructions
- Do NOT output words like "assistant" or tags like "<think>"
- Keep the answer short and clear (2–4 sentences max)

Output ONLY the final answer. No extra text.
"""

SYSTEM_PROMPT_V2 = """
You are an expert product recommendation assistant.

TASK:
Use the provided reviews to answer the question.

RULES:
- Base your answer ONLY provided Amazon product information (metadata, descriptions, and customer reviews
- Do NOT invent information
- Do NOT repeat the prompt or instructions
- Do NOT output "assistant" or "<think>"
- Avoid numbered or bullet lists unless necessary
- Keep the answer concise and readable

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
# STEP 2.4: RAG PIPELINE
# -----------------------------
def rag_pipeline(query, mode="Semantic", prompt_version="v1", top_k=5, use_tools=False):

    docs = []
    # 1. Retrieval
    if mode == "Semantic":
        docs = semantic_retrieve(query, top_k)
    elif mode == "BM25":
        docs = bm25_retrieve(query, top_k)
    else:
        raise ValueError("Invalid mode")
    
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
# STEP 5: EVALUATION LOOP
# -----------------------------
def evaluate_queries(queries, mode="Semantic"):
    """Run a list of queries through the RAG pipeline and print results."""
    results = []
    for q in queries:
        answer, prompt, docs, tool_used = rag_pipeline(q, mode=mode)
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
        answer, prompt, docs, tool_used = rag_pipeline(query, mode="Semantic", prompt_version="v1")
        print(f"ANSWER: {answer}")
