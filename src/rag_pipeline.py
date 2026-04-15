import sys
from pathlib import Path
import pickle
import duckdb
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Your existing imports
from bm25 import load_bm25_index, search_bm25
from semantic import load_semantic_index, search_semantic

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
# STEP 1: LLM PIPELINE
# -----------------------------
llm = pipeline("text-generation", model="Qwen/Qwen3.5-0.8B")

def generate(prompt):
    output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    return output


# -----------------------------
# LOAD DATA
# -----------------------------
faiss_index = load_semantic_index()
metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the sentence transformer model

bm25, tokenized_corpus, _ = load_bm25_index()


# -----------------------------
# STEP 2.1: RETRIEVERS
# -----------------------------
def semantic_retrieve(query, top_k=5):
    return search_semantic(query, faiss_index, metadata, model, top_k)


def bm25_retrieve(query, top_k=5):
    return search_bm25(query, bm25, tokenized_corpus, top_k)


# -----------------------------
# STEP 3: HYBRID RETRIEVER
# -----------------------------
def hybrid_retrieve(query, top_k=5):
    sem_results = semantic_retrieve(query, top_k)
    bm25_results = bm25_retrieve(query, top_k)

    # Merge + deduplicate (by unique id or text)
    seen = set()
    combined = []

    for doc in sem_results + bm25_results:
        text = doc.get("title", "")
        if text not in seen:
            seen.add(text)
            combined.append(doc)

    return combined[:top_k]


# -----------------------------
# STEP 2.2: CONTEXT BUILDER
# -----------------------------
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


def build_context(docs):
    context_blocks = []

    for doc in docs:
        block = f"""
Product ASIN: {doc.get('parent_asin', 'N/A')}
Title: {doc.get('title', 'N/A')}
Rating: {doc.get('rating_number', 'N/A')}/5
Review: {doc.get('review_text', get_review(doc["parent_asin"])[:200])}
"""
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)


# -----------------------------
# STEP 2.3: PROMPT TEMPLATES
# -----------------------------

# Variant 1 (strict grounding)
SYSTEM_PROMPT_V1 = """
You are a helpful Amazon shopping assistant.
Answer ONLY using the provided reviews.
Do NOT make up information.
Always cite the product ASIN.
Keep the answer concise.
"""

# Variant 2 (more flexible)
SYSTEM_PROMPT_V2 = """
You are an expert product recommendation assistant.
Use the reviews below to answer the question.
Summarize key insights and mention product ASINs when relevant.
"""


def build_prompt(query, context, version="v1"):
    system_prompt = SYSTEM_PROMPT_V1 if version == "v1" else SYSTEM_PROMPT_V2

    return f"""
{system_prompt}

Customer Reviews:
{context}

Question: {query}

Answer:
"""


# -----------------------------
# STEP 2.4: RAG PIPELINE
# -----------------------------
def rag_pipeline(query, mode="semantic", prompt_version="v1", top_k=5):

    # 1. Retrieval
    if mode == "semantic":
        docs = semantic_retrieve(query, top_k)
    elif mode == "bm25":
        docs = bm25_retrieve(query, top_k)
    elif mode == "hybrid":
        docs = hybrid_retrieve(query, top_k)
    else:
        raise ValueError("Invalid mode")

    # 2. Context
    context = build_context(docs)

    # 3. Prompt
    prompt = build_prompt(query, context, version=prompt_version)

    # 4. LLM
    answer = generate(prompt)

    return answer, docs


# -----------------------------
# STEP 5: SIMPLE EVALUATION LOOP
# -----------------------------
def evaluate_queries(queries, mode="hybrid"):
    results = []

    for q in queries:
        answer, docs = rag_pipeline(q, mode=mode)

        print("\n======================")
        print("QUERY:", q)
        print("ANSWER:", answer[:500])

        results.append({
            "query": q,
            "answer": answer
        })

    return results


# -----------------------------
# MAIN TEST
# -----------------------------
if __name__ == "__main__":

    query = "What are the best moisturizers for dry skin?"

    # Change mode: "semantic", "bm25", "hybrid"
    answer, docs = rag_pipeline(query, mode="hybrid", prompt_version="v1")

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

    print("\n=== RETRIEVED DOCS ===\n")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.get('title', 'N/A')} | Rating: {doc.get('rating_number', 'N/A')}")


