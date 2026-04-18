import sys
from pathlib import Path
import pickle
import duckdb
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATH SETUP
# -----------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "All_Beauty.parquet"

FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "faiss_index.faiss"
CORPUS_METADATA_PATH = REPO_ROOT / "data" / "processed" / "corpus_metadata.pkl"

# Your existing imports
from src.bm25 import load_bm25_index, search_bm25
from src.semantic import load_semantic_index, search_semantic
conn = duckdb.connect()
# -----------------------------
# STEP 1: LLM PIPELINE
# -----------------------------
llm = pipeline("text-generation", model="Qwen/Qwen3.5-0.8B")

def generate_llm_answer(prompt):
    output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    answer = output[len(prompt):].strip()
    
    # Remove garbage patterns
    if "<think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    if "assistant" in answer:
        answer = answer.split("assistant")[-1].strip()

    # Remove repeated numbering junk
    if "1." in answer and "2." in answer:
        lines = answer.split("\n")
        cleaned = [l for l in lines if len(l.strip()) > 10]
        answer = "\n".join(cleaned)

    return answer.strip()
 

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
        Rating: {doc.get('score', 'N/A')}
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

STRICT RULES:
- Answer ONLY using the provided reviews
- Do NOT make up information
- Do NOT repeat the question or instructions
- Do NOT output words like "assistant" or tags like "<think>"
- Do NOT generate numbered lists (1, 2, 3, etc.)
- Keep the answer short and clear (2–4 sentences max)
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


def build_prompt(query, context, version="v1"):
    system_prompt = SYSTEM_PROMPT_V1 if version == "v1" else SYSTEM_PROMPT_V2

    return f"""
    {system_prompt}

    Customer Reviews Are:
    {context}

    Question is: {query}

    FINAL ANSWER:
    """


# -----------------------------
# STEP 2.4: RAG PIPELINE
# -----------------------------
def hybrid_rag_pipeline(query, mode="Hybrid", prompt_version="v1", top_k=5):

    docs = []

    if mode == "Hybrid":
        docs = hybrid_retrieve(query, top_k)
    else:
        raise ValueError("Invalid mode")

    # 2. Context
    context = build_context(docs)

    # 3. Prompt
    prompt = build_prompt(query, context, version=prompt_version)

    # 4. LLM
    answer = generate_llm_answer(prompt)

    return answer, prompt, docs


# -----------------------------
# STEP 5: SIMPLE EVALUATION LOOP
# -----------------------------
def evaluate_queries(queries, mode="hybrid"):
    results = []

    for q in queries:
        answer, prompt, docs = hybrid_rag_pipeline(q, mode=mode)

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
    answer, prompt, docs = hybrid_rag_pipeline(query, mode="Hybrid", prompt_version="v1")

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

    # print("\n=== RETRIEVED DOCS ===\n")
    # for i, doc in enumerate(docs):
    #     print(f"{i+1}. {doc.get('title', 'N/A')} | Rating: {doc.get('score', 'N/A')}")


