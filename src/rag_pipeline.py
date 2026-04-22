import sys, os
from pathlib import Path
import pickle
import duckdb
from transformers import pipeline
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
# STEP 1: LLM PIPELINE
# -----------------------------
# USE A LOCAL MODEL
#llm = pipeline("text-generation", model="Qwen/Qwen3.5-0.8B")
# def generate_llm_answer(prompt):
#     output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
#     answer = output[len(prompt):].strip()
    
#     # Remove garbage patterns
#     if "<think>" in answer:
#         answer = answer.split("</think>")[-1].strip()

#     if "assistant" in answer:
#         answer = answer.split("assistant")[-1].strip()

#     # Remove repeated numbering junk
#     if "1." in answer and "2." in answer:
#         lines = answer.split("\n")
#         cleaned = [l for l in lines if len(l.strip()) > 10]
#         answer = "\n".join(cleaned)

#     return answer.strip()

# Setup Groq 
load_dotenv()
llm = ChatGroq(
 model="llama-3.3-70b-versatile",
 api_key=os.getenv("GROQ_API_KEY"),
 temperature=0.1    # low temperature = more focussed, less random answers   
)

def generate_llm_answer(prompt):
    response = llm.invoke(prompt)
    return response.content.strip()

# -----------------------------
# LOAD DATA
# -----------------------------
faiss_index = load_semantic_index()
metadata = pickle.load(open(CORPUS_METADATA_PATH, "rb"))
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the sentence transformer model

#bm25, tokenized_corpus, _ = load_bm25_index()
bm25, _, _ = load_bm25_index()


# -----------------------------
# STEP 2.1: RETRIEVERS
# -----------------------------
def semantic_retrieve(query, top_k=5):
    return search_semantic(query, faiss_index, metadata, model, top_k)


def bm25_retrieve(query, top_k=5):
    return search_bm25(query, bm25, metadata, top_k)


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
        Price: {doc.get('price', 'N/A')}
        Average Rating: {doc.get('average_rating', 'N/A')} out of 5
        Number of Reviews: {doc.get('rating_number', 'N/A')}
        Store: {doc.get('store', 'N/A')}
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
# STEP 5: SIMPLE EVALUATION LOOP
# -----------------------------
def evaluate_queries(queries, mode="hybrid"):
    results = []

    for q in queries:
        answer, prompt, docs, tool_used = rag_pipeline(q, mode=mode)

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
