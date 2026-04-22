# ─────────────────────────────────────────────────────────────
# hybrid.py
# Purpose: Hybrid retriever (BM25 + Semantic via RRF) and
#          hybrid RAG pipeline. Extends the base RAG pipeline
#          in rag_pipeline.py with combined retrieval.
# ─────────────────────────────────────────────────────────────
from src.rag_pipeline import (
    semantic_retrieve,
    bm25_retrieve,
    build_context,
    build_prompt,
    generate_llm_answer,
    web_search,
    should_use_web_search
)

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
        