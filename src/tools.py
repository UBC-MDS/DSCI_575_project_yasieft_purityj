# ─────────────────────────────────────────────────────────────
# tools.py
# Purpose: Tools the LLM can call when product reviews
#          don't contain enough information to answer the query.
#          Currently implements web search via Tavily API.
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for current product information that Amazon reviews 
    cannot answer: current pricing, retail availability, whether a product
    is cruelty-free/vegan, recent reformulations, or brand news.
    Use this when the user asks about price, availability, certifications,
    or anything time-sensitive not found in the review corpus.

    Args:
        query: The search query string (e.g. 'CeraVe moisturizer current price')
        max_results: Number of web results to return (default 3)

    Returns:
        Concatenated text snippets from top web results
    """
    # Scope the search to beauty/Amazon context
    scoped_query = f"{query} Amazon beauty product"

    try:
        results = tavily_client.search(scoped_query, max_results=max_results)
        snippets = [r["content"] for r in results.get("results", [])]
        return "\n\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Web searcg failed: {str(e)}"

TOOL_TRIGGER_KEYWORDS = {
    # Price / availability
    "price", "cost", "cheap", "expensive", "affordable", "buy",
    "available", "stock", "where to buy", "shipping",
    # Time-sensitive
    "latest", "new", "recent", "2024", "2025", "2026", "discontinued", "reformulated",
    # Certifications not in reviews
    "cruelty-free", "vegan", "organic", "certified", "fda", "approved",
    # Retail
    "shoppers", "walmart", "sephora", "ulta", "drugstore"
}

def should_use_web_search(query: str, docs: list) -> bool:
    """
    Heuristic to decide if web search would help.
    Triggers when the query mentions something the static Amazon
    review corpus cannot answer: pricing, availability, certifications,
    recent reformulations, or retail store locations.
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in TOOL_TRIGGER_KEYWORDS)
    


