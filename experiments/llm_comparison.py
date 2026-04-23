import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rag_pipeline import semantic_retrieve, build_context, build_prompt

MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    #"whisper-large-v3": "whisper-large-v3", # does not support chatcompletion
   # "whisper-large-v3-turbo": "whisper-large-v3-turbo",
    "qwen/qwen3-32b": "qwen/qwen3-32b"


}

TEST_QUERIES = [
    "What are the best moisturizers for dry skin?",
    "Something to keep my face hydrated all day",
    "Product for dark spots and uneven skin tone",
    "Is CeraVe good for sensitive skin?",
    "Affordable anti-aging serum",
    "Top 5 Prices of CeraVe"
]

PROMPT_VERSION = "v1"

for model_name, model_id in MODELS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_id})")
    print(f"{'='*60}")

    llm = ChatGroq(
        model=model_id,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )

    for query in TEST_QUERIES:
        docs = semantic_retrieve(query, top_k=5)
        context = build_context(docs)
        prompt = build_prompt(query, context, version=PROMPT_VERSION)
        response = llm.invoke(prompt)
        answer = response.content.strip()

        print(f"\nQUERY: {query}")
        print(f"ANSWER: {answer}")
        print("-" * 40)