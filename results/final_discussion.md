# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

We used the full **All_Beauty** category from the Amazon Reviews 2023 dataset (McAuley Lab, UC San Diego), with no sampling or truncation.

| File | Records |
|---|---|
| Product metadata | 112,590 products |
| Customer reviews | 701,528 reviews |

This exceeds the 10,000-product minimum requirement. Both the BM25 index and the FAISS semantic index were built over the full 112,590 product corpus. The dataset was chosen because it is large enough to stress-test retrieval quality while remaining manageable on a laptop without a GPU.

---

### LLM Experiment

We compared four models available via the Groq API, running identical queries with identical retrieved context and prompts across all models. Semantic retrieval (top-5 products) was used for all queries to ensure the context fed to each model was the same.

**Models compared:**

| Model | Family | Size | Provider |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Llama 3.3 | 70B | Meta via Groq |
| `llama-3.1-8b-instant` | Llama 3.1 | 8B | Meta via Groq |
| `openai/gpt-oss-120b` | GPT OSS | 120B | OpenAI via Groq |
| `qwen/qwen3-32b` | Qwen 3 | 32B | Alibaba via Groq |

**Prompt used (identical across all models):**

```
You are a helpful Amazon shopping assistant.

STRICT RULES:
- Answer using the provided Amazon product information (metadata, descriptions, and customer reviews)
- Do NOT make up information
- Do NOT repeat the question or instructions
- Do NOT output words like "assistant" or tags like "<think>"
- Keep the answer short and clear (2–4 sentences max)

Output ONLY the final answer. No extra text.

Answer using the provided Amazon product information.

Product Information and Context:
{retrieved context — top 5 products from semantic search}

Question is: {query}

FINAL ANSWER:
```

**Queries and outputs:**

---

**Query 1: "What are the best moisturizers for dry skin?"**

| Model | Output |
|---|---|
| Llama 3.3 70B | The DayTime Moisturizer for Dry Skin (4.1★) and Psoriasis & Eczema Cream (4.0★) are highly rated. The Physicians Formula Elastin & Collagen Moisture Lotion has a 5-star rating, making it a great choice for dry skin. |
| Llama 3.1 8B | The DayTime Moisturizer (B01M1A0H4Z) and Physicians Formula Elastin & Collagen Moisture Lotion (B00LTJV320) are highly rated options, with average ratings of 4.1 and 5.0 out of 5. |
| GPT-OSS 120B | The DayTime Moisturizer for Dry Skin (4.1★) is a solid choice. Highly rated alternatives include the Neolia Hydra-Screen Apricot Oil Soap (5★) and Physicians Formula Elastin & Collagen Moisture Lotion (5★), both praised for effectiveness and value. |
| Qwen 32B | Psoriasis & Eczema Cream (4.0/5, 58 reviews) with organic shea butter and coconut oil. Physicians Formula Elastin & Collagen Moisture Lotion (5.0/5). DayTime Moisturizer (4.1/5) though pricey at $70. |

---

**Query 2: "Something to keep my face hydrated all day"**

| Model | Output |
|---|---|
| Llama 3.3 70B | One Drop of All Natural Hyaluronic Acid Serum is a great option (4.5★), absorbing quickly and leaving skin silky smooth. The Hydrating Face Mask For Men and Women By ERH (4.1★) and Hydrate Mens Face Moisturizer (4.5★) are also strong options. |
| Llama 3.1 8B | The One Drop of Our All Natural Hyaluronic Acid Serum (B00H2TZ4F8) absorbs quickly and leaves skin feeling silky smooth. |
| GPT-OSS 120B | The All-Natural Hyaluronic Acid Serum (B00H2TZ4F8) is lightweight, fast-absorbing, and leaves skin hydrated all day. Users note it isn't greasy and requires no waiting time before makeup. |
| Qwen 32B | The DAKOKA Hyaluronic Acid Serum (B00H2TZ4F8) is ideal for all-day hydration (4.5★). For a moisturizer, the Hydrate Men's Face Moisturizer + Beard Conditioner (B00YZ3AB66) with hyaluronic acid also provides long-lasting hydration. |

---

**Query 3: "Product for dark spots and uneven skin tone"**

| Model | Output |
|---|---|
| Llama 3.3 70B | BREMENN CLINICAL Dark Spot Eraser and Vice Reversa Pigment Fader are popular for dark spots, with ratings of 3.8–3.9 and 3.5★. They contain vitamin C and niacinamide to visibly reduce dark spots. |
| Llama 3.1 8B | The BREMENN CLINICAL Dark Spot Eraser (B00B0ZIQ64) and TELEPURE Intimate Area Dark Spot Corrector (B099K3QQ8M) received positive reviews for reducing dark spots. |
| GPT-OSS 120B | The Bremenn Clinical Dark Spot Eraser (≈3.9/5 from 200+ reviews) visibly reduces dark spots without irritation and works on face, neck, and hands. Consider after a patch test. |
| Qwen 32B | BREMENN CLINICAL Dark Spot Eraser (3.8–3.9/5, 200+ reviews) shows visible improvement within days to weeks and is suitable for sensitive skin. Vice Reversa Pigment Fader uses niacinamide and vitamin C but is slower and pricier. |

---

**Query 4: "Is CeraVe good for sensitive skin?"**

| Model | Output |
|---|---|
| Llama 3.3 70B | Yes. Reviewers with "very allergic skin" found the CeraVe SA Cream to be the best for their skin. The CeraVe Hydrating Facial Cleanser was also praised as the best cleanser for dry sensitive skin without burning or irritation. |
| Llama 3.1 8B | Yes, CeraVe products are suitable for sensitive skin. The SA Cream helped a customer with allergic skin, and the Hydrating Facial Cleanser was praised for not burning sensitive eyes. |
| GPT-OSS 120B | Yes. Multiple CeraVe products receive high ratings from users with sensitive or allergic skin. The Hydrating Facial Cleanser and SA Cream (4.6–4.9★) are generally well-tolerated for sensitive skin. |
| Qwen 32B | Yes. The CeraVe SA Cream is praised for soothing allergic and irritated skin, and the Hydrating Facial Cleanser is noted as the best for dry, sensitive skin without eye irritation. |

---

**Query 5: "Affordable anti-aging serum"**

| Model | Output |
|---|---|
| Llama 3.3 70B | The Peptide Serum Complex by Living Earth (4.2★, 137 reviews) boosts collagen and reduces wrinkles. The Anti-Aging 50X Vitamin C Serum by DOCTORS PREFERRED (3.6★) is another option. |
| Llama 3.1 8B | The Peptide Serum Complex by Living Earth (B07D1GTDXM) is highly rated (4.2★, 137 reviews), vegan, cruelty-free, and made in the USA. |
| GPT-OSS 120B | The 0.17 oz trial-size Anti-Aging 50X Vitamin C Serum from Doctors Preferred is the most budget-friendly option with a 3.6★ rating. |
| Qwen 32B | The Peptide Serum Complex (B07D1GTDXM, 4.2★, 137 reviews) is organic and praised for reducing wrinkles. The 5.0-rated Anti Aging Facial Serum for Men (B010L93H2C) is also effective but has fewer reviews. |

---

**Key observations:**

- **Llama 3.3 70B** produced the most natural, readable answers with the best balance of detail and conciseness. It consistently followed prompt instructions and cited ratings without over-structuring the output.
- **Llama 3.1 8B** was faster but produced noticeably shorter, more mechanical answers. It leaned heavily on ASINs as identifiers rather than product descriptions, making answers feel less user-friendly.
- **GPT-OSS 120B** produced the cleanest prose and was best at synthesizing information across products. However, it occasionally added unsolicited advice ("consider a patch test") that goes slightly beyond what the retrieved context supports.
- **Qwen 32B** exposed internal `<think>` reasoning blocks in its raw output — a known behavior of the model's chain-of-thought mode. These were handled in the pipeline using `re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)`. The final answers were accurate but over-formatted with bold text and numbered lists, which conflicts with our prompt's instruction to avoid lists.

**Chosen model: `llama-3.3-70b-versatile`**

Llama 3.3 70B was selected as the default because it best followed prompt instructions, produced readable answers, and struck the right balance between answer quality and response length. The 8B model sacrificed too much quality for speed, GPT-OSS occasionally hallucinated advice not grounded in retrieved context, and Qwen required extra output cleaning and formatting overrides.

---

## Step 2: Additional Feature — Tool Integration

### What We Implemented

We added Tavily-powered web search as an optional tool the RAG pipeline can invoke at query time. This addresses a core limitation of a static product corpus: Amazon review data cannot answer questions about current pricing, retail availability, product certifications (cruelty-free, vegan, organic), or recent reformulations.

**How it works:**

The tool is governed by a keyword-based heuristic in `src/tools.py`. Before calling the LLM, the pipeline checks whether the query contains trigger keywords. If matched, it calls Tavily, appends the web results to the product context, and instructs the LLM to use both sources.

```python
TOOL_TRIGGER_KEYWORDS = {
    "price", "cost", "cheap", "expensive", "affordable", "buy",
    "available", "stock", "where to buy", "shipping",
    "latest", "new", "recent", "2024", "2025", "2026", "discontinued", "reformulated",
    "cruelty-free", "vegan", "organic", "certified", "fda", "approved",
    "sephora", "ulta", "walmart", "drugstore"
}
```

The tool is opt-in in the UI — users toggle "Augment with web search" in the RAG tab.

**Example queries where the tool was used:**

**Query 1: "Is CeraVe cruelty-free?"**

Without web search, the static corpus has no certification data. With Tavily enabled, the pipeline retrieved current information confirming CeraVe's cruelty-free status in certain markets and surfaced nuance (e.g. sold in China where animal testing may be required), which the review corpus alone could not provide.

**Query 2: "Where can I buy the CeraVe moisturizer?"**

The static corpus contains Amazon listing data but no current retail availability. With web search enabled, the pipeline returned results mentioning Walmart, Target, Ulta, and CVS alongside pricing — directly answering a question the reviews cannot.

**Query 3: "Top 5 prices of CeraVe"**

Price data in the corpus is sparse (84.3% missing in the raw dataset, which is why it was excluded from document construction). With web search, the pipeline retrieved current retail prices from multiple sources, supplementing the handful of prices present in the corpus.

**Did it improve results?**

Yes, for queries involving time-sensitive or external information. For standard product recommendation queries ("best moisturizer for dry skin"), the tool trigger correctly does not fire — the static corpus is sufficient and web search would add noise. The keyword heuristic is intentionally conservative to avoid unnecessary API calls.

---

## Step 3: Improve Documentation and Code Quality

### Documentation Update

The README was updated to reflect the current state of the project including:

- Updated project status table to reflect all three milestones
- Added RAG and tool integration to the "How It Works" section
- Added Streamlit deployment instructions and public app URL
- Updated repository structure to include `src/hybrid.py`, `src/rag_pipeline.py`, `src/tools.py`, `src/loader.py`, and `results/` files
- Added `requirements.txt` setup path alongside `environment.yml`
- Updated the data pipeline diagram to include the RAG and LLM generation steps

### Code Quality Changes

- **No hardcoded paths**: all file paths use `pathlib.Path` relative to `REPO_ROOT = Path(__file__).parent.parent`
- **No API keys in source**: all keys loaded via `python-dotenv` locally and `st.secrets` on Streamlit Cloud, with a unified `get_secret()` helper in `app.py`
- **Docstrings added** to all public functions across `bm25.py`, `semantic.py`, `utils.py`, `rag_pipeline.py`, `hybrid.py`, `loader.py`, and `tools.py`
- **Removed dead weight**: `embeddings.npy` excluded from deployment (165MB, not used at runtime — FAISS stores vectors internally); `tokenized_corpus` removed from `bm25_index.pkl` (reduced from 97MB to 53MB)
- **Batched review lookups**: replaced per-product DuckDB parquet scans in `build_context()` with a startup-time dict lookup (`review_lookup`), eliminating repeated file I/O during inference
- **Environment file updated**: `requirements.txt` created for Streamlit Cloud deployment alongside the existing `environment.yml` for local conda setup

---

## Step 4: Cloud Deployment Plan

### Current Deployment

The app is currently deployed at [https://beutyfinder-ai.streamlit.app](https://beutyfinder-ai.streamlit.app) using Streamlit Community Cloud. Index files (BM25, FAISS, corpus metadata, parquet files) are hosted on Hugging Face as a private dataset and downloaded at app startup via the `src/loader.py` module. LLM inference is handled via the Groq API (Llama 3.3 70B). Web search is handled via the Tavily API.

The primary limitation of the current deployment is Streamlit Community Cloud's 1GB memory ceiling. Loading the FAISS index (165MB), BM25 index (53MB), sentence transformer model (90MB), corpus metadata (19MB), and the review lookup dict built from `All_Beauty.parquet` (120MB) approaches this limit, causing crashes under active use. Mitigations applied include `@st.cache_resource` for all heavy objects, limiting context to top-3 documents, truncating review snippets to 150 characters, and replacing per-query parquet scans with a startup-time dict lookup.

### Proposed AWS Deployment

Below is a production-grade deployment plan on AWS that addresses the memory, concurrency, and update limitations of the current setup.

#### Data Storage

| Artifact | Storage Solution | Rationale |
|---|---|---|
| Raw JSONL data | S3 (Standard tier) | Infrequently accessed, cheap at scale |
| Processed parquet files | S3 (Standard tier) | Queryable via Athena if needed |
| FAISS vector index | S3 + EFS mount | EFS allows multiple containers to share the index without re-downloading |
| BM25 index | S3 + EFS mount | Same reasoning as FAISS |
| Corpus metadata | S3 + EFS mount | Shared across all app instances |

Raw data stays in S3 and is never loaded into application memory — only the processed indexes are mounted at runtime.

#### Compute

**App hosting**: Deploy the Streamlit app as a Docker container on **AWS ECS (Fargate)**. Fargate removes the need to manage EC2 instances and scales containers automatically based on traffic. A task with 4GB RAM and 2 vCPU would comfortably hold all indexes in memory without the crashes seen on Streamlit Cloud.

**Concurrency**: ECS auto-scaling would spin up additional Fargate tasks under high load, with an **Application Load Balancer** distributing traffic across tasks. Each task loads the indexes independently from EFS on startup. Session state is managed within each container — no shared state is needed between tasks since the pipeline is stateless (query in, answer out).

**LLM inference**: Keep using the **Groq API** for LLM inference rather than hosting a model. At 112K products, query volume is unlikely to justify the cost of a dedicated GPU instance (e.g. `g4dn.xlarge` at ~$0.53/hr). Groq provides fast inference with no infrastructure overhead. If the system scaled to millions of queries, self-hosting Llama 3.3 70B on a GPU instance with **vLLM** would become cost-competitive.

#### Streaming and Updates

**Incorporating new products**: New products would be ingested via a scheduled **AWS Lambda** function (or Glue job) that:
1. Downloads new records from the Amazon Reviews dataset (or a live product feed)
2. Appends to the parquet files in S3
3. Rebuilds the BM25 and FAISS indexes incrementally
4. Uploads the new indexes to S3 and refreshes the EFS mount

For the FAISS index specifically, `IndexFlatIP` does not support incremental addition without full rebuild. At 112K products, a full nightly rebuild takes approximately 20 minutes on a CPU instance, which is acceptable. If the corpus grew to millions of products, switching to `IndexIVFFlat` (approximate search) would reduce rebuild time significantly.

**Keeping the pipeline current**: The sentence transformer model (`all-MiniLM-L6-v2`) and the Groq-hosted Llama model are versioned externally. Model updates would be handled by pinning versions in `requirements.txt` and `environment.yml` and testing against a held-out query set before promoting to production. The Tavily web search tool ensures time-sensitive information (pricing, availability, certifications) stays current regardless of index rebuild frequency.