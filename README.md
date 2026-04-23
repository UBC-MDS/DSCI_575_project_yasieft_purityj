# Amazon Beauty Assistant
### A Smart Product Search and Recommendation System

> **DSCI 575 Project** | University of British Columbia Master of Data Science  
> Built by: Yasaman Eftekharypour and Purity Jangaya

---

🚀 **Live App**: [https://beutyfinder-ai.streamlit.app](https://beutyfinder-ai.streamlit.app)

---

Amazon Beauty Assistant is a context-aware product search and recommendation system that retrieves relevant Amazon beauty products from natural language queries and generates grounded answers using a large language model.

Example queries:

- *"gentle cleanser for sensitive skin"*
- *"something to keep my face hydrated all day"*
- *"Is CeraVe cruelty-free?"*
- *"affordable anti-aging serum under $20"*

---

## Project Status

| Milestone | Focus | Status |
|---|---|---|
| Milestone 1 | Retrieval (BM25 + Semantic) | ✅ Complete |
| Milestone 2 | RAG + LLM Integration | ✅ Complete |
| Final | Hybrid Retrieval, Tool Use, Deployment | ✅ Complete |

---

## Features

- **BM25 keyword search** — exact term and brand name matching
- **Semantic search** — meaning-based retrieval using `all-MiniLM-L6-v2` embeddings and FAISS
- **Hybrid retrieval** — combines BM25 and semantic search via Reciprocal Rank Fusion (RRF)
- **RAG pipeline** — retrieves top products and generates grounded answers using Llama 3.3 70B via Groq
- **Web search tool** — augments answers with live web results for pricing, availability, and certifications via Tavily
- **User feedback** — thumbs up/down feedback stored per query result

---

## Quick Start

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- ~2GB free disk space
- Internet connection (for data download and API calls)

There are two setup paths depending on how much time you have:

| Path | Time | When to use |
|---|---|---|
| **Fast** — download pre-built indexes | ~5 min | Just want to run the app |
| **Full pipeline** — build from scratch | ~25 min | Reproducing the full pipeline |

---

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_yasieft_purityj.git
cd DSCI_575_project_yasieft_purityj
```

### 2. Set up environment

**macOS / Linux (conda):**
```bash
make setup
conda activate 575_project
```

**Windows (conda):**
```bash
conda env create -f environment.yml
conda activate 575_project
```

**Any platform (pip):**
```bash
make install
# or: pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the repo root:

```bash
HF_TOKEN=your_huggingface_token       # https://huggingface.co/settings/tokens
GROQ_API_KEY=your_groq_api_key        # https://console.groq.com
TAVILY_API_KEY=your_tavily_api_key    # https://tavily.com (optional, for web search)
```

- `HF_TOKEN` — required to download index files from Hugging Face at startup
- `GROQ_API_KEY` — required for LLM answer generation
- `TAVILY_API_KEY` — optional, only needed if using the web search augmentation feature

### 4a. Fast path — download pre-built indexes (recommended)

Downloads the pre-built BM25, FAISS, and metadata files directly from HuggingFace. Requires `HF_TOKEN`.

```bash
make download
```

### 4b. Full pipeline — build from scratch

Downloads raw data from HuggingFace and builds all indexes locally.

```bash
make all
```

> ⚠️ First run takes ~25 minutes:
> - Data download: ~5 minutes
> - BM25 index build: ~10 seconds
> - Semantic index build: ~20 minutes (run once, saved to disk)

You can also run steps individually:

```bash
make data       # download and convert raw data only
make bm25       # build BM25 index only
make semantic   # build FAISS index only
make indexes    # build both indexes (skips data download)
```

### 5. Launch the app

```bash
make app
```

The app runs at [http://localhost:8501](http://localhost:8501).

### Other useful commands

```bash
make experiment   # run LLM comparison experiment (requires GROQ_API_KEY)
make clean        # remove processed indexes (keeps raw data)
make clean-raw    # remove raw downloaded data only
make clean-all    # remove everything and start fresh
make update-env   # update conda environment after environment.yml changes
```

---

## Repository Structure

```
DSCI_575_project_yasieft_purityj/
│
├── README.md
├── Makefile
├── environment.yml
├── requirements.txt
├── .env                              # never commit
│
├── data/
│   ├── raw/                          # gitignored
│   └── processed/                    # gitignored
│
├── notebooks/
│   ├── milestone1_exploration.ipynb
│   └── milestone2_rag.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # downloads and converts raw data
│   ├── utils.py                      # document construction and tokenization
│   ├── bm25.py                       # BM25 index build, save, load, search
│   ├── semantic.py                   # FAISS index build, save, load, search
│   ├── hybrid.py                     # RRF hybrid retriever and pipeline
│   ├── rag_pipeline.py               # base RAG pipeline (BM25 + Semantic modes)
│   ├── tools.py                      # Tavily web search tool
│   └── loader.py                     # downloads index files from Hugging Face
│
├── results/
│   ├── milestone1_discussion.md
│   ├── milestone2_discussion.md
│   └── final_discussion.md
│
└── app/
    └── app.py                        # Streamlit UI
```

---

## How It Works

### Data Pipeline

```
HuggingFace (McAuley-Lab/Amazon-Reviews-2023)
↓  src/data_loader.py
data/raw/*.jsonl             (incremental write, never fully in RAM)
↓  src/data_loader.py
data/processed/*.parquet     (columnar format, DuckDB queries)
↓  src/utils.py
Document corpus              (title + features + description + store + details)
↓
BM25 index    +    FAISS index
↓
Hybrid retriever (RRF)
↓
RAG pipeline → Llama 3.3 70B (Groq) → Answer
```

### Document Construction

Each of the 112,590 products is indexed as a single text document combining:

| Field | Why included |
|---|---|
| `title` | Most reliable field, always present |
| `features` | Rich product attributes |
| `description` | Longer product context |
| `store` | Brand names not always in details |
| `details` | Skin type, material, size |
| `price` | **Excluded** — 84.3% missing, adds noise |

### Retrieval Methods

**BM25** scores documents by term frequency and inverse document frequency. Best for exact brand names and specific product types (e.g. *"CeraVe moisturizer"*).

**Semantic search** encodes queries and documents as 384-dimensional vectors using `all-MiniLM-L6-v2` and retrieves nearest neighbours via FAISS. Best for descriptive or conceptual queries (e.g. *"something hydrating for dry skin"*).

**Hybrid retrieval** combines both using Reciprocal Rank Fusion (RRF). Each candidate is scored as the sum of `1 / (rank + 60)` from each retriever, then re-ranked. This consistently outperforms either method alone on mixed query types.

### RAG Pipeline

1. Query → Hybrid retriever → top-5 products
2. Product metadata + one review snippet per product → context block
3. Context + query → prompt → Llama 3.3 70B via Groq API
4. Answer returned and displayed with source citations

### Web Search Tool

When a query contains keywords related to pricing, availability, certifications, or recency (e.g. *"cruelty-free"*, *"where to buy"*, *"price"*), the pipeline optionally calls Tavily to augment the static product context with live web results. This is opt-in via a checkbox in the UI.

---

## Dataset

**All_Beauty** category from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) by McAuley Lab, UC San Diego.

| File | Records |
|---|---|
| Product metadata | 112,590 |
| Customer reviews | 701,528 |

**Why All_Beauty?** Manageable size, rich natural language queries, good mix of keyword (brand names) and semantic (skin concerns) queries. Scales to the larger `Beauty_and_Personal_Care` category (1M+ products) with no code changes.

---

## Development Notes

### Key Design Decisions

**Parquet + DuckDB over raw JSONL** — raw JSONL requires sequential reads. Parquet with DuckDB loads only the columns needed, keeping memory bounded on lower-end machines.

**Incremental processing** — data is processed in chunks of 1,000 records. Memory usage is bounded regardless of dataset size.

**Shared corpus metadata** — both BM25 and FAISS return integer indices. Both retrievers share a single `corpus_metadata.pkl` for result display, avoiding duplication.

**embeddings.npy excluded from deployment** — FAISS stores vectors internally in `faiss_index.faiss`. The raw embeddings file (165MB) is useful for experimentation but not needed at runtime.

**tokenized_corpus removed from BM25 pickle** — saving the tokenized corpus alongside the BM25 object inflated the index from 53MB to 97MB. At query time only `bm25.get_scores()` is called, which does not require the original corpus.

**Review lookup dict over per-query parquet scans** — the original implementation ran a DuckDB parquet scan for each product in the context window. The app now builds a `asin → snippet` dict at startup, replacing repeated file I/O with O(1) dict lookups.

### Running Individual Components

```bash
python src/data_loader.py   # download and convert data
python src/bm25.py          # build BM25 index
python src/semantic.py      # build FAISS index
streamlit run app/app.py    # launch app
```

---

## Team

| Name | GitHub |
|---|---|
| Purity Jangaya | @purityj |
| Yasaman Eftekharypour | @yasieft |

---

## License

Educational use only — UBC MDS DSCI 575.  
Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — McAuley Lab, UC San Diego.
