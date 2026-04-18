# Amazon Assistant
### A Smart Product Search System for Amazon Beauty Products

> **DSCI 575 Project** | University of British Columbia Master of Data Science  
> Built by: Yasaman Eftekharypour and Purity Jangaya

---

Amazon Assistant is a context-aware product search system that retrieves 
relevant Amazon beauty products from natural language queries.

Example queries:

- *"gentle cleanser for sensitive skin"*
- *"something to keep my face hydrated all day"*
- *"CeraVe moisturizer"*

The system uses two complementary retrieval approaches:

- **BM25** — keyword-based search, great for exact terms and brand names
- **Semantic Search** — meaning-based search, great for descriptive queries

---

## Project Status

| Milestone | Focus | Status |
|---|---|---|
| Milestone 1 | Retrieval (BM25 + Semantic) | ✅ In Progress |
| Milestone 2 | RAG + Tool Integration | 🔜 Upcoming |
| Final | Guardrails + Polish | 🔜 Upcoming |

---

## Quick Start (Just want to run the app?)

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- ~2GB free disk space (for data and indexes)
- Internet connection (for initial data download)

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_yasieft_purityj.git
cd DSCI_575_project_yasieft_purityj
```

### 2. Set up environment

#### for windows users
- create the environment: conda env create -f environment.yml

#### for limux-based systems

```bash
make setup
conda activate 575_project
```

### 3. Configure API keys (optional)

Create a `.env` file in the repo root.
Get a free HuggingFace token at: https://huggingface.co/settings/tokens and put here.
HF_TOKEN=your_huggingface_token_here

### 4. Download data and build indexes

```bash
make all
```

> ⚠️ First run takes ~25 minutes total:
> - Data download: ~5 minutes
> - BM25 index: ~10 seconds  
> - Semantic index: ~20 minutes (run once, saved to disk)

### 5. Launch the app

```bash
make app
```

## Demo

![Demo](demo_rag.gif)

---

## Developers

Follow these steps to reproduce the entire pipeline from scratch.

### Repository Structure

```

DSCI_575_project_yasieft_purityj/
│
├── README.md                        
├── Makefile                         
├── environment.yml                  
├── .env                             # never commit
│
├── data/
│   ├── raw/                         # gitignored
│   └── processed/                   # gitignored
│
├── notebooks/
│   └── milestone1_exploration.ipynb 
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               
│   ├── utils.py                     
│   ├── bm25.py                      
│   └── semantic.py                  
│
├── results/
│   └── milestone1_discussion.md     
│
└── app/
    └── app.py                       
```

### Step 1: Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate 575_project
```

Verify installation:

```bash
python -c "import rank_bm25; import sentence_transformers; import faiss; print('All good!')"
```

### Step 2: API Keys (optional)

>> reference above

### Step 3: Download Data

Downloads the All_Beauty category from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) and converts to Parquet format for efficient querying.

```bash
make data
```

This produces:

```
data/raw/meta_All_Beauty.jsonl       # 112,590 product records
data/raw/All_Beauty.jsonl            # 701,528 review records
data/processed/meta_All_Beauty.parquet
data/processed/All_Beauty.parquet
```

### Step 4: Build Search Indexes

Build both retrieval indexes:

```bash
make indexes    # BM25 (~10 seconds) + semantic (~20 minutes, run once)
```

Or build individually:

```bash
make bm25       # builds BM25 index (~10 seconds)
make semantic   # builds FAISS index (~2-7 minutes)
```

```
This produces:
data/processed/bm25_index.pkl        # fitted BM25 index
data/processed/corpus_metadata.pkl   # product metadata for display
data/processed/faiss_index.faiss     # FAISS vector index
data/processed/embeddings.npy        # raw embeddings (384-dim vectors)
```

### Step 5: Launch App

```bash
make app
```

---

## How It Works

### Data Pipeline

```
HuggingFace (raw data)
↓  data_loader.py
data/raw/.jsonl          (incremental write, never fully in RAM)
↓  data_loader.py
data/processed/.parquet  (columnar format, efficient querying)
↓  utils.py
Document corpus           (title + features + description + store + details)
↓
BM25 index    +    FAISS index
```

### Document Construction

Each product is indexed as a single text document combining:

| Field | Why included |
|---|---|
| `title` | Most reliable field, always present (0% missing) |
| `features` | Rich product attributes (missing in 84.6% — included when available) |
| `description` | Longer context (missing in 83% — included when available) |
| `store` | Brand names not always captured in details |
| `details` | Skin type, material, brand, size |
| `price` | **Excluded** — 84.3% missing, adds noise |

### Retrieval Methods

**BM25 (Keyword Search)**

- Scores documents by term frequency and inverse document frequency
- Best for: exact brand names, specific product types
- Example: *"CeraVe moisturizer"* → finds CeraVe products reliably

**Semantic Search**

- Encodes text as 384-dimensional vectors using `all-MiniLM-L6-v2`
- Finds products by meaning similarity using FAISS
- Best for: descriptive queries, synonyms, conceptual searches
- Example: *"something hydrating for dry skin"* → finds moisturizers 
  even without exact word matches

---

## Dataset

We use the **All_Beauty** category from the 
[Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) 
by McAuley Lab.

| File | Records | Description |
|---|---|---|
| Metadata | 112,590 | Product titles, features, descriptions, prices |
| Reviews | 701,528 | User ratings, review text, helpful votes |

**Why All_Beauty?**

- Manageable size — works on any laptop
- Rich natural language queries are natural for beauty products
- Good mix of keyword queries (brand names) and semantic queries 
  (skin concerns, product effects)
- Scales to the larger `Beauty_and_Personal_Care` category 
  (1M+ products) with no code changes

> See `results/milestone1_discussion.md` for full EDA findings 
> and data quality analysis.

---

## Development Notes

### Key Design Decisions

**Parquet + DuckDB over raw JSONL**  
Raw JSONL files are read sequentially — loading 112K records into 
pandas crashes on lower-end laptops. We convert to Parquet (columnar 
format) and query with DuckDB, loading only the columns we need.

**Incremental processing**  
Data is processed in chunks of 1000 records at a time. This keeps 
memory usage bounded regardless of dataset size.

**Shared corpus metadata**  
Both BM25 and semantic search use the same `corpus_metadata.pkl` 
for displaying results. Both systems return integer indices — we 
use those to look up product info from the shared metadata list.

**Type-agnostic field handling**  
Array fields (`features`, `description`) load as numpy arrays from 
Parquet, not Python lists. We use `len()` checks instead of 
`isinstance(x, list)` to handle this correctly.

### Running Individual Components

```bash
# Download and process data
python src/data_loader.py

# Build BM25 index
python src/bm25.py

# Build semantic search index  
python src/semantic.py
```

---

## Team

| Name | GitHub |
|---|---|
| Purity Jangaya | @purityj |
| Yasaman Eftekharypour | @jasmine |

---

## License

This project is for educational purposes as part of UBC MDS DSCI 575.  
Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) 
— McAuley Lab, UC San Diego.
