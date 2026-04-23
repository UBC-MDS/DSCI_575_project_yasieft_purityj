# ─────────────────────────────────────────────────────────────
# Makefile
# Purpose: Simplify running the pipeline steps so users don't
#          need to know the underlying Python commands.
#
# Usage:
#   make help          - show available commands
#   make setup         - create conda environment
#   make data          - download and convert data
#   make bm25          - build BM25 index
#   make semantic      - build semantic search index
#   make indexes       - build both indexes
#   make all           - run full pipeline from scratch
#   make app           - launch the Streamlit app
#   make download      - download index files from HuggingFace
#   make experiment    - run LLM comparison experiment
#   make clean         - remove processed files and indexes
# ─────────────────────────────────────────────────────────────

# Variables
PYTHON = python
ENV_NAME = 575_project

# ── Help ─────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Amazon Beauty Assistant — available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup         create conda environment from environment.yml"
	@echo "    make update-env    update conda environment if environment.yml changed"
	@echo "    make install       install dependencies via pip (requirements.txt)"
	@echo ""
	@echo "  Data pipeline:"
	@echo "    make data          download raw data and convert to parquet"
	@echo "    make download      download pre-built indexes from HuggingFace (faster)"
	@echo ""
	@echo "  Index building (run after make data):"
	@echo "    make bm25          build and save BM25 index (~10 seconds)"
	@echo "    make semantic      build and save FAISS semantic index (~20 minutes)"
	@echo "    make indexes       build both BM25 and semantic indexes"
	@echo ""
	@echo "  Full pipeline:"
	@echo "    make all           run full pipeline: data + indexes"
	@echo ""
	@echo "  App:"
	@echo "    make app           launch the Streamlit app locally"
	@echo ""
	@echo "  Experiments:"
	@echo "    make experiment    run LLM comparison experiment (requires GROQ_API_KEY)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         remove processed files and indexes"
	@echo "    make clean-raw     remove raw downloaded data only"
	@echo "    make clean-all     remove everything (raw + processed)"
	@echo ""

# ── Environment Setup ─────────────────────────────────────────
setup:
	@echo "Creating conda environment: $(ENV_NAME)..."
	conda env create -f environment.yml
	@echo "Done! Activate with: conda activate $(ENV_NAME)"

update-env:
	@echo "Updating conda environment..."
	conda env update -f environment.yml --prune

install:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo "Done!"

# ── Data Pipeline ─────────────────────────────────────────────
data:
	@echo "=== Downloading and processing data ==="
	@echo "Warning: downloads ~800MB from HuggingFace, takes ~5 minutes"
	$(PYTHON) src/data_loader.py
	@echo "=== Data pipeline complete ==="

# Download pre-built indexes from HuggingFace
# Faster alternative to building indexes locally (skips ~20 min semantic build)
# Requires HF_TOKEN in .env
download:
	@echo "=== Downloading pre-built index files from HuggingFace ==="
	@echo "Requires HF_TOKEN in .env"
	$(PYTHON) -c "from src.loader import download_index_files; download_index_files()"
	@echo "=== Index files downloaded to data/processed/ ==="

# ── Index Building ────────────────────────────────────────────
bm25:
	@echo "=== Building BM25 index ==="
	$(PYTHON) src/bm25.py
	@echo "=== BM25 index saved to data/processed/ ==="

semantic:
	@echo "=== Building semantic search index ==="
	@echo "Warning: encoding 112K documents takes ~20 minutes on CPU"
	@echo "This only needs to be done once - index is saved to disk"
	$(PYTHON) src/semantic.py
	@echo "=== FAISS index saved to data/processed/ ==="

indexes: bm25 semantic
	@echo "=== Both indexes built successfully ==="

# ── Full Pipeline ─────────────────────────────────────────────
all: data indexes
	@echo "=== Full pipeline complete. Run 'make app' to launch. ==="

# ── Web App ───────────────────────────────────────────────────
app:
	@echo "=== Launching Streamlit app at http://localhost:8501 ==="
	streamlit run app/app.py --server.fileWatcherType=none

# ── Experiments ───────────────────────────────────────────────
experiment:
	@echo "=== Running LLM comparison experiment ==="
	@echo "Requires GROQ_API_KEY in .env"
	$(PYTHON) experiments/llm_comparison.py
	@echo "=== Experiment complete ==="

# ── Cleanup ───────────────────────────────────────────────────
clean:
	@echo "Removing processed files and indexes..."
	rm -f data/processed/*.parquet
	rm -f data/processed/*.pkl
	rm -f data/processed/*.faiss
	rm -f data/processed/*.npy
	@echo "Clean complete. Run 'make all' to rebuild."

clean-raw:
	@echo "Removing raw downloaded data..."
	rm -f data/raw/*.jsonl
	rm -f data/raw/*.jsonl.gz
	@echo "Raw data removed. Run 'make data' to re-download."

clean-all: clean clean-raw
	@echo "All data removed. Run 'make all' to rebuild from scratch."

# ── Marks these as non-file targets ───────────────────────────
.PHONY: help setup update-env install data download bm25 semantic indexes all app experiment clean clean-raw clean-all