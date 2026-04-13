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
#   make all           - run full pipeline from scratch
# ─────────────────────────────────────────────────────────────

# Variables 
PYTHON = python
ENV_NAME = 575_project

# ── Help ─────────────────────────────────────────────────────
# Default target - runs when you just type "make" with no command
# Lists all available commands so users know what's available
help:
	@echo "Available commands:"
	@echo "  make setup      - create conda environment from environment.yml"
	@echo "  make data       - download raw data and convert to parquet"
	@echo "  make bm25       - build and save BM25 index"
	@echo "  make semantic   - build and save semantic search (FAISS) index"
	@echo "  make indexes    - build both BM25 and semantic indexes"
	@echo "  make all        - run full pipeline (data + indexes)"
	@echo "  make clean      - remove processed files and indexes"

# ── Environment Setup ─────────────────────────────────────────
# Creates the conda environment from environment.yml
# Run this once when setting up the project for the first time
setup:
	@echo "Creating conda environment: $(ENV_NAME)..."
	conda env create -f environment.yml
	@echo "Done! Activate with: conda activate $(ENV_NAME)"

# Update environment if environment.yml changes
update-env:
	@echo "Updating conda environment..."
	conda env update -f environment.yml --prune

# ── Data Pipeline ─────────────────────────────────────────────
# Downloads raw data from HuggingFace and converts to parquet
# Checks if parquet files already exist to avoid re-downloading
data:
	@echo "=== Downloading and processing data ==="
	$(PYTHON) src/data_loader.py
	@echo "=== Data pipeline complete ==="

# ── Index Building ────────────────────────────────────────────
# Build BM25 index - depends on data being downloaded first
# If data/processed/meta_All_Beauty.parquet doesn't exist, run data first
bm25: 
	@echo "=== Building BM25 index ==="
	$(PYTHON) src/bm25.py
	@echo "=== BM25 index saved to data/processed/ ==="

# Build semantic search index - depends on data being downloaded first
# Warning: this takes 2-3 minutes (encoding 112K documents)
semantic:
	@echo "=== Building semantic search index ==="
	@echo "Warning: this may take 2-3 minutes..."
	$(PYTHON) src/semantic.py
	@echo "=== FAISS index saved to data/processed/ ==="

# Build both indexes
indexes: bm25 semantic
	@echo "=== Both indexes built successfully ==="

# ── Full Pipeline ─────────────────────────────────────────────
# Runs everything from scratch in the correct order:
# data → bm25 → semantic
all: data indexes
	@echo "=== Full pipeline complete. Ready to run the app! ==="

# ── Cleanup ───────────────────────────────────────────────────
# Removes processed files and indexes so you can rebuild from scratch
# Use with caution - you'll need to re-run the pipeline after this
clean:
	@echo "Removing processed files and indexes..."
	rm -f data/processed/*.parquet
	rm -f data/processed/*.pkl
	rm -f data/processed/*.faiss
	rm -f data/processed/*.npy
	@echo "Clean complete. Run 'make all' to rebuild."

# ── Marks these as non-file targets ───────────────────────────
# Without this, make gets confused if a file named "data" exists
.PHONY: help setup update-env data bm25 semantic indexes all clean
