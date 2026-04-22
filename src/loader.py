import os
import streamlit as st
from huggingface_hub import hf_hub_download
from pathlib import Path
from dotenv import load_dotenv



load_dotenv()
HF_REPO_ID = "Purityj01/amazon_beauty_search_index"

try:
    HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

FILES_TO_DOWNLOAD = [
    "bm25_index.pkl",
    "corpus_metadata.pkl", 
    "faiss_index.faiss",
    "All_Beauty.parquet",
    "meta_All_Beauty.parquet"
]

def download_index_files(local_dir="data/processed"):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES_TO_DOWNLOAD:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"Downloading {filename} from HuggingFace...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=local_dir
            )
            print(f"    {filename} downloaded")
        else:
            print(f"    {filename} already exists, skipping")
    