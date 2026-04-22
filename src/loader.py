import os
import streamlit as st
from huggingface_hub import hf_hub_download
from pathlib import Path

HF_REPO_ID = "Purityj01/amazon_beauty_search_index"

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

    # app.py already set this from st.secrets or .env
    hf_token = os.getenv("HF_TOKEN")

    print("Starting index file downloads...")

    for filename in FILES_TO_DOWNLOAD:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"Downloading {filename} from HuggingFace...")
            st.info(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
                local_dir=local_dir
            )
            print(f"    {filename} downloaded")
        else:
            print(f"    {filename} already exists, skipping")

    st.success("All index files ready!")
    
