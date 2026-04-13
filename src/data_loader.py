import json
import gzip
import duckdb
import pandas as pd
import os
from pathlib import Path
from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# set HuggingFace token 
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("HF_TOKEN loaded successfully")
else:
    print("Warning: No HF_TOKEN found in .env - downloads may be slower")

def download_data(output_dir="data/raw"):
    """
    Downloads All_Beauty files from HuggingFace.
    Saves as .jsonl (uncompressed) for easier incremental reading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "meta_All_Beauty.jsonl"
    review_path = output_dir / "All_Beauty.jsonl"

    print("Downloading All_Beauty metadata...")
    meta_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_All_Beauty",
        split="full",
        trust_remote_code=True
    )

    # Save incrementally to jsonl
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, "w") as f:
        for record in meta_dataset:
            f.write(json.dumps(record) + "\n")

    # Load reviews
    print("Downloading All_Beauty reviews...")
    review_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True
    )

    print(f"Saving reviews to {review_path}...")
    with open(review_path, "w") as f:
        for record in review_dataset:
            f.write(json.dumps(record) + "\n")

    print("Done! Now run convert_to_parquet() to prepare for retrieval.")
    return meta_path, review_path


def convert_to_parquet(
    raw_dir="data/raw",
    processed_dir="data/processed",
    chunk_size=1000
):
    """
    Converts raw JSONL files to Parquet format incrementally.
    Reads in chunks so we never load the full file into RAM.
    chunk_size: number of lines to process at a time
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "meta_All_Beauty.jsonl": "meta_All_Beauty.parquet",
        "All_Beauty.jsonl": "All_Beauty.parquet"
    }

    for jsonl_file, parquet_file in files.items():
        input_path = raw_dir / jsonl_file
        output_path = processed_dir / parquet_file

        if not input_path.exists():
            print(f"Skipping {jsonl_file} - not found")
            continue

        print(f"Converting {jsonl_file} to parquet...")

        chunks = []
        chunk = []

        with open(input_path, "r") as f:
            for i, line in enumerate(f):
                chunk.append(json.loads(line.strip()))

                if len(chunk) == chunk_size:
                    chunks.append(pd.DataFrame(chunk))
                    chunk = []
                    print(f"  Processed {i+1} records...", end="\r")

            # last partial chunk
            if chunk:
                chunks.append(pd.DataFrame(chunk))

        # Combine and save as parquet
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(output_path, index=False)
        print(f"  Saved {len(df)} records to {output_path}")


def load_metadata_with_duckdb(
    parquet_path="data/processed/meta_All_Beauty.parquet",
    columns=None,
    limit=None
):
    """
    Load metadata using DuckDB - memory efficient, only loads what you need.
    
    columns: list of column names to load. None = all columns
    limit: max number of rows. None = all rows
    
    Example:
        df = load_metadata_with_duckdb(columns=["parent_asin", "title", "price"])
    """
    col_str = ", ".join(columns) if columns else "*"
    limit_str = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT {col_str}
        FROM read_parquet('{parquet_path}')
        {limit_str}
    """

    return duckdb.query(query).df()


def load_reviews_with_duckdb(
    parquet_path="data/processed/All_Beauty.parquet",
    columns=None,
    limit=None
):
    """
    Load review using DuckDB - memory efficient, only loads what you need.
    
    columns: list of column names to load. None = all columns
    limit: max number of rows. None = all rows
    
    Example:
        df = load_metadata_with_duckdb(columns=["parent_asin", "title", "price"])
    """
    col_str = ", ".join(columns) if columns else "*"
    limit_str = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT {col_str}
        FROM read_parquet('{parquet_path}')
        {limit_str}
    """

    return duckdb.query(query).df()


if __name__ == "__main__":
    # # Step 1: Download
    # download_data()

    # # Step 2: Convert to parquet incrementally
    # convert_to_parquet()

    # # Step 3: Quick sanity check with DuckDB
    # print("\nSanity check - first 3 metadata records:")
    # df = load_metadata_with_duckdb(
    #     columns=["parent_asin", "title", "price", "average_rating"],
    #     limit=3
    # )
    # print(df)
    
    reviews = load_reviews_with_duckdb(columns=["text"])
    print(reviews)