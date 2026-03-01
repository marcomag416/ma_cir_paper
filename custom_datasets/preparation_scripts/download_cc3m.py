import json
import os
import requests
from tqdm import tqdm

"""
 Before running, download metadata files using:
    wget "https://huggingface.co/api/datasets/pixparse/cc3m-wds/parquet/default/validation" -O .data/cc3m/metadata/validation_parquet_urls.txt
    wget "https://huggingface.co/api/datasets/pixparse/cc3m-wds/parquet/default/train" -O .data/cc3m/metadata/train_parquet_urls.txt
"""

def download_files(url_file_path, output_dir):
    """
    Reads a JSON list of URLs from a file and downloads them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    with open(url_file_path, 'r') as f:
        urls = json.load(f)

    print(f"Found {len(urls)} URLs in {url_file_path}")

    for url in urls:
        filename = url.split('/')[-1]
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}, skipping.")
            continue

        print(f"Downloading {url} to {output_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    # Define paths relative to the script location or workspace root
    # Assuming script is run from datasets/preparation_scripts/ or root
    
    # Base data directory
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cc3m")
    
    metadata_dir = os.path.join(base_data_dir, "metadata")
    
    # Train files
    train_urls_file = os.path.join(metadata_dir, "train_parquet_urls.txt")
    train_output_dir = os.path.join(base_data_dir, "train")
    
    print(f"Processing Train URLs from: {train_urls_file}")
    if os.path.exists(train_urls_file):
        download_files(train_urls_file, train_output_dir)
    else:
        print(f"File not found: {train_urls_file}")

    # Validation files
    val_urls_file = os.path.join(metadata_dir, "validation_parquet_urls.txt")
    val_output_dir = os.path.join(base_data_dir, "validation")
    
    print(f"Processing Validation URLs from: {val_urls_file}")
    if os.path.exists(val_urls_file):
        download_files(val_urls_file, val_output_dir)
    else:
        print(f"File not found: {val_urls_file}")
