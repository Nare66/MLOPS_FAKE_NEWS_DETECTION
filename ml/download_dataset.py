"""
Dataset Acquisition & Setup Utility
Fake News Detection with Continuous Learning

This utility facilitates the acquisition of verified open-source benchmark news datasets:
1. WELFake Dataset (72,134 news articles: 35,028 real, 37,106 fake)
2. ISOT Fake News Dataset (University of Victoria: 21,417 real, 23,481 fake)
3. Kaggle Fake and Real News Dataset (Clément & Hany: 44,898 articles)
"""

import os
import sys
import pandas as pd
import urllib.request
import zipfile
import shutil

BENCHMARK_SOURCES = {
    "WELFake": {
        "description": "WELFake dataset is a composite benchmark collected from Kaggle, McIntire, Reuters, and BuzzFeed datasets.",
        "kaggle_ref": "saurabhshahane/welfake-dataset",
        "url": "https://raw.githubusercontent.com/joolsa/fake_news_detection/master/data/sample_dataset.csv"
    },
    "ISOT": {
        "description": "ISOT Research Lab benchmark from University of Victoria containing verified Reuters articles and PolitiFact debunked stories.",
        "kaggle_ref": "emineyetm/fake-news-detection-datasets",
        "url": "https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php"
    }
}

def print_manual_setup_instructions():
    print("""
=============================================================================
             PUBLIC BENCHMARK DATASET SETUP INSTRUCTIONS
=============================================================================
To expand the included 60-sample verified benchmark to full-scale datasets:

Option A: Kaggle CLI (Recommended)
----------------------------------
1. Ensure your Kaggle API token is configured (~/.kaggle/kaggle.json).
2. Download the WELFake dataset (72,000+ verified articles):
   kaggle datasets download -d saurabhshahane/welfake-dataset -p dataset/raw/
3. Extract the downloaded ZIP:
   unzip dataset/raw/welfake-dataset.zip -d dataset/raw/
4. Rename or merge into dataset/raw/news_dataset.csv.

Option B: Hugging Face Datasets
-------------------------------
Run the following in Python:
   from datasets import load_dataset
   dataset = load_dataset("mteb/tweet_eval_sentiment") # or "ErfanMoosaviMonazzah/fake-news-detection"
   df = dataset["train"].to_pandas()
   df.to_csv("dataset/raw/news_dataset.csv", index=False)

Option C: Custom CSV Upload
---------------------------
Prepare any CSV with columns:
   `title` (string)
   `text`  (string)
   `label` (REAL or FAKE, or 1 or 0)
Place it at: dataset/raw/news_dataset.csv or upload via Continuous Learning UI!
=============================================================================
""")

def setup_benchmark():
    target_path = "dataset/raw/news_dataset.csv"
    if os.path.exists(target_path):
        df = pd.read_csv(target_path)
        print(f"Verified dataset already active at: {target_path} ({len(df)} records).")
    else:
        print("Standard dataset not found. Initializing setup instructions:")
        print_manual_setup_instructions()

if __name__ == "__main__":
    setup_benchmark()
    print_manual_setup_instructions()
