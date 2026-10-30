import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

def clean_text(text: str) -> str:
    """Preprocess and clean raw news text."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_label(label: Any) -> int:
    """Convert REAL/FAKE, 1/0, true/false to binary int 1 (REAL) or 0 (FAKE)."""
    if isinstance(label, (int, float, np.integer)):
        return 1 if int(label) == 1 else 0
    str_val = str(label).strip().upper()
    if str_val in ["REAL", "TRUE", "1", "GENUINE"]:
        return 1
    elif str_val in ["FAKE", "FALSE", "0", "DISINFO"]:
        return 0
    else:
        raise ValueError(f"Unknown label value: {label}")

def prepare_dataset(
    raw_csv_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    output_dir: str = "dataset/processed",
    reference_dir: str = "dataset/reference"
) -> Dict[str, str]:
    """
    Load raw CSV, clean, standardize columns, stratify split, and save processed datasets.
    Also produces reference_data.csv for Evidently AI drift monitoring baseline.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)

    # Validate column requirements
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column with values REAL/FAKE or 1/0.")

    # Determine text source
    if "text" in df.columns and "title" in df.columns:
        df["content"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    elif "title" in df.columns:
        df["content"] = df["title"].fillna("")
    else:
        raise ValueError("Dataset must contain either 'text' or 'title' column.")

    df["content"] = df["content"].apply(clean_text)
    df["label"] = df["label"].apply(normalize_label)

    # Filter out empty content rows
    df = df[df["content"].str.len() > 10].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Dataset too small ({len(df)} valid records). At least 10 records required.")

    # Stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df["label"] if len(df["label"].unique()) > 1 else None
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_ratio),
        random_state=random_state,
        stratify=temp_df["label"] if len(temp_df["label"].unique()) > 1 else None
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    reference_path = os.path.join(reference_dir, "reference_data.csv")

    cols_to_save = ["content", "label"]
    train_df[cols_to_save].to_csv(train_path, index=False)
    val_df[cols_to_save].to_csv(val_path, index=False)
    test_df[cols_to_save].to_csv(test_path, index=False)

    # The reference data for Evidently AI baseline drift is the training set
    # with text metrics like text_length
    ref_df = train_df.copy()
    ref_df["text_length"] = ref_df["content"].apply(len)
    ref_df["word_count"] = ref_df["content"].apply(lambda x: len(x.split()))
    ref_df.to_csv(reference_path, index=False)

    print(f"Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "reference": reference_path,
        "total_records": len(df),
        "real_count": int((df["label"] == 1).sum()),
        "fake_count": int((df["label"] == 0).sum())
    }

if __name__ == "__main__":
    prepare_dataset("dataset/raw/news_dataset.csv")
