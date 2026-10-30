import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for Fake News binary classification.
    0: FAKE, 1: REAL
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm is [[TN, FP], [FN, TP]] where 0 is Fake (Negative) and 1 is Real (Positive)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "sample_size": len(y_true),
        "real_count": int(sum(y_true)),
        "fake_count": int(len(y_true) - sum(y_true))
    }
    return metrics

def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")
