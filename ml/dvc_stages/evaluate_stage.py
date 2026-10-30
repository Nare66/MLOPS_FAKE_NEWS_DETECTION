import os
import json
import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml.evaluate import compute_classification_metrics, save_metrics

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    eval_cfg = params.get("evaluation", {})
    min_acc = eval_cfg.get("min_accuracy_threshold", 0.75)
    min_f1 = eval_cfg.get("min_f1_threshold", 0.75)

    test_csv = "dataset/processed/test.csv"
    model_dir = "models/candidate/model"
    eval_dir = "models/candidate/eval"
    os.makedirs(eval_dir, exist_ok=True)

    print("Running DVC Stage: evaluate")
    df = pd.read_csv(test_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = df["label"].tolist()
    y_pred = []

    for text in df["content"]:
        inputs = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = int(torch.argmax(outputs.logits, dim=1).cpu().item())
            y_pred.append(pred)

    metrics = compute_classification_metrics(y_true, y_pred)
    passed = (metrics["accuracy"] >= min_acc) and (metrics["f1"] >= min_f1)
    metrics["passed_evaluation"] = passed
    metrics["thresholds"] = {"min_accuracy": min_acc, "min_f1": min_f1}

    save_metrics(metrics, os.path.join(eval_dir, "metrics.json"))
    print(f"Evaluation complete. Accuracy: {metrics['accuracy']} | F1: {metrics['f1']} | Passed: {passed}")

if __name__ == "__main__":
    main()
