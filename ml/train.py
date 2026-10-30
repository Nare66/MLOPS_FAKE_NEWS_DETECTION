import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch

from ml.evaluate import compute_classification_metrics, save_metrics

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_bert_model(
    train_csv: str = "dataset/processed/train.csv",
    val_csv: str = "dataset/processed/val.csv",
    output_dir: str = "models/candidate",
    base_model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 128,
    model_version: str = "v1.0.0",
    mlflow_tracking_uri: str = None,
    experiment_name: str = "FakeNewsDetection-BERT"
):
    """
    Fine-tunes BERT model for Fake News Classification with MLflow experiment tracking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting BERT Fine-Tuning on {device} ===")
    print(f"Base Model: {base_model_name} | Version: {model_version} | Epochs: {epochs}")

    # Set up MLflow with SQLite backend
    os.makedirs("mlruns", exist_ok=True)
    db_path = os.path.abspath("mlruns/mlflow.db").replace("\\", "/")
    tracking_uri = mlflow_tracking_uri or f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1}
    )
    model.to(device)

    train_dataset = NewsDataset(train_df["content"], train_df["label"], tokenizer, max_length)
    val_dataset = NewsDataset(val_df["content"], val_df["label"], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    with mlflow.start_run(run_name=f"Run_{model_version}_{int(time.time())}") as run:
        # Log parameters
        mlflow.log_params({
            "base_model": base_model_name,
            "model_version": model_version,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "optimizer": "AdamW",
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "device": str(device)
        })

        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_metrics = compute_classification_metrics(all_labels, all_preds)

            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")

            # Log metrics to MLflow per epoch
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"]
            }, step=epoch)

        total_duration = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", round(total_duration, 2))

        # Save artifacts locally
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            "model_version": model_version,
            "base_model": base_model_name,
            "training_completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_duration_seconds": round(total_duration, 2),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "mlflow_run_id": run.info.run_id,
            "history": history,
            "final_validation_metrics": val_metrics
        }

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        save_metrics(val_metrics, os.path.join(output_dir, "metrics.json"))

        # Log local artifact directory to MLflow
        mlflow.log_artifacts(output_dir, artifact_path="model_package")
        print(f"Training completed successfully! Model & metrics saved to: {output_dir}")
        print(f"MLflow Run ID: {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "output_dir": output_dir,
            "metrics": val_metrics,
            "history": history,
            "metadata": metadata
        }

if __name__ == "__main__":
    train_bert_model()
