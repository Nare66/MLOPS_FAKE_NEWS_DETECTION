import os
import yaml
from ml.train import train_bert_model

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model_cfg = params.get("model", {})
    train_cfg = params.get("training", {})

    print("Running DVC Stage: train")
    train_bert_model(
        train_csv="dataset/processed/train.csv",
        val_csv="dataset/processed/val.csv",
        output_dir="models/candidate/model",
        base_model_name=model_cfg.get("base_model", "distilbert-base-uncased"),
        epochs=train_cfg.get("epochs", 3),
        batch_size=train_cfg.get("batch_size", 8),
        lr=float(train_cfg.get("learning_rate", 2e-5)),
        max_length=model_cfg.get("max_seq_length", 128),
        model_version="candidate-v1.0.0"
    )

if __name__ == "__main__":
    main()
