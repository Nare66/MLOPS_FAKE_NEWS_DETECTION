import os
import yaml
from ml.data_processor import prepare_dataset

def main():
    params_file = "params.yaml"
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    data_cfg = params.get("data", {})
    raw_path = data_cfg.get("raw_path", "dataset/raw/news_dataset.csv")
    train_split = data_cfg.get("train_split", 0.70)
    val_split = data_cfg.get("val_split", 0.15)
    test_split = data_cfg.get("test_split", 0.15)
    seed = params.get("training", {}).get("random_seed", 42)

    print("Running DVC Stage: prepare")
    res = prepare_dataset(
        raw_csv_path=raw_path,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        random_state=seed
    )
    print(f"Preparation complete. Train={res['train']}, Val={res['val']}, Test={res['test']}")

if __name__ == "__main__":
    main()
