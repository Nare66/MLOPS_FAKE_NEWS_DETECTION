import os
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
ID_TO_LABEL = {0: "FAKE", 1: "REAL"}
LABEL_TO_ID = {"FAKE": 0, "REAL": 1}

class FakeNewsModelWrapper:
    """
    Production inference wrapper for BERT-based Fake News classification.
    Supports PyTorch sequence classification with full Hugging Face transformer architecture.
    """
    def __init__(self, model_path_or_name: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path_or_name or "models/production"
        self.model = None
        self.tokenizer = None
        self.metadata = {}
        self.load_model(self.model_path)

    def load_model(self, path_or_name: str):
        print(f"Loading Fake News Classifier from: {path_or_name} (Device: {self.device})")
        # Check if local model directory exists with saved weights
        if os.path.exists(path_or_name) and os.path.exists(os.path.join(path_or_name, "config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(path_or_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path_or_name,
                num_labels=2,
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID
            )
        else:
            # Fall back to base pretrained model (e.g. distilbert-base-uncased)
            base_model = DEFAULT_MODEL_NAME
            print(f"Local weights not found at {path_or_name}, loading base: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=2,
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID
            )

        self.model.to(self.device)
        self.model.eval()

        # Load metadata if present
        meta_file = os.path.join(path_or_name, "metadata.json") if os.path.exists(path_or_name) else None
        if meta_file and os.path.exists(meta_file):
            import json
            try:
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {
                "model_version": "v1.0.0-initial",
                "base_model": DEFAULT_MODEL_NAME,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }

    def predict(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        Run inference on input news content.
        Returns predicted label, confidence score, raw probabilities, and latency.
        """
        start_time = time.perf_counter()
        
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        fake_prob = float(probs[0])
        real_prob = float(probs[1])

        pred_id = int(probs.argmax())
        label = ID_TO_LABEL[pred_id]
        confidence = float(probs[pred_id])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "FAKE": round(fake_prob, 4),
                "REAL": round(real_prob, 4)
            },
            "latency_ms": latency_ms,
            "model_version": self.metadata.get("model_version", "v1.0.0")
        }
