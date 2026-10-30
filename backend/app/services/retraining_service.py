import os
import shutil
import uuid
import time
import json
import threading
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from ml.data_processor import prepare_dataset, clean_text, normalize_label
from ml.train import train_bert_model
from ml.evaluate import compute_classification_metrics, save_metrics
from ml.drift_monitor import run_evidently_drift_monitoring
from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import RetrainingJob
from backend.app.services.model_service import model_service

class RetrainingService:
    _instance: Optional['RetrainingService'] = None
    _active_job: Dict[str, Any] = {
        "job_id": None,
        "status": "IDLE", # IDLE, QUEUED, RUNNING, COMPLETED, FAILED
        "progress": 0,
        "current_step": "System ready",
        "prior_accuracy": None,
        "new_accuracy": None,
        "prior_f1": None,
        "new_f1": None,
        "promoted": False,
        "message": "No active retraining job.",
        "logs": []
    }

    @classmethod
    def get_instance(cls) -> 'RetrainingService':
        if cls._instance is None:
            cls._instance = RetrainingService()
        return cls._instance

    def get_status(self) -> Dict[str, Any]:
        return self._active_job

    def log(self, message: str):
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp_str}] {message}"
        print(f"[RetrainingService] {entry}")
        self._active_job["logs"].append(entry)
        if len(self._active_job["logs"]) > 200:
            self._active_job["logs"].pop(0)

    def validate_csv(self, file_path: str) -> Dict[str, Any]:
        """Validate uploaded news CSV format and contents."""
        try:
            df = pd.read_csv(file_path)
            cols = [c.lower() for c in df.columns]
            df.columns = cols

            if "label" not in df.columns:
                return {"is_valid": False, "message": "Missing required 'label' column.", "total": 0, "real": 0, "fake": 0, "sample": []}

            text_col = "text" if "text" in df.columns else ("title" if "title" in df.columns else None)
            if not text_col:
                return {"is_valid": False, "message": "Missing required 'text' or 'title' column.", "total": 0, "real": 0, "fake": 0, "sample": []}

            # Validate rows
            valid_labels = 0
            real_count = 0
            fake_count = 0
            valid_rows = []

            for _, row in df.iterrows():
                try:
                    lbl = normalize_label(row["label"])
                    txt = str(row[text_col]).strip()
                    if len(txt) > 10:
                        valid_labels += 1
                        if lbl == 1:
                            real_count += 1
                        else:
                            fake_count += 1
                        valid_rows.append({"text": txt[:120] + "...", "label": "REAL" if lbl == 1 else "FAKE"})
                except Exception:
                    continue

            if valid_labels == 0:
                return {"is_valid": False, "message": "No valid rows with text length > 10 found.", "total": 0, "real": 0, "fake": 0, "sample": []}

            return {
                "is_valid": True,
                "message": f"Successfully validated {valid_labels} news records ({real_count} REAL, {fake_count} FAKE).",
                "total": valid_labels,
                "real": real_count,
                "fake": fake_count,
                "sample": valid_rows[:5]
            }
        except Exception as e:
            return {"is_valid": False, "message": f"Error parsing CSV: {str(e)}", "total": 0, "real": 0, "fake": 0, "sample": []}

    def start_retraining_async(
        self,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        version_tag: Optional[str] = None,
        force_promote: bool = False
    ) -> str:
        if self._active_job["status"] == "RUNNING":
            raise RuntimeError("A retraining job is already in progress.")

        job_id = str(uuid.uuid4())
        new_version = version_tag or f"v1.{int(time.time()) % 1000}.0"

        self._active_job = {
            "job_id": job_id,
            "status": "QUEUED",
            "progress": 5,
            "current_step": "Retraining job queued",
            "prior_accuracy": None,
            "new_accuracy": None,
            "prior_f1": None,
            "new_f1": None,
            "promoted": False,
            "message": "Retraining job starting...",
            "logs": []
        }

        thread = threading.Thread(
            target=self._run_retraining_pipeline,
            args=(job_id, epochs, batch_size, learning_rate, new_version, force_promote),
            daemon=True
        )
        thread.start()
        return job_id

    def _run_retraining_pipeline(
        self,
        job_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        version_tag: str,
        force_promote: bool
    ):
        db = SessionLocal()
        job_record = RetrainingJob(
            job_id=job_id,
            status="RUNNING",
            model_version=version_tag,
            created_at=datetime.utcnow()
        )
        db.add(job_record)
        db.commit()

        self._active_job["status"] = "RUNNING"
        self._active_job["progress"] = 10
        self.log(f"Starting Continuous Learning Retraining Pipeline [Job ID: {job_id}]")

        try:
            # Step 1: Merge any staging uploaded datasets
            self._active_job["current_step"] = "Consolidating and deduplicating training data"
            self._active_job["progress"] = 15
            self._merge_staged_datasets()

            # Step 2: Track updated dataset with DVC
            self._active_job["current_step"] = "Versioning dataset via DVC"
            self._active_job["progress"] = 25
            self.log("Versioning updated dataset with DVC...")
            try:
                from dvc.repo import Repo
                repo = Repo('.')
                repo.add('dataset/raw/news_dataset.csv')
                self.log("DVC dataset manifest updated successfully.")
            except Exception as e:
                self.log(f"Notice: DVC sync check: {e}")

            # Step 3: Run Data Preprocessor
            self._active_job["current_step"] = "Preprocessing text and creating stratified splits"
            self._active_job["progress"] = 35
            prep_res = prepare_dataset(
                raw_csv_path=settings.DATASET_RAW_PATH,
                output_dir=settings.PROCESSED_DIR,
                reference_dir=settings.REFERENCE_DIR
            )
            self.log(f"Data splits ready: Total={prep_res['total_records']}, Real={prep_res['real_count']}, Fake={prep_res['fake_count']}")

            # Step 4: Fine-tune BERT candidate model
            self._active_job["current_step"] = f"Fine-tuning BERT transformer ({epochs} epochs, lr={learning_rate})"
            self._active_job["progress"] = 50
            candidate_dir = os.path.join(settings.CANDIDATE_MODEL_DIR, "model")
            train_res = train_bert_model(
                train_csv=os.path.join(settings.PROCESSED_DIR, "train.csv"),
                val_csv=os.path.join(settings.PROCESSED_DIR, "val.csv"),
                output_dir=candidate_dir,
                base_model_name=settings.DEFAULT_BASE_MODEL,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                model_version=version_tag
            )

            # Step 5: Evaluate candidate model against fixed test set
            self._active_job["current_step"] = "Evaluating candidate model on test dataset"
            self._active_job["progress"] = 75
            test_metrics = self._evaluate_on_test(candidate_dir)
            candidate_acc = test_metrics["accuracy"]
            candidate_f1 = test_metrics["f1"]

            # Load prior production metrics for comparison
            prior_acc, prior_f1 = self._get_production_metrics()
            self._active_job["prior_accuracy"] = prior_acc
            self._active_job["new_accuracy"] = candidate_acc
            self._active_job["prior_f1"] = prior_f1
            self._active_job["new_f1"] = candidate_f1

            self.log(f"Prior Production Model: Accuracy={prior_acc}, F1={prior_f1}")
            self.log(f"Candidate Model ({version_tag}): Accuracy={candidate_acc}, F1={candidate_f1}")

            # Step 6: Promotion Gate Evaluation
            self._active_job["current_step"] = "Evaluating promotion criteria"
            self._active_job["progress"] = 85
            
            # Gating Logic:
            # 1. Candidate accuracy must not regress more than MAX_ACCURACY_REGRESSION
            # 2. Candidate accuracy must be >= MIN_ACCURACY_THRESHOLD
            # 3. Candidate F1 must be >= MIN_F1_THRESHOLD
            regressed = (prior_acc - candidate_acc) > settings.MAX_ACCURACY_REGRESSION
            meets_threshold = (candidate_acc >= settings.MIN_ACCURACY_THRESHOLD) and (candidate_f1 >= settings.MIN_F1_THRESHOLD)
            
            should_promote = (not regressed and meets_threshold) or force_promote

            if should_promote:
                self.log(f"PROMOTION CRITERIA PASSED! Promoting candidate {version_tag} to Production.")
                self._promote_candidate_to_production(candidate_dir, test_metrics, version_tag)
                model_service.reload()
                self._active_job["promoted"] = True
                self._active_job["status"] = "COMPLETED"
                self._active_job["message"] = f"Candidate model {version_tag} successfully promoted to Production!"
            else:
                reason = "Performance regression observed" if regressed else "Did not meet minimum accuracy/F1 threshold"
                self.log(f"PROMOTION REJECTED: {reason}. Retaining current production model.")
                self._active_job["promoted"] = False
                self._active_job["status"] = "COMPLETED"
                self._active_job["message"] = f"Model {version_tag} retained in candidate registry. Existing production model kept ({reason})."

            # Step 7: Run Evidently AI monitoring refresh
            self._active_job["current_step"] = "Updating Evidently AI monitoring drift reports"
            self._active_job["progress"] = 95
            try:
                run_evidently_drift_monitoring(
                    reference_csv=os.path.join(settings.REFERENCE_DIR, "reference_data.csv")
                )
                self.log("Evidently AI drift monitoring report refreshed.")
            except Exception as e:
                self.log(f"Notice on monitoring refresh: {e}")

            self._active_job["progress"] = 100
            self._active_job["current_step"] = "Retraining workflow completed"

            # Update DB
            job_record.status = self._active_job["status"]
            job_record.prior_accuracy = prior_acc
            job_record.new_accuracy = candidate_acc
            job_record.prior_f1 = prior_f1
            job_record.new_f1 = candidate_f1
            job_record.promoted = self._active_job["promoted"]
            job_record.logs = "\n".join(self._active_job["logs"])
            job_record.completed_at = datetime.utcnow()
            db.commit()

        except Exception as e:
            import traceback
            err_msg = f"Retraining failed: {str(e)}"
            self.log(err_msg)
            self.log(traceback.format_exc())
            self._active_job["status"] = "FAILED"
            self._active_job["message"] = err_msg
            self._active_job["current_step"] = "Pipeline failed"
            job_record.status = "FAILED"
            job_record.logs = "\n".join(self._active_job["logs"])
            job_record.completed_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def _merge_staged_datasets(self):
        """Append validated uploaded staging files into main dataset and remove from staging."""
        os.makedirs(settings.STAGING_DIR, exist_ok=True)
        staged_files = [f for f in os.listdir(settings.STAGING_DIR) if f.endswith(".csv")]
        if not staged_files:
            self.log("No new staged files found. Retraining on current master dataset.")
            return

        master_df = pd.read_csv(settings.DATASET_RAW_PATH)
        initial_count = len(master_df)

        for filename in staged_files:
            file_path = os.path.join(settings.STAGING_DIR, filename)
            try:
                staged_df = pd.read_csv(file_path)
                staged_df.columns = [c.lower() for c in staged_df.columns]
                
                # Standardize columns
                if "text" in staged_df.columns:
                    staged_df["text"] = staged_df["text"].apply(clean_text)
                if "title" in staged_df.columns:
                    staged_df["title"] = staged_df["title"].apply(clean_text)
                staged_df["label"] = staged_df["label"].apply(normalize_label)

                master_df = pd.concat([master_df, staged_df], ignore_index=True)
                # Deduplicate by text/title
                subset_col = ["text"] if "text" in master_df.columns else ["title"]
                master_df.drop_duplicates(subset=subset_col, inplace=True)
                os.remove(file_path)
                self.log(f"Merged staged file: {filename}")
            except Exception as e:
                self.log(f"Warning: Failed to merge staged file {filename}: {e}")

        master_df.to_csv(settings.DATASET_RAW_PATH, index=False)
        self.log(f"Dataset updated: {initial_count} -> {len(master_df)} records.")

    def _get_production_metrics(self) -> (float, float):
        metrics_file = os.path.join(settings.PRODUCTION_MODEL_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    return float(data.get("accuracy", 0.78)), float(data.get("f1", 0.78))
            except Exception:
                pass
        return 0.75, 0.75

    def _evaluate_on_test(self, candidate_dir: str) -> Dict[str, Any]:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        test_csv = os.path.join(settings.PROCESSED_DIR, "test.csv")
        df = pd.read_csv(test_csv)

        tokenizer = AutoTokenizer.from_pretrained(candidate_dir)
        model = AutoModelForSequenceClassification.from_pretrained(candidate_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        y_true = df["label"].tolist()
        y_pred = []

        for text in df["content"]:
            inputs = tokenizer(
                text,
                max_length=settings.MAX_SEQ_LENGTH,
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
        return metrics

    def _promote_candidate_to_production(self, candidate_dir: str, metrics: Dict[str, Any], version_tag: str):
        os.makedirs(settings.PRODUCTION_MODEL_DIR, exist_ok=True)
        # Copy model files
        for item in os.listdir(candidate_dir):
            src = os.path.join(candidate_dir, item)
            dst = os.path.join(settings.PRODUCTION_MODEL_DIR, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        # Update metadata
        metadata = {
            "model_version": version_tag,
            "base_model": settings.DEFAULT_BASE_MODEL,
            "promoted_at": datetime.utcnow().isoformat() + "Z",
            "production_test_metrics": metrics
        }
        with open(os.path.join(settings.PRODUCTION_MODEL_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        save_metrics(metrics, os.path.join(settings.PRODUCTION_MODEL_DIR, "metrics.json"))

retraining_service = RetrainingService.get_instance()
