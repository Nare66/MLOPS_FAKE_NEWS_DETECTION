import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "Fake News Detection & Continuous Learning Platform"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = ""
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "admin-mlops-secret-2026")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PRODUCTION_MODEL_DIR: str = os.path.join(BASE_DIR, "models", "production")
    CANDIDATE_MODEL_DIR: str = os.path.join(BASE_DIR, "models", "candidate")
    DATASET_RAW_PATH: str = os.path.join(BASE_DIR, "dataset", "raw", "news_dataset.csv")
    STAGING_DIR: str = os.path.join(BASE_DIR, "dataset", "raw", "upload_staging")
    PROCESSED_DIR: str = os.path.join(BASE_DIR, "dataset", "processed")
    REFERENCE_DIR: str = os.path.join(BASE_DIR, "dataset", "reference")
    MONITORING_DIR: str = os.path.join(BASE_DIR, "monitoring")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "monitoring", "reports")
    MLRUNS_DIR: str = os.path.join(BASE_DIR, "mlruns")
    
    # Evaluation gating criteria for continuous learning promotion
    MIN_ACCURACY_THRESHOLD: float = 0.75
    MIN_F1_THRESHOLD: float = 0.75
    MAX_ACCURACY_REGRESSION: float = 0.02
    
    # Model defaults
    DEFAULT_BASE_MODEL: str = "distilbert-base-uncased"
    MAX_SEQ_LENGTH: int = 128
    BATCH_SIZE: int = 8
    DEFAULT_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5

settings = Settings()
