from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article content to analyze")

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="'REAL' or 'FAKE'")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    model_version: str = Field(..., description="Active production model version")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    disclaimer: str = Field(
        default="Notice: This classification is an AI-generated assessment produced by a fine-tuned BERT transformer model and does not constitute certified verification.",
        description="Legal and analytical disclaimer"
    )

class PredictionHistoryItem(BaseModel):
    id: int
    text: str
    prediction: str
    confidence: float
    model_version: str
    latency_ms: float
    created_at: datetime

    class Config:
        from_attributes = True

class HistoryResponse(BaseModel):
    total: int
    items: List[PredictionHistoryItem]

class DatasetUploadResponse(BaseModel):
    filename: str
    total_records: int
    real_count: int
    fake_count: int
    is_valid: bool
    validation_message: str
    sample_records: List[Dict[str, Any]]

class RetrainRequest(BaseModel):
    epochs: Optional[int] = Field(default=3, ge=1, le=20)
    batch_size: Optional[int] = Field(default=8, ge=2, le=64)
    learning_rate: Optional[float] = Field(default=2e-5, gt=0)
    model_version_tag: Optional[str] = None
    force_promote: Optional[bool] = False

class RetrainStatusResponse(BaseModel):
    job_id: str
    status: str # QUEUED, RUNNING, COMPLETED, FAILED, REJECTED
    progress: int
    current_step: str
    prior_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    prior_f1: Optional[float] = None
    new_f1: Optional[float] = None
    promoted: bool = False
    message: str
    logs: List[str] = []

class ModelInfoResponse(BaseModel):
    model_name: str
    active_version: str
    base_model: str
    device: str
    last_trained: str
    accuracy: float
    f1_score: float
    dataset_version: str
    status: str

class MetricsResponse(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Dict[str, Any]
    training_loss: List[float]
    validation_loss: List[float]
    model_comparisons: List[Dict[str, Any]]

class MonitoringResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    status: str
    dataset_version: str
    current_model_version: str
    last_monitoring_run: str
    reference_sample_count: int
    current_sample_count: int
    column_drift_details: Dict[str, Any]
    report_available: bool
