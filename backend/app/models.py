from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
from backend.app.database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    prediction = Column(String(10), nullable=False) # REAL or FAKE
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=False, default="v1.0.0")
    latency_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(20), nullable=False, default="QUEUED") # QUEUED, RUNNING, COMPLETED, FAILED
    model_version = Column(String(50), nullable=False)
    prior_accuracy = Column(Float, nullable=True)
    new_accuracy = Column(Float, nullable=True)
    prior_f1 = Column(Float, nullable=True)
    new_f1 = Column(Float, nullable=True)
    promoted = Column(Boolean, default=False)
    dataset_size = Column(Integer, default=0)
    logs = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class MonitoringLog(Base):
    __tablename__ = "monitoring_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(Integer, nullable=False)
    run_date = Column(String(50), nullable=False)
    drift_detected = Column(Boolean, default=False)
    drift_score = Column(Float, default=0.0)
    report_path = Column(String(255), nullable=True)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
