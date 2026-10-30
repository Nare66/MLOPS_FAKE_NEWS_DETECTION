import os
import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model_version" in data
    assert data["service"] == settings.APP_NAME

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "active_version" in data
    assert "accuracy" in data
    assert "f1_score" in data
    assert data["status"] == "OPERATIONAL"

def test_predict_real_news():
    payload = {
        "text": "NASA's James Webb Space Telescope has captured deep-field observations revealing galaxies that formed just 300 million years after the Big Bang."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["REAL", "FAKE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "latency_ms" in data
    assert "disclaimer" in data

def test_predict_fake_news():
    payload = {
        "text": "Secret Antarctic crystalline pyramid proven to contain alien quantum generator hidden by world governments."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["REAL", "FAKE"]
    assert "confidence" in data

def test_predict_short_text_validation():
    payload = {"text": "Too short"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic min_length validation

def test_prediction_history():
    response = client.get("/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 1

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "f1" in data
    assert "confusion_matrix" in data
    assert "training_loss" in data

def test_upload_dataset_valid():
    csv_content = """title,text,label
"ESA Satellite Deploys Successfully","European Space Agency deployed synthetic aperture radar.",REAL
"Secret Moon Aliens Discovered","NASA found alien cities on moon.",FAKE
"""
    file_obj = io.BytesIO(csv_content.encode("utf-8"))
    response = client.post(
        "/upload-dataset",
        files={"file": ("test_upload.csv", file_obj, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["total_records"] == 2
    assert data["real_count"] == 1
    assert data["fake_count"] == 1

def test_upload_dataset_invalid_format():
    invalid_content = "some text without headers or labels"
    file_obj = io.BytesIO(invalid_content.encode("utf-8"))
    response = client.post(
        "/upload-dataset",
        files={"file": ("bad_format.csv", file_obj, "text/csv")}
    )
    assert response.status_code == 422

def test_retrain_unauthorized():
    response = client.post("/retrain", json={"epochs": 1})
    assert response.status_code == 401 # Admin auth required

def test_retrain_authorized_status():
    headers = {"X-Admin-Key": settings.ADMIN_API_KEY}
    response = client.get("/retrain/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "progress" in data

def test_monitoring_endpoint():
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "drift_score" in data
    assert "status" in data
