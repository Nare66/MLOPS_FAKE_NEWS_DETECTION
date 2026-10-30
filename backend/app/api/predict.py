import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import PredictionHistory
from backend.app.schemas import PredictRequest, PredictResponse, HistoryResponse, PredictionHistoryItem, ModelInfoResponse
from backend.app.services.model_service import model_service

router = APIRouter(tags=["Prediction & Model"])

@router.post("/predict", response_model=PredictResponse)
def predict_news(payload: PredictRequest, db: Session = Depends(get_db)):
    """
    Classify news content as REAL or FAKE using the active fine-tuned BERT model.
    Stores the prediction persistently in SQLite for auditing and monitoring.
    """
    text = payload.text.strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Input text must be at least 10 characters.")

    try:
        result = model_service.predict(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Store persistently in SQLite
    history_entry = PredictionHistory(
        text=text,
        prediction=result["label"],
        confidence=result["confidence"],
        probabilities_json=json.dumps(result["probabilities"]),
        model_version=result.get("model_version", "v1.0.0"),
        latency_ms=result.get("latency_ms", 0.0)
    )
    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)

    return PredictResponse(
        prediction=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=result.get("model_version", "v1.0.0"),
        latency_ms=result.get("latency_ms", 0.0)
    )

@router.get("/history", response_model=HistoryResponse)
def get_prediction_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Retrieve stored prediction history with timestamps and confidence scores.
    """
    total = db.query(PredictionHistory).count()
    records = db.query(PredictionHistory).order_by(PredictionHistory.id.desc()).offset(offset).limit(limit).all()
    
    items = []
    for r in records:
        items.append(PredictionHistoryItem(
            id=r.id,
            text=r.text,
            prediction=r.prediction,
            confidence=round(r.confidence, 4),
            model_version=r.model_version,
            latency_ms=round(r.latency_ms, 2),
            created_at=r.created_at
        ))

    return HistoryResponse(total=total, items=items)

@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Return active production BERT model architecture, version, device, and evaluation performance.
    """
    return model_service.get_info()
