import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.database import engine, Base
from backend.app.api.predict import router as predict_router
from backend.app.api.metrics import router as metrics_router
from backend.app.api.continuous_learning import router as cl_router
from backend.app.api.monitoring import router as monitoring_router
from backend.app.services.model_service import model_service

# Auto-create SQLite database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Full-Stack Production System for Fake News Detection with Continuous Learning, BERT, MLflow, DVC, and Evidently AI Monitoring."
)

# Enable CORS for React Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(predict_router, prefix=settings.API_V1_PREFIX)
app.include_router(metrics_router, prefix=settings.API_V1_PREFIX)
app.include_router(cl_router, prefix=settings.API_V1_PREFIX)
app.include_router(monitoring_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
def health_check():
    """API Health Check and System Architecture Overview"""
    model_info = model_service.get_info()
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_model_version": model_info["active_version"],
        "model_status": model_info["status"],
        "device": model_info["device"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
