import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from backend.app.auth import verify_admin_key
from backend.app.schemas import DatasetUploadResponse, RetrainRequest, RetrainStatusResponse
from backend.app.services.retraining_service import retraining_service
from backend.app.config import settings

router = APIRouter(tags=["Continuous Learning & MLOps Pipeline"])

@router.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a labelled CSV news dataset for continuous learning.
    Validates formatting, required columns (text/title and label), and distribution.
    Staged safely without directly altering production data until retraining is triggered.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a standard CSV file."
        )

    os.makedirs(settings.STAGING_DIR, exist_ok=True)
    temp_filename = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    temp_path = os.path.join(settings.STAGING_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    validation_result = retraining_service.validate_csv(temp_path)

    if not validation_result["is_valid"]:
        # Remove invalid upload
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV Validation Failed: {validation_result['message']}"
        )

    return DatasetUploadResponse(
        filename=file.filename,
        total_records=validation_result["total"],
        real_count=validation_result["real"],
        fake_count=validation_result["fake"],
        is_valid=True,
        validation_message=validation_result["message"],
        sample_records=validation_result["sample"]
    )

@router.post("/retrain", response_model=RetrainStatusResponse)
def trigger_retraining(
    payload: RetrainRequest = RetrainRequest(),
    is_admin: bool = Depends(verify_admin_key)
):
    """
    Protected Admin Endpoint: Trigger automated BERT fine-tuning and candidate evaluation.
    Merges validated datasets, tracks via DVC, logs MLflow run, and applies promotion gate.
    """
    try:
        job_id = retraining_service.start_retraining_async(
            epochs=payload.epochs or settings.DEFAULT_EPOCHS,
            batch_size=payload.batch_size or settings.BATCH_SIZE,
            learning_rate=payload.learning_rate or settings.LEARNING_RATE,
            version_tag=payload.model_version_tag,
            force_promote=payload.force_promote or False
        )
        status_info = retraining_service.get_status()
        return RetrainStatusResponse(
            job_id=job_id,
            status=status_info["status"],
            progress=status_info["progress"],
            current_step=status_info["current_step"],
            message="Retraining initiated successfully in background.",
            logs=status_info["logs"]
        )
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start retraining: {str(e)}")

@router.get("/retrain/status", response_model=RetrainStatusResponse)
def get_retraining_status():
    """
    Check active or latest retraining progress, step description, logs, and evaluation comparison.
    """
    status_info = retraining_service.get_status()
    return RetrainStatusResponse(
        job_id=status_info["job_id"] or "none",
        status=status_info["status"],
        progress=status_info["progress"],
        current_step=status_info["current_step"],
        prior_accuracy=status_info.get("prior_accuracy"),
        new_accuracy=status_info.get("new_accuracy"),
        prior_f1=status_info.get("prior_f1"),
        new_f1=status_info.get("new_f1"),
        promoted=status_info.get("promoted", False),
        message=status_info["message"],
        logs=status_info.get("logs", [])
    )
