FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy codebase
COPY backend /app/backend
COPY ml /app/ml
COPY dataset /app/dataset
COPY models /app/models
COPY monitoring /app/monitoring
COPY params.yaml /app/params.yaml
COPY dvc.yaml /app/dvc.yaml
COPY .dvc /app/.dvc

# Expose FastAPI port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV DVC_NO_ANALYTICS=1

# Run FastAPI backend via Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
