FROM python:3.11-slim

# Install system deps for Pillow and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller image — ~800MB vs 3GB with CUDA)
# and ONNX Runtime; no GPU needed on Railway
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    "pydantic>=2.5.0" \
    "onnxruntime>=1.17.0" \
    Pillow \
    numpy \
    httpx \
    python-dotenv

# Copy application code
COPY api/        ./api/
COPY models/     ./models/
COPY train/      ./train/
COPY data/splits/ ./data/splits/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
