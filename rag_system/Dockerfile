# ============================================================
# Dockerfile - RAG知识库系统镜像（带Web UI 和 用户认证）
# ============================================================

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /app/RAG_files /app/chroma_db /app/models /app/eval_results /app/data

COPY RAG_files/ /app/RAG_files/
COPY models/ /app/models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
