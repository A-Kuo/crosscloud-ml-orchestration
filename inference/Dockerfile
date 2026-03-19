# ---------------------------------------------------------------------------
# CrossCloud ML Inference Server
# Deployable to GCP Cloud Run (SERVE_TARGET=absa) or
#              AWS SageMaker (SERVE_TARGET=hallucination) or
#              local router mode (SERVE_TARGET=router, default)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY router/ /app/router/
COPY inference/ /app/inference/

# Pre-download model weights at build time to avoid cold-start latency
# Override PROBE_MODEL / SERVE_TARGET build args to bake different models.
ARG PROBE_MODEL=distilbert-base-uncased
ARG SERVE_TARGET=router
ENV SERVE_TARGET=${SERVE_TARGET}
ENV PROBE_MODEL=${PROBE_MODEL}

RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('${PROBE_MODEL}'); \
AutoModel.from_pretrained('${PROBE_MODEL}', output_attentions=True); \
print('Probe model cached.')"

# Port expected by Cloud Run and SageMaker
EXPOSE 8080

# Health check (Cloud Run requires /health to return 200)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
