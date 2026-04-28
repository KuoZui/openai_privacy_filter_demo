# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# git is required for `pip install git+https://...` (opf is installed from GitHub)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights into the image (~3 GB → /root/.opf/privacy_filter/).
# HF_TOKEN is passed via BuildKit secret to avoid baking it into image layers
# while still allowing authenticated downloads (much faster than anonymous).
# Build without a token also works, just slower and subject to HF rate limits.
RUN --mount=type=secret,id=hf_token,required=false \
    sh -c 'if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
           python -c "from opf._common.checkpoint_download import ensure_default_checkpoint; ensure_default_checkpoint()"'

COPY app.py demo.py ./

# Default port for local `docker run`; Railway overrides via $PORT
ENV PORT=8501
EXPOSE 8501

# Shell-form CMD so $PORT expands at container start
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
