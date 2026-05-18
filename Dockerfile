# ── Base image ────────────────────────────────────────────────────────────
# Python 3.11 slim — keeps image small
FROM python:3.11-slim

# HF Spaces runs as non-root user (uid=1000)
RUN useradd -m -u 1000 appuser

# ── System deps ────────────────────────────────────────────────────────────
# libgl1 + libglib2.0 needed by OpenCV (used by Albumentations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# CPU-only torch — much smaller image for HF free tier
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application ───────────────────────────────────────────────────────
COPY app.py .
COPY static/ ./static/

# models/ directory — place .pth files here before build OR mount at runtime
RUN mkdir -p models
COPY models/ ./models/

# ── Permissions ────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────
# HF Spaces expects port 7860
EXPOSE 7860

# ── Launch ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
