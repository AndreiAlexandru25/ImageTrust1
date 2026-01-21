# ImageTrust Docker Image
# Multi-stage build for smaller final image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="ImageTrust: AI-Generated Image Detection"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash imagetrust
RUN chown -R imagetrust:imagetrust /app
USER imagetrust

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs /app/reports /app/tmp

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV IMAGETRUST_ENV=production
ENV IMAGETRUST_LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - API server
CMD ["uvicorn", "imagetrust.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
