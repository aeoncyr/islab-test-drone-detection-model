# ============================================================
# Dockerfile — VanillaDroneDetector Training Container
# ============================================================
# Base: PyTorch 2.2.0 with CUDA 11.8 + cuDNN 8 (runtime)
# Build: docker compose build
# Run  : docker compose up
# ============================================================

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

LABEL maintainer="islab-drone-detection"
LABEL description="Vanilla from-scratch drone detector training environment"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies first (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command: show usage
CMD ["python", "train.py", "--help"]
