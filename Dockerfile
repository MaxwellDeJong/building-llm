# Requires PyTorch >= 2.10 for torch.optim.Muon.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Minimal system packages (git is needed by the HuggingFace datasets library
# to resolve dataset revisions from Hub).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# torch/torchvision are already present in the base image with CUDA support;
# reinstalling via pip would replace them with CPU-only wheels, so we filter
# them out and rely entirely on what the base image provides.
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|torchaudio)' requirements.txt \
    | pip install --no-cache-dir --break-system-packages -r /dev/stdin

COPY src/ ./src/
COPY train.py .
COPY evaluate.py .

# src/ is the Python package root for all internal imports.
ENV PYTHONPATH=/app/src

# HuggingFace Hub metadata and lock files (mapped to a named volume at
# runtime).  Parquet shards and tokenised caches live in /app/data, which is
# mounted separately via the climbmix_data volume.
ENV HF_HOME=/data/huggingface

# Unbuffered output so log lines appear immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
