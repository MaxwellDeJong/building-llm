# Requires PyTorch >= 2.10 for torch.optim.Muon.
# Update the tag below when a newer pytorch/pytorch image is available.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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
    | pip install --no-cache-dir -r /dev/stdin

# Install the project package so that `from components import ...` etc. work.
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

COPY train.py .

# src/ is the Python package root for all internal imports.
ENV PYTHONPATH=/app/src

# HuggingFace datasets and model cache (mapped to a named volume at runtime).
ENV HF_HOME=/data/huggingface

# Unbuffered output so log lines appear immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
