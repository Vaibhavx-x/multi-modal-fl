# =============================================================================
# Dockerfile  —  multi-modal-fl  (PAD-UFES-20 + Flower)
#
# Single image used by BOTH the server container and all 5 client containers.
# The CMD / command is overridden per-service in docker-compose.yml.
#
# Build (done automatically by docker compose up --build):
#   docker build -t mmfl .
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
# Official NVIDIA PyTorch image: CUDA 12.1 + cuDNN 8 runtime.
# Ships with torch + torchvision pre-built against the matching CUDA toolkit.
# Switch to a *-devel tag only if you need to compile custom CUDA extensions.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── OS-level dependencies ─────────────────────────────────────────────────────
# Required by Pillow (image loading) and some OpenCV builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Layer order matters: copy requirements.txt BEFORE source files so Docker
# re-uses the pip cache layer on source-only changes (fast rebuilds).
COPY requirements.txt .

# Install all deps from requirements.txt.
# Key packages installed:
#   flwr[simulation]  — Flower FL framework (server + client + strategy)
#   flwr-datasets     — DirichletPartitioner
#   datasets          — HuggingFace in-memory bridge for offline partitioning
#   pandas            — CSV metadata loading + tabular feature encoding
#   scikit-learn      — optional metrics (confusion matrix, per-class accuracy)
#   Pillow            — image loading
#   torchvision       — ResNet-18 pretrained weights + transforms
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download ResNet-18 ImageNet weights ───────────────────────────────────
# torchvision downloads weights to ~/.cache/torch/hub at first use.
# TRANSFORMERS_OFFLINE does NOT block this — we must bake the weights into the
# image so containers can run fully offline at runtime.
RUN python - <<'EOF'
from torchvision.models import resnet18, ResNet18_Weights
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("ResNet-18 weights cached.")
EOF

# ── Application source files ──────────────────────────────────────────────────
# The PAD-UFES-20 archive/ is NOT copied — it is mounted as a volume at
# runtime so the image stays small and the data stays on the host.
COPY dataset.py       .
COPY partitioning.py  .
COPY model.py         .
COPY client.py        .
COPY server.py        .

# ── Network port ──────────────────────────────────────────────────────────────
# Flower gRPC server listens on 8080 inside the container.
# Mapped to the host via docker-compose ports: or -p flag.
EXPOSE 8080

# ── Offline-mode environment variables ───────────────────────────────────────
# Prevents HuggingFace datasets / transformers from making Hub requests.
# Required because the Dirichlet partitioner uses an in-memory HF Dataset.
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ── Default command ───────────────────────────────────────────────────────────
# docker-compose.yml overrides this per service:
#   server    → python server.py
#   client_N  → python client.py --client-id N ...
CMD ["python", "server.py"]
