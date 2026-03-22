#!/usr/bin/env bash
# ─── ObjectVision · One-Shot Setup Script ────────────────────────────────────
# Usage:  bash setup.sh [--gpu]
#
#   --gpu   Force CUDA 12.x PyTorch build (auto-detected if nvidia-smi present)
# ─────────────────────────────────────────────────────────────────────────────

set -e

GPU_MODE=false

for arg in "$@"; do
  case $arg in
    --gpu) GPU_MODE=true ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         ObjectVision  ·  Setup               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Detect environment ─────────────────────────────────────────────────────
echo "[1/6] Detecting Python environment..."
if command -v conda &>/dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
  echo "      ✓ Conda environment detected: $CONDA_DEFAULT_ENV"
  echo "      ✓ Using $(python --version)"
  echo "      (Skipping venv — conda studio detected)"
else
  echo "      ✓ Using $(python3 --version)"
fi

PYTHON=$(command -v python || command -v python3)

# ── 2. Upgrade pip ────────────────────────────────────────────────────────────
echo "[2/6] Upgrading pip..."
$PYTHON -m pip install --quiet --upgrade pip

# ── 3. PyTorch ────────────────────────────────────────────────────────────────
TORCH_CUDA=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "notinstalled")

if [ "$TORCH_CUDA" = "True" ]; then
  echo "[3/6] PyTorch with CUDA already installed — skipping."
elif [ "$GPU_MODE" = true ]; then
  echo "[3/6] Installing PyTorch CUDA 12.1..."
  $PYTHON -m pip install --quiet torch torchvision \
      --index-url https://download.pytorch.org/whl/cu121
else
  if command -v nvidia-smi &>/dev/null; then
    echo "[3/6] NVIDIA GPU detected — installing PyTorch CUDA 12.1..."
    $PYTHON -m pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
  else
    echo "[3/6] Installing PyTorch (CPU)..."
    $PYTHON -m pip install --quiet torch torchvision
  fi
fi

# ── 4. Other requirements ─────────────────────────────────────────────────────
echo "[4/6] Installing project requirements..."
# ultralytics>=8.3.0  →  fixes PyTorch 2.6 weights_only=True error
# fastapi>=0.112.0    →  resolves litserve dependency conflict
$PYTHON -m pip install --quiet \
  "fastapi>=0.112.0" \
  "uvicorn[standard]>=0.29.0" \
  "python-multipart>=0.0.9" \
  "websockets>=12.0" \
  "ultralytics>=8.3.0" \
  "opencv-python-headless>=4.9.0.80" \
  "numpy>=1.26.0" \
  "python-dotenv>=1.0.1"
echo "      ✓ All packages installed"

# ── 5. Pre-download YOLOv8n weights ──────────────────────────────────────────
echo "[5/6] Pre-downloading YOLOv8n model weights..."
$PYTHON - <<'EOF'
import torch
print(f"  PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
from ultralytics import YOLO
for m in ["yolov8n.pt"]:
    print(f"  Downloading {m} ...")
    YOLO(m)
    print(f"  ✓ {m} ready")
EOF

# ── 6. Directories + .env ─────────────────────────────────────────────────────
echo "[6/6] Creating runtime directories..."
mkdir -p Video_storage logs data
echo "      ✓ Video_storage/  logs/  data/"

if [ ! -f ".env" ]; then
  echo "      ⚠  .env not found — please ensure it exists (included in the zip)."
else
  echo "      ✓ .env present"
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                           ║"
echo "║                                              ║"
echo "║  Generate your first API key:                ║"
echo "║    python -m src.backend.auth keygen         ║"
echo "║      --label my-key                          ║"
echo "║                                              ║"
echo "║  Start the server:                           ║"
echo "║    python app.py                             ║"
echo "║                                              ║"
echo "║  Open the UI:                                ║"
echo "║    http://localhost:8000/static/index.html   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""