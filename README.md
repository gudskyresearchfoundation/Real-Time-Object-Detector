# Real-Time-Object-Detector
Real-time object detection using YOLOv8 + FastAPI with live webcam inference — built during Gudsky Research Foundation's 5-Day AI Workshop

# ObjectVision · Real-Time Object Detector

> YOLOv8 + FastAPI + WebSocket · Built for the Gudsky Research Foundation 5-Day AI Workshop

---

## Project Structure

```
object-detection-ai/
├── app.py                          # Uvicorn entrypoint  ← START HERE
├── requirements.txt
├── setup.sh                        # One-shot install + model download
├── .env                            # All configuration (edit as needed)
├── .gitignore
│
├── src/
│   ├── server.py                   # FastAPI: REST endpoints + WebSocket
│   ├── websocket_handler.py        # 50-user cap, per-session state
│   │
│   ├── frontend/
│   │   └── index.html              # Full UI (served at /static/index.html)
│   │
│   ├── backend/                    # Reserved: auth, rate limiting
│   │   └── __init__.py
│   │
│   └── model/
│       ├── __init__.py
│       └── object_model.py         # YOLOv8 wrapper + OpenCV annotation
│
└── Video_storage/                  # Auto-created; organised by date
    └── YYYY-MM-DD/
        └── <video_uid.ext>
```

---

## Quick Start

### 1 — Setup (first time only)

```bash
# CPU (laptop / no GPU)
bash setup.sh

# GPU — NVIDIA CUDA 12.x  (workshop L4 GPU)
bash setup.sh --gpu
```

### 2 — Run

```bash
source .venv/bin/activate
python app.py
```

### 3 — Open the UI

```
http://localhost:8000/static/index.html
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/api/models` | List YOLOv8 model variants |
| POST | `/api/upload` | Upload video → saved to `Video_storage/YYYY-MM-DD/` |
| GET | `/api/storage` | List all stored videos grouped by date |
| WS | `/ws/inference` | Stream annotated frames + detections |
| GET | `/ws/status` | Active WebSocket sessions |

### WebSocket Protocol

**Client → Server (text, JSON config first):**
```json
{
  "type":       "config",
  "session_id": "SID-XXXXX",
  "confidence": 0.5,
  "classes":    ["person", "car", "dog"],
  "filename":   "myvideo.mp4",
  "model":      "yolov8n.pt"
}
```
Then immediately send the **binary video bytes**.

**Server → Client (per frame):**
```json
{
  "type":       "frame",
  "frame":      42,
  "fps":        28.3,
  "progress":   35.0,
  "image":      "<base64 JPEG>",
  "detections": [
    {"cls": "person", "conf": 0.923},
    {"cls": "car",    "conf": 0.871}
  ]
}
```

---

## Supported Models

| Model | Variant | Params | mAP50-95 |
|-------|---------|--------|----------|
| yolov8n.pt | Nano   | 3.2M  | 37.3 |
| yolov8s.pt | Small  | 11.2M | 44.9 |
| yolov8m.pt | Medium | 25.9M | 50.2 |
| yolov8l.pt | Large  | 43.7M | 52.9 |
| yolov8x.pt | XLarge | 68.2M | 53.9 |
| yolov8x6.pt| World  | 97.2M | 56.7 |

Models are auto-downloaded by Ultralytics on first use.

---

## Video Storage

Uploaded videos are automatically organised:

```
Video_storage/
├── 2026-03-25/
│   ├── demo_a1b2c3d4.mp4
│   └── test_e5f6g7h8.avi
└── 2026-03-26/
    └── campus_i9j0k1l2.mp4
```

---

## Prerequisites

- Python 3.10+
- For GPU: NVIDIA driver ≥ 525, CUDA 12.x

---

*Gudsky Research Foundation · Dronacharya Free Learning*
