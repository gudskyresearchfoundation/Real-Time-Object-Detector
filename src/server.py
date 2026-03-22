"""
src/server.py  –  FastAPI: REST + WebSocket
============================================================
Endpoints:
  GET  /                        → health check (public)
  GET  /static/index.html       → UI           (public)
  POST /api/upload              → upload video  [auth + rate-limited]
  GET  /api/storage             → list videos   [auth + rate-limited]
  GET  /api/models              → model list    [auth + rate-limited]
  WS   /ws/inference            → detection stream [auth via ?api_key=]
  GET  /ws/status               → session count [auth]
  GET  /admin/keys              → list API keys [admin]
  POST /admin/keys              → create key    [admin]
  DELETE /admin/keys/{id}       → revoke key    [admin]
  GET  /admin/logs              → tail log file [admin]
"""

import asyncio
import base64
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import (
    Body, Depends, FastAPI, File, HTTPException,
    Query, Request, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.model.object_model import ObjectDetector
from src.websocket_handler import ConnectionManager
from src.backend import (
    RequestLoggingMiddleware,
    log_inference,
    rate_limit_api,
    rate_limit_upload,
    rate_limit_ws,
    rl_headers,
    require_api_key,
    require_admin,
    verify_ws_key,
)
import src.backend.auth as _auth_module

# ─── Configuration ────────────────────────────────────────────────────────────
VIDEO_STORAGE_ROOT = Path(os.getenv("VIDEO_STORAGE_ROOT", "Video_storage"))
MAX_FILE_MB        = int(os.getenv("MAX_FILE_MB", "500"))
YOLO_MODEL         = os.getenv("YOLO_MODEL", "yolov8n.pt")
MAX_STREAM_FPS     = int(os.getenv("MAX_STREAM_FPS", "30"))
MAX_WS_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", "50"))
AUTH_ENABLED       = os.getenv("AUTH_ENABLED", "true").lower() == "true"
LOG_DIR            = Path(os.getenv("LOG_DIR", "logs"))

ALLOWED_MODELS = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolov8l.pt", "yolov8x.pt", "yolov8x6.pt",
}

MODEL_META = [
    {"id": "yolov8n.pt",  "label": "v8n",  "tag": "Nano",   "params": "3.2M",  "map50_95": 37.3},
    {"id": "yolov8s.pt",  "label": "v8s",  "tag": "Small",  "params": "11.2M", "map50_95": 44.9},
    {"id": "yolov8m.pt",  "label": "v8m",  "tag": "Medium", "params": "25.9M", "map50_95": 50.2},
    {"id": "yolov8l.pt",  "label": "v8l",  "tag": "Large",  "params": "43.7M", "map50_95": 52.9},
    {"id": "yolov8x.pt",  "label": "v8x",  "tag": "XLarge", "params": "68.2M", "map50_95": 53.9},
    {"id": "yolov8x6.pt", "label": "v8x6", "tag": "World",  "params": "97.2M", "map50_95": 56.7},
]

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ObjectVision API",
    description="Real-time YOLOv8 object detection · Auth + Rate Limiting + Logging",
    version="1.1.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

# ─── Singletons ───────────────────────────────────────────────────────────────
detector = ObjectDetector(model_path=YOLO_MODEL)
manager  = ConnectionManager(max_connections=MAX_WS_CONNECTIONS)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def today_folder() -> Path:
    folder = VIDEO_STORAGE_ROOT / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def unique_filename(original: str) -> str:
    stem   = Path(original).stem
    suffix = Path(original).suffix
    uid    = uuid.uuid4().hex[:8]
    return f"{stem}_{uid}{suffix}"


# ─── Public endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status":       "ok",
        "service":      "ObjectVision",
        "version":      "1.1.0",
        "model":        detector.model_path,
        "device":       detector.device,
        "auth_enabled": AUTH_ENABLED,
    }


# ─── Protected: Models ────────────────────────────────────────────────────────

@app.get("/api/models", tags=["Models"])
async def list_models(
    _key = Depends(require_api_key),
    _rl  = Depends(rate_limit_api),
):
    return {"models": MODEL_META, "current": detector.model_path}


# ─── Protected: Upload ────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["Storage"])
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    _key  = Depends(require_api_key),
    rl    = Depends(rate_limit_upload),
):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are accepted.")

    data    = await video.read()
    size_mb = len(data) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_MB} MB.",
        )

    dest_dir  = today_folder()
    safe_name = unique_filename(video.filename or "upload.mp4")
    dest_path = dest_dir / safe_name

    with open(dest_path, "wb") as f:
        f.write(data)

    return JSONResponse(
        {
            "status":   "saved",
            "path":     str(dest_path),
            "date":     dest_dir.name,
            "filename": safe_name,
            "size_mb":  round(size_mb, 2),
        },
        headers=rl_headers(rl),
    )


# ─── Protected: Storage list ──────────────────────────────────────────────────

@app.get("/api/storage", tags=["Storage"])
async def list_storage(
    _key = Depends(require_api_key),
    rl   = Depends(rate_limit_api),
):
    if not VIDEO_STORAGE_ROOT.exists():
        return JSONResponse({"dates": []}, headers=rl_headers(rl))

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    result = []
    for date_dir in sorted(VIDEO_STORAGE_ROOT.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        files = [
            {
                "name":    f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "path":    str(f),
            }
            for f in sorted(date_dir.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
        result.append({"date": date_dir.name, "count": len(files), "files": files})

    return JSONResponse(
        {"dates": result, "total_dates": len(result)},
        headers=rl_headers(rl),
    )


# ─── Protected: WS status ─────────────────────────────────────────────────────

@app.get("/ws/status", tags=["WebSocket"])
async def ws_status(
    _key = Depends(require_api_key),
    _rl  = Depends(rate_limit_api),
):
    return manager.summary()


# ─── Admin endpoints ──────────────────────────────────────────────────────────

@app.get("/admin/keys", tags=["Admin"])
async def admin_list_keys(_=Depends(require_admin)):
    return {"keys": _auth_module.list_keys()}


@app.post("/admin/keys", tags=["Admin"])
async def admin_create_key(
    label: str = Body(default="", embed=True),
    _=Depends(require_admin),
):
    plaintext = _auth_module.generate_key(label)
    return {
        "status":  "created",
        "key":     plaintext,
        "label":   label,
        "warning": "Save this key — it will NOT be shown again.",
    }


@app.delete("/admin/keys/{key_id}", tags=["Admin"])
async def admin_revoke_key(key_id: int, _=Depends(require_admin)):
    ok = _auth_module.revoke_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Key id={key_id} not found.")
    return {"status": "revoked", "key_id": key_id}


@app.get("/admin/logs", tags=["Admin"])
async def admin_tail_logs(
    log_type: str = Query(default="access", regex="^(access|inference)$"),
    lines:    int = Query(default=50, ge=1, le=500),
    _=Depends(require_admin),
):
    log_file = LOG_DIR / f"{log_type}.log"
    if not log_file.exists():
        return {"lines": [], "file": str(log_file), "message": "Log file not yet created."}

    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    tail = [line.rstrip() for line in all_lines[-lines:]]
    parsed = []
    for raw in tail:
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            parsed.append({"raw": raw})

    return {"file": str(log_file), "count": len(parsed), "lines": parsed}


# ─── WebSocket Inference ──────────────────────────────────────────────────────

@app.websocket("/ws/inference")
async def websocket_inference(
    ws:      WebSocket,
    api_key: Optional[str] = Query(default=None, alias="api_key"),
):
    # Auth check before accepting
    if not verify_ws_key(api_key):
        await ws.accept()
        await ws.send_text(json.dumps({
            "type":    "error",
            "message": "Unauthorized. Pass ?api_key=ov_... in the WebSocket URL.",
        }))
        await ws.close(code=4001)
        return

    session_id = uuid.uuid4().hex[:8]
    client_ip  = ws.client.host if ws.client else "unknown"

    # Rate limit: one check per connection
    try:
        rate_limit_ws(api_key or client_ip)
    except HTTPException as e:
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": str(e.detail)}))
        await ws.close(code=4029)
        return

    await manager.connect(ws, session_id)

    saved_path: Optional[Path] = None
    session_start    = time.perf_counter()
    total_detections = 0

    try:
        # Step 1: JSON config
        raw_config = await ws.receive_text()
        config     = json.loads(raw_config)

        confidence : float     = float(config.get("confidence", 0.5))
        filter_cls : List[str] = config.get("classes", [])
        filename   : str       = config.get("filename", "upload.mp4")
        model_name : str       = config.get("model", YOLO_MODEL)

        if model_name not in ALLOWED_MODELS:
            model_name = YOLO_MODEL

        if model_name != detector.model_path:
            await ws.send_text(json.dumps({"type": "status", "message": f"Loading {model_name}…"}))
            detector.load_model(model_name)
            await ws.send_text(json.dumps({"type": "status", "message": f"{model_name} ready"}))

        manager.set_status(session_id, "receiving")

        # Step 2: binary video
        video_bytes = await ws.receive_bytes()
        dest_dir    = today_folder()
        safe_name   = unique_filename(filename)
        saved_path  = dest_dir / safe_name
        saved_path.write_bytes(video_bytes)

        await ws.send_text(json.dumps({"type": "status", "message": f"Saved → {saved_path}"}))

        # Step 3: inference
        manager.set_status(session_id, "running")

        cap          = cv2.VideoCapture(str(saved_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay  = 1.0 / min(fps_src, MAX_STREAM_FPS)
        frame_idx    = 0
        loop_start   = asyncio.get_event_loop().time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            manager.increment_frame(session_id)

            detections = detector.detect(
                frame, confidence=confidence,
                filter_classes=filter_cls, annotate=True,
            )
            total_detections += len(detections)

            _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buf).decode()

            elapsed  = asyncio.get_event_loop().time() - loop_start
            live_fps = round(frame_idx / elapsed, 1) if elapsed > 0 else 0
            progress = round(frame_idx / total_frames * 100, 1)

            await ws.send_text(json.dumps({
                "type":       "frame",
                "frame":      frame_idx,
                "fps":        live_fps,
                "progress":   progress,
                "image":      img_b64,
                "detections": [
                    {"cls": d["class"], "conf": round(d["confidence"], 3)}
                    for d in detections
                ],
            }))
            await asyncio.sleep(frame_delay)

        cap.release()
        manager.set_status(session_id, "done")

        await ws.send_text(json.dumps({
            "type":         "done",
            "total_frames": frame_idx,
            "saved_path":   str(saved_path),
        }))

        log_inference(
            session_id=session_id,
            model=detector.model_path,
            confidence=confidence,
            classes_count=len(filter_cls),
            total_frames=frame_idx,
            duration_s=time.perf_counter() - session_start,
            detections_total=total_detections,
            ip=client_ip,
            api_key_id=api_key[:10] if api_key else None,
            saved_path=str(saved_path) if saved_path else "",
        )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
