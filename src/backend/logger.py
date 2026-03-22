"""
src/backend/logger.py  –  Structured Request + Inference Logging
============================================================
Produces two log streams:

  logs/access.log    – one line per HTTP request  (JSON)
  logs/inference.log – one line per WS session    (JSON)

Both rotate daily (TimedRotatingFileHandler).
Console output mirrors the log level set in .env.

Log record shape (access):
  {ts, level, type:"request", method, path, status_code,
   duration_ms, ip, api_key_id, bytes_sent}

Log record shape (inference):
  {ts, level, type:"inference", session_id, model, confidence,
   classes_count, total_frames, duration_s, detections_total,
   ip, api_key_id, saved_path}

Usage:
  from src.backend.logger import access_logger, inference_logger, RequestLoggingMiddleware

  # In server.py:
  app.add_middleware(RequestLoggingMiddleware)

  # Manual inference log:
  inference_logger.info(json.dumps({...}))
"""

from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ─── Config ───────────────────────────────────────────────────────────────────
LOG_DIR      = Path(os.getenv("LOG_DIR", "logs"))
LOG_LEVEL    = os.getenv("LOG_LEVEL", "info").upper()
LOG_ENABLED  = os.getenv("LOG_ENABLED", "true").lower() == "true"


# ─── Setup ────────────────────────────────────────────────────────────────────

def _make_logger(name: str, filename: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter("%(message)s")   # raw JSON lines

    # ── Rotating file handler (daily, keep 30 days) ───────────────────────────
    fh = TimedRotatingFileHandler(
        LOG_DIR / filename,
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    return logger


access_logger    = _make_logger("objectvision.access",    "access.log")
inference_logger = _make_logger("objectvision.inference", "inference.log")


# ─── Request Logging Middleware ───────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request as a JSON line to access_logger.
    Skips /static/* and /docs* (too noisy).
    """

    SKIP_PREFIXES = ("/static/", "/docs", "/openapi", "/redoc")

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip noisy paths
        if any(path.startswith(p) for p in self.SKIP_PREFIXES):
            return await call_next(request)

        start   = time.perf_counter()
        response: Optional[Response] = None

        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            status_code = response.status_code if response else 500

            # Best-effort IP
            ip = (
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or (request.client.host if request.client else "unknown")
            )

            record = {
                "ts":          _now(),
                "level":       "INFO" if status_code < 400 else "WARNING",
                "type":        "request",
                "method":      request.method,
                "path":        path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "ip":          ip,
                "api_key_id":  request.headers.get("X-API-Key", "")[:10] or None,
            }

            if LOG_ENABLED:
                if status_code >= 500:
                    access_logger.error(json.dumps(record))
                elif status_code >= 400:
                    access_logger.warning(json.dumps(record))
                else:
                    access_logger.info(json.dumps(record))


# ─── Inference session logger ─────────────────────────────────────────────────

def log_inference(
    session_id:       str,
    model:            str,
    confidence:       float,
    classes_count:    int,
    total_frames:     int,
    duration_s:       float,
    detections_total: int,
    ip:               str = "unknown",
    api_key_id:       Optional[str] = None,
    saved_path:       str = "",
) -> None:
    """Call once at the end of a WebSocket inference session."""
    if not LOG_ENABLED:
        return

    record = {
        "ts":               _now(),
        "level":            "INFO",
        "type":             "inference",
        "session_id":       session_id,
        "model":            model,
        "confidence":       confidence,
        "classes_count":    classes_count,
        "total_frames":     total_frames,
        "duration_s":       round(duration_s, 2),
        "fps_avg":          round(total_frames / duration_s, 1) if duration_s > 0 else 0,
        "detections_total": detections_total,
        "ip":               ip,
        "api_key_id":       api_key_id,
        "saved_path":       saved_path,
    }
    inference_logger.info(json.dumps(record))


# ─── Utility ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
