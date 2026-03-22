"""
src/backend/rate_limiter.py  –  Rate Limiting (per IP + per API key)
============================================================
Strategy: sliding-window counter stored in-process (dict).
For multi-worker deployments swap the store for Redis.

Rules (all configurable via .env):
  RATE_LIMIT_IP_RPM      – max requests/min per IP          (default 60)
  RATE_LIMIT_IP_RPH      – max requests/hour per IP         (default 500)
  RATE_LIMIT_KEY_RPM     – max requests/min per API key     (default 120)
  RATE_LIMIT_KEY_RPH     – max requests/hour per API key    (default 2000)
  RATE_LIMIT_UPLOAD_RPH  – max uploads/hour per identity    (default 20)
  RATE_LIMIT_ENABLED     – master on/off switch             (default true)

Usage (FastAPI dependency):
    @app.post("/api/upload")
    async def upload(request: Request, _=Depends(rate_limit_upload)):
        ...

    @app.get("/api/storage")
    async def storage(request: Request, _=Depends(rate_limit_api)):
        ...
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, Header, HTTPException, Query, Request, status

# ─── Config ───────────────────────────────────────────────────────────────────
ENABLED         = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
IP_RPM          = int(os.getenv("RATE_LIMIT_IP_RPM",     "60"))
IP_RPH          = int(os.getenv("RATE_LIMIT_IP_RPH",     "500"))
KEY_RPM         = int(os.getenv("RATE_LIMIT_KEY_RPM",    "120"))
KEY_RPH         = int(os.getenv("RATE_LIMIT_KEY_RPH",    "2000"))
UPLOAD_RPH      = int(os.getenv("RATE_LIMIT_UPLOAD_RPH", "20"))

_WINDOW_MINUTE  = 60
_WINDOW_HOUR    = 3600


# ─── In-process sliding-window store ─────────────────────────────────────────
# Structure:  _store[identity][window_key] = deque of unix timestamps

_store: Dict[str, Dict[str, Deque[float]]] = defaultdict(lambda: defaultdict(deque))


def _check(identity: str, window_key: str, window_secs: int, limit: int) -> int:
    """
    Sliding-window rate check.

    Returns the number of remaining requests in this window.
    Raises HTTP 429 if the limit is exceeded.
    """
    now    = time.monotonic()
    cutoff = now - window_secs
    dq     = _store[identity][window_key]

    # Drop timestamps outside the window
    while dq and dq[0] < cutoff:
        dq.popleft()

    remaining = limit - len(dq)
    if remaining <= 0:
        retry_after = int(window_secs - (now - dq[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error":       "rate_limit_exceeded",
                "limit":       limit,
                "window_secs": window_secs,
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    dq.append(now)
    return remaining - 1   # consume 1, return remaining after this request


def _get_ip(request: Request) -> str:
    """Best-effort client IP (handles reverse proxies)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _get_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key:   Optional[str] = Query(default=None,  alias="api_key"),
) -> Optional[str]:
    return x_api_key or api_key


# ─── FastAPI dependencies ─────────────────────────────────────────────────────

class RateLimitResult:
    """Carries limit metadata so endpoints can add headers if desired."""
    def __init__(self, ip: str, key: Optional[str],
                 ip_rem_min: int, ip_rem_hr: int,
                 key_rem_min: int = -1, key_rem_hr: int = -1):
        self.ip          = ip
        self.key         = key
        self.ip_rem_min  = ip_rem_min
        self.ip_rem_hr   = ip_rem_hr
        self.key_rem_min = key_rem_min
        self.key_rem_hr  = key_rem_hr


def rate_limit_api(
    request: Request,
    api_key: Optional[str] = Depends(_get_key),
) -> RateLimitResult:
    """
    Standard API rate limiter.
    Applies IP limits always; adds key limits when a key is present.
    """
    if not ENABLED:
        return RateLimitResult("disabled", None, -1, -1)

    ip = _get_ip(request)
    ip_rem_min = _check(f"ip:{ip}", "minute", _WINDOW_MINUTE, IP_RPM)
    ip_rem_hr  = _check(f"ip:{ip}", "hour",   _WINDOW_HOUR,   IP_RPH)

    key_rem_min = key_rem_hr = -1
    if api_key:
        key_rem_min = _check(f"key:{api_key}", "minute", _WINDOW_MINUTE, KEY_RPM)
        key_rem_hr  = _check(f"key:{api_key}", "hour",   _WINDOW_HOUR,   KEY_RPH)

    return RateLimitResult(ip, api_key, ip_rem_min, ip_rem_hr, key_rem_min, key_rem_hr)


def rate_limit_upload(
    request: Request,
    api_key: Optional[str] = Depends(_get_key),
) -> RateLimitResult:
    """
    Stricter limiter for the upload endpoint.
    Applies the standard API limits PLUS a lower per-hour upload cap.
    """
    result = rate_limit_api(request, api_key)   # standard checks first

    if not ENABLED:
        return result

    identity = f"key:{api_key}" if api_key else f"ip:{result.ip}"
    _check(identity, "upload_hour", _WINDOW_HOUR, UPLOAD_RPH)

    return result


def rate_limit_ws(identity: str) -> None:
    """
    Lightweight check for WebSocket connections.
    Called once per connection, not per frame.
    """
    if not ENABLED:
        return
    _check(f"ws:{identity}", "minute", _WINDOW_MINUTE, IP_RPM)
    _check(f"ws:{identity}", "hour",   _WINDOW_HOUR,   IP_RPH)


# ─── Helpers for adding rate-limit headers to responses ──────────────────────

def rl_headers(result: RateLimitResult) -> dict:
    """
    Returns a dict of X-RateLimit-* headers to attach to a response.
    Usage:  return JSONResponse(data, headers=rl_headers(rl))
    """
    h = {
        "X-RateLimit-IP-Remaining-Min":  str(result.ip_rem_min),
        "X-RateLimit-IP-Remaining-Hour": str(result.ip_rem_hr),
    }
    if result.key_rem_min >= 0:
        h["X-RateLimit-Key-Remaining-Min"]  = str(result.key_rem_min)
        h["X-RateLimit-Key-Remaining-Hour"] = str(result.key_rem_hr)
    return h
