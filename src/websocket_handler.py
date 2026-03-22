"""
src/websocket_handler.py  –  50-user cap, per-session state
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from fastapi import WebSocket


@dataclass
class Session:
    session_id  : str
    websocket   : WebSocket
    connected_at: datetime = field(default_factory=datetime.utcnow)
    frame_count : int = 0
    status      : str = "connected"   # connected | receiving | running | done | error


class ConnectionManager:
    """
    Manages up to `max_connections` concurrent WebSocket sessions.

    Usage
    -----
    manager = ConnectionManager(max_connections=50)

    # in websocket endpoint:
    await manager.connect(ws, session_id)
    ...
    manager.disconnect(session_id)
    """

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ─── Connection lifecycle ─────────────────────────────────────────────────

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            if len(self._sessions) >= self.max_connections:
                await websocket.accept()
                await websocket.send_text(
                    '{"type":"error","message":"Server at capacity (50 users). Try again later."}'
                )
                await websocket.close(code=1008)
                raise RuntimeError("Max WebSocket connections reached")

            await websocket.accept()
            self._sessions[session_id] = Session(
                session_id=session_id,
                websocket=websocket,
            )
        print(f"[WS] +connect  sid={session_id}  active={len(self._sessions)}")

    def disconnect(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        print(f"[WS] -disconnect  sid={session_id}  active={len(self._sessions)}")

    # ─── Session state helpers ────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def set_status(self, session_id: str, status: str) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.status = status

    def increment_frame(self, session_id: str) -> int:
        s = self._sessions.get(session_id)
        if s:
            s.frame_count += 1
            return s.frame_count
        return 0

    # ─── Broadcast ────────────────────────────────────────────────────────────

    async def broadcast(self, message: str, exclude: Optional[str] = None) -> None:
        dead = []
        for sid, session in list(self._sessions.items()):
            if sid == exclude:
                continue
            try:
                await session.websocket.send_text(message)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.disconnect(sid)

    # ─── Info ────────────────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def is_at_capacity(self) -> bool:
        return len(self._sessions) >= self.max_connections

    def summary(self) -> dict:
        return {
            "active":   self.active_count,
            "capacity": self.max_connections,
            "sessions": [
                {
                    "session_id":   s.session_id,
                    "connected_at": s.connected_at.isoformat(),
                    "frame_count":  s.frame_count,
                    "status":       s.status,
                }
                for s in self._sessions.values()
            ],
        }
