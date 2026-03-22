"""
src/backend/auth.py  –  API Key Authentication
============================================================
- Keys stored in SQLite  (data/keys.db)
- Keys are SHA-256 hashed at rest — plaintext never saved
- FastAPI dependency:  require_api_key  (REST)
- WebSocket helper:    verify_ws_key    (WS handshake header / query param)
- Admin CLI:           python -m src.backend.auth keygen --label "workshop"
                       python -m src.backend.auth list
                       python -m src.backend.auth revoke <key_id>

Key format:  ov_<32 random hex chars>   e.g.  ov_a1b2c3d4e5f6...
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query, status

# ─── Config ───────────────────────────────────────────────────────────────────
AUTH_ENABLED  = os.getenv("AUTH_ENABLED", "true").lower() == "true"
DB_PATH       = Path(os.getenv("AUTH_DB_PATH", "data/keys.db"))
ADMIN_KEY     = os.getenv("ADMIN_KEY", "")          # if set, only this key can call /admin/*
KEY_PREFIX    = "ov_"


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                label       TEXT    NOT NULL DEFAULT '',
                key_hash    TEXT    NOT NULL UNIQUE,
                key_prefix  TEXT    NOT NULL,          -- first 10 chars for display
                created_at  TEXT    NOT NULL,
                last_used   TEXT,
                use_count   INTEGER NOT NULL DEFAULT 0,
                is_active   INTEGER NOT NULL DEFAULT 1
            )
        """)
        con.commit()


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ─── Key management ───────────────────────────────────────────────────────────

def generate_key(label: str = "") -> str:
    """Create a new API key, store its hash, return the plaintext key."""
    _init_db()
    plaintext = KEY_PREFIX + secrets.token_hex(32)
    h         = _hash(plaintext)
    prefix    = plaintext[:10]
    now       = datetime.utcnow().isoformat()

    with _conn() as con:
        con.execute(
            "INSERT INTO api_keys (label, key_hash, key_prefix, created_at) VALUES (?,?,?,?)",
            (label, h, prefix, now),
        )
        con.commit()

    return plaintext


def revoke_key(key_id: int) -> bool:
    with _conn() as con:
        cur = con.execute(
            "UPDATE api_keys SET is_active=0 WHERE id=?", (key_id,)
        )
        con.commit()
        return cur.rowcount > 0


def list_keys() -> list[dict]:
    _init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT id, label, key_prefix, created_at, last_used, use_count, is_active "
            "FROM api_keys ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def _lookup(key: str) -> Optional[sqlite3.Row]:
    """Return the DB row if the key is valid and active, else None."""
    _init_db()
    h = _hash(key)
    with _conn() as con:
        row = con.execute(
            "SELECT id, label, use_count, is_active FROM api_keys WHERE key_hash=?",
            (h,),
        ).fetchone()
        if row and row["is_active"]:
            # Update last_used + use_count
            con.execute(
                "UPDATE api_keys SET last_used=?, use_count=use_count+1 WHERE id=?",
                (datetime.utcnow().isoformat(), row["id"]),
            )
            con.commit()
            return row
    return None


# ─── FastAPI dependencies ─────────────────────────────────────────────────────

def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key:   Optional[str] = Query(default=None,  alias="api_key"),
) -> dict:
    """
    Dependency for protected REST endpoints.
    Pass key via header  X-API-Key: ov_...
    or query param       ?api_key=ov_...
    """
    if not AUTH_ENABLED:
        return {"id": 0, "label": "auth_disabled"}

    key = x_api_key or api_key
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass X-API-Key header or ?api_key= query param.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    row = _lookup(key)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key.",
        )

    return {"id": row["id"], "label": row["label"]}


def verify_ws_key(key: Optional[str]) -> bool:
    """
    Lightweight check for WebSocket connections.
    Returns True if auth is disabled OR the key is valid.
    """
    if not AUTH_ENABLED:
        return True
    if not key:
        return False
    return _lookup(key) is not None


# ─── Admin dependency ─────────────────────────────────────────────────────────

def require_admin(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Restrict /admin/* routes to the ADMIN_KEY."""
    if not ADMIN_KEY:
        return   # no admin key configured → allow (dev mode)
    if x_api_key != ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────
# python -m src.backend.auth keygen --label "student-01"
# python -m src.backend.auth list
# python -m src.backend.auth revoke 3

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="ObjectVision API Key Manager")
    sub    = parser.add_subparsers(dest="cmd")

    kg = sub.add_parser("keygen", help="Generate a new API key")
    kg.add_argument("--label", default="", help="Human label for this key")

    sub.add_parser("list", help="List all keys")

    rv = sub.add_parser("revoke", help="Revoke a key by ID")
    rv.add_argument("key_id", type=int)

    args = parser.parse_args()

    if args.cmd == "keygen":
        key = generate_key(args.label)
        print(f"\n✓  New API key generated")
        print(f"   Label : {args.label or '(none)'}")
        print(f"   Key   : {key}")
        print(f"\n   ⚠  Save this — it will NOT be shown again.\n")

    elif args.cmd == "list":
        keys = list_keys()
        if not keys:
            print("No keys found.")
        else:
            print(f"\n{'ID':<4} {'Label':<20} {'Prefix':<12} {'Uses':<6} {'Active':<7} Created")
            print("─" * 72)
            for k in keys:
                active = "✓" if k["is_active"] else "✗"
                print(f"{k['id']:<4} {k['label']:<20} {k['key_prefix']:<12} "
                      f"{k['use_count']:<6} {active:<7} {k['created_at'][:19]}")
            print()

    elif args.cmd == "revoke":
        ok = revoke_key(args.key_id)
        print(f"{'✓ Revoked' if ok else '✗ Key not found'}  id={args.key_id}")

    else:
        parser.print_help()
