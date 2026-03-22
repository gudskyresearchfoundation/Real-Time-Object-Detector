"""
app.py  –  Uvicorn entrypoint for ObjectVision
============================================================
Run:
    python app.py
    
    OR directly with uvicorn:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open:  http://localhost:8000/static/index.html
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load .env before importing the FastAPI app so all os.getenv() calls
# inside server.py pick up the values at import time.
load_dotenv()

from src.server import app  # noqa: E402  (import after load_dotenv)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))

    print(f"""
╔══════════════════════════════════════════════╗
║         ObjectVision  ·  YOLOv8 API          ║
╠══════════════════════════════════════════════╣
║  URL  :  http://{host}:{port}              ║
║  UI   :  http://localhost:{port}/static/index.html  ║
║  Docs :  http://localhost:{port}/docs         ║
╚══════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,   # reload mode only supports 1 worker
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
