import os
from datetime import datetime, timezone

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
