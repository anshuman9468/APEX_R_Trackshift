"""One engine shared by the browser, HTTP server, and FastAPI adapter."""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .phase5 import Phase5Service

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts" / "engine-cli.cjs"
PUBLIC = ROOT / "dist"
MAX_BODY = 1_000_000


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at DESC)")
            db.commit()

    def get(self, request_id):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            row = db.execute("SELECT payload FROM decisions WHERE id=?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, record):
        payload = json.dumps(record, allow_nan=False, separators=(",", ":"))
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("INSERT OR IGNORE INTO decisions(id, created_at, payload) VALUES (?, ?, ?)", (record["id"], record["created_at"], payload))
            db.commit()
        return self.get(record["id"])

    def recent(self, limit=100):
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100")
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            rows = db.execute("SELECT payload FROM decisions ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]


class Service:
    def __init__(self, database=None):
        self.node = shutil.which("node")
        if not self.node:
            raise RuntimeError("Node.js 18 or later is required for the shared engine. The standalone HTML app can run without it.")
        self.store = Store(database or os.environ.get("APEX_DATABASE", ROOT / "runtime" / "apex.db"))
        self.slots = threading.BoundedSemaphore(4)
        self.metadata = self.execute("meta", {})
        # Phase 5 is optional at runtime; its PyTorch dependency is loaded
        # lazily by the adapter and does not block the portable app.
        self.phase5 = Phase5Service()

    def execute(self, method, payload):
        encoded = json.dumps({"method": method, "input": payload}, allow_nan=False)
        if len(encoded.encode()) > MAX_BODY:
            raise ValueError("Request is too large")
        if not self.slots.acquire(timeout=2):
            raise RuntimeError("Engine busy. Please retry.")
        try:
            process = subprocess.run([self.node, str(ENGINE)], input=encoded, text=True, capture_output=True, timeout=20, cwd=ROOT, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Engine timed out") from error
        finally:
            self.slots.release()
        if process.returncode:
            raise ValueError(process.stderr.strip()[:500] or "Invalid engine request")
        return json.loads(process.stdout)

    def compare(self, body):
        if not isinstance(body, dict) or set(body) - {"input", "provenance", "request_id"}:
            raise ValueError("Expected input, provenance, and request_id")
        request_id = body.get("request_id", "")
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{8,80}", request_id):
            raise ValueError("request_id must contain 8 to 80 letters, digits, or hyphens")
        provenance = body.get("provenance", {})
        if not isinstance(provenance, dict) or len(json.dumps(provenance)) > 20000:
            raise ValueError("Invalid provenance")
        payload = body.get("input", {})
        if not isinstance(payload, dict):
            raise ValueError("input must be an object")
        normalised = self.execute("normalise", payload)
        previous = self.store.get(request_id)
        if previous:
            if previous["result"]["input"] != normalised or previous["provenance"] != provenance:
                raise ValueError("request_id already belongs to a different decision")
            return previous["result"]
        result = self.execute("compare", normalised)
        record = {"id": request_id, "created_at": datetime.now(timezone.utc).isoformat(), "result": result, "provenance": provenance}
        stored = self.store.save(record)
        if stored["result"]["input"] != normalised or stored["provenance"] != provenance:
            raise ValueError("Concurrent request_id conflict")
        return stored["result"]

    def health(self, framework="stdlib"):
        return {"status": "ok", "engine_version": self.metadata["version"], "framework": framework, "storage": "sqlite", "offline": True, "websocket": framework == "fastapi"}

    def audit(self, limit=100):
        return {"records": self.store.recent(limit), "storage": "sqlite"}
