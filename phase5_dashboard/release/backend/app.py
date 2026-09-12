"""FastAPI adapter. All strategic computation is delegated to the shared engine."""

import asyncio
import bisect
import math
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import MAX_BODY, PUBLIC, Service


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    request_id: str = Field(min_length=8, max_length=80, pattern=r"^[a-zA-Z0-9-]+$")


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=24, ge=5, le=100, strict=True)
    seed: int = Field(default=2026, ge=0, le=4294967295, strict=True)


class Phase5InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str | None = Field(default=None, max_length=100)


@asynccontextmanager
async def lifespan(application):
    application.state.service = await asyncio.to_thread(Service)
    yield


app = FastAPI(title="APEX-R Strategy API", version="1.0.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])


@app.middleware("http")
async def local_only(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Cross-origin requests are not accepted"}, status_code=403)
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            return JSONResponse({"detail": "Use application/json"}, status_code=415)
        if len(await request.body()) > MAX_BODY:
            return JSONResponse({"detail": "Request too large"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ValueError)
async def invalid_input(request, error):
    return JSONResponse({"detail": str(error)}, status_code=422)


@app.exception_handler(RuntimeError)
async def engine_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=503)


@app.get("/api/health")
def health(request: Request):
    return request.app.state.service.health("fastapi")


@app.get("/api/scenarios")
def scenarios(request: Request):
    return request.app.state.service.metadata


@app.post("/api/compare")
def compare(body: CompareRequest, request: Request):
    return request.app.state.service.compare(body.model_dump())


@app.post("/api/validate")
def validate(body: ValidationRequest, request: Request):
    return request.app.state.service.execute("validate", body.model_dump())


@app.get("/api/audit")
def audit(request: Request, limit: int = Query(default=100, ge=1, le=100)):
    return request.app.state.service.audit(limit)


@app.get("/api/telemetry")
def telemetry(request: Request, soc: float = Query(default=42, ge=0, le=100)):
    return request.app.state.service.execute("telemetry", {"soc": soc})


@app.get("/api/phase5/metadata")
def phase5_metadata(request: Request):
    return request.app.state.service.phase5.metadata()


@app.get("/api/phase5/replay")
def phase5_replay(request: Request):
    return request.app.state.service.phase5.replay()


@app.get("/api/phase5/state")
def phase5_state(request: Request, time: float = Query(...)):
    return request.app.state.service.phase5.state(time)


@app.post("/api/phase5/inference")
def phase5_inference(body: Phase5InferenceRequest, request: Request):
    return request.app.state.service.phase5.inference(body.window_id)


@app.websocket("/ws/replay")
async def replay(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    replay_data, cached_soc = None, None
    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await websocket.send_json({"error": "Expected a replay request object"})
                continue
            t, soc = message.get("time", 0), message.get("soc", 42)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (t, soc)) or not 0 <= t <= 90 or not 0 <= soc <= 100:
                await websocket.send_json({"error": "time must be 0..90 and soc must be 0..100"})
                continue
            if soc != cached_soc:
                replay_data = await asyncio.to_thread(websocket.app.state.service.execute, "telemetry", {"soc": soc})
                cached_soc = soc
            frames = replay_data["frames"]
            index = max(0, bisect.bisect_right([f["t"] for f in frames], t) - 1)
            await websocket.send_json({"source": "synthetic", "frame": frames[index]})
    except WebSocketDisconnect:
        return
    except (ValueError, RuntimeError):
        await websocket.close(code=1011)


app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="pitwall")
