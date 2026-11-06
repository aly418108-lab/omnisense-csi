"""
OmniSense CSI - Pure Hardware Server (FastAPI + WebSockets)
Module: backend/main.py

Features:
1. 100% Pure Hardware Engine: Zero simulator fallbacks or dummy data.
2. Auto-Connect on Startup: Scans and connects to /dev/ttyACM0 @ 115200 baud.
3. Dual-Gated PCA Ballistocardiography: Interactive target manual selection via WebSocket.
4. CSIKit Linear Phase Sanitization & 2D Kalman/Spectral Localization.
5. Zero-Profile Boot with Single-Class Anomaly Distance Safeguard.
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.csi_ingestor import CSIIngestorManager, CSIPacket
from backend.dsp_engine import CSIDSPEngine
from backend.ml_biometrics import GaitBiometricClassifier, PROFILES_DB_PATH

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OmniSense.Main")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Core Production Engine Instances
ingestor = CSIIngestorManager(baudrate=115200)
dsp_engine = CSIDSPEngine()
ml_classifier = GaitBiometricClassifier()


# WebSocket Connection Manager
class TelemetryBroadcaster:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, data: Dict[str, Any]):
        if not self.active_connections:
            return
        msg = json.dumps(data)
        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(msg)
            except Exception:
                disconnected.append(connection)

        if disconnected:
            async with self._lock:
                for dead_conn in disconnected:
                    if dead_conn in self.active_connections:
                        self.active_connections.remove(dead_conn)


broadcaster = TelemetryBroadcaster()

# Background Processing Worker
processing_task: Optional[asyncio.Task] = None
last_biometric_result: Dict[str, Any] = {
    "identity": "AMBIENT_SILENCE",
    "confidence": 0.0,
    "is_known": False,
    "is_walking": False,
    "status_code": "IDLE",
    "probabilities": {},
    "color": "#00f3ff"
}


async def csi_processing_loop():
    """
    Continuous real-time loop pulling CSI frames from ESP32, executing DSP filtering,
    running ML biometric evaluation, and pushing updates to WebSockets.
    """
    global last_biometric_result
    logger.info("CSI hardware processing loop started.")
    frame_count = 0

    while True:
        try:
            # Pull packet from hardware queue
            packet: CSIPacket = await ingestor.queue.get()

            # DSP: Amplitude/Phase conversion, Disturbance Index, 2D Spatial & Gated Vitals
            dsp_result = dsp_engine.process_frame(
                subcarriers=packet.subcarriers,
                timestamp=packet.timestamp,
                rssi=packet.rssi,
                metadata=packet.metadata
            )

            # ML: Gait Feature Extraction & Classification
            amplitudes = dsp_result["amplitudes"]
            ml_classifier.add_frame(amplitudes, timestamp=packet.timestamp)

            disturbance = dsp_result["disturbance_index"]

            # Run biometric classification every 4 frames or on motion change (> 22%)
            frame_count += 1
            if frame_count % 4 == 0 or disturbance > 22.0:
                last_biometric_result = ml_classifier.classify_current_window(disturbance)

            # Assemble real-time telemetry payload
            telemetry_payload = {
                "type": "TELEMETRY_FRAME",
                "timestamp": packet.timestamp,
                "rssi": packet.rssi,
                "mac": packet.mac,
                "source": packet.source,
                "disturbance_index": disturbance,
                "state_label": dsp_result["state_label"],
                "subcarriers": amplitudes,
                "norm_subcarriers": dsp_result["norm_amplitudes"],
                "phases": dsp_result["phases"],
                "vitals": dsp_result["vitals"],
                "spatial": dsp_result["spatial"],
                "biometrics": last_biometric_result,
                "hardware": ingestor.get_hardware_status()
            }

            # Broadcast to UI
            await broadcaster.broadcast(telemetry_payload)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in processing loop: {e}", exc_info=False)
            await asyncio.sleep(0.01)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting OmniSense CSI Pure Hardware Server...")
    await ingestor.start()
    global processing_task
    processing_task = asyncio.create_task(csi_processing_loop())
    yield
    # Shutdown
    logger.info("Shutting down OmniSense CSI Radar Server...")
    if processing_task:
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass
    await ingestor.stop()


app = FastAPI(
    title="OmniSense CSI Radar API (Hardware)",
    description="Wi-Fi-Based Human Surveillance, Gait Biometrics & Vital Signs Radar",
    version="2.1.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class ReconnectRequest(BaseModel):
    port: Optional[str] = None
    baudrate: Optional[int] = 115200


class VitalsLockRequest(BaseModel):
    locked: bool


class EnrollmentRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    color: Optional[str] = "#e056fd"
    frames: List[List[float]]  # CSI amplitude frames collected during walking test


# REST Endpoints
@app.get("/api/status")
async def get_system_status():
    hw_status = ingestor.get_hardware_status()
    return {
        "status": "online",
        "system": "OmniSense CSI Radar v2.1",
        "version": "2.1.0",
        "mode": "PURE_HARDWARE",
        "hardware": hw_status,
        "connected_clients": len(broadcaster.active_connections),
        "enrolled_profiles": ml_classifier.classes_,
        "disturbance_index": dsp_engine.prev_disturbance,
        "vitals_manually_locked": dsp_engine.vitals_manually_locked
    }


@app.get("/api/ports")
async def get_available_ports():
    ports = ingestor.serial_reader.list_available_ports()
    return {"ports": ports}


@app.post("/api/reconnect")
async def reconnect_serial(req: ReconnectRequest):
    res = ingestor.reconnect(port=req.port, baudrate=req.baudrate)
    return res


@app.post("/api/calibrate")
async def calibrate_baseline():
    dsp_engine.calibrate_baseline()
    return {"status": "ok", "message": "Ambient baseline calibrated successfully."}


@app.post("/api/vitals/lock")
async def set_vitals_lock(req: VitalsLockRequest):
    dsp_engine.set_vitals_manual_lock(req.locked)
    return {"status": "ok", "vitals_manually_locked": dsp_engine.vitals_manually_locked}


@app.get("/api/profiles")
async def get_profiles():
    return {
        "profiles": ml_classifier.profiles_db.get("profiles", []),
        "classes": ml_classifier.classes_,
        "threshold": ml_classifier.CONFIDENCE_THRESHOLD
    }


@app.post("/api/enroll")
async def enroll_gait(req: EnrollmentRequest):
    """
    On-stage live enrollment of a new walking profile from live hardware frames.
    Retrains the classifier in < 0.5s.
    """
    result = ml_classifier.enroll_new_signature(
        name=req.name,
        recorded_frames=req.frames,
        description=req.description or "",
        color=req.color or "#e056fd"
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """
    Delete an enrolled profile and retrain/reset the classifier.
    """
    result = ml_classifier.delete_profile(profile_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


# WebSocket Telemetry Stream & Command Receiver
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # Handle inbound commands from frontend
            raw = await websocket.receive_text()
            try:
                cmd = json.loads(raw)
                action = cmd.get("action")
                if action == "SET_VITALS_LOCK":
                    dsp_engine.set_vitals_manual_lock(cmd.get("locked", False))
                elif action == "CALIBRATE_BASELINE":
                    dsp_engine.calibrate_baseline()
                elif action == "RECONNECT_SERIAL":
                    ingestor.reconnect(cmd.get("port"), cmd.get("baudrate", 115200))
            except Exception as e:
                logger.debug(f"WS command error: {e}")
    except WebSocketDisconnect:
        await broadcaster.disconnect(websocket)
    except Exception:
        await broadcaster.disconnect(websocket)


# Mount Frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>OmniSense CSI Radar</h1><p>Frontend loading...</p>")
