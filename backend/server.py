"""
OmniSense CSI v4.0 (IEEE Research Edition) - Tactical Command Server
FastAPI Server with Dual-Link UART Ingestion, Multi-Target Telemetry WebSocket,
RMT Crowd Counting, Digital Twin Specular Folding, and IEEE Benchmark Suite.
"""

import asyncio
import struct
import time
import os
from pathlib import Path
import serial
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional

try:
    from dsp_engine import OmniSenseDSPEngine
    from benchmark_suite import IEEEBenchmarkSuite
except ImportError:
    from backend.dsp_engine import OmniSenseDSPEngine
    from backend.benchmark_suite import IEEEBenchmarkSuite

FRAME_FORMAT = "<H I 6B b b B B 104b H"
FRAME_SIZE = 122
MAGIC_WORD = 0x55AA

app = FastAPI(title="OmniSense CSI v4.0 (IEEE Research Edition) Tactical Command Server")
engine = OmniSenseDSPEngine(fs=100.0)
benchmark_suite = IEEEBenchmarkSuite()

pkt_rate_counter = 0
measured_pkt_rate = 0
sim_mode = "single_static"  # "empty" | "single_static" | "single_walking" | "crowd_2_targets" | "crowd_4_targets" | "multi_3_targets" | "through_wall_clutter"


class PerimeterGateRequest(BaseModel):
    max_radius_m: float = 4.0
    wall_cutoff_db: float = 12.0


class RoomTwinRequest(BaseModel):
    width_m: float = 6.0
    length_m: float = 8.0
    wall_material: str = "CONCRETE"
    tx1_pos: List[float] = [-2.0, 0.0]
    tx2_pos: List[float] = [2.0, 0.0]
    rx_pos: List[float] = [0.0, 0.0]
    ap_pos: List[float] = [0.0, 6.0]


class BenchmarkLogRequest(BaseModel):
    actual_people_count: int
    estimated_people_count: int
    ground_truth_x: float = 0.0
    ground_truth_y: float = 0.0
    estimated_x: float = 0.0
    estimated_y: float = 0.0
    reference_hr_bpm: float = 0.0
    estimated_hr_bpm: float = 0.0
    reference_rr_rpm: float = 0.0
    estimated_rr_rpm: float = 0.0
    has_clutter_outside: bool = False
    clutter_rejected: bool = True
    ghost_targets_filtered: int = 0


def calc_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


async def serial_or_hil_ingestion_loop():
    global pkt_rate_counter, measured_pkt_rate, sim_mode
    serial_port = os.getenv("CSI_SERIAL_PORT")
    if not serial_port:
        if os.path.exists("/dev/ttyACM0"):
            serial_port = "/dev/ttyACM0"
        elif os.path.exists("/dev/ttyUSB0"):
            serial_port = "/dev/ttyUSB0"
        else:
            serial_port = "/dev/ttyACM0"
    baud_rate = int(os.getenv("CSI_BAUD_RATE", "921600"))
    ser = None

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=0.02)
        print(f"[OMNISENSE] Connected to ESP32 Radar RX on {serial_port} @ {baud_rate} baud")
    except Exception:
        print(f"[OMNISENSE] Hardware port {serial_port} not found. Running Physics-Accurate RF HIL Simulator @ 100 PKT/S.")

    buf = bytearray()
    last_rate_time = time.time()
    t_sim = 0.0
    sim_pkt_idx = 0

    while True:
        now = time.time()
        if now - last_rate_time >= 1.0:
            measured_pkt_rate = pkt_rate_counter
            pkt_rate_counter = 0
            last_rate_time = now

        # Periodic DSP updates every 10 frames (~100 ms)
        if engine.pkt_count % 10 == 0 and engine.pkt_count > 0:
            engine.update_gait_biometrics()
            engine.compute_vital_signs_bcg()

        if ser and ser.is_open:
            chunk = ser.read(512)
            if chunk:
                buf.extend(chunk)
                while len(buf) >= FRAME_SIZE:
                    idx = buf.find(b'\xaa\x55')
                    if idx == -1:
                        buf.clear()
                        break
                    if idx > 0:
                        del buf[:idx]
                    if len(buf) < FRAME_SIZE:
                        break
                    candidate = bytes(buf[:FRAME_SIZE])
                    expected_crc = struct.unpack("<H", candidate[-2:])[0]
                    if calc_crc16(candidate[:-2]) == expected_crc:
                        unpacked = struct.unpack(FRAME_FORMAT, candidate)
                        # Extract source_mac (bytes 6..11) for Dual-Link separation
                        mac_bytes = unpacked[2:8]
                        source_mac = ":".join(f"{b:02x}" for b in mac_bytes)
                        rssi = float(unpacked[8])
                        csi_iq = np.array(unpacked[12:116], dtype=np.int8)
                        engine.ingest_frame(csi_iq, rssi, source_mac=source_mac)
                        pkt_rate_counter += 1
                        del buf[:FRAME_SIZE]
                    else:
                        del buf[:2]
            await asyncio.sleep(0.001)
        else:
            # Physics-Accurate Multi-Mode HIL Simulation with Dual-Link Staggered Mesh
            t_sim += 0.01
            sim_pkt_idx += 1
            sc_idx = np.arange(52)
            lam = 0.1231 * (1.0 - 0.0012 * (sc_idx - 26))

            # Alternate source MAC between TX1 (Primary, ~100 Hz), TX2 (Secondary, ~75 Hz), and AP
            if sim_pkt_idx % 7 == 0:
                sim_mac = "ac:22:05:41:88:12"  # SAM 4G AP
            elif sim_pkt_idx % 2 == 0:
                sim_mac = "fc:e8:c0:df:37:b5"  # ESP32-TX2 (Secondary Injector)
            else:
                sim_mac = "fc:e8:c0:df:37:b4"  # ESP32-TX1 (Primary Injector)

            if sim_mode == "empty":
                # Pure multipath thermal noise in an empty room
                amp = 25.0 + np.random.normal(0, 0.04, 52)
                phase = np.random.normal(0, 0.02, 52)
                rssi = -55.0 + np.random.normal(0, 0.1)

            elif sim_mode == "single_static":
                # Seated stationary subject with physiological chest displacement
                resp_disp = 0.005 * np.sin(2 * np.pi * 0.28 * t_sim)  # 0.28 Hz = 16.8 RPM
                cardiac_phase = (t_sim * 1.22) % 1.0                  # 1.22 Hz = 73.2 BPM
                pqrst_wave = 0.0004 * (
                    0.2 * np.exp(-((cardiac_phase - 0.18) / 0.03) ** 2) -
                    0.3 * np.exp(-((cardiac_phase - 0.28) / 0.015) ** 2) +
                    1.5 * np.exp(-((cardiac_phase - 0.31) / 0.012) ** 2) -
                    0.4 * np.exp(-((cardiac_phase - 0.34) / 0.015) ** 2) +
                    0.35 * np.exp(-((cardiac_phase - 0.52) / 0.04) ** 2)
                )
                total_d = 4.0 + resp_disp + pqrst_wave
                phase = -2.0 * np.pi * total_d / lam + np.random.normal(0, 0.015, 52)
                amp = 28.0 + 3.0 * np.sin(sc_idx * 0.15) + np.random.normal(0, 0.08, 52)
                rssi = -51.5 + np.random.normal(0, 0.1)

            elif sim_mode == "single_walking":
                # 1 person walking back and forth (Strict Rank-1 Coherent Target)
                x_pos = 2.0 * np.sin(0.4 * np.pi * t_sim)
                y_pos = 4.0 + 1.2 * np.cos(0.4 * np.pi * t_sim)
                dist = np.hypot(x_pos, y_pos)
                doppler_shift = 2.0 * np.pi * 0.8 * np.sin(0.4 * np.pi * t_sim)
                phase = -2.0 * np.pi * dist / lam + doppler_shift * t_sim + np.random.normal(0, 0.02, 52)
                amp = 30.0 + 10.0 * np.sin(sc_idx * 0.14 + t_sim * 2.2) + np.random.normal(0, 0.15, 52)
                rssi = -48.0 - 2.5 * dist + np.random.normal(0, 0.3)

            elif sim_mode in ["crowd_2_targets"]:
                # 2 distinct moving individuals in separate sectors
                x1 = -1.8 + 0.9 * np.sin(0.6 * t_sim)
                y1 = 3.5 + 0.8 * np.cos(0.6 * t_sim)
                r1 = np.hypot(x1, y1)

                x2 = 1.8 + 0.8 * np.cos(0.45 * t_sim)
                y2 = 4.5 + 0.9 * np.sin(0.45 * t_sim)
                r2 = np.hypot(x2, y2)

                phi1 = -2.0 * np.pi * r1 / lam[:26]
                phi2 = -2.0 * np.pi * r2 / lam[26:]
                phase = np.concatenate([phi1, phi2]) + np.random.normal(0, 0.03, 52)

                amp1 = 28.0 + 7.0 * np.sin(sc_idx[:26] * 0.2 + t_sim * 3.0)
                amp2 = 29.0 + 6.5 * np.cos(sc_idx[26:] * 0.25 + t_sim * 2.1)
                amp = np.concatenate([amp1, amp2]) + np.random.normal(0, 0.15, 52)
                rssi = -49.0 + np.random.normal(0, 0.3)

            elif sim_mode in ["crowd_4_targets", "multi_3_targets"]:
                # 4 distinct individuals (crowd mode demonstrating unbounded RMT scaling)
                r1 = 3.2 + 0.8 * np.sin(0.5 * t_sim)
                r2 = 4.4 + 0.7 * np.cos(0.4 * t_sim)
                r3 = 5.2 + 0.9 * np.sin(0.3 * t_sim)
                r4 = 3.8 + 0.6 * np.cos(0.6 * t_sim)

                phi1 = -2.0 * np.pi * r1 / lam[:13]
                phi2 = -2.0 * np.pi * r2 / lam[13:26]
                phi3 = -2.0 * np.pi * r3 / lam[26:39]
                phi4 = -2.0 * np.pi * r4 / lam[39:]
                phase = np.concatenate([phi1, phi2, phi3, phi4]) + np.random.normal(0, 0.03, 52)

                amp1 = 27.0 + 6.0 * np.sin(sc_idx[:13] * 0.3 + t_sim * 2.8)
                amp2 = 28.0 + 5.5 * np.cos(sc_idx[13:26] * 0.25 + t_sim * 2.2)
                amp3 = 26.5 + 6.5 * np.sin(sc_idx[26:39] * 0.2 + t_sim * 1.9)
                amp4 = 29.0 + 5.0 * np.cos(sc_idx[39:] * 0.28 + t_sim * 3.4)
                amp = np.concatenate([amp1, amp2, amp3, amp4]) + np.random.normal(0, 0.15, 52)
                rssi = -48.5 + np.random.normal(0, 0.3)

            elif sim_mode == "through_wall_clutter":
                # External motion outside room perimeter (e.g., at r = 6.5m with low SNR)
                r_ext = 6.8 + 1.2 * np.sin(0.3 * t_sim)
                phase = -2.0 * np.pi * r_ext / lam + np.random.normal(0, 0.08, 52)
                amp = 26.0 + 1.2 * np.sin(sc_idx * 0.2 + t_sim * 1.5) + np.random.normal(0, 0.1, 52)
                rssi = -68.0 + np.random.normal(0, 0.4)

            else:
                amp = 28.0 + np.random.normal(0, 0.1, 52)
                phase = np.random.normal(0, 0.05, 52)
                rssi = -52.0

            csi_iq = np.empty(104, dtype=np.int8)
            csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
            csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
            engine.ingest_frame(csi_iq, rssi, source_mac=sim_mac)
            pkt_rate_counter += 1
            await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(serial_or_hil_ingestion_loop())


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/hud", response_class=HTMLResponse)
async def serve_hud():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return HTMLResponse("<h2>OmniSense CSI v4.0 HUD loading...</h2>")


@app.post("/api/calibrate")
async def trigger_calibration():
    engine.start_room_calibration()
    return {"status": "CALIBRATING_5S"}


@app.post("/api/enroll/{target_name}")
async def trigger_enrollment(target_name: str):
    engine.start_gait_enrollment(target_name, duration_sec=20.0)
    return {"status": "ENROLLING_20S", "target": target_name, "duration_sec": 20.0}


@app.post("/api/select_target/{track_id}")
async def select_target_endpoint(track_id: int):
    engine.select_target(track_id)
    return {
        "status": "TARGET_SELECTED" if track_id > 0 else "TARGET_UNLOCKED",
        "selected_target_id": engine.selected_target_id
    }


@app.post("/api/perimeter_gate")
async def update_perimeter_gate(req: PerimeterGateRequest):
    engine.set_perimeter_gate(req.max_radius_m, req.wall_cutoff_db)
    return {
        "status": "PERIMETER_GATE_UPDATED",
        "max_radius_m": engine.max_sensing_radius_m,
        "wall_cutoff_db": engine.wall_attenuation_cutoff_db
    }


@app.get("/api/room_twin")
async def get_room_twin():
    return engine.room_twin


@app.post("/api/room_twin")
async def update_room_twin(req: RoomTwinRequest):
    cfg = engine.configure_room_twin(
        width_m=req.width_m,
        length_m=req.length_m,
        wall_material=req.wall_material,
        tx1_pos=req.tx1_pos,
        tx2_pos=req.tx2_pos,
        rx_pos=req.rx_pos,
        ap_pos=req.ap_pos
    )
    return {"status": "ROOM_TWIN_CONFIGURED", "room_twin": cfg}


@app.post("/api/benchmark/log")
async def benchmark_log_trial(req: BenchmarkLogRequest):
    data = req.dict()
    entry = benchmark_suite.log_trial(data, data)
    return {"status": "LOGGED", "trial": entry, "metrics": benchmark_suite.get_metrics()}


@app.get("/api/benchmark/metrics")
async def benchmark_get_metrics():
    return benchmark_suite.get_metrics()


@app.get("/api/benchmark/latex")
async def benchmark_get_latex():
    return {"latex": benchmark_suite.generate_ieee_latex_table()}


@app.post("/api/benchmark/reset")
async def benchmark_reset():
    benchmark_suite.reset()
    return {"status": "RESET", "metrics": benchmark_suite.get_metrics()}


@app.post("/api/sim_mode/{mode}")
async def set_sim_mode(mode: str):
    global sim_mode
    mode_clean = mode.lower().strip()
    valid_modes = [
        "empty", "single_static", "single_walking",
        "crowd_2_targets", "crowd_4_targets", "multi_3_targets", "through_wall_clutter"
    ]
    if mode_clean in valid_modes:
        sim_mode = mode_clean
        if sim_mode == "empty":
            engine.tracks.clear()
            engine.presence_state = "EMPTY_ROOM"
        return {"status": "SIM_MODE_UPDATED", "sim_mode": sim_mode}
    return {"status": "ERROR", "message": f"Invalid mode. Available: {', '.join(valid_modes)}"}


# -----------------------------------------------------------------------------
# Real-Time Telemetry WebSocket Stream (25 FPS)
# -----------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state = engine.get_telemetry_state()
            payload = {
                "pkt_rate": measured_pkt_rate,
                "rssi": round(engine.rssi_ema, 1),
                "calibrating": engine.is_calibrating,
                "sim_mode": sim_mode,
                "presence_state": state["presence_state"],
                "crowd": state["crowd"],
                "mesh_status": state["mesh_status"],
                "perimeter_gate": state["perimeter_gate"],
                "room_twin": state["room_twin"],
                "enrolling": state["enrolling"],
                "targets": state["targets"],
                "vitals": state["vitals"],
                # Backwards compatible fields
                "pos": {
                    "x": round(float(engine.kf_state[0]), 2),
                    "y": round(float(engine.kf_state[1]), 2),
                    "vx": round(float(engine.kf_state[2]), 2),
                    "vy": round(float(engine.kf_state[3]), 2)
                },
                "biometrics": {
                    "target_id": engine.current_target_id,
                    "mahalanobis_dist": engine.current_gait_distance,
                    "profiles_count": len(engine.gait_profiles)
                },
                "subcarriers": np.round(np.abs(engine.amp_ring[-1, :]), 3).tolist()
            }
            await ws.send_json(payload)
            await asyncio.sleep(0.04)  # 25 FPS
    except WebSocketDisconnect:
        pass
