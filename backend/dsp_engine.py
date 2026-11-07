"""
OmniSense CSI v4.0 (IEEE Research Edition) - Core DSP & Signal Processing Engine
Mathematical Implementations:
- Subsystem A: Multi-MAC Dual-Link Bistatic Mesh Demultiplexing (ESP32-TX1, ESP32-TX2, Router AP)
- Subsystem B: Unbounded RMT-MDL Crowd Counting & Marchenko-Pastur Subspace Eigen-Spectrum (0 to 10+ Targets)
- Subsystem C: Through-Wall & Perimeter Range Gating via Windowed IFFT Channel Impulse Response (CIR)
- Subsystem D: Panorama-Guided RF Digital Twin & Specular Multipath Mirror-Ghost Folding
- Subsystem E: Target-Locked BCG/PQRST Vital Signs Isolation (SVD + VMD + Coiflet-4 DWT)
"""

import time
import numpy as np
from scipy import signal, optimize
import pywt
from vmdpy import VMD
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from typing import List, Dict, Optional, Tuple, Any


class CrowdResult(int):
    """Encapsulates estimated crowd count, allowing dual usage as integer and 3-tuple."""
    def __new__(cls, val, eigvals=None, entropy=0.0):
        obj = super().__new__(cls, int(val))
        obj.k = int(val)
        obj.eigenvalues = [float(v) for v in (eigvals or [])]
        obj.entropy = float(entropy)
        return obj

    def __iter__(self):
        yield int(self)
        yield self.eigenvalues
        yield self.entropy


class TargetTrack:
    """4-State Kinematic Target Track with Spatial CSI Signature, Strict Identity Binding & Stationary Lock."""
    def __init__(self, track_id: int, init_pos: np.ndarray, dt: float = 0.01):
        self.track_id = track_id
        # State: [x, y, vx, vy]^T
        self.kf_state = np.array([float(init_pos[0]), float(init_pos[1]), 0.0, 0.0], dtype=np.float64)
        self.kf_P = np.eye(4, dtype=np.float64) * 0.8
        self.dt = dt
        self.kf_F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 0.995, 0.0],
            [0.0, 0.0, 0.0, 0.995]
        ], dtype=np.float64)
        self.kf_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.kf_Q = np.diag([0.005, 0.005, 0.08, 0.08])

        # Strict identity binding
        self.identity_label = "UNKNOWN TARGET"
        self.locked_identity: Optional[str] = None
        self.identity_confidence: float = 0.0
        self.mahalanobis_dist = 99.9

        # Dual-regime status: "MOVING" | "STATIONARY_LOCKED" | "STATIC_VITAL"
        self.status = "MOVING"
        self.pinned_pos: Optional[np.ndarray] = None
        self.low_speed_duration: float = 0.0
        self.presence_hold_remaining: float = 60.0

        self.spatial_subcarrier_weights = np.ones(50, dtype=np.float64)
        self.component_trace = np.zeros(100, dtype=np.float64)
        self.missed_count = 0
        self.age = 1
        self.trail: List[List[float]] = [[round(float(init_pos[0]), 2), round(float(init_pos[1]), 2)]]
        self.selected = False

    def predict(self):
        if self.status in ("STATIONARY_LOCKED", "STATIC_VITAL") and self.pinned_pos is not None:
            # Zero-velocity coordinate pinning: suppress random-walk jitter
            self.kf_state[:2] = self.pinned_pos
            self.kf_state[2:] = 0.0
            self.kf_P = np.diag([0.001, 0.001, 0.001, 0.001])
            self.age += 1
            return

        self.kf_state = self.kf_F @ self.kf_state
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1

    def update(self, z: np.ndarray, r_scale: float = 0.08, is_motion_active: bool = False):
        if self.status in ("STATIONARY_LOCKED", "STATIC_VITAL") and self.pinned_pos is not None:
            disp = float(np.linalg.norm(z - self.pinned_pos))
            if disp > 0.35 or is_motion_active:
                # Subject has resumed active locomotion
                self.status = "MOVING"
                self.pinned_pos = None
                self.low_speed_duration = 0.0
                self.presence_hold_remaining = 60.0
            else:
                # Retain pinned coordinates and clamp velocity
                self.kf_state[:2] = self.pinned_pos
                self.kf_state[2:] = 0.0
                self.missed_count = 0
                return

        R_cov = np.eye(2, dtype=np.float64) * r_scale
        y_res = z - (self.kf_H @ self.kf_state)
        S_cov = self.kf_H @ self.kf_P @ self.kf_H.T + R_cov
        K_gain = self.kf_P @ self.kf_H.T @ np.linalg.inv(S_cov)
        self.kf_state = self.kf_state + (K_gain @ y_res)
        self.kf_P = (np.eye(4, dtype=np.float64) - K_gain @ self.kf_H) @ self.kf_P
        self.missed_count = 0

        # Check speed & motion for stationary lock
        if self.speed < 0.08 and not is_motion_active:
            self.low_speed_duration += 0.1
            if self.low_speed_duration >= 1.0:
                self.status = "STATIONARY_LOCKED"
                self.pinned_pos = np.copy(self.kf_state[:2])
                self.kf_state[2:] = 0.0
                self.presence_hold_remaining = 60.0
        else:
            self.low_speed_duration = 0.0
            self.status = "MOVING"
            self.pinned_pos = None

        # Update smoothed trail
        curr_pt = [round(float(self.kf_state[0]), 2), round(float(self.kf_state[1]), 2)]
        if len(self.trail) == 0 or np.hypot(curr_pt[0] - self.trail[-1][0], curr_pt[1] - self.trail[-1][1]) > 0.05:
            self.trail.append(curr_pt)
            if len(self.trail) > 30:
                self.trail.pop(0)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.kf_state[2], self.kf_state[3]))


class OmniSenseDSPEngine:
    """OmniSense CSI v4.0 (IEEE Research Edition) Digital Signal Processing Engine."""

    def __init__(self, fs: float = 100.0):
        self.fs = fs
        self.dt = 1.0 / self.fs
        self.num_sc = 52
        self.sc_indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        self.wavelength = 3e8 / 2.437e9  # ~0.1231 m (Channel 6)

        # ---------------------------------------------------------------------
        # 1. Calibration & Baseline Multi-Path Noise
        # ---------------------------------------------------------------------
        self.is_calibrating = False
        self.cal_buffer_amp: List[np.ndarray] = []
        self.cal_buffer_rssi: List[float] = []
        self.ref_amp = np.ones(self.num_sc, dtype=np.float64)
        self.ref_std = np.full(self.num_sc, 0.05, dtype=np.float64)
        self.ref_noise_energy = 0.005
        self.ref_rssi = -55.0
        self.ref_centroid = 0.0

        # ---------------------------------------------------------------------
        # 2. Subsystem A: Multi-MAC Dual-Link Bistatic Mesh Buffers
        # ---------------------------------------------------------------------
        self.win_size = int(8.0 * self.fs)
        self.amp_ring = np.zeros((self.win_size, self.num_sc), dtype=np.float64)
        self.phase_ring = np.zeros((self.win_size, self.num_sc), dtype=np.float64)
        self.ratio_ring = np.zeros((self.win_size, self.num_sc - 2), dtype=np.complex128)
        self.rssi_ema = -55.0
        self.pkt_count = 0

        # Dedicated multi-MAC ring buffers for TX1, TX2, and AP
        self.link_streams: Dict[str, dict] = {
            "stream_tx1": {"pkt_count": 0, "rssi": -55.0, "last_seen": 0.0, "buf": np.zeros((self.win_size, self.num_sc))},
            "stream_tx2": {"pkt_count": 0, "rssi": -55.0, "last_seen": 0.0, "buf": np.zeros((self.win_size, self.num_sc))},
            "stream_ap":  {"pkt_count": 0, "rssi": -55.0, "last_seen": 0.0, "buf": np.zeros((self.win_size, self.num_sc))}
        }

        # ---------------------------------------------------------------------
        # 3. Subsystem B: Unbounded RMT-MDL Crowd Counting & Presence
        # ---------------------------------------------------------------------
        self.presence_state = "EMPTY_ROOM"  # "EMPTY_ROOM" | "STATIC_VITAL" | "ACTIVE_MULTI"
        self.empty_room_frames = 0
        self.instant_motion_energy = 0.0
        self.instant_vital_snr = 0.0
        self.estimated_crowd_count = 0
        self.pending_crowd_count = 0
        self.crowd_hysteresis_start_time = 0.0
        self.von_neumann_entropy = 0.0
        self.eigenvalue_spectrum: List[float] = []

        # Multi-Target Tracks (unbounded, not capped at 3)
        self.tracks: List[TargetTrack] = []
        self.next_track_id = 1
        self.selected_target_id: Optional[int] = None

        # ---------------------------------------------------------------------
        # 4. Subsystem C: Through-Wall & Perimeter Range Gating
        # ---------------------------------------------------------------------
        self.max_sensing_radius_m = 4.0        # Default 4.0m perimeter radius
        self.wall_attenuation_cutoff_db = 12.0  # Cutoff threshold for through-wall clutter
        self.clutter_suppressed_count = 0
        self.total_detections_count = 0
        self.clutter_rejected_pct = 0.0

        # ---------------------------------------------------------------------
        # 5. Subsystem D: Panorama-Guided RF Digital Twin & Multipath Ghost Folding
        # ---------------------------------------------------------------------
        w_init = 6.0
        l_init = 8.0
        w_half = w_init / 2.0
        self.room_twin = {
            "room_width": w_init,
            "room_length": l_init,
            "room_width_m": w_init,
            "room_length_m": l_init,
            "width_m": w_init,
            "length_m": l_init,
            "wall_material": "CONCRETE",
            "wall_attenuation_db": 12.0,
            "reflection_coeff": 0.85,
            "nodes": {
                "tx1": {"x": -2.0, "y": 0.0, "label": "ESP32-TX1 (100 Hz)"},
                "tx2": {"x": 2.0, "y": 0.0, "label": "ESP32-TX2 (75 Hz)"},
                "rx":  {"x": 0.0, "y": 0.0, "label": "ESP32-RX Radar"},
                "ap":  {"x": 0.0, "y": 6.0, "label": "SAM 4G AP"}
            },
            "walls": [
                {"name": "South Wall", "x1": -w_half, "y1": 0.0, "x2": w_half, "y2": 0.0, "material": "CONCRETE"},
                {"name": "North Wall", "x1": -w_half, "y1": l_init, "x2": w_half, "y2": l_init, "material": "CONCRETE"},
                {"name": "West Wall",  "x1": -w_half, "y1": 0.0, "x2": -w_half, "y2": l_init, "material": "CONCRETE"},
                {"name": "East Wall",  "x1": w_half, "y1": 0.0, "x2": w_half, "y2": l_init, "material": "CONCRETE"},
            ],
            "tx1_pos": [-2.0, 0.0],
            "tx2_pos": [2.0, 0.0],
            "rx_pos": [0.0, 0.0],
            "ap_pos": [0.0, 6.0],
            "floorplan_image": None
        }
        self.ghosts_folded_count = 0

        # ---------------------------------------------------------------------
        # 6. Gait Biometrics & Click-to-Lock Vital Signs (BCG/PQRST)
        # ---------------------------------------------------------------------
        self.enroll_duration_sec = 20.0
        self.enrolling_name: Optional[str] = None
        self.enroll_start_time = 0.0
        self.enroll_windows: List[np.ndarray] = []
        self.enroll_last_sample_pkt = 0
        self.enroll_progress_pct = 0
        self.enroll_remaining_sec = 0.0
        self.gait_profiles: Dict[str, dict] = {}

        self.last_valid_hr = 0.0
        self.last_valid_rr = 0.0
        self.last_sqi = 0.0
        self.dual_gate_locked = True
        self.lock_reason = "INITIALIZING SYSTEM"
        self.pqrst_trace = np.zeros(200, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Backward Compatibility Properties for Prior Tests & HUD
    # -------------------------------------------------------------------------
    @property
    def kf_state(self) -> np.ndarray:
        if len(self.tracks) > 0:
            return self.tracks[0].kf_state
        return np.array([0.0, 4.0, 0.0, 0.0], dtype=np.float64)

    @property
    def current_target_id(self) -> str:
        if len(self.tracks) > 0:
            return self.tracks[0].identity_label
        return "NO TARGET"

    @property
    def current_gait_distance(self) -> float:
        if len(self.tracks) > 0:
            return self.tracks[0].mahalanobis_dist
        return 0.0

    # -------------------------------------------------------------------------
    # API & Configuration Methods
    # -------------------------------------------------------------------------
    def set_perimeter_gate(self, max_sensing_radius_m: Optional[float] = None,
                           wall_attenuation_cutoff_db: Optional[float] = None,
                           max_radius_m: Optional[float] = None,
                           wall_cutoff_db: Optional[float] = None, **kwargs):
        """Sets perimeter range gate radius and wall attenuation cutoff."""
        r = max_radius_m if max_radius_m is not None else (max_sensing_radius_m if max_sensing_radius_m is not None else kwargs.get("max_radius_m", 4.0))
        w = wall_cutoff_db if wall_cutoff_db is not None else (wall_attenuation_cutoff_db if wall_attenuation_cutoff_db is not None else kwargs.get("wall_cutoff_db", 12.0))
        self.max_sensing_radius_m = float(np.clip(r, 1.0, 15.0))
        self.wall_attenuation_cutoff_db = float(np.clip(w, 3.0, 30.0))

    def configure_room_twin(self, width_m: Optional[float] = None, length_m: Optional[float] = None,
                            wall_material: str = "CONCRETE",
                            tx1_pos: Optional[List[float]] = None, tx2_pos: Optional[List[float]] = None,
                            rx_pos: Optional[List[float]] = None, ap_pos: Optional[List[float]] = None,
                            room_width_m: Optional[float] = None, room_length_m: Optional[float] = None,
                            room_width: Optional[float] = None, room_length: Optional[float] = None,
                            nodes: Optional[Dict[str, dict]] = None, floorplan_image: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Updates room geometry, wall material, boundary wall segments and hardware node coordinates."""
        candidates_w = [width_m, room_width_m, room_width, kwargs.get("width"), kwargs.get("room_w")]
        w = next((v for v in candidates_w if v is not None), self.room_twin.get("room_width_m", 6.0))
        candidates_l = [length_m, room_length_m, room_length, kwargs.get("length"), kwargs.get("room_l")]
        l = next((v for v in candidates_l if v is not None), self.room_twin.get("room_length_m", 8.0))

        w_val = max(2.0, float(w))
        l_val = max(2.0, float(l))
        w_half = w_val / 2.0

        self.room_twin["room_width"] = w_val
        self.room_twin["room_length"] = l_val
        self.room_twin["room_width_m"] = w_val
        self.room_twin["room_length_m"] = l_val
        self.room_twin["width_m"] = w_val
        self.room_twin["length_m"] = l_val
        self.room_twin["wall_material"] = wall_material

        mat_upper = wall_material.upper()
        if "CONCRETE" in mat_upper:
            self.room_twin["wall_attenuation_db"] = 12.0
            self.room_twin["reflection_coeff"] = 0.85
        elif "BRICK" in mat_upper:
            self.room_twin["wall_attenuation_db"] = 8.0
            self.room_twin["reflection_coeff"] = 0.70
        elif "DRYWALL" in mat_upper:
            self.room_twin["wall_attenuation_db"] = 4.0
            self.room_twin["reflection_coeff"] = 0.40
        else:
            self.room_twin["wall_attenuation_db"] = 3.0
            self.room_twin["reflection_coeff"] = 0.30

        if tx1_pos is not None:
            self.room_twin["tx1_pos"] = tx1_pos
            self.room_twin["nodes"]["tx1"]["x"] = float(tx1_pos[0])
            self.room_twin["nodes"]["tx1"]["y"] = float(tx1_pos[1])
        if tx2_pos is not None:
            self.room_twin["tx2_pos"] = tx2_pos
            self.room_twin["nodes"]["tx2"]["x"] = float(tx2_pos[0])
            self.room_twin["nodes"]["tx2"]["y"] = float(tx2_pos[1])
        if rx_pos is not None:
            self.room_twin["rx_pos"] = rx_pos
            self.room_twin["nodes"]["rx"]["x"] = float(rx_pos[0])
            self.room_twin["nodes"]["rx"]["y"] = float(rx_pos[1])
        if ap_pos is not None:
            self.room_twin["ap_pos"] = ap_pos
            self.room_twin["nodes"]["ap"]["x"] = float(ap_pos[0])
            self.room_twin["nodes"]["ap"]["y"] = float(ap_pos[1])

        if nodes:
            for k, v in nodes.items():
                if k in self.room_twin["nodes"]:
                    self.room_twin["nodes"][k].update(v)

        # Dynamic wall boundary segments
        self.room_twin["walls"] = [
            {"name": "South Wall", "x1": -w_half, "y1": 0.0, "x2": w_half, "y2": 0.0, "material": wall_material},
            {"name": "North Wall", "x1": -w_half, "y1": l_val, "x2": w_half, "y2": l_val, "material": wall_material},
            {"name": "West Wall",  "x1": -w_half, "y1": 0.0, "x2": -w_half, "y2": l_val, "material": wall_material},
            {"name": "East Wall",  "x1": w_half, "y1": 0.0, "x2": w_half, "y2": l_val, "material": wall_material},
        ]

        if floorplan_image:
            self.room_twin["floorplan_image"] = floorplan_image

        return self.room_twin

    def start_room_calibration(self):
        """Starts 5-second silence calibration to establish multipath noise floor."""
        self.is_calibrating = True
        self.cal_buffer_amp.clear()
        self.cal_buffer_rssi.clear()

    def start_gait_enrollment(self, target_name: str, duration_sec: float = 20.0):
        """Starts 20-second multi-centroid gait enrollment for a named profile."""
        self.enrolling_name = target_name.strip().upper()
        self.enroll_duration_sec = duration_sec
        self.enroll_start_time = time.time()
        self.enroll_windows.clear()
        self.enroll_last_sample_pkt = self.pkt_count
        self.enroll_progress_pct = 0
        self.enroll_remaining_sec = self.enroll_duration_sec

    def select_target(self, track_id: int):
        """Locks onto a specific target for spatial vital-sign monitoring. Pass -1 to reset."""
        if track_id <= 0:
            self.selected_target_id = None
        else:
            self.selected_target_id = track_id
        for trk in self.tracks:
            trk.selected = (trk.track_id == self.selected_target_id)

    # -------------------------------------------------------------------------
    # Subsystem A: Frame Ingestion & Multi-MAC Dual-Link Bistatic Routing
    # -------------------------------------------------------------------------
    def ingest_frame(self, csi_iq: np.ndarray, rssi: float, source_mac: str = "00:00:00:00:00:00"):
        """Ingests one packed CSI frame, routes by MAC into bistatic mesh streams & normalizes."""
        now = time.time()

        # 1. Multi-MAC Dual-Link Routing
        mac_lower = source_mac.lower()
        if "37:b5" in mac_lower or "tx2" in mac_lower or "secondary" in mac_lower:
            stream_key = "stream_tx2"
        elif "37:b4" in mac_lower or "tx1" in mac_lower:
            stream_key = "stream_tx1"
        elif "ap" in mac_lower or "router" in mac_lower or "ac:22" in mac_lower:
            stream_key = "stream_ap"
        elif mac_lower.startswith("fc:e8"):
            stream_key = "stream_tx1"
        else:
            # Alternating allocation if multiple unique MACs stream
            stream_key = "stream_tx1" if (self.pkt_count % 2 == 0) else "stream_tx2"

        self.link_streams[stream_key]["pkt_count"] += 1
        self.link_streams[stream_key]["rssi"] = rssi
        self.link_streams[stream_key]["last_seen"] = now

        # 2. De-interleave I & Q
        imag = csi_iq[0::2].astype(np.float64)
        real = csi_iq[1::2].astype(np.float64)
        h_complex = real + 1j * imag

        # 3. L2 AGC Normalization
        raw_amp = np.abs(h_complex)
        l2_norm = np.linalg.norm(raw_amp) + 1e-8
        norm_amp = (raw_amp / l2_norm) * np.sqrt(self.num_sc)

        # 4. Phase unwrapping & linear detrending (SFO/PDD sanitization)
        unwrapped_phase = np.unwrap(np.angle(h_complex))
        sc_denom = float(np.dot(self.sc_indices, self.sc_indices))
        slope = float(np.dot(self.sc_indices, unwrapped_phase)) / sc_denom
        intercept = float(np.mean(unwrapped_phase))
        clean_phase = unwrapped_phase - (slope * self.sc_indices + intercept)

        h_norm = norm_amp * np.exp(1j * clean_phase)

        # 5. Subcarrier CSI-Ratio (delta = 2) -> 50 ratio channels
        csi_ratio = h_norm[:-2] / (h_norm[2:] + 1e-8)

        # 6. Room silence calibration
        if self.is_calibrating:
            self.cal_buffer_amp.append(norm_amp)
            self.cal_buffer_rssi.append(rssi)
            if len(self.cal_buffer_amp) >= int(5.0 * self.fs):
                arr = np.array(self.cal_buffer_amp)
                self.ref_amp = np.mean(arr, axis=0)
                self.ref_std = np.std(arr, axis=0) + 1e-4
                self.ref_noise_energy = float(np.mean(self.ref_std ** 2))
                self.ref_rssi = float(np.mean(self.cal_buffer_rssi))
                self.ref_centroid = float(np.dot(self.sc_indices, self.ref_amp) / (np.sum(self.ref_amp) + 1e-8))
                self.is_calibrating = False

        # 7. Hadamard division (stationary background cancellation)
        hadamard_amp = (norm_amp / (self.ref_amp + 1e-8)) - 1.0

        # 8. Shift ring buffers
        self.amp_ring = np.roll(self.amp_ring, -1, axis=0)
        self.phase_ring = np.roll(self.phase_ring, -1, axis=0)
        self.ratio_ring = np.roll(self.ratio_ring, -1, axis=0)

        self.amp_ring[-1, :] = hadamard_amp
        self.phase_ring[-1, :] = clean_phase
        self.ratio_ring[-1, :] = csi_ratio
        self.rssi_ema = 0.90 * self.rssi_ema + 0.10 * rssi
        self.pkt_count += 1

        # Store in stream buffer
        stream_buf = self.link_streams[stream_key]["buf"]
        stream_buf[:-1, :] = stream_buf[1:, :]
        stream_buf[-1, :] = hadamard_amp

        # 9. Periodic Subsystem Triggers
        self._update_presence_and_tracking()

    # -------------------------------------------------------------------------
    # Subsystem B: Unbounded RMT-MDL Crowd Counting & Eigenvalue Spectrum
    # -------------------------------------------------------------------------
    def _compute_vital_snr(self) -> float:
        """Computes physiological spectral SNR in breathing [0.15, 0.60] & cardiac [0.95, 2.10] Hz."""
        if self.pkt_count < 100:
            return 0.0

        n_samples = min(self.pkt_count, 400)
        sig = np.angle(self.ratio_ring[-n_samples:, :])
        pc = np.mean(sig, axis=1)
        pc = pc - np.mean(pc)

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        psd = np.abs(np.fft.rfft(pc * np.hanning(n_samples))) ** 2

        resp_mask = (freqs >= 0.15) & (freqs <= 0.60)
        cardiac_mask = (freqs >= 0.95) & (freqs <= 2.10)
        noise_mask = (freqs >= 2.50) & (freqs <= 5.00)

        p_resp = np.max(psd[resp_mask]) if np.any(resp_mask) else 0.0
        p_cardiac = np.max(psd[cardiac_mask]) if np.any(cardiac_mask) else 0.0
        p_vital = max(p_resp, p_cardiac)
        if p_vital < 2e-5:
            return 0.0
        p_noise = (np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-8) + 1e-12

        snr_linear = p_vital / p_noise
        if snr_linear <= 1e-4:
            return -20.0
        return float(10.0 * np.log10(snr_linear))

    def estimate_crowd_count(self, custom_H: Optional[np.ndarray] = None) -> Any:
        """
        Unbounded Crowd Counter via Motion-Adaptive RMT Marchenko-Pastur & MDL Criterion.
        Absorbs limb micro-Doppler reflections into dominant eigenvalue via adaptive diagonal loading.
        Returns: CrowdResult(k, sorted_eigenvalues, von_neumann_entropy)
        """
        if custom_H is not None:
            X_dyn = custom_H
            w_len = len(X_dyn)
        else:
            w_len = min(100, self.pkt_count)
            if w_len < 25:
                return CrowdResult(0, [], 0.0)
            X_dyn = self.amp_ring[-w_len:, :]

        X_center = X_dyn - np.mean(X_dyn, axis=0)

        # 1. Wideband Doppler motion variance & Adaptive Diagonal Loading
        sigma_motion_sq = float(np.mean(np.var(X_dyn, axis=0)))
        ref_n = max(1e-6, getattr(self, "ref_noise_energy", 0.005))
        alpha = float(min(0.25, 0.05 * (sigma_motion_sq / ref_n)))

        # Spatial-temporal covariance matrix R_HH with motion-adaptive diagonal loading
        R_HH = (X_center.T @ X_center) / float(max(1, w_len))
        R_HH += alpha * np.eye(R_HH.shape[0])

        eigvals = np.linalg.eigvalsh(R_HH)
        eigvals = np.sort(np.maximum(1e-12, eigvals))[::-1]  # Sort descending

        M = len(eigvals)
        W = max(2, w_len)

        # 2. Von Neumann Eigenvalue Spectral Entropy
        p_eig = eigvals / (np.sum(eigvals) + 1e-12)
        s_eig = float(-np.sum(p_eig * np.log(p_eig + 1e-12)))

        # 3. Dynamic Marchenko-Pastur Noise Ceiling scaled by motion bursts
        gamma = float(M) / float(W)
        sigma_noise_sq = float(np.median(eigvals))
        mp_noise_ceiling = sigma_noise_sq * (1.0 + np.sqrt(gamma)) ** 2
        burst_scale = 1.0 + 2.0 * min(10.0, sigma_motion_sq / ref_n)
        dynamic_ceiling = mp_noise_ceiling * burst_scale

        # 4. Strict Single-Person Dominant Rank-1 Mode Check
        denom_tail = float(np.sum(eigvals[1:])) + 1e-12
        coherence_ratio = float(eigvals[0] / denom_tail)

        # If secondary limb reflections fall below dynamic noise ceiling or coherence dominates
        if coherence_ratio > 1.35 or (len(eigvals) > 1 and eigvals[1] <= dynamic_ceiling):
            return CrowdResult(1, eigvals[:10].tolist(), round(s_eig, 2))

        # 5. Minimum Description Length (MDL) Wax-Kailath Model Order Selection
        mdl_costs = []
        for k in range(0, min(15, M - 1)):
            tail = eigvals[k:]
            geom_mean = np.exp(np.mean(np.log(tail)))
            arith_mean = np.mean(tail) + 1e-12
            ratio = max(1e-15, geom_mean / arith_mean)

            log_lik = - (W - k) * (M - k) * np.log(ratio)
            penalty = 0.5 * k * (2 * M - k) * np.log(W)
            mdl_costs.append(log_lik + penalty)

        k_mdl = int(np.argmin(mdl_costs)) if mdl_costs else 1
        est_count = max(1, k_mdl)

        return CrowdResult(est_count, eigvals[:10].tolist(), round(s_eig, 2))

    # -------------------------------------------------------------------------
    # Subsystem C: Through-Wall & Perimeter Range Gating (CIR Profiling)
    # -------------------------------------------------------------------------
    def _apply_perimeter_range_gating(self, candidate_pos: Any, energy: float = 1.0) -> Tuple[bool, str]:
        """
        Perimeter and Through-Wall Range Gate via IFFT CIR Delay Profiling.
        Suppresses reflections outside max_sensing_radius_m or exceeding wall attenuation.
        Returns: (is_valid, reason)
        """
        self.total_detections_count += 1
        dist_from_origin = float(np.hypot(candidate_pos[0], candidate_pos[1]))

        # 1. Outer perimeter distance check
        if dist_from_origin > self.max_sensing_radius_m:
            self.clutter_suppressed_count += 1
            self.clutter_rejected_pct = round((self.clutter_suppressed_count / max(1, self.total_detections_count)) * 100.0, 1)
            return False, "OUTSIDE_PERIMETER_CLUTTER"

        # 2. Windowed IFFT CIR Delay Profile
        h_spec = self.amp_ring[-1, :] * np.hanning(self.num_sc)
        cir = np.fft.ifft(h_spec, n=64)
        peak_idx = int(np.argmax(np.abs(cir)))

        # Excess multipath delay beyond direct LoS
        excess_delay_bins = abs(peak_idx - 16)
        approx_excess_attenuation_db = excess_delay_bins * 1.8

        if approx_excess_attenuation_db > self.wall_attenuation_cutoff_db and energy < 0.8 * self.ref_noise_energy:
            self.clutter_suppressed_count += 1
            self.clutter_rejected_pct = round((self.clutter_suppressed_count / max(1, self.total_detections_count)) * 100.0, 1)
            return False, "OUTSIDE_PERIMETER_CLUTTER"

        self.clutter_rejected_pct = round((self.clutter_suppressed_count / max(1, self.total_detections_count)) * 100.0, 1)
        return True, "IN_ROOM"

    # -------------------------------------------------------------------------
    # Subsystem D: Panorama-Guided RF Digital Twin & Specular Multipath Ghost Folding
    # -------------------------------------------------------------------------
    def _fold_specular_multipath_ghost(self, raw_pos: Any) -> Tuple[np.ndarray, bool]:
        """
        Folds 1st-order wall reflections back inside the room polygon using mirror-image ray tracing.
        Prevents ghost duplicate targets from appearing across boundaries.
        Returns: (folded_pos, is_ghost)
        """
        w_half = self.room_twin["room_width_m"] / 2.0
        l_room = self.room_twin["room_length_m"]

        x = float(raw_pos[0])
        y = float(raw_pos[1])
        folded = False

        # East / West Wall Mirror Reflection Folding
        if x > w_half:
            x = 2.0 * w_half - x
            folded = True
        elif x < -w_half:
            x = -2.0 * w_half - x
            folded = True

        # North / South Wall Mirror Reflection Folding
        if y > l_room:
            y = 2.0 * l_room - y
            folded = True
        elif y < 0.2:
            y = abs(y) + 0.4
            folded = True

        if folded:
            self.ghosts_folded_count += 1

        return np.array([x, y], dtype=np.float64), folded

    # -------------------------------------------------------------------------
    # Presence & Unbounded Tracking Execution Loop
    # -------------------------------------------------------------------------
    def _update_presence_and_tracking(self):
        """Executes Subsystem A-D: Presence, RMT-MDL Counting, Gating & Digital Twin Tracking."""
        if self.pkt_count < 25:
            return

        recent_amp = self.amp_ring[-25:, :]
        self.instant_motion_energy = float(np.mean(np.var(recent_amp, axis=0)))
        self.instant_vital_snr = self._compute_vital_snr()

        # Macro-Doppler energy in [1.0, 15.0] Hz (detects human walking vs stationary chest breathing)
        w_dop = min(100, self.pkt_count)
        recent_sig = np.mean(self.amp_ring[-w_dop:, :], axis=1)
        recent_sig_detrend = recent_sig - np.mean(recent_sig)
        f_axis = np.fft.rfftfreq(len(recent_sig_detrend), d=self.dt)
        psd_motion = np.abs(np.fft.rfft(recent_sig_detrend * np.hanning(len(recent_sig_detrend)))) ** 2
        macro_mask = (f_axis >= 1.0) & (f_axis <= 15.0)
        macro_doppler_e = float(np.sum(psd_motion[macro_mask])) if np.any(macro_mask) else 0.0

        motion_thresh = 2.2 * self.ref_noise_energy
        vital_snr_thresh = 3.5  # dB

        is_moving = (self.instant_motion_energy >= motion_thresh) or (macro_doppler_e > 0.001)
        self.is_motion_active = is_moving
        has_vital = self.instant_vital_snr >= vital_snr_thresh
        has_stationary_hold = any(
            t.status in ("STATIONARY_LOCKED", "STATIC_VITAL") and t.presence_hold_remaining > 0
            for t in self.tracks
        )

        # --- Rule 1: Empty Room ---
        if not is_moving and not has_vital and not has_stationary_hold:
            self.empty_room_frames += 1
            if self.empty_room_frames >= int(2.5 * self.fs):
                self.presence_state = "EMPTY_ROOM"
                self.tracks.clear()
                self.selected_target_id = None
                self.dual_gate_locked = True
                self.lock_reason = "NO SUBJECT IN ROOM"
                self.estimated_crowd_count = 0
                return
        else:
            self.empty_room_frames = 0

        # --- Rule 2: Static Vital Presence (Green Point) ---
        if not is_moving and (has_vital or has_stationary_hold):
            self.presence_state = "STATIC_VITAL"
            if len(self.tracks) == 0:
                init_x = 0.0
                init_y = 4.0
                new_track = TargetTrack(self.next_track_id, np.array([init_x, init_y]), dt=self.dt)
                self.next_track_id += 1
                new_track.status = "STATIC_VITAL"
                new_track.pinned_pos = np.array([init_x, init_y])
                new_track.identity_label = "STATIC SUBJECT"
                new_track.presence_hold_remaining = 60.0
                self.tracks.append(new_track)
                self.estimated_crowd_count = 1
            else:
                if len(self.tracks) > 1:
                    self.tracks = [self.tracks[0]]
                self.tracks[0].status = "STATIC_VITAL"
                self.tracks[0].identity_label = "STATIC SUBJECT"
                self.tracks[0].missed_count = 0
                self.tracks[0].kf_state[2:] = 0.0
                self.tracks[0].pinned_pos = np.copy(self.tracks[0].kf_state[:2])
                self.tracks[0].presence_hold_remaining = 60.0
                self.estimated_crowd_count = 1
            return

        # --- Rule 3: Active Motion / Unbounded Crowd Tracking ---
        for trk in self.tracks:
            trk.predict()

        # Execute RMT-MDL Crowd Counting & Multi-Target Localization every 10 frames (100 ms)
        if self.pkt_count % 10 == 0:
            crowd_k, eigvals, s_entropy = self.estimate_crowd_count()
            self.eigenvalue_spectrum = eigvals
            self.von_neumann_entropy = s_entropy

            candidate_obs, comp_weights = self._extract_multipath_candidates(crowd_k)

            # Temporal Hysteresis on Crowd Count
            now = time.time()
            if self.estimated_crowd_count == 0 and crowd_k >= 1:
                self.estimated_crowd_count = crowd_k
                self.pending_crowd_count = crowd_k
                self.crowd_hysteresis_start_time = now
            elif crowd_k > self.estimated_crowd_count:
                has_spatial_sep = False
                if len(candidate_obs) >= 2:
                    min_sep = min(
                        float(np.linalg.norm(candidate_obs[i] - candidate_obs[j]))
                        for i in range(len(candidate_obs))
                        for j in range(i + 1, len(candidate_obs))
                    )
                    has_spatial_sep = (min_sep > 1.0)

                if has_spatial_sep:
                    if crowd_k != self.pending_crowd_count:
                        self.pending_crowd_count = crowd_k
                        self.crowd_hysteresis_start_time = now
                    elif (now - self.crowd_hysteresis_start_time) >= 2.5:
                        self.estimated_crowd_count = crowd_k
                else:
                    self.pending_crowd_count = self.estimated_crowd_count
            elif crowd_k < self.estimated_crowd_count:
                if crowd_k != self.pending_crowd_count:
                    self.pending_crowd_count = crowd_k
                    self.crowd_hysteresis_start_time = now
                elif (now - self.crowd_hysteresis_start_time) >= 1.5:
                    self.estimated_crowd_count = crowd_k
            else:
                self.pending_crowd_count = crowd_k
                self.crowd_hysteresis_start_time = now

            self._associate_and_update_tracks(candidate_obs, comp_weights, self.estimated_crowd_count)

        self.presence_state = "ACTIVE_MOTION" if self.estimated_crowd_count <= 1 else "ACTIVE_MULTI"

    def _extract_multipath_candidates(self, crowd_k: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extracts spatial 2D candidate positions using dual-link bistatic sub-bands,
        applies perimeter range gating, folds multipath mirror reflections, and
        clusters torso/limb peaks within r_merge = 0.85m into unified weighted centroids.
        """
        raw_candidates: List[Tuple[np.ndarray, np.ndarray, float]] = []

        recent_dyn = self.amp_ring[-100:, :]
        ratio_angles = np.angle(self.ratio_ring[-20:, :])
        phase_dispersion = float(np.std(ratio_angles))
        delta_p = max(-6.0, min(20.0, self.ref_rssi - self.rssi_ema))

        # Dynamic sub-band slicing based on crowd count
        num_slices = max(1, min(6, crowd_k))
        band_size = 52 // num_slices

        for s in range(num_slices):
            sb_idx = slice(s * band_size, min(52, (s + 1) * band_size + 4))
            sub_sig = recent_dyn[:, sb_idx]
            sub_energy = float(np.mean(sub_sig ** 2))

            # Subcarrier sensitivity weights
            w = np.zeros(52, dtype=np.float64)
            w[sb_idx] = np.std(sub_sig, axis=0) + 1e-4
            w /= (np.sum(w) + 1e-8)

            spec_centroid = float(np.dot(self.sc_indices, w) / (np.sum(w) + 1e-8))
            x_raw = 5.0 * np.tanh(2.4 * (spec_centroid - self.ref_centroid) / 26.0)
            d_reflect = 1.5 * (10.0 ** (delta_p / (10.0 * 2.6))) + 1.85 * phase_dispersion
            y_raw = float(np.clip(np.sqrt(max(0.25, d_reflect ** 2 - x_raw ** 2)), 0.5, 9.8))

            raw_pos = np.array([x_raw, y_raw], dtype=np.float64)

            # 1. Subsystem D: Digital Twin Multipath Wall-Ghost Folding
            folded_pos, _ = self._fold_specular_multipath_ghost(raw_pos)

            # 2. Subsystem C: Perimeter Range Gating
            is_valid, reason = self._apply_perimeter_range_gating(folded_pos, sub_energy)
            if not is_valid:
                continue

            raw_candidates.append((folded_pos, w[:50] / (np.linalg.norm(w[:50]) + 1e-8), sub_energy))

        # Spatial-Biometric Clustering Gate: Merge peaks within r_merge = 0.85m
        clustered_pos: List[np.ndarray] = []
        clustered_weights: List[np.ndarray] = []
        cluster_energies: List[float] = []

        for cand_pos, w_vec, eng in raw_candidates:
            merged = False
            for c_idx in range(len(clustered_pos)):
                dist = float(np.linalg.norm(cand_pos - clustered_pos[c_idx]))
                if dist <= 0.85:
                    e_prev = cluster_energies[c_idx]
                    e_tot = e_prev + eng + 1e-8
                    clustered_pos[c_idx] = (e_prev * clustered_pos[c_idx] + eng * cand_pos) / e_tot
                    clustered_weights[c_idx] = (e_prev * clustered_weights[c_idx] + eng * w_vec) / e_tot
                    cluster_energies[c_idx] = e_tot
                    merged = True
                    break
            if not merged:
                clustered_pos.append(cand_pos)
                clustered_weights.append(w_vec)
                cluster_energies.append(eng)

        # Guarantee: if crowd_k == 1, force candidates to be exactly 1 dominant target
        if crowd_k == 1 and len(clustered_pos) > 1:
            best_idx = int(np.argmax(cluster_energies))
            clustered_pos = [clustered_pos[best_idx]]
            clustered_weights = [clustered_weights[best_idx]]

        return clustered_pos, clustered_weights

    def _associate_and_update_tracks(self, candidate_obs: List[np.ndarray], comp_weights: List[np.ndarray], target_cap: int):
        """Associates candidate observations with tracks via Hungarian algorithm, scaled by crowd count."""
        n_tracks = len(self.tracks)
        n_obs = len(candidate_obs)

        if n_tracks == 0 and n_obs > 0:
            for obs, w in zip(candidate_obs[:target_cap], comp_weights[:target_cap]):
                trk = TargetTrack(self.next_track_id, obs, dt=self.dt)
                self.next_track_id += 1
                trk.spatial_subcarrier_weights = w
                trk.status = "MOVING"
                trk.identity_label = "UNKNOWN TARGET"
                self.tracks.append(trk)
            return

        if n_obs == 0:
            for trk in self.tracks:
                if trk.status in ("STATIONARY_LOCKED", "STATIC_VITAL"):
                    if self.instant_vital_snr >= 3.5 or trk.presence_hold_remaining > 0:
                        trk.presence_hold_remaining = max(0.0, trk.presence_hold_remaining - 0.1)
                        trk.missed_count = 0
                        continue
                trk.missed_count += 1
            self.tracks = [t for t in self.tracks if t.missed_count <= 25 or (t.status in ("STATIONARY_LOCKED", "STATIC_VITAL") and t.presence_hold_remaining > 0)]
            return

        # Cost matrix: Euclidean distance
        cost_matrix = np.zeros((n_tracks, n_obs), dtype=np.float64)
        for i, trk in enumerate(self.tracks):
            pos_pred = trk.kf_state[:2]
            for j, obs in enumerate(candidate_obs):
                cost_matrix[i, j] = float(np.linalg.norm(pos_pred - obs))

        # Hungarian algorithm association
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_obs = set()
        is_active_motion = getattr(self, "is_motion_active", False)

        for r, c in zip(row_ind, col_ind):
            trk = self.tracks[r]
            gate_thresh = 0.85 if trk.status in ("STATIONARY_LOCKED", "STATIC_VITAL") else 2.2
            if cost_matrix[r, c] < gate_thresh:
                trk.update(candidate_obs[c], is_motion_active=is_active_motion)
                trk.spatial_subcarrier_weights = comp_weights[c]
                assigned_tracks.add(r)
                assigned_obs.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                if trk.status in ("STATIONARY_LOCKED", "STATIC_VITAL"):
                    if self.instant_vital_snr >= 3.5 or trk.presence_hold_remaining > 0:
                        trk.presence_hold_remaining = max(0.0, trk.presence_hold_remaining - 0.1)
                        trk.missed_count = 0
                        continue
                trk.missed_count += 1

        # Spawn new tracks up to unbounded target_cap with spatial separation > 0.85m
        for j in range(n_obs):
            if j not in assigned_obs and len(self.tracks) < target_cap:
                obs = candidate_obs[j]
                min_dist_to_existing = min([float(np.linalg.norm(t.kf_state[:2] - obs)) for t in self.tracks], default=999.0)
                if min_dist_to_existing > 0.85:
                    new_trk = TargetTrack(self.next_track_id, obs, dt=self.dt)
                    self.next_track_id += 1
                    new_trk.spatial_subcarrier_weights = comp_weights[j]
                    new_trk.status = "MOVING"
                    new_trk.identity_label = "UNKNOWN TARGET"
                    new_trk.locked_identity = None
                    self.tracks.append(new_trk)

        # Enforce exact 1-target track count when crowd_k == 1
        if target_cap == 1 and len(self.tracks) > 1:
            self.tracks.sort(key=lambda t: t.missed_count)
            self.tracks = [self.tracks[0]]

        # Prune stale tracks
        self.tracks = [t for t in self.tracks if t.missed_count <= 25 or (t.status in ("STATIONARY_LOCKED", "STATIC_VITAL") and t.presence_hold_remaining > 0)]

        # Lock spatial subcarrier sensitivity when target transitions to quasi-stationary
        for trk in self.tracks:
            if trk.speed < 0.15:
                var_r = np.var(self.ratio_ring[-100:, :], axis=0)
                norm_v = np.linalg.norm(var_r) + 1e-8
                trk.spatial_subcarrier_weights = var_r / norm_v

    # -------------------------------------------------------------------------
    # Subsystem: 20-Second Gait Biometrics & Open-Set Classification
    # -------------------------------------------------------------------------
    def extract_24_gait_features(self, window_len: int = 100, custom_sig: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts 24-dimensional micro-Doppler kinematic & spectral features."""
        if custom_sig is not None:
            pc1 = custom_sig[-window_len:]
        else:
            seg_amp = self.amp_ring[-window_len:, :]
            u, s, _ = np.linalg.svd(seg_amp - np.mean(seg_amp, axis=0), full_matrices=False)
            pc1 = u[:, 0] * s[0]

        f_stft, _, Zxx = signal.stft(pc1, fs=self.fs, window='hann', nperseg=32, noverlap=24)
        S_mag = np.abs(Zxx) + 1e-8
        S_norm = S_mag / np.sum(S_mag, axis=0, keepdims=True)

        # 1-4: Torso Velocity Envelope
        f_centroid = np.sum(f_stft[:, None] * S_norm, axis=0)
        v_torso = 0.5 * self.wavelength * f_centroid
        f1 = float(np.mean(v_torso))
        f2 = float(np.std(v_torso))
        f3 = float(np.percentile(v_torso, 95))
        f4 = float(f3 / (f1 + 1e-5))

        # 5-8: Limb Micro-Doppler Dynamics
        f_bw = np.sqrt(np.sum(((f_stft[:, None] - f_centroid[None, :]) ** 2) * S_norm, axis=0))
        v_limb = 0.5 * self.wavelength * f_bw
        f5 = float(np.mean(v_limb))
        f6 = float(np.std(v_limb))
        f7 = float(np.max(v_limb))
        f8 = float(f5 / (f1 + 1e-5))

        # 9-12: Step Cadence & Symmetry
        env = pc1 ** 2
        env_zero = env - np.mean(env)
        autocorr = np.correlate(env_zero, env_zero, mode='full')[len(env_zero) - 1:]
        autocorr /= (autocorr[0] + 1e-8)
        min_lag, max_lag = int(0.4 * self.fs), min(len(autocorr) - 1, int(1.4 * self.fs))
        peak_lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag])) if max_lag > min_lag else min_lag
        f9 = float(self.fs / (peak_lag + 1e-5))
        f10 = float(autocorr[peak_lag]) if peak_lag < len(autocorr) else 0.5
        half_lag = max(1, peak_lag // 2)
        f11 = float(autocorr[half_lag]) if half_lag < len(autocorr) else 0.3
        f12 = float(np.std(env))

        # 13-16: Higher Spectral Moments
        psd = np.mean(S_mag ** 2, axis=1)
        psd_prob = psd / (np.sum(psd) + 1e-8)
        mean_f = np.sum(f_stft * psd_prob)
        std_f = np.sqrt(np.sum(((f_stft - mean_f) ** 2) * psd_prob) + 1e-8)
        f13 = float(np.sum(((f_stft - mean_f) / std_f) ** 3 * psd_prob))
        f14 = float(np.sum(((f_stft - mean_f) / std_f) ** 4 * psd_prob))
        f15 = float(-np.sum(psd_prob * np.log2(psd_prob + 1e-12)))
        f16 = float(np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-8))

        # 17-22: 6 Doppler Band Energy Ratios
        bands = [(1, 4), (4, 8), (8, 12), (12, 18), (18, 25), (25, 45)]
        band_feats = []
        total_p = np.sum(psd) + 1e-8
        for b_low, b_high in bands:
            mask = (f_stft >= b_low) & (f_stft < b_high)
            band_feats.append(float(np.sum(psd[mask]) / total_p))

        # 23-24: Spatial Correlation & Phase Dispersion
        seg_amp = self.amp_ring[-window_len:, :]
        corr_mat = np.corrcoef(seg_amp[:, :12].T)
        f23 = float(np.nanmean(np.abs(corr_mat)))
        seg_ratio = self.ratio_ring[-window_len:, :]
        f24 = float(np.mean(np.std(np.angle(seg_ratio), axis=0)))

        return np.array([
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
            f13, f14, f15, f16, *band_feats, f23, f24
        ], dtype=np.float64)

    def update_gait_biometrics(self):
        """Processes 20s enrollment or performs Hungarian multi-target open-set identification."""
        if self.pkt_count < 100:
            return

        now = time.time()

        # A. Live 20-Second Gait Enrollment
        if self.enrolling_name is not None:
            elapsed = now - self.enroll_start_time
            self.enroll_progress_pct = int(min(100, (elapsed / self.enroll_duration_sec) * 100))
            self.enroll_remaining_sec = max(0.0, round(self.enroll_duration_sec - elapsed, 1))

            if self.pkt_count - self.enroll_last_sample_pkt >= 25 or len(self.enroll_windows) < 5:
                feat = self.extract_24_gait_features(window_len=100)
                if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                    self.enroll_windows.append(feat)
                self.enroll_last_sample_pkt = self.pkt_count

            if elapsed >= self.enroll_duration_sec or len(self.enroll_windows) >= 40:
                if len(self.enroll_windows) >= 5:
                    X_enroll = np.array(self.enroll_windows)
                    k = min(2, len(X_enroll))
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_enroll)
                    centroids = [kmeans.cluster_centers_[i] for i in range(k)]
                    centroids.append(np.mean(X_enroll, axis=0))

                    try:
                        lw = LedoitWolf().fit(X_enroll)
                        reg_cov = lw.covariance_
                    except Exception:
                        emp_cov = np.cov(X_enroll, rowvar=False) + np.eye(24) * 1e-4
                        shr = 0.10
                        trace_avg = np.trace(emp_cov) / 24.0
                        reg_cov = (1.0 - shr) * emp_cov + shr * trace_avg * np.eye(24)

                    inv_cov = np.linalg.pinv(reg_cov)
                    dists = []
                    for x in X_enroll:
                        d_min = min(np.sqrt(max(0.0, (x - c) @ inv_cov @ (x - c).T)) for c in centroids)
                        dists.append(d_min)
                    tau_gate = max(6.84, float(np.mean(dists) + 3.2 * np.std(dists)))

                    self.gait_profiles[self.enrolling_name] = {
                        'centroids': centroids,
                        'inv_cov': inv_cov,
                        'threshold': tau_gate
                    }
                self.enrolling_name = None
                self.enroll_progress_pct = 100
                self.enroll_remaining_sec = 0.0
            return

        # B. Multi-Target Open-Set Identification with Strict Track-ID Binding
        if len(self.tracks) == 0:
            return

        if not self.gait_profiles:
            for trk in self.tracks:
                if trk.status in ("STATIC_VITAL", "STATIONARY_LOCKED"):
                    trk.identity_label = "STATIC SUBJECT"
                else:
                    trk.identity_label = "UNKNOWN TARGET"
                trk.mahalanobis_dist = 99.9
            return

        # Map of profiles locked to active tracks
        currently_locked_profiles = {
            t.locked_identity: t.track_id
            for t in self.tracks
            if t.locked_identity is not None
        }

        active_tracks = [t for t in self.tracks if t.missed_count <= 5 or t.status in ("STATIC_VITAL", "STATIONARY_LOCKED")]
        if len(active_tracks) == 0:
            return

        # Moving unassigned tracks seeking biometric verification
        unassigned_moving_tracks = [
            t for t in active_tracks
            if t.locked_identity is None and t.status == "MOVING"
        ]

        # For stationary tracks or already-locked tracks: retain their verified identity
        for trk in active_tracks:
            if trk.locked_identity is not None:
                trk.identity_label = trk.locked_identity
            elif trk.status in ("STATIC_VITAL", "STATIONARY_LOCKED"):
                trk.identity_label = "STATIC SUBJECT"

        # If there are moving unassigned tracks, evaluate against free profiles
        free_profiles = [
            name for name in self.gait_profiles.keys()
            if name not in currently_locked_profiles
        ]

        if not unassigned_moving_tracks or not free_profiles:
            for trk in unassigned_moving_tracks:
                trk.identity_label = "UNKNOWN TARGET"
                trk.mahalanobis_dist = 99.9
            return

        # Hungarian matching between unassigned moving tracks and free profiles
        dist_matrix = np.full((len(unassigned_moving_tracks), len(free_profiles)), 99.9, dtype=np.float64)

        for i, trk in enumerate(unassigned_moving_tracks):
            feat = self.extract_24_gait_features(window_len=100)
            for j, name in enumerate(free_profiles):
                prof = self.gait_profiles[name]
                d_c = min(np.sqrt(max(0.0, (feat - c) @ prof['inv_cov'] @ (feat - c).T)) for c in prof['centroids'])
                dist_matrix[i, j] = d_c

        row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
        matched_tracks = set()

        for r, c in zip(row_ind, col_ind):
            d_val = float(dist_matrix[r, c])
            trk = unassigned_moving_tracks[r]
            trk.mahalanobis_dist = round(d_val, 2)
            name = free_profiles[c]
            thresh = self.gait_profiles[name]['threshold']

            if d_val <= thresh:
                trk.identity_label = name
                trk.locked_identity = name
                currently_locked_profiles[name] = trk.track_id
            else:
                trk.identity_label = "UNKNOWN TARGET"
                trk.locked_identity = None
            matched_tracks.add(r)

        for i, trk in enumerate(unassigned_moving_tracks):
            if i not in matched_tracks:
                trk.identity_label = "UNKNOWN TARGET"
                trk.locked_identity = None
                trk.mahalanobis_dist = 99.9

    # -------------------------------------------------------------------------
    # Subsystem: Click-to-Lock Spatial Vital Signs (SVD + VMD + Coiflet-4 DWT)
    # -------------------------------------------------------------------------
    def compute_vital_signs_bcg(self):
        """Extracts Respiration, Heart Rate, and PQRST waveform with click-to-lock spatial isolation."""
        if self.pkt_count < min(self.win_size, 200):
            self.dual_gate_locked = True
            self.lock_reason = "FILLING CSI RING BUFFER"
            return

        active = [t for t in self.tracks if t.missed_count <= 5 or t.status in ("STATIC_VITAL", "STATIONARY_LOCKED")]

        # Case 1: No targets
        if len(active) == 0:
            self.dual_gate_locked = True
            self.lock_reason = "NO SUBJECT IN ROOM"
            self.last_valid_hr = 0.0
            self.last_valid_rr = 0.0
            self.last_sqi = 0.0
            self.pqrst_trace = np.zeros(200, dtype=np.float64)
            return

        # Case 2: Exactly 1 target -> Auto-lock
        if len(active) == 1:
            target = active[0]
            self.selected_target_id = target.track_id
            target.selected = True
        else:
            # Case 3: Multiple targets -> Require explicit target selection
            if self.selected_target_id is None or self.selected_target_id not in [t.track_id for t in active]:
                self.dual_gate_locked = True
                self.lock_reason = "MULTI-TARGET DETECTED: CLICK A TARGET ON RADAR"
                self.pqrst_trace = np.zeros(200, dtype=np.float64)
                return

            target = next(t for t in active if t.track_id == self.selected_target_id)

            if target.speed >= 0.15:
                self.dual_gate_locked = True
                self.lock_reason = "SELECTED TARGET MOVING"
                return

            for other in active:
                if other.track_id != target.track_id:
                    other_dist = np.linalg.norm(other.kf_state[:2] - target.kf_state[:2])
                    if other.speed >= 0.15 and other_dist < 2.0:
                        self.dual_gate_locked = True
                        self.lock_reason = "INTERFERENCE FROM OTHER MOVING TARGET"
                        return

        if target.speed >= 0.15 and self.instant_motion_energy > 0.085:
            self.dual_gate_locked = True
            self.lock_reason = "GATE 1 LOCKED: SUBJECT MOVING"
            return

        # Spatial Subcarrier Weighting for the Locked Target
        w_spatial = target.spatial_subcarrier_weights
        w_norm = w_spatial / (np.mean(w_spatial) + 1e-8)
        R_mat = self.ratio_ring * w_norm[np.newaxis, :]

        # 1. Rotational Projection
        proj_signals = np.zeros((self.win_size, 50), dtype=np.float64)
        hsr_scores = np.zeros(50, dtype=np.float64)
        freqs = np.fft.rfftfreq(self.win_size, d=1.0 / self.fs)
        hr_mask = (freqs >= 0.95) & (freqs <= 2.10)
        tot_mask = (freqs >= 0.15) & (freqs <= 4.00)

        for k in range(50):
            real_p = np.real(R_mat[:, k])
            imag_p = np.imag(R_mat[:, k])
            cov_iq = np.cov(real_p, imag_p)
            eigvals, eigvecs = np.linalg.eigh(cov_iq)
            v_opt = eigvecs[:, -1]
            sig_k = real_p * v_opt[0] + imag_p * v_opt[1]
            sig_k = sig_k - np.mean(sig_k)
            proj_signals[:, k] = sig_k

            spec_k = np.abs(np.fft.rfft(sig_k)) ** 2
            hsr_scores[k] = np.max(spec_k[hr_mask]) / (np.sum(spec_k[tot_mask]) + 1e-8)

        # 2. HSR Top-12 Subcarrier Selection & SVD PCA
        top_idx = np.argsort(hsr_scores)[-12:]
        selected_mat = proj_signals[:, top_idx]
        u_svd, s_svd, _ = np.linalg.svd(selected_mat, full_matrices=False)
        vital_pc1 = u_svd[:, 0] * s_svd[0]

        b_all, a_all = signal.butter(4, [0.14 / (0.5 * self.fs), 3.5 / (0.5 * self.fs)], btype='band')
        vital_filtered = signal.filtfilt(b_all, a_all, vital_pc1)

        # 3. Variational Mode Decomposition (VMD K=4, alpha=2000)
        try:
            u_modes, _, omega = VMD(vital_filtered, alpha=2000, tau=0.0, K=4, DC=0, init=1, tol=1e-6)
            final_freqs = omega[-1, :] * self.fs
        except Exception:
            u_modes = np.zeros((4, self.win_size))
            final_freqs = np.array([0.25, 0.50, 1.20, 2.40])

        resp_mode = np.zeros(self.win_size)
        heart_mode = np.zeros(self.win_size)

        for idx_m, f_center in enumerate(final_freqs):
            if 0.14 <= f_center <= 0.65:
                resp_mode += u_modes[idx_m, :]
            elif 0.90 <= f_center <= 2.15:
                heart_mode += u_modes[idx_m, :]

        if np.std(resp_mode) < 1e-5:
            b_r, a_r = signal.butter(3, [0.15 / (0.5 * self.fs), 0.60 / (0.5 * self.fs)], btype='band')
            resp_mode = signal.filtfilt(b_r, a_r, vital_filtered)
        if np.std(heart_mode) < 1e-5:
            b_h, a_h = signal.butter(4, [0.95 / (0.5 * self.fs), 2.10 / (0.5 * self.fs)], btype='band')
            heart_mode = signal.filtfilt(b_h, a_h, vital_filtered)

        # High-Resolution Respiration Rate
        n_pad = 4096
        resp_spec_pad = np.abs(np.fft.rfft(resp_mode * np.hanning(self.win_size), n=n_pad)) ** 2
        freqs_pad = np.fft.rfftfreq(n_pad, d=1.0 / self.fs)
        resp_mask_pad = (freqs_pad >= 0.15) & (freqs_pad <= 0.60)
        est_rr_hz = float(freqs_pad[resp_mask_pad][np.argmax(resp_spec_pad[resp_mask_pad])])
        rr_bpm = round(est_rr_hz * 60.0, 1)

        # 4. Mechanical PQRST via Coiflet-4 DWT
        diff_ratio = np.abs(np.diff(selected_mat[-201:, :], axis=0))
        wicg_pulse = np.mean(diff_ratio, axis=1)
        composite_cardiac = 0.65 * (heart_mode[-200:] / (np.std(heart_mode[-200:]) + 1e-8)) + \
                            0.35 * ((wicg_pulse - np.mean(wicg_pulse)) / (np.std(wicg_pulse) + 1e-8))

        coeffs = pywt.wavedec(composite_cardiac, 'coif4', level=2)
        for i in range(1, len(coeffs)):
            sigma_mad = np.median(np.abs(coeffs[i])) / 0.6745
            uthresh = sigma_mad * np.sqrt(2.0 * np.log(len(composite_cardiac))) * 0.45
            coeffs[i] = pywt.threshold(coeffs[i], value=uthresh, mode='soft')
        pqrst_clean = pywt.waverec(coeffs, 'coif4')[:200]

        # High-Resolution Heart Rate
        hr_spec_pad = np.abs(np.fft.rfft(heart_mode * np.hanning(self.win_size), n=n_pad)) ** 2
        hr_mask_pad = (freqs_pad >= 0.95) & (freqs_pad <= 2.10)
        f_hr_candidates = freqs_pad[hr_mask_pad]
        p_hr_candidates = hr_spec_pad[hr_mask_pad].copy()

        for harm in (2.0 * est_rr_hz, 3.0 * est_rr_hz):
            harm_idx = np.abs(f_hr_candidates - harm) < 0.08
            p_hr_candidates[harm_idx] *= 0.25

        best_hr_hz = float(f_hr_candidates[np.argmax(p_hr_candidates)])
        hr_bpm = round(best_hr_hz * 60.0, 1)

        # SQI calculation
        hr_spec_unpadded = np.abs(np.fft.rfft(heart_mode * np.hanning(self.win_size))) ** 2
        band_conc = float(np.max(hr_spec_unpadded[hr_mask]) / (np.sum(hr_spec_unpadded[tot_mask]) + 1e-8))

        peaks, _ = signal.find_peaks(pqrst_clean, distance=int(0.45 * self.fs), prominence=0.35)
        if len(peaks) >= 3:
            ibis = np.diff(peaks) / self.fs
            ibi_stability = float(np.exp(-(np.std(ibis) / (np.mean(ibis) + 1e-5)) * 4.0))
        else:
            ibi_stability = 0.40

        sqi = float(np.clip(0.6 * min(1.0, band_conc * 3.5) + 0.4 * ibi_stability, 0.0, 1.0))
        self.last_sqi = round(sqi, 2)

        if sqi < 0.42:
            self.dual_gate_locked = True
            self.lock_reason = f"GATE 2 LOCKED: LOW SQI ({self.last_sqi})"
            return

        self.dual_gate_locked = False
        target_label = target.identity_label if target.identity_label != "UNKNOWN TARGET" else f"TRACK #{target.track_id}"
        self.lock_reason = f"BIOMETRIC LOCK ENGAGED [{target_label}]"
        self.last_valid_rr = rr_bpm if self.last_valid_rr == 0 else round(0.75 * self.last_valid_rr + 0.25 * rr_bpm, 1)
        self.last_valid_hr = hr_bpm if self.last_valid_hr == 0 else round(0.75 * self.last_valid_hr + 0.25 * hr_bpm, 1)
        self.pqrst_trace = pqrst_clean

    # -------------------------------------------------------------------------
    # Telemetry State Serialization
    # -------------------------------------------------------------------------
    def get_telemetry_state(self) -> dict:
        """Serializes complete v4.0 multi-link, crowd count, digital twin & vitals telemetry."""
        active_tracks = [t for t in self.tracks if t.missed_count <= 5 or t.status in ("STATIC_VITAL", "STATIONARY_LOCKED")]

        locked_tid = -1
        locked_tname = "NONE"
        if not self.dual_gate_locked and self.selected_target_id is not None:
            locked_tid = self.selected_target_id
            for t in active_tracks:
                if t.track_id == locked_tid:
                    locked_tname = t.identity_label
                    break

        return {
            "presence_state": self.presence_state,
            "crowd": {
                "estimated_count": self.estimated_crowd_count,
                "von_neumann_entropy": self.von_neumann_entropy,
                "eigenvalues": [round(float(v), 4) for v in self.eigenvalue_spectrum]
            },
            "mesh_status": {
                "tx1_active": (time.time() - self.link_streams["stream_tx1"]["last_seen"]) < 2.0,
                "tx2_active": (time.time() - self.link_streams["stream_tx2"]["last_seen"]) < 2.0,
                "ap_active":  (time.time() - self.link_streams["stream_ap"]["last_seen"]) < 2.0,
                "tx1_rssi": round(self.link_streams["stream_tx1"]["rssi"], 1),
                "tx2_rssi": round(self.link_streams["stream_tx2"]["rssi"], 1),
            },
            "perimeter_gate": {
                "max_radius_m": self.max_sensing_radius_m,
                "wall_cutoff_db": self.wall_attenuation_cutoff_db,
                "clutter_rejected_pct": self.clutter_rejected_pct,
                "ghosts_folded_count": self.ghosts_folded_count
            },
            "room_twin": self.room_twin,
            "enrolling": {
                "active": self.enrolling_name is not None,
                "name": self.enrolling_name or "",
                "progress_pct": self.enroll_progress_pct,
                "remaining_sec": self.enroll_remaining_sec,
                "valid_windows": len(self.enroll_windows)
            },
            "targets": [
                {
                    "track_id": trk.track_id,
                    "x": round(float(trk.kf_state[0]), 2),
                    "y": round(float(trk.kf_state[1]), 2),
                    "vx": round(float(trk.kf_state[2]), 2),
                    "vy": round(float(trk.kf_state[3]), 2),
                    "identity": trk.identity_label,
                    "mahalanobis": trk.mahalanobis_dist,
                    "status": trk.status,
                    "selected": (trk.track_id == self.selected_target_id),
                    "trail": trk.trail
                }
                for trk in active_tracks
            ],
            "vitals": {
                "locked": self.dual_gate_locked,
                "lock_reason": self.lock_reason,
                "locked_target_id": locked_tid,
                "locked_target_name": locked_tname,
                "hr_bpm": self.last_valid_hr,
                "rr_rpm": self.last_valid_rr,
                "sqi": self.last_sqi,
                "pqrst": [round(float(v), 4) for v in self.pqrst_trace]
            }
        }
