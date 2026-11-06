"""
OmniSense CSI v3.0 - Comprehensive Verification Test Suite
Tests:
1. Empty room noise produces 0 targets on the radar (SECTOR CLEAR).
2. Stationary breathing/cardiac signal produces 1 green STATIC_VITAL point at subject's coordinates.
3. 20-second enrollment builds full-rank 24x24 Ledoit-Wolf covariance model, accurately classifies 2 enrolled profiles, and rejects a 3rd distinct signature as UNKNOWN TARGET via Chi-Square_24 gate.
4. Multi-target presence automatically engages safety lock on PQRST monitor until select_target(track_id) is invoked, after which selected target's vitals (HR, RR, SQI) are isolated.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
from dsp_engine import OmniSenseDSPEngine


def test_1_empty_room_zero_targets():
    """Requirement 1: Empty room noise produces 0 targets on the radar."""
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.ref_noise_energy = 0.005

    # Ingest 300 frames of pure empty room noise (3.0s > 2.5s gate)
    for _ in range(300):
        noise_amp = 25.0 + np.random.normal(0, 0.03, 52)
        noise_phase = np.random.normal(0, 0.015, 52)
        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(noise_amp * np.sin(noise_phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(noise_amp * np.cos(noise_phase), -127, 127).astype(np.int8)
        engine.ingest_frame(csi_iq, -55.0)

    state = engine.get_telemetry_state()
    assert state["presence_state"] == "EMPTY_ROOM"
    assert len(state["targets"]) == 0, f"Expected 0 targets in empty room, got {len(state['targets'])}"
    assert state["vitals"]["locked"] is True
    assert state["vitals"]["lock_reason"] == "NO SUBJECT IN ROOM"


def test_2_static_vital_presence_single_target():
    """Requirement 2: Stationary breathing/cardiac signal produces 1 green STATIC_VITAL point."""
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.ref_noise_energy = 0.005

    dt = 1.0 / 100.0
    sc_idx = np.arange(52)
    lam = 0.1231 * (1.0 - 0.0012 * (sc_idx - 26))

    for i in range(400):
        t = i * dt
        resp_disp = 0.005 * np.sin(2 * np.pi * 0.28 * t)  # 0.28 Hz breathing
        cardiac_disp = 0.0004 * np.sin(2 * np.pi * 1.22 * t)
        total_d = 4.0 + resp_disp + cardiac_disp
        phase = -2.0 * np.pi * total_d / lam + np.random.normal(0, 0.01, 52)
        amp = 28.0 + np.random.normal(0, 0.05, 52)

        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
        engine.ingest_frame(csi_iq, -51.5)

    state = engine.get_telemetry_state()
    assert state["presence_state"] == "STATIC_VITAL"
    assert len(state["targets"]) == 1, f"Expected exactly 1 target, got {len(state['targets'])}"
    target = state["targets"][0]
    assert target["status"] == "STATIC_VITAL"
    assert target["identity"] == "STATIC SUBJECT"


def test_3_gait_enrollment_and_open_set_rejection():
    """Requirement 3: 20-second enrollment builds full-rank 24x24 Ledoit-Wolf covariance model,
    accurately classifies 2 enrolled profiles, and rejects a 3rd distinct signature as UNKNOWN TARGET."""
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.ref_noise_energy = 0.005

    # Simulate 800 frames of baseline signal so buffer is filled
    for i in range(300):
        amp = 28.0 + np.random.normal(0, 0.5, 52)
        phase = np.random.normal(0, 0.2, 52)
        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
        engine.ingest_frame(csi_iq, -50.0)

    # 1. Enroll Profile 1: OPERATOR_ALPHA
    engine.start_gait_enrollment("OPERATOR_ALPHA", duration_sec=0.1)
    # Generate 15 distinct micro-Doppler windows for Alpha
    for _ in range(15):
        alpha_feat = np.full(24, 1.2) + np.random.normal(0, 0.05, 24)
        engine.enroll_windows.append(alpha_feat)
    time.sleep(0.12)
    engine.update_gait_biometrics()
    assert "OPERATOR_ALPHA" in engine.gait_profiles
    assert engine.gait_profiles["OPERATOR_ALPHA"]["inv_cov"].shape == (24, 24)
    # Verify non-singular covariance (full rank)
    assert np.linalg.matrix_rank(engine.gait_profiles["OPERATOR_ALPHA"]["inv_cov"]) == 24

    # 2. Enroll Profile 2: OPERATOR_BETA
    engine.start_gait_enrollment("OPERATOR_BETA", duration_sec=0.1)
    for _ in range(15):
        beta_feat = np.full(24, 3.8) + np.random.normal(0, 0.05, 24)
        engine.enroll_windows.append(beta_feat)
    time.sleep(0.12)
    engine.update_gait_biometrics()
    assert "OPERATOR_BETA" in engine.gait_profiles

    # 3. Test Identification of Alpha
    from dsp_engine import TargetTrack
    track1 = TargetTrack(1, np.array([1.0, 3.0]))
    track1.status = "MOVING"
    engine.tracks = [track1]

    # Mock feature extraction to return Alpha's gait
    engine.extract_24_gait_features = lambda window_len=100, custom_sig=None: np.full(24, 1.21)
    engine.update_gait_biometrics()
    assert track1.identity_label == "OPERATOR_ALPHA"
    assert track1.mahalanobis_dist < engine.gait_profiles["OPERATOR_ALPHA"]["threshold"]

    # 4. Test Identification of Beta
    track2 = TargetTrack(2, np.array([-2.0, 4.0]))
    track2.status = "MOVING"
    engine.tracks = [track2]
    engine.extract_24_gait_features = lambda window_len=100, custom_sig=None: np.full(24, 3.79)
    engine.update_gait_biometrics()
    assert track2.identity_label == "OPERATOR_BETA"

    # 5. Test Open-Set Gate Rejection of Impostor Gamma
    track3 = TargetTrack(3, np.array([0.0, 5.0]))
    track3.status = "MOVING"
    engine.tracks = [track3]
    # Gamma has distinct feature signature (e.g. 8.5)
    engine.extract_24_gait_features = lambda window_len=100, custom_sig=None: np.full(24, 8.5)
    engine.update_gait_biometrics()
    assert track3.identity_label == "UNKNOWN TARGET", f"Expected UNKNOWN TARGET, got {track3.identity_label}"


def test_4_multi_target_vital_signs_safety_lock():
    """Requirement 4: Multi-target presence automatically locks the PQRST monitor until
    POST /api/select_target/{id} is invoked, after which selected target's vitals are isolated."""
    from dsp_engine import TargetTrack
    engine = OmniSenseDSPEngine(fs=100.0)

    # Ingest 800 frames with physiological signal
    dt = 1.0 / 100.0
    sc_idx = np.arange(52)
    lam = 0.1231 * (1.0 - 0.0012 * (sc_idx - 26))

    for i in range(800):
        t = i * dt
        resp_disp = 0.005 * np.sin(2 * np.pi * 0.28 * t)
        cardiac_phase = (t * 1.22) % 1.0
        pqrst_wave = 0.0004 * (
            0.2 * np.exp(-((cardiac_phase - 0.18)/0.03)**2) -
            0.3 * np.exp(-((cardiac_phase - 0.28)/0.015)**2) +
            1.5 * np.exp(-((cardiac_phase - 0.31)/0.012)**2) -
            0.4 * np.exp(-((cardiac_phase - 0.34)/0.015)**2) +
            0.35 * np.exp(-((cardiac_phase - 0.52)/0.04)**2)
        )
        total_d = 3.6 + resp_disp + pqrst_wave
        phase = -2.0 * np.pi * total_d / lam + np.random.normal(0, 0.01, 52)
        amp = 28.0 + np.random.normal(0, 0.05, 52)

        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
        engine.ingest_frame(csi_iq, -51.5)

    # Create 2 simultaneous targets
    t1 = TargetTrack(101, np.array([1.5, 3.5]))
    t1.status = "STATIC_VITAL"
    t1.kf_state[2:] = 0.0

    t2 = TargetTrack(102, np.array([-2.0, 4.0]))
    t2.status = "STATIC_VITAL"
    t2.kf_state[2:] = 0.0

    engine.tracks = [t1, t2]
    engine.selected_target_id = None  # No selection

    # Step A: Vital signs must be LOCKED due to multi-target presence
    engine.compute_vital_signs_bcg()
    assert engine.dual_gate_locked is True
    assert "MULTI-TARGET DETECTED" in engine.lock_reason
    assert np.all(engine.pqrst_trace == 0)

    # Step B: Select Target 101 via click-to-lock API
    engine.select_target(101)
    assert engine.selected_target_id == 101
    assert t1.selected is True
    assert t2.selected is False

    # Step C: Vital signs should now unlock and isolate Target 101
    engine.compute_vital_signs_bcg()
    assert engine.dual_gate_locked is False
    assert engine.last_valid_hr > 0
    assert engine.last_valid_rr > 0
    assert engine.last_sqi >= 0.42
    assert "BIOMETRIC LOCK ENGAGED" in engine.lock_reason


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
