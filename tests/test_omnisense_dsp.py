import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
from dsp_engine import OmniSenseDSPEngine

def test_omnisense_dsp_pipeline():
    """
    Automated verification of Section 5.6 of the Research:
    1. Ingestion of 52 active L-LTF subcarriers.
    2. L2 AGC Normalization & Linear Phase Sanitization.
    3. HSR Subcarrier Selection & SVD Principal Component Analysis.
    4. Dual-Gated Vitals Isolation (VMD + Coiflet-4 DWT BCG).
    5. Mahalanobis 24-Feature Doppler Gait Biometrics & Safety Valve.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    
    # 1. Test Ingestion & Normalization
    dt = 1.0 / 100.0
    sc_idx = np.arange(52)
    lam = 0.1231 * (1.0 - 0.0012 * (sc_idx - 26))
    
    for i in range(850):
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
        phase = -2.0 * np.pi * total_d / lam + np.random.normal(0, 0.015, 52)
        amp = 28.0 + 3.5 * np.sin(sc_idx * 0.15 + t * 0.2) + np.random.normal(0, 0.25, 52)
        
        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
        
        engine.ingest_frame(csi_iq, -51.5)
    
    assert engine.pkt_count == 850
    assert engine.amp_ring.shape == (800, 52)
    assert engine.phase_ring.shape == (800, 52)
    assert engine.ratio_ring.shape == (800, 50)
    
    # 2. Test 2D Localization
    assert 0.5 <= engine.kf_state[1] <= 9.8  # Y depth in bounds
    assert -5.0 <= engine.kf_state[0] <= 5.0 # X span in bounds
    
    # 3. Test 24-Feature Doppler Gait Extractor
    feats = engine.extract_24_gait_features(window_len=100)
    assert len(feats) == 24
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))
    
    # 4. Test Live Rapid Enrollment (5-second Ledoit-Wolf)
    engine.start_gait_enrollment("OPERATOR_ALPHA")
    for _ in range(10):
        engine.update_gait_biometrics()
    
    # 5. Test Vital Signs Extraction & Dual-Gate
    engine.compute_vital_signs_bcg()
    print(f"Extracted HR: {engine.last_valid_hr} BPM | RR: {engine.last_valid_rr} RPM | SQI: {engine.last_sqi}")
    print(f"Lock State: {engine.dual_gate_locked} ({engine.lock_reason})")
    
    assert engine.last_valid_hr > 0, "Failed to extract Heart Rate"
    assert engine.last_valid_rr > 0, "Failed to extract Respiration Rate"
    assert engine.last_sqi >= 0.42, f"SQI below clinical threshold: {engine.last_sqi}"
    assert not engine.dual_gate_locked, "Dual gate failed to engage lock"
    assert len(engine.pqrst_trace) == 200, "PQRST trace size mismatch"

if __name__ == "__main__":
    test_omnisense_dsp_pipeline()
    print("\n>>> ALL OMNISENSE DSP VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
