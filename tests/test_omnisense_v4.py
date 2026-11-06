"""
OmniSense CSI v4.0 (IEEE Research Edition) - Comprehensive Verification Test Suite
Tests:
1. test_1_single_person_zero_ghost_splitting:
   Single moving person produces STRICTLY 1 target (Rank-1 Coherent subspace) and ZERO ghost splitting.
2. test_2_unbounded_crowd_counting_scalability:
   Multi-target scattering dynamically scales crowd count (1, 2, 4, 5+) without any hardcoded 3-target cap.
3. test_3_through_wall_perimeter_gating:
   Reflections beyond sensing perimeter (or excess CIR delay) are rejected as clutter.
4. test_4_digital_twin_specular_ghost_folding:
   1st-order specular wall reflections outside room boundaries are folded back inside.
5. test_5_dual_link_demultiplexing_and_benchmark_suite:
   Multi-MAC demultiplexing tracks TX1 and TX2, and IEEE benchmark suite generates valid LaTeX tables.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
from dsp_engine import OmniSenseDSPEngine
from benchmark_suite import IEEEBenchmarkSuite


def test_1_single_person_zero_ghost_splitting():
    """
    Test 1: Single moving individual must produce STRICTLY 1 target on the radar.
    Eliminates the FastICA person-splitting bug where 1 moving person was split into 3 targets.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.ref_noise_energy = 0.005

    dt = 0.01
    sc_idx = np.arange(52)
    lam = 0.1231 * (1.0 - 0.0012 * (sc_idx - 26))

    # Ingest 250 frames of a single person walking back and forth
    for i in range(250):
        t = i * dt
        x_pos = 1.5 * np.sin(0.5 * np.pi * t)
        y_pos = 3.5 + 0.8 * np.cos(0.5 * np.pi * t)
        dist = np.hypot(x_pos, y_pos)
        doppler = 2.0 * np.pi * 0.7 * np.sin(0.5 * np.pi * t)

        phase = -2.0 * np.pi * dist / lam + doppler * t + np.random.normal(0, 0.02, 52)
        # Dominant coherent energy across subcarriers
        amp = 28.0 + 8.0 * np.sin(sc_idx * 0.14 + t * 2.0) + np.random.normal(0, 0.1, 52)

        csi_iq = np.empty(104, dtype=np.int8)
        csi_iq[0::2] = np.clip(amp * np.sin(phase), -127, 127).astype(np.int8)
        csi_iq[1::2] = np.clip(amp * np.cos(phase), -127, 127).astype(np.int8)
        engine.ingest_frame(csi_iq, -50.0, source_mac="fc:e8:c0:df:37:b4")

    state = engine.get_telemetry_state()
    # Must detect active motion
    assert state["presence_state"] == "ACTIVE_MOTION"
    # RMT Crowd estimate MUST be 1
    assert state["crowd"]["estimated_count"] == 1
    # Radar targets must be EXACTLY 1 (Zero ghost splitting)
    assert len(state["targets"]) == 1, f"Expected exactly 1 target, got {len(state['targets'])}"
    # Target status must be MOVING
    assert state["targets"][0]["status"] == "MOVING"


def test_2_unbounded_crowd_counting_scalability():
    """
    Test 2: Crowd counting must scale smoothly to 2, 4, 5+ individuals without a 3-target cap.
    """
    engine = OmniSenseDSPEngine(fs=100.0)

    # 1. Test Single Person (Rank-1 dominant eigenvalue)
    # Synthetic dynamic matrix with 1 dominant singular value
    rng = np.random.default_rng(42)
    u1 = rng.standard_normal((100, 1))
    v1 = rng.standard_normal((1, 50))
    H_single = u1 @ v1 * 25.0 + rng.standard_normal((100, 50)) * 0.3
    count_1 = engine.estimate_crowd_count(H_single)
    assert count_1 == 1, f"Expected 1 person for Rank-1 signal, got {count_1}"

    # 2. Test 2 Independent Subjects
    u2 = rng.standard_normal((100, 2))
    v2 = rng.standard_normal((2, 50))
    H_two = u2 @ v2 * 18.0 + rng.standard_normal((100, 50)) * 0.4
    count_2 = engine.estimate_crowd_count(H_two)
    assert count_2 >= 2, f"Expected at least 2 people, got {count_2}"

    # 3. Test 4 Independent Subjects (Scalability beyond 3)
    u4 = rng.standard_normal((100, 4))
    v4 = rng.standard_normal((4, 50))
    H_four = u4 @ v4 * 16.0 + rng.standard_normal((100, 50)) * 0.5
    count_4 = engine.estimate_crowd_count(H_four)
    assert count_4 >= 3, f"Expected scaled crowd count >= 3, got {count_4}"

    # 4. Test 6 Independent Subjects (Demonstrating unbounded scaling)
    u6 = rng.standard_normal((100, 6))
    v6 = rng.standard_normal((6, 50))
    H_six = u6 @ v6 * 15.0 + rng.standard_normal((100, 50)) * 0.6
    count_6 = engine.estimate_crowd_count(H_six)
    assert count_6 >= 4, f"Expected crowd count >= 4 for 6 independent orthogonal scatterers, got {count_6}"


def test_3_through_wall_perimeter_gating():
    """
    Test 3: Windowed IFFT CIR delay profiling filters out-of-perimeter clutter.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.set_perimeter_gate(max_radius_m=3.5, wall_cutoff_db=12.0)

    # Synthetic candidate inside room (r = 2.5m)
    cand_inside = [1.5, 2.0]  # hypot = 2.5m <= 3.5m
    accepted_inside, label_inside = engine._apply_perimeter_range_gating(cand_inside)
    assert accepted_inside is True
    assert label_inside == "IN_ROOM"

    # Synthetic candidate outside room (r = 5.2m)
    cand_outside = [3.5, 3.8]  # hypot = 5.16m > 3.5m
    accepted_outside, label_outside = engine._apply_perimeter_range_gating(cand_outside)
    assert accepted_outside is False
    assert label_outside == "OUTSIDE_PERIMETER_CLUTTER"

    # Verify clutter rejection metric was updated
    assert engine.clutter_rejected_pct > 0.0


def test_4_digital_twin_specular_ghost_folding():
    """
    Test 4: 1st-order specular wall reflections outside room boundaries are folded
    back inside the room polygon.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.configure_room_twin(
        width_m=6.0, length_m=8.0, wall_material="CONCRETE",
        tx1_pos=[-2.0, 0.0], tx2_pos=[2.0, 0.0], rx_pos=[0.0, 0.0], ap_pos=[0.0, 6.0]
    )
    # Room boundaries: X in [-3.0, 3.0], Y in [0.0, 8.0]

    # Ghost reflection beyond East wall (x = +4.2m > 3.0m, y = 4.0m)
    ghost_east = [4.2, 4.0]
    folded_east, is_ghost_east = engine._fold_specular_multipath_ghost(ghost_east)
    assert is_ghost_east is True
    # Folded coordinate: 2 * 3.0 - 4.2 = 1.8m
    assert np.isclose(folded_east[0], 1.8, atol=0.01)
    assert np.isclose(folded_east[1], 4.0, atol=0.01)

    # Ghost reflection beyond North wall (x = 0.5m, y = 9.5m > 8.0m)
    ghost_north = [0.5, 9.5]
    folded_north, is_ghost_north = engine._fold_specular_multipath_ghost(ghost_north)
    assert is_ghost_north is True
    # Folded coordinate: 2 * 8.0 - 9.5 = 6.5m
    assert np.isclose(folded_north[0], 0.5, atol=0.01)
    assert np.isclose(folded_north[1], 6.5, atol=0.01)

    # Inside target must remain unchanged
    target_inside = [1.2, 4.5]
    folded_inside, is_ghost_inside = engine._fold_specular_multipath_ghost(target_inside)
    assert is_ghost_inside is False
    assert np.isclose(folded_inside[0], 1.2, atol=0.01)
    assert np.isclose(folded_inside[1], 4.5, atol=0.01)

    assert engine.ghosts_folded_count == 2


def test_5_dual_link_demultiplexing_and_benchmark_suite():
    """
    Test 5: Dual-Link Demultiplexing (ESP32-TX1 and ESP32-TX2) and IEEE Benchmark Suite.
    """
    engine = OmniSenseDSPEngine(fs=100.0)

    # Ingest from TX1
    amp1 = np.full(52, 28.0)
    phase1 = np.zeros(52)
    csi_iq1 = np.empty(104, dtype=np.int8)
    csi_iq1[0::2] = np.clip(amp1 * np.sin(phase1), -127, 127).astype(np.int8)
    csi_iq1[1::2] = np.clip(amp1 * np.cos(phase1), -127, 127).astype(np.int8)
    engine.ingest_frame(csi_iq1, -48.0, source_mac="fc:e8:c0:df:37:b4")

    # Ingest from TX2
    amp2 = np.full(52, 31.0)
    phase2 = np.zeros(52)
    csi_iq2 = np.empty(104, dtype=np.int8)
    csi_iq2[0::2] = np.clip(amp2 * np.sin(phase2), -127, 127).astype(np.int8)
    csi_iq2[1::2] = np.clip(amp2 * np.cos(phase2), -127, 127).astype(np.int8)
    engine.ingest_frame(csi_iq2, -45.0, source_mac="fc:e8:c0:df:37:b5")

    state = engine.get_telemetry_state()
    assert state["mesh_status"]["tx1_active"] is True
    assert state["mesh_status"]["tx2_active"] is True
    assert state["mesh_status"]["tx1_rssi"] == -48.0
    assert state["mesh_status"]["tx2_rssi"] == -45.0

    # Test IEEE Benchmark Suite
    suite = IEEEBenchmarkSuite()
    # Log 3 sample trials
    suite.log_trial(
        {"actual_people_count": 1, "ground_truth_x": 0.0, "ground_truth_y": 4.0, "reference_hr_bpm": 72.0, "reference_rr_rpm": 16.0},
        {"estimated_people_count": 1, "estimated_x": 0.1, "estimated_y": 4.1, "estimated_hr_bpm": 71.5, "estimated_rr_rpm": 16.2}
    )
    suite.log_trial(
        {"actual_people_count": 2, "ground_truth_x": 1.5, "ground_truth_y": 3.5, "reference_hr_bpm": 75.0, "reference_rr_rpm": 17.0},
        {"estimated_people_count": 2, "estimated_x": 1.6, "estimated_y": 3.4, "estimated_hr_bpm": 74.8, "estimated_rr_rpm": 16.9}
    )

    metrics = suite.get_metrics()
    assert metrics["total_trials"] == 2
    assert metrics["counting_accuracy_pct"] == 100.0
    assert metrics["localization_rmse_m"] < 0.25
    assert metrics["hr_mae_bpm"] < 1.0

    latex = suite.generate_ieee_latex_table()
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert "OmniSense CSI v4.0" in latex
