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


def test_6_motion_adaptive_diagonal_loading_person_splitting():
    """
    Test 6: Motion-adaptive diagonal loading absorbs vigorous limb micro-Dopplers,
    preventing 1 rapidly moving person from being falsely split into 2 or 3 targets.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.ref_noise_energy = 0.005

    rng = np.random.default_rng(123)
    # Dominant torso component
    u_torso = rng.standard_normal((100, 1))
    v_torso = rng.standard_normal((1, 50))
    # Secondary high-frequency limb swinging burst
    u_limb = rng.standard_normal((100, 1)) * 0.45
    v_limb = rng.standard_normal((1, 50))

    H_motion_burst = (u_torso @ v_torso * 20.0) + (u_limb @ v_limb * 7.5) + rng.standard_normal((100, 50)) * 0.25
    crowd_res = engine.estimate_crowd_count(H_motion_burst)

    assert crowd_res == 1, f"Expected strictly 1 person during motion burst, got {crowd_res}"


def test_7_identity_hijacking_prevention_and_independent_spawning():
    """
    Test 7: Stationary target retains its verified identity ('OPERATOR_ALPHA'),
    and new motion appearing outside the spatial gate spawns a distinct 'UNKNOWN TARGET'
    without identity jumping or profile hijacking.
    """
    from dsp_engine import TargetTrack
    engine = OmniSenseDSPEngine(fs=100.0)
    engine.pkt_count = 150

    # 1. Register profile OPERATOR_ALPHA
    engine.start_gait_enrollment("OPERATOR_ALPHA", duration_sec=0.1)
    for _ in range(15):
        engine.enroll_windows.append(np.full(24, 1.25) + np.random.normal(0, 0.04, 24))
    time.sleep(0.12)
    engine.update_gait_biometrics()
    assert "OPERATOR_ALPHA" in engine.gait_profiles

    # 2. Track 1 is locked as OPERATOR_ALPHA and becomes stationary at (1.0, 3.0)
    trk1 = TargetTrack(10, np.array([1.0, 3.0]))
    trk1.identity_label = "OPERATOR_ALPHA"
    trk1.locked_identity = "OPERATOR_ALPHA"
    trk1.status = "STATIONARY_LOCKED"
    trk1.pinned_pos = np.array([1.0, 3.0])
    trk1.kf_state[2:] = 0.0
    engine.tracks = [trk1]

    # 3. New candidate observation appears 2.8m away at (-1.5, 4.0)
    candidate_obs = [np.array([-1.5, 4.0])]
    comp_weights = [np.ones(50) / 50.0]
    engine._associate_and_update_tracks(candidate_obs, comp_weights, target_cap=2)

    # Must now have exactly 2 tracks
    assert len(engine.tracks) == 2, f"Expected 2 tracks, got {len(engine.tracks)}"

    # Track 1 must still be stationary at (1.0, 3.0) with locked identity
    t1 = next(t for t in engine.tracks if t.track_id == 10)
    assert t1.identity_label == "OPERATOR_ALPHA"
    assert t1.locked_identity == "OPERATOR_ALPHA"
    assert np.allclose(t1.kf_state[:2], [1.0, 3.0], atol=0.01)

    # Track 2 must be independently spawned as UNKNOWN TARGET
    t2 = next(t for t in engine.tracks if t.track_id != 10)
    assert t2.identity_label == "UNKNOWN TARGET"
    assert t2.locked_identity is None

    # 4. Trigger gait biometrics update: Track 2 MUST NOT hijack OPERATOR_ALPHA
    engine.extract_24_gait_features = lambda window_len=100, custom_sig=None: np.full(24, 1.25)
    engine.update_gait_biometrics()

    assert t1.identity_label == "OPERATOR_ALPHA"
    assert t2.identity_label == "UNKNOWN TARGET", f"Track 2 hijacked identity! Got: {t2.identity_label}"


def test_8_stationary_target_zero_jitter_pinning():
    """
    Test 8: Moving target stopping transitions to STATIONARY_LOCKED, freezes coordinates
    to prevent random walk jitter, and retains 60-second presence hold.
    """
    from dsp_engine import TargetTrack
    engine = OmniSenseDSPEngine(fs=100.0)

    trk = TargetTrack(1, np.array([0.5, 3.0]))
    trk.status = "MOVING"
    trk.kf_state[2:] = [0.02, 0.01]  # speed < 0.08 m/s

    # Simulate 1.2s of near-zero velocity updates (12 cycles of 0.1s)
    z = np.array([0.5, 3.0])
    for _ in range(12):
        trk.update(z, is_motion_active=False)

    assert trk.status == "STATIONARY_LOCKED"
    assert trk.pinned_pos is not None
    assert np.allclose(trk.pinned_pos, [0.5, 3.0], atol=0.01)
    assert trk.speed == 0.0

    # Execute 20 predict cycles: position must remain clamped with zero drift
    for _ in range(20):
        trk.predict()
        assert np.allclose(trk.kf_state[:2], [0.5, 3.0], atol=1e-5)
        assert trk.speed == 0.0

    assert trk.presence_hold_remaining == 60.0


def test_9_digital_twin_synchronization_and_walls():
    """
    Test 9: Updating Digital Twin geometry dynamically computes metric walls,
    positions all hardware nodes (TX1, TX2, RX, AP), and exposes them to telemetry.
    """
    engine = OmniSenseDSPEngine(fs=100.0)
    twin = engine.configure_room_twin(
        room_width=5.0,
        room_length=7.0,
        wall_material="DRYWALL",
        tx1_pos=[-1.8, 0.5],
        tx2_pos=[1.8, 0.5],
        rx_pos=[0.0, 0.2],
        ap_pos=[0.0, 5.5]
    )

    assert twin["room_width"] == 5.0
    assert twin["room_length"] == 7.0
    assert twin["wall_material"] == "DRYWALL"
    assert twin["wall_attenuation_db"] == 4.0

    # Verify 4 physical walls
    walls = twin["walls"]
    assert len(walls) == 4
    # South wall: y=0, x from -2.5 to +2.5
    s_wall = next(w for w in walls if w["name"] == "South Wall")
    assert s_wall["x1"] == -2.5 and s_wall["x2"] == 2.5 and s_wall["y1"] == 0.0
    # North wall: y=7.0, x from -2.5 to +2.5
    n_wall = next(w for w in walls if w["name"] == "North Wall")
    assert n_wall["x1"] == -2.5 and n_wall["x2"] == 2.5 and n_wall["y1"] == 7.0

    # Verify nodes
    assert twin["nodes"]["tx1"]["x"] == -1.8 and twin["nodes"]["tx1"]["y"] == 0.5
    assert twin["nodes"]["tx2"]["x"] == 1.8 and twin["nodes"]["tx2"]["y"] == 0.5
    assert twin["nodes"]["rx"]["x"] == 0.0 and twin["nodes"]["rx"]["y"] == 0.2
    assert twin["nodes"]["ap"]["x"] == 0.0 and twin["nodes"]["ap"]["y"] == 5.5
