import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from hospital_monitor.video.kinematics import analyze_repetitions, angle, detect_falls, frame_features, stitch_tracks
from hospital_monitor.video.pose import OPENPOSE_TO_COCO

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "simulate_videos.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def test_angle():
    assert abs(angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) - 90) < 1e-6


def _session():
    rows = []
    for i in range(int(30 * 12.5)):
        t = i / 12.5
        kl, kr, tr = sim.physio_angles(t)
        pts = sim.physio_pose(kl, kr, tr)
        kps = np.hstack([pts, np.full((18, 1), 0.9)])[OPENPOSE_TO_COCO]
        box = np.array([kps[:, 0].min(), kps[:, 1].min(), kps[:, 0].max(), kps[:, 1].max()])
        rows.append({**frame_features(kps, box), "time_s": t, "track_id": 1})
    return pd.DataFrame(rows)


def test_simulated_knee_angles_match_protocol():
    f = _session()
    assert 80 < f["knee_mean"].min() < 95
    assert f["knee_mean"].max() > 170


def test_repetition_analysis_flags_known_deviations():
    reps = analyze_repetitions(_session(), {"target_min_knee_angle": 100, "max_trunk_inclination": 45,
                                            "max_asymmetry": 20})
    assert len(reps) == 8
    assert all(not r.issues for r in reps[:4])
    assert all(r.issues for r in reps[4:])
    assert any("assimetria" in i for i in reps[6].issues)
    assert any("tronco" in i for i in reps[5].issues)


def test_fall_detection_and_track_stitching():
    t = np.arange(0, 6, 0.08)
    standing = t < 3
    df = pd.DataFrame({"time_s": t, "track_id": np.where(standing, 1, 2),
                       "hip_x": 500.0, "hip_y": np.where(standing, 400.0, 620.0),
                       "box_h": np.where(standing, 480.0, 140.0), "box_w": np.where(standing, 180.0, 450.0)})
    df["aspect_ratio"] = df["box_w"] / df["box_h"]
    stitched = stitch_tracks(df)
    assert stitched["track_id"].nunique() == 1
    falls = detect_falls(stitched)
    assert len(falls) == 1 and 2.9 < falls[0]["time_s"] < 3.3
