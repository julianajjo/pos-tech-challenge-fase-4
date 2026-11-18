"""Cálculos biomecânicos a partir dos keypoints e regras do protocolo de exercício."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .pose import KP


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo ABC em graus."""
    ba, bc = a[:2] - b[:2], c[:2] - b[:2]
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))


def frame_features(kps: np.ndarray, box: np.ndarray, kp_thr: float = 0.3) -> dict[str, float]:
    """Ângulos articulares, inclinação do tronco e posição do quadril de uma pessoa."""
    def ok(*names: str) -> bool:
        return all(kps[KP[n], 2] >= kp_thr for n in names)

    def p(n: str) -> np.ndarray:
        return kps[KP[n]]

    f: dict[str, float] = {}
    for side in ("l", "r"):
        f[f"{side}_knee"] = angle(p(f"{side}_hip"), p(f"{side}_knee"), p(f"{side}_ankle")) \
            if ok(f"{side}_hip", f"{side}_knee", f"{side}_ankle") else np.nan
        f[f"{side}_hip_angle"] = angle(p(f"{side}_shoulder"), p(f"{side}_hip"), p(f"{side}_knee")) \
            if ok(f"{side}_shoulder", f"{side}_hip", f"{side}_knee") else np.nan
        f[f"{side}_elbow"] = angle(p(f"{side}_shoulder"), p(f"{side}_elbow"), p(f"{side}_wrist")) \
            if ok(f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist") else np.nan
        f[f"{side}_shoulder_angle"] = angle(p(f"{side}_hip"), p(f"{side}_shoulder"), p(f"{side}_elbow")) \
            if ok(f"{side}_hip", f"{side}_shoulder", f"{side}_elbow") else np.nan
    if ok("l_shoulder", "r_shoulder", "l_hip", "r_hip"):
        sh = (p("l_shoulder")[:2] + p("r_shoulder")[:2]) / 2
        hp = (p("l_hip")[:2] + p("r_hip")[:2]) / 2
        v = sh - hp
        f["trunk_inclination"] = float(np.degrees(np.arctan2(abs(v[0]), -v[1] + 1e-6)))  # 0 = ereto
        f["hip_x"], f["hip_y"] = float(hp[0]), float(hp[1])
    else:
        f["trunk_inclination"] = np.nan
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        f["hip_x"], f["hip_y"] = float(cx), float(cy)
    w, h = box[2] - box[0], box[3] - box[1]
    f["box_w"], f["box_h"] = float(w), float(h)
    f["aspect_ratio"] = float(w / max(h, 1e-6))
    f["knee_mean"] = float(np.nanmean([f["l_knee"], f["r_knee"]])) if not (
        np.isnan(f["l_knee"]) and np.isnan(f["r_knee"])) else np.nan
    f["knee_asymmetry"] = abs(f["l_knee"] - f["r_knee"]) if not (
        np.isnan(f["l_knee"]) or np.isnan(f["r_knee"])) else np.nan
    return f


@dataclass
class Repetition:
    index: int
    start_s: float
    bottom_s: float
    end_s: float
    min_knee: float
    max_trunk: float
    max_asymmetry: float
    issues: list[str]


def analyze_repetitions(track: pd.DataFrame, ex: dict[str, Any]) -> list[Repetition]:
    """Conta repetições (máquina de estados no ângulo do joelho) e avalia cada uma."""
    down, up = ex.get("rep_down_angle", 120), ex.get("rep_up_angle", 160)
    target = ex.get("target_min_knee_angle", 100)
    max_trunk = ex.get("max_trunk_inclination", 45)
    max_asym = ex.get("max_asymmetry", 20)
    knee = track["knee_mean"].interpolate(limit=5).rolling(3, center=True, min_periods=1).median()
    reps, state, start_i = [], "up", None
    for i, k in enumerate(knee.to_numpy()):
        if np.isnan(k):
            continue
        if state == "up" and k < down:
            state, start_i = "down", i
            # recua até o início da descida
            while start_i > 0 and knee.iloc[start_i - 1] > knee.iloc[start_i] and knee.iloc[start_i - 1] < up:
                start_i -= 1
        elif state == "down" and k > up:
            seg = track.iloc[start_i:i + 1]
            ks = knee.iloc[start_i:i + 1]
            issues = []
            mn = float(ks.min())
            mt = float(np.nanmax(seg["trunk_inclination"])) if seg["trunk_inclination"].notna().any() else np.nan
            ma = float(np.nanmax(seg["knee_asymmetry"])) if seg["knee_asymmetry"].notna().any() else np.nan
            if mn > target:
                issues.append(f"amplitude insuficiente (joelho mínimo {mn:.0f}° > alvo {target}°)")
            if not np.isnan(mt) and mt > max_trunk:
                issues.append(f"inclinação excessiva do tronco ({mt:.0f}°)")
            if not np.isnan(ma) and ma > max_asym:
                issues.append(f"assimetria entre joelhos ({ma:.0f}°)")
            reps.append(Repetition(len(reps) + 1, float(seg["time_s"].iloc[0]),
                                   float(track["time_s"].iloc[start_i + int(np.argmin(ks.to_numpy()))]),
                                   float(seg["time_s"].iloc[-1]), mn, mt, ma, issues))
            state = "up"
    return reps


def detect_falls(track: pd.DataFrame, drop_ratio: float = 0.25, lying_ratio: float = 1.3,
                 window_s: float = 1.0) -> list[dict[str, Any]]:
    """Queda: quadril desce > ``drop_ratio`` x altura do corpo em até ``window_s`` e a pessoa
    termina em posição horizontal (largura/altura da caixa > ``lying_ratio``)."""
    events = []
    t = track["time_s"].to_numpy()
    hy = track["hip_y"].to_numpy()
    bh = track["box_h"].rolling(15, min_periods=1).max().to_numpy()  # altura de referência em pé
    ar = track["aspect_ratio"].rolling(3, min_periods=1).median().to_numpy()
    last = -1e9
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - window_s)
        if j >= i:
            continue
        drop = (hy[i] - hy[j:i].min()) / max(bh[i], 1)
        lying_after = ar[i:np.searchsorted(t, t[i] + 1.0) + 1]
        if drop > drop_ratio and len(lying_after) and np.nanmax(lying_after) > lying_ratio and t[i] - last > 3:
            events.append({"time_s": float(t[i]), "hip_drop_ratio": float(drop),
                           "aspect_ratio": float(np.nanmax(lying_after))})
            last = t[i]
    return events


def stitch_tracks(frames: pd.DataFrame, max_gap_s: float = 1.0, max_dist_factor: float = 1.5) -> pd.DataFrame:
    """Une trajetórias interrompidas pelo rastreador.

    Em quedas a silhueta muda bruscamente (em pé → deitada) e o rastreador costuma
    criar um novo ID. Um track que começa até ``max_gap_s`` após outro terminar, a
    menos de ``max_dist_factor`` x altura do corpo de distância, recebe o mesmo ID.
    """
    f = frames.copy()
    spans = f.groupby("track_id").agg(t0=("time_s", "min"), t1=("time_s", "max"))
    mapping: dict[int, int] = {}
    for tid, s in spans.sort_values("t0").iterrows():
        first = f[(f["track_id"] == tid)].sort_values("time_s").iloc[0]
        best, best_d = None, np.inf
        for oid, o in spans.iterrows():
            if oid == tid or not (0 <= s["t0"] - o["t1"] <= max_gap_s):
                continue
            last = f[(f["track_id"] == oid)].sort_values("time_s").iloc[-1]
            d = np.hypot(first["hip_x"] - last["hip_x"], first["hip_y"] - last["hip_y"])
            if d < max_dist_factor * max(last["box_h"], 1) and d < best_d:
                best, best_d = oid, d
        if best is not None:
            mapping[tid] = mapping.get(best, best)
    f["track_id"] = f["track_id"].map(lambda t: mapping.get(t, t))
    return f


def detect_lying(track: pd.DataFrame, lying_ratio: float = 1.3, min_s: float = 2.0) -> list[dict[str, Any]]:
    """Pessoa em posição horizontal por pelo menos ``min_s`` segundos (ex.: no chão)."""
    lying = track["aspect_ratio"] > lying_ratio
    seg = (lying != lying.shift()).cumsum()
    out = []
    for _, g in track[lying].groupby(seg[lying]):
        dur = g["time_s"].iloc[-1] - g["time_s"].iloc[0]
        if dur >= min_s:
            out.append({"time_s": float(g["time_s"].iloc[0]), "duration_s": float(dur)})
    return out
