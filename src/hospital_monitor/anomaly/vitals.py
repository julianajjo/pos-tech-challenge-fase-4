"""Detecção de anomalias em séries temporais de sinais vitais.

Combina três camadas complementares:

1. **Filtro de artefatos** – remove leituras fisiologicamente impossíveis ou picos
   isolados (ex.: sensor de oximetria desconectado), reduzindo falsos alarmes;
2. **Escore clínico NEWS2** (National Early Warning Score 2) – regra validada
   clinicamente para deterioração aguda;
3. **Modelos estatísticos/ML por paciente** – z-score robusto em janela móvel
   (mediana/MAD) e Isolation Forest multivariado treinado na linha de base do
   próprio paciente, capazes de captar desvios que não atingem os limiares do NEWS2
   (ex.: crise hipertensiva, mudança brusca de padrão).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity

SIGNALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
SIGNAL_NAMES = {"hr": "FC", "sbp": "PAS", "dbp": "PAD", "spo2": "SpO2", "rr": "FR", "temp": "Temp"}
SEVERITY_SCORE = {Severity.CRITICA: 0.9, Severity.ALTA: 0.75, Severity.MEDIA: 0.5, Severity.BAIXA: 0.3,
                  Severity.INFO: 0.1}
# piso do desvio evita z gigantes em sinais muito estáveis (ex.: temperatura)
MAD_FLOOR = {"hr": 2, "sbp": 3, "dbp": 2, "spo2": 0.7, "rr": 1, "temp": 0.1}
DEFAULT_LIMITS = {"hr": (20, 250), "sbp": (40, 260), "dbp": (20, 180), "spo2": (50, 100),
                  "rr": (4, 60), "temp": (30, 43)}


# --------------------------------------------------------------------------- NEWS2
def _news_component(x: np.ndarray, bins: list[float], scores: list[int]) -> np.ndarray:
    """``bins`` são os limites superiores (inclusivos) de cada faixa."""
    x = np.asarray(x, dtype=float)
    idx = np.searchsorted(np.asarray(bins), x, side="left")
    out = np.asarray(scores)[np.clip(idx, 0, len(scores) - 1)].astype(float)
    out[np.isnan(x)] = 0
    return out


def news2_components(df: pd.DataFrame) -> pd.DataFrame:
    """Pontuação NEWS2 por parâmetro (escala 1 de SpO2, paciente alerta, sem O2)."""
    return pd.DataFrame({
        "rr": _news_component(df["rr"], [8, 11, 20, 24, np.inf], [3, 1, 0, 2, 3]),
        "spo2": _news_component(df["spo2"], [91, 93, 95, np.inf], [3, 2, 1, 0]),
        "sbp": _news_component(df["sbp"], [90, 100, 110, 219, np.inf], [3, 2, 1, 0, 3]),
        "hr": _news_component(df["hr"], [40, 50, 90, 110, 130, np.inf], [3, 1, 0, 1, 2, 3]),
        "temp": _news_component(df["temp"], [35.0, 36.0, 38.0, 39.0, np.inf], [3, 1, 0, 1, 2]),
    }, index=df.index)


def news2_score(df: pd.DataFrame) -> pd.Series:
    return news2_components(df).sum(axis=1)


# ------------------------------------------------------------------ pré-processamento
def clean_artifacts(df: pd.DataFrame, limits: dict[str, tuple[float, float]] | None = None,
                    spike_k: float = 8.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marca como NaN leituras impossíveis e picos de 1 amostra; retorna (limpo, máscara)."""
    limits = limits or DEFAULT_LIMITS
    clean = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=SIGNALS)
    for s in SIGNALS:
        if s not in clean:
            continue
        lo, hi = limits.get(s, DEFAULT_LIMITS[s])
        x = clean[s].astype(float)
        bad = (x < lo) | (x > hi)
        med = x.rolling(5, center=True, min_periods=3).median()
        mad = np.maximum((x - med).abs().rolling(61, center=True, min_periods=10).median() * 1.4826,
                         MAD_FLOOR[s])
        spike = ((x - med).abs() / mad > spike_k) & ((x.shift(1) - med).abs() / mad < spike_k / 2) & (
            (x.shift(-1) - med).abs() / mad < spike_k / 2)
        m = bad | spike
        mask[s] = m
        clean.loc[m, s] = np.nan
        clean[s] = clean[s].interpolate(limit=3, limit_direction="both")
    return clean, mask


def _features(df: pd.DataFrame) -> pd.DataFrame:
    f = df[SIGNALS].copy()
    for s in SIGNALS:
        f[f"{s}_d15"] = df[s] - df[s].rolling(15, min_periods=1).mean().shift(15).bfill()
    f["shock_index"] = df["hr"] / df["sbp"].clip(lower=1)
    return f.ffill().bfill()


def robust_zscores(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """z-score robusto de cada sinal em relação à janela *anterior* (sem olhar o futuro)."""
    z = pd.DataFrame(index=df.index)
    for s in SIGNALS:
        x = df[s].astype(float)
        med = x.rolling(window, min_periods=window // 3).median().shift(1)
        mad = (x - med).abs().rolling(window, min_periods=window // 3).median().shift(1) * 1.4826
        z[s] = (x - med) / np.maximum(mad, MAD_FLOOR[s])
    return z.fillna(0)


# ------------------------------------------------------------------- detector batch
@dataclass
class VitalEpisode:
    patient_id: str
    start: datetime
    end: datetime
    max_news2: int
    signals: list[str]
    methods: list[str]
    severity: Severity
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60 + 1


class VitalSignsAnomalyDetector:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.baseline_hours = cfg.get("baseline_hours", 6)
        self.window = cfg.get("rolling_window_min", 60)
        self.z_thr = cfg.get("zscore_threshold", 4.0)
        self.contamination = cfg.get("isolation_contamination", 0.01)
        self.min_episode = cfg.get("min_episode_minutes", 3)
        self.limits = {k: tuple(v) for k, v in cfg.get("limits", DEFAULT_LIMITS).items()}

    # -- por paciente
    def score_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp").reset_index(drop=True)
        clean, art = clean_artifacts(df, self.limits)
        comps = news2_components(clean)
        out = clean.copy()
        out["artifact"] = art.any(axis=1)
        out["news2"] = comps.sum(axis=1)
        out["news2_red"] = (comps == 3).any(axis=1)
        z = robust_zscores(clean, self.window)
        out["z_max"] = z.abs().max(axis=1)
        out["z_signals"] = z.abs().gt(self.z_thr).apply(lambda r: ",".join(r.index[r]), axis=1)

        feats = _features(clean)
        t0 = clean["timestamp"].iloc[0]
        base = feats[clean["timestamp"] < t0 + pd.Timedelta(hours=self.baseline_hours)]
        iso = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=0)
        iso.fit(base.values)
        out["iso_score"] = -iso.decision_function(feats.values)  # maior = mais anômalo
        out["iso_flag"] = iso.predict(feats.values) == -1
        # confirmação temporal do Isolation Forest (≥3 de 5 minutos)
        out["iso_flag"] = out["iso_flag"].rolling(5, center=True, min_periods=1).sum() >= 3

        out["anomaly"] = (
            (out["news2"] >= 5)
            | out["news2_red"]
            | (out["iso_flag"] & (out["z_max"] > self.z_thr))
            | (out["iso_flag"] & (out["news2"] >= 3))
        )
        return out

    def _episodes(self, scored: pd.DataFrame, patient_id: str) -> list[VitalEpisode]:
        flags = scored["anomaly"].to_numpy()
        eps: list[VitalEpisode] = []
        i, n = 0, len(flags)
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j, gap = i, 0
            while j + 1 < n and (flags[j + 1] or gap < 5):
                j += 1
                gap = 0 if flags[j] else gap + 1
            while j > i and not flags[j]:
                j -= 1
            seg = scored.iloc[i:j + 1]
            if len(seg) >= self.min_episode:
                eps.append(self._make_episode(seg, patient_id))
            i = j + 1
        return eps

    def _make_episode(self, seg: pd.DataFrame, pid: str) -> VitalEpisode:
        comps = news2_components(seg)
        sig = set()
        for c in comps.columns:
            if (comps[c] >= 2).any():
                sig.add(c)
        for zs in seg["z_signals"]:
            sig.update([s for s in zs.split(",") if s])
        methods = []
        if (seg["news2"] >= 5).any() or seg["news2_red"].any():
            methods.append("NEWS2")
        if (seg["z_max"] > self.z_thr).any():
            methods.append("z-score robusto")
        if seg["iso_flag"].any():
            methods.append("Isolation Forest")
        mx = int(seg["news2"].max())
        if mx >= 7:
            sev = Severity.CRITICA
        elif mx >= 5:
            sev = Severity.ALTA
        elif seg["news2_red"].any() or len(methods) >= 2:
            sev = Severity.MEDIA
        else:
            sev = Severity.BAIXA
        worst = seg.loc[seg["news2"].idxmax()] if mx > 0 else seg.loc[seg["z_max"].idxmax()]
        summary = {s: float(worst[s]) for s in SIGNALS}
        summary["iso_score_max"] = round(float(seg["iso_score"].max()), 3)
        return VitalEpisode(pid, seg["timestamp"].iloc[0].to_pydatetime(), seg["timestamp"].iloc[-1].to_pydatetime(),
                            mx, sorted(sig), methods, sev, summary)

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[VitalEpisode]]:
        scored, episodes = [], []
        for pid, g in df.groupby("patient_id"):
            s = self.score_patient(g)
            scored.append(s)
            episodes.extend(self._episodes(s, pid))
        return pd.concat(scored, ignore_index=True), episodes

    @staticmethod
    def to_alerts(episodes: list[VitalEpisode]) -> list[Alert]:
        alerts = []
        for e in episodes:
            v = e.summary
            names = ", ".join(SIGNAL_NAMES.get(s, s) for s in e.signals) or "padrão multivariado"
            desc = (f"Episódio de {e.duration_min:.0f} min ({e.start:%d/%m %H:%M}-{e.end:%H:%M}). "
                    f"Sinais alterados: {names}. Pior momento: FC {v['hr']:.0f} bpm, PA {v['sbp']:.0f}/"
                    f"{v['dbp']:.0f} mmHg, SpO2 {v['spo2']:.0f}%, FR {v['rr']:.0f} irpm, T {v['temp']:.1f}°C. "
                    f"NEWS2 máximo {e.max_news2}. Detectado por: {', '.join(e.methods)}.")
            if e.max_news2 >= 7:
                action = "Acionar time de resposta rápida e avaliação médica imediata."
            elif e.max_news2 >= 5:
                action = "Avaliação médica urgente e monitorização contínua."
            else:
                action = "Reavaliar paciente e conferir posicionamento dos sensores."
            alerts.append(Alert(e.patient_id, "vitals", e.severity, "Anomalia em sinais vitais", desc,
                                timestamp=e.start, evidence={**v, "news2": e.max_news2, "signals": e.signals,
                                                             "methods": e.methods,
                                                             "duration_min": e.duration_min},
                                recommended_action=action,
                                score=min(1.0, SEVERITY_SCORE[e.severity] + 0.05 * (len(e.methods) - 1))))
        return alerts


# --------------------------------------------------------------- monitor em tempo real
class StreamingVitalsMonitor:
    """Processa uma amostra por vez (ex.: vinda de um monitor multiparamétrico/IoT).

    Usa um modelo Isolation Forest treinado na linha de base e uma janela móvel para
    o z-score. Emite alerta no início de um episódio e quando a gravidade aumenta.
    """

    def __init__(self, patient_id: str, baseline: pd.DataFrame, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.patient_id = patient_id
        self.window = cfg.get("rolling_window_min", 60)
        self.z_thr = cfg.get("zscore_threshold", 4.0)
        self.limits = {k: tuple(v) for k, v in cfg.get("limits", DEFAULT_LIMITS).items()}
        base_clean, _ = clean_artifacts(baseline, self.limits)
        self.iso = IsolationForest(n_estimators=200, contamination=cfg.get("isolation_contamination", 0.01),
                                   random_state=0).fit(_features(base_clean).values)
        self.buffer: deque[dict] = deque(base_clean[["timestamp", *SIGNALS]].to_dict("records")[-self.window - 30:],
                                         maxlen=self.window + 30)
        self.current_severity: Severity | None = None
        self.consecutive = 0
        self.normal_streak = 0

    def update(self, sample: dict[str, Any]) -> Alert | None:
        sample = dict(sample)
        for s in SIGNALS:
            lo, hi = self.limits[s]
            v = sample.get(s)
            if v is None or not (lo <= float(v) <= hi):  # artefato: repete o último valor válido
                sample[s] = self.buffer[-1][s] if self.buffer else np.nan
        self.buffer.append({k: sample[k] for k in ["timestamp", *SIGNALS]})
        win = pd.DataFrame(self.buffer)
        comps = news2_components(win.iloc[[-1]]).iloc[0]
        news = int(comps.sum())
        z = robust_zscores(win, self.window).iloc[-1].abs()
        iso_flag = self.iso.predict(_features(win).iloc[[-1]].values)[0] == -1
        abnormal = news >= 5 or (comps == 3).any() or (iso_flag and (z.max() > self.z_thr or news >= 3))
        self.consecutive = self.consecutive + 1 if abnormal else 0
        self.normal_streak = 0 if abnormal else self.normal_streak + 1
        if self.normal_streak >= 15:
            self.current_severity = None
        if self.consecutive < 3:  # confirmação de 3 minutos evita alarmes por ruído
            return None
        sev = Severity.CRITICA if news >= 7 else Severity.ALTA if news >= 5 else Severity.MEDIA
        if self.current_severity is not None and sev <= self.current_severity:
            return None
        self.current_severity = sev
        changed = [SIGNAL_NAMES[s] for s in SIGNALS if (s in comps and comps[s] >= 2) or z.get(s, 0) > self.z_thr]
        return Alert(self.patient_id, "vitals", sev, "Anomalia em sinais vitais (tempo real)",
                     f"NEWS2={news}; sinais: {', '.join(changed) or 'padrão multivariado'}; "
                     f"FC {sample['hr']:.0f}, PAS {sample['sbp']:.0f}, SpO2 {sample['spo2']:.0f}, FR {sample['rr']:.0f}.",
                     timestamp=pd.Timestamp(sample["timestamp"]).to_pydatetime(),
                     evidence={"news2": news, "z_max": float(z.max()), "iso_flag": bool(iso_flag)},
                     recommended_action="Avaliação à beira-leito.", score=SEVERITY_SCORE[sev])
