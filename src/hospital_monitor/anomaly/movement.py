"""Detecção de anomalias nos padrões de movimentação do paciente durante a internação.

Entrada: índice de atividade (0-100) em intervalos regulares, vindo de sensores
(acelerômetro, sensor de pressão do leito) ou da análise de vídeo do quarto
(deslocamento dos keypoints da pessoa; ver ``video.analyzer``).

Detecta:
- **imobilidade prolongada** (risco de lesão por pressão / rebaixamento de consciência);
- **agitação noturna** (possível delirium), comparando com o perfil horário do paciente;
- **assinatura de queda** (pico de impacto seguido de inatividade);
- desvios gerais em relação ao perfil horário (z-score robusto por hora do dia).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..alerts import Alert, Severity


class MovementAnomalyDetector:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.immobility_h = cfg.get("immobility_hours", 2)
        self.night = tuple(cfg.get("night_hours", [23, 5]))
        self.z_thr = cfg.get("zscore_threshold", 3.0)
        self.idle_level = cfg.get("idle_level", 5.0)
        self.baseline_h = cfg.get("baseline_hours", 24)  # perfil horário aprendido no 1º dia

    def _is_night(self, hours: pd.Series) -> pd.Series:
        a, b = self.night
        return (hours >= a) | (hours < b) if a > b else (hours >= a) & (hours < b)

    def score_patient(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        step_min = g["timestamp"].diff().dt.total_seconds().median() / 60
        g["hour"] = g["timestamp"].dt.hour
        # perfil horário do próprio paciente (mediana/MAD por hora do dia)
        base = g[g["timestamp"] < g["timestamp"].iloc[0] + pd.Timedelta(hours=self.baseline_h)]
        prof = base.groupby("hour")["activity"].agg(
            med="median", mad=lambda x: (x - x.median()).abs().median() * 1.4826)
        g = g.join(prof, on="hour")
        g["z"] = (g["activity"] - g["med"]) / g["mad"].clip(lower=3)
        g["idle"] = g["activity"] < self.idle_level
        run = (g["idle"] != g["idle"].shift()).cumsum()
        g["idle_minutes"] = g.groupby(run)["idle"].cumsum() * step_min * g["idle"]
        g["night"] = self._is_night(g["hour"])
        g.attrs["step_min"] = step_min
        return g

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Alert]]:
        out, alerts = [], []
        for pid, g in df.groupby("patient_id"):
            s = self.score_patient(g)
            out.append(s)
            step = s.attrs["step_min"]
            # imobilidade prolongada
            thr = self.immobility_h * 60
            hits = s[(s["idle_minutes"] >= thr) & (s["idle_minutes"].shift(1) < thr)]
            for _, r in hits.iterrows():
                start = r["timestamp"] - pd.Timedelta(minutes=r["idle_minutes"] - step)
                if s.loc[s["timestamp"] == start, "night"].any() and r["night"]:
                    continue  # sono noturno não é alerta por si só
                alerts.append(Alert(pid, "movement", Severity.MEDIA, "Imobilidade prolongada",
                                    f"Sem movimentação significativa há {r['idle_minutes'] / 60:.1f} h "
                                    f"(desde {start:%d/%m %H:%M}), em período diurno.",
                                    timestamp=r["timestamp"].to_pydatetime(),
                                    evidence={"idle_hours": round(r["idle_minutes"] / 60, 2)},
                                    recommended_action="Reposicionar o paciente e avaliar nível de consciência.",
                                    score=0.45))
            # agitação noturna: sequência de atividade alta à noite
            agit = (s["night"] & (s["z"] > self.z_thr)).astype(int)
            win = int(round(60 / step))
            roll = agit.rolling(win, min_periods=win).sum()
            starts = s[(roll >= 0.6 * win) & (roll.shift(1) < 0.6 * win)]
            for _, r in starts.iterrows():
                alerts.append(Alert(pid, "movement", Severity.MEDIA, "Agitação noturna",
                                    f"Atividade muito acima do perfil do paciente durante a madrugada "
                                    f"(índice {r['activity']:.0f}, z={r['z']:.1f}). Possível delirium ou dor.",
                                    timestamp=r["timestamp"].to_pydatetime(), evidence={"z": round(r["z"], 2)},
                                    recommended_action="Avaliar delirium (CAM-ICU), dor e segurança do leito.",
                                    score=0.4))
            # assinatura de queda: pico extremo seguido de >= 30 min de inatividade
            for i in np.where((s["activity"] >= 90) & (s["z"] > self.z_thr))[0]:
                after = s.iloc[i + 1:i + 1 + int(30 / step)]
                if len(after) and after["idle"].mean() >= 0.8:
                    r = s.iloc[i]
                    alerts.append(Alert(pid, "movement", Severity.ALTA, "Possível queda",
                                        f"Pico de impacto (índice {r['activity']:.0f}) seguido de "
                                        f"{len(after) * step:.0f} min de inatividade.",
                                        timestamp=r["timestamp"].to_pydatetime(),
                                        evidence={"peak": float(r["activity"])},
                                        recommended_action="Verificar o paciente imediatamente.", score=0.7))
        return pd.concat(out, ignore_index=True), alerts


def activity_from_video_tracks(tracks: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Converte deslocamento dos keypoints (saída do analisador de vídeo) em índice de atividade."""
    t = tracks.sort_values("time_s").copy()
    t["disp"] = np.hypot(t["hip_x"].diff(), t["hip_y"].diff()).fillna(0)
    t["timestamp"] = pd.to_datetime(t["time_s"], unit="s")
    act = t.set_index("timestamp")["disp"].resample(freq).sum()
    act = 100 * act / max(act.max(), 1e-6)
    return act.rename("activity").reset_index()
