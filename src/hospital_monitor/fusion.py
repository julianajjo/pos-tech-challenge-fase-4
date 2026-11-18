"""Fusão multimodal de risco (late fusion).

Cada modalidade (sinais vitais, áudio, vídeo, movimentação, prescrições) gera alertas
com um escore 0-1. Para cada paciente, em uma janela deslizante, a fusão combina:

    risco = 0,5 · max(escores) + 0,5 · média ponderada(escores) + bônus · (k − 1)

onde k é o número de modalidades com escore ≥ 0,4. O bônus de corroboração reflete
que achados independentes concordantes (ex.: voz com dispneia + SpO2 caindo) são
mais confiáveis do que um único sensor. Só gera alerta de fusão quando k ≥ 2.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from .alerts import Alert, Severity

DEFAULT_WEIGHTS = {"vitals": 0.35, "audio": 0.2, "video": 0.15, "movement": 0.15, "prescription": 0.15}


class MultimodalFusion:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.window = timedelta(hours=cfg.get("window_hours", 6))
        self.weights = cfg.get("weights", DEFAULT_WEIGHTS)
        self.bonus = cfg.get("corroboration_bonus", 0.15)
        th = cfg.get("thresholds", {"MEDIA": 0.35, "ALTA": 0.55, "CRITICA": 0.75})
        self.thresholds = sorted(((Severity[k], v) for k, v in th.items()), key=lambda x: -x[1])

    def score(self, alerts: list[Alert]) -> tuple[float, dict[str, float]]:
        per_mod: dict[str, float] = {}
        for a in alerts:
            if a.source == "fusion":
                continue
            per_mod[a.source] = max(per_mod.get(a.source, 0.0), a.score)
        if not per_mod:
            return 0.0, {}
        wsum = sum(self.weights.get(m, 0.1) for m in per_mod)
        weighted = sum(self.weights.get(m, 0.1) * s for m, s in per_mod.items()) / wsum
        k = sum(s >= 0.4 for s in per_mod.values())
        risk = min(1.0, 0.5 * max(per_mod.values()) + 0.5 * weighted + self.bonus * max(0, k - 1))
        return risk, per_mod

    def _severity(self, risk: float) -> Severity | None:
        for sev, thr in self.thresholds:
            if risk >= thr:
                return sev
        return None

    def run(self, alerts: list[Alert]) -> list[Alert]:
        out: list[Alert] = []
        by_patient: dict[str, list[Alert]] = {}
        for a in sorted(alerts, key=lambda a: a.timestamp):
            if a.source != "fusion":
                by_patient.setdefault(a.patient_id, []).append(a)
        for pid, items in by_patient.items():
            last_sev: Severity | None = None
            last_time = None
            for a in items:
                win = [x for x in items if a.timestamp - self.window <= x.timestamp <= a.timestamp]
                risk, per_mod = self.score(win)
                k = sum(s >= 0.4 for s in per_mod.values())
                sev = self._severity(risk)
                if last_time and a.timestamp - last_time > self.window:
                    last_sev = None
                if sev is None or k < 2 or (last_sev is not None and sev <= last_sev):
                    continue
                last_sev, last_time = sev, a.timestamp
                mods = ", ".join(f"{m} {s:.2f}" for m, s in sorted(per_mod.items(), key=lambda x: -x[1]))
                drivers = [x.title for x in win if x.score >= 0.4][-4:]
                out.append(Alert(pid, "fusion", sev, "Risco multimodal elevado",
                                 f"Risco combinado {risk:.2f} a partir de {k} fontes concordantes nas últimas "
                                 f"{self.window.total_seconds() / 3600:.0f} h ({mods}). Principais achados: "
                                 + "; ".join(drivers) + ".",
                                 timestamp=a.timestamp, evidence={"risk": round(risk, 3), "modalities": per_mod},
                                 recommended_action=("Acionar médico responsável e time de resposta rápida."
                                                     if sev >= Severity.ALTA else "Reavaliação clínica prioritária."),
                                 score=risk))
        return out

    def patient_summary(self, alerts: list[Alert]) -> dict[str, dict[str, Any]]:
        """Risco atual (última janela) por paciente, para o painel."""
        res = {}
        by_patient: dict[str, list[Alert]] = {}
        for a in alerts:
            by_patient.setdefault(a.patient_id, []).append(a)
        for pid, items in by_patient.items():
            items.sort(key=lambda a: a.timestamp)
            peak, peak_mods, peak_t = 0.0, {}, None
            for a in items:
                win = [x for x in items if a.timestamp - self.window <= x.timestamp <= a.timestamp]
                r, m = self.score(win)
                # em empate de risco, prefere a janela com mais modalidades envolvidas
                if (round(r, 3), len(m), sum(m.values())) > (round(peak, 3), len(peak_mods), sum(peak_mods.values())):
                    peak, peak_mods, peak_t = r, m, a.timestamp
            res[pid] = {"peak_risk": round(peak, 3), "modalities": peak_mods, "peak_time": peak_t,
                        "severity": (self._severity(peak) or Severity.BAIXA).name,
                        "n_alerts": len([a for a in items if a.source != "fusion"])}
        return res
