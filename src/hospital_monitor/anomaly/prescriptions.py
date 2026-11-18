"""Detecção de alterações inesperadas na evolução das prescrições.

Regras de segurança do paciente (baseadas em práticas do ISMP para
medicamentos de alta vigilância) + modelo estatístico (Isolation Forest) sobre as
características de cada alteração de prescrição.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity

# dose diária máxima de referência para adultos (valores de bula, simplificados para o protótipo)
MAX_DAILY_DOSE = {
    "dipirona": (4000, "mg"),
    "paracetamol": (4000, "mg"),
    "morfina": (60, "mg"),  # via EV/SC em paciente não tolerante
    "varfarina": (10, "mg"),
    "enoxaparina": (200, "mg"),
    "aas": (325, "mg"),
    "insulina regular": (100, "UI"),
    "metformina": (2550, "mg"),
    "losartana": (100, "mg"),
    "ceftriaxona": (4000, "mg"),
    "omeprazol": (80, "mg"),
}

HIGH_ALERT = {"morfina", "insulina regular", "varfarina", "enoxaparina", "heparina", "potássio"}

INTERACTIONS = {
    frozenset({"varfarina", "aas"}): ("Risco aumentado de sangramento (anticoagulante + antiagregante).", Severity.ALTA),
    frozenset({"varfarina", "enoxaparina"}): ("Dupla anticoagulação.", Severity.ALTA),
    frozenset({"morfina", "midazolam"}): ("Risco de depressão respiratória.", Severity.ALTA),
}


class PrescriptionAnomalyDetector:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.jump_ratio = cfg.get("dose_jump_ratio", 2.0)
        self.max_changes = cfg.get("max_changes_24h", 3)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["patient_id", "drug", "timestamp"]).copy()
        df["drug_key"] = df["drug"].str.lower().str.strip()
        df["daily_dose"] = df["dose"] * df["times_per_day"]
        df["prev_daily"] = df.groupby(["patient_id", "drug_key"])["daily_dose"].shift(1)
        df["ratio"] = df["daily_dose"] / df["prev_daily"]
        df["log_ratio"] = np.log2(df["ratio"].fillna(1.0))
        df["hours_since_prev"] = (
            df["timestamp"] - df.groupby(["patient_id", "drug_key"])["timestamp"].shift(1)
        ).dt.total_seconds().div(3600).fillna(999)
        df["max_ratio"] = df.apply(
            lambda r: r["daily_dose"] / MAX_DAILY_DOSE[r["drug_key"]][0] if r["drug_key"] in MAX_DAILY_DOSE else 0.5,
            axis=1)

        def _changes_24h(g: pd.DataFrame) -> pd.Series:
            ts = g["timestamp"]
            return ts.apply(lambda t: int(((ts > t - pd.Timedelta(hours=24)) & (ts <= t)).sum()) - 1)

        df["changes_24h"] = df.groupby(["patient_id", "drug_key"], group_keys=False)[["timestamp"]].apply(_changes_24h)
        df["high_alert"] = df["drug_key"].isin(HIGH_ALERT)
        return df

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Alert]]:
        f = self.features(df)
        alerts: list[Alert] = []

        X = f[["log_ratio", "max_ratio", "changes_24h"]].assign(high=f["high_alert"].astype(int)).values
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
        f["iso_score"] = -iso.decision_function(X)
        f["iso_flag"] = iso.predict(X) == -1
        f["rules"] = ""

        for idx, r in f.iterrows():
            rules = []
            drug = r["drug"]
            # 1) dose acima do máximo diário
            if r["max_ratio"] > 1.0:
                rules.append("dose_acima_maximo")
                alerts.append(self._alert(r, Severity.CRITICA if r["high_alert"] else Severity.ALTA,
                                          f"Dose acima do máximo diário: {drug}",
                                          f"{drug} {r['dose']:g} {r['unit']} {r['times_per_day']}x/dia = "
                                          f"{r['daily_dose']:g} {r['unit']}/dia (máximo de referência "
                                          f"{MAX_DAILY_DOSE[r['drug_key']][0]:g}). Possível erro de digitação/dose.",
                                          "Suspender administração e confirmar a prescrição com o prescritor e a farmácia."))
            # 2) salto de dose em relação à prescrição anterior
            if pd.notna(r["ratio"]) and (r["ratio"] > self.jump_ratio or r["ratio"] < 1 / self.jump_ratio):
                rules.append("salto_de_dose")
                direction = "aumento" if r["ratio"] > 1 else "redução"
                sev = Severity.ALTA if r["high_alert"] else Severity.MEDIA
                alerts.append(self._alert(r, sev, f"Alteração brusca de dose: {drug}",
                                          f"{direction.capitalize()} da dose diária de {drug} (fator {r['ratio']:.1f}x: "
                                          f"{r['prev_daily']:g} → {r['daily_dose']:g} {r['unit']}/dia) "
                                          f"em {r['hours_since_prev']:.0f} h. Prescritor: {r['prescriber']}.",
                                          "Confirmar a justificativa clínica da alteração antes da próxima dose."))
            # 3) muitas alterações em 24 h
            if r["changes_24h"] >= self.max_changes:
                rules.append("alteracoes_frequentes")
                alerts.append(self._alert(r, Severity.MEDIA, f"Alterações frequentes: {drug}",
                                          f"{int(r['changes_24h']) + 1} prescrições de {drug} em 24 h, indicando "
                                          "instabilidade do controle ou falta de padronização entre plantões.",
                                          "Revisar protocolo (ex.: escala de insulina) com a equipe."))
            f.at[idx, "rules"] = ",".join(rules)

        # 4) interações e duplicidade terapêutica entre medicamentos ativos
        for pid, g in f.sort_values("timestamp").groupby("patient_id"):
            active: dict[str, pd.Series] = {}
            seen_pairs = set()
            for _, r in g.iterrows():
                for other_key, o in active.items():
                    pair = frozenset({r["drug_key"], other_key})
                    if pair in seen_pairs or other_key == r["drug_key"]:
                        continue
                    if pair in INTERACTIONS:
                        msg, sev = INTERACTIONS[pair]
                        seen_pairs.add(pair)
                        alerts.append(self._alert(r, sev, f"Interação medicamentosa: {o['drug']} + {r['drug']}",
                                                  msg, "Avaliar risco-benefício e monitorar sinais de sangramento/toxicidade."))
                    elif o["drug_class"] == r["drug_class"] and r["drug_class"] in {"anticoagulante", "opioide", "insulina"}:
                        seen_pairs.add(pair)
                        alerts.append(self._alert(r, Severity.ALTA, f"Duplicidade terapêutica: {r['drug_class']}",
                                                  f"{o['drug']} e {r['drug']} ativos ao mesmo tempo.",
                                                  "Verificar se a prescrição anterior deveria ter sido suspensa."))
                active[r["drug_key"]] = r

        # 5) padrões raros detectados apenas pelo modelo estatístico
        for _, r in f[f["iso_flag"] & (f["iso_score"] > 0.05) & (f["rules"] == "") & f["prev_daily"].notna()].iterrows():
            alerts.append(self._alert(r, Severity.BAIXA, f"Padrão atípico de prescrição: {r['drug']}",
                                      f"Isolation Forest classificou a alteração como atípica (score {r['iso_score']:.2f}).",
                                      "Revisão farmacêutica."))
        return f, alerts

    @staticmethod
    def _alert(r: pd.Series, sev: Severity, title: str, desc: str, action: str) -> Alert:
        return Alert(r["patient_id"], "prescription", sev, title, desc,
                     timestamp=pd.Timestamp(r["timestamp"]).to_pydatetime(),
                     evidence={"drug": r["drug"], "dose": float(r["dose"]), "unit": r["unit"],
                               "times_per_day": int(r["times_per_day"]), "prescriber": r["prescriber"]},
                     recommended_action=action, score={4: 0.9, 3: 0.7, 2: 0.45, 1: 0.25}.get(int(sev), 0.1))
