"""Geração de dados sintéticos de internação para demonstração e testes.

Os sinais vitais seguem as faixas e a resolução (1 amostra/min) de bases como o
MIMIC-III Waveform/Numerics e o BIDMC (PhysioNet). Episódios clínicos conhecidos
são inseridos para que seja possível medir se os detectores os encontram.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

START = datetime(2026, 11, 10, 0, 0)


@dataclass
class InjectedEpisode:
    patient_id: str
    kind: str
    start: datetime
    end: datetime
    description: str


def _circadian(minutes: np.ndarray, amp: float, phase_h: float = 15.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * (minutes / 60.0 - phase_h + 6) / 24.0)


def _ramp(n: int, start: int, length: int) -> np.ndarray:
    r = np.zeros(n)
    end = min(n, start + length)
    r[start:end] = np.linspace(0, 1, end - start)
    r[end:] = 1.0
    return r


def _bump(n: int, start: int, length: int, rise: int = 5) -> np.ndarray:
    """Pulso trapezoidal (sobe, platô, desce)."""
    b = np.zeros(n)
    rise = max(1, min(rise, length // 3))
    seg = np.concatenate([np.linspace(0, 1, rise), np.ones(length - 2 * rise), np.linspace(1, 0, rise)])
    end = min(n, start + length)
    b[start:end] = seg[: end - start]
    return b


def generate_vitals(hours: int = 48, seed: int = 7) -> tuple[pd.DataFrame, list[InjectedEpisode]]:
    """Sinais vitais minuto a minuto para 4 pacientes com episódios conhecidos."""
    rng = np.random.default_rng(seed)
    n = hours * 60
    t = np.arange(n)
    ts = [START + timedelta(minutes=int(m)) for m in t]
    frames, episodes = [], []

    profiles = {
        "P001": dict(hr=78, sbp=122, dbp=76, spo2=97, rr=15, temp=36.7),
        "P002": dict(hr=84, sbp=135, dbp=84, spo2=95.5, rr=17, temp=36.6),
        "P003": dict(hr=70, sbp=118, dbp=72, spo2=98, rr=14, temp=36.5),
        "P004": dict(hr=74, sbp=128, dbp=80, spo2=96.5, rr=15, temp=36.6),
    }
    for pid, p in profiles.items():
        hr = p["hr"] + _circadian(t, 5) + rng.normal(0, 2.0, n)
        sbp = p["sbp"] + _circadian(t, 6) + rng.normal(0, 3.0, n)
        dbp = p["dbp"] + _circadian(t, 4) + rng.normal(0, 2.0, n)
        spo2 = p["spo2"] + rng.normal(0, 0.6, n)
        rr = p["rr"] + _circadian(t, 1) + rng.normal(0, 0.8, n)
        temp = p["temp"] + _circadian(t, 0.25) + rng.normal(0, 0.05, n)

        if pid == "P001" and n > 30 * 60 + 240:  # deterioração compatível com sepse (h30 em diante)
            s = 30 * 60
            r = _ramp(n, s, 240)
            hr += 45 * r
            sbp -= 38 * r
            dbp -= 22 * r
            rr += 11 * r
            temp += 2.1 * r
            spo2 -= 4.5 * r
            episodes.append(InjectedEpisode(pid, "deterioracao_sepse", ts[s], ts[-1],
                                            "Taquicardia, hipotensão, taquipneia e febre progressivas"))
        if pid == "P002" and n > 20 * 60 + 30:  # dessaturação aguda (h20, 25 min)
            s, L = 20 * 60, 25
            b = _bump(n, s, L, rise=4)
            spo2 -= 11 * b
            hr += 22 * b
            rr += 9 * b
            episodes.append(InjectedEpisode(pid, "dessaturacao", ts[s], ts[s + L], "Queda de SpO2 para ~84%"))
        if pid == "P004" and n > 36 * 60 + 45:  # crise hipertensiva curta (h36, 40 min) logo após a queda
            s, L = 36 * 60, 40
            b = _bump(n, s, L, rise=6)
            sbp += 55 * b
            dbp += 25 * b
            hr += 18 * b
            episodes.append(InjectedEpisode(pid, "crise_hipertensiva", ts[s], ts[s + L], "PAS > 180 mmHg"))

        # artefatos de sensor (não devem gerar alerta clínico)
        for _ in range(3):
            k = int(rng.integers(60, n - 60))
            spo2[k] = rng.choice([0.0, 35.0])
            hr[k + 1] = rng.choice([0.0, 300.0])

        df = pd.DataFrame({
            "timestamp": ts, "patient_id": pid,
            "hr": hr.round(0), "sbp": sbp.round(0), "dbp": dbp.round(0),
            "spo2": np.clip(spo2, 0, 100).round(1), "rr": rr.round(0), "temp": temp.round(1),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True), episodes


def generate_prescriptions(seed: int = 11) -> pd.DataFrame:
    """Histórico de prescrições com alterações esperadas e inesperadas."""
    d = lambda h: START + timedelta(hours=h)  # noqa: E731
    rows = [
        # P001: erro de dose de morfina (vírgula/zero a mais)
        ("P001", d(1), "Dipirona", 1000, "mg", 4, "analgesico", "Dr. Silva"),
        ("P001", d(1), "Morfina", 2, "mg", 6, "opioide", "Dr. Silva"),
        ("P001", d(14), "Morfina", 2, "mg", 6, "opioide", "Dr. Silva"),
        ("P001", d(26), "Morfina", 20, "mg", 6, "opioide", "Dra. Costa"),
        ("P001", d(31), "Ceftriaxona", 1000, "mg", 2, "antibiotico", "Dra. Costa"),
        ("P001", d(31), "Paracetamol", 750, "mg", 4, "analgesico", "Dra. Costa"),
        # P002: interação varfarina + AAS e duplicidade de anticoagulante
        ("P002", d(2), "Varfarina", 5, "mg", 1, "anticoagulante", "Dr. Lima"),
        ("P002", d(2), "Omeprazol", 20, "mg", 1, "ibp", "Dr. Lima"),
        ("P002", d(18), "AAS", 100, "mg", 1, "antiagregante", "Dr. Souza"),
        ("P002", d(22), "Enoxaparina", 40, "mg", 1, "anticoagulante", "Dr. Souza"),
        # P003: ajustes esperados e dentro do protocolo
        ("P003", d(3), "Losartana", 50, "mg", 1, "anti-hipertensivo", "Dra. Rocha"),
        ("P003", d(27), "Losartana", 50, "mg", 2, "anti-hipertensivo", "Dra. Rocha"),
        ("P003", d(3), "Dipirona", 1000, "mg", 4, "analgesico", "Dra. Rocha"),
        # P004: mudanças frequentes de insulina e salto de dose
        ("P004", d(1), "Insulina regular", 4, "UI", 3, "insulina", "Dr. Alves"),
        ("P004", d(9), "Insulina regular", 6, "UI", 3, "insulina", "Dr. Alves"),
        ("P004", d(13), "Insulina regular", 8, "UI", 3, "insulina", "Dr. Alves"),
        ("P004", d(16), "Insulina regular", 20, "UI", 3, "insulina", "Plantonista"),
        ("P004", d(20), "Insulina regular", 8, "UI", 3, "insulina", "Plantonista"),
        ("P004", d(2), "Metformina", 850, "mg", 2, "antidiabetico", "Dr. Alves"),
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "timestamp", "drug", "dose", "unit",
                                     "times_per_day", "drug_class", "prescriber"])
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def generate_movement(days: int = 3, seed: int = 5) -> pd.DataFrame:
    """Índice de atividade (0-100) a cada 5 min, como em sensores de leito/acelerômetro."""
    rng = np.random.default_rng(seed)
    step = 5
    n = days * 24 * 60 // step
    ts = [START + timedelta(minutes=step * i) for i in range(n)]
    hours = np.array([t.hour + t.minute / 60 for t in ts])
    frames = []
    for pid in ["P001", "P002", "P003", "P004"]:
        day_activity = np.where((hours >= 7) & (hours < 22), 22, 5)
        act = day_activity + rng.gamma(2.0, 3.0, n)
        # reposicionamento periódico (~a cada 90 min)
        act[(np.arange(n) % 18) == 0] += 25
        if pid == "P001":  # imobilidade progressiva no 2º dia (risco de lesão por pressão)
            s = 32 * 12
            act[s:s + 4 * 12] = rng.gamma(1.0, 0.4, 4 * 12)
        if pid == "P002":  # agitação noturna (possível delirium) na 2ª e 3ª noite
            for night in (1, 2):
                s = night * 24 * 12 + 1 * 12  # 01h
                act[s:s + 3 * 12] = 55 + rng.normal(0, 10, 3 * 12)
        if pid == "P004":  # queda: pico seguido de inatividade
            s = 24 * 12 + 11 * 12 + 6  # 11h30 do 2º dia
            act[s] = 100
            act[s + 1:s + 1 + 8] = rng.gamma(1.0, 0.5, 8)
        frames.append(pd.DataFrame({"timestamp": ts, "patient_id": pid,
                                    "activity": np.clip(act, 0, 100).round(1)}))
    return pd.concat(frames, ignore_index=True)


CONSULTATION_TEXTS = {
    "P003": (
        "Bom dia doutora. Estou me sentindo bem melhor hoje. Dormi bem a noite toda, "
        "consegui caminhar no corredor pela manhã e estou me alimentando normalmente. "
        "A dor na perna diminuiu bastante."
    ),
    "P001": (
        "Doutora, eu estou muito cansada. Desde ontem sinto falta de ar quando falo, "
        "e uma fraqueza no corpo todo. Tive febre e calafrios a noite. "
        "Estou com tontura quando levanto e não tenho vontade de comer."
    ),
}


def synthesize_consultation_audio(out_dir, espeak: str = "espeak-ng") -> dict[str, str]:
    """Cria áudios de consulta com voz sintética (pt-BR).

    - P003: fala fluente (referência);
    - P001: fala lenta, monótona e com pausas (simula fadiga/dispneia).
    Requer `espeak-ng` instalado. Retorna {patient_id: caminho_wav}.
    """
    import shutil
    import subprocess
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which(espeak):
        raise RuntimeError("espeak-ng não encontrado; instale com 'apt install espeak-ng'.")
    params = {
        "P003": ["-s", "165", "-p", "55", "-g", "2"],
        "P001": ["-s", "95", "-p", "30", "-g", "22", "-a", "70"],
    }
    paths = {}
    for pid, text in CONSULTATION_TEXTS.items():
        wav = out_dir / f"consulta_{pid}.wav"
        if pid == "P001":  # pausas respiratórias entre frases curtas
            text = text.replace(",", ". ").replace(" e ", ". e ")
        subprocess.run([espeak, "-v", "pt-br", *params[pid], "-w", str(wav), text], check=True)
        (out_dir / f"consulta_{pid}.txt").write_text(CONSULTATION_TEXTS[pid], encoding="utf-8")
        paths[pid] = str(wav)
    return paths
