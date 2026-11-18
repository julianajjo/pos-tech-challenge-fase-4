"""Leitura de dados do PhysioNet para o detector de sinais vitais.

Formatos suportados:
- BIDMC PPG and Respiration Dataset (``bidmc_XX_Numerics.csv``):
  colunas ``Time [s], HR, PULSE, RESP, SpO2``;
- MIMIC-III/IV Waveform "numerics" (via ``wfdb``, opcional);
- CSV genérico já no padrão do projeto
  (``timestamp, patient_id, hr, sbp, dbp, spo2, rr, temp``).

Download (exemplo BIDMC):
    https://physionet.org/content/bidmc/1.0.0/bidmc_csv/bidmc_01_Numerics.csv
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = ["timestamp", "patient_id", "hr", "sbp", "dbp", "spo2", "rr", "temp"]


def load_bidmc_numerics(path: str | Path, patient_id: str | None = None,
                        start: datetime | None = None, resample: str = "1min") -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    start = start or datetime(2026, 1, 1)
    df["timestamp"] = [start + timedelta(seconds=float(s)) for s in df["Time [s]"]]
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "hr": df.get("HR"),
        "spo2": df.get("SpO2"),
        "rr": df.get("RESP"),
    }).set_index("timestamp")
    if resample:
        out = out.resample(resample).mean()
    out = out.reset_index()
    out["patient_id"] = patient_id or path.stem.replace("_Numerics", "")
    for c in STANDARD_COLUMNS:
        if c not in out:
            out[c] = float("nan")
    return out[STANDARD_COLUMNS]


def load_wfdb_numerics(record: str, pn_dir: str | None = None, patient_id: str | None = None) -> pd.DataFrame:
    """Lê um registro numérico do MIMIC (requer ``pip install wfdb``)."""
    import wfdb

    rec = wfdb.rdrecord(record, pn_dir=pn_dir)
    df = pd.DataFrame(rec.p_signal, columns=[s.upper() for s in rec.sig_name])
    start = rec.base_datetime or datetime(2026, 1, 1)
    df["timestamp"] = [start + timedelta(seconds=i / rec.fs) for i in range(len(df))]
    mapping = {"HR": "hr", "ABPSYS": "sbp", "ABP SYS": "sbp", "NBPSYS": "sbp", "ABPDIAS": "dbp",
               "ABP DIAS": "dbp", "NBPDIAS": "dbp", "SPO2": "spo2", "%SPO2": "spo2", "RESP": "rr"}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df["patient_id"] = patient_id or record
    for c in STANDARD_COLUMNS:
        if c not in df:
            df[c] = float("nan")
    return df[STANDARD_COLUMNS]


def load_vitals_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    for c in missing:
        df[c] = float("nan")
    return df[STANDARD_COLUMNS]
