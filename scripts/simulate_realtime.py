"""Simula o monitoramento em tempo real de sinais vitais (uma amostra por minuto).

As amostras do arquivo de exemplo são "tocadas" em velocidade acelerada e passam pelo
``StreamingVitalsMonitor``; os alertas aparecem no console e em outputs/alerts/.

    python scripts/simulate_realtime.py --paciente P002 --inicio "2026-11-10 19:30" --minutos 60 --velocidade 0.05
"""
import argparse
import time

import pandas as pd

import _path  # noqa: F401
from hospital_monitor.alerts import AlertManager
from hospital_monitor.anomaly.vitals import StreamingVitalsMonitor
from hospital_monitor.config import PROJECT_ROOT, load_config
from hospital_monitor.pipeline import prepare_samples

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paciente", default="P002")
    ap.add_argument("--inicio", default="2026-11-10 19:30")
    ap.add_argument("--minutos", type=int, default=60)
    ap.add_argument("--velocidade", type=float, default=0.05, help="segundos reais por minuto simulado")
    a = ap.parse_args()
    cfg = load_config()
    files = prepare_samples(PROJECT_ROOT / cfg["paths"]["samples_dir"])
    df = pd.read_csv(files["vitals"], parse_dates=["timestamp"])
    df = df[df["patient_id"] == a.paciente].sort_values("timestamp")
    t0 = pd.Timestamp(a.inicio)
    baseline = df[df["timestamp"] < df["timestamp"].iloc[0] + pd.Timedelta(hours=cfg["vitals"]["baseline_hours"])]
    history = df[df["timestamp"] < t0].tail(90)
    mon = StreamingVitalsMonitor(a.paciente, pd.concat([baseline, history]).drop_duplicates("timestamp"), cfg["vitals"])
    manager = AlertManager.from_config(cfg)
    stream = df[(df["timestamp"] >= t0) & (df["timestamp"] < t0 + pd.Timedelta(minutes=a.minutos))]
    print(f"Monitorando {a.paciente}: {len(stream)} amostras a partir de {t0}")
    for _, row in stream.iterrows():
        print(f"\r{row['timestamp']:%d/%m %H:%M}  FC {row['hr']:>4.0f}  PA {row['sbp']:>3.0f}/{row['dbp']:<3.0f} "
              f"SpO2 {row['spo2']:>5.1f}  FR {row['rr']:>3.0f}  T {row['temp']:.1f}", end="", flush=True)
        alert = mon.update(row.to_dict())
        if alert:
            print()
            manager.emit(alert)
        time.sleep(a.velocidade)
    print()
