"""Analisa o áudio de uma consulta (Azure Speech + Azure Text Analytics + biomarcadores vocais).

Exemplos:
    python scripts/run_audio.py data/samples/consulta_P001.wav --paciente P001
    python scripts/run_audio.py minha_gravacao.m4a --paciente P010
Sem as chaves do Azure no .env, usa os modos locais.
"""
import argparse
import json
from datetime import datetime

import _path  # noqa: F401
from hospital_monitor.alerts import AlertManager
from hospital_monitor.audio.analyzer import AudioAnalyzer
from hospital_monitor.config import load_config, output_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--paciente", default="P000")
    a = ap.parse_args()
    cfg = load_config()
    aa = AudioAnalyzer(cfg)
    rep = aa.analyze(a.audio, a.paciente, output_dir(cfg, "audio"))
    print(f"Transcrição ({rep.transcription_provider}): {rep.transcript}\n")
    print("Sentimento:", json.dumps(rep.sentiment, ensure_ascii=False))
    print("Termos críticos:", [(t["term"], t["severity"], "negado" if t["negated"] else "") for t in rep.critical_terms])
    if rep.health_entities:
        print("Entidades de saúde (Azure):", [(e["text"], e["category"]) for e in rep.health_entities])
    print("Indicadores vocais:", {k: v["score"] for k, v in rep.indicators.items() if k != "componentes"}, "\n")
    AlertManager.from_config(cfg).emit_many(aa.to_alerts(rep, datetime.now()))
