"""Executa o fluxo multimodal completo e gera o painel da equipe médica.

Uso:
    python scripts/run_demo.py                 # tudo (vídeo, áudio, sinais vitais, prescrições, movimentação)
    python scripts/run_demo.py --sem-video     # mais rápido
    python scripts/run_demo.py --video meu_video.mp4 P010 fisioterapia
"""
import argparse
import json
import logging

import _path  # noqa: F401
from hospital_monitor.pipeline import run_demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-video", action="store_true")
    ap.add_argument("--sem-audio", action="store_true")
    ap.add_argument("--video", nargs=3, action="append", metavar=("ARQUIVO", "PACIENTE", "MODO"),
                    help="vídeo extra; MODO = fisioterapia | cirurgia | monitoramento")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    res = run_demo(with_video=not a.sem_video, with_audio=not a.sem_audio, extra_videos=a.video)
    s = res["summary"]
    print("\n===== RESUMO =====")
    print(json.dumps({k: s[k] for k in ("alertas_total", "alertas_por_origem", "alertas_por_severidade")},
                     ensure_ascii=False, indent=2))
    for pid, r in s["risco_por_paciente"].items():
        print(f"{pid}: risco máximo {r['peak_risk']:.2f} ({r['severity']})")
    print("Painel:", s["painel"])
