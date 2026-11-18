"""Analisa um vídeo clínico e gera relatório automático.

Exemplos:
    python scripts/run_video.py data/samples/queda_simulada.mp4 --paciente P004 --modo quarto
    python scripts/run_video.py minha_sessao.mp4 --paciente P010 --modo fisioterapia
    python scripts/run_video.py cirurgia.mp4 --modo cirurgia --backend openpose
    python scripts/run_video.py sessao.mp4 --modo fisioterapia --openpose-json saida_openpose/
"""
import argparse
from datetime import datetime

import _path  # noqa: F401
from hospital_monitor.alerts import AlertManager
from hospital_monitor.config import load_config, output_dir
from hospital_monitor.video.analyzer import VideoAnalyzer, events_to_alerts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--paciente", default="P000")
    ap.add_argument("--modo", default="monitoramento", choices=["fisioterapia", "cirurgia", "monitoramento", "quarto"])
    ap.add_argument("--backend", choices=["yolo", "openpose"], default=None)
    ap.add_argument("--openpose-json", default=None, help="pasta com a saída --write_json do OpenPose")
    ap.add_argument("--max-segundos", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config()
    vcfg = cfg["video"]
    if a.backend:
        vcfg["pose_backend"] = a.backend
    if a.openpose_json:
        vcfg["pose_backend"], vcfg["openpose_json_dir"] = "openpose_json", a.openpose_json
    start = datetime.now()
    rep = VideoAnalyzer(vcfg).analyze(a.video, a.paciente, a.modo, output_dir(cfg, "video"), start, a.max_segundos)
    manager = AlertManager.from_config(cfg)
    manager.emit_many(events_to_alerts(rep, start))
    print("\nRelatório:", rep.outputs["plot"].replace("linha_do_tempo.png", "relatorio_video.md"))
    print("Vídeo anotado:", rep.outputs.get("annotated_video"))
