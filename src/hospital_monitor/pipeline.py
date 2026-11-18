"""Orquestração do fluxo multimodal completo (demonstração ponta a ponta)."""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .alerts import Alert, AlertManager, Severity
from .anomaly.movement import MovementAnomalyDetector
from .anomaly.prescriptions import PrescriptionAnomalyDetector
from .anomaly.vitals import VitalSignsAnomalyDetector
from .config import PROJECT_ROOT, load_config, output_dir
from .data import synthetic
from .fusion import MultimodalFusion
from .report import build_dashboard, plot_alert_timeline, plot_movement, plot_vitals

logger = logging.getLogger(__name__)

# agenda da demonstração: quando cada consulta/vídeo aconteceu
AUDIO_SCHEDULE = {"P001": datetime(2026, 11, 11, 8, 30), "P003": datetime(2026, 11, 11, 9, 0)}
VIDEO_SCHEDULE = {
    "fisioterapia_simulada.mp4": ("P001", "fisioterapia", datetime(2026, 11, 11, 10, 0)),
    "queda_simulada.mp4": ("P004", "monitoramento", datetime(2026, 11, 11, 11, 29, 55)),
}


def prepare_samples(samples: Path) -> dict[str, Path]:
    """Gera (se necessário) os dados sintéticos tabulares e os áudios de exemplo."""
    samples.mkdir(parents=True, exist_ok=True)
    files = {"vitals": samples / "sinais_vitais.csv", "prescriptions": samples / "prescricoes.csv",
             "movement": samples / "movimentacao.csv", "episodes": samples / "episodios_inseridos.csv"}
    if not files["vitals"].exists():
        v, eps = synthetic.generate_vitals()
        v.to_csv(files["vitals"], index=False)
        pd.DataFrame([e.__dict__ for e in eps]).to_csv(files["episodes"], index=False)
    if not files["prescriptions"].exists():
        synthetic.generate_prescriptions().to_csv(files["prescriptions"], index=False)
    if not files["movement"].exists():
        synthetic.generate_movement().to_csv(files["movement"], index=False)
    if not (samples / "consulta_P001.wav").exists() and shutil.which("espeak-ng"):
        synthetic.synthesize_consultation_audio(samples)
    return files


def evaluate_vitals(episodes, injected: pd.DataFrame, scored: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compara episódios detectados com os episódios clínicos inseridos nos dados e com
    o que seria obtido usando apenas a regra NEWS2 >= 5."""
    res, tp_ids = [], set()
    for _, inj in injected.iterrows():
        news_first = None
        if scored is not None:
            g = scored[(scored["patient_id"] == inj["patient_id"]) & (scored["timestamp"] >= pd.Timestamp(inj["start"]))
                       & (scored["timestamp"] <= pd.Timestamp(inj["end"])) & (scored["news2"] >= 5)]
            news_first = g["timestamp"].min() if len(g) else None
        hits = [e for e in episodes if e.patient_id == inj["patient_id"]
                and e.start <= pd.Timestamp(inj["end"]) and e.end >= pd.Timestamp(inj["start"])]
        first = min((e.start for e in hits), default=None)
        tp_ids.update(id(e) for e in hits)
        res.append({"paciente": inj["patient_id"], "episodio": inj["kind"], "inicio_real": str(inj["start"]),
                    "detectado": bool(hits), "primeira_deteccao": str(first) if first else "-",
                    "atraso_min": round((first - pd.Timestamp(inj["start"])).total_seconds() / 60, 0) if first else None,
                    "somente_news2_ge5": str(news_first) if news_first is not None else "não detectaria",
                    "metodos": sorted({m for e in hits for m in e.methods})})
    fp = [e for e in episodes if id(e) not in tp_ids]
    return {"episodios": res, "falsos_positivos": len(fp),
            "recall": sum(r["detectado"] for r in res) / max(len(res), 1)}


def run_demo(cfg: dict[str, Any] | None = None, with_video: bool = True, with_audio: bool = True,
             extra_videos: list[tuple[str, str, str]] | None = None, console: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    out = output_dir(cfg)
    for sub in ("alerts", "figuras"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    jl = out / "alerts" / "alerts.jsonl"
    jl.unlink(missing_ok=True)
    samples = PROJECT_ROOT / cfg.get("paths", {}).get("samples_dir", "data/samples")
    files = prepare_samples(samples)
    manager = AlertManager.from_config(cfg, console=console)
    raw: list[Alert] = []
    images: dict[str, str] = {}
    summary: dict[str, Any] = {}

    # 1) sinais vitais
    vitals = pd.read_csv(files["vitals"], parse_dates=["timestamp"])
    vdet = VitalSignsAnomalyDetector(cfg.get("vitals", {}))
    scored, episodes = vdet.detect(vitals)
    raw += vdet.to_alerts(episodes)
    if files["episodes"].exists():
        summary["avaliacao_sinais_vitais"] = evaluate_vitals(episodes, pd.read_csv(files["episodes"]), scored)
    summary["artefatos_removidos"] = int(scored["artifact"].sum())
    for pid in sorted(vitals["patient_id"].unique()):
        images[f"Sinais vitais {pid}"] = plot_vitals(scored, episodes, pid, out / "figuras" / f"vitais_{pid}.png")

    # 2) prescrições
    presc = pd.read_csv(files["prescriptions"], parse_dates=["timestamp"])
    _, palerts = PrescriptionAnomalyDetector(cfg.get("prescriptions", {})).detect(presc)
    raw += palerts

    # 3) movimentação
    mov = pd.read_csv(files["movement"], parse_dates=["timestamp"])
    mscored, malerts = MovementAnomalyDetector(cfg.get("movement", {})).detect(mov)
    raw += malerts
    images["Movimentação"] = plot_movement(mscored, malerts, out / "figuras" / "movimentacao.png")

    # 4) áudio das consultas
    if with_audio:
        from .audio.analyzer import AudioAnalyzer

        aa = AudioAnalyzer(cfg)
        summary["audio"] = {}
        for pid, when in AUDIO_SCHEDULE.items():
            wav = samples / f"consulta_{pid}.wav"
            if not wav.exists():
                continue
            rep = aa.analyze(wav, pid, out / "audio")
            raw += aa.to_alerts(rep, when)
            images[f"Áudio da consulta {pid}"] = str(out / "audio" / f"audio_{wav.stem}.png")
            summary["audio"][pid] = {"transcricao": rep.transcription_provider, "texto": rep.text_provider,
                                     "indicadores": {k: v["score"] for k, v in rep.indicators.items()
                                                     if k != "componentes"},
                                     "termos": [t["term"] for t in rep.critical_terms if not t["negated"]],
                                     "sentimento": rep.sentiment.get("label")}

    # 5) vídeos
    if with_video:
        from .video.analyzer import VideoAnalyzer, events_to_alerts
        from .video.pose import OpenPoseJsonEstimator

        summary["video"] = {}
        vcfg = cfg.get("video", {})
        yolo = None
        jobs = [(samples / name, pid, mode, when) for name, (pid, mode, when) in VIDEO_SCHEDULE.items()]
        jobs += [(Path(p), pid, mode, datetime(2026, 11, 11, 14, 0)) for p, pid, mode in (extra_videos or [])]
        for path, pid, mode, when in jobs:
            if not path.exists():
                continue
            json_dir = path.parent / path.name.replace("_simulada.mp4", "_openpose_json")
            if json_dir.exists():
                va = VideoAnalyzer(vcfg, pose_estimator=OpenPoseJsonEstimator(json_dir), detector=_detector(vcfg))
            else:
                yolo = yolo or VideoAnalyzer(vcfg)
                va = yolo
            rep = va.analyze(path, pid, mode, out / "video", start_time=when)
            raw += events_to_alerts(rep, when)
            images[f"Vídeo {path.name}"] = rep.outputs["plot"]
            for e in rep.events:
                if e.snapshot and e.severity >= Severity.ALTA:
                    images[f"Evento {e.kind} ({path.name}, {e.time_s:.1f}s)"] = e.snapshot
            summary["video"][path.name] = {"eventos": [(round(e.time_s, 1), e.kind, e.severity.name) for e in rep.events],
                                           "repeticoes": len(rep.repetitions)}

    # 6) fusão + distribuição dos alertas
    fusion = MultimodalFusion(cfg.get("fusion", {}))
    fused = fusion.run(raw)
    manager.emit_many(raw + fused)
    manager.to_json(out / "alerts" / "alerts.json")
    psum = fusion.patient_summary(manager.history)
    images = {"Linha do tempo dos alertas": plot_alert_timeline(manager.history, out / "figuras" / "alertas_timeline.png"),
              **images}
    dash = build_dashboard(manager.history, psum, images, out / "painel_equipe_medica.html")
    summary.update({
        "alertas_total": len(manager.history), "alertas_suprimidos_cooldown": manager.suppressed,
        "alertas_por_origem": pd.Series([a.source for a in manager.history]).value_counts().to_dict(),
        "alertas_por_severidade": pd.Series([a.severity.name for a in manager.history]).value_counts().to_dict(),
        "risco_por_paciente": {k: {**v, "peak_time": str(v["peak_time"])} for k, v in psum.items()},
        "painel": dash,
    })
    with open(out / "resumo_resultados.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return {"summary": summary, "alerts": manager.history, "episodes": episodes, "scored_vitals": scored}


def _detector(vcfg):
    from ultralytics import YOLO

    return YOLO(vcfg.get("detection_model", "yolov8n.pt"))
