"""Pipeline de análise de áudio de consultas.

áudio → [Azure Speech to Text] → transcrição
      → [biomarcadores vocais (librosa)] → fadiga / disartria / dificuldade respiratória
transcrição → [Azure Text Analytics + Health] → sentimento, entidades clínicas, negações
            → [léxico de termos críticos] → alertas
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..alerts import Alert, Severity
from ..azure.language import build_text_analyzer
from ..azure.speech import Transcript, build_transcriber
from .features import extract_features, vocal_indicators
from .local_text import find_critical_terms

logger = logging.getLogger(__name__)

INDICATOR_NAMES = {"fadiga": "fadiga/cansaço", "disartria": "alteração articulatória (disartria)",
                   "dificuldade_respiratoria": "dificuldade respiratória"}


@dataclass
class AudioReport:
    audio: str
    patient_id: str
    transcript: str
    transcription_provider: str
    text_provider: str
    features: dict[str, Any]
    indicators: dict[str, Any]
    sentiment: dict[str, Any]
    critical_terms: list[dict[str, Any]]
    key_phrases: list[str] = field(default_factory=list)
    health_entities: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    sentences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioAnalyzer:
    def __init__(self, cfg: dict[str, Any], transcriber=None, text_analyzer=None):
        self.cfg = cfg.get("audio", {})
        az = cfg.get("azure", {})
        self.transcriber = transcriber or build_transcriber(az)
        self.text = text_analyzer or build_text_analyzer(az)

    def analyze(self, audio_path: str | Path, patient_id: str, out_dir: str | Path | None = None) -> AudioReport:
        import librosa

        audio_path = Path(audio_path)
        y, sr = librosa.load(str(audio_path), sr=self.cfg.get("sample_rate", 16000), mono=True)
        feats = extract_features(y, sr, self.cfg.get("top_db", 30), self.cfg.get("thresholds", {}).get("long_pause_s", 1.0))
        ind = vocal_indicators(feats, self.cfg.get("thresholds", {}))

        transcript: Transcript = self.transcriber.transcribe(audio_path)
        words = transcript.words
        if len(words) > 5:  # com marcação por palavra do Azure, mede palavras/min reais
            span = words[-1]["end_s"] - words[0]["start_s"]
            feats_extra = {"words_per_min": round(60 * len(words) / max(span, 1e-6), 1)}
        else:
            feats_extra = {}
        text_res = self.text.analyze(transcript.text) if transcript.text else {
            "provider": "n/a", "sentiment": {}, "sentences": [], "key_phrases": [], "health_entities": []}
        terms = find_critical_terms(transcript.text, text_res.get("health_entities"))

        rep = AudioReport(str(audio_path), patient_id, transcript.text, transcript.provider, text_res["provider"],
                          {**feats.to_dict(), **feats_extra}, ind, text_res.get("sentiment", {}), terms,
                          text_res.get("key_phrases", []), text_res.get("health_entities", []),
                          transcript.segments, text_res.get("sentences", []))
        if out_dir:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"audio_{audio_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(rep.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            self.plot(y, sr, rep, out / f"audio_{audio_path.stem}.png")
        return rep

    @staticmethod
    def plot(y, sr, rep: AudioReport, path: Path) -> None:
        import librosa
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from .features import speech_segments

        fig, ax = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [2, 1]})
        t = np.arange(len(y)) / sr
        ax[0].plot(t, y, lw=0.4, color="#475569")
        for a, b in speech_segments(librosa.util.normalize(y), sr):
            ax[0].axvspan(a, b, color="#2563eb", alpha=0.12)
        ax[0].set_title(f"{Path(rep.audio).name} – trechos de fala (azul) e pausas")
        ax[0].set_xlabel("Tempo (s)")
        names = ["fadiga", "disartria", "dificuldade_respiratoria"]
        vals = [rep.indicators[n]["score"] for n in names]
        colors = ["#dc2626" if v >= 0.7 else "#d97706" if v >= 0.45 else "#16a34a" for v in vals]
        ax[1].barh(["Fadiga", "Disartria", "Dif. respiratória"], vals, color=colors)
        ax[1].set_xlim(0, 1)
        ax[1].axvline(0.45, ls="--", color="#6b7280", lw=1)
        ax[1].set_title("Indicadores vocais (0 = normal, 1 = muito alterado)")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    @staticmethod
    def to_alerts(rep: AudioReport, when: datetime) -> list[Alert]:
        alerts = []
        for name, label in INDICATOR_NAMES.items():
            s = rep.indicators[name]["score"]
            if s >= 0.45:
                sev = Severity.ALTA if s >= 0.7 else Severity.MEDIA
                ev = ", ".join(e.replace("_", " ") for e in rep.indicators[name]["evidencias"])
                alerts.append(Alert(rep.patient_id, "audio", sev, f"Voz: sinais de {label}",
                                    f"Escore {s:.2f}. Evidências acústicas: {ev}. Taxa de articulação "
                                    f"{rep.features['articulation_rate']:.1f} sílabas/s, pausas "
                                    f"{rep.features['pause_ratio']:.0%} do tempo, "
                                    f"{rep.features['pauses_per_min']:.0f} pausas/min.",
                                    timestamp=when, evidence={"indicator": name, "score": s,
                                                              "audio": Path(rep.audio).name},
                                    recommended_action="Correlacionar com sinais vitais e exame físico.", score=s))
        active = [t for t in rep.critical_terms if not t["negated"]]
        if active:
            worst = max(Severity[t["severity"]] for t in active)
            terms = ", ".join(sorted({t["term"] for t in active}))
            neg = rep.sentiment.get("negative", 0)
            if worst >= Severity.MEDIA:
                alerts.append(Alert(rep.patient_id, "audio", worst, "Consulta: termos clínicos críticos",
                                    f"Paciente relatou: {terms}. Sentimento predominante: "
                                    f"{rep.sentiment.get('label', 'n/d')} (negativo {neg:.0%}). "
                                    f"Transcrição via {rep.transcription_provider}; análise de texto via "
                                    f"{rep.text_provider}.",
                                    timestamp=when, evidence={"terms": [t["term"] for t in active],
                                                              "sentiment": rep.sentiment},
                                    recommended_action="Revisar queixas com o paciente e registrar no prontuário.",
                                    score={4: 0.9, 3: 0.7, 2: 0.45}.get(int(worst), 0.2)))
        return alerts
