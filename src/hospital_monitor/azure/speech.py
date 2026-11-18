"""Integração com o Azure AI Speech (Speech to Text).

Transcreve o áudio da consulta com reconhecimento contínuo, retornando segmentos
com tempo de início/fim, confiança e marcação de tempo por palavra (usada para
medir velocidade de fala e pausas).

Variáveis de ambiente: ``AZURE_SPEECH_KEY`` e ``AZURE_SPEECH_REGION``.
"""
from __future__ import annotations

import json
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "pt-BR"
    provider: str = "azure"

    @property
    def words(self) -> list[dict[str, Any]]:
        return [w for s in self.segments for w in s.get("words", [])]


def _to_wav16k(path: str | Path) -> str:
    """O SDK aceita WAV PCM 16 kHz mono; converte qualquer formato suportado pelo librosa."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(path), sr=16000, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, 16000, subtype="PCM_16")
    return tmp.name


class AzureSpeechTranscriber:
    def __init__(self, key: str, region: str, language: str = "pt-BR", timeout_s: float = 600):
        if not key or not region:
            raise ValueError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION não configurados.")
        import azure.cognitiveservices.speech as speechsdk

        self.sdk = speechsdk
        self.config = speechsdk.SpeechConfig(subscription=key, region=region)
        self.config.speech_recognition_language = language
        self.config.output_format = speechsdk.OutputFormat.Detailed
        self.config.request_word_level_timestamps()
        self.language = language
        self.timeout_s = timeout_s

    def transcribe(self, audio_path: str | Path) -> Transcript:
        sdk = self.sdk
        wav = _to_wav16k(audio_path)
        audio_config = sdk.audio.AudioConfig(filename=wav)
        recognizer = sdk.SpeechRecognizer(speech_config=self.config, audio_config=audio_config)
        segments: list[dict[str, Any]] = []
        errors: list[str] = []
        done = threading.Event()

        def on_recognized(evt):
            res = evt.result
            if res.reason != sdk.ResultReason.RecognizedSpeech or not res.text:
                return
            detail = json.loads(res.json)
            best = (detail.get("NBest") or [{}])[0]
            segments.append({
                "text": res.text,
                "start_s": res.offset / 1e7,
                "end_s": (res.offset + res.duration) / 1e7,
                "confidence": best.get("Confidence"),
                "words": [{"word": w["Word"], "start_s": w["Offset"] / 1e7,
                           "end_s": (w["Offset"] + w["Duration"]) / 1e7} for w in best.get("Words", [])],
            })

        def on_canceled(evt):
            det = evt.cancellation_details
            if det.reason == sdk.CancellationReason.Error:
                errors.append(f"{det.error_code}: {det.error_details}")
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(lambda _: done.set())
        recognizer.start_continuous_recognition()
        done.wait(self.timeout_s)
        recognizer.stop_continuous_recognition()
        Path(wav).unlink(missing_ok=True)
        if errors:
            raise RuntimeError("Azure Speech retornou erro: " + "; ".join(errors))
        segments.sort(key=lambda s: s["start_s"])
        return Transcript(" ".join(s["text"] for s in segments), segments, self.language, "azure")


class LocalTranscriber:
    """Alternativa sem nuvem: usa ``faster-whisper`` se instalado ou, para os dados de
    demonstração, a transcrição de referência ``<arquivo>.txt`` ao lado do áudio."""

    def __init__(self, language: str = "pt", model_size: str = "small"):
        self.language = language
        self.model_size = model_size

    def transcribe(self, audio_path: str | Path) -> Transcript:
        audio_path = Path(audio_path)
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_size, compute_type="int8")
            segs, _ = model.transcribe(str(audio_path), language=self.language, word_timestamps=True)
            segments = [{"text": s.text.strip(), "start_s": s.start, "end_s": s.end, "confidence": None,
                         "words": [{"word": w.word.strip(), "start_s": w.start, "end_s": w.end} for w in (s.words or [])]}
                        for s in segs]
            return Transcript(" ".join(s["text"] for s in segments), segments, self.language, "faster-whisper")
        except ImportError:
            pass
        sidecar = audio_path.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            return Transcript(text, [{"text": text, "start_s": 0.0, "end_s": None, "words": []}], self.language,
                              "transcricao_referencia")
        logger.warning("Sem Azure, faster-whisper ou transcrição de referência para %s", audio_path)
        return Transcript("", [], self.language, "indisponivel")


def build_transcriber(azure_cfg: dict[str, Any]):
    if azure_cfg.get("speech_key") and azure_cfg.get("speech_region"):
        return AzureSpeechTranscriber(azure_cfg["speech_key"], azure_cfg["speech_region"],
                                      azure_cfg.get("speech_language", "pt-BR"))
    logger.info("Azure Speech não configurado: usando transcrição local.")
    return LocalTranscriber()
