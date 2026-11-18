"""Biomarcadores vocais extraídos do áudio da consulta.

As medidas são aproximações acústicas usadas na literatura de fala patológica:

- **taxa de articulação** (núcleos silábicos por segundo de fala) e **pausas**:
  fala lentificada e pausas longas aparecem em fadiga, depressão e disartria;
- **variação de pitch (F0)** em semitons: monotonia em fadiga/disartria hipocinética;
- **jitter/shimmer** (instabilidade de frequência/amplitude): qualidade vocal;
- **pausas respiratórias por minuto** e **duração média das frases**: frases curtas
  interrompidas para respirar indicam dispneia;
- **planura espectral** em trechos vozeados: soprosidade;
- **queda de energia** entre o início e o fim da fala: cansaço ao longo da consulta.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class VoiceFeatures:
    duration_s: float
    speech_s: float
    pause_ratio: float
    n_pauses: int
    long_pauses: int
    mean_pause_s: float
    pauses_per_min: float
    mean_phrase_s: float
    syllables: int
    articulation_rate: float  # sílabas por segundo de fala
    rhythm_cv: float
    f0_median_hz: float
    f0_std_st: float
    f0_range_st: float
    jitter: float
    shimmer: float
    spectral_flatness: float
    energy_decline_db: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in asdict(self).items()}


def speech_segments(y: np.ndarray, sr: int, top_db: float = 30, min_gap: float = 0.15) -> list[tuple[float, float]]:
    import librosa

    iv = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    segs: list[list[float]] = []
    for a, b in iv / sr:
        if segs and a - segs[-1][1] < min_gap:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs if b - a > 0.05]


def count_syllables(y: np.ndarray, sr: int, segs: list[tuple[float, float]]) -> tuple[int, np.ndarray]:
    """Picos do envelope de energia (banda de voz) como estimativa de núcleos silábicos."""
    import librosa
    from scipy.signal import butter, find_peaks, sosfiltfilt

    sos = butter(4, [300, 3000], btype="band", fs=sr, output="sos")
    yb = sosfiltfilt(sos, y)
    hop = int(0.01 * sr)
    env = librosa.feature.rms(y=yb, frame_length=int(0.04 * sr), hop_length=hop)[0]
    env = np.convolve(env, np.hanning(7) / np.hanning(7).sum(), mode="same")
    times = np.arange(len(env)) * hop / sr
    peaks, _ = find_peaks(env, distance=int(0.1 / 0.01), prominence=0.12 * np.percentile(env, 95))
    in_speech = [p for p in peaks if any(a <= times[p] <= b for a, b in segs)]
    return len(in_speech), times[in_speech]


def extract_features(y: np.ndarray, sr: int, top_db: float = 30, long_pause_s: float = 1.0) -> VoiceFeatures:
    import librosa

    y = librosa.util.normalize(y.astype(float))
    duration = len(y) / sr
    segs = speech_segments(y, sr, top_db)
    if not segs:
        raise ValueError("Nenhum trecho de fala detectado no áudio.")
    speech = sum(b - a for a, b in segs)
    span = segs[-1][1] - segs[0][0]
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    pauses = [g for g in gaps if g >= 0.25]
    n_syl, syl_t = count_syllables(y, sr, segs)
    ioi = np.diff(syl_t)
    ioi = ioi[ioi < 0.6]  # intervalos dentro da mesma frase
    rhythm_cv = float(np.std(ioi) / np.mean(ioi)) if len(ioi) > 3 else 0.0

    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    f0v = f0[voiced & ~np.isnan(f0)]
    if len(f0v) > 10:
        med = float(np.median(f0v))
        st = 12 * np.log2(f0v / med)
        f0_std, f0_range = float(np.std(st)), float(np.percentile(st, 95) - np.percentile(st, 5))
        periods = 1 / f0v
        jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    else:
        med = f0_std = f0_range = jitter = 0.0
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    n = min(len(rms), len(voiced))
    amp = rms[:n][voiced[:n]]
    shimmer = float(np.mean(np.abs(np.diff(amp))) / np.mean(amp)) if len(amp) > 10 else 0.0
    flat = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)[0][:n]
    flatness = float(np.mean(flat[voiced[:n]])) if voiced[:n].any() else float(np.mean(flat))

    # energia (dB) no primeiro e no último terço do tempo de fala
    speech_rms = np.concatenate([rms[int(a * sr / 256):int(b * sr / 256)] for a, b in segs])
    third = max(1, len(speech_rms) // 3)
    db = 20 * np.log10(speech_rms + 1e-8)
    decline = float(np.mean(db[:third]) - np.mean(db[-third:]))

    return VoiceFeatures(
        duration_s=duration, speech_s=speech, pause_ratio=1 - speech / max(span, 1e-6), n_pauses=len(pauses),
        long_pauses=sum(p >= long_pause_s for p in pauses), mean_pause_s=float(np.mean(pauses)) if pauses else 0.0,
        pauses_per_min=60 * len(pauses) / max(span, 1e-6), mean_phrase_s=speech / len(segs), syllables=n_syl,
        articulation_rate=n_syl / max(speech, 1e-6), rhythm_cv=rhythm_cv, f0_median_hz=med, f0_std_st=f0_std,
        f0_range_st=f0_range, jitter=jitter, shimmer=shimmer, spectral_flatness=flatness,
        energy_decline_db=decline,
    )


def _scale(x: float, normal: float, abnormal: float) -> float:
    """0 no valor normal, 1 no valor alterado (funciona nos dois sentidos)."""
    if abnormal == normal:
        return 0.0
    return float(np.clip((x - normal) / (abnormal - normal), 0, 1))


def vocal_indicators(f: VoiceFeatures, thr: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Escores (0-1) de fadiga, disartria e dificuldade respiratória, com justificativas."""
    rate_low = thr.get("speech_rate_low", 2.6)
    pause_hi = thr.get("pause_ratio_high", 0.35)
    mono = thr.get("pitch_std_low_st", 1.4)
    jit = thr.get("jitter_high", 0.025)
    shim = thr.get("shimmer_high", 0.30)
    bpm = thr.get("breath_pauses_per_min_high", 18)
    dec = thr.get("energy_decline_db", 3.0)

    comp = {
        "fala_lenta": _scale(f.articulation_rate, rate_low + 1.5, rate_low),
        "pausas": _scale(f.pause_ratio, pause_hi - 0.2, pause_hi),
        "pausas_longas": _scale(f.long_pauses, 0, 4),
        "monotonia": _scale(f.f0_std_st, mono + 1.5, mono),
        "queda_energia": _scale(f.energy_decline_db, 0, dec),
        "jitter": _scale(f.jitter, jit / 2, jit),
        "shimmer": _scale(f.shimmer, shim / 2, shim),
        "ritmo_irregular": _scale(f.rhythm_cv, 0.35, 0.7),
        "pausas_respiratorias": _scale(f.pauses_per_min, bpm / 2, bpm),
        "frases_curtas": _scale(f.mean_phrase_s, 2.5, 1.0),
        "soprosidade": _scale(f.spectral_flatness, 0.02, 0.08),
    }
    groups = {
        "fadiga": {"fala_lenta": 0.3, "monotonia": 0.25, "queda_energia": 0.2, "pausas_longas": 0.25},
        "disartria": {"fala_lenta": 0.3, "ritmo_irregular": 0.25, "jitter": 0.25, "shimmer": 0.2},
        "dificuldade_respiratoria": {"pausas_respiratorias": 0.35, "frases_curtas": 0.3, "pausas": 0.2,
                                     "soprosidade": 0.15},
    }
    out = {}
    for name, weights in groups.items():
        score = sum(comp[k] * w for k, w in weights.items())
        reasons = [k for k in weights if comp[k] >= 0.5]
        out[name] = {"score": round(float(score), 3), "evidencias": reasons}
    out["componentes"] = {k: round(v, 3) for k, v in comp.items()}
    return out
