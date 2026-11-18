"""Análise de vídeos clínicos (fisioterapia, cirurgia, monitoramento de quarto/corredor).

Fluxo:
1. leitura do vídeo (1 a cada ``frame_stride`` frames);
2. pose humana (YOLOv8-pose ou OpenPose) com rastreamento por pessoa;
3. detecção de objetos (YOLOv8) e verificação de zonas críticas;
4. extração de características biomecânicas por pessoa/frame;
5. detecção de eventos fora do padrão (quedas, execução incorreta do exercício,
   postura atípica via Isolation Forest, objetos/pessoas em área crítica, instrumento
   remanescente no campo);
6. geração de vídeo anotado, imagens dos eventos, gráficos e relatório automático.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity
from .kinematics import analyze_repetitions, detect_falls, detect_lying, frame_features, stitch_tracks
from .pose import SKELETON, build_pose_estimator

logger = logging.getLogger(__name__)


@dataclass
class VideoEvent:
    time_s: float
    kind: str
    severity: Severity
    description: str
    track_id: int = -1
    snapshot: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoReport:
    video: str
    patient_id: str
    mode: str
    fps: float
    duration_s: float
    frames_processed: int
    people_tracked: int
    object_counts: dict[str, int]
    events: list[VideoEvent]
    repetitions: list[dict[str, Any]]
    outputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for e in d["events"]:
            e["severity"] = Severity(e["severity"]).name
        return d


class VideoAnalyzer:
    def __init__(self, cfg: dict[str, Any], pose_estimator=None, detector=None):
        self.cfg = cfg
        self.pose = pose_estimator or build_pose_estimator(cfg)
        if detector is None:
            from ultralytics import YOLO

            detector = YOLO(cfg.get("detection_model", "yolov8n.pt"))
        self.detector = detector
        self.stride = cfg.get("frame_stride", 2)
        self.conf = cfg.get("confidence", 0.35)
        self.kp_thr = cfg.get("keypoint_confidence", 0.3)

    # ------------------------------------------------------------------ util
    def _zones(self, w: int, h: int, mode: str) -> list[dict[str, Any]]:
        zones = []
        cfg_z = self.cfg.get("zones", {})
        for z in (cfg_z.get(mode, []) if isinstance(cfg_z, dict) else cfg_z) or []:
            poly = (np.array(z["polygon"], dtype=float) * [w, h]).astype(np.int32)
            zones.append({**z, "poly": poly})
        return zones

    @staticmethod
    def _in(poly: np.ndarray, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def _detect_objects(self, frame: np.ndarray) -> list[dict[str, Any]]:
        res = self.detector(frame, conf=self.conf, verbose=False)[0]
        names = res.names
        out = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            out.append({"cls": names[int(k)], "conf": float(c), "box": b.tolist()})
        return out

    # --------------------------------------------------------------- análise
    def analyze(self, video_path: str | Path, patient_id: str = "P000", mode: str = "monitoramento",
                out_dir: str | Path = "outputs/video", start_time: datetime | None = None,
                max_seconds: float | None = None, write_video: bool = True) -> VideoReport:
        video_path = Path(video_path)
        out_dir = Path(out_dir) / video_path.stem
        (out_dir / "eventos").mkdir(parents=True, exist_ok=True)
        start_time = start_time or datetime.now()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zones = self._zones(w, h, mode)
        rows, obj_rows, overlays = [], [], {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if max_seconds and t > max_seconds:
                break
            if idx % self.stride == 0:
                people = self.pose(frame, frame_idx=idx)
                objs = self._detect_objects(frame) if mode != "fisioterapia" else []
                overlays[idx] = (people, objs)
                for p in people:
                    f = frame_features(p.keypoints, p.box, self.kp_thr)
                    foot_x, foot_y = (p.box[0] + p.box[2]) / 2, p.box[3]
                    f.update(frame=idx, time_s=t, track_id=p.track_id, conf=p.score,
                             **{f"in_{z['name']}": self._in(z["poly"], foot_x, foot_y) for z in zones})
                    rows.append(f)
                for o in objs:
                    cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
                    obj_rows.append({"frame": idx, "time_s": t, "cls": o["cls"], "conf": o["conf"],
                                     **{f"in_{z['name']}": self._in(z["poly"], cx, cy) for z in zones}})
            idx += 1
        cap.release()
        duration = idx / fps
        frames = pd.DataFrame(rows)
        if not frames.empty:
            frames = stitch_tracks(frames)
        objects = pd.DataFrame(obj_rows)
        frames.to_csv(out_dir / "pose_frames.csv", index=False)
        objects.to_csv(out_dir / "objetos.csv", index=False)

        events: list[VideoEvent] = []
        reps: list[dict[str, Any]] = []
        if not frames.empty:
            events += self._fall_events(frames)
            events += self._zone_events(frames, objects, zones, duration)
            if mode == "fisioterapia":
                rep_events, reps = self._exercise_events(frames)
                events += rep_events
                events += self._postural_outliers(frames)
        events.sort(key=lambda e: e.time_s)

        outputs = {"frames_csv": str(out_dir / "pose_frames.csv"), "objects_csv": str(out_dir / "objetos.csv")}
        self._snapshots(video_path, events, overlays, zones, out_dir / "eventos")
        if write_video:
            outputs["annotated_video"] = self._write_annotated(video_path, overlays, events, zones, fps,
                                                               out_dir / "video_anotado.mp4")
        outputs["plot"] = self._plot(frames, events, reps, mode, out_dir / "linha_do_tempo.png")
        counts = objects["cls"].value_counts().to_dict() if not objects.empty else {}
        report = VideoReport(str(video_path), patient_id, mode, fps, duration, len(overlays),
                             int(frames["track_id"].nunique()) if not frames.empty else 0,
                             {k: int(v) for k, v in counts.items()}, events, reps, outputs)
        self._write_report(report, out_dir, start_time)
        self.start_time = start_time
        return report

    # ------------------------------------------------------------- eventos
    def _fall_events(self, frames: pd.DataFrame) -> list[VideoEvent]:
        fc = self.cfg.get("fall", {})
        ev = []
        for tid, g in frames.groupby("track_id"):
            if len(g) < 5:
                continue
            g = g.sort_values("time_s")
            falls = detect_falls(g, fc.get("hip_drop_ratio", 0.25), fc.get("lying_aspect_ratio", 1.3))
            for f in falls:
                ev.append(VideoEvent(f["time_s"], "queda", Severity.CRITICA,
                                     f"Possível queda da pessoa #{tid}: quadril desceu {f['hip_drop_ratio']:.0%} da "
                                     f"altura corporal em <1 s e terminou em posição horizontal.", int(tid), data=f))
            for ly in detect_lying(g, fc.get("lying_aspect_ratio", 1.3), fc.get("lying_min_s", 2.0)):
                if any(abs(ly["time_s"] - f["time_s"]) < 3 for f in falls):
                    continue  # já coberto pelo evento de queda
                ev.append(VideoEvent(ly["time_s"], "pessoa_no_chao", Severity.ALTA,
                                     f"Pessoa #{tid} em posição horizontal por {ly['duration_s']:.0f} s "
                                     "fora do leito (sem queda observada).", int(tid), data=ly))
        return ev

    def _zone_events(self, frames: pd.DataFrame, objects: pd.DataFrame, zones: list[dict], duration: float):
        ev = []
        instr = set(self.cfg.get("instrument_classes", []))
        for z in zones:
            col = f"in_{z['name']}"
            if col in frames:
                per_frame = frames[frames[col]].groupby("time_s")["track_id"].nunique()
                over = per_frame[per_frame > z.get("max_people", 99)]
                if len(over) >= 3:
                    t0 = float(over.index[0])
                    ev.append(VideoEvent(t0, "lotacao_zona", Severity.MEDIA,
                                         f"{int(over.max())} pessoas simultâneas na zona '{z['name']}' "
                                         f"(limite {z['max_people']}) por {len(over)} amostras.",
                                         data={"zone": z["name"], "max_people": int(over.max())}))
            if not objects.empty and col in objects:
                forb = objects[objects[col] & objects["cls"].isin(z.get("forbidden_classes", []))]
                for cls, g in forb.groupby("cls"):
                    if len(g) >= 3:  # evita falsos positivos de 1 frame
                        ev.append(VideoEvent(float(g["time_s"].iloc[0]), "objeto_zona_critica",
                                             Severity.ALTA if cls in instr else Severity.MEDIA,
                                             f"Objeto '{cls}' detectado na zona '{z['name']}' "
                                             f"({len(g)} detecções, confiança média {g['conf'].mean():.2f}).",
                                             data={"zone": z["name"], "class": cls}))
                tail = forb[(forb["time_s"] > duration * 0.9) & forb["cls"].isin(instr)]
                if len(tail) >= 2:
                    ev.append(VideoEvent(float(tail["time_s"].iloc[0]), "instrumento_remanescente", Severity.CRITICA,
                                         f"Instrumento ('{tail['cls'].iloc[0]}') ainda presente no campo ao final "
                                         "do procedimento. Conferir contagem de instrumentais.",
                                         data={"zone": z["name"]}))
        return ev

    def _main_track(self, frames: pd.DataFrame) -> pd.DataFrame:
        main = frames.groupby("track_id")["box_h"].agg(["size", "median"])
        tid = (main["size"] * main["median"]).idxmax()
        return frames[frames["track_id"] == tid].sort_values("time_s").reset_index(drop=True)

    def _exercise_events(self, frames: pd.DataFrame):
        ex = self.cfg.get("exercise", {})
        track = self._main_track(frames)
        reps = analyze_repetitions(track, ex)
        ev = []
        for r in reps:
            if r.issues:
                ev.append(VideoEvent(r.bottom_s, "execucao_incorreta", Severity.MEDIA,
                                     f"Repetição {r.index} de {ex.get('name', 'exercício')}: " + "; ".join(r.issues) + ".",
                                     int(track["track_id"].iloc[0]), data=asdict(r)))
        bad = sum(1 for r in reps if r.issues)
        if reps and bad / len(reps) >= 0.5:
            ev.append(VideoEvent(reps[-1].end_s, "sessao_fora_do_padrao", Severity.ALTA,
                                 f"{bad} de {len(reps)} repetições fora do padrão esperado. "
                                 "Sugere fadiga, dor ou necessidade de ajustar o exercício."))
        return ev, [asdict(r) for r in reps]

    def _postural_outliers(self, frames: pd.DataFrame) -> list[VideoEvent]:
        track = self._main_track(frames)
        cols = ["knee_mean", "trunk_inclination", "aspect_ratio", "l_elbow", "r_elbow", "l_hip_angle", "r_hip_angle"]
        X = track[cols].interpolate(limit_direction="both").fillna(0)
        X["hip_vy"] = track["hip_y"].diff().fillna(0) / track["box_h"].clip(lower=1)
        if len(X) < 30:
            return []
        iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=0).fit(X.values)
        flag = pd.Series(iso.predict(X.values) == -1)
        seg = (flag != flag.shift()).cumsum()
        ev = []
        for _, g in track[flag].groupby(seg[flag]):
            if g["time_s"].iloc[-1] - g["time_s"].iloc[0] >= 0.4:
                ev.append(VideoEvent(float(g["time_s"].iloc[0]), "postura_atipica", Severity.BAIXA,
                                     f"Padrão postural atípico por {g['time_s'].iloc[-1] - g['time_s'].iloc[0]:.1f} s "
                                     "(Isolation Forest sobre ângulos articulares).", int(g["track_id"].iloc[0])))
        return ev[:5]

    # -------------------------------------------------------------- desenho
    def _draw(self, frame: np.ndarray, people, objs, zones, banners: list[str]) -> np.ndarray:
        for z in zones:
            ov = frame.copy()
            cv2.fillPoly(ov, [z["poly"]], (0, 0, 180))
            frame = cv2.addWeighted(ov, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [z["poly"]], True, (0, 0, 220), 2)
            cv2.putText(frame, z["name"], tuple(int(v) for v in z["poly"][0] + [5, 20]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 220), 2)
        for o in objs:
            if o["cls"] == "person":
                continue
            x1, y1, x2, y2 = map(int, o["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f"{o['cls']} {o['conf']:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 1)
        for p in people:
            k = p.keypoints
            for a, b in SKELETON:
                if k[a, 2] > self.kp_thr and k[b, 2] > self.kp_thr:
                    cv2.line(frame, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), (80, 220, 80), 2)
            for x, y, c in k:
                if c > self.kp_thr:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 1)
            cv2.putText(frame, f"#{p.track_id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
        for i, text in enumerate(banners[:3]):
            y = 30 + i * 30
            cv2.rectangle(frame, (0, y - 22), (frame.shape[1], y + 8), (0, 0, 160), -1)
            cv2.putText(frame, text[:95], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _ascii(s: str) -> str:
        import unicodedata

        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    def _banners(self, events: list[VideoEvent], t: float) -> list[str]:
        return [self._ascii(f"ALERTA {e.severity.name}: {e.kind.replace('_', ' ')} (t={e.time_s:.1f}s)")
                for e in events if e.time_s <= t <= e.time_s + 2.5 and e.severity >= Severity.MEDIA]

    def _nearest_overlay(self, overlays: dict, idx: int):
        keys = [k for k in overlays if k <= idx]
        return overlays[max(keys)] if keys else ([], [])

    def _snapshots(self, video_path, events, overlays, zones, out: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for i, e in enumerate(events):
            fidx = int(round(e.time_s * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            people, objs = self._nearest_overlay(overlays, fidx)
            img = self._draw(frame, people, objs, zones, self._banners([e], e.time_s))
            path = out / f"evento_{i + 1:02d}_{e.kind}.jpg"
            cv2.imwrite(str(path), img)
            e.snapshot = str(path)
        cap.release()

    def _write_annotated(self, video_path, overlays, events, zones, fps, out: Path) -> str:
        cap = cv2.VideoCapture(str(video_path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = out.with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or idx > max(overlays, default=0) + self.stride:
                break
            people, objs = self._nearest_overlay(overlays, idx)
            writer.write(self._draw(frame, people, objs, zones, self._banners(events, idx / fps)))
            idx += 1
        writer.release()
        cap.release()
        if shutil.which("ffmpeg"):  # converte para H.264 (compatível com navegadores/YouTube)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "26", str(out)], check=False)
            if out.exists():
                tmp.unlink(missing_ok=True)
                return str(out)
        tmp.rename(out)
        return str(out)

    def _plot(self, frames, events, reps, mode, out: Path) -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        if not frames.empty:
            if mode == "fisioterapia":
                tr = self._main_track(frames)
                axes[0].plot(tr["time_s"], tr["knee_mean"], label="Ângulo do joelho (°)", color="#2563eb")
                axes[0].plot(tr["time_s"], tr["trunk_inclination"], label="Inclinação do tronco (°)", color="#d97706")
                axes[0].axhline(self.cfg.get("exercise", {}).get("target_min_knee_angle", 100), ls="--",
                                color="#6b7280", lw=1, label="Alvo de flexão")
                for r in reps:
                    axes[0].axvspan(r["start_s"], r["end_s"], color="#dc2626" if r["issues"] else "#16a34a",
                                    alpha=0.12)
                axes[0].set_title("Sessão de fisioterapia: repetições (verde = adequada, vermelho = fora do padrão)")
            else:
                cnt = frames.groupby("time_s")["track_id"].nunique()
                axes[0].plot(cnt.index, cnt.values, color="#2563eb", label="Pessoas no quadro")
                for c in [c for c in frames.columns if c.startswith("in_")]:
                    zc = frames[frames[c]].groupby("time_s")["track_id"].nunique()
                    axes[0].plot(zc.index, zc.values, color="#dc2626", label=f"Pessoas em {c[3:]}")
                axes[0].set_title("Ocupação ao longo do vídeo")
            axes[0].legend(loc="upper right", fontsize=8)
        colors = {Severity.BAIXA: "#2563eb", Severity.MEDIA: "#d97706", Severity.ALTA: "#dc2626",
                  Severity.CRITICA: "#7f1d1d"}
        for e in events:
            axes[1].scatter(e.time_s, int(e.severity), color=colors.get(e.severity, "#6b7280"), s=60)
            axes[1].annotate(e.kind.replace("_", " "), (e.time_s, int(e.severity)), fontsize=7, rotation=20,
                             xytext=(3, 5), textcoords="offset points")
        axes[1].set_yticks([1, 2, 3, 4], ["BAIXA", "MÉDIA", "ALTA", "CRÍTICA"])
        axes[1].set_ylim(0.5, 4.8)
        axes[1].set_xlabel("Tempo (s)")
        axes[1].set_title("Eventos detectados")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return str(out)

    def _write_report(self, rep: VideoReport, out_dir: Path, start_time: datetime) -> None:
        with open(out_dir / "relatorio_video.json", "w", encoding="utf-8") as f:
            json.dump(rep.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        lines = [f"# Relatório automático de vídeo – {Path(rep.video).name}", "",
                 f"- Paciente: **{rep.patient_id}** | Modo: **{rep.mode}**",
                 f"- Duração: {rep.duration_s:.1f} s | FPS: {rep.fps:.0f} | Frames analisados: {rep.frames_processed}",
                 f"- Pessoas rastreadas: {rep.people_tracked}",
                 f"- Objetos detectados (YOLOv8): " + (", ".join(f"{k} ({v})" for k, v in
                                                          sorted(rep.object_counts.items(), key=lambda x: -x[1])[:8]) or "nenhum"),
                 "", "## Desvios e eventos detectados", ""]
        if not rep.events:
            lines.append("Nenhum desvio em relação ao padrão esperado.")
        else:
            lines += ["| Tempo | Severidade | Tipo | Descrição |", "|---|---|---|---|"]
            for e in rep.events:
                ts = start_time + timedelta(seconds=e.time_s)
                lines.append(f"| {e.time_s:.1f} s ({ts:%H:%M:%S}) | {e.severity.name} | {e.kind} | {e.description} |")
        if rep.repetitions:
            lines += ["", "## Repetições do exercício", "",
                      "| # | Início (s) | Joelho mín. (°) | Tronco máx. (°) | Assimetria (°) | Problemas |",
                      "|---|---|---|---|---|---|"]
            for r in rep.repetitions:
                lines.append(f"| {r['index']} | {r['start_s']:.1f} | {r['min_knee']:.0f} | {r['max_trunk']:.0f} | "
                             f"{r['max_asymmetry']:.0f} | {'; '.join(r['issues']) or 'adequada'} |")
        lines += ["", "![Linha do tempo](linha_do_tempo.png)", ""]
        (out_dir / "relatorio_video.md").write_text("\n".join(lines), encoding="utf-8")


def events_to_alerts(report: VideoReport, start_time: datetime) -> list[Alert]:
    actions = {
        "queda": "Enviar equipe ao local imediatamente.",
        "execucao_incorreta": "Fisioterapeuta deve corrigir a execução/ajustar carga.",
        "sessao_fora_do_padrao": "Interromper a sessão e reavaliar dor/fadiga do paciente.",
        "objeto_zona_critica": "Verificar o campo/área crítica.",
        "instrumento_remanescente": "Realizar contagem de instrumentais antes do fechamento.",
        "lotacao_zona": "Reduzir circulação na área crítica.",
        "postura_atipica": "Revisar trecho do vídeo.",
    }
    alerts = []
    for e in report.events:
        if e.kind == "execucao_incorreta":
            continue  # consolidado no evento de sessão (evita excesso de alertas)
        alerts.append(Alert(report.patient_id, "video", e.severity, f"Vídeo: {e.kind.replace('_', ' ')}",
                            e.description, timestamp=start_time + timedelta(seconds=e.time_s),
                            evidence={"video": Path(report.video).name, "time_s": round(e.time_s, 1),
                                      "snapshot": e.snapshot}, recommended_action=actions.get(e.kind, ""),
                            score={4: 0.9, 3: 0.7, 2: 0.45, 1: 0.2}.get(int(e.severity), 0.1)))
    return alerts
