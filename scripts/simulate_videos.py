"""Gera os vídeos de demonstração usados no projeto.

1. ``fisioterapia_simulada.mp4`` + ``fisioterapia_openpose_json/``
   Sessão de agachamento (8 repetições) gerada a partir de um esqueleto biomecânico
   2D. Os keypoints são gravados no mesmo formato JSON do OpenPose (COCO-18), e o
   vídeo mostra um manequim desenhado sobre o esqueleto. Repetições 5-8 possuem
   desvios conhecidos (amplitude insuficiente, inclinação do tronco, assimetria),
   permitindo validar o analisador.

2. ``queda_simulada.mp4``
   Pessoa recortada (segmentação YOLOv8-seg da imagem de exemplo ``bus.jpg`` do
   Ultralytics) composta em um cenário de quarto e animada caminhando e caindo.
   O vídeo é processado pelo YOLOv8-pose real.

Vídeos clínicos reais (ex.: gravações próprias de fisioterapia) podem ser
analisados diretamente com ``scripts/run_video.py``.

Uso:  python scripts/simulate_videos.py --out data/samples
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25


def room(label: str) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (214, 224, 228)
    cv2.rectangle(bg, (0, 470), (W, H), (150, 160, 168), -1)
    for x in range(0, W, 80):
        cv2.line(bg, (x, 470), (int(W / 2 + (x - W / 2) * 1.8), H), (140, 150, 158), 1)
    cv2.putText(bg, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 130), 2, cv2.LINE_AA)
    return bg


# ----------------------------------------------------------------- fisioterapia
REPS = [  # (joelho_min_esq, joelho_min_dir, tronco_max) por repetição
    (88, 90, 30), (86, 88, 32), (90, 92, 28), (87, 89, 33),
    (112, 114, 26),  # 5: amplitude insuficiente
    (90, 92, 62),  # 6: inclinação excessiva do tronco
    (86, 118, 34),  # 7: assimetria
    (114, 116, 50),  # 8: fadiga (pouca amplitude + compensação de tronco)
]


def _leg(ankle, theta_deg, L1=130, L2=135):
    a = math.radians((180 - theta_deg) / 2)
    knee = ankle + L1 * np.array([math.sin(a), -math.cos(a)])
    d1 = np.array([-math.sin(a), math.cos(a)])  # joelho -> tornozelo
    t = math.radians(theta_deg)
    cands = []
    for sgn in (1, -1):  # o quadril fica atrás do joelho (menor x)
        rot = np.array([[math.cos(t), -sgn * math.sin(t)], [sgn * math.sin(t), math.cos(t)]])
        cands.append(knee + L2 * (rot @ d1))
    hip = min(cands, key=lambda h: h[0])
    return knee, hip


def physio_pose(theta_l, theta_r, trunk_deg, x0=600, floor=660):
    ank_l = np.array([x0, floor], float)
    ank_r = np.array([x0 + 14, floor - 8], float)
    knee_l, hip_l = _leg(ank_l, theta_l)
    knee_r, hip_r = _leg(ank_r, theta_r)
    hip_c = (hip_l + hip_r) / 2
    phi = math.radians(trunk_deg)
    sh_c = hip_c + 175 * np.array([math.sin(phi), -math.cos(phi)])
    sh_l, sh_r = sh_c + [-6, 4], sh_c + [8, -4]
    reach = np.array([math.cos(phi * 0.3), math.sin(phi * 0.3)])
    elb_l, elb_r = sh_l + 75 * reach, sh_r + 75 * reach + [6, -4]
    wri_l, wri_r = elb_l + 70 * reach, elb_r + 70 * reach
    head = sh_c + 55 * np.array([math.sin(phi), -math.cos(phi)]) + [14, 0]
    neck = sh_c
    # COCO-18: nose, neck, rsho, relb, rwri, lsho, lelb, lwri, rhip, rknee, rank, lhip, lknee, lank, reye, leye, rear, lear
    pts = [head + [22, 5], neck, sh_r, elb_r, wri_r, sh_l, elb_l, wri_l, hip_r, knee_r, ank_r, hip_l, knee_l, ank_l,
           head + [14, -8], head + [16, -10], head + [-6, -4], head + [-4, -6]]
    return np.array(pts)


def physio_angles(t: float):
    """Retorna ângulos (joelho esq., joelho dir., tronco) no instante t."""
    rest = (176, 176, 8)
    t0, period = 2.0, 3.2
    k = int((t - t0) // period)
    if t < t0 or k >= len(REPS):
        return rest
    u = (t - t0 - k * period) / period
    if u < 0.4:
        s = 0.5 - 0.5 * math.cos(math.pi * u / 0.4)
    elif u < 0.5:
        s = 1.0
    elif u < 0.9:
        s = 0.5 + 0.5 * math.cos(math.pi * (u - 0.5) / 0.4)
    else:
        s = 0.0
    kl, kr, tr = REPS[k]
    return (rest[0] - (rest[0] - kl) * s, rest[1] - (rest[1] - kr) * s, rest[2] + (tr - rest[2]) * s)


def draw_mannequin(img, p):
    limb = lambda a, b, c, t: cv2.line(img, tuple(map(int, p[a])), tuple(map(int, p[b])), c, t, cv2.LINE_AA)  # noqa
    pants, shirt, skin = (90, 60, 40), (160, 120, 60), (150, 180, 215)
    limb(8, 9, (70, 45, 30), 24); limb(9, 10, (70, 45, 30), 20)  # perna dir (fundo)
    limb(11, 12, pants, 26); limb(12, 13, pants, 22)
    cv2.line(img, tuple(map(int, (p[8] + p[11]) / 2)), tuple(map(int, p[1])), shirt, 46, cv2.LINE_AA)
    limb(2, 3, (130, 95, 45), 16); limb(3, 4, skin, 13)
    limb(5, 6, shirt, 18); limb(6, 7, skin, 14)
    for a in (10, 13):
        cv2.ellipse(img, (int(p[a][0]) + 18, int(p[a][1]) + 8), (26, 10), 0, 0, 360, (40, 40, 40), -1)
    head = (p[14] + p[16]) / 2
    cv2.circle(img, tuple(map(int, head + [4, 4])), 30, skin, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, head + [-5, -8])), 30, (40, 40, 60), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, head + [6, 6])), 26, skin, -1, cv2.LINE_AA)


def make_physio(out: Path, seed: int = 3):
    rng = np.random.default_rng(seed)
    json_dir = out / "fisioterapia_openpose_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for old in json_dir.glob("*.json"):
        old.unlink()
    tmp = out / "fisioterapia_simulada.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg = room("SESSAO DE FISIOTERAPIA - AGACHAMENTO (SIMULACAO)")
    cv2.rectangle(bg, (900, 380), (1180, 470), (120, 150, 170), -1)
    cv2.rectangle(bg, (60, 140), (200, 470), (200, 205, 210), -1)  # espaldar
    for yy in range(160, 470, 40):
        cv2.line(bg, (60, yy), (200, yy), (150, 120, 90), 6)  # maca ao fundo
    total = 2.0 + len(REPS) * 3.2 + 2.0
    for i in range(int(total * FPS)):
        t = i / FPS
        kl, kr, tr = physio_angles(t)
        pts = physio_pose(kl, kr, tr) + rng.normal(0, 1.2, (18, 2))
        frame = bg.copy()
        draw_mannequin(frame, pts)
        cv2.putText(frame, f"t = {t:4.1f} s", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
        vw.write(frame)
        kp = np.hstack([pts, np.full((18, 1), 0.9)]).ravel().round(2).tolist()
        with open(json_dir / f"fisioterapia_{i:06d}_keypoints.json", "w") as f:
            json.dump({"version": 1.3, "people": [{"person_id": [1], "pose_keypoints_2d": kp}]}, f)
    vw.release()
    _h264(tmp, out / "fisioterapia_simulada.mp4")


# ------------------------------------------------------------------------ queda
def person_cutout(cache: Path) -> np.ndarray:
    if cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_UNCHANGED)
    import ultralytics
    from ultralytics import YOLO

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    r = YOLO("yolov8n-seg.pt")(img, verbose=False)[0]
    best = max((i for i, c in enumerate(r.boxes.cls.cpu().numpy()) if int(c) == 0),
               key=lambda i: float(r.boxes.xyxy[i, 0] < 300) * float(r.boxes.xyxy[i, 3] - r.boxes.xyxy[i, 1]))
    mask = cv2.resize(r.masks.data[best].cpu().numpy(), (img.shape[1], img.shape[0]))
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[best].cpu().numpy())
    rgba = np.dstack([img, (mask * 255).astype(np.uint8)])[y1:y2, x1:x2]
    cv2.imwrite(str(cache), rgba)
    return rgba


def _compose(frame, img, foot_x, foot_y, angle):
    ph, pw = img.shape[:2]
    C = int(np.hypot(ph, pw)) * 2 + 10
    canvas = np.zeros((C, C, 4), np.uint8)
    ox, oy = C // 2 - pw // 2, C // 2 - ph
    canvas[oy:oy + ph, ox:ox + pw] = img
    rot = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((C / 2, C / 2), -angle, 1.0), (C, C))
    x0, y0 = int(foot_x - C / 2), int(foot_y - C / 2)
    xs, ys, xe, ye = max(0, x0), max(0, y0), min(W, x0 + C), min(H, y0 + C)
    crop = rot[ys - y0:ye - y0, xs - x0:xe - x0]
    a = crop[:, :, 3:4] / 255.0
    frame[ys:ye, xs:xe] = (crop[:, :, :3] * a + frame[ys:ye, xs:xe] * (1 - a)).astype(np.uint8)
    return frame


def make_fall(out: Path):
    person = cv2.resize(person_cutout(out / ".pessoa_recorte.png"), None, fx=0.95, fy=0.95)
    bg = room("QUARTO 12 - LEITO 2 (SIMULACAO)")
    cv2.rectangle(bg, (60, 330), (420, 500), (235, 235, 235), -1)
    cv2.rectangle(bg, (60, 300), (110, 500), (120, 120, 130), -1)
    cv2.rectangle(bg, (80, 330), (400, 360), (250, 250, 250), -1)
    cv2.rectangle(bg, (1050, 150), (1200, 470), (160, 130, 110), -1)
    tmp = out / "queda_simulada.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(int(10 * FPS)):
        t = i / FPS
        if t < 4.0:
            fx, fy, ang = 300 + t * 60, 640, 0
        elif t < 4.7:
            u = (t - 4.0) / 0.7
            fx, fy, ang = 540 + u * 40, 640 + u * 20, 88 * u ** 2
        else:
            fx, fy, ang = 580, 660, 88
        vw.write(cv2.GaussianBlur(_compose(bg.copy(), person, fx, fy, ang), (3, 3), 0))
    vw.release()
    _h264(tmp, out / "queda_simulada.mp4")


def _h264(src: Path, dst: Path):
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(dst)], check=True)
        src.unlink()
    else:
        src.rename(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    make_physio(out)
    make_fall(out)
    print("Vídeos gerados em", out)
