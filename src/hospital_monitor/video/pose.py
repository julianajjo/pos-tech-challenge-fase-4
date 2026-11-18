"""Estimadores de pose humana.

Dois back-ends com saída no mesmo formato (17 keypoints no padrão COCO):

* ``YoloPoseEstimator`` – YOLOv8-pose (Ultralytics), multi-pessoa e com rastreamento
  (ByteTrack). É o padrão por rodar bem em CPU/Colab.
* ``OpenPoseJsonEstimator`` – importa a saída JSON do binário oficial do OpenPose;
* ``OpenPoseEstimator`` – modelo OpenPose (CMU, COCO-18) executado via OpenCV DNN.
  Requer os arquivos ``pose_deploy_linevec.prototxt`` e ``pose_iter_440000.caffemodel``
  (ver README). Implementação para uma pessoa (sessões de fisioterapia).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
# OpenPose COCO-18 -> COCO-17
OPENPOSE_TO_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
# OpenPose BODY_25 -> COCO-17
BODY25_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


@dataclass
class PersonPose:
    keypoints: np.ndarray  # (17, 3): x, y, confiança (pixels)
    box: np.ndarray  # x1, y1, x2, y2
    track_id: int = -1
    score: float = 1.0
    extra: dict = field(default_factory=dict)


class YoloPoseEstimator:
    def __init__(self, model: str = "yolov8n-pose.pt", conf: float = 0.35, track: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf
        self.track = track

    def __call__(self, frame: np.ndarray, **_) -> list[PersonPose]:
        if self.track:
            res = self.model.track(frame, conf=self.conf, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        else:
            res = self.model(frame, conf=self.conf, verbose=False)[0]
        people = []
        if res.keypoints is None or res.boxes is None:
            return people
        kps = res.keypoints.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else -np.ones(len(boxes), int)
        for k, b, c, i in zip(kps, boxes, confs, ids):
            people.append(PersonPose(k, b, int(i), float(c)))
        return people


class OpenPoseEstimator:
    def __init__(self, proto: str | Path, weights: str | Path, in_size: int = 368, thr: float = 0.1):
        import cv2

        if not Path(proto).exists() or not Path(weights).exists():
            raise FileNotFoundError("Arquivos do OpenPose não encontrados. Veja a seção 'OpenPose' do README.")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.in_size = in_size
        self.thr = thr

    def __call__(self, frame: np.ndarray, **_) -> list[PersonPose]:
        import cv2

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.in_size, self.in_size), (0, 0, 0), swapRB=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 57, H, W): 19 heatmaps + 38 PAFs
        kps18 = np.zeros((18, 3))
        for i in range(18):
            hm = out[0, i]
            _, conf, _, pt = cv2.minMaxLoc(hm)
            kps18[i] = (pt[0] * w / hm.shape[1], pt[1] * h / hm.shape[0], conf if conf > self.thr else 0)
        kps = kps18[OPENPOSE_TO_COCO]
        valid = kps[kps[:, 2] > 0]
        if len(valid) < 5:
            return []
        box = np.array([valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()])
        return [PersonPose(kps, box, 1, float(valid[:, 2].mean()))]


class OpenPoseJsonEstimator:
    """Lê os keypoints exportados pelo binário oficial do OpenPose (``--write_json``).

    Permite rodar o OpenPose completo (multi-pessoa, PAFs, GPU) fora do Python e
    importar o resultado no pipeline. Aceita os modelos COCO-18 e BODY_25.
    Os arquivos ``*_keypoints.json`` são associados aos frames pela ordem.
    """

    def __init__(self, json_dir: str | Path, kp_thr: float = 0.05):
        import json

        self.frames = []
        for fp in sorted(Path(json_dir).glob("*_keypoints.json")):
            with open(fp, encoding="utf-8") as f:
                self.frames.append(json.load(f).get("people", []))
        if not self.frames:
            raise FileNotFoundError(f"Nenhum *_keypoints.json em {json_dir}")
        self.kp_thr = kp_thr

    def __call__(self, frame: np.ndarray | None = None, frame_idx: int = 0, **_) -> list[PersonPose]:
        if frame_idx >= len(self.frames):
            return []
        people = []
        for i, p in enumerate(self.frames[frame_idx]):
            arr = np.asarray(p["pose_keypoints_2d"], dtype=float).reshape(-1, 3)
            mapping = BODY25_TO_COCO if len(arr) == 25 else OPENPOSE_TO_COCO
            kps = arr[mapping]
            valid = kps[kps[:, 2] > self.kp_thr]
            if len(valid) < 5:
                continue
            box = np.array([valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()])
            pid = p.get("person_id", [-1])
            pid = pid[0] if isinstance(pid, list) else pid
            people.append(PersonPose(kps, box, int(pid) if pid not in (-1, None) else i + 1,
                                     float(valid[:, 2].mean())))
        return people


def build_pose_estimator(cfg: dict) -> YoloPoseEstimator | OpenPoseEstimator:
    if cfg.get("pose_backend") == "openpose_json":
        return OpenPoseJsonEstimator(cfg["openpose_json_dir"])
    if cfg.get("pose_backend", "yolo") == "openpose":
        from ..config import PROJECT_ROOT

        return OpenPoseEstimator(PROJECT_ROOT / cfg["openpose_proto"], PROJECT_ROOT / cfg["openpose_weights"])
    return YoloPoseEstimator(cfg.get("pose_model", "yolov8n-pose.pt"), cfg.get("confidence", 0.35))
