"""
src/model/object_model.py  –  YOLOv8 model wrapper + OpenCV frame annotation
============================================================
Responsibilities:
  - Load YOLOv8 model at startup (auto-selects CUDA / CPU)
  - Hot-swap model weights without restarting the server
  - Run inference on a single BGR OpenCV frame
  - Filter detections by class name list (pre-filtered at YOLO level for speed)
  - Draw annotated bounding boxes + labels directly on the frame (in-place)
  - Return structured detection dicts for JSON serialisation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ─── COCO 80 class list (index → name) ───────────────────────────────────────
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Distinct RGB colour palette for 80 classes
_PALETTE_RGB = [
    (0, 229, 255), (255, 61, 113), (0, 255, 136), (255, 230, 0),  (255, 149, 0),
    (180, 68, 255),(255, 107, 53), (0, 212, 170), (255, 133, 161),(127, 255, 0),
    (255, 77, 166),(0, 191, 255),  (255, 200, 0), (0, 255, 200),  (200, 255, 0),
    (255, 0, 128), (0, 128, 255),  (128, 255, 0), (255, 128, 0),  (0, 255, 128),
]


def _class_color_bgr(cls_name: str) -> tuple:
    """Deterministic per-class BGR colour."""
    h = 0
    for c in cls_name:
        h = (h * 31 + ord(c)) % len(_PALETTE_RGB)
    r, g, b = _PALETTE_RGB[h]
    return (b, g, r)   # OpenCV uses BGR


# ─── ObjectDetector ───────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Thin wrapper around YOLOv8 with:
      - Auto GPU/CPU selection
      - Hot-swap of model weights via load_model()
      - Per-class colour annotation
      - Structured detection output

    Parameters
    ----------
    model_path : str
        Path or name of the YOLO weights file (e.g. "yolov8n.pt").
        Ultralytics auto-downloads official weights on first use.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed.\n"
                "Run:  pip install ultralytics"
            )

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = ""       # set by load_model()
        self._name_to_id: Dict[str, int] = {}
        self.model = None

        self.load_model(model_path)

    # ─── Model loading ────────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> None:
        """
        Load (or hot-swap) YOLOv8 weights.
        Safe to call while the server is running — subsequent detect() calls
        will use the new model.
        """
        print(f"[ObjectDetector] Loading → {model_path}  (device: {self.device.upper()})")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model_path = model_path

        # Rebuild class-name ↔ index map
        self._name_to_id = {
            name: idx for idx, name in self.model.names.items()
        }
        print(f"[ObjectDetector] Ready  → {model_path}  |  {len(self._name_to_id)} classes")

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        filter_classes: Optional[List[str]] = None,
        annotate: bool = True,
    ) -> List[Dict]:
        """
        Run YOLOv8 on a single BGR OpenCV frame.

        Parameters
        ----------
        frame          : H×W×3 uint8 BGR array.  Modified in-place if annotate=True.
        confidence     : minimum confidence threshold  [0, 1]
        filter_classes : if provided, only return detections whose class name is
                         in this list.  Filtering happens inside YOLO (faster).
        annotate       : draw coloured boxes + labels on `frame`

        Returns
        -------
        List of dicts  {class, confidence, bbox:[x1,y1,x2,y2]}
        """
        if self.model is None:
            return []

        # Resolve class names → numeric IDs for YOLO's internal filter
        cls_ids: Optional[List[int]] = None
        if filter_classes:
            cls_ids = [
                self._name_to_id[n]
                for n in filter_classes
                if n in self._name_to_id
            ]
            if not cls_ids:
                return []   # all requested classes unknown → skip inference

        results = self.model.predict(
            source  = frame,
            conf    = confidence,
            classes = cls_ids,
            device  = self.device,
            verbose = False,
        )

        detections: List[Dict] = []

        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "class":      cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox":       [x1, y1, x2, y2],
                })

                if annotate:
                    self._draw_box(frame, cls_name, conf_val, x1, y1, x2, y2)

        return detections

    # ─── Annotation helper ────────────────────────────────────────────────────

    @staticmethod
    def _draw_box(
        frame: np.ndarray,
        cls_name: str,
        conf: float,
        x1: int, y1: int, x2: int, y2: int,
        thickness: int = 2,
    ) -> None:
        color = _class_color_bgr(cls_name)
        label = f"{cls_name} {conf * 100:.0f}%"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + baseline), color, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 3, ly - 2),
            font, font_scale,
            (20, 20, 20), 1, cv2.LINE_AA,
        )

    # ─── Utilities ────────────────────────────────────────────────────────────

    @property
    def available_classes(self) -> List[str]:
        """All class names supported by the currently loaded model."""
        return list(self.model.names.values()) if self.model else []
