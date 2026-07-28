"""
detector.py
-----------
Handles all interaction with the YOLOv8-Pose model (Ultralytics).

This module is responsible ONLY for detection - given a video frame, it
returns a clean list of detected people, each with:
    - bounding box (x1, y1, x2, y2)
    - confidence score
    - 17 COCO-format pose keypoints (x, y, confidence) each

It does NOT know anything about table zones or occupancy logic - that is
handled by tracker.py. This separation of concerns keeps each module
focused on a single responsibility (a core OOP/software-engineering
principle).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ultralytics import YOLO
from utils import Config


# --------------------------------------------------------------------------
# DATA STRUCTURE FOR A SINGLE DETECTION
# --------------------------------------------------------------------------
@dataclass
class PersonDetection:
    """
    Represents a single detected person in one video frame.

    Attributes:
        bbox (Tuple[float, float, float, float]): (x1, y1, x2, y2) box corners.
        confidence (float): Detection confidence score (0.0 - 1.0).
        keypoints (List[Tuple[float, float, float]]): 17 COCO keypoints,
            each as (x, y, confidence). Order follows COCO standard:
            0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
            5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
            9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
            13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle.
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    keypoints: List[Tuple[float, float, float]]


# --------------------------------------------------------------------------
# SKELETON CONNECTIONS (for drawing pose lines between keypoints)
# --------------------------------------------------------------------------
# Each tuple is a pair of keypoint indices that should be connected with a
# line to visually form a human skeleton (COCO standard skeleton layout).
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]


# --------------------------------------------------------------------------
# PERSON DETECTOR CLASS
# --------------------------------------------------------------------------
class PersonDetector:
    """
    Wraps the Ultralytics YOLOv8-Pose model to provide a simple, clean
    interface: give it a frame, get back a list of PersonDetection objects.

    Using a class (instead of a bare function) lets us load the model
    ONCE in the constructor and reuse it efficiently across thousands of
    video frames, instead of reloading it every call.
    """

    def __init__(self, model_path: str = Config.MODEL_PATH,
                 confidence_threshold: float = Config.CONFIDENCE_THRESHOLD,
                 device: str = Config.DEVICE):
        """
        Load the YOLOv8-Pose model once when the detector is created.

        Args:
            model_path: path to the .pt weights file. Ultralytics will
                auto-download 'yolov8n-pose.pt' on first run if not present.
            confidence_threshold: minimum confidence to keep a detection.
            device: "cpu" or "cuda" (GPU) - depends on the machine running it.
        """
        print(f"[Detector] Loading YOLOv8-Pose model from '{model_path}' ...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        print("[Detector] Model loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Run pose detection on a single video frame and return a list of
        PersonDetection objects (one per detected person).

        Args:
            frame: a single video frame as a numpy array (BGR, from OpenCV).

        Returns:
            List[PersonDetection]: all people detected above the confidence
            threshold, each with bounding box + keypoints.
        """
        # Run inference. verbose=False keeps the console clean (no per-frame logs).
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
            classes=[0],  # class 0 = "person" in the COCO dataset used by YOLO
        )

        detections: List[PersonDetection] = []

        # Ultralytics returns a list of Results objects (one per input image;
        # we only pass one frame at a time, so we take the first result).
        if not results:
            return detections

        result = results[0]

        # If no boxes were detected in this frame, return an empty list.
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()          # shape (N, 4)
        confidences = result.boxes.conf.cpu().numpy()     # shape (N,)

        # Pose keypoints may be None if the model somehow returns no keypoints
        # (defensive check to avoid crashing the whole pipeline).
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()      # (N, 17, 2)
            keypoints_conf = result.keypoints.conf.cpu().numpy()  # (N, 17)
        else:
            keypoints_xy = None
            keypoints_conf = None

        # Build a clean PersonDetection object for each detected person.
        for i in range(len(boxes)):
            bbox = tuple(boxes[i])  # (x1, y1, x2, y2)
            conf = float(confidences[i])

            person_keypoints: List[Tuple[float, float, float]] = []
            if keypoints_xy is not None:
                for k in range(keypoints_xy.shape[1]):
                    kx, ky = keypoints_xy[i][k]
                    kconf = float(keypoints_conf[i][k])
                    person_keypoints.append((float(kx), float(ky), kconf))

            detections.append(PersonDetection(
                bbox=bbox,
                confidence=conf,
                keypoints=person_keypoints
            ))

        return detections
