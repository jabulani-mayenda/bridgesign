"""
Pose Detector – MediaPipe Pose Landmarker (upper-body landmarks).

Returns 33 pose landmarks (shoulders, elbows, wrists, hips, etc.)
used by the Motion Recorder to drive avatar arm/body/head animation.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import config

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)
_POSE_MODEL_PATH = os.path.join(config.MODELS_DIR, "pose_landmarker_lite.task")


def _ensure_pose_model():
    """Download the pose landmarker model if not already present."""
    if not os.path.exists(_POSE_MODEL_PATH):
        import urllib.request
        print("[PoseDetector] Downloading pose landmarker model…")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        print(f"[PoseDetector] Model saved to {_POSE_MODEL_PATH}")


class PoseDetector:
    """Detects 33 pose landmarks using MediaPipe Pose Landmarker task API."""

    # Upper-body landmark indices we care about for the avatar
    UPPER_BODY_INDICES = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
    }

    def __init__(self, detection_con=0.5, track_con=0.5):
        _ensure_pose_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=_POSE_MODEL_PATH
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_con,
            min_pose_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts = 0
        self.results = None

    def detect(self, img):
        """
        Run pose detection on a BGR image.

        Returns a list of 33 landmark dicts [{x, y, z, visibility}, …]
        or None if no pose was detected.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        self.results = self._detector.detect_for_video(mp_img, self._frame_ts)

        if (
            self.results
            and self.results.pose_landmarks
            and len(self.results.pose_landmarks) > 0
        ):
            return self.results.pose_landmarks[0]  # first person
        return None

    def get_landmark_payload(self, pose_landmarks):
        """
        Convert MediaPipe NormalizedLandmark list → JSON-safe list.
        Returns all 33 landmarks with x, y, z, visibility.
        """
        if not pose_landmarks:
            return []
        return [
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(getattr(lm, "z", 0.0)),
                "visibility": float(getattr(lm, "visibility", 1.0)),
            }
            for lm in pose_landmarks
        ]

    def get_upper_body_parts(self, pose_landmarks):
        """Return list of detected upper-body part names."""
        if not pose_landmarks:
            return []
        parts = []
        for idx, name in self.UPPER_BODY_INDICES.items():
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                vis = float(getattr(lm, "visibility", 0))
                if vis > 0.3:
                    parts.append(name)
        return parts
