import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import config

# Download model if not present
_MODEL_PATH = os.path.join(config.MODELS_DIR, "hand_landmarker.task")

def _ensure_model():
    """Download the hand landmarker model if not already present."""
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"[HandDetector] Downloading hand landmarker model...")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, _MODEL_PATH)
        print(f"[HandDetector] Model saved to {_MODEL_PATH}")

_ensure_model()

class HandDetector:
    def __init__(self,
                 mode=False,
                 max_hands=config.MAX_NUM_HANDS,
                 detection_con=config.MIN_DETECTION_CONFIDENCE,
                 track_con=config.MIN_TRACKING_CONFIDENCE):

        self.max_hands = max_hands
        self.results = None

        running_mode = mp_vision.RunningMode.IMAGE if mode else mp_vision.RunningMode.VIDEO

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )
        self._mode = running_mode
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0  # monotonically increasing timestamp for VIDEO mode

    def find_hands(self, img, draw=True):
        """Detect hands and optionally draw landmarks. Returns (img, results)."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # avoids an internal copy inside MediaPipe
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == mp_vision.RunningMode.VIDEO:
            self._frame_ts += 33  # approx 30fps
            self.results = self._detector.detect_for_video(mp_img, self._frame_ts)
        else:
            self.results = self._detector.detect(mp_img)

        if draw and self.results and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand_lm in self.results.hand_landmarks:
                # Draw connections manually
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                # Draw points
                for pt in pts:
                    cv2.circle(img, pt, 5, (0, 215, 255), cv2.FILLED)
                # Draw a few key connections (palm)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                for a, b in connections:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)

        return img, self.results

    def get_landmarks(self, img, hand_no=0):
        """Extract landmark list [id, x, y] for a specific hand."""
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if len(self.results.hand_landmarks) > hand_no:
                h, w, _ = img.shape
                hand_lm = self.results.hand_landmarks[hand_no]
                for id, lm in enumerate(hand_lm):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list
