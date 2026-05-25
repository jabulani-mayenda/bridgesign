import os
import tempfile

# Read from environment for cloud deployment (Render, Heroku, etc.)
# Falls back to dev defaults when running locally.
SECRET_KEY = os.environ.get("SECRET_KEY", "bridgesign_secret_key_2024")
PORT       = int(os.environ.get("PORT", 5000))
APP_NAME = "BridgeSign"
APP_VERSION = "1.0.0"

# Camera Settings
CAMERA_INDEX = 0
INFER_EVERY_N_FRAMES = 2   # Every 2nd frame (~15 detections/sec) — balances speed vs CPU
CONSECUTIVE_THRESHOLD = 2  # Confirm after 2 matching frames for snappy detection
JPEG_QUALITY = 70          # Lower = faster stream
CAMERA_MAX_INDEX = 4
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Hand-switch grace period
# When the hand disappears, wait this many seconds before treating it as
# "no hand" for word-boundary purposes.  Absorbs brief tracking drops when
# repositioning or switching hands without triggering a false word boundary.
HAND_LOST_GRACE_SEC = 0.5  # 0.5 s feels instant but covers a hand-swap

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.5    # Lowered from 0.7 to 0.5: industry standard default, significantly more robust in poor lighting/angles
MIN_TRACKING_CONFIDENCE  = 0.5
MAX_NUM_HANDS            = 1    # Single hand = half the MediaPipe work

# Classifier confidence gate
# Predictions below this value are discarded as "not sure".
# RandomForest distributes probability across 500 trees / 26 classes so
# max confidence is typically 0.20–0.40 even on correct predictions.
# The consecutive-frame filter (CONSECUTIVE_THRESHOLD) provides the real
# false-positive protection, so keep this gate low.
MIN_PREDICTION_CONFIDENCE = 0.25  # raised from 0.15 — fewer false positives, better live accuracy

# Path Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_dir(path, fallback_name=None):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError as e:
        if fallback_name:
            fallback = os.path.join(tempfile.gettempdir(), fallback_name)
            os.makedirs(fallback, exist_ok=True)
            print(f"[Config] Could not write to {path!r}: {e}. Using {fallback!r}.")
            return fallback
        print(f"[Config] Could not create {path!r}: {e}.")
        return path


MODELS_DIR = _ensure_dir(os.path.join(BASE_DIR, "models"))
DATA_DIR = _ensure_dir(
    os.environ.get("BRIDGESIGN_DATA_DIR", os.path.join(BASE_DIR, "data")),
    "bridgesign-data",
)
DATASET_DIR = _ensure_dir(os.path.join(BASE_DIR, "dataset"))

# TTS Settings
TTS_RATE = 150
TTS_VOLUME = 1.0

# Colors for UI (BGR format for OpenCV)
COLOR_PRIMARY = (255, 153, 51) # Blueish
COLOR_SECONDARY = (51, 204, 255) # Orangeish
COLOR_TEXT = (255, 255, 255)
COLOR_SUCCESS = (0, 255, 0)
COLOR_WARNING = (0, 0, 255)
