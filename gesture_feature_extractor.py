"""
BridgeSign – Gesture Feature Extractor
=======================================
Captures a sliding window of hand-landmark frames and computes
temporal (motion) features for gesture recognition.

The static FeatureExtractor looks at ONE frame → shape of hand.
This module looks at N frames → movement of hand over time.

Used by gesture_classifier.py and gesture_collector.py.
"""

import numpy as np
from collections import deque

# How many frames make up one gesture sample
GESTURE_WINDOW = 15          # ~0.5s at 30fps
STATIC_FEATURES = 42        # 21 landmarks × 2 coords per frame
GESTURE_FEATURE_SIZE = GESTURE_WINDOW * STATIC_FEATURES   # 630 raw
VELOCITY_FEATURES = (GESTURE_WINDOW - 1) * STATIC_FEATURES  # 588 deltas
SUMMARY_FEATURES = 42 + 1   # displacement per landmark + total magnitude

TOTAL_GESTURE_FEATURES = GESTURE_FEATURE_SIZE + VELOCITY_FEATURES + SUMMARY_FEATURES
# = 630 + 588 + 43 = 1261 features


class GestureFeatureExtractor:
    """
    Maintains a sliding window of static feature vectors and produces
    a rich temporal feature vector when the window is full.
    """

    def __init__(self, window_size=GESTURE_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def push_frame(self, static_features):
        """
        Add one frame's static features (42-dim) to the buffer.

        Parameters
        ----------
        static_features : np.ndarray of shape (42,) or None
            Output from FeatureExtractor.extract_features().
            If None, the frame is skipped.
        """
        if static_features is not None:
            self.buffer.append(static_features.copy())

    def is_ready(self):
        """True when the buffer has enough frames to extract a gesture."""
        return len(self.buffer) >= self.window_size

    def get_raw_sequence(self):
        """
        Return the raw frame buffer as a (window, 42) numpy array.
        Used by the LSTM model which takes sequence input directly.
        Returns None if buffer isn't full.
        """
        if not self.is_ready():
            return None
        return np.array(list(self.buffer), dtype=np.float32)  # (15, 42)


    def get_motion_magnitude(self):
        """
        Quick check: how much total motion is in the buffer?
        Used by the decider to know if the hand is moving or static.
        Returns 0.0 if buffer isn't full.
        """
        if not self.is_ready():
            return 0.0
        frames = np.array(list(self.buffer))    # (window, 42)
        deltas = np.diff(frames, axis=0)         # (window-1, 42)
        return float(np.sum(np.abs(deltas)))

    def extract_gesture_features(self):
        """
        Extract the full temporal feature vector from the current buffer.

        Returns
        -------
        np.ndarray of shape (TOTAL_GESTURE_FEATURES,) or None if not ready.
        """
        if not self.is_ready():
            return None

        frames = np.array(list(self.buffer), dtype=np.float32)  # (15, 42)

        # 1. Raw flattened sequence (captures exact positions over time)
        raw = frames.flatten()                       # 630

        # 2. Frame-to-frame velocity (captures movement direction/speed)
        deltas = np.diff(frames, axis=0).flatten()   # 588

        # 3. Summary: displacement from first to last frame per landmark
        displacement = (frames[-1] - frames[0])      # 42
        total_mag = np.array([np.linalg.norm(displacement)])  # 1

        summary = np.concatenate([displacement, total_mag])   # 43

        return np.concatenate([raw, deltas, summary])

    def clear(self):
        """Reset the buffer (e.g., when hand disappears)."""
        self.buffer.clear()


def extract_from_sequence(frame_list):
    """
    Utility: extract gesture features from a pre-recorded list of
    static feature vectors. Used by the trainer.

    Parameters
    ----------
    frame_list : list of np.ndarray, each shape (42,), length = GESTURE_WINDOW

    Returns
    -------
    np.ndarray of shape (TOTAL_GESTURE_FEATURES,) or None
    """
    ext = GestureFeatureExtractor(window_size=len(frame_list))
    for f in frame_list:
        ext.push_frame(f)
    return ext.extract_gesture_features()
