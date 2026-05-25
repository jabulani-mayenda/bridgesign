import numpy as np

# Key landmark indices (MediaPipe 21-point hand model)
# Fingertip: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
# MCP joints: 1(thumb), 5(index), 9(middle), 13(ring), 17(pinky)
# Wrist: 0
_FINGERTIPS  = [4, 8, 12, 16, 20]
_MCP_JOINTS  = [1, 5,  9, 13, 17]
_PIP_JOINTS  = [2, 6, 10, 14, 18]
_WRIST       = 0
_XY_FEATURE_COUNT = 42  # 21 landmarks * 2 coords

# Pairs used for pairwise distance features
_DIST_PAIRS = [
    (4, 8),  (4, 12), (4, 16), (4, 20),   # thumb tip ↔ each finger tip
    (8, 12), (8, 16), (8, 20),             # index tip ↔ others
    (12,16), (12,20),                       # middle ↔ others
    (16,20),                               # ring ↔ pinky
    (0, 4),  (0, 8),  (0, 12), (0,16), (0,20),  # wrist ↔ fingertips
    (0, 5),  (0, 9),  (0,13),             # wrist ↔ MCP knuckles
    (5, 8),  (9,12),  (13,16), (17,20),   # knuckle ↔ its own fingertip
]
FEATURE_VECTOR_SIZE = _XY_FEATURE_COUNT


class FeatureExtractor:
    """
    Extracts the 42-feature landmark representation used by the saved
    BridgeSign dataset/model.

    The current project artifacts in data/dataset.csv and models/sign_model.pkl
    were collected/trained on wrist-normalized XY coordinates only. Keep the
    runtime extractor aligned with that representation unless the dataset/model
    are rebuilt together.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    def extract_features(self, lm_list):
        """
        Parameters
        ----------
        lm_list : list of [id, x, y]  (pixels, from HandDetector)

        Returns
        -------
        np.ndarray of shape (42,) or None if input is invalid.
        """
        if not lm_list or len(lm_list) < 21:
            return None

        # Build a plain (21,2) array of pixel coordinates
        pts = np.array([[cx, cy] for _, cx, cy in lm_list], dtype=np.float32)

        # ── 1. Translation: centre on wrist (landmark 0) ───────────────
        base = pts[_WRIST].copy()
        pts -= base

        # ── Scale: use wrist→middle-MCP distance (landmark 9) ──────────
        scale = np.linalg.norm(pts[9])
        if scale < 1e-6:
            scale = 1.0
        pts /= scale

        # Keep the live feature vector identical to the saved training data.
        return pts.flatten()

    @staticmethod
    def flip_x(features):
        """Mirror hand pose on X (matches horizontally flipped camera frames)."""
        flipped = np.asarray(features, dtype=np.float32).copy()
        flipped[0::2] *= -1.0
        return flipped
