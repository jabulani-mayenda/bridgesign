"""Quick offline check: model load + sanity + optional webcam frame."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from classifier import Classifier
from feature_extractor import FeatureExtractor
from hand_detector import HandDetector


def main():
    print("=== BridgeSign live pipeline test ===\n")
    clf = Classifier()
    if clf.pipeline is None:
        print("FAIL: sign_model.pkl did not load")
        return 1
    print("OK: sign_model.pkl loaded")

    fe = FeatureExtractor()
    det = HandDetector(mode=True)
    sample = os.path.join(ROOT, "data", "_sample_A.jpg")
    if os.path.exists(sample):
        import cv2
        frame = cv2.imread(sample)
        frame, res = det.find_hands(frame, draw=False)
        lm = det.get_landmarks(frame, 0)
        if not lm:
            print(f"WARN: no hand in {sample}")
        else:
            feats = fe.extract_features(lm)
            raw_l, raw_c = clf.predict(feats)
            flip_l, flip_c = clf.predict(fe.flip_x(feats))
            print(f"Sample image: raw={raw_l}({raw_c:.3f}) flipped={flip_l}({flip_c:.3f})")
    else:
        print(f"SKIP: no {sample}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
