"""
BridgeSign — Rebuild dataset.csv from dataset/ image folders + retrain.
Uses build_dataset.py logic inline so no args are needed.
"""
import os, sys, cv2, time
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(BASE_DIR, "dataset")
DATA_DIR    = os.path.join(BASE_DIR, "data")
CSV_PATH    = os.path.join(DATA_DIR, "dataset.csv")
CSV_BACKUP  = os.path.join(DATA_DIR, "dataset_old_backup.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Back up old CSV ───────────────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    import shutil
    shutil.copy2(CSV_PATH, CSV_BACKUP)
    print(f"[Info] Old CSV backed up to {CSV_BACKUP}")

# ── Import helpers ────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor

detector  = HandDetector(mode=True, max_hands=1)   # IMAGE mode for offline files
extractor = FeatureExtractor()

# ── Extract landmarks from every image in dataset/<SIGN>/ ────────────────────
print(f"\n[Build] Scanning {IMAGES_DIR} ...\n")

total_ok    = 0
total_fail  = 0
class_stats = {}

with open(CSV_PATH, "w") as out:
    for sign_label in sorted(os.listdir(IMAGES_DIR)):
        sign_dir = os.path.join(IMAGES_DIR, sign_label)
        if not os.path.isdir(sign_dir):
            continue

        ok = 0
        fail = 0
        for img_name in sorted(os.listdir(sign_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(sign_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                fail += 1
                continue

            _, results = detector.find_hands(frame, draw=False)
            lm_list = detector.get_landmarks(frame, hand_no=0)

            if lm_list:
                features = extractor.extract_features(lm_list)
                if features is not None:
                    out.write(f"{sign_label}," + ",".join(f"{v:.8f}" for v in features) + "\n")
                    ok += 1
                    total_ok += 1
                else:
                    fail += 1
                    total_fail += 1
            else:
                fail += 1
                total_fail += 1

        class_stats[sign_label] = (ok, fail)
        print(f"  {sign_label:2s}:  {ok:4d} OK   {fail:3d} failed")

print(f"\n[Build] Done. {total_ok} rows written, {total_fail} images failed landmark extraction.")
print(f"[Build] CSV saved -> {CSV_PATH}\n")
print("Now run:  python model_trainer.py")
