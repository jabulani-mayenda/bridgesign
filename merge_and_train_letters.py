"""
BridgeSign – Merge Image Dataset + Live Data, then Train Letter Model
=====================================================================
This script:
  1. Extracts hand landmarks from images in dataset/ (A–Z folders)
  2. Merges with the existing live-captured data/dataset.csv
  3. Deduplicates and saves to data/dataset_merged.csv
  4. Trains sign_model.pkl on the merged dataset

This gives the letter model diversity from:
  - Real webcam captures (hand at various angles, lighting)
  - Image dataset (different people's hands, cleaner shots)

Usage:
  python merge_and_train_letters.py

  # If you only want to retrain without re-extracting images:
  python merge_and_train_letters.py --skip-extract
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import config
from feature_extractor import FEATURE_VECTOR_SIZE

IMAGES_DIR    = "dataset"
LIVE_CSV      = os.path.join(config.DATA_DIR, "dataset.csv")
MERGED_CSV    = os.path.join(config.DATA_DIR, "dataset_merged.csv")
MODEL_PATH    = os.path.join(config.MODELS_DIR, "sign_model.pkl")
CLASSES_PATH  = MODEL_PATH.replace(".pkl", "_classes.json")
METRICS_PATH  = MODEL_PATH.replace(".pkl", "_metrics.json")

AUG_NOISE_STD = 0.015
AUG_COPIES    = 3
MIRROR_EXCLUDE = frozenset({"J", "P", "Q", "Z"})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip image extraction, use existing dataset_merged.csv")
    return p.parse_args()


# ── Step 1: Extract features from image dataset ──────────────────────────────

def extract_from_images(images_dir, output_csv):
    """Run HandDetector + FeatureExtractor on every image in images_dir/LABEL/."""
    import cv2
    from hand_detector import HandDetector
    from feature_extractor import FeatureExtractor

    detector  = HandDetector(mode=True, max_hands=1)
    extractor = FeatureExtractor()

    letters = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )

    print(f"\n  Extracting landmarks from images in '{images_dir}/'...")
    total_ok = 0
    total_fail = 0

    with open(output_csv, "w") as f_out:
        for label in letters:
            label_dir = os.path.join(images_dir, label)
            ok = 0
            fail = 0
            for fname in sorted(os.listdir(label_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(label_dir, fname))
                if img is None:
                    fail += 1
                    continue
                frame, _ = detector.find_hands(img, draw=False)
                lm_list  = detector.get_landmarks(frame, hand_no=0)
                if lm_list:
                    feats = extractor.extract_features(lm_list)
                    if feats is not None and len(feats) == FEATURE_VECTOR_SIZE:
                        f_out.write(label + "," + ",".join(map(str, feats)) + "\n")
                        ok += 1
                    else:
                        fail += 1
                else:
                    fail += 1
            total_ok   += ok
            total_fail += fail
            print(f"    {label}: {ok:3d} ok, {fail} failed")

    print(f"\n  Image extraction done: {total_ok} features, {total_fail} failed")
    return total_ok


# ── Step 2: Merge image CSV with live CSV ─────────────────────────────────────

def merge_csvs(image_csv, live_csv, merged_csv):
    """
    Concatenate image-extracted features with live-captured features.
    Also mirrored (x-flipped) copies of non-directional letters so the model
    handles both front (mirrored webcam) and back camera orientations.
    """
    print(f"\n  Merging datasets...")

    rows_image = 0
    rows_live  = 0
    label_counts = {}

    with open(merged_csv, "w") as out:
        # Live data first (higher priority — real camera conditions)
        if os.path.exists(live_csv):
            with open(live_csv, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        label = line.split(",")[0]
                        label_counts[label] = label_counts.get(label, 0) + 1
                        out.write(line + "\n")
                        rows_live += 1

        # Image data
        if os.path.exists(image_csv):
            with open(image_csv, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        label = line.split(",")[0]
                        label_counts[label] = label_counts.get(label, 0) + 1
                        out.write(line + "\n")
                        rows_image += 1

    print(f"    Live samples  : {rows_live}")
    print(f"    Image samples : {rows_image}")
    print(f"    Total merged  : {rows_live + rows_image}")
    print(f"\n  Per-class counts:")
    for k in sorted(label_counts):
        print(f"    {k:4s}: {label_counts[k]:5d}")

    return rows_live + rows_image


# ── Step 3: Load merged CSV ───────────────────────────────────────────────────

def load_dataset(path):
    labels, features = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            labels.append(parts[0])
            features.append([float(x) for x in parts[1:]])
    X = np.array(features, dtype=np.float32)
    return X, labels


def encode_labels(labels_raw):
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int32)
    return y, classes


# ── Step 4: Augmentation ─────────────────────────────────────────────────────

def augment_dataset(X, y, n_copies=AUG_COPIES, noise_std=AUG_NOISE_STD, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    aug_X, aug_y = [X], [y]
    for i in range(n_copies):
        intensity = noise_std * (0.7 + 0.6 * i / max(n_copies - 1, 1))
        noisy = X + rng.normal(0, intensity, X.shape).astype(np.float32)
        aug_X.append(noisy)
        aug_y.append(y)
        scale = rng.uniform(0.95, 1.05, size=(X.shape[0], 1)).astype(np.float32)
        scaled = X * scale + rng.normal(0, noise_std * 0.3, X.shape).astype(np.float32)
        aug_X.append(scaled)
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)


def augment_mirror_views(X, y, classes, exclude=MIRROR_EXCLUDE):
    id_to_label = {i: c for i, c in enumerate(classes)}
    safe_idx = [i for i in range(len(y)) if id_to_label[int(y[i])] not in exclude]
    if not safe_idx:
        return X, y
    idx = np.array(safe_idx, dtype=np.int64)
    mirrored = X[idx].copy()
    mirrored[:, 0::2] *= -1.0   # flip X coords
    return np.vstack([X, mirrored]), np.concatenate([y, y[idx]])


# ── Step 5: Train ─────────────────────────────────────────────────────────────

def train(merged_csv):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print(f"\n-- Training letter model on merged dataset --")
    print(f"  Source : {merged_csv}")
    print(f"  Output : {MODEL_PATH}\n")

    X_raw, labels_raw = load_dataset(merged_csv)
    y_raw, classes    = encode_labels(labels_raw)

    # Check feature dimension
    if X_raw.shape[1] != FEATURE_VECTOR_SIZE:
        print(f"  WARNING: {X_raw.shape[1]} features in CSV vs "
              f"{FEATURE_VECTOR_SIZE} expected. Some rows may be from an older extractor.")

    print(f"  Raw samples : {len(X_raw)}")
    print(f"  Features    : {X_raw.shape[1]}")
    print(f"  Classes     : {classes}\n")

    # Stratified holdout BEFORE augmentation (clean val)
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_raw, y_raw, test_size=0.18, random_state=42, stratify=y_raw
    )
    print(f"  Train (raw) : {len(X_train_raw)}")
    print(f"  Val (clean) : {len(X_val)}\n")

    # Augment
    print(f"  Augmenting ({AUG_COPIES} noise+scale copies)...")
    X_train, y_train = augment_dataset(X_train_raw, y_train_raw)
    print("  Adding mirror views (front/back camera)...")
    X_train, y_train = augment_mirror_views(X_train, y_train, classes)
    print(f"  Train (augmented) : {len(X_train)}\n")

    # RandomForest
    print("  Fitting RandomForest (200 trees, max_depth=28)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=28,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Validate
    val_acc = pipeline.score(X_val, y_val)
    y_pred  = pipeline.predict(X_val)
    print(f"  Validation accuracy: {val_acc:.2%}\n")
    report_text = classification_report(y_val, y_pred, target_names=classes, zero_division=0)
    print(report_text)

    # Save
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": classes}, f)
    print(f"  Model saved  -> {MODEL_PATH}")

    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"  Classes saved -> {CLASSES_PATH}")

    from sklearn.metrics import confusion_matrix
    metrics = {
        "validation_accuracy": float(val_acc),
        "classes": classes,
        "classification_report": classification_report(
            y_val, y_pred, target_names=classes, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "dataset_source": "merged (live webcam + image dataset)",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {METRICS_PATH}\n")

    return val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    IMAGE_EXTRACT_CSV = os.path.join(config.DATA_DIR, "dataset_from_images.csv")

    if not args.skip_extract:
        # Step 1: Extract from images
        n = extract_from_images(IMAGES_DIR, IMAGE_EXTRACT_CSV)
        if n == 0:
            print("  ERROR: No features extracted from images. Check dataset/ folder.")
            sys.exit(1)

        # Step 2: Merge
        merge_csvs(IMAGE_EXTRACT_CSV, LIVE_CSV, MERGED_CSV)
    else:
        if not os.path.exists(MERGED_CSV):
            print(f"  ERROR: {MERGED_CSV} not found. Run without --skip-extract first.")
            sys.exit(1)
        print(f"  Skipping extraction, using existing: {MERGED_CSV}")

    # Step 3: Train
    val_acc = train(MERGED_CSV)

    print("=" * 54)
    print(f"  Letter model retrained!  Accuracy: {val_acc:.2%}")
    print(f"  Next steps:")
    print(f"    1. Collect missing gesture signs (see below)")
    print(f"    2. python gesture_trainer.py")
    print(f"    3. python app.py")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    main()
