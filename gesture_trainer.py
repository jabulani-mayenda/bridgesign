"""
BridgeSign – Gesture Model Trainer
====================================
Trains a RandomForest classifier on gesture (motion) data collected
by gesture_collector.py.

Each sample in gesture_dataset.csv is a sequence of 15 frames of
hand landmarks, capturing the full movement of a gesture.

Usage:
  python gesture_trainer.py

Output:
  models/gesture_model.pkl          <- trained pipeline
  models/gesture_model_classes.json <- class labels
  models/gesture_model_metrics.json <- accuracy metrics
"""

import os
import json
import pickle
import numpy as np
import config
from gesture_feature_extractor import (
    GESTURE_WINDOW, STATIC_FEATURES, TOTAL_GESTURE_FEATURES,
    extract_from_sequence
)

DATA_PATH    = os.path.join(config.DATA_DIR, "gesture_dataset.csv")
MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_model.pkl")
CLASSES_PATH = MODEL_PATH.replace(".pkl", "_classes.json")
METRICS_PATH = MODEL_PATH.replace(".pkl", "_metrics.json")


def load_gesture_dataset(path):
    """
    Load gesture_dataset.csv.
    Each row: label, f1_1, f1_2, ..., f15_42
    (1 label + 15×42 = 630 raw values)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gesture dataset not found at '{path}'.\n"
            "Run gesture_collector.py first."
        )

    labels, sequences = [], []
    expected_vals = GESTURE_WINDOW * STATIC_FEATURES  # 630

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            label = parts[0]
            vals = [float(x) for x in parts[1:]]

            if len(vals) < expected_vals:
                print(f"  Warning: line {line_no} has {len(vals)} values, "
                      f"expected {expected_vals} — skipping")
                continue

            # Reshape into frames and extract temporal features
            raw_frames = np.array(vals[:expected_vals], dtype=np.float32)
            frame_list = raw_frames.reshape(GESTURE_WINDOW, STATIC_FEATURES)
            gesture_feats = extract_from_sequence(list(frame_list))

            if gesture_feats is not None:
                labels.append(label)
                sequences.append(gesture_feats)

    X = np.array(sequences, dtype=np.float32)
    return X, labels


def augment_gestures(X, y, n_copies=3, noise_std=0.01, rng=None):
    """
    Augment gesture data with noise and time-shift.
    Safe transforms only — no flipping (directional gestures).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    aug_X, aug_y = [X], [y]

    for i in range(n_copies):
        # Noise copy
        intensity = noise_std * (0.7 + 0.6 * i / max(n_copies - 1, 1))
        noisy = X + rng.normal(0, intensity, X.shape).astype(np.float32)
        aug_X.append(noisy)
        aug_y.append(y)

        # Scale jitter
        scale = rng.uniform(0.95, 1.05, size=(X.shape[0], 1)).astype(np.float32)
        scaled = X * scale
        scaled += rng.normal(0, noise_std * 0.3, scaled.shape).astype(np.float32)
        aug_X.append(scaled)
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print("\n-- BridgeSign Gesture Trainer --------")
    print(f"  Dataset : {DATA_PATH}")
    print(f"  Output  : {MODEL_PATH}\n")

    # 1. Load
    X, labels_raw = load_gesture_dataset(DATA_PATH)
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int32)

    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Gestures : {classes}")

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.18, random_state=42, stratify=y
    )
    print(f"\n  Train : {len(X_train)}")
    print(f"  Val   : {len(X_val)}\n")

    # 3. Augment
    print("  Augmenting (3 copies: noise + scale jitter)...")
    X_train, y_train = augment_gestures(X_train, y_train)
    print(f"  Train (augmented): {len(X_train)}\n")

    # 4. Train
    print("  Training RandomForest (300 trees)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Validate
    val_acc = pipeline.score(X_val, y_val)
    y_pred = pipeline.predict(X_val)
    print(f"  Validation accuracy: {val_acc:.2%}")
    print(classification_report(y_val, y_pred, target_names=classes, zero_division=0))

    # 6. Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": classes}, f)
    print(f"  Model saved -> {MODEL_PATH}")

    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)

    metrics = {
        "validation_accuracy": float(val_acc),
        "classes": classes,
        "samples_per_class": {c: int(np.sum(y == label_to_id[c])) for c in classes},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n-- Gesture training complete! --------")
    print(f"  Next: python app.py\n")


if __name__ == "__main__":
    main()
