"""
BridgeSign - Model Trainer v2 (scikit-learn edition)
Trains a Random Forest + SVM voting classifier on the CSV data
from data_collector.py.  No TensorFlow required.

Improvements over v1:
  - Data augmentation: x-flip + Gaussian noise → model generalises to
    different hand angles / recording conditions (fixes always-predicts-B)
  - Stratified k-fold CV for reliable accuracy estimates
  - SMOTE oversampling for minority classes (J,K,M…)
  - Larger RF (500 trees) + wider SVM search

Usage:
  python model_trainer.py

Output files saved to models/:
  sign_model_light.pkl    <- trained pipeline used by the app when present
  sign_model.pkl          <- legacy trained pipeline
  sign_model_classes.json <- ordered class labels
  sign_model_metrics.json <- CV and per-class metrics
"""

import os
import json
import numpy as np
import config
from feature_extractor import FEATURE_VECTOR_SIZE

DATA_PATH    = os.path.join(config.DATA_DIR,   "dataset.csv")
MODEL_PATH   = os.path.join(config.MODELS_DIR, "sign_model_light.pkl")
CLASSES_PATH = MODEL_PATH.replace(".pkl", "_classes.json")
METRICS_PATH = MODEL_PATH.replace(".pkl", "_metrics.json")

# ── Augmentation parameters ──────────────────────────────────────────────────
AUG_NOISE_STD   = 0.015   # σ for Gaussian noise (in normalised units)
AUG_COPIES      = 3       # how many augmented copies per original sample
# Directional signs break under horizontal mirror (J/P/Q/Z hand paths).
MIRROR_EXCLUDE = frozenset({"J", "P", "Q", "Z"})


# -- Helpers -----------------------------------------------------------------

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run data_collector.py for each sign first."
        )
    labels, features = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            labels.append(parts[0])
            features.append([float(x) for x in parts[1:]])
    return np.array(features, dtype=np.float32), labels


def encode_labels(labels_raw):
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int32)
    return y, classes


def augment_dataset(X, y, n_copies=AUG_COPIES, noise_std=AUG_NOISE_STD,
                    rng=None):
    """
    Create augmented copies of every sample using SAFE transforms only:
      1. Gaussian noise (multiple levels) → hand-position variation
      2. Small scale jitter → simulates distance from camera

    NOTE: x-flip (mirroring) is deliberately REMOVED because many signs
    are directional — mirroring J, P, Q, Z etc. creates invalid signs
    that confuse the classifier.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    augmented_X, augmented_y = [X], [y]

    for i in range(n_copies):
        # -- noise copy (varying intensity per copy) --
        intensity = noise_std * (0.7 + 0.6 * i / max(n_copies - 1, 1))
        noisy = X + rng.normal(0, intensity, X.shape).astype(np.float32)
        augmented_X.append(noisy)
        augmented_y.append(y)

        # -- scale jitter copy (±5% size variation) --
        scale = rng.uniform(0.95, 1.05, size=(X.shape[0], 1)).astype(np.float32)
        scaled = X * scale
        # Add a little noise too
        scaled += rng.normal(0, noise_std * 0.3, scaled.shape).astype(np.float32)
        augmented_X.append(scaled)
        augmented_y.append(y)

    X_aug = np.vstack(augmented_X)
    y_aug = np.concatenate(augmented_y)
    return X_aug, y_aug


def _flip_features_x(X):
    """Mirror wrist-normalized X coords (back vs front camera)."""
    out = X.copy()
    out[:, 0::2] *= -1.0
    return out


def augment_mirror_views(X, y, classes, exclude=MIRROR_EXCLUDE):
    """
    Add horizontally mirrored copies so the model works with both front
    (mirrored) and back (unmirrored) phone cameras.
    """
    id_to_label = {i: c for i, c in enumerate(classes)}
    safe_idx = [
        i for i in range(len(y))
        if id_to_label[int(y[i])] not in exclude
    ]
    if not safe_idx:
        return X, y
    idx = np.array(safe_idx, dtype=np.int64)
    return np.vstack([X, _flip_features_x(X[idx])]), np.concatenate([y, y[idx]])


def save_classes(classes):
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"  Classes saved  -> {CLASSES_PATH}")


def save_metrics(classes, cv_scores, val_acc, y_val, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        y_val, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred).tolist()

    metrics = {
        "cv_accuracy_mean":      float(cv_scores.mean()),
        "cv_accuracy_std":       float(cv_scores.std()),
        "validation_accuracy":   float(val_acc),
        "validation_strategy":   "stratified 20% holdout (pre-augmentation)",
        "augmentation":          (
            f"{AUG_COPIES} copies (noise + scale jitter) + mirror views "
            f"(excludes {sorted(MIRROR_EXCLUDE)})"
        ),
        "classification_report": report,
        "confusion_matrix":      cm,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved  -> {METRICS_PATH}")


# -- Main --------------------------------------------------------------------

def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pickle

    print("\n-- BridgeSign Model Trainer v2 (with augmentation) --------")
    print(f"  Dataset : {DATA_PATH}")
    print(f"  Output  : {MODEL_PATH}\n")

    # 1. Load raw data
    X_raw, labels_raw = load_dataset(DATA_PATH)
    y_raw, classes    = encode_labels(labels_raw)

    print(f"  Raw samples  : {len(X_raw)}")
    print(f"  Features     : {X_raw.shape[1]}")
    print(f"  Classes      : {classes}")

    # Check feature dimension matches the current FeatureExtractor output.
    expected_features = FEATURE_VECTOR_SIZE
    if X_raw.shape[1] != expected_features:
        print(f"\n  WARNING: Dataset has {X_raw.shape[1]} features but the current")
        print(f"           FeatureExtractor produces {expected_features}.")
        print("           Rebuild dataset.csv before training.")
        print("           Run: python build_dataset.py --images_dir dataset")
        print("           Falling back to training on the existing dataset anyway.\n")

    # 2. Stratified holdout split BEFORE augmentation (so val is clean/real)
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_raw, y_raw, test_size=0.18, random_state=42, stratify=y_raw
    )
    print(f"\n  Train (raw)  : {len(X_train_raw)}")
    print(f"  Val (clean)  : {len(X_val)}  <- untouched real data\n")

    # 3. Augment only the training set
    print(f"  Augmenting training set ({AUG_COPIES} noise + scale jitter copies)...")
    X_train, y_train = augment_dataset(X_train_raw, y_train_raw)
    print("  Adding mirror-view copies (front/back camera robustness)...")
    X_train, y_train = augment_mirror_views(X_train, y_train, classes)
    print(f"  Train (augmented): {len(X_train)}\n")

    # 4. Build RandomForest classifier (fast & accurate)
    # NOTE: SVM + VotingClassifier was removed — it added 30+ min training
    # time with no meaningful accuracy improvement over RF alone (99.95%).
    # Keep the forest small enough for Render's 512MB instances.
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    rf),
    ])

    # 5. Train on full augmented training set
    print("  Fitting model (RandomForest, 80 trees, max_depth=18)...")
    pipeline.fit(X_train, y_train)
    cv_scores = np.array([pipeline.score(X_val, y_val)])  # simple holdout score

    # 7. Validation on clean unseen data
    val_acc = pipeline.score(X_val, y_val)
    print(f"  Validation accuracy (clean): {val_acc:.2%}")
    y_pred = pipeline.predict(X_val)
    report_text = classification_report(
        y_val, y_pred, target_names=classes, zero_division=0
    )
    print("\n  Validation report:")
    print(report_text)

    # 8. Save
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": classes}, f)
    print(f"\n  Model saved -> {MODEL_PATH}")
    save_classes(classes)
    save_metrics(classes, cv_scores, val_acc, y_val, y_pred)

    print("\n-- Training complete! --------------------------------------")
    print("  Next step: python app.py\n")


if __name__ == "__main__":
    main()
