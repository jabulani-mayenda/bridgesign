"""Daily BridgeSign accuracy report.

Prints the current model metrics, dataset coverage, and the next collection
targets for improving live hand-sign accuracy.
"""

import csv
import json
import string
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))

try:
    from word_signs import PHASE_1, PHASE_2, display_name
except Exception:
    PHASE_1 = []
    PHASE_2 = []
    display_name = lambda value: str(value).replace("_", " ")

STATIC_TARGET_TOTAL = 220
GESTURE_TARGET_NEW = 40
GESTURE_TARGET_TOTAL = 180


def load_counts(path):
    counts = {}
    if not path.exists():
        return counts
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label = row[0].strip().upper()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def load_json(path, default):
    if not path.exists():
        return default
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return default


def pct(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def low_sample_labels(counts, labels, target):
    return sorted(
        ((label, counts.get(label, 0)) for label in labels),
        key=lambda item: (item[1], item[0]),
    )


def weak_report_labels(metrics, min_f1=0.92):
    report = metrics.get("classification_report") or {}
    weak = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        score = values.get("f1-score")
        if isinstance(score, (int, float)) and score < min_f1:
            weak.append((label, score))
    return sorted(weak, key=lambda item: item[1])


def print_collection_block(title, items, command):
    print(title)
    if not items:
        print("  None")
        return
    for label, count in items:
        print(f"  {display_name(label):14s} {count:4d} samples")
    print(f"  Next command: {command}")


def main():
    static_counts = load_counts(DATA_DIR / "dataset.csv")
    gesture_counts = load_counts(DATA_DIR / "gesture_dataset.csv")
    sign_metrics = load_json(MODELS_DIR / "sign_model_metrics.json", {})
    gesture_metrics = load_json(MODELS_DIR / "gesture_model_metrics.json", {})
    lstm_meta = load_json(MODELS_DIR / "gesture_lstm_meta.json", {})

    alphabet = list(string.ascii_uppercase)
    gesture_targets = ["STATIC", "J", "Z"] + list(PHASE_1) + list(PHASE_2)

    low_static = [
        item for item in low_sample_labels(static_counts, alphabet, STATIC_TARGET_TOTAL)
        if item[1] < STATIC_TARGET_TOTAL
    ][:8]
    low_gesture = [
        item for item in low_sample_labels(gesture_counts, gesture_targets, GESTURE_TARGET_TOTAL)
        if item[1] < GESTURE_TARGET_TOTAL
    ][:8]
    weak_static = weak_report_labels(sign_metrics)

    print("=" * 64)
    print("BridgeSign Daily Accuracy Report")
    print("=" * 64)
    print()

    print("Model accuracy")
    print(f"  Static hand signs validation : {pct(sign_metrics.get('validation_accuracy'))}")
    print(f"  Static hand signs CV mean    : {pct(sign_metrics.get('cv_accuracy_mean'))}")
    print(f"  Motion RF validation         : {pct(gesture_metrics.get('validation_accuracy'))}")
    print(f"  Motion LSTM best validation  : {pct(lstm_meta.get('best_val_accuracy'))}")
    print()

    print("Dataset coverage")
    print(f"  Static samples  : {sum(static_counts.values())} across {len(static_counts)} labels")
    print(f"  Gesture samples : {sum(gesture_counts.values())} across {len(gesture_counts)} labels")
    print()

    static_labels = " ".join(label for label, _ in low_static)
    gesture_labels = " ".join(label for label, _ in low_gesture)
    print_collection_block(
        "Lowest static hand-sign sample counts",
        low_static,
        f"python data_collector.py --batch {static_labels} --samples {STATIC_TARGET_TOTAL}" if static_labels else "n/a",
    )
    print()
    print_collection_block(
        "Lowest live motion/word-sign sample counts",
        low_gesture,
        f"python gesture_collector.py --batch {gesture_labels} --samples {GESTURE_TARGET_NEW}" if gesture_labels else "n/a",
    )
    print()

    print("Weak static classes by f1-score")
    if weak_static:
        for label, score in weak_static[:8]:
            print(f"  {display_name(label):14s} {score * 100:.2f}% f1")
    else:
        print("  None below threshold")
    print()

    print("Daily retrain checklist")
    print("  1. Collect the lowest static labels until each reaches the target total.")
    print("  2. Collect 40 new samples for the lowest gesture labels.")
    print("  3. Run: python model_trainer.py")
    print("  4. Run: python gesture_trainer.py")
    print("  5. Run: python lstm_gesture_trainer.py")
    print("  6. Run: python _system_audit.py")


if __name__ == "__main__":
    main()
