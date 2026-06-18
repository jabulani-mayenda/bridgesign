"""BridgeSign system audit: coverage for alphabet, gestures, avatar, and gaps."""

import csv
import json
import string
from pathlib import Path

from word_signs import ALL_WORD_SIGNS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ANIM_DIR = BASE_DIR / "static" / "avatar" / "animations"

ALPHABET = list(string.ascii_uppercase)
FACE_SCAN_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "dataset",
    "models",
    "node_modules",
    "static/vendor",
    "static/lib",
}


def load_counts(path: Path):
    counts = {}
    if not path.exists():
        return counts
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label = row[0].strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
    return counts


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def print_list(title, items):
    print(title)
    if items:
        print("  " + ", ".join(items))
    else:
        print("  None")
    print()


def project_files_for_face_scan():
    import os
    for root, dirs, files in os.walk(BASE_DIR):
        # Prune directories starting with . or in FACE_SCAN_SKIP_DIRS in-place
        dirs[:] = [d for d in dirs if d not in FACE_SCAN_SKIP_DIRS and not d.startswith(".")]
        for f in files:
            yield Path(root) / f



def main():
    static_counts = load_counts(DATA_DIR / "dataset.csv")
    gesture_counts = load_counts(DATA_DIR / "gesture_dataset.csv")

    sign_classes = load_json(MODELS_DIR / "sign_model_classes.json", [])
    gesture_lstm_meta = load_json(MODELS_DIR / "gesture_lstm_meta.json", {})
    gesture_lstm_classes = gesture_lstm_meta.get("classes", [])
    gesture_rf_classes = load_json(MODELS_DIR / "gesture_model_classes.json", [])

    animation_labels = {p.stem for p in ANIM_DIR.glob("*.json")}
    gesture_words_trained = sorted(
        label for label in gesture_lstm_classes
        if label not in {"STATIC"} and len(label.replace("_", "")) > 1
    )
    target_word_signs = sorted(set(ALL_WORD_SIGNS))

    missing_static_dataset = sorted(set(ALPHABET) - set(static_counts))
    missing_static_model = sorted(set(ALPHABET) - set(sign_classes))
    missing_static_avatar = sorted(set(ALPHABET) - animation_labels)

    missing_trained_word_gestures = sorted(set(target_word_signs) - set(gesture_words_trained))
    missing_collected_word_gestures = sorted(
        set(target_word_signs) - {label for label in gesture_counts if len(label.replace("_", "")) > 1}
    )
    missing_avatar_for_live_gestures = sorted(
        {label for label in gesture_lstm_classes if label != "STATIC"} - animation_labels
    )

    face_related = sorted(
        p.relative_to(BASE_DIR).as_posix() for p in project_files_for_face_scan()
        if p.is_file() and ("face" in p.name.lower() or "emotion" in p.name.lower())
    )

    print("=" * 64)
    print("BridgeSign System Audit")
    print("=" * 64)
    print()

    print("Alphabet coverage")
    print(f"  Dataset classes : {len(static_counts)} / 26")
    print(f"  Model classes   : {len(sign_classes)} / 26")
    print(f"  Avatar clips    : {len([x for x in animation_labels if len(x) == 1 and x.isalpha()])} / 26")
    print()
    print_list("Missing alphabet dataset labels:", missing_static_dataset)
    print_list("Missing alphabet model labels:", missing_static_model)
    print_list("Missing alphabet avatar clips:", missing_static_avatar)

    print("Gesture coverage")
    print(f"  Gesture dataset labels : {len(gesture_counts)}")
    print(f"  LSTM live classes      : {len(gesture_lstm_classes)}")
    print(f"  RF fallback classes    : {len(gesture_rf_classes)}")
    best_acc = gesture_lstm_meta.get("best_val_accuracy")
    if best_acc is not None:
        print(f"  LSTM validation acc    : {best_acc:.2%}")
    print()
    print_list("Live gesture classes:", [label for label in gesture_lstm_classes if label != "STATIC"])
    print_list("Word-sign gestures still missing from the trained live model:", missing_trained_word_gestures)
    print_list("Word-sign gestures still missing from the collected dataset:", missing_collected_word_gestures)
    print_list("Avatar clips missing for trained live gestures:", missing_avatar_for_live_gestures)

    print("Avatar coverage")
    print(f"  Total animation clips : {len(animation_labels)}")
    print(f"  Word gesture clips    : {len([x for x in animation_labels if len(x.replace('_', '')) > 1])}")
    print()

    print("Facial module")
    if face_related:
        print("  Face-related files found:")
        for name in face_related:
            print(f"  - {name}")
    else:
        print("  No facial recognition / facial expression module is implemented yet.")
    print()

    if missing_collected_word_gestures:
        preview = " ".join(missing_collected_word_gestures[:12])
        print("Suggested next collection batch")
        print(f"  python gesture_collector.py --batch {preview} --samples 200")
        print()


if __name__ == "__main__":
    main()
