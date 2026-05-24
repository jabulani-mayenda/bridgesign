"""
recollect_signs.py — Re-record specific hand signs.

Purges the existing rows for the target letters from dataset.csv,
then walks you through re-recording them one at a time.

Usage:
    python recollect_signs.py
"""

import cv2
import os
import sys
import time
import shutil
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor

try:
    from word_signs import SIGN_TIPS, display_name
except ImportError:
    SIGN_TIPS    = {}
    display_name = lambda s: s.replace("_", " ")

# ── Letters that need to be re-recorded ──────────────────────────────────────
REDO_LETTERS = ["G","J","K","M","N","P","Q","R","S","T","U","V","X","Y"]
SAMPLES_PER_LETTER = 120          # increase from typical 100 for better coverage
SAVE_INTERVAL      = 0.05         # seconds between saved frames
DATASET_PATH       = os.path.join(config.DATA_DIR, "dataset.csv")
BACKUP_PATH        = os.path.join(config.DATA_DIR, "dataset_before_recollect.csv")

# ─────────────────────────────────────────────────────────────────────────────

def purge_letters(dataset_path: str, letters: list[str]) -> int:
    """Remove all rows for the given letters from dataset_path. Returns removed count."""
    if not os.path.exists(dataset_path):
        return 0

    upper = {l.upper() for l in letters}
    kept, removed = [], 0

    with open(dataset_path, "r") as f:
        for line in f:
            label = line.split(",")[0].strip().upper()
            if label in upper:
                removed += 1
            else:
                kept.append(line)

    with open(dataset_path, "w") as f:
        f.writelines(kept)

    return removed


def draw_overlay(frame, sign: str, count: int, target: int, recording: bool,
                 tip: str, letter_idx: int, total_letters: int):
    """Draw a clean HUD onto the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 140), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Progress through letter list
    cv2.putText(frame,
                f"Letter {letter_idx + 1} of {total_letters}",
                (w - 200, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Big sign label
    cv2.putText(frame, f"TARGET: {sign}", (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 160, 40), 2)

    if recording:
        bar_w = int((count / target) * (w - 40))
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 12), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 12), (0, 200, 80), -1)
        cv2.putText(frame, f"Recording: {count}/{target}",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 28, 24), 10, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (w - 60, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
    else:
        cv2.putText(frame,
                    "Press [S] to start recording  |  [N] skip  |  [Q] quit",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Tip text
    words = tip.split()
    line, lines = "", []
    for w_txt in words:
        if len(line + " " + w_txt) > 68:
            lines.append(line); line = w_txt
        else:
            line = (line + " " + w_txt).strip()
    if line:
        lines.append(line)
    for i, ln in enumerate(lines[-2:]):
        cv2.putText(frame, ln, (20, h - 130 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 200, 255), 1)

    return frame


def show_completion(frame, sign: str, target: int, next_sign: str | None):
    h, w = frame.shape[:2]
    black = frame.copy()
    cv2.rectangle(black, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.putText(black, f"✓ {sign} — {target} samples done!",
                (w // 2 - 220, h // 2 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 80), 2)
    if next_sign:
        cv2.putText(black, f"Next up: {next_sign}  — press [S] or wait…",
                    (w // 2 - 200, h // 2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    else:
        cv2.putText(black, "All letters complete! Press [Q] to finish.",
                    (w // 2 - 220, h // 2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 200, 255), 1)
    return black


def main():
    print("\n" + "=" * 60)
    print("  BridgeSign — Targeted Re-Collection")
    print(f"  Letters : {', '.join(REDO_LETTERS)}")
    print(f"  Samples : {SAMPLES_PER_LETTER} per letter")
    print(f"  Dataset : {DATASET_PATH}")
    print("=" * 60)

    # ── Step 1: Backup ────────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        shutil.copy2(DATASET_PATH, BACKUP_PATH)
        print(f"\n  📦 Backup saved → {BACKUP_PATH}")

    # ── Step 2: Purge old rows ────────────────────────────────────
    removed = purge_letters(DATASET_PATH, REDO_LETTERS)
    print(f"  🗑  Purged {removed} existing rows for: {', '.join(REDO_LETTERS)}")
    print("\n  Controls inside camera window:")
    print("    [S]  — start / next letter")
    print("    [N]  — skip current letter")
    print("    [Q]  — quit (samples already saved are kept)\n")
    input("  Press ENTER to open camera and begin…\n")

    # ── Step 3: Camera setup ──────────────────────────────────────
    detector  = HandDetector(max_hands=1)
    extractor = FeatureExtractor()
    cap       = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FPS)

    letter_idx = 0
    total      = len(REDO_LETTERS)

    try:
        while letter_idx < total:
            sign      = REDO_LETTERS[letter_idx]
            tip       = SIGN_TIPS.get(sign, "Hold the sign naturally in front of the camera.")
            next_sign = REDO_LETTERS[letter_idx + 1] if letter_idx + 1 < total else None
            recording  = False
            count      = 0
            last_saved = 0
            completed  = False

            print(f"\n  ── [{letter_idx + 1}/{total}] Sign: {sign} ──────────────────")

            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05); continue

                frame = cv2.flip(frame, 1)
                frame, _ = detector.find_hands(frame, draw=True)

                if completed:
                    disp = show_completion(frame, sign, SAMPLES_PER_LETTER, next_sign)
                    cv2.imshow("BridgeSign — Re-Collection", disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        raise KeyboardInterrupt
                    # auto-advance after 2.5 s or on S
                    if (time.time() - completed > 2.5) or k == ord("s"):
                        break
                    continue

                if recording and count < SAMPLES_PER_LETTER:
                    now = time.time()
                    if now - last_saved >= SAVE_INTERVAL:
                        lm_list = detector.get_landmarks(frame, hand_no=0)
                        if lm_list:
                            features = extractor.extract_features(lm_list)
                            with open(DATASET_PATH, "a") as f:
                                f.write(f"{sign}," + ",".join(map(str, features)) + "\n")
                            count     += 1
                            last_saved = now
                            if count % 25 == 0:
                                print(f"    {count}/{SAMPLES_PER_LETTER} samples…")

                    if count >= SAMPLES_PER_LETTER:
                        recording = False
                        completed = time.time()
                        print(f"  ✅  {sign} done — {SAMPLES_PER_LETTER} samples saved.")
                        continue

                frame = draw_overlay(frame, sign, count, SAMPLES_PER_LETTER,
                                     recording, tip, letter_idx, total)
                cv2.imshow("BridgeSign — Re-Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s") and not recording:
                    recording = True
                    print(f"  ▶  Recording {sign}…")
                elif key == ord("n"):
                    print(f"  ⏭  Skipped {sign}.")
                    break
                elif key == ord("q"):
                    raise KeyboardInterrupt

            letter_idx += 1

    except KeyboardInterrupt:
        print("\n  ⛔  Interrupted — samples collected so far are saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  Re-collection complete!")
    print(f"  Dataset : {DATASET_PATH}")
    print("\n  Next step — retrain the model:")
    print("    python build_dataset.py")
    print("    python model_trainer.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
