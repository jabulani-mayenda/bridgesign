"""
BridgeSign – Gesture Data Collector
====================================
Collects training data for MOTION-BASED signs (J, Z, word signs).

Unlike the static data_collector.py which saves one frame per sample,
this captures a SEQUENCE of 15 frames per sample — recording the
full movement of the gesture.

Usage:
  python gesture_collector.py --sign J --samples 200
  python gesture_collector.py --batch J Z --samples 200

Controls:
  [S] = Perform the gesture NOW (records next 15 frames)
  [N] = Skip this sign
  [Q] = Quit

Data is saved to: data/gesture_dataset.csv
Format: label, f1_1, f1_2, ..., f1_42, f2_1, ..., f15_42
        (1 label + 15 frames × 42 features = 631 values per row)
"""

import cv2
import os
import time
import argparse
import numpy as np
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from gesture_feature_extractor import GESTURE_WINDOW

try:
    from word_signs import SIGN_TIPS, display_name
except ImportError:
    SIGN_TIPS = {}
    display_name = lambda s: s.replace("_", " ")

# Extra tips for gesture letters
GESTURE_TIPS = {
    "J": "Start with pinky up (I shape), then trace a J hook downward and curl",
    "Z": "Start with index finger pointing, trace a Z zigzag in the air",
    "STATIC": "Keep hand still or make small random repositioning movements",
}
GESTURE_TIPS.update(SIGN_TIPS)

SAVE_FILE = os.path.join(config.DATA_DIR, "gesture_dataset.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BridgeSign Gesture Collector — collect motion-based sign data.",
        epilog="""
Examples:
  python gesture_collector.py --sign J --samples 200
  python gesture_collector.py --batch J Z --samples 200
  python gesture_collector.py --batch HELP STOP --samples 150
        """
    )
    parser.add_argument("--sign",    type=str, default=None, help="Single gesture to collect")
    parser.add_argument("--batch",   type=str, nargs="+", default=None, help="Multiple gestures")
    parser.add_argument("--samples", type=int, default=200, help="Samples per gesture (default: 200)")
    args = parser.parse_args()
    if args.sign is None and args.batch is None:
        parser.error("Specify --sign or --batch")
    return args


def count_existing(save_path):
    counts = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


def draw_hud(frame, sign, count, target, state_text, tip, existing=0):
    """Draw collection HUD onto frame."""
    h, w = frame.shape[:2]

    # Dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 140), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Sign label top left
    cv2.putText(frame, f"GESTURE: {display_name(sign)}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 50), 2)

    # State indicator
    if state_text == "RECORDING":
        color = (0, 0, 255)
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (w - 28, 24), 10, color, -1)
        cv2.putText(frame, "CAPTURING GESTURE...", (w - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    elif state_text == "READY":
        cv2.putText(frame, "Press [S] to perform gesture", (16, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    # Progress
    total = existing + count
    cv2.putText(frame, f"Collected: {count}/{target}  (total: {total})",
                (16, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Progress bar
    if target > 0:
        bar_w = int((count / target) * (w - 40))
        cv2.rectangle(frame, (20, h - 28), (w - 20, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 28), (20 + bar_w, h - 10), (0, 200, 80), -1)

    # Tip
    cv2.putText(frame, tip[:80], (16, h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 200, 255), 1)

    # Existing count
    if existing > 0:
        cv2.putText(frame, f"Existing: {existing}", (w - 180, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1)

    return frame


def collect_one_gesture(sign, target, save_path, cap, detector, extractor, existing=0):
    """
    Collect gesture samples for one sign.

    Each sample = user presses [S], then we capture GESTURE_WINDOW frames
    of landmarks as one sequence.

    Returns: number of new samples collected, or -1 if quit.
    """
    tip = GESTURE_TIPS.get(sign, "Perform the gesture naturally in front of the camera.")
    count = 0
    capturing = False
    capture_buffer = []

    print(f"\n  ── Collecting gesture: {display_name(sign)} ──")
    print(f"     Existing: {existing}  |  Target new: {target}")
    print(f"     Press [S] to perform the gesture, we'll capture {GESTURE_WINDOW} frames")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        frame, _ = detector.find_hands(frame, draw=True)

        if capturing:
            # Capture frames for one gesture sample
            lm_list = detector.get_landmarks(frame, hand_no=0)
            if lm_list:
                features = extractor.extract_features(lm_list)
                if features is not None:
                    capture_buffer.append(features)

            # Check if we have enough frames
            if len(capture_buffer) >= GESTURE_WINDOW:
                # Save this sequence as one sample
                flat = np.concatenate(capture_buffer[:GESTURE_WINDOW])
                with open(save_path, "a") as f:
                    f.write(f"{sign}," + ",".join(map(str, flat)) + "\n")
                count += 1
                capturing = False
                capture_buffer = []
                if count % 10 == 0:
                    print(f"     {count}/{target} gesture samples collected...")

            frame = draw_hud(frame, sign, count, target, "RECORDING", tip, existing)
        else:
            frame = draw_hud(frame, sign, count, target, "READY", tip, existing)

        cv2.imshow("BridgeSign - Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not capturing:
            capturing = True
            capture_buffer = []
        elif key == ord("n"):
            print(f"     Skipped {display_name(sign)} ({count} collected)")
            return count
        elif key == ord("q"):
            print(f"     Quit ({count} collected for {display_name(sign)})")
            return -1

    print(f"\n     ✅  Done! {count} gesture samples for '{display_name(sign)}'")
    return count


def main():
    args = parse_args()

    if args.batch:
        signs = [s.upper() for s in args.batch]
    else:
        signs = [args.sign.upper()]

    existing_counts = count_existing(SAVE_FILE)

    print("\n" + "=" * 62)
    print(f"  BridgeSign Gesture Collector")
    print(f"  Signs    : {', '.join(display_name(s) for s in signs)}")
    print(f"  Samples  : {args.samples} per sign")
    print(f"  Window   : {GESTURE_WINDOW} frames per gesture")
    print(f"  Save to  : {SAVE_FILE}")
    print("=" * 62)

    for s in signs:
        cur = existing_counts.get(s, 0)
        print(f"    {display_name(s):12s}  {cur:4d} existing")

    print(f"\n  Controls: [S] perform gesture  [N] skip  [Q] quit\n")

    detector = HandDetector(max_hands=1)
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    total_collected = {}

    try:
        for sign in signs:
            existing = existing_counts.get(sign, 0)
            needed = max(0, args.samples - existing)
            if needed == 0:
                print(f"\n  ⏭  {display_name(sign)} already has {existing} samples — skipping")
                total_collected[sign] = 0
                continue

            result = collect_one_gesture(sign, needed, SAVE_FILE,
                                         cap, detector, extractor, existing)
            if result == -1:
                total_collected[sign] = 0
                break
            total_collected[sign] = result

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 62)
        print("  Gesture Collection Summary")
        print("=" * 62)
        grand_total = 0
        for s in signs:
            new = total_collected.get(s, 0)
            old = existing_counts.get(s, 0)
            grand_total += new
            print(f"    {display_name(s):12s}  +{new:4d}  (total: {old + new})")
        print(f"\n  Total new samples: {grand_total}")
        if grand_total > 0:
            print(f"  Saved to: {SAVE_FILE}")
            print(f"\n  Next step: python gesture_trainer.py")
        print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
