# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BridgeSign – Video Gesture Extractor
======================================
Trains the gesture model from VIDEO FILES instead of live camera capture.
This is better when:
  - Your PC lags during live collection
  - You want to record with a phone/better camera
  - You want to review and delete bad takes before training

HOW TO USE
----------
1. Create a folder called:  gesture_videos/
2. Inside it, create one folder per gesture (name = exact label):

   gesture_videos/
     AGAIN/
       take1.mp4
       take2.mp4
     AMBULANCE/
       take1.mp4
     ANGRY/
       take1.mp4
     DANGER/
       take1.mp4
     DOCTOR/
       take1.mp4
     EMERGENCY/
       take1.mp4
     FIRE/
       take1.mp4
     FOOD/
       take1.mp4
     STATIC/
       take1.mp4   ← record yourself holding letter positions still

3. Run:
   python video_gesture_collector.py

   To preview what's being extracted:
   python video_gesture_collector.py --preview

   To only process specific gestures:
   python video_gesture_collector.py --gestures AGAIN FIRE STATIC

HOW TO RECORD GOOD VIDEOS
--------------------------
- Use your phone camera (better than PC webcam for this)
- Good lighting, plain background
- For motion signs (AGAIN, FIRE etc.): perform the gesture 5–10 times in one video
- For STATIC: hold different letter shapes for 2–3 seconds each in one video
- Videos can be mp4, avi, mov — any format OpenCV can read
- Landscape or portrait, any resolution — script handles it

OUTPUT
------
Appends to: data/gesture_dataset.csv  (same file gesture_collector.py uses)
Then run:   python gesture_trainer.py
"""

import cv2
import os
import argparse
import numpy as np
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from gesture_feature_extractor import GESTURE_WINDOW

VIDEOS_DIR = "gesture_videos"
SAVE_FILE  = os.path.join(config.DATA_DIR, "gesture_dataset.csv")

# Sliding window stride: extract one sequence every STRIDE frames.
# Lower = more sequences per video (more data but more overlap).
# Higher = fewer sequences but more different.
STRIDE = 5

# Minimum fraction of frames in a window that must have a hand detected.
# 1.0 = all 15 frames must have a hand (strict — recommended)
MIN_HAND_FRAMES_RATIO = 0.90  # 90% of 15 frames = at least 13/15

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args():
    p = argparse.ArgumentParser(
        description="BridgeSign Video Gesture Extractor — build gesture training data from video files."
    )
    p.add_argument("--videos_dir", type=str, default=VIDEOS_DIR,
                   help=f"Folder containing gesture subfolders (default: {VIDEOS_DIR})")
    p.add_argument("--output",     type=str, default=SAVE_FILE,
                   help=f"Output CSV path (default: {SAVE_FILE})")
    p.add_argument("--stride",     type=int, default=STRIDE,
                   help=f"Extract a sequence every N frames (default: {STRIDE})")
    p.add_argument("--preview",    action="store_true",
                   help="Show a preview window while extracting (useful to check quality)")
    p.add_argument("--gestures",   nargs="+", default=None,
                   help="Only process these gesture labels (space-separated)")
    p.add_argument("--max_per_video", type=int, default=200,
                   help="Max sequences to extract per video file (default: 200)")
    return p.parse_args()


def count_existing(save_path):
    counts = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


def extract_from_video(video_path, label, save_path, detector, extractor,
                        stride, preview, max_per_video):
    """
    Extract gesture sequences from one video file.
    Returns number of sequences saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      FAIL Could not open: {os.path.basename(video_path)}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Read all frames into memory (videos are short so this is fine)
    all_frames    = []
    all_landmarks = []   # list of (lm_list or None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # mirror so it matches webcam orientation
        frame, _ = detector.find_hands(frame, draw=False)
        lm_list  = detector.get_landmarks(frame, hand_no=0)
        all_frames.append(frame)
        all_landmarks.append(lm_list if lm_list else None)

    cap.release()

    if len(all_frames) < GESTURE_WINDOW:
        print(f"      SKIP Video too short ({len(all_frames)} frames < {GESTURE_WINDOW} needed): "
              f"{os.path.basename(video_path)}")
        return 0

    # Sliding window extraction
    saved = 0
    i     = 0

    while i <= len(all_frames) - GESTURE_WINDOW and saved < max_per_video:
        window_lms = all_landmarks[i : i + GESTURE_WINDOW]

        # Check enough frames have a hand
        hand_count = sum(1 for lm in window_lms if lm is not None)
        if hand_count < int(GESTURE_WINDOW * MIN_HAND_FRAMES_RATIO):
            i += max(1, stride // 2)  # move forward a bit and retry
            continue

        # Extract features from each frame that has a hand; fill missing from neighbours
        feat_sequence = []
        last_valid    = None
        for lm in window_lms:
            if lm is not None:
                feats = extractor.extract_features(lm)
                if feats is not None:
                    last_valid = feats
                    feat_sequence.append(feats)
                else:
                    feat_sequence.append(last_valid)
            else:
                feat_sequence.append(last_valid)

        # Filter out windows where we never had a valid feature
        if any(f is None for f in feat_sequence):
            i += stride
            continue

        # Save this sequence
        flat = np.concatenate(feat_sequence[:GESTURE_WINDOW])
        with open(save_path, "a") as f:
            f.write(f"{label}," + ",".join(map(str, flat)) + "\n")
        saved += 1

        # Preview
        if preview:
            frame = all_frames[i + GESTURE_WINDOW // 2].copy()
            cv2.putText(frame, f"{label} seq#{saved}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Video Extractor Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        i += stride

    return saved


def main():
    args = parse_args()

    if not os.path.isdir(args.videos_dir):
        print(f"\n  ERROR Videos folder not found: {os.path.abspath(args.videos_dir)}")
        print(f"\n  Create it with one subfolder per gesture, e.g.:")
        print(f"    {args.videos_dir}/AGAIN/take1.mp4")
        print(f"    {args.videos_dir}/FIRE/take1.mp4")
        print(f"    {args.videos_dir}/STATIC/take1.mp4")
        return

    # Find gesture folders
    gesture_dirs = sorted(
        d for d in os.listdir(args.videos_dir)
        if os.path.isdir(os.path.join(args.videos_dir, d))
    )
    if not gesture_dirs:
        print(f"  ERROR No gesture subfolders found in {args.videos_dir}")
        return

    if args.gestures:
        gesture_dirs = [g for g in gesture_dirs if g.upper() in [x.upper() for x in args.gestures]]

    existing_counts = count_existing(args.output)

    print("\n" + "=" * 60)
    print("  BridgeSign Video Gesture Extractor")
    print(f"  Videos dir : {os.path.abspath(args.videos_dir)}")
    print(f"  Output     : {args.output}")
    print(f"  Window     : {GESTURE_WINDOW} frames per sequence")
    print(f"  Stride     : every {args.stride} frames")
    print("=" * 60)

    print(f"\n  Gestures to process: {gesture_dirs}")
    print(f"\n  Existing gesture counts:")
    for g in gesture_dirs:
        ex = existing_counts.get(g, 0)
        print(f"    {g:15s}: {ex} existing")

    os.makedirs(config.DATA_DIR, exist_ok=True)

    detector  = HandDetector(mode=False, max_hands=1,
                             detection_con=0.4, track_con=0.4)
    extractor = FeatureExtractor()

    grand_total = 0
    results     = {}

    for gesture in gesture_dirs:
        gesture_dir = os.path.join(args.videos_dir, gesture)
        videos = [
            f for f in sorted(os.listdir(gesture_dir))
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        ]

        if not videos:
            print(f"\n  WARNING {gesture}: no video files found (checked {gesture_dir})")
            continue

        print(f"\n  -- {gesture} ({len(videos)} video(s)) --")
        gesture_total = 0

        for vid_file in videos:
            vid_path = os.path.join(gesture_dir, vid_file)
            n = extract_from_video(
                vid_path, gesture, args.output,
                detector, extractor,
                args.stride, args.preview, args.max_per_video
            )
            print(f"      {vid_file:<30s}  +{n} sequences")
            gesture_total += n

        existing = existing_counts.get(gesture, 0)
        print(f"    Total new: {gesture_total}  (total including old: {existing + gesture_total})")
        results[gesture]  = gesture_total
        grand_total      += gesture_total

    if args.preview:
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  Extraction Summary")
    print("=" * 60)
    for g in gesture_dirs:
        new = results.get(g, 0)
        old = existing_counts.get(g, 0)
        print(f"    {g:15s}  +{new:4d}  (total: {old + new})")
    print(f"\n  Total new sequences: {grand_total}")
    if grand_total > 0:
        print(f"  Saved to: {args.output}")
        print(f"\n  Next step: python gesture_trainer.py")
    else:
        print(f"\n  WARNING Nothing saved. Check that your videos contain visible hands.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
