# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
BridgeSign - WLASL Smart Downloader
=====================================
Downloads ONLY the gesture signs we need from the WLASL dataset.
Uses direct MP4 URLs (handspeak.com, aslbricks.org, etc.) first —
no youtube-dl/yt-dlp needed for the majority of clips.

Signs downloaded:
  AGAIN, AMBULANCE, ANGRY, DANGER, DOCTOR, EMERGENCY, FIRE, FOOD

Output:
  gesture_videos/AGAIN/0.mp4, 1.mp4 ...
  gesture_videos/AMBULANCE/0.mp4 ...
  ... etc.

Then run:
  python video_gesture_collector.py
  python gesture_trainer.py
"""

import os
import json
import time
import sys
import urllib.request
import urllib.error
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)

# Signs to download — must match WLASL gloss names (lowercase)
TARGET_SIGNS = {
    # Previously live-recorded — now supplemented with professional WLASL videos
    "goodbye":   "GOODBYE",
    "hello":     "HELLO",
    "help":      "HELP",
    "no":        "NO",
    "please":    "PLEASE",
    "sorry":     "SORRY",
    "stop":      "STOP",
    "thank you": "THANK_YOU",   # WLASL gloss has a space
    "water":     "WATER",
    "yes":       "YES",
    "fire":      "FIRE",
    "j":         "J",
    # New signs added this session
    "again":     "AGAIN",
    "hospital":  "HOSPITAL",
    "angry":     "ANGRY",
    "danger":    "DANGER",
    "doctor":    "DOCTOR",
    "emergency": "EMERGENCY",
    "food":      "FOOD",
    # NOTE: 'ambulance' and 'z' not in WLASL — keep live-recorded data for those
}

OUTPUT_DIR = "gesture_videos"

# Max clips to download per sign (direct MP4 urls only — fast and reliable)
MAX_CLIPS_PER_SIGN = 8

# Sources that provide direct, reliable MP4 downloads
PREFERRED_SOURCES = {
    "handspeak",
    "aslbrick",
    "aslbricks",
    "startasl",
    "signingsavvy",
    "aslsearch",
    "asldeafined",
    "aslsignbank",
    "signschool",
}

YOUTUBE_SOURCES = {"youtube", "valencia-asl", "northtexas", "asllex", "lillybauer", "aslu"}

TIMEOUT = 20   # seconds per download attempt


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_json(url):
    print(f"  Fetching WLASL JSON from GitHub...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"  Loaded {len(data)} signs from WLASL.")
    return data


def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url


def is_direct_mp4(url):
    u = url.lower()
    return (
        u.endswith(".mp4") or u.endswith(".webm") or u.endswith(".mov")
    ) and not is_youtube_url(url)


def download_url(url, dest_path, timeout=TIMEOUT):
    """Download a direct URL to dest_path. Returns True on success."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120",
                "Referer": "https://www.google.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 5000:   # < 5 KB = probably an error page
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        return False


def try_yt_dlp(url, dest_path, frame_start, frame_end, fps=25):
    """Try to download a YouTube clip using yt-dlp if installed."""
    try:
        import subprocess
        # Check if yt-dlp is available
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return False
    except Exception:
        return False

    # Calculate time range (add 1s padding on each side)
    t_start = max(0, (frame_start - 1) / fps - 1.0)
    if frame_end > 0:
        t_end = (frame_end / fps) + 1.0
        section = f"*{t_start:.1f}-{t_end:.1f}"
    else:
        section = f"*{t_start:.1f}-inf"

    try:
        tmp_path = dest_path.replace(".mp4", "_yt_tmp.mp4")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "--download-sections", section,
            "-o", tmp_path,
            "--no-playlist",
            "--quiet",
            url,
        ]
        result = subprocess.run(cmd, timeout=120, capture_output=True)
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.rename(tmp_path, dest_path)
            return True
    except Exception:
        pass

    return False


def trim_video(src_path, dest_path, frame_start, frame_end, fps=25):
    """
    Trim video to [frame_start, frame_end] using OpenCV.
    Saves trimmed clip to dest_path.
    Returns True on success.
    """
    if frame_start <= 1 and frame_end == -1:
        # No trimming needed — just move/copy
        if src_path != dest_path:
            os.rename(src_path, dest_path)
        return True

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    f_start = max(0, frame_start - 1)
    f_end   = total_frames if frame_end == -1 else min(frame_end, total_frames)

    if f_start >= f_end:
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    out = cv2.VideoWriter(
        dest_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (w, h),
    )

    for _ in range(f_end - f_start):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    return size > 5000


def validate_video(path):
    """Check that the video is readable and has at least 15 frames."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count >= 15
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  BridgeSign — WLASL Gesture Downloader")
    print(f"  Downloading: {', '.join(TARGET_SIGNS.values())}")
    print("=" * 60)

    # 1. Fetch WLASL JSON
    try:
        wlasl = fetch_json(WLASL_JSON_URL)
    except Exception as e:
        print(f"\n  ERROR fetching WLASL JSON: {e}")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    # 2. Build lookup: gloss → instances
    target_lower = set(TARGET_SIGNS.keys())
    sign_data    = {}
    for entry in wlasl:
        gloss = entry["gloss"].lower()
        if gloss in target_lower:
            sign_data[gloss] = entry["instances"]

    missing = target_lower - set(sign_data.keys())
    if missing:
        print(f"\n  WARNING: These signs not found in WLASL: {missing}")

    print(f"\n  Found {len(sign_data)} signs in WLASL. Starting downloads...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}

    for gloss, label in TARGET_SIGNS.items():
        if gloss not in sign_data:
            print(f"  ✗ {label}: not in WLASL — skip")
            continue

        instances = sign_data[gloss]
        out_dir   = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        # Count already downloaded
        existing = len([f for f in os.listdir(out_dir) if f.endswith(".mp4")])
        needed   = MAX_CLIPS_PER_SIGN - existing

        if needed <= 0:
            print(f"  OK {label}: already has {existing} clips -- skip")
            summary[label] = existing
            continue

        print(f"  -- {label} ({len(instances)} instances, need {needed} more) --")

        # Sort: direct MP4 sources first, YouTube last
        direct   = [i for i in instances if is_direct_mp4(i["url"])]
        youtube  = [i for i in instances if is_youtube_url(i["url"])]
        ordered  = direct + youtube

        downloaded = existing
        clip_idx   = existing

        for inst in ordered:
            if downloaded - existing >= needed:
                break

            url         = inst["url"]
            frame_start = inst.get("frame_start", 1)
            frame_end   = inst.get("frame_end", -1)
            source      = inst.get("source", "unknown")
            is_yt       = is_youtube_url(url)

            clip_path     = os.path.join(out_dir, f"{clip_idx}.mp4")
            tmp_raw_path  = os.path.join(out_dir, f"_tmp_{clip_idx}_raw.mp4")

            if is_yt:
                print(f"    [{clip_idx}] YouTube ({source}) — trying yt-dlp...", end=" ")
                ok = try_yt_dlp(url, clip_path, frame_start, frame_end)
                if ok and validate_video(clip_path):
                    print("✓")
                    downloaded += 1
                    clip_idx   += 1
                else:
                    print("✗ skipped (yt-dlp unavailable or failed)")
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
            else:
                print(f"    [{clip_idx}] Direct MP4 ({source})...", end=" ")
                ok = download_url(url, tmp_raw_path)
                if not ok:
                    print("✗ download failed")
                    if os.path.exists(tmp_raw_path):
                        os.remove(tmp_raw_path)
                    continue

                # Trim to exact sign frames
                ok2 = trim_video(tmp_raw_path, clip_path, frame_start, frame_end)
                if os.path.exists(tmp_raw_path) and tmp_raw_path != clip_path:
                    try:
                        os.remove(tmp_raw_path)
                    except Exception:
                        pass

                if ok2 and validate_video(clip_path):
                    size_kb = os.path.getsize(clip_path) // 1024
                    print(f"✓ ({size_kb} KB, frames {frame_start}→{frame_end})")
                    downloaded += 1
                    clip_idx   += 1
                else:
                    print("✗ invalid/too short after trim")
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

            time.sleep(0.3)   # polite delay

        new = downloaded - existing
        summary[label] = downloaded
        print(f"    → {new} new clips downloaded  (total: {downloaded})\n")

    # Summary
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    total_clips = 0
    for label in TARGET_SIGNS.values():
        n = summary.get(label, 0)
        total_clips += n
        status = "✓" if n > 0 else "✗"
        print(f"    {status} {label:15s}: {n} clips")

    print(f"\n  Total clips ready: {total_clips}")
    print(f"  Videos saved to  : {os.path.abspath(OUTPUT_DIR)}/")

    if total_clips > 0:
        print(f"\n  Next steps:")
        print(f"    1. Record STATIC yourself (hold letter signs still):")
        print(f"       Put video in gesture_videos/STATIC/take1.mp4")
        print(f"    2. python video_gesture_collector.py")
        print(f"    3. python gesture_trainer.py")
    else:
        print(f"\n  No clips downloaded. Check internet connection.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
