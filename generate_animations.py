"""
BridgeSign -- Avatar Animation Generator (VRM edition)
======================================================
Builds lightweight animation clips from:

  1. data/dataset.csv          -> static alphabet poses (A-Z)
  2. data/gesture_dataset.csv  -> motion clips (J, Z, HELLO, HELP, ...)

Output format: JSON animation clips with VRM humanoid bone names.
These are loaded by the AvatarController and applied to VRM bone nodes.
"""

import csv
import json
from pathlib import Path

import numpy as np

STATIC_DATA_FILE = Path("data/dataset.csv")
GESTURE_DATA_FILE = Path("data/gesture_dataset.csv")
OUT_DIR = Path("static/avatar/animations")

STATIC_FEATURE_COUNT = 42
GESTURE_WINDOW = 15

# MediaPipe landmark indices (21-point model)
FINGER_LANDMARKS = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20],
}
WRIST_IDX = 0

# Map finger names to VRM humanoid bone names
VRM_BONE_MAP = {
    ("Thumb",  0): "rightThumbMetacarpal",
    ("Thumb",  1): "rightThumbProximal",
    ("Thumb",  2): "rightThumbDistal",
    ("Index",  0): "rightIndexProximal",
    ("Index",  1): "rightIndexIntermediate",
    ("Index",  2): "rightIndexDistal",
    ("Middle", 0): "rightMiddleProximal",
    ("Middle", 1): "rightMiddleIntermediate",
    ("Middle", 2): "rightMiddleDistal",
    ("Ring",   0): "rightRingProximal",
    ("Ring",   1): "rightRingIntermediate",
    ("Ring",   2): "rightRingDistal",
    ("Pinky",  0): "rightLittleProximal",
    ("Pinky",  1): "rightLittleIntermediate",
    ("Pinky",  2): "rightLittleDistal",
}


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (radians) to quaternion [x, y, z, w]."""
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [float(qx), float(qy), float(qz), float(qw)]


def _angle_between(v1, v2):
    """Angle in radians between two 2D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _compute_finger_curl(pts, landmarks):
    """
    Compute curl angles for each joint of a finger from landmark chain.
    Returns [curl1, curl2, curl3] in radians.
    """
    mcp_idx, pip_idx, dip_idx, tip_idx = landmarks
    wrist = pts[WRIST_IDX]
    mcp, pip, dip, tip = pts[mcp_idx], pts[pip_idx], pts[dip_idx], pts[tip_idx]

    v_wrist_mcp = mcp - wrist
    v_mcp_pip = pip - mcp
    v_pip_dip = dip - pip
    v_dip_tip = tip - dip

    return [
        _angle_between(v_wrist_mcp, v_mcp_pip),
        _angle_between(v_mcp_pip, v_pip_dip),
        _angle_between(v_pip_dip, v_dip_tip),
    ]


def _compute_finger_spread(pts, finger_name):
    """Compute lateral spread (abduction) of a finger."""
    landmarks = FINGER_LANDMARKS[finger_name]
    mcp, tip = pts[landmarks[0]], pts[landmarks[3]]
    wrist = pts[WRIST_IDX]
    base_dir = mcp - wrist
    finger_dir = tip - mcp
    if np.linalg.norm(base_dir) < 1e-8 or np.linalg.norm(finger_dir) < 1e-8:
        return 0.0
    cross_z = float(base_dir[0] * finger_dir[1] - base_dir[1] * finger_dir[0])
    return float(np.clip(cross_z * 2.0, -0.5, 0.5))


def pose_tracks_from_features(features):
    """
    Convert a 42-value hand sample into VRM bone quaternion targets.
    """
    pts = np.array([(features[i * 2], features[i * 2 + 1]) for i in range(21)],
                   dtype=np.float64)
    tracks = {}

    for finger_name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        landmarks = FINGER_LANDMARKS[finger_name]
        curls = _compute_finger_curl(pts, landmarks)
        spread = _compute_finger_spread(pts, finger_name)

        for j in range(3):
            vrm_bone = VRM_BONE_MAP[(finger_name, j)]
            raw_angle = curls[j]

            if finger_name == "Thumb":
                if j == 0:
                    curl_val = float(np.clip(raw_angle * 0.8, 0.0, 1.2))
                    spread_val = spread * 0.6
                    quat = euler_to_quaternion(curl_val, spread_val * 0.5, spread_val)
                else:
                    curl_val = float(np.clip(raw_angle * 0.9, 0.0, 1.4))
                    quat = euler_to_quaternion(curl_val, 0.0, 0.0)
            else:
                curl_val = float(np.clip(raw_angle * 1.0, 0.0, 1.6))
                spread_val = spread * 0.3 if j == 0 else 0.0
                quat = euler_to_quaternion(curl_val, 0.0, spread_val)

            tracks[f"{vrm_bone}.quaternion"] = quat

    # Wrist rotation
    wrist = pts[WRIST_IDX]
    mid_mcp, idx_mcp = pts[9], pts[5]
    palm_dir = mid_mcp - wrist
    if np.linalg.norm(palm_dir) > 1e-8:
        wrist_pitch = float(np.arctan2(palm_dir[0], -palm_dir[1])) * 0.4
        lateral = idx_mcp - mid_mcp
        wrist_yaw = float(np.arctan2(lateral[1], lateral[0])) * 0.3
    else:
        wrist_pitch, wrist_yaw = 0.0, 0.0

    tracks["rightHand.quaternion"] = euler_to_quaternion(0.0, wrist_pitch, wrist_yaw)

    return tracks


def _rest_pose(target_pose):
    """Generate a neutral rest pose (all identity quaternions)."""
    return {key: [0.0, 0.0, 0.0, 1.0] for key in target_pose}


def build_clip(label, pose_series, duration):
    """Create a JSON animation clip from a list of poses."""
    if not pose_series:
        return None

    if len(pose_series) == 1:
        # Static: transition from rest to target
        times = [0.0, duration * 0.15, duration]
        rest = _rest_pose(pose_series[0])
        pose_series = [rest, pose_series[0], pose_series[0]]
    else:
        times = np.linspace(0.0, duration, num=len(pose_series),
                            dtype=np.float32).tolist()

    tracks = []
    for track_name in sorted(pose_series[0].keys()):
        values = []
        for pose in pose_series:
            values.extend(pose[track_name])
        tracks.append({
            "name": track_name,
            "type": "quaternion",
            "times": times,
            "values": values,
        })

    return {"name": label, "duration": duration, "tracks": tracks}


def iter_static_samples(path, count=5):
    """Yield averaged samples per letter for stable poses."""
    samples = {}
    if not path.exists():
        return
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < STATIC_FEATURE_COUNT + 1:
                continue
            label = row[0]
            if label not in samples:
                samples[label] = []
            if len(samples[label]) >= count:
                continue
            samples[label].append(
                np.array([float(x) for x in row[1:1 + STATIC_FEATURE_COUNT]],
                         dtype=np.float64))

    for label, feat_list in samples.items():
        yield label, np.mean(feat_list, axis=0)


def iter_gesture_sequences(path, count=5):
    """Yield the most dynamic gesture sequence per label."""
    expected_values = GESTURE_WINDOW * STATIC_FEATURE_COUNT
    samples = {}
    if not path.exists():
        return
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < expected_values + 1:
                continue
            label = row[0]
            if label == "STATIC":
                continue
            if label not in samples:
                samples[label] = []
            if len(samples[label]) >= count:
                continue
            values = [float(x) for x in row[1:1 + expected_values]]
            samples[label].append(
                np.array(values, dtype=np.float64).reshape(
                    GESTURE_WINDOW, STATIC_FEATURE_COUNT))

    for label, seqs in samples.items():
        best = max(seqs, key=lambda s: float(np.sum(np.abs(np.diff(s, axis=0)))))
        yield label, best


def save_clip(label, clip):
    out_path = OUT_DIR / f"{label}.json"
    with out_path.open("w") as f:
        json.dump(clip, f, indent=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating VRM avatar animations...")
    print(f"  Static dataset : {STATIC_DATA_FILE}")
    print(f"  Gesture dataset: {GESTURE_DATA_FILE}")
    print(f"  Output dir     : {OUT_DIR}\n")

    static_count = 0
    for label, features in iter_static_samples(STATIC_DATA_FILE, count=5):
        pose = pose_tracks_from_features(features)
        clip = build_clip(label, [pose], duration=1.0)
        if clip:
            save_clip(label, clip)
            static_count += 1
            non_id = sum(1 for v in pose.values()
                         if not (abs(v[0]) < 0.001 and abs(v[1]) < 0.001
                                 and abs(v[2]) < 0.001 and abs(v[3] - 1.0) < 0.001))
            print(f"    {label}: {non_id}/{len(pose)} tracks have movement")

    gesture_count = 0
    for label, sequence in iter_gesture_sequences(GESTURE_DATA_FILE, count=5):
        poses = [pose_tracks_from_features(frame) for frame in sequence]
        clip = build_clip(label, poses, duration=1.2)
        if clip:
            save_clip(label, clip)
            gesture_count += 1
            first_vals = list(poses[0].values())
            last_vals = list(poses[-1].values())
            delta = sum(abs(a - b) for fv, lv in zip(first_vals, last_vals)
                        for a, b in zip(fv, lv))
            print(f"    {label}: frame delta = {delta:.3f}")

    total = len(list(OUT_DIR.glob("*.json")))
    print(f"\n  Static clips  : {static_count}")
    print(f"  Gesture clips : {gesture_count}")
    print(f"  Total files   : {total}")
    print("\nDone.")


if __name__ == "__main__":
    main()
