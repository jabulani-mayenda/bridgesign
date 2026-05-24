"""Quick diagnostic: tests whether the sign_model.pkl can actually predict."""
import pickle, json, os, sys
import numpy as np

MODEL_PATH = "models/sign_model.pkl"
CSV_PATH   = "data/dataset.csv"

# ── 1. Load model ────────────────────────────────────────────────────────────
print("\n=== Model check ===")
if not os.path.exists(MODEL_PATH):
    print(f"MISSING: {MODEL_PATH}")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

pipeline = data["pipeline"]
classes  = data["classes"]
n_feat   = getattr(pipeline, "n_features_in_", "unknown")
print(f"  classes ({len(classes)}): {classes}")
print(f"  n_features_in_: {n_feat}")

# ── 2. Test with hardcoded A sample ─────────────────────────────────────────
A_SAMPLE = [0.0,0.0,0.44396728,-0.13154586,0.78105354,-0.5508483,0.88793457,
            -0.9701507,1.0441452,-1.2496856,0.6495077,-0.88793457,0.6988374,
            -1.1756911,0.600178,-0.8714913,0.5508483,-0.6659509,0.36997274,
            -0.92904264,0.38641596,-1.1921344,0.32064304,-0.7974968,0.3124214,
            -0.5919564,0.09043778,-0.93726426,0.098659396,-1.2085776,0.082216166,
            -0.78927517,0.106881015,-0.5755131,-0.18909718,-0.9125994,-0.16443233,
            -1.134583,-0.13154586,-0.83038324,-0.098659396,-0.6577293]

print(f"\n  A sample has {len(A_SAMPLE)} features")
try:
    proba   = pipeline.predict_proba([A_SAMPLE])[0]
    idx     = int(np.argmax(proba))
    print(f"  A test => predicted='{classes[idx]}', conf={proba[idx]:.3f}")
except Exception as e:
    print(f"  PREDICT ERROR: {e}")

# ── 3. Check config.MIN_PREDICTION_CONFIDENCE ────────────────────────────────
print("\n=== Config check ===")
import config
print(f"  MIN_PREDICTION_CONFIDENCE: {config.MIN_PREDICTION_CONFIDENCE}")
print(f"  CONSECUTIVE_THRESHOLD:     {config.CONSECUTIVE_THRESHOLD}")
print(f"  HAND_LOST_GRACE_SEC:       {config.HAND_LOST_GRACE_SEC}")

# ── 4. Check dataset.csv ──────────────────────────────────────────────────────
print("\n=== Dataset CSV check ===")
if not os.path.exists(CSV_PATH):
    print(f"  MISSING: {CSV_PATH}")
else:
    with open(CSV_PATH) as f:
        rows = f.readlines()
    labels = sorted(set(r.split(",")[0] for r in rows if r.strip()))
    feat_len = len(rows[0].split(",")) - 1 if rows else 0
    print(f"  Rows: {len(rows)}")
    print(f"  Feature length per row: {feat_len}")
    print(f"  Classes in CSV: {labels}")
    # Per-class counts
    from collections import Counter
    counts = Counter(r.split(",")[0] for r in rows if r.strip())
    print("  Per-class counts:")
    for lbl in sorted(counts):
        print(f"    {lbl}: {counts[lbl]}")

print("\n=== Done ===")
