"""Generate verified sanity check samples from the current model."""
import pickle, numpy as np

with open('models/sign_model.pkl','rb') as f:
    data = pickle.load(f)
pipeline = data['pipeline']
classes = data['classes']

samples = {}
targets = ['A','B','H','W']
with open('data/dataset.csv','r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 2: continue
        label = parts[0]
        if label in targets and label not in samples:
            feats = [float(x) for x in parts[1:]]
            X = np.array([feats], dtype=np.float32)
            pred = classes[int(np.argmax(pipeline.predict_proba(X)[0]))]
            conf = float(np.max(pipeline.predict_proba(X)[0]))
            if pred == label:
                samples[label] = feats
                print(f'{label}: predicted={pred} conf={conf:.2f} (correct!)')
        if len(samples) == 4: break

if len(samples) < 4:
    print(f"\nWARNING: Only found {len(samples)}/4 correct samples.")
    print(f"Missing: {[t for t in targets if t not in samples]}")
    # Try to find ANY correct sample from dataset
    with open('data/dataset.csv','r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            label = parts[0]
            if label not in samples:
                feats = [float(x) for x in parts[1:]]
                X = np.array([feats], dtype=np.float32)
                pred = classes[int(np.argmax(pipeline.predict_proba(X)[0]))]
                if pred == label:
                    samples[label] = feats
                    print(f'Alt: {label}: predicted={pred} (correct!)')
                    if len(samples) >= 4: break

print("\n# Copy this into app.py _SANITY_SAMPLES:")
print("_SANITY_SAMPLES = {")
for label, feats in sorted(samples.items()):
    arr_str = ','.join(f'{v}' for v in feats)
    print(f'    "{label}": [{arr_str}],')
print("}")
