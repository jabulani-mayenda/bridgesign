"""Quick diagnostic: tests camera -> hand detection -> features -> classifier."""
import cv2, time, sys
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import Classifier
import config

detector = HandDetector()
extractor = FeatureExtractor()
classifier = Classifier()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

print("Camera opened. Show a hand sign and hold it steady.")
print(f"Confidence threshold: {config.MIN_PREDICTION_CONFIDENCE}")
print()

for i in range(60):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: No frame captured")
        continue

    frame, results = detector.find_hands(frame, draw=False)
    hand_ok = results and results.hand_landmarks and len(results.hand_landmarks) > 0

    if hand_ok:
        lm_list = detector.get_landmarks(frame, hand_no=0)
        if lm_list:
            features = extractor.extract_features(lm_list)
            if features is not None:
                label, conf = classifier.predict(features)
                status = "PASS" if conf >= config.MIN_PREDICTION_CONFIDENCE else "BLOCKED"
                print(f"Frame {i:3d}: HAND -> {label:>2s} conf={conf:.3f}  [{status}]")
            else:
                print(f"Frame {i:3d}: HAND detected but extract_features returned None")
        else:
            print(f"Frame {i:3d}: HAND detected but get_landmarks returned empty")
    else:
        print(f"Frame {i:3d}: No hand detected")

    time.sleep(0.15)

cap.release()
print("\nDiagnostic complete.")
