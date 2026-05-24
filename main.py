import cv2
import time
import argparse
import config
from camera import Camera
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import Classifier
from text_to_speech import TextToSpeech
from word_assembler import WordAssembler

def parse_args():
    parser = argparse.ArgumentParser(description="BridgeSign - Sign Language Translator")
    parser.add_argument("--webcam", type=int, default=config.CAMERA_INDEX, help="Webcam index")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[{config.APP_NAME} v{config.APP_VERSION}] Initialising components...")

    detector          = HandDetector()
    feature_extractor = FeatureExtractor()
    classifier        = Classifier()
    tts               = TextToSpeech()
    assembler         = WordAssembler()

    prev_label   = ""
    consecutive  = 0
    last_hand_ts = 0.0
    last_emitted_label = ""

    print("Starting video stream — sign letters to build words, pause to speak them.")
    try:
        with Camera(index=args.webcam) as cam:
            while True:
                ret, frame = cam.get_frame()
                if not ret:
                    break

                frame, results = detector.find_hands(frame, draw=True)
                lm_list        = detector.get_landmarks(frame, hand_no=0)
                hand_present   = bool(lm_list)
                label          = ""
                confidence     = 0.0

                if hand_present:
                    features        = feature_extractor.extract_features(lm_list)
                    label, confidence = classifier.predict(features)

                    if confidence < config.MIN_PREDICTION_CONFIDENCE or label in ("", "Unknown", "Error"):
                        label = ""
                        confidence = 0.0
                        consecutive = 0
                        prev_label = ""
                    else:
                        if label == prev_label:
                            consecutive += 1
                        else:
                            consecutive = 1
                            prev_label  = label

                        # Stable letter confirmed — push to word assembler (no TTS yet)
                        if consecutive >= config.CONSECUTIVE_THRESHOLD:
                            if label != last_emitted_label:
                                assembler.push_letter(label)
                                last_emitted_label = label
                            consecutive = 0
                else:
                    consecutive = 0
                    prev_label = ""
                    last_emitted_label = ""

                # Grace period for hand-switch tolerance
                now = time.time()
                if hand_present:
                    last_hand_ts = now
                grace_present = hand_present or (now - last_hand_ts) < config.HAND_LOST_GRACE_SEC

                state = assembler.tick(grace_present)

                # ── Word completed → speak the whole word ──────────────────
                if state["completed_word"]:
                    word = state["completed_word"]
                    print(f"[Word] {word}")
                    tts.speak_async(word)

                # ── Sentence completed → speak the full sentence ────────────
                if state["completed_sentence"]:
                    sentence = state["completed_sentence"]
                    print(f"[Sentence] {sentence}")
                    if len(sentence.split()) > 1:
                        tts.speak_async(sentence)

                # ── OpenCV display ─────────────────────────────────────
                cv2.rectangle(frame, (0, 0), (config.FRAME_WIDTH, 80), (15, 15, 15), cv2.FILLED)

                # Current letter being signed
                cv2.putText(frame, f"Letter : {label or '...'}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 40), 2)

                # Buffer (letters being assembled)
                buf = state["word_buffer"]
                cv2.putText(frame, f"Buffer : {buf or '-'}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

                # Last completed word
                if state["last_word"]:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 82), (config.FRAME_WIDTH, 130), (20, 20, 20), cv2.FILLED)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                    cv2.putText(frame, f"Word   : {state['last_word']}",
                                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 100), 2)

                # Sentence
                if state["sentence"]:
                    overlay2 = frame.copy()
                    cv2.rectangle(overlay2, (0, 132), (config.FRAME_WIDTH, 172), (10, 10, 10), cv2.FILLED)
                    frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)
                    cv2.putText(frame, f"Sentence: {state['sentence']}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 120, 255), 2)

                cv2.imshow(f"{config.APP_NAME} — Sign to speak", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
