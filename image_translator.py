import cv2
import numpy as np
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import Classifier

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

MIN_HAND_BBOX_AREA_RATIO = 0.004
MAX_HAND_BBOX_ASPECT = 4.0

class ImageTranslator:
    def __init__(self):
        self.detector = HandDetector(mode=True) # static mode for single images
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()

    def _resize_for_detection(self, img):
        max_dim = int(getattr(config, "IMAGE_MAX_DIM", 768))
        if img is None or max_dim <= 0:
            return img
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return img
        scale = max_dim / float(longest)
        return cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _read_image(self, img_path):
        """
        Read an uploaded photo in the same orientation the user sees.

        Phone photos often carry EXIF orientation instead of physically rotating
        pixels. OpenCV can miss that in some environments, which makes MediaPipe
        see a sideways hand and fail detection.
        """
        if Image is not None and ImageOps is not None:
            try:
                with Image.open(img_path) as pil_img:
                    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                    return self._resize_for_detection(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[ImageTranslator] PIL image load failed, falling back to OpenCV: {e}")
        return self._resize_for_detection(cv2.imread(img_path))

    def _predict_landmarks(self, lm_list):
        features = self.feature_extractor.extract_features(lm_list)
        if features is None:
            return "No hand detected", 0.0, "none"

        if hasattr(self.classifier, "predict_many"):
            (raw_label, raw_conf), (flip_label, flip_conf) = self.classifier.predict_many([
                features,
                FeatureExtractor.flip_x(features),
            ])
        else:
            raw_label, raw_conf = self.classifier.predict(features)
            flip_label, flip_conf = self.classifier.predict(FeatureExtractor.flip_x(features))

        if flip_conf > raw_conf:
            return flip_label, flip_conf, "feature_mirror"
        return raw_label, raw_conf, "raw"

    def _predict_image(self, img):
        result_img = img.copy()
        result_img, results = self.detector.find_hands(result_img, draw=False)
        if not self._has_plausible_hand(results, result_img):
            return "No hand detected", 0.0, "none", result_img
        lm_list = self.detector.get_landmarks(result_img, hand_no=0)
        label, confidence, orientation = self._predict_landmarks(lm_list)
        return label, confidence, orientation, result_img

    @staticmethod
    def _has_plausible_hand(results, img):
        if (
            results is None
            or not getattr(results, "hand_landmarks", None)
            or len(results.hand_landmarks) == 0
        ):
            return False

        h, w = img.shape[:2]
        if w < 1 or h < 1:
            return False

        hand_landmarks = results.hand_landmarks[0]
        xs = [float(lm.x) for lm in hand_landmarks]
        ys = [float(lm.y) for lm in hand_landmarks]
        if not xs or not ys:
            return False

        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        if bbox_w * bbox_h < MIN_HAND_BBOX_AREA_RATIO:
            return False

        if bbox_w > 0 and bbox_h > 0:
            aspect = max(bbox_w / bbox_h, bbox_h / bbox_w)
            if aspect > MAX_HAND_BBOX_ASPECT:
                return False

        return True

    def translate(self, img_path):
        """
        Loads an image from path, detects hands, extracts features, 
        and returns the classification prediction.
        
        To handle both standard direct photos and mirrored selfies, we run
        predictions on both the original image and a horizontally flipped copy,
        and select the one with the higher confidence score.
        """
        img = self._read_image(img_path)
        if img is None:
            return "Error: Image not found", 0.0, None

        # 1. Predict from the detected landmarks, checking raw + mirrored
        # feature vectors. This matches the live camera path.
        orig_label, orig_conf, orig_orientation, orig_img = self._predict_image(img)

        # Skip flipped pass entirely if we have a high-confidence prediction
        if orig_label != "No hand detected" and orig_conf >= 0.70:
            print(
                f"[ImageTranslator] High confidence on original image: {orig_label} ({orig_conf:.2%}, {orig_orientation}). "
                "Skipping pixel-flipped pass."
            )
            label, confidence, result_img = orig_label, orig_conf, orig_img
        elif orig_label != "No hand detected" and orig_conf >= 0.50:
            # Medium confidence: Try a pixel-flipped image to see if it improves
            flipped_img = cv2.flip(img, 1)
            flipped_label, flipped_conf, flipped_orientation, flipped_img_res = self._predict_image(flipped_img)

            # Compare confidences and select the best result
            if flipped_conf > orig_conf:
                print(
                    "[ImageTranslator] Pixel-flipped image matched better: "
                    f"{flipped_label} ({flipped_conf:.2%}, {flipped_orientation}) vs "
                    f"{orig_label} ({orig_conf:.2%}, {orig_orientation})"
                )
                label, confidence, result_img = flipped_label, flipped_conf, flipped_img_res
            else:
                print(
                    "[ImageTranslator] Original image matched better: "
                    f"{orig_label} ({orig_conf:.2%}, {orig_orientation}) vs "
                    f"{flipped_label} ({flipped_conf:.2%}, {flipped_orientation})"
                )
                label, confidence, result_img = orig_label, orig_conf, orig_img
        else:
            # Low confidence or no hand detected: Skip flipped pass entirely
            if orig_label != "No hand detected":
                print(
                    f"[ImageTranslator] Low confidence ({orig_conf:.2%}) on original image: {orig_label}. "
                    "Skipping pixel-flipped pass."
                )
            label, confidence, result_img = orig_label, orig_conf, orig_img

        if label != "No hand detected" and confidence >= config.MIN_PREDICTION_CONFIDENCE:
            # Put translated text on the chosen output image
            cv2.putText(result_img, f'{label} ({confidence:.2%})', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        config.COLOR_PRIMARY, 2)
            
        return label, confidence, result_img

    def translate_from_array(self, img_array):
        """
        Translates directly from a cv2 numpy array (e.g. pasted image).
        Uses dual-orientation prediction to handle mirrored/unmirrored states.
        """
        img_array = self._resize_for_detection(img_array)
        orig_label, orig_conf, _, orig_img = self._predict_image(img_array)
        flipped_img = cv2.flip(img_array, 1)
        flipped_label, flipped_conf, _, flipped_img = self._predict_image(flipped_img)

        if flipped_conf > orig_conf:
            return flipped_label, flipped_conf, flipped_img
        return orig_label, orig_conf, orig_img
