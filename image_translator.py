import cv2
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import Classifier

class ImageTranslator:
    def __init__(self):
        self.detector = HandDetector(mode=True) # static mode for single images
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()

    def translate(self, img_path):
        """
        Loads an image from path, detects hands, extracts features, 
        and returns the classification prediction.
        
        To handle both standard direct photos and mirrored selfies, we run
        predictions on both the original image and a horizontally flipped copy,
        and select the one with the higher confidence score.
        """
        img = cv2.imread(img_path)
        if img is None:
            return "Error: Image not found", 0.0, None

        # 1. Predict on original image
        orig_img = img.copy()
        orig_img, orig_results = self.detector.find_hands(orig_img, draw=True)
        orig_lm = self.detector.get_landmarks(orig_img, hand_no=0)
        
        orig_label, orig_conf = "No hand detected", 0.0
        if orig_lm:
            orig_feats = self.feature_extractor.extract_features(orig_lm)
            orig_label, orig_conf = self.classifier.predict(orig_feats)

        # 2. Predict on horizontally flipped image
        flipped_img = cv2.flip(img, 1)
        flipped_img, flipped_results = self.detector.find_hands(flipped_img, draw=True)
        flipped_lm = self.detector.get_landmarks(flipped_img, hand_no=0)
        
        flipped_label, flipped_conf = "No hand detected", 0.0
        if flipped_lm:
            flipped_feats = self.feature_extractor.extract_features(flipped_lm)
            flipped_label, flipped_conf = self.classifier.predict(flipped_feats)

        # 3. Compare confidences and select the best result
        if flipped_conf > orig_conf:
            print(f"[ImageTranslator] Flipped image matched better: {flipped_label} ({flipped_conf:.2%}) vs {orig_label} ({orig_conf:.2%})")
            label, confidence, result_img = flipped_label, flipped_conf, flipped_img
        else:
            if orig_lm:
                print(f"[ImageTranslator] Original image matched better: {orig_label} ({orig_conf:.2%}) vs {flipped_label} ({flipped_conf:.2%})")
            label, confidence, result_img = orig_label, orig_conf, orig_img

        if label != "No hand detected":
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
        # 1. Original
        orig_img = img_array.copy()
        orig_img, _ = self.detector.find_hands(orig_img, draw=True)
        orig_lm = self.detector.get_landmarks(orig_img, hand_no=0)
        orig_label, orig_conf = "No hand detected", 0.0
        if orig_lm:
            orig_feats = self.feature_extractor.extract_features(orig_lm)
            orig_label, orig_conf = self.classifier.predict(orig_feats)

        # 2. Flipped
        flipped_img = cv2.flip(img_array, 1)
        flipped_img, _ = self.detector.find_hands(flipped_img, draw=True)
        flipped_lm = self.detector.get_landmarks(flipped_img, hand_no=0)
        flipped_label, flipped_conf = "No hand detected", 0.0
        if flipped_lm:
            flipped_feats = self.feature_extractor.extract_features(flipped_lm)
            flipped_label, flipped_conf = self.classifier.predict(flipped_feats)

        # 3. Best selection
        if flipped_conf > orig_conf:
            return flipped_label, flipped_conf, flipped_img
        return orig_label, orig_conf, orig_img
