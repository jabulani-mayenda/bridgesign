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
        """
        img = cv2.imread(img_path)
        if img is None:
            return "Error: Image not found", 0.0, None

        # Create a copy so we don't modify original read
        result_img = img.copy()
        
        # Detect hands in static image mode
        result_img, _ = self.detector.find_hands(result_img, draw=True)
        lm_list = self.detector.get_landmarks(result_img, hand_no=0)
        
        if lm_list:
            features = self.feature_extractor.extract_features(lm_list)
            label, confidence = self.classifier.predict(features)
            
            # Put translated text on image
            cv2.putText(result_img, f'{label} ({confidence:.2f})', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        config.COLOR_PRIMARY, 2)
            return label, confidence, result_img
            
        return "No hand detected", 0.0, result_img

    def translate_from_array(self, img_array):
        """
        Translates directly from a cv2 numpy array (e.g. pasted image).
        """
        result_img = img_array.copy()
        result_img, _ = self.detector.find_hands(result_img, draw=True)
        lm_list = self.detector.get_landmarks(result_img, hand_no=0)
        
        if lm_list:
            features = self.feature_extractor.extract_features(lm_list)
            label, confidence = self.classifier.predict(features)
            return label, confidence, result_img
            
        return "No hand detected", 0.0, result_img
