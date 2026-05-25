import config

class LearningMode:
    def __init__(self):
        self.detector = None
        # Predefined correct landmark states for basic signs (concept)
        # In a real app this would load recorded templates
        self.library = {
            "Hello": "Raise hand flat, thumb out, swipe right",
            "Thank you": "Hand flat at chin, move down and forward",
            "Yes": "Make a fist, bob it up and down like nodding",
            "No": "Pointer, middle, thumb extended, snap them together"
        }

    def get_lesson(self, sign):
        return self.library.get(sign, "Lesson not found.")

    def _get_detector(self):
        if self.detector is None:
            from hand_detector import HandDetector
            self.detector = HandDetector()
        return self.detector

    def evaluate_sign(self, frame, target_sign):
        """
        Process the frame and evaluate if the user is making 
        the target sign correctly. 
        Returns (Frame, Feedback String)
        """
        import cv2

        result_img = frame.copy()
        detector = self._get_detector()
        result_img, _ = detector.find_hands(result_img, draw=True)
        lm_list = detector.get_landmarks(result_img)
        
        if not lm_list:
            return result_img, "No hand detected. Raise your hand."
            
        # Placeholder evaluation logic
        # Here we would compare lm_list to target_sign templates
        feedback = f"Attempting '{target_sign}': Keep practicing!"
        
        cv2.putText(result_img, feedback, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, config.COLOR_SECONDARY, 2)
                    
        return result_img, feedback
