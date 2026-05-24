import os
import cv2
import argparse
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
import config

def process_images_to_csv(images_dir, output_csv):
    # Set mode=True for IMAGE processing instead of VIDEO
    detector = HandDetector(mode=True, max_hands=1)
    extractor = FeatureExtractor()
    
    # Initialize output file
    with open(output_csv, 'w') as f:
        pass
        
    total_processed = 0
    total_failed = 0
    
    # Iterate through class folders
    for sign_label in sorted(os.listdir(images_dir)):
        sign_dir = os.path.join(images_dir, sign_label)
        if not os.path.isdir(sign_dir):
            continue
            
        count = 0
        failed = 0
        for img_name in os.listdir(sign_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(sign_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                failed += 1
                continue
                
            frame, _ = detector.find_hands(frame, draw=False)
            lm_list = detector.get_landmarks(frame, hand_no=0)
            
            if lm_list:
                features = extractor.extract_features(lm_list)
                with open(output_csv, 'a') as f:
                    f.write(f"{sign_label}," + ",".join(map(str, features)) + "\n")
                count += 1
                total_processed += 1
            else:
                failed += 1
                total_failed += 1
                
        print(f"[{sign_label}] Extracted {count} images (No hands found in {failed})")
        
    print(f"\nDone! Successfully extracted landmarks from {total_processed} images.")
    print(f"Failed to find hands in {total_failed} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process downloaded image dataset into CSV format for training.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the downloaded dataset folder (e.g., dataset/asl_alphabet_train)")
    args = parser.parse_args()
    
    out_csv = os.path.join(config.DATA_DIR, "dataset.csv")
    print(f"Extraction started...\n")
    print(f"Source Directory: {os.path.abspath(args.images_dir)}")
    print(f"Output CSV path: {os.path.abspath(out_csv)}\n")
    
    process_images_to_csv(args.images_dir, out_csv)
    print("\nNext step: Run 'python model_trainer.py' to train your model!")
