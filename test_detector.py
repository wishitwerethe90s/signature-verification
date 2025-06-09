import cv2
import os
from src.detection.detector import detect_signatures

# --- Configuration ---
MODEL_PATH = 'models/detection_yolov5.pt'
# IMPORTANT: Change this to the name of your test image file
TEST_IMAGE_PATH = 'data/raw_documents/2.png' 
OUTPUT_DIR = 'output/detection_results'

# --- Main execution ---
if __name__ == "__main__":
    print("--- Starting Signature Detection Test ---")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if the test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Test image not found at '{TEST_IMAGE_PATH}'")
        print("Please place a test image with that name and path, or update the TEST_IMAGE_PATH variable.")
    else:
        # Run the detection function
        cropped_sigs = detect_signatures(image_path=TEST_IMAGE_PATH, model_path=MODEL_PATH)

        if cropped_sigs:
            print(f"\nSuccessfully detected {len(cropped_sigs)} signature(s).")
            # Save the cropped signatures to the output directory
            for i, sig_img in enumerate(cropped_sigs):
                save_path = os.path.join(OUTPUT_DIR, f"cropped_signature_{i+1}.jpg")
                cv2.imwrite(save_path, sig_img)
                print(f"Saved cropped signature to '{save_path}'")
        else:
            print("\nNo signatures were detected in the image.")

    print("\n--- Detection Test Finished ---")