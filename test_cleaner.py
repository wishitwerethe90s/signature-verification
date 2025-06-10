import cv2
import os
# The function name is now different to reflect it's a wrapper
from src.cleaning.cleaner import clean_signature

# --- Configuration ---
# This path is passed to our function but the underlying script has its own path logic.
# We keep it for consistency.
MODEL_PATH = 'models/latest_net_G.pth'
# Use one of the outputs from the detection test
INPUT_IMAGE_PATH = 'output/detection_results/cropped_signature_1.jpg'
OUTPUT_DIR = 'output/cleaning_results'

# --- Main execution ---
if __name__ == "__main__":
    # ... (The rest of the test script can remain the same as the last version) ...
    print("--- Starting Signature Cleaning Test ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'")
    else:
        cropped_sig_img = cv2.imread(INPUT_IMAGE_PATH)
        
        if cropped_sig_img is not None:
            cleaned_img = clean_signature(signature_image=cropped_sig_img, model_path=MODEL_PATH)
            
            if cleaned_img is not None:
                output_filename = os.path.basename(INPUT_IMAGE_PATH).replace('cropped', 'cleaned')
                save_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(save_path, cleaned_img)
                print(f"Successfully cleaned signature and saved to '{save_path}'")
            else:
                print("Cleaning function failed to produce an image.")
        else:
            print(f"ERROR: Could not read the input image at {INPUT_IMAGE_PATH}")

    print("\n--- Cleaning Test Finished ---")