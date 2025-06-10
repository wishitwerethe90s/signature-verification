from src.verification.verifier import get_feature_vector, get_verification_score
import os
import numpy as np

# --- Configuration ---
# NOTE: The weights path in main.py was 'weights/signet/signet.h5'.
# Ensure you have this file and place it in 'models/verification_siamese.h5'
# or update the path here.
MODEL_WEIGHTS = 'models/verification_siamese.h5' 

# NOTE: The main.py used specific data paths.
# Please ensure you have these images or replace them with your own test images.
# A genuine-genuine pair
GENUINE_1 = '/home/paarthgupta/hdfc/innovation/signature/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/data/01_050.png'
GENUINE_2 = '/home/paarthgupta/hdfc/innovation/signature/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/data/02_050.png'
# A genuine-forged pair
FORGED = '/home/paarthgupta/hdfc/innovation/signature/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/data/02_067.png'

# --- Main execution ---
if __name__ == "__main__":
    print("--- Starting Signature Verification Test ---")

    # Check if model and data files exist
    if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(GENUINE_1):
        print("\nERROR: Model weights or test data not found.")
        print(f"Please make sure '{MODEL_WEIGHTS}', '{GENUINE_1}', etc., exist.")
    else:
        # --- Test 1: Feature Extraction ---
        print("\n--- Testing Feature Extraction (1:N) ---")
        try:
            feature_vec = get_feature_vector(GENUINE_1, MODEL_WEIGHTS)
            print(f"Successfully extracted feature vector for '{GENUINE_1}'.")
            print(f"Vector Shape: {feature_vec.shape}")
            print(f"Vector Preview: {feature_vec[0, :5]}...")
        except Exception as e:
            print(f"An error occurred during feature extraction: {e}")

        # --- Test 2: Verification ---
        print("\n--- Testing Verification (1:1) ---")
        try:
            # Test a genuine-genuine pair (should have a low score)
            genuine_score = get_verification_score(GENUINE_1, GENUINE_2, MODEL_WEIGHTS)
            print(f"Verification score for two genuine signatures: {genuine_score:.4f}")
            if genuine_score <= 0.5:
                print("Result: Correctly identified as 'Genuine'")
            else:
                print("Result: Incorrectly identified as 'Forged'")

            # Test a genuine-forged pair (should have a high score)
            forged_score = get_verification_score(GENUINE_1, FORGED, MODEL_WEIGHTS)
            print(f"\nVerification score for a genuine vs. forged signature: {forged_score:.4f}")
            if forged_score > 0.5:
                print("Result: Correctly identified as 'Forged'")
            else:
                print("Result: Incorrectly identified as 'Genuine'")
        except Exception as e:
            print(f"An error occurred during verification: {e}")

    print("\n--- Verification Test Finished ---")