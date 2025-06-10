import cv2
import os
import sys
import shutil
import glob
import numpy as np

# --- THIS IS THE ONLY LINE THAT HAS CHANGED ---
# It now correctly points to the 'src' directory, allowing Python to find 'model_sources.gan'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the test module from its new location
from SOURCE.gan_files import test

# Define the hardcoded paths required by the author's script
GAN_INPUT_DIR = os.path.abspath('results/gan/gan_signdata_kaggle/gan_ips/testB')
GAN_OUTPUT_DIR = os.path.abspath('results/gan/gan_signdata_kaggle/test_latest/images')

def clean_signature(signature_image: np.ndarray, model_path: str) -> np.ndarray:
    """
    Cleans a signature image by using the author's original test.py script.
    This function orchestrates the required file I/O operations.

    Args:
        signature_image (np.ndarray): The input signature image (from cv2.imread).
        model_path (str): Path to the model. While not used directly in the call, 
                          it's kept for consistency and future modifications. The test.py
                          script has the model path hardcoded in its options.
    Returns:
        np.ndarray: The cleaned signature image as a NumPy array.
    """
    try:
        # 1. Prepare the input directory
        if os.path.exists(GAN_INPUT_DIR):
            shutil.rmtree(GAN_INPUT_DIR)
        os.makedirs(GAN_INPUT_DIR)

        # 2. Save the signature image to the required input location.
        temp_input_path = os.path.join(GAN_INPUT_DIR, 'temp_signature.png')
        cv2.imwrite(temp_input_path, signature_image)

        # 3. Call the author's cleaning function.
        print("Invoking the author's cleaning script...")
        test.clean()
        print("Cleaning script finished.")

        # 4. Retrieve the output image.
        output_files = glob.glob(os.path.join(GAN_OUTPUT_DIR, '*_fake.png'))
        
        if not output_files:
            raise FileNotFoundError("Cleaning script did not produce an output file.")

        cleaned_image_path = output_files[0]
        cleaned_image = cv2.imread(cleaned_image_path)
        
        if cleaned_image is None:
            raise IOError(f"Could not read the cleaned image at {cleaned_image_path}")
            
        return cleaned_image

    except Exception as e:
        print(f"An error occurred during the cleaning process: {e}")
        return None