import torch
import cv2
import numpy as np

def detect_signatures(image_path: str, model_path: str) -> list:
    """
    Detects signatures in an image using a custom YOLOv5 model.

    Args:
        image_path (str): The path to the input image file.
        model_path (str): The path to the pre-trained YOLOv5 .pt model file.

    Returns:
        list: A list of cropped signature images as NumPy arrays.
    """
    # Load the custom YOLOv5 model
    # Note: If this is the first time, it might download YOLOv5 dependencies.
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        # If it fails due to cache or internet issues, try force_reload
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        except Exception as e_reload:
             print(f"Critical Error: Could not load model even with force_reload. {e_reload}")
             return []


    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    # Perform inference
    results = model(img)

    # Process results and crop signatures
    cropped_signatures = []
    # The results.xyxy[0] tensor contains bounding box coordinates [xmin, ymin, xmax, ymax, confidence, class]
    for *box, conf, cls in results.xyxy[0]:
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Crop the image using NumPy slicing
        cropped_img = img[y1:y2, x1:x2]
        cropped_signatures.append(cropped_img)
        print(f"Detected signature with confidence {conf:.2f} at location [{x1}, {y1}, {x2}, {y2}]")

    return cropped_signatures