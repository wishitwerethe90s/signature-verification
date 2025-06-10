import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense
import cv2
import numpy as np
import os

# Add the project's root directory to the Python path to find the SOURCE folder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the necessary components from the SOURCE directory
from SOURCE.signet_files.network import create_base_network_signet
from SOURCE.signet_files.utils import euclidean_distance, eucl_dist_output_shape

# Define image dimensions based on the source main.py
IMG_W, IMG_H = 150, 220
INPUT_SHAPE = (IMG_H, IMG_W, 1)

def _preprocess_image(image_path):
    """Helper function to read and preprocess a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    img = cv2.resize(img, (IMG_W, IMG_H))

    img = np.array(img, dtype=np.float64)
    img /= 255.0
    img = img[..., np.newaxis]
    return img

def get_feature_vector(image_path, model_weights_path):
    """
    Loads the base network, processes a single image, and returns its feature vector.
    This is used for the 1:N Recognition task.
    """
    base_network = create_base_network_signet(INPUT_SHAPE)
    
    # To get the feature vector, we only need the base_network part of the model.
    # We must load the weights into the full Siamese model and then extract the base_network layer.
    input_a = Input(shape=(INPUT_SHAPE))
    input_b = Input(shape=(INPUT_SHAPE))
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    outputs = Dense(1, activation="sigmoid")(distance)
    full_model = Model(inputs=[input_a, input_b], outputs=outputs)
    full_model.load_weights(model_weights_path)

    # Preprocess the input image
    img = _preprocess_image(image_path)
    
    # Now, use the extracted base_network to predict (generate the feature vector)
    feature_vector = base_network.predict(np.expand_dims(img, axis=0))
    return feature_vector

def get_verification_score(image_path1, image_path2, model_weights_path):
    """
    Loads the full Siamese model, processes a pair of images, and returns the verification score.
    This is used for the 1:1 Verification task.
    """
    base_network = create_base_network_signet(INPUT_SHAPE)
    
    input_a = Input(shape=(INPUT_SHAPE))
    input_b = Input(shape=(INPUT_SHAPE))
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[input_a, input_b], outputs=outputs)

    model.load_weights(model_weights_path)
    
    # Preprocess both images
    img1 = _preprocess_image(image_path1)
    img2 = _preprocess_image(image_path2)
    
    # Predict the similarity score
    score = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])[0][0]
    return score