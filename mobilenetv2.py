

import cv2
import numpy as np
from keras.applications import MobileNetV2, preprocess_input, decode_predictions

def load_model():
    # Load MobileNetV2 pre-trained model (excluding classification head)
    model = MobileNetV2(weights='imagenet', include_top=False)
    return model

def preprocess_image(image_path):
    # Load and preprocess the input image for the model
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # MobileNetV2 input size
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def classify_image(model, image_path):
    # Preprocess the input image
    image = preprocess_image(image_path)

    # Perform inference to get class predictions
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions)

    # Get the top predicted class and its probability
    top_prediction = decoded_predictions[0][0]
    class_name, class_description, confidence = top_prediction
    return class_name, confidence

if __name__ == '__main__':
    # Load the pre-trained model
    model = load_model()

    # Path to the image for classification
    image_path = 'path_to_image/image.jpg'

    # Classify the image
    class_name, confidence = classify_image(model, image_path)

    # Print the result
    print(f"Predicted class: {class_name}, Confidence: {confidence:.2f}")
