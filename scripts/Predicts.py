#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np

def predict(image_path):
    # Path to the saved model
    model_path = '/Users/larrymutoni/Documents/DatasetTest/SavedModel2'

    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_path)

    # ImageDataGenerator for resizing and preprocessing
    data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = data_augmentation.standardize(img_array)

    # Make predictions
    predictions = loaded_model.predict(processed_img)

    # Decode the prediction
    predicted_class = "MSV" if predictions[0] > 0.5 else "HEALTHY"

    # Return prediction results
    return {
        "image_path": image_path,
        "raw_predictions": predictions.tolist(),
        "predicted_class": predicted_class,
        "confidence": float(predictions[0])
    }

if __name__ == "__main__":
    # Expecting the image path as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction_results = predict(image_path)

    # Print the results (you can modify this part based on your needs)
    print(f"Image: {prediction_results['image_path']}")
    print(f"Raw predictions: {prediction_results['raw_predictions']}")
    print(f"Predicted Class: {prediction_results['predicted_class']}")
    print(f"Confidence: {prediction_results['confidence']}")

