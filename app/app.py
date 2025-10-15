import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import os


# Load the model from the correct path
# The '..' navigates up one directory to find the 'model' folder
try:
    from tensorflow.keras.models import load_model

    model = load_model("mnist_model.h5")

except Exception as e:
    st.error(f"Error loading the model. Make sure 'mnist_model.h5' is in the 'model' directory at the root level of your project. Error: {e}")
    st.stop()


def predictImage(image):
    """
    Preprocesses the input image and makes a prediction using the loaded model.
    """
    # Convert the input image to grayscale
    image = ImageOps.grayscale(image)

    # Convert to MNIST size (28x28 pixels)
    img = image.resize((28, 28))

    # Normalize pixel values to be between 0 and 1
    img = np.array(img, dtype="float32") / 255.0

    # Reshape the image to fit the model's input shape (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Predict the number
    pred = model.predict(img)
    return np.argmax(pred[0])

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a single digit to see the prediction.")

# File uploader with the correct function name
file_ = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if file_ is not None:
    # Open the image
    image = Image.open(file_)
    st.image(image, caption="Uploaded Image", width=150)

    # Predict button
    if st.button("Predict Now"):
        # Make sure the user has uploaded a file before predicting
        if file_ is not None:
            result = predictImage(image=image)
            st.header("Predicted Digit: " + str(result))
        else:
            st.warning("Please upload an image first.")