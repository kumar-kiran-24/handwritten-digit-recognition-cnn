import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import tflite_runtime.interpreter as tflite

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = "mnist_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please make sure it is in the project root.")
    st.stop()

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_image(image):
    """
    Preprocess the uploaded image and predict using the TFLite model.
    """
    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Resize to MNIST 28x28
    img = image.resize((28, 28))

    # Normalize pixel values
    img = np.array(img, dtype=np.float32) / 255.0

    # Reshape for the model: (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction[0])

    return predicted_class


# --------------------------
# Streamlit UI
# --------------------------
st.title("🖊️ Handwritten Digit Recognition (MNIST)")
st.write("Upload an image of a handwritten digit (0–9) and I’ll predict the number!")

file_ = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file_ is not None:
    image = Image.open(file_)
    st.image(image, caption="Uploaded Image", width=150)

    if st.button("🔍 Predict"):
        result = predict_image(image)
        st.success(f"Predicted Digit: {result}")
