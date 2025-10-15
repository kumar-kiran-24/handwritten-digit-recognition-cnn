import os
import subprocess
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# --------------------------
# Try to import TensorFlow Lite runtime, install if missing
# --------------------------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    with st.spinner("Installing TensorFlow Lite runtime..."):
        subprocess.run(
            [
                "pip",
                "install",
                "-q",
                "https://github.com/google-coral/pycoral/releases/download/release-frogfish/"
                "tflite_runtime-2.14.0-cp310-cp310-manylinux2014_x86_64.whl",
            ],
            check=True,
        )
    import tflite_runtime.interpreter as tflite

# --------------------------
# Load the TFLite model
# --------------------------
MODEL_PATH = "mnist_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ Model file '{MODEL_PATH}' not found.\n\n"
        "➡️ Please make sure 'mnist_model.tflite' is in the root folder of your project."
    )
    st.stop()

# Initialize the TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Prediction Function
# --------------------------
def predict_image(image):
    """Preprocess image and run prediction using the TFLite model."""
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to MNIST format (28x28)
    img = image.resize((28, 28))
    # Normalize to [0, 1]
    img = np.array(img, dtype=np.float32) / 255.0
    # Reshape for model: (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], img)
    # Run inference
    interpreter.invoke()
    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]["index"])
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    return predicted_class, confidence

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="centered")

st.title("✍️ Handwritten Digit Recognition (MNIST)")
st.write("Upload an image of a handwritten digit (0–9) and the model will predict it!")

file_ = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if file_ is not None:
    image = Image.open(file_)
    st.image(image, caption="Uploaded Image", width=150)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing the image..."):
            result, confidence = predict_image(image)
        st.success(f"**Predicted Digit:** {result}")
        st.info(f"Confidence: {confidence:.2f}%")
