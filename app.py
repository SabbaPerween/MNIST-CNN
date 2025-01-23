import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Streamlit app title
st.title("Handwritten Digit Recognition")

# Add a canvas for drawing
st.write("Draw a digit (0-9) in the box below:")
canvas_result = st_canvas(
    fill_color="black",  # Background color of the canvas
    stroke_width=15,     # Thickness of the drawing stroke
    stroke_color="white",# Color of the drawing stroke
    background_color="black",  # Background color of the canvas
    width=280,           # Width of the canvas
    height=280,          # Height of the canvas
    drawing_mode="freedraw",  # Allow free drawing
    key="canvas",
)

# Preprocess the drawn image
if canvas_result.image_data is not None:
    # Convert the canvas image to grayscale
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (black background to white)
    img = img.resize((28, 28))  # Resize to 28x28 pixels

    # Convert the image to a numpy array and normalize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display the prediction
    st.write(f"Predicted Digit: **{predicted_digit}**")

    # Show the processed image
    st.image(img, caption="Processed Image", width=100)

    # Display the prediction probabilities
    st.write("Prediction Probabilities:")
    st.bar_chart(prediction[0])