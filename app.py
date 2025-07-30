import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load model
model = load_model("model.h5")
categories = ["chrysanthemum", "gumamela", "rose", "sampaguita", "sunflower", "tulip"]

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌼 Flower Classifier")
st.markdown("Upload an image of a flower and get the predicted type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess
    image_resized = image.resize((64, 64))
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = categories[np.argmax(prediction)]

    st.success(f"🌸 Prediction: **{predicted_label}**")
