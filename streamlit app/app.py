import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import os

# ✅ Paths
MODEL_PATH = "../models/final_model.h5"
CLASS_MAP_PATH = "../models/class_indices.json"
IMG_SIZE = (224, 224)

# ✅ Load model once
@st.cache_resource
def load_the_model():
    model = load_model(MODEL_PATH)
    return model

# ✅ Load class mapping
@st.cache_data
def load_class_map():
    with open(CLASS_MAP_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # invert mapping

# Load everything
model = load_the_model()
inv_class_map = load_class_map()

# ✅ App UI
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to predict its species.")

# File uploader
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    top_label = inv_class_map[top_idx]
    top_prob = float(preds[top_idx])

    # Display main result
    st.markdown(f"### 🎯 Prediction: **{top_label}**")
    st.progress(int(top_prob * 100))
    st.write(f"Confidence: {top_prob:.2%}")

    # Display top-3 predictions
    st.write("#### Top 3 predictions:")
    top_3 = np.argsort(preds)[::-1][:3]
    for i in top_3:
        st.write(f"- {inv_class_map[i]}: {preds[i]:.2%}")
