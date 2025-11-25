import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import cv2
from matplotlib import cm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DRIVE_ID = "1XlZArIYbtkG3_NRyP2hsRBViY0C5T67f"
MODEL_PATH = "model.weights.h5"  # model filename

st.title("Skin Lesion Classifier with Grad-CAM++")

# -----------------------------
# GRAD-CAM++ FUNCTIONS
# -----------------------------
def compute_gradcam_plus_plus(model, img_array, layer_name=None):
    if layer_name is None:
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
        if not conv_layers:
            return None
        layer_name = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    grads_power_2 = tf.square(grads)
    grads_power_3 = grads_power_2 * grads
    sum_conv = tf.reduce_sum(conv_outputs * grads_power_3, axis=(1, 2), keepdims=True)
    alpha_denom = 2 * grads_power_2 + sum_conv
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = grads_power_2 / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))

    cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)

    cam = cam.numpy()[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_gradcam(img, heatmap, alpha=0.5, colormap=cm.jet):
    if isinstance(img, Image.Image):
        img = np.array(img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = colormap(heatmap)[:, :, :3]
    overlay = heatmap_colored * alpha + img / 255.0
    overlay = np.clip(overlay, 0, 1)
    return overlay

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -----------------------------
# IMAGE UPLOADER
# -----------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    # Predict
    class_names = [
        'Melanoma → Cancer (malignant)',
        'Melanocytic Nevus → Usually benign (moles, not cancer)',
        'Basal Cell Carcinoma (BCC) → Cancer (skin cancer, usually slow-growing)',
        'Actinic Keratosis (AK) → Pre-cancerous (can turn into squamous cell carcinoma if untreated)']
    pred = model.predict(arr)[0]

    st.subheader("Prediction Result")
    if pred.shape[0] > 1:
        idx = int(np.argmax(pred))
        confidence = float(pred[idx])
        st.write(f"Top class: **{class_names[idx]}** ({confidence:.4f})")
        st.write("**Full probabilities:**")
        for i, p in enumerate(pred):
            st.write(f"{class_names[i]}: {p:.4f}")

        st.subheader("Simple Report")
        st.write(f"The model predicts **{class_names[idx]}** with **{confidence:.2%}** confidence.")
    else:
        p = float(pred[0])
        st.write(f"Probability: {p:.4f}")

    # -----------------------------
    # DISPLAY ORIGINAL AND GRAD-CAM SIDE BY SIDE
    # -----------------------------
    heatmap = compute_gradcam_plus_plus(model, arr)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM++ Heatmap")
        if heatmap is not None:
            overlay = overlay_gradcam(img_resized, heatmap)
            st.image(overlay, use_column_width=True)
        else:
            st.warning("Grad-CAM++ could not be generated.")

else:
    st.info("""
    ### How to use:
    1. Upload a clear skin lesion image (jpg, jpeg, png)
    2. Wait for the model to predict the class
    3. Review the prediction result and confidence
    4. View Grad-CAM++ heatmap highlighting important regions
    """)
