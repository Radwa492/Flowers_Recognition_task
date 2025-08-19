import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------------
# Prediction function
# -------------------------
def predict_image_class(img_file, model, class_indices, target_size=(128, 128)):
    """
    Predicts the class of an image given the uploaded file.
    """
    img = image.load_img(img_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    # Reverse mapping index -> class
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_index]

    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Flower Classifier 🌸", page_icon="🌼", layout="centered")

st.markdown(
    """
    <div style="text-align: center;">
        <h1>🌸 Flower Classification App 🌻</h1>
        <p style="font-size:18px;">Upload a flower image and I’ll predict whether it’s a <b>Daisy</b>, 
        <b>Dandelion</b>, <b>Rose</b>, <b>Sunflower</b>, or <b>Tulip</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_flower_model.keras")  # replace with your model
    return model

model = load_model()

# ⚠️ You must pass your training generator class_indices here
# Example: class_indices = train.class_indices
# For demo, define manually:
class_indices = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("🔍 Analyzing the flower..."):
        predicted_class, confidence = predict_image_class(uploaded_file, model, class_indices)

    # Result
    st.markdown(
        f"""
        <div style="
            background-color:#f0f8ff;
            border-radius:15px;
            padding:20px;
            margin-top:20px;
            text-align:center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        ">
            <h2 style="color:#4CAF50;">✅ Prediction: {predicted_class.title()}</h2>
            <p style="font-size:18px;">Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("👆 Please upload an image to start classification.")
