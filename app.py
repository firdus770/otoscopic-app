import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  

st.title("Otoscopic Image Classifier")
st.markdown("Upload an otoscopic image to classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("### Prediction:")
    results = model.predict(image, imgsz=224)
    top1 = results[0].probs.top1
    conf = results[0].probs.top1conf
    label = model.names[top1]

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {conf:.2f}")
