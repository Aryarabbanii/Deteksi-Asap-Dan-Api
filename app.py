import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model YOLOv11
model = YOLO("best.pt")

st.set_page_config(page_title="Deteksi Asap & Api", layout="centered")
st.title("🔥 Deteksi Asap dan Api dari Gambar")
st.write("Upload gambar untuk mendeteksi keberadaan asap atau api menggunakan model YOLOv11.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi file ke array numpy
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Jalankan deteksi
    results = model(img_np)
    annotated_img = results[0].plot()

    # Tampilkan hasil deteksi
    st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)

    # Jika ingin simpan hasil
    # cv2.imwrite("hasil_deteksi.jpg", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
