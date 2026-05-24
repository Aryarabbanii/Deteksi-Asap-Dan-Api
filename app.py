import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Cache model agar tidak load ulang
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Asap & Api", layout="centered")

st.title("🔥 Deteksi Asap dan Api dari Gambar")
st.write("Upload gambar untuk mendeteksi keberadaan asap atau api menggunakan model YOLOv11.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Tampilkan gambar asli
    st.image(image, caption="Gambar Asli", use_container_width=True)

    # Jalankan deteksi
    with st.spinner("Sedang mendeteksi..."):
        results = model(img_np)

    # Ambil hasil gambar dengan bounding box
    annotated_img = results[0].plot()

    # Tampilkan hasil
    st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

    # Tampilkan jumlah objek terdeteksi
    num_detected = len(results[0].boxes)
    st.success(f"Jumlah objek terdeteksi: {num_detected}")
