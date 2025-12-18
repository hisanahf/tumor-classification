import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import preprocess_input
import base64


#custom warna
st.markdown("""
<style>

/* === GLASSMORPHISM UNTUK SEMUA BOX ALERT === */
.stAlert {
    background: rgba(255, 255, 255, 0.35) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    padding: 12px !important;
}

/* === STYLE TEKS DI DALAM ALERT === */
.stAlert p {
    color: #102040 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* === BORDER KHUSUS BERDASARKAN TIPE ALERT === */
.stAlert[data-baseweb="notification"][kind="info"] {
    border-left: 4px solid #5fa8f0 !important;
}

.stAlert[data-baseweb="notification"][kind="success"] {
    border-left: 4px solid #57cc99 !important;
}

.stAlert[data-baseweb="notification"][kind="warning"] {
    border-left: 4px solid #f6c445 !important;
}

</style>
""", unsafe_allow_html=True)


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

def add_bg_from_local(image_file):
    encoded_bg = get_base64_image(image_file)
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_bg}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

#background
add_bg_from_local("ungu.jpg")

MODEL_PATH = "best_mobilenetv2.h5" 

@st.cache_resource
def load_model():
    # Gunakan compile=False agar tidak mengecek metadata training lama
    return keras.models.load_model("best_mobilenetv2.h5", compile=False)

model = load_model()

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (200, 200)

#SIDE BAR MENU
with st.sidebar:
    st.title("🧭 Petunjuk Menu ")
    menu = st.radio("Pilih Menu", [
        "Klasifikasi Hasil MRI",
        "Tentang Tumor Otak",
        "Tips Kesehatan Otak",
        "Tentang Aplikasi"
    ])
    st.markdown("---")
    st.info("Aplikasi ini menggunakan AI untuk mendeteksi pola pada citra MRI.")

def predict(image):
    img = image.resize(IMG_SIZE)
    img = img.convert("RGB")
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    return CLASS_NAMES[class_id], pred[0]


#Halaman 1

if menu == "Klasifikasi Hasil MRI":
    st.title("🧠 Brain Tumor Classification")
    st.markdown("### Deteksi Dini Menggunakan Deep Learning")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Silahkan upload gambar MRI kepala untuk mendeteksi jenis tumor.")
        uploaded_file = st.file_uploader("Upload gambar MRI (jpg/png)", type=["jpg", "jpeg", "png"])

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Diunggah", width=300)

            if st.button("🔍 Analisis Gambar"):
                if model is None:
                    st.error("Model tidak ditemukan.")
                else:
                    with st.spinner('Sedang menganalisis gambar...'):
                        label, probabilities = predict(image)

                    st.markdown("---")
                    st.subheader(f"🧾 Hasil Prediksi: **{label.upper()}**")

                    st.write("Probabilitas Deteksi:")
                    for cls, prob in zip(CLASS_NAMES, probabilities):
                        col_name, col_bar, col_val = st.columns([1, 3, 1])
                        with col_name:
                            st.write(f"**{cls.capitalize()}**")
                        with col_bar:
                            st.progress(float(prob))
                        with col_val:
                            st.write(f"{prob*100:.2f}%")

                    st.markdown("---")
                    st.markdown("### 🩺 Penjelasan Medis:")

                    if label == "notumor":
                        st.success("✔ **Tidak terdeteksi adanya tumor.**")
                        st.info("Tetap konsultasikan ke dokter untuk hasil akurat.")
                    else:
                        st.warning(f"⚠ Terindikasi adanya **{label.capitalize()} Tumor**.")
                        st.info("Hasil ini bukan diagnosis final. Segera konsultasikan ke dokter.")


#Halaman 2

elif menu == "Tentang Tumor Otak":
    st.header("💡Edukasi Tumor Otak")
    tab1, tab2, tab3 = st.tabs(["Glioma", "Meningioma", "Pituitary"])
    
    with tab1:
        st.header("Glioma")
        st.write("Tumor yang tumbuh pada sel glial otak atau sel pendukung otak.")
        st.warning("Jenis ini seringkali tumbuh di jaringan otak itu sendiri.")
        
    with tab2:
        st.header("Meningioma")
        st.write("Tumbuh pada meninges atau selaput otak yang mengelilingi bagian luar otak.")
        st.info("Seringkali jinak tetapi dapat menekan otak jika membesar.")
        
    with tab3:
        st.header("Pituitary")
        st.write("Tumbuh pada kelenjar pituitari atau kelenjar hormon di dasar otak.")
        st.success("Dapat mempengaruhi keseimbangan hormon tubuh.")

#Halaman 3

elif menu == "Tips Kesehatan Otak":
    st.header("🧘 Tips Menjaga Kesehatan Otak")
    st.write("Berikut tips sederhana untuk menjaga fungsi otak:")

    st.subheader("🍎 1. Nutrisi")
    st.write("Mengkonsumsi Buah seperti Blueberry, kacang-kacangan, ikan salmon, dan alpukat.")

    st.subheader("🏃 2. Olahraga")
    st.write("Olahraga ringan 20–30 menit per hari, dapat meningkatkan aliran darah ke otak.")

    st.subheader("🧠 3. Kemampuan Kognitif")
    st.write("Membaca buku, bermain puzzle, menghafal nama Negara dapat meningkatkan kemampuan kognitif")

    st.subheader("😴 4. Tidur Berkualitas")
    st.write("Tidur 7–9 jam sangat penting untuk memori dan fokus.")

    st.subheader("💧 5. Asupan Air")
    st.write("Perbanyak Minum karena kurang minum dapat membuat otak sulit berkonsentrasi.")

    st.subheader("🧘 6. Kelola Stres")
    st.write("Lakukan meditasi ringan atau aktivitas menenangkan lainnya.")

    st.success("💡 *Kebiasaan sehat sehari-hari dapat membantu menjaga kesehatan otak dan meningkatkan kualitas hidup yang baik.*")


#Halaman 4

elif menu == "Tentang Aplikasi":
    st.header("ℹ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk melakukan deteksi awal tumor otak 
    menggunakan model deep learning.

    Fitur aplikasi:
    - Upload MRI & prediksi tumor 
    - Penjelasan sederhana mengenai tumor otak  
    - Tips kesehatan otak  

    Aplikasi ini **bukan alat diagnosis resmi dan akurat 100%**, ini merupakan alat bantu diagnosa awal dan edukasi bagi pengguna.
    
             
    Aplikasi by hisanahf""")
   