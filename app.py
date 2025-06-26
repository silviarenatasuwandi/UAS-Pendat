import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("heart_failure_scaler.pkl")

# Header Aplikasi
st.set_page_config(page_title="Prediksi Gagal Jantung", layout="centered")
st.title("🫀 Prediksi Kematian Akibat Gagal Jantung")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi kemungkinan kematian selama perawatan.")

# Form input pengguna
with st.form("input_form"):
    age = st.number_input("Usia", min_value=0.0)
    anaemia = st.selectbox("Anaemia (Hemoglobin Rendah)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0.0)
    diabetes = st.selectbox("Riwayat Diabetes", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0.0)
    high_blood_pressure = st.selectbox("Tekanan Darah Tinggi", [0, 1])
    platelets = st.number_input("Jumlah Platelet", min_value=0.0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0.0)
    sex = st.selectbox("Jenis Kelamin", options=[(1, "Laki-laki"), (0, "Perempuan")], format_func=lambda x: x[1])[0]
    smoking = st.selectbox("Perokok", [0, 1])
    time = st.number_input("Waktu Follow-up (hari)", min_value=0.0)

    submit = st.form_submit_button("🔍 Prediksi")

# Prediksi saat tombol ditekan
if submit:
    input_data = np.array([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
        smoking, time
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("❌ Pasien diprediksi **MENINGGAL** selama perawatan.")
    else:
        st.success("✅ Pasien diprediksi **BERTAHAN HIDUP** selama perawatan.")
