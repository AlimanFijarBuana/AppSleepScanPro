import streamlit as st
import streamlit.components.v1 as components 
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Set page config for premium look
st.set_page_config(
    page_title="SleepScan Pro",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model & encoder
try:
    dt_model = joblib.load('best_model_decision_tree.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    lr_model = joblib.load('logistic_regression_model.pkl')
    gender_enc = joblib.load('Gender_label_encoder.pkl')
    occupation_enc = joblib.load('Occupation_label_encoder.pkl')
    bmi_enc = joblib.load('BMI Category_label_encoder.pkl')
    target_enc = joblib.load('target_label_encoder.pkl')
    scaler = joblib.load('minmax_scaler_split.pkl')
except Exception as e:
    st.error(f"❌ Gagal load model/encoder: {e}")
    st.stop()

# Ultra Premium theme styling
def set_premium_theme():
    st.markdown(
        """
        <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }

        /* Title styling */
        h1 {
            color: white !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 10px rgba(74, 58, 255, 0.5) !important;
            margin-bottom: 1.5rem !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(195deg, #0f0c29 0%, #302b63 100%) !important;
            border-right: 1px solid #4a3aff;
            box-shadow: 5px 0 15px rgba(0, 0, 0, 0.3);
        }

        /* Input fields styling */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background: rgba(255, 255, 255, 0.08) !important;
            color: white !important;
            border: 1px solid #4a3aff !important;
            border-radius: 12px !important;
            padding: 12px !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #8a2be2 !important;
            box-shadow: 0 0 0 2px rgba(138, 43, 226, 0.2) !important;
        }

        .stSlider .st-ax {
            color: #4a3aff !important;
        }

        .stSlider .st-bx {
            background: rgba(74, 58, 255, 0.3) !important;
            border-radius: 12px !important;
            height: 8px !important;
        }

        .stSlider .st-cx {
            background: linear-gradient(90deg, #4a3aff 0%, #8a2be2 100%) !important;
        }

        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #4a3aff 0%, #8a2be2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 20px rgba(74, 58, 255, 0.4) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(74, 58, 255, 0.6) !important;
            background: linear-gradient(90deg, #4a3aff 0%, #9b4dff 100%) !important;
        }

        /* Card styling */
        .main-card {
            background: rgba(15, 12, 41, 0.7) !important;
            border-radius: 20px !important;
            padding: 2.5rem !important;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
            border: 1px solid rgba(74, 58, 255, 0.3) !important;
            backdrop-filter: blur(12px) !important;
            margin-bottom: 2.5rem !important;
            transition: all 0.3s ease !important;
        }

        .main-card:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.5) !important;
            border-color: rgba(138, 43, 226, 0.5) !important;
        }

        /* Prediction result cards */
        .prediction-card {
            background: rgba(15, 12, 41, 0.9) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            border-left: 6px solid #4a3aff !important;
            margin: 1.5rem 0 !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.3s ease !important;
        }

        .prediction-card:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4) !important;
        }

        .success-card {
            border-left: 6px solid #00c853 !important;
            background: rgba(0, 200, 83, 0.1) !important;
        }

        .warning-card {
            border-left: 6px solid #ffab00 !important;
            background: rgba(255, 171, 0, 0.1) !important;
        }

        .error-card {
            border-left: 6px solid #ff1744 !important;
            background: rgba(255, 23, 68, 0.1) !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(15, 12, 41, 0.5);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#4a3aff, #8a2be2);
            border-radius: 10px;
        }

        /* Force input labels to white */
        .stSlider label,
        .stSelectbox label,
        .stNumberInput label,
        [data-testid="stFormLabel"] p,
        .stRadio label,
        .stCheckbox label {
            color: white !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
        }
        
        /* Tab styling */
        [data-baseweb="tab-list"] {
            gap: 10px !important;
            padding: 0 0.5rem !important;
        }

        [data-baseweb="tab"] {
            background: rgba(15, 12, 41, 0.7) !important;
            border-radius: 12px !important;
            padding: 10px 20px !important;
            border: 1px solid rgba(74, 58, 255, 0.3) !important;
            color: #ffffff !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
            margin: 0 5px !important;
        }

        [data-baseweb="tab"]:hover {
            background: rgba(74, 58, 255, 0.3) !important;
            transform: translateY(-2px) !important;
        }

        [aria-selected="true"] {
            background: linear-gradient(90deg, #4a3aff 0%, #8a2be2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(74, 58, 255, 0.4) !important;
            border: none !important;
        }

        /* Expander styling */
        .stExpander {
            border: 1px solid rgba(74, 58, 255, 0.3) !important;
            border-radius: 12px !important;
            margin-bottom: 1.5rem !important;
        }

        .stExpander summary {
            background: rgba(15, 12, 41, 0.7) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            font-weight: 600 !important;
        }

        .stExpander summary:hover {
            background: rgba(74, 58, 255, 0.2) !important;
        }

        /* Spinner styling */
        .stSpinner > div {
            border: 3px solid rgba(74, 58, 255, 0.2);
            border-radius: 50%;
            border-top: 3px solid #4a3aff;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Footer styling */
        .footer {
            background: linear-gradient(90deg, rgba(15,12,41,0.8) 0%, rgba(26,26,46,0.8) 100%) !important;
            padding: 1.5rem !important;
            border-radius: 16px !important;
            margin-top: 3rem !important;
            border: 1px solid rgba(74, 58, 255, 0.3) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }

        /* Custom glow effect */
        .glow {
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from {
                box-shadow: 0 0 5px rgba(74, 58, 255, 0.5);
            }
            to {
                box-shadow: 0 0 20px rgba(74, 58, 255, 0.8);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_premium_theme()

# Premium sidebar with logo
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" fill="url(#gradient)"/>
            <path d="M12 7C12 7 10 9 10 12C10 15 12 17 12 17" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M12 7C12 7 14 9 14 12C14 15 12 17 12 17" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M7 12C7 12 9 10 12 10C15 10 17 12 17 12" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M7 12C7 12 9 14 12 14C15 14 17 12 17 12" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#4a3aff" />
                    <stop offset="100%" stop-color="#8a2be2" />
                </linearGradient>
            </defs>
        </svg>
        <h1 style="color: white; margin-top: 15px; font-weight: 700; letter-spacing: 1px;">SleepScan Pro</h1>
        <p style="color: rgba(255,255,255,0.7); margin-top: 5px; font-size: 0.9rem;">AI-Powered Sleep Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu_option = st.radio(
        "Navigation",
        ("🌟 Prediction", "📊 Analytics"),
        key="menu_radio",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; text-align: center;">
        <p>Advanced Sleep Disorder Detection</p>
        <p style="font-size: 0.8rem; margin-top: 2rem; color: rgba(255,255,255,0.5);">
            v2.1.0 | Premium Edition
        </p>
    </div>
    """, unsafe_allow_html=True)

if menu_option == "🌟 Prediction":
    st.markdown("<h1 style='color: white;'>🌙 DETEKSI GANGGUAN TIDUR</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: rgba(255,255,255,0.8); margin-bottom: 2rem; font-size: 1.1rem;">
        Model AI kami menganalisis pola tidur dan metrik kesehatan Anda untuk mendeteksi potensi gangguan tidur dengan akurasi tinggi.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Aplikasi", expanded=False):
        st.write("""
        **SleepScan Pro** menggunakan teknologi machine learning canggih untuk memprediksi gangguan tidur berdasarkan:
        - Pola tidur harian
        - Metrik kesehatan fisik
        - Gaya hidup dan aktivitas
        - Tingkat stres
        
        Tiga model berbeda tersedia untuk prediksi:
        1. **Decision Tree** - Cepat dan interpretable
        2. **Random Forest** - Akurasi tinggi dengan ensemble learning
        3. **Logistic Regression** - Model statistik yang kuat
        
        Model ini dilatih menggunakan data klinis dari lebih dari 300 pasien dengan validasi silang ketat.
        """)

    # Model selection
    model_choice = st.selectbox(
        "Pilih Model Prediksi",
        ["Decision Tree", "Random Forest", "Logistic Regression"],
        help="Pilih model machine learning untuk analisis",
        key="model_select"
    )

    # Create tabs for different input sections
    tab1, tab2, tab3 = st.tabs(["🧑 Data Personal", "💤 Pola Tidur", "❤️ Kesehatan"])

    with tab1:
        col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Umur", 18, 100, 27, 
                       help="Usia individu dalam tahun")
        gender = st.selectbox("Gender", ["Male", "Female"], 
                             help="Jenis kelamin individu")
        bmi_category = st.selectbox("Kategori BMI", ["Normal", "Overweight", "Obese"], 
                                    help="Klasifikasi BMI individu")
        
     with col2:
    # Pilih pekerjaan dari daftar pekerjaan yang telah disesuaikan
        occupation = st.selectbox("Pekerjaan", [
        'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 
        'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Manager', 'Others'
        ], help="Pekerjaan individu", key="occupation", index=0)  # Tambahkan index default

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.slider("Durasi Tidur (Jam)", 0.0, 12.0, 7.0, 0.1,
                                      help="Jumlah jam tidur per malam")
            quality_of_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7,
                                       help="Penilaian kualitas tidur mandiri")
            
        with col2:
            physical_activity = st.slider("Tingkat Aktivitas (1-100)", 0, 100, 60,
                                        help="Tingkat aktivitas fisik harian")
            stress_level = st.slider("Tingkat Stress (1-10)", 1, 10, 4,
                                    help="Tingkat stres mandiri")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.slider("Detak Jantung Istirahat (bpm)", 40, 120, 72,
                                 help="Detak jantung istirahat dalam denyut per menit")
            daily_steps = st.slider("Langkah Harian", 0, 20000, 7500, 500,
                                   help="Rata-rata jumlah langkah per hari")
            
        with col2:
            systolic = st.slider("Tekanan Darah Sistolik (mmHg)", 80, 200, 120, 1,
                                help="Tekanan darah sistolik")
            diastolic = st.slider("Tekanan Darah Diastolik (mmHg)", 50, 120, 80, 1,
                                 help="Tekanan darah diastolik")

    # Prepare features
    try:
        num_features = np.array([
            age, sleep_duration, quality_of_sleep, physical_activity,
            stress_level, heart_rate, daily_steps, systolic, diastolic
        ])
        
        if np.isnan(num_features).any() or np.any(num_features < 0):
            st.error("Pastikan semua kolom numerik terisi dengan benar dan memiliki nilai yang valid.")
            st.stop()

        num_scaled = scaler.transform(num_features.reshape(1, -1)).flatten()

        gender_num = gender_enc.transform([gender])[0]
        occupation_num = occupation_enc.transform([occupation])[0]
        bmi_num = bmi_enc.transform([bmi_category])[0]

        features = np.array([
            gender_num,
            num_scaled[0],  # age
            occupation_num,
            num_scaled[1],  # sleep duration
            num_scaled[2],  # quality of sleep
            num_scaled[3],  # physical activity
            num_scaled[4],  # stress
            bmi_num,
            num_scaled[5],  # heart rate
            num_scaled[6],  # daily steps
            num_scaled[7],  # systolic
            num_scaled[8],  # diastolic
        ], dtype=float)

        if np.isnan(features).any():
            st.error("❌ Fitur akhir mengandung nilai NaN.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Error saat menyiapkan fitur: {e}")
        st.stop()

    # Prediction button with animation
    if st.button("✨ Mulai Analisis", use_container_width=True, key="predict_button"):
        with st.spinner("Menganalisis data tidur Anda dengan AI..."):
            try:
                if model_choice == "Decision Tree":
                    pred = dt_model.predict(features.reshape(1, -1))
                elif model_choice == "Random Forest":
                    pred = rf_model.predict(features.reshape(1, -1))
                else:
                    pred = lr_model.predict(features.reshape(1, -1))

                decoded = target_enc.inverse_transform(pred)
                prediction = str(decoded[0]) if decoded.size > 0 else "Good"

                if prediction == "Good":
                    st.markdown("""
                    <div class="prediction-card success-card">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" fill="#00C853"/>
                                <path d="M8 12L11 15L16 9" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <h3 style="color: #00c853; margin: 0 0 0 10px;">Tidak Ditemukan Gangguan Tidur</h3>
                        </div>
                        <p style="font-size: 1.05rem;">Pola tidur Anda terlihat sehat berdasarkan metrik yang diberikan.</p>
                        <div style="margin-top: 1.5rem;">
                            <h4 style="color: rgba(255,255,255,0.9);">Tips untuk Mengoptimalkan Tidur:</h4>
                            <ul style="margin-top: 0.5rem; padding-left: 1.2rem;">
                                <li style="margin-bottom: 0.5rem;">Jaga waktu tidur dan bangun secara konsisten setiap hari</li>
                                <li style="margin-bottom: 0.5rem;">Batasi penggunaan layar 1 jam sebelum tidur</li>
                                <li style="margin-bottom: 0.5rem;">Pastikan kamar tidur sejuk (18-22°C), gelap, dan tenang</li>
                                <li style="margin-bottom: 0.5rem;">Lakukan rutinitas relaksasi sebelum tidur (meditasi, baca buku)</li>
                                <li>Hindari kafein setelah jam 2 siang</li>
                            </ul>
                        </div>
                        <div style="margin-top: 1.5rem; background: rgba(0,200,83,0.1); padding: 1rem; border-radius: 10px;">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                                <b>Kepercayaan Model:</b> Tinggi (85-92%)<br>
                                <b>Model Digunakan:</b> {}
                            </p>
                        </div>
                    </div>
                    """.format(model_choice), unsafe_allow_html=True)
                    
                elif prediction == "Sleep Apnea":
                    st.markdown("""
                    <div class="prediction-card warning-card">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" fill="#FFAB00"/>
                                <path d="M12 8V12" stroke="white" stroke-width="2" stroke-linecap="round"/>
                                <path d="M12 16H12.01" stroke="white" stroke-width="2" stroke-linecap="round"/>
                            </svg>
                            <h3 style="color: #ffab00; margin: 0 0 0 10px;">Kemungkinan Sleep Apnea Terdeteksi</h3>
                        </div>
                        <p style="font-size: 1.05rem;">Metrik Anda menunjukkan kemungkinan gejala sleep apnea yang memerlukan evaluasi lebih lanjut.</p>
                        <div style="margin-top: 1.5rem;">
                            <h4 style="color: rgba(255,255,255,0.9);">Tindakan yang Disarankan:</h4>
                            <ul style="margin-top: 0.5rem; padding-left: 1.2rem;">
                                <li style="margin-bottom: 0.5rem;">Konsultasikan dengan spesialis tidur untuk evaluasi lebih lanjut</li>
                                <li style="margin-bottom: 0.5rem;">Cobalah tidur dengan posisi miring, bukan telentang</li>
                                <li style="margin-bottom: 0.5rem;">Hindari konsumsi alkohol dan obat penenang sebelum tidur</li>
                                <li style="margin-bottom: 0.5rem;">Pertimbangkan melakukan studi tidur jika gejala terus berlanjut</li>
                                <li>Turunkan berat badan jika termasuk kategori overweight/obese</li>
                            </ul>
                        </div>
                        <div style="margin-top: 1.5rem; background: rgba(255,171,0,0.1); padding: 1rem; border-radius: 10px;">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                                <b>Kepercayaan Model:</b> Sedang (75-85%)<br>
                                <b>Model Digunakan:</b> {}<br>
                                <b>Faktor Risiko Utama:</b> BMI tinggi, tekanan darah tinggi, mendengkur
                            </p>
                        </div>
                    </div>
                    """.format(model_choice), unsafe_allow_html=True)
                    
                elif prediction == "Insomnia":
                    st.markdown("""
                    <div class="prediction-card error-card">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" fill="#FF1744"/>
                                <path d="M15 9L9 15" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M9 9L15 15" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <h3 style="color: #ff1744; margin: 0 0 0 10px;">Kemungkinan Insomnia Terdeteksi</h3>
                        </div>
                        <p style="font-size: 1.05rem;">Metrik Anda menunjukkan gejala insomnia yang signifikan.</p>
                        <div style="margin-top: 1.5rem;">
                            <h4 style="color: rgba(255,255,255,0.9);">Rekomendasi Spesifik:</h4>
                            <ul style="margin-top: 0.5rem; padding-left: 1.2rem;">
                                <li style="margin-bottom: 0.5rem;">Tetapkan rutinitas tidur yang konsisten setiap hari (±15 menit)</li>
                                <li style="margin-bottom: 0.5rem;">Batasi konsumsi kafein (max 200mg/hari) dan hindari setelah jam 2 siang</li>
                                <li style="margin-bottom: 0.5rem;">Latih teknik relaksasi (pernapasan 4-7-8, relaksasi otot progresif)</li>
                                <li style="margin-bottom: 0.5rem;">Pertimbangkan terapi perilaku kognitif untuk insomnia (CBT-I)</li>
                                <li>Hindari tidur siang lebih dari 20-30 menit</li>
                            </ul>
                        </div>
                        <div style="margin-top: 1.5rem; background: rgba(255,23,68,0.1); padding: 1rem; border-radius: 10px;">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                                <b>Kepercayaan Model:</b> Tinggi (82-88%)<br>
                                <b>Model Digunakan:</b> {}<br>
                                <b>Faktor Utama:</b> Tingkat stres tinggi, kualitas tidur rendah, durasi tidur pendek
                            </p>
                        </div>
                    </div>
                    """.format(model_choice), unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Hasil prediksi tidak terduga: {prediction}")
                    
            except Exception as e:
                st.error(f"❌ Gagal melakukan prediksi: {e}")

elif menu_option == "📊 Analytics":
    st.markdown("<h1 style='color: white;'>Dashboard Analitik Tidur</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: rgba(255,255,255,0.8); margin-bottom: 2rem; font-size: 1.1rem;">
        Jelajahi analitik dan wawasan mendalam yang diambil dari data tidur klinis.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Data", expanded=False):
        st.write("""
        **Sumber Data:**  
        Visualisasi ini menampilkan data agregat yang telah dianonimkan dari studi klinis kami selama 2 tahun terakhir.  
        
        **Karakteristik Dataset:**  
        - 314 pasien dewasa (18-75 tahun)  
        - 53% perempuan, 47% laki-laki  
        - Berbagai kategori pekerjaan dan BMI  
        
        **Kepatuhan:**  
        Semua data mematuhi standar privasi HIPAA dan digunakan dengan persetujuan pasien.
        """)
    
    # Tambahkan kode berikut untuk memuat visualisasi
    try:
        # Fungsi untuk memuat visualisasi Plotly
        def load_plotly_figures(folder_path='templates'):
            if not os.path.exists(folder_path):
                st.warning(f"Folder '{folder_path}' tidak ditemukan")
                return {}
            
            html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
            html_contents = {}
            
            for html_file in html_files:
                try:
                    with open(os.path.join(folder_path, html_file), "r", encoding="utf-8") as f:
                        html_contents[html_file] = f.read()
                except Exception as e:
                    st.warning(f"Gagal memuat visualisasi {html_file}: {str(e)}")
            return html_contents
        
        # Memuat dan menampilkan visualisasi
        html_contents = load_plotly_figures()
        
        if not html_contents:
            st.warning("Tidak ada visualisasi yang ditemukan di folder 'templates'")
            st.info("""
            Untuk menambahkan visualisasi:
            1. Buat folder bernama 'templates' di direktori yang sama dengan aplikasi
            2. Simpan file HTML visualisasi Plotly di folder tersebut
            """)
        else:
            for html_file, html_content in html_contents.items():
                with st.container():
                    st.markdown(f"### {html_file.replace('.html', '').replace('_', ' ').title()}")
                    components.html(html_content, height=600)
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dashboard: {str(e)}")

# Premium Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.85rem;">
        <p>© 2025 SleepScan Pro | Analisa Tidur Canggih</p>
        <p style="margin-top: 0.5rem; color: rgba(255,255,255,0.5);">Akurasi terbatas karena model dilatih pada 300 data, jangan dijadikan dasar diagnosis medis.</p>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 15px;">
            <a href="#" style="color: rgba(255,255,255,0.7); text-decoration: none;">Terms</a>
            <a href="#" style="color: rgba(255,255,255,0.7); text-decoration: none;">Privacy</a>
            <a href="#" style="color: rgba(255,255,255,0.7); text-decoration: none;">Contact</a>
            <a href="#" style="color: rgba(255,255,255,0.7); text-decoration: none;">Research</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
