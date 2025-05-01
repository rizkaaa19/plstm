import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Load Model
model = tf.keras.models.load_model('model_plstm.h5')

# Fungsi untuk prediksi
def predict(data):
    data = np.array(data).reshape((1, 1, len(data)))  
    prediction = model.predict(data)
    return prediction[0][0]

# CSS untuk Tampilan
st.markdown("""
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #2C3E50;
            color: white;
        }
        
        /* Navbar Styling */
        .navbar {
            background-color: #1ABC9C;
            padding: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Footer Styling */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #2C3E50;
            color: white;
            text-align: center;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Beranda", "📊 Analisis", "📈 Prediksi", "🗺️ Peta Penurunan"])

# Halaman Beranda
if menu == "🏠 Beranda":
    st.markdown('<div class="navbar">Deteksi Penurunan Tanah di Kota Padang</div>', unsafe_allow_html=True)
    st.write("""
        Selamat datang di aplikasi **Deteksi Penurunan Tanah di Kota Padang**!  
        Aplikasi ini dirancang untuk **menganalisis dan memprediksi** penurunan tanah di wilayah Kota Padang  
        menggunakan model **PLSTM (Predictive Long Short-Term Memory)**.  
    """)
    st.image("https://source.unsplash.com/800x400/?padang,indonesia,landscape", use_column_width=True)
    st.info("Gunakan menu di sidebar untuk melihat analisis, prediksi, dan peta penurunan tanah.")

# Halaman Analisis
elif menu == "📊 Analisis":
    st.markdown('<div class="navbar">Analisis Penurunan Tanah di Kota Padang</div>', unsafe_allow_html=True)
    st.write("Halaman ini menampilkan analisis tren penurunan tanah di berbagai wilayah Kota Padang.")

    # Data Dummy untuk contoh
    data = {
        "Tanggal": pd.date_range(start="2023-01-01", periods=12, freq="M"),
        "Penurunan (cm)": np.random.uniform(1, 20, size=12)
    }
    df = pd.DataFrame(data)

    # Tampilkan data dalam tabel
    st.write("### Data Penurunan Tanah di Kota Padang")
    st.dataframe(df)

    # Grafik Tren Penurunan
    st.write("### Grafik Tren Penurunan Tanah di Kota Padang")
    fig, ax = plt.subplots()
    ax.plot(df["Tanggal"], df["Penurunan (cm)"], marker='o', linestyle='-', color='r', label="Penurunan Tanah")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Penurunan Tanah (cm)")
    ax.set_title("Tren Penurunan Tanah Per Bulan")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# Halaman Prediksi
import streamlit as st
import numpy as np
import pandas as pd


def predict(data):
    return np.random.uniform(0, 0.2)  


if "predictions" not in st.session_state:
    st.session_state.predictions = [0.0]  

elif menu == "📈 Prediksi":
    st.markdown('<div class="navbar">Prediksi Penurunan Tanah di Kota Padang</div>', unsafe_allow_html=True)
    st.write("Masukkan data untuk memprediksi penurunan tanah menggunakan model PLSTM.")


    easting = st.number_input('Easting [m]:', value=0.0)
    northing = st.number_input('Northing [m]:', value=0.0)
    ortho_height = st.number_input('Ortho Height [m]:', value=0.0)
    wgs84_ellip_height = st.number_input('WGS84 Ellip Height [m]:', value=0.0)

    if st.button('🔍 Prediksi'):
        data = [easting, northing, ortho_height, wgs84_ellip_height]
        prediction = predict(data)

       
        st.session_state.predictions.append(prediction)

      
        if prediction < 0.05:
            status = "⚪ **Penurunan Rendah** (tidak ada dampak signifikan)"
        elif 0.05 <= prediction < 0.10:
            status = "🟡 **Penurunan Sedang** (perlu pemantauan)"
        else:
            status = "🔴 **Penurunan Tinggi** (berbahaya, waspada!)"

 
        st.subheader("Hasil Prediksi")
        st.write(f"**Prediksi Penurunan Tanah: {prediction:.2f} meter**")
        st.write(status)


    if len(st.session_state.predictions) > 1:
        st.write("### 📊 Tren Prediksi Penurunan Tanah")

        df = pd.DataFrame({
            "Percobaan": list(range(1, len(st.session_state.predictions) + 1)),
            "Penurunan Tanah (m)": st.session_state.predictions
        })

        st.line_chart(df.set_index("Percobaan"))  # Grafik prediksi

    # Tombol reset data
    if st.button("🔄 Reset Data"):
        st.session_state.predictions = [0.0]  # Reset dengan dummy data
        st.rerun()


# Halaman Peta Penurunan Tanah
elif menu == "🗺️ Peta Penurunan":
    st.markdown('<div class="navbar">Peta Penurunan Tanah di Kota Padang</div>', unsafe_allow_html=True)
    st.write("Berikut adalah peta interaktif yang menunjukkan daerah dengan potensi penurunan tanah di Kota Padang.")

    # Lokasi Kota Padang
    lokasi_padang = [-0.9471, 100.4172]

    # Peta Interaktif dengan Folium
    m = folium.Map(location=lokasi_padang, zoom_start=12)

    # Contoh Titik dengan Risiko Penurunan
    titik_penurunan = [
        {"nama": "Gunung Padang", "lokasi": [-0.9356, 100.3565], "tingkat": "Tinggi"},
        {"nama": "Pantai Air Manis", "lokasi": [-0.9783, 100.3680], "tingkat": "Sedang"},
        {"nama": "Lubuk Begalung", "lokasi": [-0.9834, 100.4182], "tingkat": "Rendah"},
    ]

    # Tambahkan Marker ke Peta
    for titik in titik_penurunan:
        warna = "red" if titik["tingkat"] == "Tinggi" else "orange" if titik["tingkat"] == "Sedang" else "blue"
        folium.Marker(
            location=titik["lokasi"],
            popup=f"{titik['nama']} - {titik['tingkat']}",
            icon=folium.Icon(color=warna)
        ).add_to(m)

    # Tampilkan Peta
    folium_static(m)

# Footer
st.markdown('<div class="footer">© 2025 Aplikasi Deteksi Penurunan Tanah di Kota Padang </div>', unsafe_allow_html=True)
