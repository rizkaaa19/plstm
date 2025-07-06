# %%
import chardet

files = ['cpdg2021.csv', 'cpdg2022.csv', 'cpdg2023.csv', 'cpdg2024.csv']

for file in files:
    with open(file, 'rb') as f:
        result = chardet.detect(f.read(100000))
        print(f"{file}: {result['encoding']}")


# %%
import pandas as pd

files = ['cpdg2021.csv', 'cpdg2022.csv', 'cpdg2023.csv', 'cpdg2024.csv']
dfs = []

for file in files:
    df = pd.read_csv(file, encoding='windows-1252', sep=';')  # Gunakan delimiter yang benar
    dfs.append(df)

# Gabungkan semua data menjadi satu DataFrame
df_final = pd.concat(dfs, ignore_index=True)

# Tampilkan beberapa baris pertama untuk memastikan data terbaca dengan benar
print(df_final.head())



# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi 1: Histogram untuk distribusi ketinggian ortometrik
plt.figure(figsize=(8, 5))
sns.histplot(df_final['Ortho. Height [m]'], bins=30, kde=True)
plt.title("Distribusi Ketinggian Ortometrik")
plt.xlabel("Ortho. Height [m]")
plt.ylabel("Frekuensi")
plt.grid()
plt.show()

# Visualisasi 2: Scatter plot untuk hubungan antara Easting dan Northing
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_final['Easting [m]'], y=df_final['Northing [m]'], alpha=0.5)
plt.title("Scatter Plot: Easting vs Northing")
plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
plt.grid()
plt.show()


# %%
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9\s\[\]._-]', '', regex=True)
print("Kolom setelah dibersihkan:", df.columns)


# %%
df = df[['Point Id', 'Start Time', 'WGS84 Latitude []', 'WGS84 Longitude []', 
         'Easting [m]', 'Northing [m]', 'Ortho. Height [m]', 
         'WGS84 Ellip. Height [m]', 'Geoid Separation [m]']]


# %%
print(df.columns.tolist())  # Lihat semua nama kolom setelah dibersihkan


# %%
print(df.columns.tolist())


# %%
df = df[['Point Id', 'Start Time', 'WGS84 Latitude []', 'WGS84 Longitude []', 
         'Easting [m]', 'Northing [m]', 'Ortho. Height [m]', 
         'WGS84 Ellip. Height [m]', 'Geoid Separation [m]']]


# %%
df.columns = df.columns.str.strip()  # Hilangkan spasi berlebih
df.columns = df.columns.str.replace(r'[^\w\s\[\]]', '', regex=True)  # Bersihkan karakter aneh
print(df.columns.tolist())  # Cek lagi nama kolom setelah dibersihkan


# %%
df = df[['Point Id', 'Start Time', 'WGS84 Latitude []', 'WGS84 Longitude []', 
         'Easting [m]', 'Northing [m]', 'Ortho Height [m]', 
         'WGS84 Ellip Height [m]', 'Geoid Separation [m]']]


# %%
# Cek ringkasan statistik
print(df.describe())

# Cek missing values
print(df.isnull().sum())


# %%
import re

def dms_to_dd(dms_str):
    """Konversi koordinat dari format DMS ke Decimal Degrees (DD)"""
    dms_str = dms_str.strip()
    match = re.match(r'(\d+)° (\d+)\'.*?([\d.]+)\" (\w)', dms_str)
    
    if match:
        degrees, minutes, seconds, direction = match.groups()
        dd = float(degrees) + float(minutes)/60 + float(seconds)/3600
        if direction in ['S', 'W']:  # South dan West negatif
            dd *= -1
        return dd
    else:
        return None  # Kalau format salah, biarkan kosong

# Terapkan ke kolom Latitude & Longitude
df['WGS84 Latitude []'] = df['WGS84 Latitude []'].apply(dms_to_dd)
df['WGS84 Longitude []'] = df['WGS84 Longitude []'].apply(dms_to_dd)

# Cek apakah konversi berhasil
print(df[['WGS84 Latitude []', 'WGS84 Longitude []']].head())


# %%
# Fungsi untuk membersihkan angka yang memiliki format titik pemisah ribuan
def convert_to_float(value):
    try:
        # Hapus pemisah ribuan (titik) lalu ubah ke float
        return float(str(value).replace('.', '').replace(',', '.'))
    except ValueError:
        return None  # Jika gagal, isi dengan NaN

# Terapkan ke semua kolom numerik
numeric_columns = ['Easting [m]', 'Northing [m]', 'Ortho Height [m]', 'WGS84 Ellip Height [m]', 'Geoid Separation [m]']
df[numeric_columns] = df[numeric_columns].applymap(convert_to_float)

# Cek apakah masih ada data string
print(df.dtypes)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Scatterplot lokasi berdasarkan Latitude & Longitude
plt.figure(figsize=(8,6))
plt.scatter(df['WGS84 Longitude []'], df['WGS84 Latitude []'], c='blue', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Peta Persebaran Titik Pengukuran')
plt.grid()
plt.show()

# Heatmap Korelasi Antar Variabel
plt.figure(figsize=(10,6))
sns.heatmap(df.drop(columns=['Point Id', 'Start Time']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Antar Variabel')
plt.show()


# %%
import matplotlib.pyplot as plt

# Scatter plot antara Easting dan Northing (koordinat)
plt.figure(figsize=(8,6))
plt.scatter(df['Easting [m]'], df['Northing [m]'], alpha=0.5, c='blue', edgecolors='k')
plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('Scatter Plot Easting vs Northing')
plt.grid(True)
plt.show()


# %%
import seaborn as sns

# Plot histogram untuk Ortho Height dan WGS84 Ellip Height
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['Ortho Height [m]'], bins=20, kde=True, color='blue')
plt.xlabel('Ortho Height [m]')
plt.ylabel('Frekuensi')
plt.title('Distribusi Ortho Height')

plt.subplot(1,2,2)
sns.histplot(df['WGS84 Ellip Height [m]'], bins=20, kde=True, color='green')
plt.xlabel('WGS84 Ellip Height [m]')
plt.ylabel('Frekuensi')
plt.title('Distribusi WGS84 Ellip Height')

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10,5))

sns.boxplot(data=df[['Ortho Height [m]', 'WGS84 Ellip Height [m]']], palette="Set2")

plt.title('Boxplot Ketinggian')
plt.ylabel('Nilai')
plt.xticks([0, 1], ['Ortho Height [m]', 'WGS84 Ellip Height [m]'])

plt.show()


# %%
plt.figure(figsize=(12,6))

plt.plot(df['Start Time'], df['Ortho Height [m]'], label='Ortho Height', marker='o')
plt.plot(df['Start Time'], df['WGS84 Ellip Height [m]'], label='WGS84 Ellip Height', marker='s')

plt.xlabel('Start Time')
plt.ylabel('Height (m)')
plt.title('Tren Perubahan Ketinggian dari Waktu ke Waktu')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.show()


# %%
from sklearn.cluster import KMeans

# Tentukan jumlah cluster (misalnya 3)
num_clusters = 3

# Ambil data koordinat
X = df[['Easting [m]', 'Northing [m]']]

# Inisialisasi dan jalankan K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plot hasil clustering
plt.figure(figsize=(10,6))
for cluster in range(num_clusters):
    plt.scatter(df[df['Cluster'] == cluster]['Easting [m]'],
                df[df['Cluster'] == cluster]['Northing [m]'], label=f'Cluster {cluster}')

# Tandai pusat cluster
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            color='black', marker='X', s=200, label='Centroids')

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('Clustering Titik Lokasi')
plt.legend()
plt.grid()
plt.show()


# %%
print(df['Start Time'].head(20))  # Menampilkan 20 data pertama


# %%
df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')


# %%
print(df[df['Start Time'].isna()])


# %%
df = df.sort_values(by='Start Time')


# %%
import pandas as pd

# Konversi 'Start Time' ke format datetime
df['Start Time'] = pd.to_datetime(df['Start Time'])

# Urutkan berdasarkan waktu
df = df.sort_values(by='Start Time')

# Set 'Start Time' sebagai index
df.set_index('Start Time', inplace=True)

# Lihat contoh data setelah diurutkan
print(df.head())


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Ortho Height [m]'], label='Ortho Height', color='blue')
plt.plot(df.index, df['WGS84 Ellip Height [m]'], label='Ellip Height', color='red', linestyle='dashed')

plt.xlabel('Time')
plt.ylabel('Height (m)')
plt.title('Tren Perubahan Ketinggian')
plt.legend()
plt.grid()
plt.show()


# %%
correlation = df[['Ortho Height [m]', 'WGS84 Ellip Height [m]']].corr()
print(correlation)


# %%
# Menambahkan kolom lain ke dalam analisis korelasi
correlation = df[['Ortho Height [m]', 'WGS84 Ellip Height [m]', 'Easting [m]', 'Northing [m]', 'Geoid Separation [m]']].corr()

print(correlation)


# %%
import matplotlib.pyplot as plt

# Visualisasi perubahan ketinggian dan posisi seiring waktu
plt.figure(figsize=(14, 8))

# Plot untuk perubahan ketinggian
plt.subplot(2, 2, 1)
plt.plot(df.index, df['Ortho Height [m]'], label='Ortho Height', color='blue')
plt.plot(df.index, df['WGS84 Ellip Height [m]'], label='Ellip Height', color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Height (m)')
plt.title('Tren Perubahan Ketinggian')
plt.legend()
plt.grid()

# Plot untuk perubahan Easting
plt.subplot(2, 2, 2)
plt.plot(df.index, df['Easting [m]'], label='Easting [m]', color='green')
plt.xlabel('Time')
plt.ylabel('Easting [m]')
plt.title('Tren Perubahan Easting')
plt.legend()
plt.grid()

# Plot untuk perubahan Northing
plt.subplot(2, 2, 3)
plt.plot(df.index, df['Northing [m]'], label='Northing [m]', color='purple')
plt.xlabel('Time')
plt.ylabel('Northing [m]')
plt.title('Tren Perubahan Northing')
plt.legend()
plt.grid()

# Plot untuk perubahan Geoid Separation
plt.subplot(2, 2, 4)
plt.plot(df.index, df['Geoid Separation [m]'], label='Geoid Separation [m]', color='orange')
plt.xlabel('Time')
plt.ylabel('Geoid Separation [m]')
plt.title('Tren Perubahan Geoid Separation')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Misalkan kita memiliki data yang disiapkan dalam bentuk DataFrame df
# Pastikan data sudah disortir berdasarkan waktu (timestamp)

# Normalisasi data (contoh menggunakan MinMaxScaler)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Ortho Height [m]', 'WGS84 Ellip Height [m]', 'Easting [m]', 'Northing [m]', 'Geoid Separation [m]']])

# Membuat generator untuk data time-series
look_back = 10  # Menggunakan 10 data sebelumnya untuk memprediksi data berikutnya
generator = TimeseriesGenerator(scaled_data, scaled_data, length=look_back, batch_size=32)

# Membangun model LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(look_back, scaled_data.shape[1])))
model.add(Dropout(0.2))  # Menghindari overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(scaled_data.shape[1]))  # Output layer untuk setiap variabel yang diprediksi
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(generator, epochs=50, verbose=1)

# Prediksi untuk data uji (test data)
predictions = model.predict(generator)

# Kembalikan prediksi ke skala asli
predictions_rescaled = scaler.inverse_transform(predictions)


# %%
import numpy as np

# Menghitung Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')


# %%
plt.figure(figsize=(12,6))
plt.plot(df.index[look_back:], actual_values, label='Actual Values', color='blue')
plt.plot(df.index[look_back:], predictions_rescaled, label='Predictions', color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Height (m)')
plt.title('Actual vs Predicted Heights')
plt.legend()
plt.grid()
plt.show()


# %%
pip install streamlit


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Menyiapkan data contoh
# Asumsikan 'data' adalah array numpy yang sudah diolah dari CSV kamu
# Contoh data, ubah dengan data asli yang kamu punya
X_train = np.random.randn(100, 10, 4)  # 100 samples, 10 timesteps, 4 fitur (misalnya Easting, Northing, Ortho Height, dll)
y_train = np.random.randn(100, 1)  # Target output (misalnya prediksi penurunan tanah)

# Membangun model PLSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer

# Compile dan latih model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Simpan model ke file
model.save('model_plstm.h5')  # Menyimpan model ke file .h5



