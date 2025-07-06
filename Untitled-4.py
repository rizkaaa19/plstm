# %%
import pandas as pd
import os

# Fungsi untuk memuat data CSV dengan pengecekan error
def load_data(file_path):
    """
    Memuat data dari file CSV dengan encoding ISO-8859-1.
    """
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding="ISO-8859-1")
        except Exception as e:
            print(f"Error membaca file {file_path}: {e}")
            return None
    else:
        print(f"File tidak ditemukan: {file_path}")
        return None

# Memuat semua data dari tahun 2021-2024
data_2021 = load_data("cpdg2021.csv")
data_2022 = load_data("cpdg2022.csv")
data_2023 = load_data("cpdg2023.csv")
data_2024 = load_data("cpdg2024.csv")

# Gabungkan semua dataset menjadi satu DataFrame
datasets = [data_2021, data_2022, data_2023, data_2024]

# Pastikan semua dataset berhasil dimuat sebelum digabungkan
if any(dataset is None or dataset.empty for dataset in datasets):
    print("Beberapa file gagal dimuat atau kosong. Proses dihentikan.")
else:
    df = pd.concat(datasets, ignore_index=True)
    print("Data berhasil digabungkan!")
    print(df.head())  # Tampilkan 5 baris pertama


# %%
# Baca ulang data dengan pemisah titik koma (;)
def load_data_fixed(file_path):
    """
    Memuat data CSV dengan pemisah titik koma (;).
    """
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding="ISO-8859-1", sep=";")
        except Exception as e:
            print(f"Error membaca file {file_path}: {e}")
            return None
    else:
        print(f"File tidak ditemukan: {file_path}")
        return None

# Muat ulang semua data
data_2021 = load_data_fixed("cpdg2021.csv")
data_2022 = load_data_fixed("cpdg2022.csv")
data_2023 = load_data_fixed("cpdg2023.csv")
data_2024 = load_data_fixed("cpdg2024.csv")

# Gabungkan ulang semua dataset
datasets = [data_2021, data_2022, data_2023, data_2024]

if any(dataset is None or dataset.empty for dataset in datasets):
    print("Beberapa file gagal dimuat atau kosong. Periksa kembali.")
else:
    df = pd.concat(datasets, ignore_index=True)
    print("Data berhasil digabungkan setelah diperbaiki!")
    print(df.head())  # Tampilkan 5 baris pertama dengan format yang benar


# %%
import matplotlib.pyplot as plt

# Cek apakah ada kolom "Start Time" untuk melihat distribusi data per tahun
if 'Start Time' in df.columns:
    df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')  # Konversi ke datetime
    df['Year'] = df['Start Time'].dt.year  # Ambil tahun

    plt.figure(figsize=(8, 5))
    df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Data")
    plt.title("Distribusi Data per Tahun")
    plt.show()
else:
    print("Kolom 'Start Time' tidak ditemukan.")


# %%
print(df.isnull().sum())



