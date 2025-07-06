# %%
import pandas as pd
import os

# Fungsi untuk memuat data CSV dengan pengecekan error dan menggunakan encoding ISO-8859-1
def load_data(file_path, encoding="ISO-8859-1"):
    """
    Fungsi untuk memuat data dari file CSV.
    """
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            print(f"Error membaca file {file_path}: {e}")
            return None
    else:
        print(f"File tidak ditemukan: {file_path}")
        return None


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk mengecek dan menangani missing values
def handle_missing_values(df):
    """
    Menangani missing values dengan mengisi nilai kosong menggunakan rata-rata kolom.
    """
    print("Missing values per kolom:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Visualisasi missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    # Mengisi missing values dengan rata-rata kolom numerik
    df_filled = df.fillna(df.mean())
    return df_filled


# %%
def show_basic_info(df):
    """
    Menampilkan informasi dasar dataset seperti 5 baris pertama dan tipe data.
    """
    print("Data berhasil dimuat dan digabungkan. Berikut adalah 5 baris pertama:")
    print(df.head())

    print("Informasi dataset:")
    print(df.info())


# %%
# Fungsi untuk memuat semua data
def load_all_data():
    """
    Memuat seluruh data dari beberapa file CSV.
    """
    data_2021 = load_data("./cpdg2021.csv")
    data_2022 = load_data("./cpdg2022.csv")
    data_2023 = load_data("./cpdg2023.csv")
    data_2024 = load_data("./cpdg2024.csv")
    
    datasets = [data_2021, data_2022, data_2023, data_2024]
    
    # Memeriksa apakah ada DataFrame kosong
    if any(dataset is None or dataset.empty for dataset in datasets):
        print("Beberapa file gagal dimuat atau kosong. Proses dihentikan.")
        return None
    else:
        # Gabungkan semua dataset menjadi satu DataFrame
        df = pd.concat(datasets, ignore_index=True)
        return df


# %%
import pandas as pd

# Menentukan file path
file_path = "./cpdg2021.csv"

# Coba untuk membaca file dengan lebih banyak pengaturan
try:
    df = pd.read_csv(file_path, quotechar='"', sep=";", encoding="ISO-8859-1")
    print(df.head())  # Menampilkan 5 baris pertama
except Exception as e:
    print(f"Error membaca file {file_path}: {e}")


# %%
import pandas as pd

# Menangani tanda kutip dalam data
def clean_data(df):
    # Menghapus tanda kutip ganda atau karakter yang tidak diinginkan
    for column in df.columns:
        df[column] = df[column].str.replace('"', '', regex=True)
    return df

try:
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df = clean_data(df)  # Pembersihan tanda kutip jika ada
    print(df.head())
except Exception as e:
    print(f"Error membaca file {file_path}: {e}")


# %%
# Fungsi utama untuk menjalankan seluruh proses
def main():
    # Memuat semua data
    df = load_all_data()
    
    if df is not None:
        # Menampilkan informasi dasar dataset
        show_basic_info(df)
        
        # Menangani missing values
        df_filled = handle_missing_values(df)
        
        # Tampilkan dataset setelah pengisian missing values
        print("Dataset setelah pengisian missing values:")
        print(df_filled.head())

# Jalankan fungsi utama
main()



