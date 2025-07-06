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
import pandas as pd

# Fungsi untuk membersihkan karakter non-standar
def clean_columns(df):
    # Menghapus karakter kutip ganda dan karakter lain yang mungkin bermasalah
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", '')  # Menghapus kutip ganda atau tanda kutip
    return df

# Fungsi untuk memuat data CSV dengan pengecekan error dan pemisah yang tepat
def load_data(file_path, encoding="ISO-8859-1", sep=";"):
    try:
        # Memuat data dengan pemisah dan encoding yang sesuai
        df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine="python")
        df = clean_columns(df)  # Membersihkan kolom setelah dimuat
        return df
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

# Path file CSV
file_path = "./cpdg2021.csv"

# Memuat data
df = load_data(file_path)

# Menampilkan hasil jika berhasil
if df is not None:
    print(df.head())


# %%
import pandas as pd

# Tentukan tipe data untuk setiap kolom (misalnya, ubah kolom yang berisi string menjadi objek)
def load_data_with_dtype(file_path, encoding="ISO-8859-1", sep=";"):
    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine="python", dtype=str)  # Menetapkan tipe data string untuk semua kolom
        return df
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

# Path file CSV
file_path = "./cpdg2021.csv"

# Memuat data dengan dtype yang disesuaikan
df = load_data_with_dtype(file_path)

# Menampilkan hasil jika berhasil
if df is not None:
    print(df.head())


# %%
import pandas as pd

# Fungsi untuk memuat data dengan pemisah desimal koma
def load_data_decimal_comma(file_path, encoding="ISO-8859-1", sep=";"):
    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine="python", decimal=",")
        return df
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

# Path file CSV
file_path = "./cpdg2021.csv"

# Memuat data dengan pengaturan desimal koma
df = load_data_decimal_comma(file_path)

# Menampilkan hasil jika berhasil
if df is not None:
    print(df.head())


# %%
# Fungsi untuk memeriksa kolom yang berisi teks panjang
def check_for_text_columns(df):
    for column in df.columns:
        # Memeriksa apakah ada nilai teks panjang di kolom
        if df[column].apply(lambda x: isinstance(x, str) and len(x) > 100).any():
            print(f"Kolom '{column}' mengandung teks panjang.")
            
check_for_text_columns(df)


# %%
# Menghapus kolom yang mengandung teks panjang atau yang tidak relevan
def remove_long_text_columns(df):
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, str) and len(x) > 100).any():
            print(f"Menghapus kolom '{column}' karena berisi teks panjang.")
            df.drop(column, axis=1, inplace=True)
    return df

# Hapus kolom teks panjang
df = remove_long_text_columns(df)

# Tampilkan hasil setelah penghapusan
print(df.head())


# %%
# Mengimpor hanya 10 baris pertama dari file untuk debugging
df_sample = pd.read_csv(file_path, nrows=10, encoding="ISO-8859-1", sep=";", engine="python")
print(df_sample)


# %%
# Coba baca file dengan pengecualian baris yang salah
try:
    df = pd.read_csv("./cpdg2021.csv", encoding="ISO-8859-1", error_bad_lines=False)
    print(df.head())  # Menampilkan beberapa baris pertama
except Exception as e:
    print(f"Error: {e}")


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


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
def load_all_data():
    try:
        df1 = pd.read_csv('file1.csv')  # Gantilah dengan path file yang sesuai
        df2 = pd.read_csv('file2.csv')  # Gantilah dengan path file yang sesuai
        df = pd.concat([df1, df2], ignore_index=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Fungsi untuk menangani missing values
def handle_missing_values(df):
    # Menangani missing values dengan strategi yang sesuai
    df_filled = df.fillna(df.mean(numeric_only=True))  # Misalnya, mengisi missing value dengan rata-rata
    return df_filled

# Fungsi untuk menampilkan informasi dasar tentang dataset
def show_basic_info(df):
    print(f"Shape of the dataset: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values count:\n{df.isnull().sum()}")
    print(f"First few rows of the dataset:\n{df.head()}")

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
        
        # Visualisasi distribusi ketinggian
        plt.figure(figsize=(10,6))
        sns.histplot(df_filled['WGS84 Ellip. Height [m]'], kde=True)
        plt.title('Distribusi Ketinggian')
        plt.xlabel('Ketinggian (m)')
        plt.ylabel('Frekuensi')
        plt.show()

# Jalankan fungsi utama
main()



