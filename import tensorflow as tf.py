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
