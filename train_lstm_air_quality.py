import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("IoT_Indoor_Air_Quality_Dataset.csv")

# Xóa khoảng trắng tên cột
df.columns = df.columns.str.strip()

# Đổi tên cột cho dễ dùng
df.rename(columns={
    'Temperature (?C)': 'Temperature',
    'Humidity (%)': 'Humidity',
    'CO2 (ppm)': 'CO2',
    'PM2.5 (?g/m?)': 'PM2.5',
    'PM10 (?g/m?)': 'PM10',
    'TVOC (ppb)': 'TVOC',
    'CO (ppm)': 'CO'
}, inplace=True)

print("Columns after rename:")
print(df.columns)

# Bỏ giá trị thiếu
df = df.dropna()

# =====================================
# 2. TẠO NHÃN 0-Good / 1-Moderate / 2-Poor
# =====================================

def label_air_quality(row):
    if (row['PM2.5'] <= 12 and
        row['CO2'] <= 1000 and
        row['TVOC'] <= 220 and
        row['CO'] <= 4.4):
        return 0  # Good
    
    elif (row['PM2.5'] > 35 or
          row['CO2'] > 1500 or
          row['TVOC'] > 400 or
          row['CO'] > 9):
        return 2  # Poor
    
    else:
        return 1  # Moderate

df['Air_Quality_Label'] = df.apply(label_air_quality, axis=1)

print("\nClass distribution:")
print(df['Air_Quality_Label'].value_counts())

# =====================================
# 3. CHỌN FEATURE
# =====================================

features = ['Temperature', 'Humidity', 'PM2.5', 'CO2', 'TVOC', 'CO']
data = df[features].values
labels = df['Air_Quality_Label'].values

# =====================================
# 4. SCALE DATA
# =====================================

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

joblib.dump(scaler, "scaler.save")

# =====================================
# 5. TẠO SEQUENCE CHO LSTM
# =====================================

sequence_length = 10

X, y = [], []

for i in range(sequence_length, len(data)):
    X.append(data[i-sequence_length:i])
    y.append(labels[i])

X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=3)

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

# =====================================
# 6. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =====================================
# 7. BUILD MODEL
# =====================================

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================
# 8. TRAIN MODEL
# =====================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=64,
    callbacks=[early_stop]
)

# =====================================
# 9. EVALUATE
# =====================================

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# =====================================
# 10. SAVE MODEL
# =====================================

model.save("air_quality_lstm_model.h5")

print("\nModel saved as air_quality_lstm_model.h5")
print("Scaler saved as scaler.save")
