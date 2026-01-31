import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATASET
# =========================

file_path = "indoor_data.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# 2. CHỌN 6 FEATURE
# =========================
# field2 → MQ135
# field3 → Temperature
# field4 → Humidity
# field5 → CO2 (JW01)
# field6 → TVOC (JW01)
# field7 → Dust (GP2Y1014AU)

X = df[['field2','field3','field4','field5','field6','field7']]

# =========================
# 3. TẠO LABEL (CẢI TIẾN)
# =========================

def create_label(row):
    co2 = row['field5']
    dust = row['field7']
    tvoc = row['field6']
    mq135 = row['field2']
    temp = row['field3']
    hum  = row['field4']

    score = 0

    # ===== CO2 =====
    if co2 >= 1200:
        score += 2
    elif co2 >= 800:
        score += 1

    # ===== Dust =====
    if dust >= 75:
        score += 2
    elif dust >= 35:
        score += 1

    # ===== TVOC =====
    if tvoc >= 600:
        score += 2
    elif tvoc >= 300:
        score += 1

    # ===== MQ135 =====
    if mq135 > 300:
        score += 1

    # ===== Temperature =====
    if temp < 18 or temp > 32:
        score += 1

    # ===== Humidity =====
    if hum < 30 or hum > 70:
        score += 1

    # ===== Final Class =====
    if score <= 2:
        return 0   # Good
    elif score <= 5:
        return 1   # Moderate
    else:
        return 2   # Poor

df['label'] = df.apply(create_label, axis=1)
y = df['label']

print("\nLabel distribution:")
print(df['label'].value_counts())

# =========================
# 4. CHUẨN HÓA
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n===== COPY CHO ESP32 =====")
print("MEAN =", scaler.mean_)
print("STD  =", scaler.scale_)

# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 6. BUILD MODEL (TỐI ƯU NHẸ CHO ESP32)
# =========================

model = keras.Sequential([
    keras.layers.Dense(12, activation='relu', input_shape=(6,)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 7. TRAIN
# =========================

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =========================
# 8. EVALUATE
# =========================

loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# =========================
# 9. VẼ ĐỒ THỊ
# =========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.show()

# =========================
# 10. CONVERT TFLITE (TinyML)
# =========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("air_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nTFLite model saved as air_model.tflite")

# =========================
# 11. EXPORT WEIGHTS (CHO C THUẦN)
# =========================

weights = model.get_weights()

print("\n================ EXPORT FOR ARDUINO ================")

layer_names = [
    "W1 (6x12)", "b1 (12)",
    "W2 (12x6)", "b2 (6)",
    "W3 (6x3)", "b3 (3)"
]

for name, w in zip(layer_names, weights):
    print("\n", name)
    print(np.round(w, 6))

print("\n===== TRAINING HOÀN THÀNH =====")
