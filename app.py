from flask import Flask, request, jsonify
import numpy as np
import joblib
import csv
import os
from datetime import datetime
from collections import deque
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================= LOAD MODEL =================
lstm_model = load_model("air_quality_lstm_model.h5")
scaler = joblib.load("scaler.save")

sequence_length = 10  # phải giống lúc train

feature_names = ["Temperature", "Humidity", "PM2.5", "CO2", "TVOC", "CO"]

data_buffer = deque(maxlen=sequence_length)

# ================= LABEL MAP =================
label_map = {
    0: "Good",
    1: "Moderate",
    2: "Poor"
}

# ================= LOG FILE =================
LOG_FILE = "air_quality_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "Temperature",
            "Humidity",
            "PM2.5",
            "CO2",
            "TVOC",
            "CO",
            "prediction"
        ])

# ================= ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # ===== Nhận dữ liệu từ ESP32 =====
        temperature = float(data["Temperature"])
        humidity = float(data["Humidity"])
        pm25 = float(data["PM2.5"])
        co2 = float(data["CO2"])
        tvoc = float(data["TVOC"])
        co = float(data["CO"])

        features = [
            temperature,
            humidity,
            pm25,
            co2,
            tvoc,
            co
        ]

        # ===== Chuẩn hóa =====
        features_scaled = scaler.transform([features])[0]

        # ===== Thêm vào buffer =====
        data_buffer.append(features_scaled)

        if len(data_buffer) < sequence_length:
            return jsonify({"status": "waiting for more data"})

        # ===== Tạo input LSTM =====
        input_sequence = np.array(data_buffer)
        input_sequence = np.expand_dims(input_sequence, axis=0)

        prediction = lstm_model.predict(input_sequence)
        predicted_class = np.argmax(prediction)
        prediction_label = label_map[predicted_class]

        # ===== LƯU LOG =====
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                temperature,
                humidity,
                pm25,
                co2,
                tvoc,
                co,
                prediction_label
            ])

        return jsonify({
            "prediction": prediction_label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= RUN =================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
