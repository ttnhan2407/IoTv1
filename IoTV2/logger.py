import serial
import csv
import time
from datetime import datetime

# ======= CHỈNH COM PORT =======
SERIAL_PORT = "COM5"   # đổi đúng COM của bạn
BAUD_RATE = 115200

filename = f"air_log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Logging started...")
print("Saving to:", filename)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Ghi header chuẩn
    writer.writerow([
        "datetime",
        "time_ms",
        "temperature",
        "humidity",
        "mq135",
        "co2",
        "tvoc",
        "dust",
        "label"
    ])

    while True:
        try:
            line = ser.readline().decode().strip()

            if not line:
                continue

            if "CSV_HEADER" in line:
                continue

            parts = line.split(",")

            if len(parts) == 8:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([now] + parts)
                file.flush()
                print("Saved:", parts)

        except KeyboardInterrupt:
            print("Stopped logging.")
            break
        except Exception as e:
            print("Error:", e)
