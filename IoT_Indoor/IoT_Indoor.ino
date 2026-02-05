#include <WiFi.h>
#include <HTTPClient.h>
#include <HardwareSerial.h>
#include <DHT.h>

// ================== WIFI ==================
const char* ssid = "Nhu Kim";
const char* password = "0776519922";
const char* serverName = "http://192.168.1.45:5000/predict"; 
// ⚠ đổi IP theo máy đang chạy Flask

// ================== DHT22 ==================
#define DHTPIN 26
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// ================== MQ135 ==================
#define MQ135_PIN 32

// ================== GP2Y1014AU ==================
#define DUST_LED_PIN 25
#define DUST_SENSOR_PIN 34

// ================== JW01 ==================
#define SSAM_RX_PIN 16
#define SSAM_TX_PIN 17
HardwareSerial ssam(2);

const uint8_t FRAME_LEN = 9;
const uint8_t SOF1 = 0x2C;
const uint8_t SOF2 = 0xE4;

uint8_t buf[FRAME_LEN];
uint8_t idx = 0;
bool inFrame = false;

#define SAMPLE_COUNT 10

// ====== Đọc trung bình analog ======
int readAnalogAverage(int pin) {
  long sum = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sum += analogRead(pin);
    delay(5);
  }
  return sum / SAMPLE_COUNT;
}

// ====== Checksum JW01 ======
bool validChecksum() {
  uint16_t sum = 0;
  for (uint8_t i = 0; i < FRAME_LEN - 1; i++)
    sum += buf[i];
  return uint8_t(sum & 0xFF) == buf[FRAME_LEN - 1];
}

// ====== Decode JW01 ======
void decodeJW01(float &tvoc, float &hcho, int &co2) {
  uint16_t rawTVOC = (uint16_t(buf[2]) << 8) | buf[3];
  uint16_t rawHCHO = (uint16_t(buf[4]) << 8) | buf[5];
  uint16_t rawCO2  = (uint16_t(buf[6]) << 8) | buf[7];

  tvoc = rawTVOC * 0.001f;   // mg/m3
  hcho = rawHCHO * 0.001f;   // mg/m3
  co2  = rawCO2;             // ppm
}

void setup() {
  Serial.begin(115200);

  dht.begin();

  pinMode(DUST_LED_PIN, OUTPUT);
  digitalWrite(DUST_LED_PIN, HIGH);

  ssam.begin(9600, SERIAL_8N1, SSAM_RX_PIN, SSAM_TX_PIN);

  // ===== WIFI CONNECT =====
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }

  Serial.println("WiFi Connected!");
  Serial.println("===== AIR QUALITY SYSTEM STARTED =====");
}

void loop() {

  // ================= DHT22 =================
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // ================= MQ135 =================
  int mq135_raw = readAnalogAverage(MQ135_PIN);

  // ⚠ scale demo (chưa phải chuẩn khoa học)
  float co_ppm = mq135_raw * (5.0 / 4095.0);

  // ================= GP2Y PM2.5 =================
  long dustSum = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    digitalWrite(DUST_LED_PIN, LOW);
    delayMicroseconds(280);
    dustSum += analogRead(DUST_SENSOR_PIN);
    delayMicroseconds(40);
    digitalWrite(DUST_LED_PIN, HIGH);
    delay(10);
  }

  int dust_raw = dustSum / SAMPLE_COUNT;
  float voltage = dust_raw * (3.3 / 4095.0);
  float dustDensity = (voltage - 0.6) / 0.5;

  if (dustDensity < 0) dustDensity = 0;
  dustDensity *= 1000;  // ug/m3

  // ================= JW01 =================
  float tvoc = -1;
  float hcho = -1;
  int co2 = -1;

  while (ssam.available()) {
    uint8_t b = ssam.read();

    if (!inFrame) {
      if (b == SOF1) {
        buf[0] = b;
        idx = 1;
        inFrame = true;
      }
    } else {
      buf[idx++] = b;
      if (idx == FRAME_LEN) {
        inFrame = false;

        if (buf[1] == SOF2 && validChecksum()) {
          decodeJW01(tvoc, hcho, co2);
        }
        idx = 0;
      }
    }
  }

  float tvoc_ppb = tvoc * 1000.0;  // mg/m3 -> ppb

  // ================= SERIAL DISPLAY =================
  Serial.println("\n====== AIR QUALITY DATA ======");
  Serial.print("Temperature: "); Serial.println(temperature);
  Serial.print("Humidity: "); Serial.println(humidity);
  Serial.print("PM2.5: "); Serial.println(dustDensity);
  Serial.print("CO2: "); Serial.println(co2);
  Serial.print("TVOC (ppb): "); Serial.println(tvoc_ppb);
  Serial.print("CO (ppm approx): "); Serial.println(co_ppm);
  Serial.println("===============================");

  // ================= SEND TO AI SERVER =================
  if (WiFi.status() == WL_CONNECTED && co2 != -1) {

    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"Temperature\":" + String(temperature) + ",";
    json += "\"Humidity\":" + String(humidity) + ",";
    json += "\"PM2.5\":" + String(dustDensity) + ",";
    json += "\"CO2\":" + String(co2) + ",";
    json += "\"TVOC\":" + String(tvoc_ppb) + ",";
    json += "\"CO\":" + String(co_ppm);
    json += "}";

    int httpResponseCode = http.POST(json);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("AI Prediction:");
      Serial.println(response);
    } else {
      Serial.println("Error sending data to AI server");
    }

    http.end();
  }

  delay(5000);
}
