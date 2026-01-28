#include <HardwareSerial.h>
#include <DHT.h>

// ================== DHT22 ==================
#define DHTPIN 26
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// ================== MQ135 ==================
#define MQ135_PIN 32

// ================== GP2Y1014AU ==================
#define DUST_LED_PIN 25
#define DUST_SENSOR_PIN 34

// ================== JW01 / SSAM01 ==================
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

// ====== Hàm đọc trung bình ======
int readAnalogAverage(int pin) {
  long sum = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sum += analogRead(pin);
    delay(5);
  }
  return sum / SAMPLE_COUNT;
}

// ====== Checksum ======
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

  tvoc = rawTVOC * 0.001f;
  hcho = rawHCHO * 0.001f;
  co2  = rawCO2;
}

void setup() {
  Serial.begin(115200);

  dht.begin();

  pinMode(DUST_LED_PIN, OUTPUT);
  digitalWrite(DUST_LED_PIN, HIGH);

  ssam.begin(9600, SERIAL_8N1, SSAM_RX_PIN, SSAM_TX_PIN);

  Serial.println("===== AIR QUALITY SYSTEM STARTED =====");
}

void loop() {

  // ================= DHT22 =================
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // ================= MQ135 =================
  int mq135_raw = readAnalogAverage(MQ135_PIN);

  // ================= GP2Y =================
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

  // ================= HIỂN THỊ =================
  Serial.println("\n====== AIR QUALITY DATA ======");

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print("MQ135 Raw: ");
  Serial.println(mq135_raw);

  Serial.print("PM2.5: ");
  Serial.print(dustDensity);
  Serial.println(" ug/m3");

  if (co2 != -1) {
    Serial.print("CO2: ");
    Serial.print(co2);
    Serial.println(" ppm");

    Serial.print("TVOC: ");
    Serial.print(tvoc, 3);
    Serial.println(" mg/m3");

    Serial.print("HCHO: ");
    Serial.print(hcho, 3);
    Serial.println(" mg/m3");
  } else {
    Serial.println("JW01: Waiting data...");
  }

  Serial.println("===============================");

  delay(3000);
}
