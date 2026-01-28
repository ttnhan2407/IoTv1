#include <HardwareSerial.h>
#include <DHT.h>
#include <math.h>

// ================== PIN CONFIG ==================
#define DHTPIN 26
#define DHTTYPE DHT22
#define MQ135_PIN 32
#define DUST_LED_PIN 25
#define DUST_SENSOR_PIN 34
#define SSAM_RX_PIN 16
#define SSAM_TX_PIN 17

DHT dht(DHTPIN, DHTTYPE);
HardwareSerial ssam(2);

#define SAMPLE_COUNT 10

// ================== JW01 ==================
const uint8_t FRAME_LEN = 9;
const uint8_t SOF1 = 0x2C;
const uint8_t SOF2 = 0xE4;
uint8_t buf[FRAME_LEN];
uint8_t idx = 0;
bool inFrame = false;

// ================== SCALER ==================
float mean_vals[6] = {40.11927982, 32.80750188, 55.38184546,
                      650.11327832, 37.63315829, 14.77569392};

float std_vals[6]  = {2.77893082, 0.98171008, 9.54433099,
                      147.20558343, 22.56306681, 72.15103528};

// ================== WEIGHTS ==================

float W1[6][16] = {
{-0.184627,-0.829904,-0.214566,0.011538,0.246981,-0.038216,-0.134606,0.113448,-0.232328,0.165485,-0.411465,-0.450981,-0.376153,0.589515,-0.004987,-0.127511},
{-0.259812,0.722922,0.385542,0.028737,0.086238,-0.422099,-0.49175,0.441256,-0.176191,-0.356138,-0.081686,-0.41763,-0.225251,0.55833,-0.158185,-0.382126},
{0.604825,-0.014608,-0.607051,-0.375816,0.61132,0.161344,-0.637454,0.491605,0.733096,-0.052215,0.10353,0.057153,-0.108321,-0.043038,-0.136572,0.114792},
{0.06654,0.226078,0.486641,-0.114725,-0.365188,-0.116869,-0.344208,0.36362,-0.332942,-0.324702,0.043053,0.475487,-0.494153,-0.023047,0.473107,0.791673},
{-0.005274,-0.065514,-0.349654,-0.300955,0.144299,0.686084,0.179298,0.853362,0.217252,0.114803,-0.193194,0.325444,-0.671347,0.008915,0.507448,0.834229},
{-0.549903,0.575702,0.243251,-1.423541,-0.76033,-0.105165,-0.684342,0.373355,-0.913303,1.152408,-0.287969,-0.079764,0.653356,-0.025921,0.619307,0.707827}
};

float b1[16] = {
0.419142,0.206553,0.333462,0.60786,0.439479,0.102079,0.626089,0.499205,
0.585425,0.302085,0.314504,-0.147743,0.346266,0.330833,-0.29063,-0.339491
};

float W2[16][8] = {
{-0.136402,-0.445048,-0.111479,0.751866,0.178338,0.225416,-0.533526,0.285784},
{-0.128043,0.366167,-0.540183,0.194384,-0.353236,-0.050019,-0.453613,0.123317},
{0.226556,0.433571,0.357939,0.094082,-0.167333,0.456603,0.049403,-0.67986},
{-0.581973,-0.783818,0.125509,0.802256,-0.166631,0.552201,0.385973,-0.557684},
{0.021772,-0.322485,-0.107447,0.327874,-1.01073,0.502972,-0.167833,-0.477026},
{0.054555,0.410812,-0.541164,-0.093259,0.235077,0.308932,0.218556,-0.246997},
{-0.032256,-0.203056,0.418731,0.961188,-0.837535,0.629611,0.050356,-0.382929},
{-0.019497,0.590909,0.061094,0.511113,0.148465,-0.032971,-0.457701,-0.04348},
{-0.286382,0.010557,0.345508,1.126558,-0.133525,0.64913,0.304415,0.163784},
{0.499886,0.700766,0.007404,-0.03266,-0.742245,-0.555567,-0.428629,-0.200536},
{-0.045467,0.046096,-0.419975,0.156826,-1.098924,0.558556,-0.374437,0.093364},
{-0.09547,0.095198,-0.017761,0.067499,0.150232,-0.084617,-0.465543,0.492719},
{0.82683,0.767374,-0.231499,-0.350848,-0.528901,-0.104301,0.049121,-0.38577},
{0.218494,0.126769,0.088116,0.069597,-0.725453,0.430121,-0.251429,-0.007882},
{0.069417,0.384508,-0.523734,-0.545929,0.708551,-0.75539,0.081,-0.057603},
{-0.021839,0.830065,0.25164,-0.550647,0.301411,-0.755273,0.200701,-0.222243}
};

float b2[8] = {
0.208363,0.128146,0.167999,0.648099,-0.370415,0.276844,-0.10197,-0.05989
};

float W3[8][3] = {
{-0.356864,1.090148,-0.467943},
{-1.333861,0.743158,0.101601},
{0.6914,-0.515614,-0.8049},
{0.298175,-0.823841,-1.739972},
{-0.305634,-0.870196,0.922987},
{1.056563,-1.009702,-0.938894},
{-0.345168,-0.723215,0.632622},
{-0.780267,0.741205,0.543997}
};

float b3[3] = {0.148441,0.593835,-0.588131};


// ⚠️ DÁN TOÀN BỘ MA TRẬN W1, W2, W3, b1, b2, b3 Ở ĐÂY
// (Giữ nguyên như mình đã gửi ở tin trước)

// ================== RELU ==================
float relu(float x){ return x > 0 ? x : 0; }

// ================== SOFTMAX ==================
void softmax(float* input, float* output, int size){
  float maxVal = input[0];
  for(int i=1;i<size;i++) if(input[i]>maxVal) maxVal=input[i];

  float sum=0;
  for(int i=0;i<size;i++){
    output[i]=exp(input[i]-maxVal);
    sum+=output[i];
  }
  for(int i=0;i<size;i++)
    output[i]/=sum;
}

// ================== NN ==================
void predict(float inputData[6], float output[3]){
  float l1[16], l2[8], l3[3];

  for(int j=0;j<16;j++){
    l1[j]=b1[j];
    for(int i=0;i<6;i++)
      l1[j]+=inputData[i]*W1[i][j];
    l1[j]=relu(l1[j]);
  }

  for(int j=0;j<8;j++){
    l2[j]=b2[j];
    for(int i=0;i<16;i++)
      l2[j]+=l1[i]*W2[i][j];
    l2[j]=relu(l2[j]);
  }

  for(int j=0;j<3;j++){
    l3[j]=b3[j];
    for(int i=0;i<8;i++)
      l3[j]+=l2[i]*W3[i][j];
  }

  softmax(l3,output,3);
}

// ================== READ MQ ==================
int readAnalogAverage(int pin){
  long sum=0;
  for(int i=0;i<SAMPLE_COUNT;i++){
    sum+=analogRead(pin);
    delay(5);
  }
  return sum/SAMPLE_COUNT;
}

// ================== READ DUST ==================
float readDust(){
  long dustSum=0;
  for(int i=0;i<SAMPLE_COUNT;i++){
    digitalWrite(DUST_LED_PIN,LOW);
    delayMicroseconds(280);
    dustSum+=analogRead(DUST_SENSOR_PIN);
    delayMicroseconds(40);
    digitalWrite(DUST_LED_PIN,HIGH);
    delay(10);
  }
  float raw=dustSum/SAMPLE_COUNT;
  float voltage=raw*(3.3/4095.0);
  float density=(voltage-0.6)/0.5;
  if(density<0) density=0;
  return density*1000;
}

// ================== JW01 ==================
bool validChecksum(){
  uint16_t sum=0;
  for(int i=0;i<FRAME_LEN-1;i++)
    sum+=buf[i];
  return uint8_t(sum&0xFF)==buf[FRAME_LEN-1];
}

void readJW01(float &tvoc,int &co2){
  while(ssam.available()){
    uint8_t b=ssam.read();
    if(!inFrame){
      if(b==SOF1){
        buf[0]=b;
        idx=1;
        inFrame=true;
      }
    }else{
      buf[idx++]=b;
      if(idx==FRAME_LEN){
        inFrame=false;
        if(buf[1]==SOF2 && validChecksum()){
          uint16_t rawTVOC=(buf[2]<<8)|buf[3];
          uint16_t rawCO2=(buf[6]<<8)|buf[7];
          tvoc=rawTVOC*0.001f;
          co2=rawCO2;
        }
        idx=0;
      }
    }
  }
}

// ================== ALERT ==================
unsigned long poorStart=0;
bool poorActive=false;

void setup(){
  Serial.begin(115200);
  dht.begin();
  pinMode(DUST_LED_PIN,OUTPUT);
  digitalWrite(DUST_LED_PIN,HIGH);
  ssam.begin(9600,SERIAL_8N1,SSAM_RX_PIN,SSAM_TX_PIN);
  Serial.println("AIR QUALITY + TinyML RUNNING");
}

void loop(){

  float temperature=dht.readTemperature();
  float humidity=dht.readHumidity();
  int mq135=readAnalogAverage(MQ135_PIN);
  float dust=readDust();

  float tvoc=0;
  int co2=0;
  readJW01(tvoc,co2);

  // ======= Chuẩn hóa =======
  float inputData[6]={
    (mq135-mean_vals[0])/std_vals[0],
    (temperature-mean_vals[1])/std_vals[1],
    (humidity-mean_vals[2])/std_vals[2],
    (co2-mean_vals[3])/std_vals[3],
    (tvoc-mean_vals[4])/std_vals[4],
    (dust-mean_vals[5])/std_vals[5]
  };

  float output[3];
  predict(inputData,output);

  int label=0;
  float maxVal=output[0];
  for(int i=1;i<3;i++)
    if(output[i]>maxVal){ maxVal=output[i]; label=i; }

  // ======= HIỂN THỊ =======
  Serial.println("\n===== SENSOR DATA =====");
  Serial.print("MQ135: "); Serial.println(mq135);
  Serial.print("Temp: "); Serial.println(temperature);
  Serial.print("Humidity: "); Serial.println(humidity);
  Serial.print("CO2: "); Serial.println(co2);
  Serial.print("TVOC: "); Serial.println(tvoc);
  Serial.print("Dust: "); Serial.println(dust);

  Serial.print("GOOD: "); Serial.print(output[0],4);
  Serial.print(" | MODERATE: "); Serial.print(output[1],4);
  Serial.print(" | POOR: "); Serial.println(output[2],4);

  if(label==0) Serial.println("Status: GOOD");
  else if(label==1) Serial.println("Status: MODERATE");
  else Serial.println("Status: POOR");

  // ======= CẢNH BÁO 1 PHÚT =======
  if(label==2){
    if(!poorActive){
      poorStart=millis();
      poorActive=true;
    }
    if(millis()-poorStart>60000){
      Serial.println("⚠️ POOR LEVEL CONFIRMED > 1 MINUTE");
    }
  }
  else{
    poorActive=false;
  }

  delay(3000);
}
