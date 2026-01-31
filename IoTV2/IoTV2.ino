#define BLYNK_TEMPLATE_ID "TMPL6TFgOwiX_"
#define BLYNK_TEMPLATE_NAME "IoTv1"
#define BLYNK_AUTH_TOKEN "p7ru9IUTN5X89npP33UMFrjHW60R9UdD"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <HardwareSerial.h>
#include <DHT.h>
#include <math.h>

char ssid[] = "FOREVER";
char pass[] = "12122018";

// ================= PIN =================
#define DHTPIN 26
#define DHTTYPE DHT22
#define MQ135_PIN 32
#define DUST_LED_PIN 25
#define DUST_SENSOR_PIN 34
#define SSAM_RX_PIN 16
#define SSAM_TX_PIN 17

DHT dht(DHTPIN, DHTTYPE);
HardwareSerial ssam(2);
BlynkTimer timer;

// ================= JW01 =================
const uint8_t FRAME_LEN = 9;
const uint8_t SOF1 = 0x2C;
const uint8_t SOF2 = 0xE4;
uint8_t buf[FRAME_LEN];
uint8_t idx = 0;
bool inFrame = false;

// ================= NORMALIZATION =================
float mean[6] = {
  40.11927982,
  32.80750188,
  55.38184546,
  650.11327832,
  37.63315829,
  14.77569392
};

float stdv[6] = {
  2.77893082,
  0.98171008,
  9.54433099,
  147.20558343,
  22.56306681,
  72.15103528
};

// ===== W1 (6x12) =====
float W1[6][12] = {
{ 0.146598, 0.272057, 0.401548,-0.382356, 0.349175, 0.17845 , 0.23029 , 0.391831,-0.353868,-0.132671,-0.194797,-0.012603},
{ 0.250479,-0.367105, 0.184571, 0.317717, 0.379979, 0.181579,-0.749418,-0.274089,-0.377889, 0.51168 ,-0.284979, 0.419092},
{ 0.459803,-0.348514, 0.252365,-0.271471, 0.219689, 0.256609, 0.918369, 0.340687, 0.418771,-0.210492,-0.65339 , 0.231643},
{ 0.717148,-0.169457, 0.101977,-0.177637,-0.354602, 0.014137,-0.486095,-0.711879, 0.533175,-0.215852, 0.097415,-0.536624},
{ 0.572225, 0.36302 ,-0.315951, 0.172212, 0.330333,-0.385581,-0.251075, 0.348865, 0.330801,-0.304392,-0.50529 , 0.4795  },
{-0.029227,-0.994559, 0.647148,-0.692757,-0.820703,-0.786743,-0.261906, 0.193094,-0.856346,-0.736912, 0.436381,-0.048977}
};

float b1[12] = {
0.129865,0.688476,0.168968,0.282566,0.322997,0.644629,
0.279677,0.218509,0.333209,0.507396,0.275206,-0.147923
};

// ===== W2 (12x6) =====
float W2[12][6] = {
{-0.086432, 0.601979,-0.064888, 0.375804, 0.734324,-0.351104},
{ 0.856514,-0.985408,-0.526612, 0.294824,-0.126225,-0.06833 },
{-0.427965, 1.057272,-0.40848 , 0.127072, 0.837159,-0.069803},
{ 0.954202,-0.198496,-0.328501, 0.404243,-0.268897,-0.446974},
{ 0.843496,-0.293724,-0.012742, 0.401675,-0.77036 , 0.221157},
{ 0.71465 ,-0.98794 , 0.15671 , 0.461155,-0.470399,-0.478237},
{ 0.399834,-0.358163, 0.302483, 0.422994,-0.330477,-0.430886},
{ 0.168296, 0.76071 ,-0.648736, 0.247846,-0.04178 ,-0.023656},
{ 0.880935,-0.058979, 0.017464, 0.878651,-0.481034, 0.030033},
{ 0.303625, 0.262646, 0.245112, 0.861926,-0.512774,-0.014817},
{-0.059054, 0.580558, 0.271263, 0.146909,-0.078073,-0.177051},
{ 0.121852,-0.283288, 0.376258, 0.102798,-0.344099, 0.029616}
};

float b2[6] = {0.204242,0.785727,-0.079803,0.082079,0.13145,-0.010906};

// ===== W3 (6x3) =====
float W3[6][3] = {
{ 1.316938,-1.437466,-1.068521},
{-0.733939, 0.450678,-1.614398},
{-0.582342, 0.299842,-0.758035},
{ 0.162262,-0.671634,-1.031603},
{-1.03168 , 0.114413,-0.263791},
{-0.59813 , 0.591675,-0.319428}
};

float b3[3] = {0.058498,0.054127,-0.202144};

// ================= RELU =================
float relu(float x){ return x>0?x:0; }

// ================= SOFTMAX =================
void softmax(float* input,float* output,int size){
  float maxVal=input[0];
  for(int i=1;i<size;i++)
    if(input[i]>maxVal) maxVal=input[i];

  float sum=0;
  for(int i=0;i<size;i++){
    output[i]=exp(input[i]-maxVal);
    sum+=output[i];
  }
  for(int i=0;i<size;i++)
    output[i]/=sum;
}

// ================= NN =================
void predict(float inputData[6],float output[3]){
  float l1[16],l2[8],l3[3];

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

// ================= READ DUST =================
float readDust(){
  digitalWrite(DUST_LED_PIN,LOW);
  delayMicroseconds(280);
  int raw=analogRead(DUST_SENSOR_PIN);
  delayMicroseconds(40);
  digitalWrite(DUST_LED_PIN,HIGH);

  float voltage=raw*(3.3/4095.0);
  float density=(voltage-0.6)/0.5;
  if(density<0) density=0;
  return density*1000;
}

// ================= READ JW01 =================
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

// ================= ALERT =================
unsigned long poorStart=0;
bool poorActive=false;

// ================= MAIN =================
void sendData(){

  float temperature=dht.readTemperature();
  float humidity=dht.readHumidity();
  int mq135=analogRead(MQ135_PIN);
  float dust=readDust();

  float tvoc=0;
  int co2=0;
  readJW01(tvoc,co2);

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
    if(output[i]>maxVal){maxVal=output[i];label=i;}

  String status;
  if(label==0) status="GOOD";
  else if(label==1) status="MODERATE";
  else status="POOR";

  // ===== SERIAL DEBUG =====
  Serial.println("\n===== SENSOR DATA =====");
  Serial.print("Temp: ");Serial.println(temperature);
  Serial.print("Humidity: ");Serial.println(humidity);
  Serial.print("MQ135: ");Serial.println(mq135);
  Serial.print("CO2: ");Serial.println(co2);
  Serial.print("TVOC: ");Serial.println(tvoc);
  Serial.print("Dust: ");Serial.println(dust);
  Serial.println("Status: "+status);

  // ===== CSV LOGGER =====
  Serial.print(millis()); Serial.print(",");
  Serial.print(temperature); Serial.print(",");
  Serial.print(humidity); Serial.print(",");
  Serial.print(mq135); Serial.print(",");
  Serial.print(co2); Serial.print(",");
  Serial.print(tvoc); Serial.print(",");
  Serial.print(dust); Serial.print(",");
  Serial.println(label);

  // ===== BLYNK =====
  Blynk.virtualWrite(V0,mq135);
  Blynk.virtualWrite(V1,temperature);
  Blynk.virtualWrite(V2,humidity);
  Blynk.virtualWrite(V3,co2);
  Blynk.virtualWrite(V4,tvoc);
  Blynk.virtualWrite(V5,dust);
  Blynk.virtualWrite(V6,status);

  // ===== ALERT =====
  if(label==2){
    if(!poorActive){
      poorStart=millis();
      poorActive=true;
    }
    if(millis()-poorStart>60000){
      Blynk.logEvent("air_alert","Air quality POOR > 1 minute");
    }
  }else{
    poorActive=false;
  }
}

void setup(){
  Serial.begin(115200);
  dht.begin();
  pinMode(DUST_LED_PIN,OUTPUT);
  digitalWrite(DUST_LED_PIN,HIGH);
  ssam.begin(9600,SERIAL_8N1,SSAM_RX_PIN,SSAM_TX_PIN);

  Blynk.begin(BLYNK_AUTH_TOKEN,ssid,pass);
  timer.setInterval(3000L,sendData);
}

void loop(){
  Blynk.run();
  timer.run();
}
