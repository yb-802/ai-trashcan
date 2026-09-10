#include "HX711.h"

// 接腳定義
#define DOUT 11
#define CLK 10
const int buttonPin = 4;
const int greenLED = 12;
const int redLED = 13;
const int motorIn3 = 5;
const int motorIn4 = 6;

HX711 scale;

float weightA = 0;
float weightB = 0;

bool buttonPressed = false;
bool sentStartSignal = false;
bool scaleReady = false;

void setup() {
  delay(1000); // 等待穩定

  while (Serial.available()) Serial.read(); // 清空 Serial 緩衝區
  Serial.begin(9600);

  // 不在這裡啟用 scale，延後至按下按鈕再啟用
  // scale.begin(DOUT, CLK);
  // scale.set_scale(2280.f);  // 根據已知校正值調整
  // scale.tare();

  pinMode(buttonPin, INPUT_PULLUP);  // 按鈕為低電位觸發
  pinMode(greenLED, OUTPUT);
  pinMode(redLED, OUTPUT);
  pinMode(motorIn3, OUTPUT);
  pinMode(motorIn4, OUTPUT);

  digitalWrite(greenLED, HIGH);
  digitalWrite(redLED, LOW);
  analogWrite(motorIn3, 0);
  analogWrite(motorIn4, 0);
}

void loop() {
  // 偵測按鈕按下
  if (digitalRead(buttonPin) == LOW && !buttonPressed) {
    delay(50);  // 去彈跳
    buttonPressed = true;

    // 亮紅燈，熄綠燈
    digitalWrite(greenLED, LOW);
    digitalWrite(redLED, HIGH);

    // 啟用秤重（此時才開始感測）
    scaleReady = true;

    // 紀錄初始重量
    if (scaleReady) {
      scale.tare();  // 歸零
      delay(500);
      weightA = scale.get_units(5);  // 初始重量
      Serial.println(weightA);
    }
    
    // 傳送開啟訊號給 Python
    Serial.println("buttonPressed");  
    sentStartSignal = true;
  }

  // 接收分類結果
  if (sentStartSignal && Serial.available()) {
    String result = Serial.readStringUntil('\n');
    result.trim();


    if (result.length() > 0) {
      if (result == "trash") {
        rotateMotor("right");
      } else {
        rotateMotor("left");
        delay(3000);  // 等待物品掉落
        weightB = scale.get_units(5);
        // Serial.println(weightB);
        float diff = weightB - weightA;
        Serial.println(diff);

      }

      // 重設
      delay(1000);
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, HIGH);
      buttonPressed = false;
      sentStartSignal = false;
    }
  }
}

void rotateMotor(String direction) {
  if (direction == "right") {
    analogWrite(motorIn3, 80);
    analogWrite(motorIn4, 0);
    delay(1000);  // 轉動 120 度（需依馬達調整時間）
    analogWrite(motorIn3, 0);
    analogWrite(motorIn4, 0);
    delay(1000);
    analogWrite(motorIn3, 0);  // 回正
    analogWrite(motorIn4, 80);
    delay(1000);  

  } else {
    analogWrite(motorIn3, 0);
    analogWrite(motorIn4, 80);
    delay(1000);
        analogWrite(motorIn3, 0);
    analogWrite(motorIn4, 0);
    delay(1000);
    analogWrite(motorIn3, 80);  // 回正
    analogWrite(motorIn4, 0);
    delay(1000);  
  }

  analogWrite(motorIn3, 0);
  analogWrite(motorIn4, 0);
}