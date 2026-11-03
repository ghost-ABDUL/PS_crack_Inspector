const int leds[8] = {2,3,4,5,6,7,8,9};
int currentLED = -1;

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 8; i++) {
    pinMode(leds[i], OUTPUT);
    digitalWrite(leds[i], HIGH);   // HIGH = OFF
  }
  delay(50);
  Serial.println("READY");
}

void allOff() {
  for (int i = 0; i < 8; i++) digitalWrite(leds[i], HIGH);
  currentLED = -1;
}

void allOn() {
  for (int i = 0; i < 8; i++) digitalWrite(leds[i], LOW);
}

void setLED(int idx) {
  allOff();
  delayMicroseconds(500);
  digitalWrite(leds[idx], LOW);
  currentLED = idx;
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd >= '1' && cmd <= '8') {
      setLED(cmd - '1');
      delay(10);
      Serial.println("ON");
    }
    else if (cmd == '9') {
      allOn();
      Serial.println("ON");
    }
    else if (cmd == '0') {
      allOff();
      Serial.println("OFF");
    }
  }
}