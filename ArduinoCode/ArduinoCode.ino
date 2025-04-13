const int TOP_LEFT_PIN = 2;
const int TOP_RIGHT_PIN = 3;
const int KICK_PIN = 4;
const int CLAP_PIN = 5;

bool top_left_pressed = false;
bool top_right_pressed = false;
bool kick_pressed = false;
bool clap_pressed = false;

void setup() {
  pinMode(TOP_LEFT_PIN, INPUT_PULLUP);
  pinMode(TOP_RIGHT_PIN, INPUT_PULLUP);
  pinMode(KICK_PIN, INPUT_PULLUP);
  pinMode(CLAP_PIN, INPUT_PULLUP);

  Serial.begin(9600);
}

void loop() {
  // TOP LEFT
  if (digitalRead(TOP_LEFT_PIN) == LOW && !top_left_pressed) {
    top_left_pressed = true;
    Serial.println("TL");
  }
  if (digitalRead(TOP_LEFT_PIN) == HIGH && top_left_pressed) {
    top_left_pressed = false;
  }

  // TOP RIGHT
  if (digitalRead(TOP_RIGHT_PIN) == LOW && !top_right_pressed) {
    top_right_pressed = true;
    Serial.println("TR");
  }
  if (digitalRead(TOP_RIGHT_PIN) == HIGH && top_right_pressed) {
    top_right_pressed = false;
  }

  // KICK
  if (digitalRead(KICK_PIN) == LOW && !kick_pressed) {
    kick_pressed = true;
    Serial.println("BL");
  }
  if (digitalRead(KICK_PIN) == HIGH && kick_pressed) {
    kick_pressed = false;
  }

  // CLAP
  if (digitalRead(CLAP_PIN) == LOW && !clap_pressed) {
    clap_pressed = true;
    Serial.println("BR");
  }
  if (digitalRead(CLAP_PIN) == HIGH && clap_pressed) {
    clap_pressed = false;
  }
}
