/**
 * ============================================================================
 * Project: Computer Vision Controlled Custom 8x8 LED Matrix
 * Target:  ESP32-C3 (DevKit / SuperMini)
 * 
 * Hardware Configuration:
 *   - 8 Anode Rows: Switched to +5V via IRF9540N P-FETs + BC547 NPN level shifters
 *   - 8 Cathode Cols: Switched to GND via 2N7000 N-FETs + 220 Ohm series resistors
 * 
 * Multiplexing Architecture:
 *   - Row scanning: 100 Hz frame refresh (1250 us per row)
 *   - Dead-time blanking: 15 us to eliminate IRF9540N gate capacitance ghosting
 *   - Sub-cycle PWM: Smooth 0-100% brightness modulation
 * 
 * Supported Commands:
 *   - LED:<1-8>          -> Illuminates selected 1-of-8 position indicator
 *   - LED:<row>,<col>    -> Illuminates individual coordinate in 8x8 grid (0-7)
 *   - BRIGHTNESS:<0-100> -> Adjusts LED intensity in real time
 *   - MESSAGE:<text>     -> Starts non-blocking scrolling text ("HELLO", etc.)
 *   - PATTERN:HEART      -> Displays static aesthetic 8x8 heart
 *   - PATTERN:SMILE      -> Displays smile bitmap
 *   - PATTERN:CLEAR      -> Clears the display
 *   - ANIMATION:PULSE    -> Triggers pulsing heartbeat animation
 * ============================================================================
 */

#include <WiFi.h>
#include <WiFiUdp.h>

// ==================== WI-FI SETTINGS ====================
// Replace with your local Wi-Fi credentials
const char* WIFI_SSID     = "Your_WiFi_SSID";
const char* WIFI_PASS     = "Your_WiFi_Password";
const uint16_t UDP_PORT   = 8888;

// ==================== PIN DEFINITIONS ====================
// Row Anodes (Driven HIGH via BC547 -> IRF9540N)
const uint8_t ROW_PINS[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// Column Cathodes (Driven HIGH via 2N7000 -> 220 Ohm -> GND)
const uint8_t COL_PINS[8] = {8, 10, 18, 19, 20, 21, 9, 11};

// ==================== TIMING CONSTANTS ====================
// Full frame: 8 rows * 1250 us = 10,000 us = 10 ms = 100 Hz refresh rate
const uint32_t ROW_DWELL_US = 1250; 
const uint32_t DEAD_TIME_US = 15;   // Hardware turn-off blanking for ghosting prevention

// ==================== DISPLAY BUFFERS & STATE ====================
// Active display buffer (frameBuffer[r] bit c = LED at row r, col c)
volatile uint8_t frameBuffer[8] = {0};

// Brightness: 0 (OFF) to 100 (Full intensity)
volatile uint8_t currentBrightness = 75;

enum DisplayMode {
  MODE_STATIC_BITMAP,
  MODE_SINGLE_POSITION,
  MODE_SCROLLING_TEXT,
  MODE_ANIMATION
};
volatile DisplayMode currentMode = MODE_STATIC_BITMAP;

// Text Scrolling Engine State
String scrollText = "HELLO";
int scrollOffset = -8;
unsigned long lastScrollTick = 0;
const uint16_t SCROLL_SPEED_MS = 110;

// Pulse Animation State
unsigned long lastAnimTick = 0;
uint8_t animFrame = 0;

// Network
WiFiUDP udp;
char udpPacketBuffer[256];
String serialCommandBuffer = "";

// ==================== 8x8 BITMAP ASSETS ====================
// Aesthetic Heart (Optimized for 8x8 red matrix)
const uint8_t BITMAP_HEART_LARGE[8] = {
  0b00000000,
  0b01100110,
  0b11111111,
  0b11111111,
  0b01111110,
  0b00111100,
  0b00011000,
  0b00000000
};

// Small Heart for pulsing animation
const uint8_t BITMAP_HEART_SMALL[8] = {
  0b00000000,
  0b00000000,
  0b00100100,
  0b01111110,
  0b00111100,
  0b00011000,
  0b00000000,
  0b00000000
};

// Smile Face
const uint8_t BITMAP_SMILE[8] = {
  0b00111100,
  0b01000010,
  0b10100101,
  0b10000001,
  0b10100101,
  0b10011001,
  0b01000010,
  0b00111100
};

// ==================== 5x7 ASCII FONT TABLE ====================
// ASCII 32 (' ') to 90 ('Z')
const uint8_t FONT_5X7[][5] = {
  {0x00, 0x00, 0x00, 0x00, 0x00}, // 32 ' '
  {0x00, 0x00, 0x5F, 0x00, 0x00}, // 33 '!'
  {0x00, 0x07, 0x00, 0x07, 0x00}, // 34 '"'
  {0x14, 0x7F, 0x14, 0x7F, 0x14}, // 35 '#'
  {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // 36 '$'
  {0x23, 0x13, 0x08, 0x64, 0x62}, // 37 '%'
  {0x36, 0x49, 0x55, 0x22, 0x50}, // 38 '&'
  {0x00, 0x05, 0x03, 0x00, 0x00}, // 39 '''
  {0x00, 0x1C, 0x22, 0x41, 0x00}, // 40 '('
  {0x00, 0x41, 0x22, 0x1C, 0x00}, // 41 ')'
  {0x14, 0x08, 0x3E, 0x08, 0x14}, // 42 '*'
  {0x08, 0x08, 0x3E, 0x08, 0x08}, // 43 '+'
  {0x00, 0x50, 0x30, 0x00, 0x00}, // 44 ','
  {0x08, 0x08, 0x08, 0x08, 0x08}, // 45 '-'
  {0x00, 0x60, 0x60, 0x00, 0x00}, // 46 '.'
  {0x20, 0x10, 0x08, 0x04, 0x02}, // 47 '/'
  {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 48 '0'
  {0x00, 0x42, 0x7F, 0x40, 0x00}, // 49 '1'
  {0x42, 0x61, 0x51, 0x49, 0x46}, // 50 '2'
  {0x21, 0x41, 0x45, 0x4B, 0x31}, // 51 '3'
  {0x18, 0x14, 0x12, 0x7F, 0x10}, // 52 '4'
  {0x27, 0x45, 0x45, 0x45, 0x39}, // 53 '5'
  {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 54 '6'
  {0x01, 0x71, 0x09, 0x05, 0x03}, // 55 '7'
  {0x36, 0x49, 0x49, 0x49, 0x36}, // 56 '8'
  {0x06, 0x49, 0x49, 0x29, 0x1E}, // 57 '9'
  {0x00, 0x36, 0x36, 0x00, 0x00}, // 58 ':'
  {0x00, 0x56, 0x36, 0x00, 0x00}, // 59 ';'
  {0x08, 0x14, 0x22, 0x41, 0x00}, // 60 '<'
  {0x14, 0x14, 0x14, 0x14, 0x14}, // 61 '='
  {0x00, 0x41, 0x22, 0x14, 0x08}, // 62 '>'
  {0x02, 0x01, 0x51, 0x09, 0x06}, // 63 '?'
  {0x32, 0x49, 0x79, 0x41, 0x3E}, // 64 '@'
  {0x7E, 0x11, 0x11, 0x11, 0x7E}, // 65 'A'
  {0x7F, 0x49, 0x49, 0x49, 0x36}, // 66 'B'
  {0x3E, 0x41, 0x41, 0x41, 0x22}, // 67 'C'
  {0x7F, 0x41, 0x41, 0x22, 0x1C}, // 68 'D'
  {0x7F, 0x49, 0x49, 0x49, 0x41}, // 69 'E'
  {0x7F, 0x09, 0x09, 0x09, 0x01}, // 70 'F'
  {0x3E, 0x41, 0x49, 0x49, 0x7A}, // 71 'G'
  {0x7F, 0x08, 0x08, 0x08, 0x7F}, // 72 'H'
  {0x00, 0x41, 0x7F, 0x41, 0x00}, // 73 'I'
  {0x20, 0x40, 0x41, 0x3F, 0x01}, // 74 'J'
  {0x7F, 0x08, 0x14, 0x22, 0x41}, // 75 'K'
  {0x7F, 0x40, 0x40, 0x40, 0x40}, // 76 'L'
  {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // 77 'M'
  {0x7F, 0x04, 0x08, 0x10, 0x7F}, // 78 'N'
  {0x3E, 0x41, 0x41, 0x41, 0x3E}, // 79 'O'
  {0x7F, 0x09, 0x09, 0x09, 0x06}, // 80 'P'
  {0x3E, 0x41, 0x51, 0x21, 0x5E}, // 81 'Q'
  {0x7F, 0x09, 0x19, 0x29, 0x46}, // 82 'R'
  {0x46, 0x49, 0x49, 0x49, 0x31}, // 83 'S'
  {0x01, 0x01, 0x7F, 0x01, 0x01}, // 84 'T'
  {0x3F, 0x40, 0x40, 0x40, 0x3F}, // 85 'U'
  {0x1F, 0x20, 0x40, 0x20, 0x1F}, // 86 'V'
  {0x3F, 0x40, 0x38, 0x40, 0x3F}, // 87 'W'
  {0x63, 0x14, 0x08, 0x14, 0x63}, // 88 'X'
  {0x07, 0x08, 0x70, 0x08, 0x07}, // 89 'Y'
  {0x61, 0x51, 0x49, 0x45, 0x43}  // 90 'Z'
};

// Forward Declarations
void handleCommand(String cmd);
void parseIncomingPacket();
void updateDisplayAnimations();
void displaySingleRow(uint8_t row);

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[SYSTEM] ESP32-C3 8x8 LED Matrix Controller Initializing...");

  // Initialize GPIO outputs
  for (uint8_t i = 0; i < 8; i++) {
    pinMode(ROW_PINS[i], OUTPUT);
    digitalWrite(ROW_PINS[i], LOW); // BC547 OFF -> IRF9540N OFF
    
    pinMode(COL_PINS[i], OUTPUT);
    digitalWrite(COL_PINS[i], LOW); // 2N7000 OFF
  }

  // Load default pattern (Aesthetic Heart)
  for (uint8_t r = 0; r < 8; r++) {
    frameBuffer[r] = BITMAP_HEART_LARGE[r];
  }

  // Wi-Fi Initialization
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[NETWORK] Connecting to Wi-Fi: ");
  Serial.print(WIFI_SSID);

  uint8_t wifiTimeout = 0;
  while (WiFi.status() != WL_CONNECTED && wifiTimeout < 20) {
    delay(500);
    Serial.print(".");
    wifiTimeout++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[NETWORK] Wi-Fi Connected!");
    Serial.print("[NETWORK] IP Address: ");
    Serial.println(WiFi.localIP());
    udp.begin(UDP_PORT);
    Serial.printf("[NETWORK] UDP Listener active on port %d\n", UDP_PORT);
  } else {
    Serial.println("\n[NETWORK] Wi-Fi not connected (Running in USB Serial mode)");
  }

  Serial.println("[SYSTEM] Ready for commands. Available commands:");
  Serial.println("  LED:<1-8> | BRIGHTNESS:<0-100> | MESSAGE:<text> | PATTERN:<HEART|SMILE|CLEAR>");
}

// ==================== MAIN LOOP ====================
void loop() {
  // 1. Process network packets (UDP)
  parseIncomingPacket();

  // 2. Process USB Serial commands
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialCommandBuffer.length() > 0) {
        handleCommand(serialCommandBuffer);
        serialCommandBuffer = "";
      }
    } else {
      serialCommandBuffer += c;
    }
  }

  // 3. Update Text Scrolling / Animations
  updateDisplayAnimations();

  // 4. Multiplex all 8 rows sequentially (1 full frame = 10 ms = 100 Hz)
  for (uint8_t row = 0; row < 8; row++) {
    displaySingleRow(row);
  }
}

// ==================== MULTIPLEXING & PWM ENGINE ====================
void displaySingleRow(uint8_t row) {
  // STEP 1: Turn OFF previous row to prevent ghosting
  for (uint8_t r = 0; r < 8; r++) {
    digitalWrite(ROW_PINS[r], LOW); // BC547 OFF -> IRF9540N gate pulled to +5V -> OFF
  }

  // STEP 2: Blank all columns
  for (uint8_t c = 0; c < 8; c++) {
    digitalWrite(COL_PINS[c], LOW); // 2N7000 OFF
  }

  // STEP 3: Hardware Dead-Time
  // Allows IRF9540N gate capacitance (Ciss=1300pF) to fully discharge via 10k pull-up
  delayMicroseconds(DEAD_TIME_US);

  // If brightness is 0 or no LEDs lit in this row, skip turning on
  uint8_t rowData = frameBuffer[row];
  if (currentBrightness == 0 || rowData == 0) {
    delayMicroseconds(ROW_DWELL_US - DEAD_TIME_US);
    return;
  }

  // STEP 4: Set column cathode states for current row
  for (uint8_t col = 0; col < 8; col++) {
    if (rowData & (1 << (7 - col))) {
      digitalWrite(COL_PINS[col], HIGH); // 2N7000 ON -> Cathode pulled to GND
    } else {
      digitalWrite(COL_PINS[col], LOW);  // 2N7000 OFF
    }
  }

  // STEP 5: Turn ON active row anode
  digitalWrite(ROW_PINS[row], HIGH); // BC547 ON -> IRF9540N gate pulled to GND -> ON (+5V)

  // STEP 6: Sub-cycle Software PWM for Intensity Control
  uint32_t activeDwellUs = (ROW_DWELL_US * currentBrightness) / 100;
  if (activeDwellUs > DEAD_TIME_US) {
    delayMicroseconds(activeDwellUs - DEAD_TIME_US);
  }

  // Blank row early if brightness < 100%
  if (currentBrightness < 100) {
    digitalWrite(ROW_PINS[row], LOW);
    uint32_t remainingDwellUs = ROW_DWELL_US - activeDwellUs;
    if (remainingDwellUs > 0) {
      delayMicroseconds(remainingDwellUs);
    }
  }
}

// ==================== ANIMATION & SCROLLING ENGINE ====================
void updateDisplayAnimations() {
  unsigned long now = millis();

  // Scrolling Text Engine
  if (currentMode == MODE_SCROLLING_TEXT) {
    if (now - lastScrollTick >= SCROLL_SPEED_MS) {
      lastScrollTick = now;

      // Calculate total text width in pixel columns
      // Each character is 5 columns wide + 1 space column = 6 columns
      int totalColumns = scrollText.length() * 6;

      // Clear frame buffer
      for (uint8_t r = 0; r < 8; r++) {
        frameBuffer[r] = 0;
      }

      // Render visible 8 columns starting at scrollOffset
      for (uint8_t screenCol = 0; screenCol < 8; screenCol++) {
        int textCol = scrollOffset + screenCol;
        if (textCol >= 0 && textCol < totalColumns) {
          int charIndex = textCol / 6;
          int charCol = textCol % 6;

          if (charCol < 5) { // 5 pixel columns of the character
            char c = scrollText.charAt(charIndex);
            // Convert lowercase to uppercase for 5x7 font
            if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
            if (c >= 32 && c <= 90) {
              uint8_t fontColBits = FONT_5X7[c - 32][charCol];
              // Map 7 bits into rows 0..6
              for (uint8_t r = 0; r < 7; r++) {
                if (fontColBits & (1 << r)) {
                  frameBuffer[r] |= (1 << (7 - screenCol));
                }
              }
            }
          }
        }
      }

      scrollOffset++;
      if (scrollOffset > totalColumns) {
        scrollOffset = -8; // Loop back from right edge
      }
    }
  }

  // Pulsing Heart Animation
  else if (currentMode == MODE_ANIMATION) {
    if (now - lastAnimTick >= 400) {
      lastAnimTick = now;
      animFrame = !animFrame;
      const uint8_t* pattern = animFrame ? BITMAP_HEART_LARGE : BITMAP_HEART_SMALL;
      for (uint8_t r = 0; r < 8; r++) {
        frameBuffer[r] = pattern[r];
      }
    }
  }
}

// ==================== NETWORK & COMMAND PARSING ====================
void parseIncomingPacket() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(udpPacketBuffer, sizeof(udpPacketBuffer) - 1);
    if (len > 0) {
      udpPacketBuffer[len] = '\0';
      String cmd = String(udpPacketBuffer);
      cmd.trim();
      handleCommand(cmd);
    }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  Serial.print("[COMMAND RECEIVED] ");
  Serial.println(cmd);

  // 1. LED Selection Command: "LED:<pos>" or "LED:<r>,<c>"
  if (cmd.startsWith("LED:")) {
    String param = cmd.substring(4);
    param.trim();

    int commaIndex = param.indexOf(',');
    if (commaIndex > 0) {
      // Coordinate format: LED:row,col (0-7)
      int r = param.substring(0, commaIndex).toInt();
      int c = param.substring(commaIndex + 1).toInt();
      if (r >= 0 && r < 8 && c >= 0 && c < 8) {
        currentMode = MODE_SINGLE_POSITION;
        for (uint8_t i = 0; i < 8; i++) frameBuffer[i] = 0;
        frameBuffer[r] = (1 << (7 - c));
        Serial.printf("[ACTION] LED set at (%d, %d)\n", r, c);
      }
    } else {
      // 1-of-8 Selection: LED:1 through LED:8
      int pos = param.toInt();
      if (pos >= 1 && pos <= 8) {
        currentMode = MODE_SINGLE_POSITION;
        for (uint8_t i = 0; i < 8; i++) frameBuffer[i] = 0;
        // Turn ON indicator on Row 3 (center) at column (pos - 1)
        uint8_t col = pos - 1;
        frameBuffer[3] = (1 << (7 - col));
        Serial.printf("[ACTION] Selected Position %d illuminated\n", pos);
    }
  }

  // 1b. Toggle LED Command: "TOGGLE:<r>,<c>"
  else if (cmd.startsWith("TOGGLE:")) {
    String param = cmd.substring(7);
    param.trim();
    int commaIndex = param.indexOf(',');
    if (commaIndex > 0) {
      int r = param.substring(0, commaIndex).toInt();
      int c = param.substring(commaIndex + 1).toInt();
      if (r >= 0 && r < 8 && c >= 0 && c < 8) {
        currentMode = MODE_STATIC_BITMAP;
        frameBuffer[r] ^= (1 << (7 - c)); // Toggle specific LED
        Serial.printf("[ACTION] LED toggled at (%d, %d) -> state: %s\n", 
                      r, c, (frameBuffer[r] & (1 << (7 - c))) ? "ON" : "OFF");
      }
    }
  }

  // 2. Brightness Command: "BRIGHTNESS:<0-100>"
  else if (cmd.startsWith("BRIGHTNESS:")) {
    int val = cmd.substring(11).toInt();
    if (val < 0) val = 0;
    if (val > 100) val = 100;
    currentBrightness = (uint8_t)val;
    Serial.printf("[ACTION] Brightness set to %d%%\n", currentBrightness);
  }

  // 3. Scrolling Message Command: "MESSAGE:<text>"
  else if (cmd.startsWith("MESSAGE:")) {
    String msg = cmd.substring(8);
    msg.trim();
    if (msg.length() > 0) {
      scrollText = msg;
      scrollOffset = -8;
      currentMode = MODE_SCROLLING_TEXT;
      Serial.printf("[ACTION] Scrolling message set: \"%s\"\n", scrollText.c_str());
    }
  }

  // 4. Predefined Pattern Command: "PATTERN:<name>"
  else if (cmd.startsWith("PATTERN:")) {
    String pat = cmd.substring(8);
    pat.toUpperCase();
    pat.trim();

    currentMode = MODE_STATIC_BITMAP;
    if (pat == "HEART") {
      for (uint8_t r = 0; r < 8; r++) frameBuffer[r] = BITMAP_HEART_LARGE[r];
      Serial.println("[ACTION] Pattern set: HEART");
    } else if (pat == "SMILE") {
      for (uint8_t r = 0; r < 8; r++) frameBuffer[r] = BITMAP_SMILE[r];
      Serial.println("[ACTION] Pattern set: SMILE");
    } else if (pat == "CLEAR") {
      for (uint8_t r = 0; r < 8; r++) frameBuffer[r] = 0;
      Serial.println("[ACTION] Pattern set: CLEAR");
    }
  }

  // 5. Animation Command: "ANIMATION:<name>"
  else if (cmd.startsWith("ANIMATION:")) {
    String anim = cmd.substring(10);
    anim.toUpperCase();
    anim.trim();

    if (anim == "PULSE" || anim == "HEART") {
      currentMode = MODE_ANIMATION;
      Serial.println("[ACTION] Animation set: PULSE HEART");
    }
  }
}
