/*
 * AIRTOUCH-88
 * -----------
 * 8x8 red LED matrix controller based on ESP32-C3, two 74HC595
 * shift registers, MOSFET row/column switching, and Wi-Fi UDP.
 *
 * Architecture:
 *   PC / OpenCV -> UDP -> ESP32-C3 -> 74HC595 x2 -> 8x8 LED Matrix
 *
 * Hardware:
 *   GPIO4 -> 74HC595 #1 SER/DS (pin 14)
 *   GPIO6 -> SHCP/SRCLK (pin 11) on both 74HC595s
 *   GPIO7 -> STCP/RCLK (pin 12) on both 74HC595s
 *
 * Shift-register chain:
 *   ESP32-C3 -> 74HC595 #1 -> 74HC595 #2
 *   #1 drives rows    -> 2N7000
 *   #2 drives columns -> IRF9540N
 *
 * Network:
 *   UDP port: 8888
 *
 * Commands:
 *   FRAME:<64 bits>     Replace the complete 8x8 framebuffer.
 *   LED:<1-64>          Turn on a numbered LED.
 *   LED:<row,col>       Turn on a specific pixel (0-7 coordinates).
 *   TOGGLE:<row,col>    Toggle a specific pixel.
 *   BRIGHTNESS:<0-100>  Set display brightness.
 *   ALL_ON              Turn all pixels on.
 *   CLEAR               Turn all pixels off.
 *   PATTERN:HEART       Display the heart pattern.
 *   PATTERN:SMILE       Display the smile pattern.
 *
 * FRAME format:
 *   64 bits, sent row-by-row. Each character is one pixel:
 *   row 0 = bits  0- 7, row 1 = bits  8-15, ... row 7 = bits 56-63
 *   0 = OFF, 1 = ON
 *
 */

#include <WiFi.h>
#include <WiFiUdp.h>


// -----------------------------------------------------------------------------
// Wi-Fi configuration
// -----------------------------------------------------------------------------

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

WiFiUDP udp;

const uint16_t UDP_PORT = 8888;

bool udpStarted = false;


// -----------------------------------------------------------------------------
// Wi-Fi retry timing
// -----------------------------------------------------------------------------

unsigned long lastWiFiAttempt = 0;

const unsigned long WIFI_RETRY_INTERVAL = 5000;


// -----------------------------------------------------------------------------
// ESP32-C3 pin assignments
// -----------------------------------------------------------------------------

const int DATA_PIN  = 4;
const int CLOCK_PIN = 6;
const int LATCH_PIN = 7;


// -----------------------------------------------------------------------------
// 8x8 framebuffer
// -----------------------------------------------------------------------------

// One byte represents one logical row.
// Bit 0 = column 0, bit 7 = column 7.
volatile uint8_t frameBuffer[8] = {
  0,0,0,0,0,0,0,0
};


// -----------------------------------------------------------------------------
// Matrix orientation
// -----------------------------------------------------------------------------

// Set a flag to true if the physical matrix appears mirrored.
const bool REVERSE_ROWS    = false;
const bool REVERSE_COLUMNS = false;


// -----------------------------------------------------------------------------
// Display brightness
// -----------------------------------------------------------------------------

uint8_t brightness = 75;


// -----------------------------------------------------------------------------
// Multiplexing / scan timing
// -----------------------------------------------------------------------------

uint8_t currentRow = 0;

unsigned long lastScanTime = 0;

const unsigned long ROW_PERIOD_US = 1250;
const unsigned long DEAD_TIME_US  = 15;


// -----------------------------------------------------------------------------
// 74HC595 low-level interface
// -----------------------------------------------------------------------------

void shiftByte(uint8_t value)
{

  for (int i = 7; i >= 0; i--)
  {

    digitalWrite(CLOCK_PIN, LOW);

    if (value & (1 << i))
      digitalWrite(DATA_PIN, HIGH);
    else
      digitalWrite(DATA_PIN, LOW);

    digitalWrite(CLOCK_PIN, HIGH);
  }

  digitalWrite(CLOCK_PIN, LOW);
}


// -----------------------------------------------------------------------------
// Latch shifted data
// -----------------------------------------------------------------------------

void latchData()
{

  digitalWrite(LATCH_PIN, HIGH);

  delayMicroseconds(1);

  digitalWrite(LATCH_PIN, LOW);
}


// -----------------------------------------------------------------------------
// Safety: disable all LED paths
// -----------------------------------------------------------------------------

void allOutputsOff()
{

  digitalWrite(LATCH_PIN, LOW);

  // 595 #2 = columns.
  // IRF9540N is active LOW: HIGH = OFF.
  shiftByte(0xFF);

  // 595 #1 = rows.
  // 2N7000 gate is active HIGH: LOW = OFF.
  shiftByte(0x00);

  latchData();
}


// -----------------------------------------------------------------------------
// Logical-to-physical row mapping
// -----------------------------------------------------------------------------

uint8_t physicalRow(uint8_t row)
{

  if (REVERSE_ROWS)
    return 7 - row;

  return row;
}


// -----------------------------------------------------------------------------
// Logical-to-physical column mapping
// -----------------------------------------------------------------------------

uint8_t physicalColumns(uint8_t value)
{

  if (!REVERSE_COLUMNS)
    return value;

  uint8_t result = 0;

  for (int i = 0; i < 8; i++)
  {

    if (value & (1 << i))
      result |= (1 << (7 - i));
  }

  return result;
}


// -----------------------------------------------------------------------------
// Multiplex one row
// -----------------------------------------------------------------------------

void displayRow(uint8_t row)
{

  // Turn everything OFF first
  allOutputsOff();

  delayMicroseconds(DEAD_TIME_US);


  // Brightness zero
  if (brightness == 0)
    return;


  // Read current row
  uint8_t columnData;

  noInterrupts();

  columnData = frameBuffer[row];

  interrupts();


  // Physical row
  uint8_t r = physicalRow(row);

  uint8_t rowData = (1 << r);


  // Physical columns
  columnData = physicalColumns(columnData);


  // Calculate ON time
  unsigned long onTime =
      ((unsigned long)ROW_PERIOD_US * brightness) / 100;


  if (onTime <= DEAD_TIME_US)
    return;


  // ----------------------------------------------------------
  // SEND TO SHIFT REGISTERS
  // ----------------------------------------------------------

  digitalWrite(LATCH_PIN, LOW);


  // FAR 595 (#2) FIRST
  // Columns active LOW
  shiftByte(~columnData);


  // NEAR 595 (#1) SECOND
  // Rows active HIGH
  shiftByte(rowData);


  latchData();


  // ----------------------------------------------------------
  // DISPLAY
  // ----------------------------------------------------------

  delayMicroseconds(onTime - DEAD_TIME_US);


  // ----------------------------------------------------------
  // OFF
  // ----------------------------------------------------------

  allOutputsOff();
}


// -----------------------------------------------------------------------------
// Framebuffer operations
// -----------------------------------------------------------------------------

void clearMatrix()
{

  noInterrupts();

  for (int i = 0; i < 8; i++)
    frameBuffer[i] = 0;

  interrupts();

  Serial.println("[MATRIX] CLEAR");
}


// -----------------------------------------------------------------------------
// Turn every pixel on
// -----------------------------------------------------------------------------

void allLEDsOn()
{

  noInterrupts();

  for (int i = 0; i < 8; i++)
    frameBuffer[i] = 0xFF;

  interrupts();

  Serial.println("[MATRIX] ALL ON");
}


// -----------------------------------------------------------------------------
// Set one pixel
// -----------------------------------------------------------------------------

void setLED(int row, int col, bool state)
{

  if (row < 0 || row > 7)
    return;

  if (col < 0 || col > 7)
    return;


  noInterrupts();

  if (state)
    frameBuffer[row] |= (1 << col);
  else
    frameBuffer[row] &= ~(1 << col);

  interrupts();
}


// -----------------------------------------------------------------------------
// Toggle one pixel
// -----------------------------------------------------------------------------

void toggleLED(int row, int col)
{

  if (row < 0 || row > 7)
    return;

  if (col < 0 || col > 7)
    return;


  noInterrupts();

  frameBuffer[row] ^= (1 << col);

  interrupts();
}


// -----------------------------------------------------------------------------
// Built-in patterns
// -----------------------------------------------------------------------------

void heartPattern()
{

  const uint8_t heart[8] =
  {
    0b01100110,
    0b11111111,
    0b11111111,
    0b11111111,
    0b01111110,
    0b00111100,
    0b00011000,
    0b00000000
  };


  noInterrupts();

  for (int i = 0; i < 8; i++)
    frameBuffer[i] = heart[i];

  interrupts();


  Serial.println("[MATRIX] HEART");
}


void smilePattern()
{

  const uint8_t smile[8] =
  {
    0b00111100,
    0b01000010,
    0b10100101,
    0b10000001,
    0b10100101,
    0b10011001,
    0b01000010,
    0b00111100
  };


  noInterrupts();

  for (int i = 0; i < 8; i++)
    frameBuffer[i] = smile[i];

  interrupts();


  Serial.println("[MATRIX] SMILE");
}


// -----------------------------------------------------------------------------
// FRAME:<64 bits> parser
// -----------------------------------------------------------------------------

bool processFrame(String data)
{

  data.trim();


  if (!data.startsWith("FRAME:"))
    return false;


  String bits = data.substring(6);

  bits.trim();


  if (bits.length() != 64)
  {

    Serial.print("[FRAME] ERROR: ");
    Serial.print(bits.length());
    Serial.println(" bits received. Need 64.");

    return false;
  }


  uint8_t newFrame[8] =
  {
    0,0,0,0,0,0,0,0
  };


  for (int row = 0; row < 8; row++)
  {

    uint8_t value = 0;


    for (int col = 0; col < 8; col++)
    {

      char c = bits[row * 8 + col];


      if (c == '1')
      {
        value |= (1 << col);
      }

      else if (c != '0')
      {

        Serial.println(
          "[FRAME] ERROR: Invalid character"
        );

        return false;
      }
    }


    newFrame[row] = value;
  }


  noInterrupts();

  for (int i = 0; i < 8; i++)
    frameBuffer[i] = newFrame[i];

  interrupts();


  return true;
}


// -----------------------------------------------------------------------------
// LED command parser
// -----------------------------------------------------------------------------

void processLED(String data)
{

  data.trim();


  if (!data.startsWith("LED:"))
    return;


  String value = data.substring(4);

  int comma = value.indexOf(',');


  // ----------------------------------------------------------
  // LED:1 ... LED:64
  // ----------------------------------------------------------

  if (comma == -1)
  {

    int led = value.toInt();


    if (led >= 1 && led <= 64)
    {

      int index = led - 1;

      int row = index / 8;

      int col = index % 8;

      setLED(row, col, true);


      Serial.print("[LED] ON ");
      Serial.print(row);
      Serial.print(",");
      Serial.println(col);
    }

    return;
  }


  // ----------------------------------------------------------
  // LED:ROW,COLUMN
  // ----------------------------------------------------------

  int row =
      value.substring(0, comma).toInt();

  int col =
      value.substring(comma + 1).toInt();


  setLED(row, col, true);
}


// -----------------------------------------------------------------------------
// TOGGLE command parser
// -----------------------------------------------------------------------------

void processToggle(String data)
{

  data.trim();


  if (!data.startsWith("TOGGLE:"))
    return;


  String value = data.substring(7);

  int comma = value.indexOf(',');


  if (comma == -1)
    return;


  int row =
      value.substring(0, comma).toInt();

  int col =
      value.substring(comma + 1).toInt();


  toggleLED(row, col);
}


// -----------------------------------------------------------------------------
// Display brightness
// -----------------------------------------------------------------------------

void processBrightness(String data)
{

  data.trim();


  if (!data.startsWith("BRIGHTNESS:"))
    return;


  int value =
      data.substring(11).toInt();


  brightness = constrain(value, 0, 100);


  Serial.print("[BRIGHTNESS] ");

  Serial.print(brightness);

  Serial.println("%");
}


// -----------------------------------------------------------------------------
// PATTERN command parser
// -----------------------------------------------------------------------------

void processPattern(String data)
{

  data.trim();


  if (!data.startsWith("PATTERN:"))
    return;


  String pattern =
      data.substring(8);


  pattern.toUpperCase();


  if (pattern == "HEART")
  {
    heartPattern();
  }

  else if (pattern == "SMILE")
  {
    smilePattern();
  }

  else if (pattern == "CLEAR")
  {
    clearMatrix();
  }
}


// -----------------------------------------------------------------------------
// Command dispatcher
// -----------------------------------------------------------------------------

void processCommand(String command)
{

  command.trim();


  if (command.length() == 0)
    return;


  Serial.print("[CMD] ");

  Serial.println(command);


  if (command.startsWith("FRAME:"))
  {
    processFrame(command);
    return;
  }


  if (command.startsWith("LED:"))
  {
    processLED(command);
    return;
  }


  if (command.startsWith("TOGGLE:"))
  {
    processToggle(command);
    return;
  }


  if (command.startsWith("BRIGHTNESS:"))
  {
    processBrightness(command);
    return;
  }


  if (command.startsWith("PATTERN:"))
  {
    processPattern(command);
    return;
  }


  if (command == "ALL_ON")
  {
    allLEDsOn();
    return;
  }


  if (command == "CLEAR")
  {
    clearMatrix();
    return;
  }


  Serial.println("[CMD] Unknown command");
}


// -----------------------------------------------------------------------------
// Wi-Fi connection
// -----------------------------------------------------------------------------
//
// IMPORTANT:
// We DO NOT repeatedly call WiFi.begin() while connecting.
//
// ============================================================

void connectToWiFi()
{

  Serial.println();

  Serial.println(
    "[NETWORK] Starting Wi-Fi connection..."
  );


  // Completely stop previous connection
  WiFi.disconnect(true);

  delay(300);


  // Station mode
  WiFi.mode(WIFI_STA);

  delay(100);


  Serial.print(
    "[NETWORK] SSID: "
  );

  Serial.println(WIFI_SSID);


  Serial.println(
    "[NETWORK] Calling WiFi.begin()..."
  );


  WiFi.begin(
    WIFI_SSID,
    WIFI_PASS
  );


  // ----------------------------------------------------------
  // WAIT FOR CONNECTION
  // ----------------------------------------------------------

  unsigned long startTime = millis();


  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - startTime < 20000
  )
  {

    delay(500);

    Serial.print(".");
  }


  Serial.println();


  // ----------------------------------------------------------
  // SUCCESS
  // ----------------------------------------------------------

  if (WiFi.status() == WL_CONNECTED)
  {

    Serial.println();

    Serial.println(
      "================================"
    );

    Serial.println(
      "       WIFI CONNECTED"
    );

    Serial.println(
      "================================"
    );


    Serial.print("SSID: ");

    Serial.println(
      WiFi.SSID()
    );


    Serial.print("IP: ");

    Serial.println(
      WiFi.localIP()
    );


    Serial.print("RSSI: ");

    Serial.print(
      WiFi.RSSI()
    );

    Serial.println(" dBm");


    Serial.print("UDP PORT: ");

    Serial.println(
      UDP_PORT
    );


    Serial.println(
      "================================"
    );

    Serial.println();


    // Start UDP ONLY after Wi-Fi succeeds
    udp.begin(UDP_PORT);

    udpStarted = true;

    return;
  }


  // ----------------------------------------------------------
  // FAILURE
  // ----------------------------------------------------------

  udpStarted = false;


  Serial.println();

  Serial.println(
    "[NETWORK] Wi-Fi connection FAILED."
  );


  Serial.print(
    "[NETWORK] Status = "
  );

  Serial.println(
    WiFi.status()
  );


  if (WiFi.status() == WL_CONNECT_FAILED)
  {

    Serial.println(
      "[NETWORK] WL_CONNECT_FAILED"
    );

    Serial.println(
      "[NETWORK] Check SSID/password and hotspot."
    );
  }


  Serial.println(
    "[NETWORK] Will retry."
  );
}


// -----------------------------------------------------------------------------
// Wi-Fi state check / reconnect
// -----------------------------------------------------------------------------

void checkWiFi()
{

  if (WiFi.status() == WL_CONNECTED)
  {

    if (!udpStarted)
    {

      udp.begin(UDP_PORT);

      udpStarted = true;


      Serial.println(
        "[NETWORK] UDP restarted."
      );
    }


    return;
  }


  // ----------------------------------------------------------
  // DISCONNECTED
  // ----------------------------------------------------------

  udpStarted = false;


  unsigned long now = millis();


  if (
    now - lastWiFiAttempt >=
    WIFI_RETRY_INTERVAL
  )
  {

    lastWiFiAttempt = now;

    connectToWiFi();
  }
}


// -----------------------------------------------------------------------------
// UDP receiver
// -----------------------------------------------------------------------------

void checkUDP()
{

  if (!udpStarted)
    return;


  int packetSize =
      udp.parsePacket();


  if (packetSize <= 0)
    return;


  String command = "";


  while (udp.available())
  {

    command +=
      (char)udp.read();
  }


  processCommand(command);
}


// -----------------------------------------------------------------------------
// Serial command receiver
// -----------------------------------------------------------------------------

void checkSerial()
{

  if (!Serial.available())
    return;


  String command =
      Serial.readStringUntil('\n');


  processCommand(command);
}


// -----------------------------------------------------------------------------
// Matrix scan scheduler
// -----------------------------------------------------------------------------

void scanMatrix()
{

  unsigned long now = micros();


  if (
    now - lastScanTime <
    ROW_PERIOD_US
  )
    return;


  lastScanTime = now;


  displayRow(currentRow);


  currentRow++;


  if (currentRow >= 8)
    currentRow = 0;
}


// -----------------------------------------------------------------------------
// Arduino setup
// -----------------------------------------------------------------------------

void setup()
{

  Serial.begin(115200);


  delay(1000);


  Serial.println();

  Serial.println(
    "======================================"
  );

  Serial.println(
    "       AIRTOUCH-88 CONTROLLER"
  );

  Serial.println(
    "======================================"
  );


  // ----------------------------------------------------------
  // GPIO
  // ----------------------------------------------------------

  pinMode(
    DATA_PIN,
    OUTPUT
  );

  pinMode(
    CLOCK_PIN,
    OUTPUT
  );

  pinMode(
    LATCH_PIN,
    OUTPUT
  );


  digitalWrite(
    DATA_PIN,
    LOW
  );

  digitalWrite(
    CLOCK_PIN,
    LOW
  );

  digitalWrite(
    LATCH_PIN,
    LOW
  );


  // ----------------------------------------------------------
  // MATRIX OFF
  // ----------------------------------------------------------

  allOutputsOff();


  Serial.println(
    "[SYSTEM] Matrix initialized."
  );


  // ----------------------------------------------------------
  // WIFI
  // ----------------------------------------------------------

  connectToWiFi();


  // ----------------------------------------------------------
  // COMMANDS
  // ----------------------------------------------------------

  Serial.println();

  Serial.println(
    "[SYSTEM] Commands:"
  );

  Serial.println(
    "FRAME:<64 bits>"
  );

  Serial.println(
    "LED:<1-64>"
  );

  Serial.println(
    "LED:<row,col>"
  );

  Serial.println(
    "TOGGLE:<row,col>"
  );

  Serial.println(
    "BRIGHTNESS:<0-100>"
  );

  Serial.println(
    "PATTERN:HEART"
  );

  Serial.println(
    "PATTERN:SMILE"
  );

  Serial.println(
    "ALL_ON"
  );

  Serial.println(
    "CLEAR"
  );

  Serial.println();
}


// -----------------------------------------------------------------------------
// Arduino main loop
// -----------------------------------------------------------------------------

void loop()
{

  // Matrix
  scanMatrix();


  // Wi-Fi
  checkWiFi();


  // UDP
  checkUDP();


  // Serial
  checkSerial();


  delay(1);
}
