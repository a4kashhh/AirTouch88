# AirTouch-88: Computer Vision Controlled Custom 8×8 LED Matrix

This project connects a custom-built, discrete MOSFET-switched 8×8 red LED matrix powered by an **ESP32-C3** to a real-time **Computer Vision** interface running on Python (OpenCV + MediaPipe).

---

## Project Structure

```
├── esp32_c3_matrix/
│   └── esp32_c3_matrix.ino       # ESP32-C3 Arduino firmware
└── python_cv/
    ├── requirements.txt          # Python dependencies
    └── cv_matrix_controller.py   # MediaPipe OpenCV tracking application
```

---

## 1. Hardware Architecture & Wiring Summary

### Components
* 64 × Red LEDs (Anodes on Rows 0–7, Cathodes on Columns 0–7)
* 8 × IRF9540N P-channel MOSFETs (High-side Row switches connected to +5V)
* 8 × BC547 NPN transistors (Level shifters & gate inverters for IRF9540N)
* 8 × 10 kΩ pull-up resistors (Between IRF9540N Gates and +5V)
* 8 × 1 kΩ base resistors (Between ESP32 GPIOs and BC547 Bases)
* 8 × 2N7000 N-channel MOSFETs (Low-side Column switches connected to GND)
* 8 × 220 Ω resistors (Current limiting, **placed on Column lines**)
* 8 × 100 kΩ pull-down resistors (On 2N7000 Gates to GND)
* 1 × 470 µF electrolytic capacitor (Decoupling across +5V and GND rail)
* 5 V 2 A DC power supply

### Pin Mapping (ESP32-C3)
* **Row Anodes (High-side via BC547 $\to$ IRF9540N)**:
  `GPIO 0, 1, 2, 3, 4, 5, 6, 7`
* **Column Cathodes (Low-side via 220 Ω $\to$ 2N7000)**:
  `GPIO 8, 10, 18, 19, 20, 21, 9, 11`

---

## 2. Setting Up the ESP32-C3 Firmware

1. Open `esp32_c3_matrix/esp32_c3_matrix.ino` in the Arduino IDE.
2. Under **Tools $\to$ Board**, select **ESP32C3 Dev Module**.
3. Update your Wi-Fi credentials:
   ```cpp
   const char* WIFI_SSID = "Your_WiFi_SSID";
   const char* WIFI_PASS = "Your_WiFi_Password";
   ```
4. Flash the code to the ESP32-C3.
5. Open Serial Monitor (115200 baud) to note the assigned IP address (e.g., `10.194.177.102`).

---

## 3. Running the Python Computer Vision App

1. Open terminal and create/activate a virtual environment:
   ```bash
   cd python_cv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the controller with your ESP32-C3 IP address:
   ```bash
   python3 cv_matrix_controller.py --ip 10.194.177.102
   ```
   *(If testing with a USB cable plugged in, you can pass `--serial /dev/cu.usbmodem...`)*

---

## 4. Interaction Controls

* **Point Index Finger**: Point across the 8 zones on the screen. The debounced zone sends `LED:<1-8>` to turn ON only that position indicator.
* **Vertical Height Tracking (BMW-Style)**: Move your hand up and down to smoothly adjust brightness between 0% and 100%.
* **Pinch Distance Tracking**: Press `'g'` to toggle to pinch-based brightness control (distance between thumb and index fingertip).
* **Peace Sign / V Gesture**: Starts scrolling text `"HELLO"` (`MESSAGE:HELLO`).
* **Fist Gesture**: Starts scrolling text `"HOW YOU DOING?"` (`MESSAGE:HOW YOU DOING?`).
* **Keyboard Hotkeys**:
  * `1`–`8`: Select LED position 1 to 8
  * `H`: Display Heart
  * `S`: Display Smile
  * `M`: Scroll "HELLO"
  * `D`: Scroll "HOW YOU DOING?"
  * `P`: Pulsing Heartbeat animation
  * `C`: Clear screen
  * `G`: Toggle Brightness mode
  * `Q`: Exit
