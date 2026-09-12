#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ==================== WI-FI SETTINGS ====================
extern const char* WIFI_SSID;
extern const char* WIFI_PASS;
const uint16_t UDP_PORT = 8888;

// ==================== PIN DEFINITIONS ====================
// Row Anodes (Driven HIGH via BC547 -> IRF9540N P-FETs)
const uint8_t ROW_PINS[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// Column Cathodes (Driven HIGH via 2N7000 N-FETs -> 220 Ohm -> GND)
const uint8_t COL_PINS[8] = {8, 10, 18, 19, 20, 21, 9, 11};

// ==================== HARDWARE TIMINGS ====================
// 8 rows * 2500 us = 20 ms full frame (50 Hz refresh rate)
const uint32_t ROW_DWELL_US = 2500; 
const uint32_t DEAD_TIME_US = 15;   // Hardware turn-off blanking for ghosting prevention

#endif // CONFIG_H
