#ifndef BITMAPS_H
#define BITMAPS_H

#include <Arduino.h>

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

#endif // BITMAPS_H
