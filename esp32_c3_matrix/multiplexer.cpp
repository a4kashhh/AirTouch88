#include "multiplexer.h"

void initMultiplexer() {
  for (uint8_t i = 0; i < 8; i++) {
    pinMode(ROW_PINS[i], OUTPUT);
    digitalWrite(ROW_PINS[i], LOW); // BC547 OFF -> IRF9540N OFF
    
    pinMode(COL_PINS[i], OUTPUT);
    digitalWrite(COL_PINS[i], LOW); // 2N7000 OFF
  }
}

void displaySingleRow(uint8_t row, const volatile uint8_t* buffer, uint8_t brightness) {
  // STEP 1: Blank previous row to eliminate ghosting
  for (uint8_t r = 0; r < 8; r++) {
    digitalWrite(ROW_PINS[r], LOW);
  }

  // STEP 2: Blank all columns
  for (uint8_t c = 0; c < 8; c++) {
    digitalWrite(COL_PINS[c], LOW);
  }

  // STEP 3: Hardware Dead-Time
  delayMicroseconds(DEAD_TIME_US);

  uint8_t rowData = buffer[row];
  if (brightness == 0 || rowData == 0) {
    delayMicroseconds(ROW_DWELL_US - DEAD_TIME_US);
    return;
  }

  // STEP 4: Set column cathode states
  for (uint8_t col = 0; col < 8; col++) {
    if (rowData & (1 << (7 - col))) {
      digitalWrite(COL_PINS[col], HIGH); // 2N7000 ON -> sink to GND
    } else {
      digitalWrite(COL_PINS[col], LOW);
    }
  }

  // STEP 5: Turn ON active row anode
  digitalWrite(ROW_PINS[row], HIGH); // BC547 ON -> IRF9540N ON

  // STEP 6: Sub-cycle Software PWM
  uint32_t activeDwellUs = (ROW_DWELL_US * brightness) / 100;
  if (activeDwellUs > DEAD_TIME_US) {
    delayMicroseconds(activeDwellUs - DEAD_TIME_US);
  }

  // Blank row early if brightness < 100%
  if (brightness < 100) {
    digitalWrite(ROW_PINS[row], LOW);
    uint32_t remainingDwellUs = ROW_DWELL_US - activeDwellUs;
    if (remainingDwellUs > 0) {
      delayMicroseconds(remainingDwellUs);
    }
  }
}
