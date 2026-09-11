#ifndef MULTIPLEXER_H
#define MULTIPLEXER_H

#include <Arduino.h>
#include "config.h"

void initMultiplexer();
void displaySingleRow(uint8_t row, const volatile uint8_t* buffer, uint8_t brightness);

#endif // MULTIPLEXER_H
