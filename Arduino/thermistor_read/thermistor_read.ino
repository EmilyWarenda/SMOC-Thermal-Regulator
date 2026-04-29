/*
  thermistor_read.ino  —  ESP32-S3-WROOM-1 edition
  ──────────────────────────────────────────────────
  Reads three NTC thermistors and prints temperatures to Serial Monitor.

  Wiring (repeat for each thermistor):
    3.3V ── 10kΩ ── GPIO4 ── NTC thermistor ── GND
    3.3V ── 10kΩ ── GPIO5 ── NTC thermistor ── GND

  Board setting in Arduino IDE:
    Tools → Board   → ESP32S3 Dev Module
    Tools → Port    → (your port)
    Serial Monitor baud: 115200
*/

#include <Arduino.h>
#include <math.h>   // for log()

// ── Pin assignments (safe ADC1 pins on ESP32-S3-WROOM-1) ─────────────────────
//
//  ADC1 lives on GPIO1–GPIO10 on the S3.
//  We skip GPIO1–3 (can interfere with UART/boot on some boards)
//  and use GPIO4, 5, 6 — clean, no special functions, ADC1 capable.
//
#define PIN_THERM_1   1
#define PIN_THERM_2   2

// ── Thermistor parameters — update from your datasheet ───────────────────────
#define R_FIXED       10000.0f   // series resistor value (Ω)
#define R_NOMINAL     10000.0f   // thermistor resistance at T_NOMINAL (Ω)
#define T_NOMINAL_C   25.0f      // reference temperature (°C)
#define BETA          3950.0f    // B-coefficient (find on your datasheet)

// ── ADC settings ──────────────────────────────────────────────────────────────
#define ADC_RESOLUTION  12              // 12-bit → counts 0–4095
#define ADC_MAX         4095
#define V_SUPPLY        3.3f
#define N_SAMPLES       16              // oversample to reduce noise

// ── Safe ADC count range ──────────────────────────────────────────────────────
//  The ESP32-S3 ADC is inaccurate below ~100 and above ~3950 counts.
//  Readings outside this range mean the thermistor is open, shorted,
//  or the temperature is beyond what this divider can measure.
#define ADC_COUNT_MIN   100
#define ADC_COUNT_MAX   3950

// ── Print interval ────────────────────────────────────────────────────────────
#define PRINT_INTERVAL_MS  1000


// ─────────────────────────────────────────────────────────────────────────────
//  readThermistor(pin)
//
//  Oversamples the ADC, converts to resistance via voltage divider,
//  then applies the Beta equation to get °C.
//  Returns -999.0f on error (open/short circuit or out-of-range).
// ─────────────────────────────────────────────────────────────────────────────
float readThermistor(int pin)
{
  // 1. Oversample
  long sum = 0;
  for (int i = 0; i < N_SAMPLES; i++)
  {
    sum += analogRead(pin);
    delayMicroseconds(100);
  }
  int raw = (int)(sum / N_SAMPLES);

  // 2. Range check — reject rail-saturated readings
  if (raw < ADC_COUNT_MIN || raw > ADC_COUNT_MAX) 
  {
    return -999.0f;
  }

  // 3. Count → voltage
  float v_mid = (float)raw * V_SUPPLY / (float)ADC_MAX;

  // 4. Voltage → thermistor resistance
  //    (voltage divider: v_mid = V_SUPPLY * R_therm / (R_FIXED + R_therm))
  float r_therm = R_FIXED * v_mid / (V_SUPPLY - v_mid);

  // 5. Beta equation → Kelvin → Celsius
  float t_nom_k  = T_NOMINAL_C + 273.15f;
  float t_kelvin = 1.0f / (1.0f / t_nom_k + log(r_therm / R_NOMINAL) / BETA);

  return t_kelvin - 273.15f;
}


// ─────────────────────────────────────────────────────────────────────────────
//  printTemp()  —  formatted output for one sensor
// ─────────────────────────────────────────────────────────────────────────────

void printTemp(const char* label, int pin, float tempC)
{
  Serial.print(label);
  Serial.print(" (GPIO");
  Serial.print(pin);
  Serial.print("): ");

  if (tempC <= -998.0f) 
  {
    Serial.println("ERROR — check wiring or thermistor range");
  } 
  else 
  {
    Serial.print(tempC, 2);
    Serial.print(" C  |  ");
    Serial.print(tempC * 9.0f / 5.0f + 32.0f, 2);
    Serial.println(" F");
  }
}


// ─────────────────────────────────────────────────────────────────────────────

void setup()
{
  Serial.begin(115200);
  delay(500);

  // Set ADC resolution to 12-bit
  analogReadResolution(ADC_RESOLUTION);

  // Set attenuation to 11dB on each pin → full 0–3.3V input range.
  // Note: some newer ESP32 Arduino core versions use ADC_ATTEN_DB_12
  // instead of ADC_11db. If you get a compile error here, change
  // ADC_11db → ADC_ATTEN_DB_12 in all three lines below.
  analogSetPinAttenuation(PIN_THERM_1, ADC_11db);
  analogSetPinAttenuation(PIN_THERM_2, ADC_11db);

  Serial.println();
  Serial.println("======================================");
  Serial.println("  ESP32-S3-WROOM-1  Thermistor Reader");
  Serial.println("======================================");
  Serial.print  ("  Beta:      "); Serial.println(BETA);
  Serial.print  ("  R_fixed:   "); Serial.print(R_FIXED); Serial.println(" Ohm");
  Serial.print  ("  R_nominal: "); Serial.print(R_NOMINAL); Serial.println(" Ohm @ 25C");
  Serial.println("======================================");
}


// ─────────────────────────────────────────────────────────────────────────────

void loop() {
  float t1 = readThermistor(PIN_THERM_1);
  float t2 = readThermistor(PIN_THERM_2);

  Serial.println("--------------------------------------");
  printTemp("Thermistor 1", PIN_THERM_1, t1);
  printTemp("Thermistor 2", PIN_THERM_2, t2);

  // Average of whichever sensors are valid
  int n = 0;
  float total = 0.0f;
  if (t1 > -998.0f) { total += t1; n++; }
  if (t2 > -998.0f) { total += t2; n++; }

  if (n > 0) {
    Serial.print("Average:               ");
    Serial.print(total / (float)n, 2);
    Serial.println(" C");
  } else {
    Serial.println("Average:               No valid sensors");
  }

  delay(PRINT_INTERVAL_MS);
}
