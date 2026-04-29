// =============================================================
//  Hotplate PRBS System Identification
//  Adapted from ovenPRBS.ino for dual-NTC / IRFZ44N MOSFET hotplate
//
//  HOW TO USE — run this sketch FOUR times, one per configuration:
//
//    Run 1:  ACTIVE_HEATER = 1,  FAN_ENABLED = false  → save as heater1_fanoff.csv
//    Run 2:  ACTIVE_HEATER = 2,  FAN_ENABLED = false  → save as heater2_fanoff.csv
//    Run 3:  ACTIVE_HEATER = 1,  FAN_ENABLED = true   → save as heater1_fanon.csv
//    Run 4:  ACTIVE_HEATER = 2,  FAN_ENABLED = true   → save as heater2_fanon.csv
//
//  Output CSV columns match calculateMPCMatrices.py exactly:
//    time_ms, t1_c, t2_c, pwm1, pwm2
//  (Only the active heater's pwm column is non-zero.)
//
//  Send 's' over Serial at any time to replay the full stored log.
// =============================================================

#include <Arduino.h>
#include <math.h>

//  --- CONFIGURE BEFORE EACH RUN ---
#define ACTIVE_HEATER  1      // 1 = right (H1),  2 = left (H2)

// --- Thermistor pins & parameters (from thermistor_read.ino) ---
#define PIN_THERM_1      1  // Right thermistor
#define PIN_THERM_2      2  // Left thermistor
#define R_FIXED          10000.0f
#define R_NOMINAL        10000.0f
#define T0_KELVIN        298.15f
#define BETA             3950.0f
#define ADC_MAX          4095.0f
#define ADC_COUNT_MIN    100
#define ADC_COUNT_MAX    3950
#define N_OVERSAMPLE     16       // ADC oversampling passes

// --- MOSFET gate pins ---
// Each IRFZ44N only needs a single PWM pin: ESP32 GPIO → gate resistor → gate.
// Drain connects to heater negative terminal; source to GND.
// NOTE: IRFZ44N is a 10V gate-drive part. At 3.3V from the ESP32 it will
// turn on but with higher Rds(on) than spec (~0.1Ω vs 0.028Ω). This is fine
// for a low-duty-cycle heater. If you see thermal runaway in the FET itself,
// add a small N-channel logic-level driver (e.g. 2N7000) or gate driver IC.
#define H1_GATE_PIN  17  // Right heater MOSFET gate
#define H2_GATE_PIN  18  // Left heater MOSFET gate

// With direct MOSFET drive the full 8-bit PWM range is available.
// PWM_MAX_KNEE stays at 225 — this was a current/power limit of the
// heater circuit, not an L298N artifact, so it should remain the same.
#define PWM_MAX_KNEE  225

// Safety cutoff — heaters off above this temperature
#define TEMP_CUTOFF_C 70.0f

// =============================================================
//  PRBS Timing
// =============================================================
//  Oven (advisor reference):
//    measurememtUpdateRate_ms = 200,000 ms (200 s)
//    Longest ON cycle = 31 × 200 s  ≈ 103 minutes
//    Rise time ≈ 15 minutes
//
//  Hotplate (rise time ≈ 100× faster ≈ 5–10 seconds):
//    measurememtUpdateRate_ms = 2,000 ms  (matches MPC sample rate)
//    Shortest ON/OFF cycle   = 1 × 2 s   =  2 seconds
//    Longest  ON/OFF cycle   = 31 × 2 s  = 62 seconds  (~1 min)
//
//  Full sequence length (5-bit LFSR visits all 31 non-zero states):
//    Total = Σ(1..31) × 2000 ms = 496 × 2 s = 992 s ≈ 16.5 minutes/run
//    Four runs ≈ 66 minutes total — well within the experiment budget.
//
//  If your observed rise time is slower (e.g. 20–30 s), increase
//  measurememtUpdateRate_ms to 4000–6000 ms accordingly.
// =============================================================

static const uint32_t MEASUREMENT_RATE_MS = 2000;

// 5-bit maximal-length LFSR — same tap polynomial as ovenPRBS
// Period = 2^5 − 1 = 31 states, covers all non-zero 5-bit values
#define LFSR_INITIAL_STATE  0x16
uint16_t lfsr = LFSR_INITIAL_STATE;

// Data log — sized for one full run plus margin
// 16.5 min × (1 sample / 2 s) = ~495 samples; 800 gives comfortable headroom
#define MAX_RECORDS  800

struct HotplateLogEntry
{
  uint32_t timestamp_ms; // Time since start of run
  float t1_c;            // Thermistor 1 temperature (°C)
  float t2_c;            // Thermistor 2 temperature (°C)
  int pwm1;              // Heater 1 PWM command (0–255)
  int pwm2;              // Heater 2 PWM command (0–255)
};

HotplateLogEntry dataLog[MAX_RECORDS];
static size_t dataCount = 0;

// Tracks the live PWM command for telemetry
static int activePWM = 0;

float readThermistor(int pin)
{
  long sum = 0;
  for (int i = 0; i < N_OVERSAMPLE; i++) 
  {
    sum += analogRead(pin);
    delayMicroseconds(100);
  }
  int raw = (int)(sum / N_OVERSAMPLE);

  if (raw < ADC_COUNT_MIN || raw > ADC_COUNT_MAX) 
  {
    return -999.0f; // sensor error / out of range
  }

  // Voltage divider → resistance (thermistor tied to GND, fixed resistor to 3.3V)
  float resistance = R_FIXED * ((float)raw / (ADC_MAX - (float)raw));

  // Beta equation → Kelvin → Celsius
  float tempK = 1.0f / (1.0f / T0_KELVIN + (1.0f / BETA) * log(resistance / R_NOMINAL));
  return tempK - 273.15f;
}

void setHeaterPWM(int heater, int pwmValue) 
{
  // Simple low-side MOSFET: PWM the gate directly.
  // HIGH gate = FET on = current through heater.
  pwmValue = constrain(pwmValue, 0, PWM_MAX_KNEE);
  uint8_t pin = (heater == 1) ? H1_GATE_PIN : H2_GATE_PIN;
  analogWrite(pin, pwmValue);
}

void allHeatersOff() 
{
  analogWrite(H1_GATE_PIN, 0);
  analogWrite(H2_GATE_PIN, 0);
  activePWM = 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // ADC config
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_THERM_1, ADC_11db);
  analogSetPinAttenuation(PIN_THERM_2, ADC_11db);

  // MOSFET gate pins — single OUTPUT per heater, no direction pins needed
  pinMode(H1_GATE_PIN, OUTPUT);
  pinMode(H2_GATE_PIN, OUTPUT);
  allHeatersOff();

  // Print configuration banner
  Serial.println("# ============================================");
  Serial.println("# Hotplate PRBS System Identification");
  Serial.printf ("# Active heater : H%d\n", ACTIVE_HEATER);
  //Serial.printf ("# Fan           : %s\n",  FAN_ENABLED ? "ON" : "OFF");
  Serial.printf ("# Sample rate   : %lu ms\n", MEASUREMENT_RATE_MS);
  Serial.printf ("# Shortest cycle: %lu ms (%.1f s)\n",
                  1  * MEASUREMENT_RATE_MS, 1  * MEASUREMENT_RATE_MS / 1000.0f);
  Serial.printf ("# Longest cycle : %lu ms (%.1f s)\n",
                  31 * MEASUREMENT_RATE_MS, 31 * MEASUREMENT_RATE_MS / 1000.0f);
  Serial.printf ("# Est. run time : ~%.0f min\n",
                  496.0f * MEASUREMENT_RATE_MS / 60000.0f);
  Serial.println("# Safety cutoff : " + String(TEMP_CUTOFF_C) + " C");
  Serial.println("# Send 's' to dump stored CSV log.");
  Serial.println("# ============================================");

  // CSV header — columns match calculateMPCMatrices.py exactly
  Serial.println("time_ms,t1_c,t2_c,pwm1,pwm2");
}

void loop()
{
  static bool     areWeFinished  = false;
  static bool     shouldMeasure  = true;
  static uint32_t lastPrbsTime   = MEASUREMENT_RATE_MS; // first PRBS tick
  static uint32_t prbsHoldMs     = 0;                   // duration of current state
  static uint32_t lastSampleTime = 0;
  static bool     pinToggle      = false;

  // ----------------------------------------------------------
  //  PRBS State Machine
  // ----------------------------------------------------------

  if (millis() >= lastPrbsTime + prbsHoldMs)
  {
    lastPrbsTime += prbsHoldMs;

    // New hold duration = base_rate × current LFSR value  (same as ovenPRBS)
    prbsHoldMs = MEASUREMENT_RATE_MS * lfsr;

    // Toggle heater state
    pinToggle ^= 1;

    if (!areWeFinished)
    {
      activePWM = pinToggle ? PWM_MAX_KNEE : 0;
      setHeaterPWM(ACTIVE_HEATER, activePWM);
    } 
    else
    {
      allHeatersOff();
    }

    // Advance 5-bit maximal LFSR  (taps: bits 4 and 2)
    uint8_t bit = (((lfsr >> 4) ^ (lfsr >> 2)) & 1);
    lfsr = ((lfsr << 1) | bit) & 0x1F;

    if (lfsr == LFSR_INITIAL_STATE)
    {
      areWeFinished = true;
      allHeatersOff();
      Serial.println("# PRBS sequence complete — heaters off.");
    }

    shouldMeasure = true; // always log at a PRBS transition
  }

  // ----------------------------------------------------------
  //  Periodic Measurement (every MEASUREMENT_RATE_MS)
  // ----------------------------------------------------------

  if (millis() >= lastSampleTime + MEASUREMENT_RATE_MS) 
  {
    shouldMeasure = true;
  }

  if (shouldMeasure) 
  {
    shouldMeasure = false;
    lastSampleTime += MEASUREMENT_RATE_MS;

    float t1 = readThermistor(PIN_THERM_1);
    float t2 = readThermistor(PIN_THERM_2);

    // --- Safety cutoff ---
    float maxTemp = max(t1, t2);
    if (maxTemp > TEMP_CUTOFF_C)
    {
      allHeatersOff();
      areWeFinished = true;
      Serial.printf("# SAFETY CUTOFF: %.1f C exceeded %.0f C limit — heaters off!\n",
                    maxTemp, TEMP_CUTOFF_C);
    }

    uint32_t timeNow = millis();
    int pwm1 = (ACTIVE_HEATER == 1) ? activePWM : 0;
    int pwm2 = (ACTIVE_HEATER == 2) ? activePWM : 0;

    // Store in RAM buffer
    if (dataCount < MAX_RECORDS) {
      dataLog[dataCount++] = { timeNow, t1, t2, pwm1, pwm2 };
    }

    // Live stream to Serial (log from Arduino Serial Monitor / plotter)
    Serial.printf("%lu,%.2f,%.2f,%d,%d\n", timeNow, t1, t2, pwm1, pwm2);
  }

  // ----------------------------------------------------------
  //  's' command — dump full RAM log as CSV
  // ----------------------------------------------------------
  
  if (Serial.available()) 
  {
    char c = Serial.read();
    if (c == 's') 
    {
      Serial.println("# --- Begin CSV Dump ---");
      Serial.println("time_ms,t1_c,t2_c,pwm1,pwm2");
      for (size_t i = 0; i < dataCount; i++) 
      {
        Serial.printf("%lu,%.2f,%.2f,%d,%d\n",
          dataLog[i].timestamp_ms,
          dataLog[i].t1_c,
          dataLog[i].t2_c,
          dataLog[i].pwm1,
          dataLog[i].pwm2);
      }
      Serial.println("# --- End CSV Dump ---");
    }
  }
}
