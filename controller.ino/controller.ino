// =============================================================
//  Dual-Zone MIMO Receding Horizon Controller (SMOC / MPC)
//  Includes Actuator Saturation limits & Ambient Offsetting
// =============================================================
#include <Arduino.h>

#define SAMPLE_INTERVAL_MS 2000 // Must match the 2000ms T_sample from Python
#define HORIZON_STEPS      20   // Prediction horizon N

// --- Hardware Calibration ---
const float R_FIXED       = 10000.0f; 
const float T0_KELVIN     = 298.15f;  
const float BETA          = 3950.0f;  
const float R0            = 10000.0f; 
const float ADC_MAX       = 4095.0f;  

const float PWM_MAX_KNEE  = 225.0f;   // The hardware current limit saturation point

// --- Data Structures ---
struct Thermistor {
  uint8_t pin;       
  float temp;        
};

struct Heater {
  uint8_t pinA;      
  uint8_t pinB;      
  uint8_t pinEN;     
  int currentPWM;    // Tracks live PWM command for CSV telemetry
};

struct MIMO_SMOC {
  float KX[2][2];
  float KU1[2][2];
  float KYD[2][HORIZON_STEPS * 2]; 
  float L[2][2];
  float A[2][2];
  float B[2][2];
  float C[2][2]; 
  float x_hat[2];    
  float u_prev[2];   
  float setpoint[2]; 
};

// --- Instantiation ---
Thermistor t1 = {1, 0.0f};  
Thermistor t2 = {2, 0.0f};  
Heater h1 = {6, 7, 5, 0};   
Heater h2 = {8, 9, 10, 0};  

// State Tracking Variables
unsigned long lastSampleTime = 0; 
float ambientT[2] = {0.0f, 0.0f}; // Stores room temperature baseline

// =============================================================
//  The Optimal Control Matrices (Imported from Python)
// =============================================================
MIMO_SMOC controller = {
  // KX (State Penalty)
  {{-4.8494f, -0.3241f},
   {-0.3224f, -4.6813f}},
  // KU1 (Input Smoothness Penalty)
  {{ 0.2392f,  0.0228f},
   { 0.0228f,  0.2257f}},
  // KYD (Trajectory Tracking)
  {{ 2.2736f, -0.0539f,  1.7545f,  0.0883f,  0.8213f,  0.1171f,  0.2201f,  0.0702f, -0.0262f,  0.0208f, -0.0745f, -0.0042f, -0.0527f, -0.0095f, -0.0235f, -0.0065f, -0.0057f, -0.0027f,  0.0013f, -0.0005f,  0.0025f,  0.0003f,  0.0017f,  0.0003f,  0.0007f,  0.0002f,  0.0002f,  0.0000f, -0.0001f, -0.0000f, -0.0001f, -0.0000f, -0.0001f, -0.0000f, -0.0000f,  0.0000f, -0.0000f,  0.0000f,  0.0000f, -0.0000f},
   {-0.0572f,  2.3093f,  0.0851f,  1.7029f,  0.1155f,  0.7480f,  0.0701f,  0.1737f,  0.0213f, -0.0412f, -0.0037f, -0.0722f, -0.0093f, -0.0459f, -0.0066f, -0.0184f, -0.0028f, -0.0033f, -0.0006f,  0.0019f,  0.0002f,  0.0023f,  0.0003f,  0.0013f,  0.0002f,  0.0005f,  0.0001f,  0.0001f,  0.0000f, -0.0001f, -0.0000f, -0.0001f, -0.0000f, -0.0000f, -0.0000f, -0.0000f, -0.0000f, -0.0000f, -0.0000f,  0.0000f}},
  // Kalman L (Observer Gain)
  {{ 0.2661f,  0.0088f},
   { 0.0088f,  0.2633f}},
  // Model A (System Dynamics)
  {{ 0.9932f,  0.0141f},
   { 0.0152f,  0.9884f}},
  // Model B (Input Actuation)
  {{ 0.0962f, -0.0122f},
   {-0.0121f,  0.1036f}},
  // Model C (Output Mapping)
  {{ 1.0000f,  0.0000f},
   { 0.0000f,  1.0000f}},
  {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f} // Initialize x_hat, u_prev, setpoint to pure zero (off)
};

// =============================================================
//  Hardware Functions
// =============================================================
void readTemp(Thermistor &t) {
  int adcVal = analogRead(t.pin); 
  if (adcVal <= 0 || adcVal >= ADC_MAX) return; 
  
  // Algebra fix: Assumes Thermistor is tied to ground
  float resistance = R_FIXED * ((float)adcVal / (ADC_MAX - (float)adcVal));       
  float tempK = 1.0f / (1.0f / T0_KELVIN + (1.0f / BETA) * log(resistance / R0)); 
  t.temp = tempK - 273.15f; 
}

void setPower(Heater &h, float ratio) {
  ratio = constrain(ratio, 0.0f, 1.0f);            // Ensure MPC ratio is 0-100%
  
  // SATURATION FIX: Maps 1.0 to our specific 2.5A knee limit
  int pwmValue = (int)(ratio * PWM_MAX_KNEE);      
  h.currentPWM = pwmValue;                         // Store for telemetry
  
  digitalWrite(h.pinA, pwmValue > 0 ? HIGH : LOW); 
  digitalWrite(h.pinB, LOW);                       
  analogWrite(h.pinEN, pwmValue);                  
}

// =============================================================
//  MIMO MPC Execution
// =============================================================
void executeMIMO() {
  // Convert physical absolute temperatures to Relative Delta Temperatures
  float dt1 = t1.temp - ambientT[0];
  float dt2 = t2.temp - ambientT[1];

  // 1. Observer Update (Kalman Filter)
  float x_pred[2], y_pred[2], innov[2];
  
  x_pred[0] = (controller.A[0][0]*controller.x_hat[0] + controller.A[0][1]*controller.x_hat[1]) + 
              (controller.B[0][0]*controller.u_prev[0] + controller.B[0][1]*controller.u_prev[1]);
  x_pred[1] = (controller.A[1][0]*controller.x_hat[0] + controller.A[1][1]*controller.x_hat[1]) + 
              (controller.B[1][0]*controller.u_prev[0] + controller.B[1][1]*controller.u_prev[1]);

  y_pred[0] = (controller.C[0][0]*x_pred[0] + controller.C[0][1]*x_pred[1]);
  y_pred[1] = (controller.C[1][0]*x_pred[0] + controller.C[1][1]*x_pred[1]);

  innov[0] = dt1 - y_pred[0];
  innov[1] = dt2 - y_pred[1];

  controller.x_hat[0] = x_pred[0] + (controller.L[0][0]*innov[0] + controller.L[0][1]*innov[1]);
  controller.x_hat[1] = x_pred[1] + (controller.L[1][0]*innov[0] + controller.L[1][1]*innov[1]);

  // 2. Trajectory Calculation
  // We only feed the MPC math a target if the user setpoint is greater than 0.
  float target_delta[2] = {0.0f, 0.0f};
  if (controller.setpoint[0] > 0.0f) target_delta[0] = max(0.0f, controller.setpoint[0] - ambientT[0]);
  if (controller.setpoint[1] > 0.0f) target_delta[1] = max(0.0f, controller.setpoint[1] - ambientT[1]);

  float trajSum[2] = {0.0f, 0.0f};
  for(int i = 0; i < HORIZON_STEPS; i++) {
    trajSum[0] += (controller.KYD[0][i*2] * target_delta[0]) + (controller.KYD[0][i*2 + 1] * target_delta[1]);
    trajSum[1] += (controller.KYD[1][i*2] * target_delta[0]) + (controller.KYD[1][i*2 + 1] * target_delta[1]);
  }

  // 3. Control Law Computation
  float u_k[2] = {0.0f, 0.0f};
  
  // Only calculate effort if the user wants the heater ON
  if (controller.setpoint[0] > 0.0f) {
    u_k[0] = (controller.KX[0][0]*controller.x_hat[0] + controller.KX[0][1]*controller.x_hat[1]) + 
             (controller.KU1[0][0]*controller.u_prev[0] + controller.KU1[0][1]*controller.u_prev[1]) + trajSum[0];
  }
  if (controller.setpoint[1] > 0.0f) {
    u_k[1] = (controller.KX[1][0]*controller.x_hat[0] + controller.KX[1][1]*controller.x_hat[1]) + 
             (controller.KU1[1][0]*controller.u_prev[0] + controller.KU1[1][1]*controller.u_prev[1]) + trajSum[1];
  }

  // 4. Actuator Saturation & Memory
  u_k[0] = constrain(u_k[0], 0.0f, 1.0f);
  u_k[1] = constrain(u_k[1], 0.0f, 1.0f);

  controller.u_prev[0] = u_k[0];
  controller.u_prev[1] = u_k[1];

  setPower(h1, u_k[0]);
  setPower(h2, u_k[1]);
}

// =============================================================
//  Setup & Main Loop
// =============================================================
void setup() {
  Serial.begin(115200); 
  analogReadResolution(12); 
  
  pinMode(h1.pinA, OUTPUT); pinMode(h1.pinB, OUTPUT); pinMode(h1.pinEN, OUTPUT); 
  pinMode(h2.pinA, OUTPUT); pinMode(h2.pinB, OUTPUT); pinMode(h2.pinEN, OUTPUT); 
  
  setPower(h1, 0.0f); 
  setPower(h2, 0.0f); 

  Serial.println("\n# Initializing MPC...");
  Serial.println("# Capturing thermal baseline (DO NOT TOUCH)...");
  
  delay(3000); 

  // Grab a smooth average to set our 'zero' baseline for the MPC math
  for(int i = 0; i < 10; i++) {
    readTemp(t1); readTemp(t2);
    ambientT[0] += t1.temp;
    ambientT[1] += t2.temp;
    delay(100);
  }
  ambientT[0] /= 10.0f;
  ambientT[1] /= 10.0f;

  Serial.printf("# Baseline Locked -> T1: %.2f C | T2: %.2f C\n", ambientT[0], ambientT[1]);
  Serial.println("# Commands: '1 [temp]' for Z1, '2 [temp]' for Z2, 'x' to stop.");
  Serial.println("time_ms,sp1_c,t1_c,pwm1_cmd,sp2_c,t2_c,pwm2_cmd"); 
}

void loop() {
  // Command Parsing
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); 
    input.trim(); 
    
    if (input.equalsIgnoreCase("x")) {
      controller.setpoint[0] = 0.0f; 
      controller.setpoint[1] = 0.0f; 
    }
    else if (input.startsWith("1 ")) {
      controller.setpoint[0] = input.substring(2).toFloat(); 
    }
    else if (input.startsWith("2 ")) {
      controller.setpoint[1] = input.substring(2).toFloat(); 
    }
    
    // Immediately execute so we don't wait for the 2-second timer
    executeMIMO();
  }

  // Synchronous MPC Execution
  unsigned long now = millis(); 
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) { 
    lastSampleTime = now; 
    
    readTemp(t1); 
    readTemp(t2); 
    
    executeMIMO(); 
    
    // Formats CSV exactly as requested
    Serial.printf("%lu,%.1f,%.2f,%d,%.1f,%.2f,%d\n", 
                  now, controller.setpoint[0], t1.temp, h1.currentPWM, 
                  controller.setpoint[1], t2.temp, h2.currentPWM);
  }
}