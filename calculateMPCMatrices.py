import numpy as np
import pandas as pd
import scipy.linalg

# =============================================================
#  1. Data Ingestion & Formatting
# =============================================================
PWM_MAX = 225.0  # The hardware saturation knee we discovered

def process_run(filename):
    """Loads CSV data and formats it into state (x) and input (u) matrices."""
    # Load data without headers, assign column names
    df = pd.read_csv(filename, header=None, names=['time', 't1', 't2', 'pwm1', 'pwm2'])
    
    # Normalize control effort (0.0 to 1.0)
    u = df[['pwm1', 'pwm2']].values / PWM_MAX
    
    # Convert absolute temperatures to relative Delta T to keep equilibrium at 0,0
    T_ambient = df[['t1', 't2']].iloc[0].values
    x = df[['t1', 't2']].values - T_ambient
    
    return x, u

# Load both datasets
x1, u1 = process_run('heater1.csv')
x2, u2 = process_run('heater2.csv')

# =============================================================
#  2. Least Squares System Identification
# =============================================================
# To solve X_next = A*X_curr + B*U_curr, we stack the matrices:
# Y = H * Theta, where H = [X_curr, U_curr] and Theta = [A^T, B^T]^T

def build_ls_matrices(x, u):
    X_curr = x[:-1, :]  # All states except the very last one
    X_next = x[1:, :]   # All states except the very first one
    U_curr = u[:-1, :]  # Inputs applied at X_curr
    return X_curr, X_next, U_curr

X1_c, X1_n, U1_c = build_ls_matrices(x1, u1)
X2_c, X2_n, U2_c = build_ls_matrices(x2, u2)

# Vertically stack both runs to give the optimizer the full MIMO picture
X_curr = np.vstack((X1_c, X2_c))
X_next = np.vstack((X1_n, X2_n))
U_curr = np.vstack((U1_c, U2_c))

# Create the H matrix [X_curr | U_curr]
H = np.hstack((X_curr, U_curr))

# Solve for Theta using ordinary least squares
Theta, residuals, rank, s = np.linalg.lstsq(H, X_next, rcond=None)

# Extract A and B matrices (transposing because lstsq solves for Theta^T)
A = Theta[0:2, :].T
B = Theta[2:4, :].T

# =============================================================
#  3. SMOC MPC Gain Calculation
# =============================================================
# Uses the exact math from your smoc_hotplate.py script

N = 20        # Prediction Horizon
Q_val = 50.0  # Aggression: Penalty for deviation from target temperature
R_val = 0.5   # Smoothness: Penalty for changing heater power too violently

nx = 2  # Number of states
nu = 2  # Number of inputs
ny = 2  # Number of outputs
C = np.eye(2) # Identity matrix: We measure both states directly

# Build Prediction Matrices (Psi and Gamma)
Psi = np.zeros((N * ny, N * nu))
Gamma = np.zeros((N * ny, nx))

Ak = np.copy(A)
for i in range(N):
    Gamma[i*ny:(i+1)*ny, :] = C @ Ak
    Ak = Ak @ A

for i in range(N):
    Ak = np.eye(nx)
    for j in range(i + 1):
        Psi[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C @ Ak @ B
        Ak = A @ Ak

# Build Cost Matrices
Q_bar = Q_val * np.eye(N * ny)

R_block = R_val * np.eye(nu)
R_bar = np.zeros((N * nu, N * nu))
for i in range(N):
    R_bar[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = 2 * R_block
    if i > 0:
        R_bar[(i-1)*nu:i*nu, i*nu:(i+1)*nu] = -R_block
        R_bar[i*nu:(i+1)*nu, (i-1)*nu:i*nu] = -R_block
R_bar[(N-1)*nu:, (N-1)*nu:] = R_block

R_hat = np.zeros((N * nu, nu))
R_hat[:nu, :nu] = R_block

# Solve for Control Law Matrices
FU = Psi.T @ Q_bar @ Psi + R_bar
FU_inv = np.linalg.inv(FU)

KX_full = -FU_inv @ Psi.T @ Q_bar @ Gamma
KU1_full = FU_inv @ R_hat
KYD_full = FU_inv @ Psi.T @ Q_bar

# Extract Receding Horizon subset
KX = KX_full[:nu, :]
KU1 = KU1_full[:nu, :]
KYD = KYD_full[:nu, :]

# =============================================================
#  4. Kalman Filter Observer (DLQR based)
# =============================================================
# Solves the Discrete Algebraic Riccati Equation to find the optimal L gain
Q_noise = 0.1 * np.eye(nx)
R_noise = 1.0 * np.eye(ny)
P = scipy.linalg.solve_discrete_are(A.T, C.T, Q_noise, R_noise)
L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_noise)

# =============================================================
#  5. C++ Struct Output Generation
# =============================================================
def print_matrix(mat):
    """Formats numpy arrays into C++ nested brace syntax."""
    rows = []
    for row in mat:
        rows.append("{" + ", ".join([f"{val:7.4f}f" for val in row]) + "}")
    return "{" + ",\n   ".join(rows) + "}"

print("/* --- COPY THIS BLOCK INTO YOUR ESP32 CODE --- */\n")
print("MIMO_SMOC controller = {")
print("  // KX (State Penalty)")
print("  " + print_matrix(KX) + ",")
print("  // KU1 (Input Smoothness Penalty)")
print("  " + print_matrix(KU1) + ",")
print("  // KYD (Trajectory Tracking)")
print("  " + print_matrix(KYD) + ",")
print("  // Kalman L (Observer Gain)")
print("  " + print_matrix(L) + ",")
print("  // Model A (System Dynamics)")
print("  " + print_matrix(A) + ",")
print("  // Model B (Input Actuation)")
print("  " + print_matrix(B) + ",")
print("  // Model C (Output Mapping)")
print("  " + print_matrix(C) + ",")
print("  {20.0f, 20.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}")
print("};")
print("\n/* -------------------------------------------- */")