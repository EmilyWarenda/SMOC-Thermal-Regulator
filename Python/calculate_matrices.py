import numpy as np
import pandas as pd
import scipy.linalg
import sys
from datetime import datetime

# =============================================================
#  Configuration
# =============================================================
PWM_MAX = 225.0   # Hardware saturation knee

# MPC tuning parameters
N     = 20    # Prediction horizon (steps)
Q_val = 50.0  # Aggression:  penalty for temperature error
R_val = 0.5   # Smoothness:  penalty for rapid heater changes

# Kalman filter noise covariances.
# The fan is an unmeasured disturbance — the Kalman observer detects it as
# a mismatch between predicted and actual temperature and corrects for it.
# Increasing R_noise_val makes the observer react more aggressively to that
# mismatch, which gives faster disturbance rejection when the fan kicks in,
# at the cost of amplifying sensor noise. Start here and tune if needed.
Q_noise_val = 0.1   # Process noise  (increase → trust model less)
R_noise_val = 1.0   # Measurement noise (increase → trust sensors less)

# All four PRBS run files. The fan-on and fan-off runs are stacked together
# so the identified model captures average dynamics across both conditions.
# This makes the controller robust to the fan switching as an unmeasured
# disturbance rather than requiring a separate model for each state.
ALL_FILES = [
    "Data/heater1_fanoff.csv",
    "Data/heater2_fanoff.csv",
    "Data/heater1_fanon.csv",
    "Data/heater2_fanon.csv",
]

# =============================================================
#  1. Data Ingestion & Formatting
# =============================================================
def load_csv(filename):
    """
    Loads a PRBS CSV, robustly stripping ESP32 boot messages and # comment
    lines that appear at the top of Serial Monitor captures.
    Returns a clean DataFrame with named columns.
    """
    rows = []
    header_found = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_found:
                if "time_ms" in line.lower():
                    header_found = True
                continue   # skip the header row itself
            rows.append(line)

    if not rows:
        raise ValueError(f"No data rows found in '{filename}'. "
                         f"Check the file contains PRBS output.")

    records = []
    for row in rows:
        parts = row.split(",")
        if len(parts) < 5:
            continue
        try:
            records.append({
                "time":  float(parts[0]),
                "t1":    float(parts[1]),
                "t2":    float(parts[2]),
                "pwm1":  float(parts[3]),
                "pwm2":  float(parts[4]),
            })
        except ValueError:
            continue

    if not records:
        raise ValueError(f"Could not parse any valid rows from '{filename}'.")

    df = pd.DataFrame(records)
    duration = (df["time"].iloc[-1] - df["time"].iloc[0]) / 60000.0
    print(f"  Loaded '{filename}': {len(df)} rows, {duration:.1f} min")
    return df


def process_run(filename):
    """
    Converts a raw PRBS CSV into state (x) and input (u) matrices.

    x: temperature as DeltaT from initial ambient — keeps equilibrium at (0, 0)
    u: PWM normalised to [0, 1]
    """
    df = load_csv(filename)
    u = df[["pwm1", "pwm2"]].values / PWM_MAX
    T_ambient = df[["t1", "t2"]].iloc[0].values
    x = df[["t1", "t2"]].values - T_ambient
    return x, u


def build_ls_matrices(x, u):
    """Splits a run into current/next state pairs for least squares."""
    X_curr = x[:-1, :]
    X_next = x[1:,  :]
    U_curr = u[:-1, :]
    return X_curr, X_next, U_curr


# =============================================================
#  2. Load and stack all four PRBS runs
# =============================================================
print("\nLoading PRBS runs...")

X_curr_all, X_next_all, U_curr_all = [], [], []
loaded_files = []

for fname in ALL_FILES:
    try:
        x, u = process_run(fname)
        Xc, Xn, Uc = build_ls_matrices(x, u)
        X_curr_all.append(Xc)
        X_next_all.append(Xn)
        U_curr_all.append(Uc)
        loaded_files.append(fname)
    except FileNotFoundError:
        print(f"  WARNING: '{fname}' not found — skipping.")
    except ValueError as e:
        print(f"  WARNING: {e} — skipping.")

if len(loaded_files) < 2:
    print("\nERROR: At least two PRBS run files are needed for identification.")
    sys.exit(1)

if len(loaded_files) < 4:
    print(f"\nNOTE: Only {len(loaded_files)}/4 files loaded. Identification will "
          f"still work but the model may be less robust to fan disturbances.")

X_curr = np.vstack(X_curr_all)
X_next = np.vstack(X_next_all)
U_curr = np.vstack(U_curr_all)

print(f"\nStacked {len(loaded_files)} runs -> {X_curr.shape[0]} total samples")

# =============================================================
#  3. Least Squares System Identification
# =============================================================
# Fit:   x[k+1] = A * x[k] + B * u[k]
#
# Rewrite as:  X_next = [X_curr | U_curr] * [A^T; B^T]  =  H * Theta
#
# lstsq finds the single A, B that minimises squared error across ALL runs
# simultaneously — fan-on and fan-off dynamics averaged into one model.

H = np.hstack((X_curr, U_curr))
Theta, residuals, rank, sv = np.linalg.lstsq(H, X_next, rcond=None)

A = Theta[0:2, :].T
B = Theta[2:4, :].T

# --- Sanity checks ---
""" eigs = np.abs(np.linalg.eigvals(A))
print(f"\nIdentified A eigenvalues (magnitudes): {np.round(eigs, 4)}")
if np.any(eigs >= 1.0):
    print("WARNING: A has eigenvalue(s) >= 1 — identified model is unstable!")
    print("         This usually means noisy data or too-short a run.")
    print("         Check your CSV files before using these matrices.")
else:
    print(f"  Model is stable  (all |lambda| < 1)  OK")

if rank < H.shape[1]:
    print(f"WARNING: H matrix is rank-deficient (rank={rank}, expected {H.shape[1]}).")
    print("         The system may not be sufficiently excited. Check PRBS data.") """

# --- Eigenvalue stabilisation ---
# If any eigenvalue magnitude >= 1, project it back inside the unit circle.
# A value of 1.0015 from ambient drift gets clipped to 0.999 — close enough
# that the MPC and Kalman filter are unaffected, but now provably stable.
STABILITY_MARGIN = 0.999  # maximum allowed eigenvalue magnitude

eigenvalues, eigenvectors = np.linalg.eig(A)
modified = False
for i, ev in enumerate(eigenvalues):
    if abs(ev) >= 1.0:
        eigenvalues[i] = ev / abs(ev) * STABILITY_MARGIN
        modified = True

if modified:
    A = np.real(eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors))
    print(f"  A stabilised — new eigenvalues: {np.round(np.abs(np.linalg.eigvals(A)), 4)}")

print(f"\nA =\n{np.round(A, 4)}")
print(f"\nB =\n{np.round(B, 4)}")

# =============================================================
#  4. SMOC Receding-Horizon MPC Gain Calculation
# =============================================================
nx = 2
nu = 2
ny = 2
C  = np.eye(2)   # We measure both states directly

# Build prediction matrices Psi (input -> output) and Gamma (state -> output)
Psi   = np.zeros((N * ny, N * nu))
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

# Build cost matrices
Q_bar   = Q_val * np.eye(N * ny)
R_block = R_val * np.eye(nu)

# R_bar encodes penalty on *changes* in input (du formulation from SMOC paper)
R_bar = np.zeros((N * nu, N * nu))
for i in range(N):
    R_bar[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = 2 * R_block
    if i > 0:
        R_bar[(i-1)*nu:i*nu,  i*nu:(i+1)*nu] = -R_block
        R_bar[i*nu:(i+1)*nu, (i-1)*nu:i*nu]  = -R_block
R_bar[(N-1)*nu:, (N-1)*nu:] = R_block

R_hat           = np.zeros((N * nu, nu))
R_hat[:nu, :nu] = R_block

# Solve for the full-horizon control gain matrices
FU     = Psi.T @ Q_bar @ Psi + R_bar
FU_inv = np.linalg.inv(FU)

KX_full  = -FU_inv @ Psi.T @ Q_bar @ Gamma
KU1_full =  FU_inv @ R_hat
KYD_full =  FU_inv @ Psi.T @ Q_bar

# Extract first nu rows only — receding horizon (only first step is applied)
KX  = KX_full[:nu, :]
KU1 = KU1_full[:nu, :]
KYD = KYD_full[:nu, :]

# =============================================================
#  5. Kalman Filter Observer (DARE-based)
# =============================================================
# Higher R_noise -> observer corrects more aggressively when the fan turns on
# and creates a mismatch between predicted and actual temperature.
Q_noise = Q_noise_val * np.eye(nx)
R_noise = R_noise_val * np.eye(ny)
P = scipy.linalg.solve_discrete_are(A.T, C.T, Q_noise, R_noise)
L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_noise)

print(f"\nKalman L =\n{np.round(L, 4)}")

# =============================================================
#  6. C++ Struct Output — written to mpc_matrices.h
# =============================================================
def fmt_matrix(mat):
    """Formats a numpy array as a C++ nested-brace float literal."""
    rows = []
    for row in mat:
        rows.append("{" + ", ".join([f"{v:8.4f}f" for v in row]) + "}")
    return "{\n   " + ",\n   ".join(rows) + "}"

OUT_FILE  = "Arduino/mpc_matrices.h"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
src_list  = ",  ".join(loaded_files)

lines = []
lines.append("/* ================================================================")
lines.append(f"   Auto-generated by calculateMPCMatrices.py")
lines.append(f"   {timestamp}")
lines.append(f"   Source files : {src_list}")
lines.append(f"   Runs stacked : {len(loaded_files)}/4  "
             f"(fan-on + fan-off averaged for disturbance robustness)")
lines.append(f"   Tuning       : Q={Q_val}  R={R_val}  N={N}  "
             f"Q_noise={Q_noise_val}  R_noise={R_noise_val}")
lines.append("")
lines.append("   Paste the struct below into controller_mosfet.ino,")
lines.append("   replacing the existing MIMO_SMOC controller = { ... }; block.")
lines.append("   ================================================================ */\n")

lines.append("MIMO_SMOC controller = {")
lines.append(f"  // KX -- state feedback gain  (Q={Q_val}, R={R_val}, N={N})")
lines.append("  " + fmt_matrix(KX) + ",")
lines.append("  // KU1 -- input smoothness gain")
lines.append("  " + fmt_matrix(KU1) + ",")
lines.append(f"  // KYD -- trajectory tracking gain  [{KYD.shape[0]} x {KYD.shape[1]}]")
lines.append("  " + fmt_matrix(KYD) + ",")
lines.append(f"  // Kalman L -- observer gain  (Q_noise={Q_noise_val}, R_noise={R_noise_val})")
lines.append("  " + fmt_matrix(L) + ",")
lines.append("  // Model A -- discrete state transition  (averaged across fan on/off)")
lines.append("  " + fmt_matrix(A) + ",")
lines.append("  // Model B -- input actuation")
lines.append("  " + fmt_matrix(B) + ",")
lines.append("  // Model C -- output mapping (identity: we measure both states)")
lines.append("  " + fmt_matrix(C) + ",")
lines.append("  {0.0f, 0.0f},  // x_hat  (observer state, starts at zero)")
lines.append("  {0.0f, 0.0f},  // u_prev (previous input, starts at zero)")
lines.append("  {0.0f, 0.0f}   // setpoint (degrees C, set via Serial)")
lines.append("};")
lines.append("\n/* ================================================================ */")

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"\nMatrices written to '{OUT_FILE}'  OK")