"""
Dual-Zone Heater MIMO MPC — Data Plotter
Columns: time_ms, t1_c, t2_c, pwm1_cmd, pwm2_cmd
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
H1_FILE = "heater1.csv"
H2_FILE = "heater2.csv"

COLORS = {
    "h1_t1":  "#185FA5",   # blue  — H1 active zone
    "h1_t2":  "#85B7EB",   # light blue — H1 coupled zone
    "h2_t2":  "#993C1D",   # coral — H2 active zone
    "h2_t1":  "#F0997B",   # light coral — H2 coupled zone
    "pwm1":   "#185FA5",
    "pwm2":   "#993C1D",
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(",")
            if len(cols) < 5:
                continue
            try:
                data.append([float(c) for c in cols[:5]])
            except ValueError:
                continue   # skip header if present
    return np.array(data)   # [time_ms, t1, t2, pwm1, pwm2]

h1 = load(H1_FILE)
h2 = load(H2_FILE)

# ── Build unified time axis ───────────────────────────────────────────────────
all_ts = np.union1d(h1[:, 0], h2[:, 0])
t0     = all_ts[0]
xs     = (all_ts - t0) / 1000.0   # seconds

def align(data, all_ts, col):
    """Map a column from one experiment onto the global time axis (NaN for gaps)."""
    mapping = dict(zip(data[:, 0], data[:, col]))
    return np.array([mapping.get(t, np.nan) for t in all_ts])

H1_T1   = align(h1, all_ts, 1)
H1_T2   = align(h1, all_ts, 2)
H1_PWM1 = align(h1, all_ts, 3)
H2_T1   = align(h2, all_ts, 1)
H2_T2   = align(h2, all_ts, 2)
H2_PWM2 = align(h2, all_ts, 4)

# ── Metrics ───────────────────────────────────────────────────────────────────
def rise(arr):
    v = arr[~np.isnan(arr)]
    return np.max(v) - np.min(v)

metrics = {
    "Duration":         f"{xs[-1]:.0f} s",
    "H1 T1 rise":       f"+{rise(H1_T1):.1f} °C",
    "H2 T2 rise":       f"+{rise(H2_T2):.1f} °C",
    "H1→T2 coupling":   f"+{rise(H1_T2):.1f} °C",
    "H2→T1 coupling":   f"+{rise(H2_T1):.1f} °C",
    "H1 peak T1":       f"{np.nanmax(H1_T1):.1f} °C",
    "H2 peak T2":       f"{np.nanmax(H2_T2):.1f} °C",
}

print("── Metrics ──────────────────────────")
for k, v in metrics.items():
    print(f"  {k:<22} {v}")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax_temp = fig.add_subplot(gs[0])
ax_pwm  = fig.add_subplot(gs[1], sharex=ax_temp)

# — Temperature panel —
ax_temp.plot(xs, H1_T1, color=COLORS["h1_t1"], lw=2,   label="H1 · T1 (active zone)")
ax_temp.plot(xs, H1_T2, color=COLORS["h1_t2"], lw=1.5, ls="--", label="H1 · T2 (coupled)")
ax_temp.plot(xs, H2_T2, color=COLORS["h2_t2"], lw=2,   label="H2 · T2 (active zone)")
ax_temp.plot(xs, H2_T1, color=COLORS["h2_t1"], lw=1.5, ls="--", label="H2 · T1 (coupled)")

ax_temp.set_ylabel("Temperature (°C)", fontsize=11)
ax_temp.legend(loc="upper left", fontsize=9, framealpha=0.7)
ax_temp.grid(True, color="#e0e0e0", linewidth=0.6)
ax_temp.set_title("Dual-Zone Heater — Measured Temperature & Actuator Output", fontsize=12, pad=10)
plt.setp(ax_temp.get_xticklabels(), visible=False)

# — PWM panel —
ax_pwm.fill_between(xs, H1_PWM1, step="post", alpha=0.15, color=COLORS["pwm1"])
ax_pwm.step(xs, H1_PWM1, color=COLORS["pwm1"], lw=1.8, where="post", label="PWM1 (H1 exp)")
ax_pwm.fill_between(xs, H2_PWM2, step="post", alpha=0.15, color=COLORS["pwm2"])
ax_pwm.step(xs, H2_PWM2, color=COLORS["pwm2"], lw=1.8, where="post", label="PWM2 (H2 exp)")

ax_pwm.set_ylim(0, 290)
ax_pwm.set_yticks([0, 128, 225, 255])
ax_pwm.set_ylabel("PWM count", fontsize=11)
ax_pwm.set_xlabel("Time (s)", fontsize=11)
ax_pwm.legend(loc="upper left", fontsize=9, framealpha=0.7)
ax_pwm.grid(True, color="#e0e0e0", linewidth=0.6)

# — Metrics annotation box —
metric_text = "\n".join(f"{k}: {v}" for k, v in metrics.items())
ax_temp.text(
    0.99, 0.04, metric_text,
    transform=ax_temp.transAxes,
    fontsize=8, verticalalignment="bottom", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85)
)

plt.tight_layout()
plt.savefig("heater_dual_zone_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → heater_dual_zone_plot.png")
