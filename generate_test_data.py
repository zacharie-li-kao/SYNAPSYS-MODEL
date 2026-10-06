"""
Generate realistic bogus experimental data for testing data_converter.py pipeline
Outputs CSV files + metadata text file

Modified so that:
 - 365 nm => potentiation (UV potentiation)
 - wavelengths >= 450 nm => depression, with depression peak at 550 nm
 - inversion of previous behavior (visible now causes depression)
 
USAGE:
    python generate_test_data.py [output_directory] [--seed N]

    If no directory specified, opens GUI folder picker (if available).
    Otherwise creates 'test_data_output' in current directory.

    --seed N makes the pseudo-random noise and dataset-to-dataset variability
    reproducible. Omitted, the generator is unseeded as before, so interactive
    use still produces a fresh realisation each run. The verification harness
    always passes a seed: without one, a regression in a fitted parameter is
    indistinguishable from ordinary scatter.
"""

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys

print("="*70)
print("TEST DATA GENERATOR - STDP/SRDP Enhanced")
print("="*70)
print("\nThis script generates synthetic characterization data including:")
print("  • Wavelength sweep files (potentiation and depression)")
print("  • Nonlinearity measurements")
print("  • Retention/decay data")
print("  • Cycling experiments")
print("  • STDP timing window data (NEW)")
print("  • SRDP frequency response data (NEW)")
print("="*70 + "\n")

# Ground truth synapse parameters (modified/inverted behavior)
true_params = {
    'G_min': 1e-6,
    'G_max': 1e-4,
    'alpha': 0.75,
    'beta': 0.75,
    # Potentiation (UV) peak
    'lambda_pot_peak': 365,
    # Depression (visible) peak
    'lambda_dep_peak': 550,
    # width (shared for simplicity; tune if needed)
    'lambda_width': 30,
    'A_peak': 8e-4,   # potentiation amplitude scale
    'B_peak': 6e-4,   # depression amplitude scale
    'decay_tau': 100
}

noise = 0.015  # 1.5% multiplicative noise on each recorded conductance

# --- Declared experimental conditions -------------------------------------
#
# Every generation site below reads these, and the METADATA_INSTRUCTIONS.txt
# written at the end is interpolated from the SAME names. That equality is the
# whole point (audit M22): the generator previously drove the wavelength sweep
# at 20 mW/cm², the single-wavelength potentiation at 50 and the retention
# pre-potentiation at 30, while declaring 20 for all of them. A pipeline
# validated against a misdeclared ground truth cannot detect an error in
# itself, which is how every defect in the audit went unnoticed.
INTENSITY_SWEEP_MW_CM2 = 20.0        # wavelength sweep, LTP nonlinearity, cycling
INTENSITY_STRONG_MW_CM2 = 50.0       # single-wavelength potentiation at 365 nm
INTENSITY_RETENTION_MW_CM2 = 30.0    # pre-potentiation before the decay measurement

PULSE_WIDTH_MS = 100.0               # every pulsed experiment
PULSE_DT_S = PULSE_WIDTH_MS / 1000.0 # integration step, in seconds

ELECTRICAL_VOLTAGE_V = 2.0

# Per CLAUDE.md the STDP rule already carries its own 0.01 factor:
#     ΔG = A_plus · exp(−dt/τ) · (G_max − G_min) · 0.01
# so A_plus = 0.5 means 0.5 PERCENT of the dynamic range. The generator used
# 0.15 here (audit M23) — fifteen times the specified rule — while fitting.py
# used yet a third value (0.1). The round trip generator → fitter → network
# could not be the identity with three different constants for one quantity.
STDP_RULE_SCALE = 0.01

# Repetitions averaged per Δt point in the STDP sweep.
#
# This is not a convenience: at the specified rule scale a single pairing
# changes G by 0.5% of the dynamic range — 0.495 µS against a ~50.6 µS
# operating point — while the read noise is 1.5% of that operating point,
# 0.759 µS. The signal sits BELOW the noise (SNR 0.65), and no estimator can
# recover A_plus from one pairing per point. Real STDP protocols resolve this
# the same way, by repeating each pairing and averaging; the noise then falls
# as 1/sqrt(N). At N = 250 the effective read noise is 0.048 µS, giving
# SNR ≈ 10.
#
# The generator previously hid this by using a rule scale of 0.15 — fifteen
# times the spec — which made a single pairing large enough to see. That
# inflated every fitted STDP amplitude by the same factor.
STDP_REPEATS_PER_POINT = 250

def relaxation(G):
    """Spontaneous relaxation toward G_min over one pulse period.

    ALWAYS applied, including during stimulation — see CLAUDE.md. Relaxation is
    thermodynamic and does not pause because a stimulus is present, so a
    generator that omitted it during drive was producing traces the simulator
    would never reproduce. At the default parameters this is ~6% of the drive
    term early in a potentiation train and grows as G rises, so it is not
    negligible.
    """
    return -(G - true_params['G_min']) / true_params['decay_tau'] * PULSE_DT_S


def wavelength_sensitivity_pot(wavelength_nm):
    # Gaussian sensitivity for potentiation centered at lambda_pot_peak
    return np.exp(-((wavelength_nm - true_params['lambda_pot_peak']) / true_params['lambda_width'])**2)

def wavelength_sensitivity_dep(wavelength_nm):
    # Gaussian sensitivity for depression centered at lambda_dep_peak
    return np.exp(-((wavelength_nm - true_params['lambda_dep_peak']) / true_params['lambda_width'])**2)

# --- Seed handling ---
# Parsed out of argv before the directory logic so that `--seed N` in any
# position does not get mistaken for an output directory.
_argv = sys.argv[1:]
_seed = None
if "--seed" in _argv:
    _i = _argv.index("--seed")
    try:
        _seed = int(_argv[_i + 1])
    except (IndexError, ValueError):
        raise SystemExit("--seed requires an integer argument, e.g. --seed 20260731")
    del _argv[_i:_i + 2]

if _seed is not None:
    np.random.seed(_seed)
    print(f"Random seed: {_seed} (reproducible run)")

# --- Output directory setup ---
if _argv:
    # Command line argument provided
    output_dir = _argv[0]
    print(f"Using output directory from command line: {output_dir}")
else:
    # Try to use GUI folder picker
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        print("Please select the output directory to save the generated data...")
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        
        if not output_dir:
            print("No directory selected. Using default location.")
            output_dir = os.path.join(os.getcwd(), "test_data_output")
        else:
            print(f"Selected output directory: {output_dir}")
            
    except ImportError:
        # tkinter not available, use default
        print("GUI not available. Using default output directory.")
        output_dir = os.path.join(os.getcwd(), "test_data_output")
    except Exception as e:
        # Any other error with GUI
        print(f"GUI selection failed ({e}). Using default output directory.")
        output_dir = os.path.join(os.getcwd(), "test_data_output")

os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")
print("="*70 + "\n")

# ============================================================================
# EXPANDED WAVELENGTH SWEEP (UV → NIR)
# ============================================================================
wavelength_sweep = [340, 360, 380, 400, 420, 440, 460, 480,
                    500, 520, 540, 560, 580, 600, 650, 700, 750]
intensity = INTENSITY_SWEEP_MW_CM2

for wavelength in wavelength_sweep:
    # add small dataset-to-dataset variability (±5%)
    variability = 1.0 + np.random.normal(0, 0.05)
    
    if wavelength < 450:
        # POTENTIATION (UV region)
        G = true_params['G_min'] * 1.5
        conductance = [G]
        A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(wavelength) * intensity * variability

        for pulse in range(50):
            dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
            G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])
            G_noisy = G * (1 + np.random.normal(0, noise))
            conductance.append(G_noisy)

        df = pd.DataFrame({
            'pulse_number': range(len(conductance)),
            'conductance_S': conductance
        })
        df.to_csv(os.path.join(output_dir, f"potentiation_{wavelength}nm.csv"), index=False)

    else:
        # DEPRESSION (visible region)
        G = true_params['G_max'] * 0.9
        conductance = [G]
        # Documented ODE exactly: B_eff = B_peak · |S(λ)| · I. The undeclared
        # ×10 that used to sit here (audit M24) made every fitted B_peak ten
        # times the true value, and nothing recorded that the factor existed.
        B_eff = true_params['B_peak'] * wavelength_sensitivity_dep(wavelength) * intensity * variability

        for pulse in range(50):
            dG = -B_eff * (G - true_params['G_min'])**true_params['beta'] * PULSE_DT_S + relaxation(G)
            G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])
            G_noisy = G * (1 + np.random.normal(0, noise))
            conductance.append(G_noisy)

        df = pd.DataFrame({
            'pulse_number': range(len(conductance)),
            'conductance_S': conductance
        })
        df.to_csv(os.path.join(output_dir, f"depression_{wavelength}nm.csv"), index=False)


# ============================================================================
# 9. LONG POTENTIATION (for nonlinearity) -- use 365 nm (UV potentiation)
# ============================================================================
wavelength = 365
G = true_params['G_min'] * 1.2
conductance = [G]

A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(wavelength) * intensity

for pulse in range(100):
    dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
    G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])
    G_noisy = G * (1 + np.random.normal(0, noise))
    conductance.append(G_noisy)

df = pd.DataFrame({
    'pulse_number': range(len(conductance)),
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "ltp_nonlinearity_365nm.csv"), index=False)

# ============================================================================
# 10. SINGLE-WAVELENGTH POTENTIATION at 365 nm (replaces previous UV depression)
# ============================================================================
wavelength = 365
G = true_params['G_min'] * 1.5
conductance = [G]

A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(wavelength) * INTENSITY_STRONG_MW_CM2

for pulse in range(80):
    dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
    G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])
    G_noisy = G * (1 + np.random.normal(0, noise))
    conductance.append(G_noisy)

df = pd.DataFrame({
    'pulse_number': range(len(conductance)),
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "potentiation_365nm.csv"), index=False)

# ============================================================================
# 11. POTENTIATION-DEPRESSION CYCLE
#    Now: potentiation at 365 nm, depression at 550 nm
# ============================================================================
pulse_num = []
conductance = []
pulse_count = 0

# Start at mid-range
G = (true_params['G_min'] + true_params['G_max']) / 2

for cycle in range(5):
    # Potentiation phase (UV at 365 nm)
    A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(365) * intensity
    for _ in range(30):
        conductance.append(G * (1 + np.random.normal(0, noise)))
        pulse_num.append(pulse_count)
        pulse_count += 1
        dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
        G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])

    # Depression phase (visible at 550 nm)
    B_eff = true_params['B_peak'] * wavelength_sensitivity_dep(550) * intensity
    for _ in range(30):
        conductance.append(G * (1 + np.random.normal(0, noise)))
        pulse_num.append(pulse_count)
        pulse_count += 1
        dG = -B_eff * (G - true_params['G_min'])**true_params['beta'] * PULSE_DT_S + relaxation(G)
        G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])

df = pd.DataFrame({
    'pulse_number': pulse_num,
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "cycling_5cycles_uv_pot_vis_dep.csv"), index=False)

# ============================================================================
# 12. RETENTION
#    First potentiate with UV (365 nm), then measure decay
# ============================================================================
# First potentiate
G = true_params['G_min'] * 1.5
A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(365) * INTENSITY_RETENTION_MW_CM2

for _ in range(50):
    dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
    G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])

G_initial = G

# Measure decay
times = [1, 10, 30, 60, 100, 200, 300, 500, 800, 1000]
conductance = []

for t in times:
    G_t = true_params['G_min'] + (G_initial - true_params['G_min']) * np.exp(-t / true_params['decay_tau'])
    G_noisy = G_t * (1 + np.random.normal(0, noise))
    conductance.append(G_noisy)

df = pd.DataFrame({
    'time_s': times,
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "retention_decay_after_uv_pot.csv"), index=False)

# ============================================================================
# 13. ELECTRICAL DEPRESSION (voltage pulses) - unchanged (wavelength-independent)
# ============================================================================
voltage = ELECTRICAL_VOLTAGE_V
G = true_params['G_max'] * 0.9
conductance = [G]

# Electrical depression: wavelength-independent. CLAUDE.md specifies
# B_eff = B_peak · V · 10; the extra ×5 that used to be here (audit M24) was
# undeclared and made electrical and light depression mutually inconsistent.
B_eff = true_params['B_peak'] * voltage * 10

for pulse in range(80):
    dG = -B_eff * (G - true_params['G_min'])**true_params['beta'] * PULSE_DT_S + relaxation(G)
    G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])
    G_noisy = G * (1 + np.random.normal(0, noise))
    conductance.append(G_noisy)

df = pd.DataFrame({
    'pulse_number': range(len(conductance)),
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "depression_electrical_2V.csv"), index=False)

# ============================================================================
# 14. MIXED: LIGHT POTENTIATION (UV 365) + ELECTRICAL DEPRESSION CYCLES
# ============================================================================
pulse_num = []
conductance = []
pulse_count = 0

# Start at mid-range
G = (true_params['G_min'] + true_params['G_max']) / 2

for cycle in range(5):
    # Potentiation phase (UV at 365 nm)
    A_eff = true_params['A_peak'] * wavelength_sensitivity_pot(365) * intensity
    for _ in range(30):
        conductance.append(G * (1 + np.random.normal(0, noise)))
        pulse_num.append(pulse_count)
        pulse_count += 1
        dG = A_eff * (true_params['G_max'] - G)**true_params['alpha'] * PULSE_DT_S + relaxation(G)
        G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])

    # Depression phase (ELECTRICAL at 2V)
    B_eff = true_params['B_peak'] * ELECTRICAL_VOLTAGE_V * 10  # Electrical depression, per spec
    for _ in range(30):
        conductance.append(G * (1 + np.random.normal(0, noise)))
        pulse_num.append(pulse_count)
        pulse_count += 1
        dG = -B_eff * (G - true_params['G_min'])**true_params['beta'] * PULSE_DT_S + relaxation(G)
        G = np.clip(G + dG, true_params['G_min'], true_params['G_max'])

df = pd.DataFrame({
    'pulse_number': pulse_num,
    'conductance_S': conductance
})
df.to_csv(os.path.join(output_dir, "cycling_mixed_uv_light_electrical_dep.csv"), index=False)

# ============================================================================
# 15. STDP (Spike-Timing-Dependent Plasticity) DATA
# ============================================================================
# STDP window parameters
A_plus = 0.5
A_minus = 0.3
tau_plus = 20.0  # ms
tau_minus = 20.0  # ms

delta_t_list = np.linspace(-100, 100, 20)
G_initial = (true_params['G_min'] + true_params['G_max']) / 2  # Start at middle conductance
conductance_stdp = []
delta_g_stdp = []  # Store actual changes for verification

for delta_t_ms in delta_t_list:
    # Calculate STDP weight change
    if delta_t_ms > 0:
        # Pre before Post → LTP (potentiation)
        weight_change = A_plus * np.exp(-delta_t_ms / tau_plus)
    elif delta_t_ms < 0:
        # Post before Pre → LTD (depression)
        weight_change = -A_minus * np.exp(delta_t_ms / tau_minus)
    else:
        weight_change = A_plus - A_minus

    # The spec rule, exactly: ΔG = weight_change · (G_max − G_min) · 0.01.
    # A_plus = 0.5 therefore produces 0.5% of the dynamic range at dt → 0.
    delta_g = weight_change * (true_params['G_max'] - true_params['G_min']) * STDP_RULE_SCALE
    G_final = G_initial + delta_g

    # Averaged over STDP_REPEATS_PER_POINT independent pairings, as a real
    # protocol does — the standard error of the mean falls as 1/sqrt(N).
    stdp_noise = noise / np.sqrt(STDP_REPEATS_PER_POINT)
    G_final_noisy = G_final * (1 + np.random.normal(0, stdp_noise))
    G_final_noisy = np.clip(G_final_noisy, true_params['G_min'], true_params['G_max'])

    conductance_stdp.append(G_final_noisy)
    delta_g_stdp.append(delta_g)

# Verify STDP shape
print(f"\nSTDP data verification:")
print(f"  G_initial: {G_initial*1e6:.2f} µS")
print(f"  Delta_G range: {min(delta_g_stdp)*1e6:.2f} to {max(delta_g_stdp)*1e6:.2f} µS")
print(f"  Final G range: {min(conductance_stdp)*1e6:.2f} to {max(conductance_stdp)*1e6:.2f} µS")
# Selected by the SIGN of Δt, which is what actually defines the two branches.
# This used to be `delta_g_stdp.index(dg)` inside a comprehension — O(n²), and
# index() returns the FIRST occurrence, so the split was by position in the
# list rather than by Δt and duplicated values were misattributed.
_ltp = [dg * 1e6 for dt, dg in zip(delta_t_list, delta_g_stdp) if dt > 0]
_ltd = [dg * 1e6 for dt, dg in zip(delta_t_list, delta_g_stdp) if dt < 0]
print(f"  LTP side (Δt>0), first 3: {[f'{v:+.3f}' for v in _ltp[:3]]} µS")
print(f"  LTD side (Δt<0), last 3:  {[f'{v:+.3f}' for v in _ltd[-3:]]} µS")
assert all(v > 0 for v in _ltp), "LTP branch must be strictly positive"
assert all(v < 0 for v in _ltd), "LTD branch must be strictly negative"

df = pd.DataFrame({
    'delta_t_ms': delta_t_list,
    'conductance_S': conductance_stdp
})
df.to_csv(os.path.join(output_dir, "stdp_timing_window.csv"), index=False)

# ============================================================================
# 16. SRDP (Spike-Rate-Dependent Plasticity) DATA
# ============================================================================
frequencies = np.logspace(-1, 2, 15)  # 0.1 Hz to 100 Hz
G_initial = (true_params['G_min'] + true_params['G_max']) / 2
conductance_srdp = []

# Transition frequency and slope of the rate response.
#
# The sigmoid is in LOG10 FREQUENCY (audit M7). Spike-rate-dependent
# plasticity is log-frequency-dependent — that is why SRDP sweeps are
# log-spaced in the first place — and the generator previously emitted a
# sigmoid in LINEAR frequency, `1/(1+exp(-(f-10)/10))`. Fitted in log
# frequency, as the physics requires, a linear-frequency sigmoid is simply a
# different function, so f0 could not be recovered no matter how good the fit
# was. Generator and fitter now describe the same physics.
SRDP_F0_HZ = 10.0
SRDP_SLOPE_DECADES = 0.3
SRDP_MAX_CHANGE_FRACTION = 0.2

for freq_hz in frequencies:
    freq_factor = 1.0 / (1.0 + np.exp(
        -(np.log10(freq_hz) - np.log10(SRDP_F0_HZ)) / SRDP_SLOPE_DECADES))

    # Total conductance change
    base_change = SRDP_MAX_CHANGE_FRACTION * freq_factor
    noise_factor = 1 + np.random.uniform(-0.05, 0.05)
    delta_g = base_change * (true_params['G_max'] - true_params['G_min']) * noise_factor


    G_final = G_initial + delta_g
    
    # Add noise
    G_final_noisy = G_final * (1 + np.random.normal(0, noise))
    G_final_noisy = np.clip(G_final_noisy, true_params['G_min'], true_params['G_max'])
    
    conductance_srdp.append(G_final_noisy)

df = pd.DataFrame({
    'frequency_hz': frequencies,
    'conductance_S': conductance_srdp
})
df.to_csv(os.path.join(output_dir, "srdp_frequency_response.csv"), index=False)

# ============================================================================
# METADATA FILE (updated to reflect inverted behavior + STDP/SRDP)
# ============================================================================
metadata_text = f"""METADATA FOR DATA_CONVERTER.PY
================================

WAVELENGTH SWEEP FILES:
-----------------------
Files with wavelengths < 450 nm are generated as POTENTIATION (UV-driven potentiation).
Files with wavelengths >= 450 nm are generated as DEPRESSION (visible-driven depression).
Depression peak: {true_params['lambda_dep_peak']} nm
Potentiation peak: {true_params['lambda_pot_peak']} nm

"""

for wl in wavelength_sweep:
    if wl < 450:
        metadata_text += f"""FILE: potentiation_{wl}nm.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: potentiation
Stimulus type: light
Wavelength (nm) - potentiation: {wl}
Wavelength (nm) - depression: (leave empty)
Light intensity (mW/cm²): {INTENSITY_SWEEP_MW_CM2:g}
Pulse width (ms): {PULSE_WIDTH_MS:g}
Pulse frequency (Hz): 10
Number of pulses: 50
Read pulse timing relative to write (ms): 10
Notes: Potentiation at {wl}nm (UV side)

"""
    else:
        metadata_text += f"""FILE: depression_{wl}nm.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: depression
Stimulus type: light
Wavelength (nm) - potentiation: (leave empty)
Wavelength (nm) - depression: {wl}
Light intensity (mW/cm²): {INTENSITY_SWEEP_MW_CM2:g}
Pulse width (ms): {PULSE_WIDTH_MS:g}
Pulse frequency (Hz): 10
Number of pulses: 50
Read pulse timing relative to write (ms): 10
Notes: Depression at {wl}nm (visible side; peak depression around {true_params['lambda_dep_peak']} nm)

"""

metadata_text += f"""FILE: ltp_nonlinearity_365nm.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: potentiation
Stimulus type: light
Wavelength (nm) - potentiation: 365
Wavelength (nm) - depression: (leave empty)
Light intensity (mW/cm²): {INTENSITY_SWEEP_MW_CM2:g}
Pulse width (ms): {PULSE_WIDTH_MS:g}
Pulse frequency (Hz): 10
Number of pulses: 100
Read pulse timing relative to write (ms): 10
Notes: Long-term potentiation at 365 nm (UV potentiation)

FILE: potentiation_365nm.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: potentiation
Stimulus type: light
Wavelength (nm) - potentiation: 365
Wavelength (nm) - depression: (leave empty)
Light intensity (mW/cm²): {INTENSITY_STRONG_MW_CM2:g}
Pulse width (ms): {PULSE_WIDTH_MS:g}
Pulse frequency (Hz): 10
Number of pulses: 80
Read pulse timing relative to write (ms): 10
Notes: Single-wavelength potentiation at 365 nm (UV)

FILE: cycling_5cycles_uv_pot_vis_dep.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: potentiation_depression_cycle
Stimulus type: light
Wavelength (nm) - potentiation: 365
Wavelength (nm) - depression: 550
Light intensity (mW/cm²): {INTENSITY_SWEEP_MW_CM2:g}
Pulse width (ms): {PULSE_WIDTH_MS:g}
Pulse frequency (Hz): 10
Number of pulses per train: 30
Read pulse timing relative to write (ms): 10
Number of cycles: 5
Notes: 5 cycles: UV potentiation (365 nm) and visible depression (550 nm)

FILE: retention_decay_after_uv_pot.csv
----------------------------
X-axis represents: time
X-axis units: s
Y-axis represents: conductance
Y-axis units: S
Experiment type: retention
Stimulus type: light
Wavelength (nm) - potentiation: 365
Wavelength (nm) - depression: (leave empty)
Light intensity (mW/cm²): {INTENSITY_RETENTION_MW_CM2:g}
Notes: Retention/decay measurement after UV potentiation

FILE: depression_electrical_2V.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: depression
Stimulus type: electrical
Voltage (V): 2.0
Number of pulses: 80
Notes: Electrical depression using voltage pulses (no light)

FILE: cycling_mixed_uv_light_electrical_dep.csv
----------------------------
X-axis represents: pulse_number
X-axis units: #
Y-axis represents: conductance
Y-axis units: S
Experiment type: potentiation_depression_cycle
Stimulus type: light+electrical
Wavelength (nm) - potentiation: 365
Wavelength (nm) - depression: (electrical)
Notes: Mixed stimulation - UV light potentiation and electrical depression

FILE: stdp_timing_window.csv
----------------------------
X-axis represents: delta_t_ms
X-axis units: ms
Y-axis represents: conductance
Y-axis units: S
Experiment type: stdp
Notes: Spike-Timing-Dependent Plasticity window. Delta_t is post-pre spike timing difference. Positive delta_t = LTP, negative = LTD.

FILE: srdp_frequency_response.csv
----------------------------
X-axis represents: frequency_hz
X-axis units: Hz
Y-axis represents: conductance
Y-axis units: S
Experiment type: srdp
Notes: Spike-Rate-Dependent Plasticity. Conductance change as function of spike frequency.

========================================
GROUND TRUTH PARAMETERS (for validation)
========================================
G_min: {true_params['G_min']} S
G_max: {true_params['G_max']} S
alpha: {true_params['alpha']}
beta: {true_params['beta']}
lambda_pot_peak: {true_params['lambda_pot_peak']} nm
lambda_dep_peak: {true_params['lambda_dep_peak']} nm
lambda_width: {true_params['lambda_width']} nm
decay_tau: {true_params['decay_tau']} s
A_peak: {true_params['A_peak']}
B_peak: {true_params['B_peak']}

STDP Parameters:
A_plus: 0.5
A_minus: 0.3
tau_plus: 20.0 ms
tau_minus: 20.0 ms

SRDP Parameters:
Transition frequency: 10 Hz
Max change: 20%

These are the TRUE parameters used to generate the data.
Compare fitted parameters against these to validate the pipeline.
"""

with open(os.path.join(output_dir, "METADATA_INSTRUCTIONS.txt"), "w") as f:
    f.write(metadata_text)

print(f"✓ Generated CSV files in '{output_dir}' directory (inverted behavior)")
print("\nFiles include:")
print("  • wavelength sweep files (400-750nm) with pot for <450nm and dep for >=450nm")
print("  • long potentiation at 365 nm")
print("  • single-wavelength potentiation at 365 nm")
print("  • depression_electrical_2V.csv (unchanged)")
print("  • cycling files with UV potentiation and visible depression")
print("  • stdp_timing_window.csv (STDP characterization)")
print("  • srdp_frequency_response.csv (SRDP characterization)")
print("\nNext steps:")
print("1. Run this script to regenerate the dataset.")
print("2. Load files in data_converter.py and check metadata fields.")
print("3. Use STDP/SRDP data for spiking neural network simulations.")
print(f"\nGround truth (high level): pot peak={true_params['lambda_pot_peak']}nm, dep peak={true_params['lambda_dep_peak']}nm")
print(f"STDP: A+={0.5}, A-={0.3}, tau+={20}ms, tau-={20}ms")

print("\n" + "="*70)
print("DATA GENERATION COMPLETE!")
print("="*70)

# Hold the window open only when a human is watching. When invoked from a
# script, a pipeline or a test harness there is nobody to press Enter, and an
# unguarded input() raises EOFError — a non-zero exit that makes a fully
# successful generation look like a failure.
#
# isatty() alone is not enough: on Windows the NUL device reports True, so a
# headless run still reaches input() and immediately hits EOF. Catch it.
if sys.stdin is not None and sys.stdin.isatty():
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass  # No interactive console after all; exit normally.