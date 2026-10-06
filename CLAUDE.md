# SYNAPSYS Suite — Project Context

Address the user as **THE PROTAGONIST**.

## Ground Rules
- **Physics must never be simplified.** Every equation, exponent, spectral response, and boundary condition must be physically rigorous.
- **No simplified fallbacks or hardcoded values/behaviors.** Silent degradation is silent failure. If something cannot be computed, it must fail loudly or be explicitly flagged — never silently substituted with a dummy value.
- **No placeholders.** Every feature is implemented fully. No stubs, no `# TODO`, no `pass` bodies, no "simplified for now" shortcuts.

## What SYNAPSYS Is

A full-stack neuromorphic device characterization and simulation platform. It spans from real Keithley 2600-series SMU hardware control to spiking neural network simulation, all grounded in the physics of memristive/optoelectronic synaptic devices.

**Author:** Zacharie Jehl Li-Kao (zacharie.jehl@upc.edu)

## Architecture Overview

```
hub.py ─── launches via subprocess.Popen ───┐
  ├── keithley_analyser.py                           (hardware control)
  ├── data_converter.py                              (CSV/metadata → JSON)
  ├── generate_test_data.py                          (synthetic data)
  └── Main.py                                        (simulation + fitting + network)
         ├── fitting.py          (model extraction)
         ├── network.py          (spatial arrays + SNN)
         ├── snn_visualization.py (spike trains + plots)
         ├── excel_patterns.py   (custom patterns from Excel)
         └── assembly_helpers.py (JSON suite builder)

synapse_engine.py ←── used by Keithley controller
synapse_cycle.py  ←── used by Keithley controller (cycling)
```

## Core Physics Model — Memristive Synapse ODE

Central to everything. Forward-Euler integrated at dt = 1 ms (stimulation) or 10 ms (rest).

**Potentiation:**
```
dG = A_eff * (G_max - G)^alpha * dt_s
A_eff = A_peak * |S(lambda)| * I     (light)
A_eff = A_peak * V * 10              (electrical)
```

**Depression:**
```
dG = -B_eff * (G - G_min)^beta * dt_s
B_eff = B_peak * |S(lambda)| * I     (light)
B_eff = B_peak * V * 10              (electrical)
```

**Spontaneous decay (ALWAYS applied, not rest-only):**
```
dG_decay = -(G - G_min) / decay_tau * dt_s
```
Relaxation is thermodynamic — trapped charge detraps, ions back-diffuse — and does not pause
while a stimulus is applied. Stimulation adds a driving term; it does not suspend relaxation.
This term was previously gated on `intensity == 0`, which made part of a spatial pattern's
apparent contrast an artefact of which pixels happened to be lit (audit M15). Gating it is a
simplification of the physics and the ground rules forbid it.

**Light branch selection.** When a **fitted signed** wavelength curve is loaded, the *wavelength*
determines the branch (positive lobe → potentiation, negative → depression) and the requested
`mode` cannot override it — that is the physics. When there is **no fitted curve**, the fallback is
a single positive Gaussian which carries responsivity *magnitude* and no sign at all, so it cannot
select a branch: `mode` is honoured and `|S(λ)|` supplies the magnitude. Reading the fallback's
sign as "always potentiation" made depression unreachable in the default state. Either way,
`VisualSynapse.last_stimulus_report` records the branch actually taken and why, whenever it differs
from what was requested.

G is clipped to [G_min, G_max] after each step.

**Spectral sensitivity S(lambda):**
- Single-peak: `exp(-((lambda - lambda_peak) / lambda_width)^2)` → [0, 1]
- Multi-Gaussian (fitted): sum of N Gaussians, signed (positive = potentiation, negative = depression), normalized to [-1, +1] with independent per-branch normalization

**Default parameters:** G_min=1e-6 S, G_max=1e-4 S, alpha=0.8, beta=0.8, A_peak=8e-4, B_peak=6e-4, lambda_peak=550 nm, lambda_width=100 nm, decay_tau=100 s

## STDP Learning Rule

```
delta_G = +A_plus * exp(-dt/tau_plus) * (G_max - G_min) * 0.01   (dt > 0, LTP)
delta_G = -A_minus * exp(dt/tau_minus) * (G_max - G_min) * 0.01  (dt < 0, LTD)
```
Defaults: A_plus=0.5, A_minus=0.3, tau_plus=tau_minus=20 ms. Eligibility window: 5*max(tau) = 100 ms.

## LIF Neuron Model

```
tau_m * dV/dt = -(V - V_rest) + R * I_syn
```
Defaults: tau_m=20 ms, V_rest=V_reset=-70 mV, V_thresh=-50 mV, R=10 MOhm, refract=2 ms.

## Suite-Wide Invariants (critical — do not regress; introduced 2026-07-31)

These invariants were introduced to close verified defects (see `AUDIT_2026-07-31.md` §10). They are
deliberate design, not decoration — do not "simplify" them away.

- **Execution-path parity.** `synapse_engine.pulse_read_sequence` has two implementations of one
  measurement (on-instrument LUA and PC-timed standard). They must return the same thing:
  `n_pulses + 1` points with **index 0 = pre-stimulus baseline**, **fixed** current and voltage
  ranging (never autorange), voltage compliance taken from the instrument (never a hardcoded 10 V),
  **NaN conductance at zero read bias** (never 0 — a fabricated value that averages and plots as
  though measured), and NPLC/averaging derived from the shared `derive_measurement_averaging()`.
  Only the timing source may differ (host clock vs instrument timer). `results['execution_path']`
  is stamped with what actually ran and written into the exported CSV header alongside
  `pulse_index_0=pre_stimulus_baseline`.
- **Single source of truth for instrument capability.** `MODEL_CAPABILITIES` in
  `synapse_engine.py` is the only place model facts (voltage/current ceilings, LUA support, 10 A
  pulse support) live. Never re-introduce per-function hardcoded model lists. An unidentifiable
  instrument raises `InstrumentIdentificationError`; it must never fall back to a "conservative
  default" — guessing a voltage ceiling is a safety decision, not a fallback.
- **Canonical mode vocabulary.** Engine modes are `electrical`, `visual`, `memristor_pulse`,
  `srdp`, `stdp`. GUI display text never reaches the engine:
  `keithley_analyser.SYNAPSE_MODE_BY_LABEL` translates combo labels to canonical modes (reverse map
  `synapse_mode_display()` for display). An unrecognised label or unregistered mode **raises**
  rather than falling through to a different execution path. LUA-capable pulse-read modes are
  enumerated in `synapse_engine.LUA_PULSE_READ_MODES`; a fall to the standard path on a
  LUA-capable instrument prints an explicit warning.
- **Console encoding.** Every entry point calls `console_io.enable_utf8_console()` before anything
  prints (wired into `hub.py`, `keithley_analyser.py`, `Main.py`, `fitting.py`,
  `data_converter.py`, `generate_test_data.py`, `synapse_engine.py`). Non-ASCII physics notation
  (µS, Δt, λ) in `print()` is safe *because of this* and unsafe without it — on a stock cp1252
  Windows console an unshimmed `✓` once caused a completed LUA measurement to be reported as an
  instrument failure and silently re-run on the slow path.
- **`keithley_analyser.py` is importable.** Its `root.mainloop()` is guarded by
  `if __name__ == "__main__":`. Module-level widget construction still runs at import —
  only the event loop is guarded — and `hub.py` launches the file as a script, so the
  GUI path is unchanged. Without the guard `import keithley_analyser` blocks forever,
  which makes `SYNAPSE_MODE_BY_LABEL` and `canonical_synapse_mode()` — the mode-dispatch
  contract the engine depends on — impossible to verify. Do not remove it.
- **Exception scope around LUA execution.** The `try` block guards **only** the instrument call
  (`execute_lua_script_fast`). Result processing and console output stay outside it, so a software
  fault can never masquerade as an instrument failure and trigger a spurious execution-path change.
- **STDP amplitude units — percent of dynamic range.** The rule
  `ΔG = A_plus · exp(−dt/τ) · (G_max − G_min) · 0.01` carries the `0.01` **inside it**, so
  `A_plus = 0.5` means **0.5 percent** of the dynamic range, not half of it. Consequently:
  `fitting.py` fits against `100 · ΔG / (G_max − G_min)` — normalised by the **dynamic range**,
  never by the baseline conductance `G_initial`, which is operating-point dependent and which no
  constant divisor can reconcile with the rule — and **no consumer divides by 100 afterwards**
  (`network.py`'s rule and both `Main.py` call sites apply the `0.01` themselves). Reading
  "fraction of range" intuitively and dividing somewhere lands you exactly 100× off. Also keep
  `stdp_model`'s output array `dtype=float`: a whole-millisecond hardware sweep gives an integer
  Δt array, and `np.zeros_like` would inherit that and truncate the entire LTP branch to zeros at
  the correct ~0.5 amplitudes.
- **Bounded `curve_fit` needs `_FIT_TOL`.** Passing `bounds` switches SciPy from `lm` to `trf`,
  whose stopping tests are far looser. On conductance data (residuals ~1e-6) `trf` reads that as
  "already converged" and **returns the initial guess unchanged, reporting success** — which is how
  `alpha` came back as exactly 0.7999 (from `p0 = 0.8`) and `decay_tau` as exactly 100.0 s
  (numerically identical to the documented default). Every bounded fit in `fitting.py` passes
  `**_FIT_TOL` and `x_scale='jac'`, and the critical ones assert via `_assert_moved_from_start`
  that the parameter actually moved. Removing either silently reintroduces fits that are not fits.
- **Spectral responsivity is the initial rate.** `_build_wavelength_response` measures
  `dG/dn|₀`, not ΔG after a fixed pulse count: the soft-bound update saturates, so an endpoint
  reading cannot distinguish a strongly-driven wavelength from a very strongly-driven one, and the
  fitted width came out 63% too wide against a plateau that R² fitted faithfully. Rates are divided
  by intensity × pulse width so the curve is a responsivity, not `S(λ)·I(λ)`, and normalisation is
  **per branch** (strongest potentiation → +1, strongest depression → −1) as the spec requires.

## Verification

`python -m pytest verification/ -q` — ~5 s, no hardware, no GUI. Proves the
invariants above have not regressed and pins every fitted parameter of the synthetic
pipeline. See `verification/README.md`. Run it before and after any change to
`synapse_engine.py`, `fitting.py`, `generate_test_data.py` or `data_converter.py`.

`generate_test_data.py` accepts `--seed N` for reproducible runs; the harness always
passes one. Unseeded interactive behaviour is unchanged.

## Module Responsibilities

### keithley_analyser.py
GUI instrument controller. Modes: Solar cell JV, Transistor, Synapse (Basic pulse-read, Visual/self-powered, SRDP, STDP, Cycling, Multi-device). Communicates via pyvisa (GPIB/RS-232/LAN). Generates on-instrument LUA scripts for fast acquisition on 2600-series. Safety-checks voltage (40V low-voltage models, 200V high-voltage). PV parameter extraction (Voc, Jsc, FF, PCE). Exports CSV data and JSON characterization suites via assembly_helpers.

### synapse_engine.py
Hardware measurement + simulation engine. Generates LUA scripts for pulse-read, visual synapse, STDP, and **cycle** sequences. Handles VISA communication, buffer management, error queue. Simulation models: exponential convergence for LTP/LTD, SRDP (log-frequency-dependent), STDP (Bi & Poo 1998 asymmetric exponential window). Metrics: PPF, delta_G, delta_G%.

**LUA timing architecture (critical — do not regress):**
- All LUA pulse loops use **absolute timer-based period enforcement**: `timer.reset()` before the loop, `timer.measure.t()` at each iteration end to compute true remainder. This prevents cumulative drift from measurement time, `source.levelv` writes, and buffer appends. Never use chained `delay()` for period timing.
- The cycle LUA script uses a **single monotonic `t_abs` accumulator** across all trains and cycles with one `timer.reset()` at script start. No timer reset between trains or cycles — prevents boundary drift.
- Script upload uses **concatenated chunks** (~512 bytes) to minimize VISA write count — never line-by-line.
- `execute_lua_script_fast` and `execute_cycle_lua` wait for completion via a **TSP `print(LUA_DONE_TOKEN)` sentinel** emitted as the script's last statement, read with an extended timeout. Do **not** use `*OPC?` here — it latches on anonymous `loadandrunscript` and stalls 60–120 s (rationale documented at `synapse_engine.py:17-22`). Fallback is sleep-based (never poll `nvbuffer1.n` during execution — that causes TSP context switches and timing jitter; the `.n` poll happens only after the fallback wait).
- STDP spikes and single-channel cycle write+read use **level changes** (`source.levelv`), not `OUTPUT_ON`/`OUTPUT_OFF` relay toggling.
- All LUA scripts use **anonymous `loadandrunscript`/`endscript`** — `endscript` triggers immediate execution. No named scripts, no `script.delete`, no error -292 on front panel. Never use `loadscript <name>` (named scripts require deletion, and all deletion patterns — `if script.user.scripts.X`, `pcall(script.delete, 'X')` — generate error -292 on the 2636A front panel despite Lua-level error suppression).
- Buffer retrieval uses **chunked `printbuffer`** (500 points/read) to avoid VISA output buffer overflow.
- Post-execution **timing drift analysis** from buffer timestamps is reported in the console for all modes (pulse-read and cycle).

**Cycle LUA generator** (`generate_cycle_lua_script`):
- Supports two channel topologies per train: **single-channel** (same SMU writes and reads via voltage switching) and **dual-channel** (one SMU writes, another reads at constant bias).
- Two independent sequential trains (A then B) per cycle, N-cycle repetition.
- Each train's topology, channels, and timing are independently configurable.
- Buffer management: one buffer per read channel. When both trains share a read channel, data is interleaved in one buffer; when they differ, separate buffers on separate channels.
- Execution helpers: `_upload_and_execute_lua()`, `_retrieve_buffer()`, `execute_cycle_lua()`.

### synapse_cycle.py
Unified LUA-based potentiation-depression cycling engine. All execution runs on-instrument via LUA for sub-ms timing. Supports single-channel write+read, dual-channel stim/read, two independent sequential trains per cycle, and N-cycle repetition. Entry point: `cycle_sequence()`. Metrics: delta_G mean/std per phase, CV% (reproducibility), dynamic range (S), ON/OFF ratio. Presets: Standard, Fast, High Endurance, Asymmetric.

### fitting.py (v3.0)
Model extraction pipeline. DataParser handles v2.0 JSON (from data_converter) and legacy formats. Fits:
- Wavelength response: 1-3 Gaussians, AIC model selection, bounds [300,900] nm
- Nonlinearity: power law `dG = A*(G_max_est - G)^alpha`
- Depression nonlinearity: `|dG| = B*(G - G_min)^beta`
- Retention: `G(t) = G_min_est + (G0 - G_min_est)*exp(-t/tau)`
- Rate constants A_peak, B_peak: minimize_scalar on first 30 pulses
- STDP window: separate LTP/LTD exponential fits
- SRDP: sigmoid `max_change / (1 + exp(-(f-f0)/slope))`

Uses scipy.optimize.curve_fit (Levenberg-Marquardt). R^2 reported. Graceful degradation to physical defaults when data is missing — but defaults are explicitly flagged in extraction_report, never silent.

### generate_test_data.py
Produces synthetic CSV files for the full characterization suite. UV (365 nm) → potentiation, visible (550 nm) → depression. Nonlinear soft-bound update with 1.5% noise. Generates: wavelength sweeps, nonlinearity, cycling, retention decay, STDP window, SRDP sigmoid. Outputs METADATA_INSTRUCTIONS.txt for batch loading.

### data_converter.py (v2.0)
GUI tool: loads CSV/Excel measurement files, collects metadata via dialog, computes derived metrics (PPF, delta_G, ON/OFF ratio, retention tau via curve fit), exports structured JSON with datasets array.

### Main.py
`VisualSynapse` class: the canonical physics implementation. 4-tab GUI: Simulation (interactive pulse/rest), Model Fitting (extract_synapse_model + compare), Network (spatial array with pattern library), SNN (Poisson inputs + STDP learning). Loads experimental JSON characterization → rebuilds VisualSynapse with fitted multi-Gaussian wavelength curve.

### network.py
`SynapseNetwork`: N×M grid of VisualSynapse objects, 10% device-to-device variability. `SpatialPatterns`: 20+ masks (bars, diagonals, checkerboard, circle, duck, thumbs_up, raised_fist, etc.). `LIFNeuron`: leaky integrate-and-fire. `SpikingSynapseNetwork`: 2-layer input→hidden SNN with STDP. `MultiLayerSpikingNetwork`: arbitrary-depth feedforward + optional recurrent connections.

### snn_visualization.py
Poisson and rate-coded spike train generation. Raster plots, weight heatmaps, firing rate estimation, STDP/SRDP curve visualization with fitted analytic overlays. `image_to_spike_trains`: deterministic rate coding from 2D images.

### hub.py
Launcher. Subprocess-based, supports PyInstaller frozen executables. No physics.

### assembly_helpers.py
Builds and validates JSON characterization suites from multiple stored measurements. Used by the Keithley controller's export function.

### excel_patterns.py
Loads custom spatial stimulus patterns from Excel files for the network simulation.

### console_io.py
`enable_utf8_console()`: reconfigures stdout/stderr to UTF-8 with `errors='replace'`; safe when
stdout is absent (pythonw / frozen GUI). Called at the top of every entry point — see Suite-Wide
Invariants. No physics.

## Data Flow

```
Hardware (Keithley SMU) → CSV files (synapse_engine, synapse_cycle)
CSV files → data_converter.py → v2.0 JSON
v2.0 JSON → fitting.py → fitted_model dict
fitted_model → Main.py (VisualSynapse with experimental params)
fitted_model → network.py (SynapseNetwork / SNN arrays)
```

## Key Data Structures

**params dict** (synapse modes): mode, stim_drive_type, stim_level, stim_width_ms, stim_period_ms, n_pulses, read_voltage, read_delay_ms, compliance_A, nplc, line_freq_hz (50/60, or None/absent = Auto: hardware entry points call `resolve_line_freq()` which adopts the instrument's power-up-detected `localnode.linefreq` — never assume a fixed grid, a 50 Hz default once put a 60 Hz user's mains pickup beyond any NPLC's rejection; the GUI combo defaults to "Auto (detect)"), wire_mode, wavelength_nm, intensity_mW_cm2, sample_id

**train_config dict** (cycle mode): topology ('single'/'dual'), write_ch, read_ch, stim_drive_type, stim_level, stim_width_ms, stim_period_ms, n_pulses, read_voltage, read_delay_ms, compliance_A, nplc, line_freq_hz (50/60, or None/absent = Auto — resolved from the instrument by `cycle_sequence` via `resolve_line_freq()` and stamped into both train configs), settle_ms, wire_mode, sample_id

**fitted_model dict**: G_min, G_max, alpha, beta, A_peak, B_peak, lambda_peak, lambda_width, decay_tau, wavelength_curve (callable), wavelength_n_peaks, stdp_*, srdp_*, extraction_report

**v2.0 JSON**: `{datasets: [{filename, x_data, y_data, metadata, derived_metrics}], n_datasets, creation_time, tool_version}`

## Tech Stack
Python 3, CustomTkinter, matplotlib (FigureCanvasTkAgg), numpy, scipy.optimize, pyvisa, PIL/Pillow, openpyxl. PyInstaller-compatible (resource_path, sys.frozen).

## File Naming
- Active files are at project root. `Save *` directories are historical backups — do not modify.
- `keithley_analyser.py` is the canonical Keithley controller. Version is tracked via `__version__` inside the module.
- `Keithley_Dual_SMU_Parameters_Analyser_4_2_2.py` exists as a historical variant — not canonical.
