"""
Phase 1: Single Visual Synapse Proof-of-Concept
================================================

A minimal standalone tool to understand synapse dynamics.

What this demonstrates:
- How light stimuli change synapse conductance
- Effect of wavelength on learning rate
- Nonlinear potentiation behavior
- Conductance saturation (G_min, G_max)

Run this script to see a synapse "learn" in real-time.
"""

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from fitting import (SyntheticCharacterization, extract_synapse_model, 
                     compare_models, plot_fitting_results)

from network import (SynapseNetwork, SpatialPatterns, plot_network_state,
                     plot_network_evolution, create_animation, SpikingSynapseNetwork,
                     SINGLE_SYNAPSE_DV_FRACTION)

from excel_patterns import ExcelPatternLoader, validate_excel_file

from snn_visualization import (generate_poisson_spike_trains, visualize_snn_simulation,
                               plot_weight_evolution, plot_firing_rates)

import json
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as tk_messagebox

# Set appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# =============================================================================
# PEAK OVERRIDE DIALOG
# =============================================================================

class PeakOverrideDialog(ctk.CTkToplevel):
    """Dialog to let user override automatic wavelength peak detection."""
    
    def __init__(self, parent, fitted_model):
        super().__init__(parent)
        
        self.result = None  # Will store selected n_peaks or None
        self.fitted_model = fitted_model
        
        self.title("Wavelength Fitting Options")
        # Tall enough for the 3-peak case, which needs ~423 px at 100% scaling
        # and ~522 px at 125%. At the old fixed 500x400 the Apply and Skip
        # buttons were starved to 5 px of 28 and their labels did not render —
        # and because this dialog is modal with no keyboard bindings, the only
        # way out was the window X, which is indistinguishable from clicking
        # Skip. minsize keeps them reachable however the user resizes.
        self.geometry("520x560")
        self.minsize(480, 480)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Keyboard escape hatches. A modal dialog whose buttons can be clipped
        # must remain dismissible from the keyboard.
        self.bind("<Return>", lambda _event: self.on_apply())
        self.bind("<Escape>", lambda _event: self.on_skip())
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the override dialog UI."""
        
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Wavelength Fitting Results",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 15))
        
        # Show automatic detection results
        auto_n_peaks = self.fitted_model.get('wavelength_n_peaks', 1)
        auto_r2 = self.fitted_model.get('wavelength_fit_R2', 0)
        
        info_text = f"Automatically detected: {auto_n_peaks} peak(s)\n"
        info_text += f"Fit quality R²: {auto_r2:.3f}\n\n"
        
        # Show peak details
        peaks = self.fitted_model.get('wavelength_peaks', [])
        if peaks:
            info_text += "Detected peaks:\n"
            for i, peak in enumerate(peaks):
                peak_type = peak.get('type', 'unknown')
                wavelength = peak.get('wavelength_nm', 0)
                info_text += f"  Peak {i+1}: {wavelength:.0f} nm ({peak_type})\n"
        
        info_label = ctk.CTkLabel(
            main_frame,
            text=info_text,
            font=("Arial", 12),
            justify="left"
        )
        info_label.pack(pady=(0, 20))
        
        # Override options
        override_label = ctk.CTkLabel(
            main_frame,
            text="Override number of peaks?",
            font=("Arial", 12, "bold")
        )
        override_label.pack(pady=(0, 10))
        
        # Radio buttons
        self.peak_choice = ctk.StringVar(value="auto")
        
        radio_frame = ctk.CTkFrame(main_frame)
        radio_frame.pack(pady=(0, 20))
        
        ctk.CTkRadioButton(
            radio_frame,
            text=f"Use automatic ({auto_n_peaks} peak{'s' if auto_n_peaks > 1 else ''})",
            variable=self.peak_choice,
            value="auto"
        ).pack(anchor="w", pady=2)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="Force 1 peak",
            variable=self.peak_choice,
            value="1"
        ).pack(anchor="w", pady=2)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="Force 2 peaks",
            variable=self.peak_choice,
            value="2"
        ).pack(anchor="w", pady=2)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="Force 3 peaks",
            variable=self.peak_choice,
            value="3"
        ).pack(anchor="w", pady=2)
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        apply_btn = ctk.CTkButton(
            button_frame,
            text="Apply",
            command=self.on_apply,
            width=120
        )
        apply_btn.pack(side="left", padx=5)
        
        skip_btn = ctk.CTkButton(
            button_frame,
            text="Skip Override",
            command=self.on_skip,
            width=120
        )
        skip_btn.pack(side="left", padx=5)
    
    def on_apply(self):
        """User wants to apply override."""
        choice = self.peak_choice.get()
        if choice == "auto":
            self.result = None  # Use automatic
        else:
            self.result = int(choice)  # Force specific number
        self.destroy()
    
    def on_skip(self):
        """User wants to skip override and use automatic."""
        self.result = None
        self.destroy()


# =============================================================================
# SYNAPSE MODEL: The Core Physics
# =============================================================================

def resolve_fitted_parameter(fitted_model, key, default, report_names=()):
    """Resolve one parameter from a fitted model, and say where it came from.

    Returns (value, state) where state is one of:
        'fitted'  — genuinely extracted from the data
        'DEFAULT' — the extractor fell back; the report says so
        'FAILED'  — the extractor failed; the value is not trustworthy
        'MISSING' — absent from the model entirely; `default` substituted

    C8: every key used to be read as `.get(key, <hardcoded default>)` with no
    presence check, after which the GUI printed "Experimental parameters
    applied to synapse model" regardless. A decay_tau that was never measured
    was displayed identically to a genuinely fitted G_min. `fitting.py`
    produces exactly the diagnostics needed to prevent this — per-fit R²
    values, and an extraction_report whose entire purpose is to flag fallbacks
    — and none of them were read.

    `report_names` are the extraction_report entries that govern this
    parameter, since one extractor can produce several parameters (the
    wavelength fit produces both lambda_peak and lambda_width, and the report
    names the extractor, not each output).

    Module-level rather than nested inside the loader so it can be tested
    without a file dialog.
    """
    report = (fitted_model.get('extraction_report') or {}) if fitted_model else {}
    defaults_used = set(report.get('defaults_used', []))
    failed_items = set(report.get('failed', []))
    names = tuple(report_names) or (key,)

    if not fitted_model or key not in fitted_model or fitted_model[key] is None:
        return default, 'MISSING'
    if any(n in defaults_used for n in names):
        return fitted_model[key], 'DEFAULT'
    if any(n in failed_items for n in names):
        return fitted_model[key], 'FAILED'
    return fitted_model[key], 'fitted'


# Electrical/optical drive equivalence, in (mW/cm^2) per volt.
#
# A_peak and B_peak are denominated per mW/cm^2 of illumination, so driving the
# same rate constants with a VOLTAGE requires a conversion factor. This one
# asserts that 1 V of electrical bias drives the device as hard as 10 mW/cm^2
# of light.
#
# It is a MODELLING CHOICE, not a measured property, and it sets the entire
# scale of every electrical simulation in the suite. It was previously a bare
# `* 10` inline in two expressions with no name and no explanation, which the
# project's own ground rules forbid: an unexplained hardcoded value that
# determines physical behaviour. Named here so it is visible, greppable, and
# can be replaced by a per-device fitted value when one is measured.
ELECTRICAL_OPTICAL_EQUIVALENCE = 10.0


class VisualSynapse:
    """
    Simple visual synapse model with wavelength-dependent learning.
    
    The conductance G evolves according to:
        dG/dt = A(λ) × I_light × (G_max - G)^α  [potentiation]
                - B(λ) × I_light × (G - G_min)^β  [depression]
                - decay_rate × (G - G_min)      [relaxation]
    
    Parameters:
        G_min, G_max: Conductance bounds (Siemens)
        alpha: Nonlinearity exponent for potentiation
        beta: Nonlinearity exponent for depression
        A_peak: Peak potentiation rate at optimal wavelength
        B_peak: Peak depression rate at optimal wavelength
        lambda_peak: Optimal wavelength (nm)
        lambda_width: Spectral width (nm)
        decay_tau: Relaxation time constant (seconds)
    """
    
    def __init__(self, G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                 lambda_peak=550, lambda_width=100, decay_tau=100, 
                 A_peak=8e-4, B_peak=6e-4, wavelength_curve=None, wavelength_range=None):
        # State variable.
        #
        # G_min * 1.5 lies outside [G_min, G_max] for any device with an
        # ON/OFF ratio below 1.5, and the first clip then silently pins the
        # device to G_max. Clipped here instead, so the initial state is always
        # inside the physical bounds whatever they are.
        self.G = float(np.clip(G_min * 1.5, G_min, G_max))

        # Set by apply_stimulus: describes which branch actually ran and why,
        # whenever that differs from what the caller asked for. None means the
        # request was carried out as stated.
        self.last_stimulus_report = None

        # Physical bounds
        self.G_min = G_min
        self.G_max = G_max
        
        # Learning parameters
        self.alpha = alpha  # Potentiation nonlinearity
        self.beta = beta    # Depression nonlinearity
        self.A_peak = A_peak  # Peak potentiation rate
        self.B_peak = B_peak  # Peak depression rate
        self.lambda_peak = lambda_peak
        self.lambda_width = lambda_width
        self.decay_tau = decay_tau
        
        # Fitted wavelength response curve (can be None for single Gaussian)
        self.wavelength_curve = wavelength_curve
        self.wavelength_range = wavelength_range if wavelength_range else [200, 2000]
        
        # History tracking
        self.history_G = [self.G]
        self.history_t = [0]
        self.history_stimulus = [0]
    
    def wavelength_sensitivity(self, wavelength_nm):
        """
        Get wavelength response - can return negative (depression) or positive (potentiation).
        
        Uses fitted multi-Gaussian curve if available, otherwise simple Gaussian.

        Returns:
            float: Response factor normalized to [-1, 1] range
                  Positive = potentiation, Negative = depression
        """
        # Normalisation constants are computed once and cached — see
        # _branch_normalisation.
        if self.wavelength_curve is not None:
            # Use fitted multi-Gaussian curve (can be negative)
            response = self.wavelength_curve(np.array([wavelength_nm]))[0]
            max_positive, max_negative = self._branch_normalisation()

            # Normalize based on the sign of the response
            if response > 0:
                normalized_response = response / max_positive if max_positive > 0 else 0
            elif response < 0:
                normalized_response = response / max_negative if max_negative > 0 else 0
            else:
                normalized_response = 0

            return normalized_response
        else:
            # Fall back to simple Gaussian (always positive)
            return np.exp(-((wavelength_nm - self.lambda_peak) / self.lambda_width)**2)
    
    # Resolution of the sweep used to find each branch's extremum, in nm.
    #
    # M12: this used 100 points over a range that defaults to [200, 2000] nm —
    # about 18 nm per step. A fitted Gaussian narrower than the grid can have
    # its peak fall between samples, so the "maximum" found is smaller than the
    # true one and the normalised response then EXCEEDS 1 (measured on the
    # synthetic suite: S(365) = +1.006, S(550) = -1.002). That matters more now
    # than when the audit was written, because the spectral fix brought fitted
    # widths down to ~27-32 nm.
    #
    # 1 nm resolves any physically plausible absorption band. The sweep is
    # evaluated once per (curve, range) and cached, so this is cheaper than the
    # 100-point version was — that one ran on EVERY call, i.e. inside the ODE
    # integration loop.
    NORMALISATION_STEP_NM = 1.0

    def _branch_normalisation(self):
        """(max_positive, max_abs_negative) for the fitted curve, cached.

        Per-branch normalisation is the single convention across the suite —
        `fitting.py` builds the spectral response this way and CLAUDE.md
        specifies it — so the strongest potentiation response maps to +1 and
        the strongest depression response to -1, independently.
        """
        cache = getattr(self, '_branch_norm_cache', None)
        key = (id(self.wavelength_curve), tuple(self.wavelength_range))
        if cache is not None and cache[0] == key:
            return cache[1]

        wl_min, wl_max = self.wavelength_range
        # Extend by 20% each side so a peak sitting at the edge of the measured
        # range is still captured.
        extension = (wl_max - wl_min) * 0.2
        wl_min = max(200.0, wl_min - extension)
        wl_max = min(2000.0, wl_max + extension)

        n_points = max(64, int(round((wl_max - wl_min) / self.NORMALISATION_STEP_NM)) + 1)
        grid = np.linspace(wl_min, wl_max, n_points)
        responses = np.asarray(self.wavelength_curve(grid), dtype=float)

        positive = responses[responses > 0]
        negative = responses[responses < 0]
        max_positive = float(np.max(positive)) if positive.size else 1.0
        max_negative = float(np.abs(np.min(negative))) if negative.size else 1.0

        result = (max_positive, max_negative)
        self._branch_norm_cache = (key, result)
        return result

    def apply_stimulus(self, intensity_mW_cm2, wavelength_nm, duration_ms,
                      mode='potentiation', stimulus_type='light', dt_ms=1.0):
        """
        Apply stimulus and update conductance.
        
        Args:
            intensity_mW_cm2: Light intensity (mW/cm²) or voltage magnitude (V)
            wavelength_nm: Light wavelength (nm) - ignored for electrical
            duration_ms: Stimulus duration (milliseconds)
            mode: 'potentiation' or 'depression' - ONLY used for electrical stimuli.
                 For light stimuli, the wavelength response curve determines whether
                 potentiation or depression occurs (mode parameter is ignored).
            stimulus_type: 'light' or 'electrical'
            dt_ms: Time step for integration (milliseconds)
        """
        # Whole steps plus one shortened remainder step, so exactly duration_ms is integrated.
        n_full = int(np.floor(duration_ms / dt_ms + 1e-9))
        remainder_ms = duration_ms - n_full * dt_ms
        if remainder_ms < 1e-9 * max(1.0, duration_ms):
            remainder_ms = 0.0
        step_lengths_ms = [dt_ms] * n_full + ([remainder_ms] if remainder_ms > 0 else [])
        steps = len(step_lengths_ms)

        # A stimulus shorter than one integration step used to be a silent
        # no-op: steps == 0, the loop never ran, and the caller got no
        # indication that nothing had happened. Refuse it instead.
        if n_full < 1 and duration_ms > 0:
            raise ValueError(
                f"Stimulus duration {duration_ms} ms is shorter than the "
                f"integration step {dt_ms} ms, so no step would be taken. "
                "Reduce dt_ms or lengthen the stimulus."
            )
        
        # Reset the per-call branch report. Callers read this to tell the user
        # what actually happened, which is the difference between a physically
        # meaningful branch selection and a silent substitution.
        self.last_stimulus_report = None

        # Calculate effective rates based on stimulus type
        if stimulus_type == 'light':
            # Light-based: wavelength-dependent
            response = self.wavelength_sensitivity(wavelength_nm)

            if self.wavelength_curve is not None:
                # A FITTED multi-Gaussian curve is SIGNED: positive lobes
                # potentiate, negative lobes depress. The wavelength therefore
                # determines the branch, and that is the physics — a device
                # does not depress under its potentiation band because the
                # operator asked it to.
                if response >= 0:
                    A_eff = self.A_peak * abs(response) * intensity_mW_cm2
                    B_eff = 0
                    actual_mode = 'potentiation'
                else:
                    A_eff = 0
                    B_eff = self.B_peak * abs(response) * intensity_mW_cm2
                    actual_mode = 'depression'

                # The requested mode is still meaningful information: if the
                # operator asked for depression at a wavelength whose fitted
                # response potentiates, that is a real conflict between intent
                # and device physics, and the GUI must say so rather than
                # quietly doing the opposite of what was asked.
                if mode != actual_mode:
                    self.last_stimulus_report = {
                        'requested_mode': mode,
                        'actual_mode': actual_mode,
                        'reason': (
                            f"the fitted spectral response at {wavelength_nm:.0f} nm "
                            f"is {response:+.3f} ({actual_mode})"
                        ),
                        'wavelength_nm': wavelength_nm,
                        'response': float(response),
                    }
            else:
                # NO fitted curve: the fallback is a single positive Gaussian
                # (see wavelength_sensitivity), which describes responsivity
                # MAGNITUDE and carries no sign at all. It therefore cannot
                # select a branch, and reading its sign as "always
                # potentiation" made depression unreachable in the default
                # state — selecting Depression raised G toward G_max while the
                # GUI reported depression.
                #
                # With no spectral sign information available, the requested
                # mode is the only branch information there is, so it is
                # honoured, and |S(λ)| supplies the magnitude.
                magnitude = abs(response)
                if mode == 'depression':
                    A_eff = 0
                    B_eff = self.B_peak * magnitude * intensity_mW_cm2
                    actual_mode = 'depression'
                else:
                    A_eff = self.A_peak * magnitude * intensity_mW_cm2
                    B_eff = 0
                    actual_mode = 'potentiation'

                self.last_stimulus_report = {
                    'requested_mode': mode,
                    'actual_mode': actual_mode,
                    'reason': (
                        "no fitted spectral response is loaded, so the branch "
                        "was taken from the requested mode rather than from "
                        "the wavelength; |S(λ)| supplied the magnitude only"
                    ),
                    'wavelength_nm': wavelength_nm,
                    'response': float(response),
                    'unsigned_fallback': True,
                }
        else:
            # Electrical: wavelength-independent, voltage-dependent
            # Use intensity_mW_cm2 as voltage magnitude (V)
            # Mode parameter determines effect (voltage sign in real devices)
            voltage = abs(intensity_mW_cm2)
            if mode == 'potentiation':
                A_eff = self.A_peak * voltage * ELECTRICAL_OPTICAL_EQUIVALENCE
                B_eff = 0
                actual_mode = 'potentiation'
            else:
                A_eff = 0
                B_eff = self.B_peak * voltage * ELECTRICAL_OPTICAL_EQUIVALENCE
                actual_mode = 'depression'
        
        t_start = self.history_t[-1]
        
        t_elapsed_ms = 0.0
        for step in range(steps):
            dt_s = step_lengths_ms[step] / 1000.0
            t_elapsed_ms += step_lengths_ms[step]
            if actual_mode == 'potentiation':
                # Potentiation term
                dG = A_eff * (self.G_max - self.G)**self.alpha * dt_s
            else:  # depression
                # Depression term
                dG = -B_eff * (self.G - self.G_min)**self.beta * dt_s
            
            # Spontaneous relaxation toward G_min.
            #
            # ALWAYS ON (audit M15). This used to be gated on
            # `intensity_mW_cm2 == 0`, i.e. applied only at rest. Relaxation is
            # a thermodynamic process — trapped charge detraps, ions
            # back-diffuse — and it does not pause because a light is shining.
            # Stimulation ADDS a driving term; it does not suspend relaxation.
            #
            # Gating it produced a specific artefact: within one spatial
            # pattern the unlit pixels relaxed while the lit ones had
            # relaxation suppressed, so part of the apparent contrast was an
            # artefact of which pixels happened to be illuminated rather than
            # of the device response.
            #
            # This changes simulated results relative to the pre-2026-07-31
            # behaviour. The effect is small whenever the stimulus dominates
            # (dt/decay_tau is 1e-5 per step at the defaults) and matters
            # exactly where it should: long stimuli, weak drive, and the
            # steady state a sustained stimulus settles into.
            dG_decay = -(self.G - self.G_min) / self.decay_tau * dt_s

            # Update state
            self.G += dG + dG_decay
            self.G = np.clip(self.G, self.G_min, self.G_max)
            
            # Record history. The timestamp is the time at the END of the step
            # just integrated, hence (step + 1): using `step` made each
            # stimulus finish one dt short and duplicated the previous
            # segment's final timestamp as the new segment's first sample.
            self.history_G.append(self.G)
            self.history_t.append(t_start + t_elapsed_ms / 1000.0)
            
            # Record stimulus intensity (negative for depression visualization)
            stim_value = intensity_mW_cm2 if actual_mode == 'potentiation' else -intensity_mW_cm2
            self.history_stimulus.append(stim_value if intensity_mW_cm2 > 0 else 0)
    
    def rest(self, duration_ms, dt_ms=10.0):
        """Allow synapse to relax with no stimulus.

        dt_ms was previously passed positionally into the `stimulus_type`
        slot, so every rest routed through the ELECTRICAL branch and
        integrated at the 1 ms default instead of the spec's 10 ms. The
        numerical result was unaffected (a finer step is more accurate and
        decay still applied), but `dt_ms` was dead and each rest generated ten
        times the history inside a loop that re-renders it.
        """
        self.apply_stimulus(0, 0, duration_ms, mode='potentiation',
                            stimulus_type='electrical', dt_ms=dt_ms)

    def reset(self):
        """Reset synapse to initial state."""
        self.G = float(np.clip(self.G_min * 1.5, self.G_min, self.G_max))
        self.history_G = [self.G]
        self.history_t = [0]
        self.history_stimulus = [0]
    
    def read_current(self, V_read=0.1):
        """Read synapse current at given voltage (like hardware)."""
        return self.G * V_read


# =============================================================================
# CUSTOMTKINTER GUI
# =============================================================================

class SynapseSimulatorApp(ctk.CTk):
    """Interactive CustomTkinter GUI for synapse simulation."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Visual Synapse Simulator - Phase 1 Proof of Concept")
        self.geometry("1400x800")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Unhandled exceptions in any Tk callback become a visible dialog
        # instead of a stderr traceback nobody reads. Without this, a handler
        # that raised left its button disabled forever with no explanation.
        from gui_errors import install_tk_error_reporter
        install_tk_error_reporter(self, "Visual Synapse Simulator",
                                  on_error=self._restore_action_buttons)
        
        # Create synapse
        self.synapse = VisualSynapse()
        self.using_experimental_params = False  # Flag to track if experimental data is loaded
        self.loaded_experimental_data = None  # Store loaded experimental data
        self.loaded_experimental_params = None  # Store fitted parameters
        self.loaded_fitted_model = None  # Store full fitted model with metrics
        
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel: Scrollable Controls
        left_panel_container = ctk.CTkFrame(main_container, width=350)
        left_panel_container.pack(side="left", fill="both", padx=(0, 10))
        left_panel_container.pack_propagate(False)
        
        # Create scrollable frame inside left panel
        left_panel = ctk.CTkScrollableFrame(left_panel_container, width=330)
        left_panel.pack(fill="both", expand=True)
        
        # Right panel: Plots
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="left", fill="both", expand=True)
        
        self.setup_controls(left_panel)
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(right_panel)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab 1: Simulation (existing plots)
        self.tab_simulation = self.notebook.add("Simulation")
        
        # Tab 2: Model Fitting
        self.tab_fitting = self.notebook.add("Model Fitting")
        
        # Tab 3: Network Simulation
        self.tab_network = self.notebook.add("Network")
        
        # Tab 4: SNN Simulation
        self.tab_snn = self.notebook.add("SNN")
        
        self.setup_plots(self.tab_simulation)
        self.setup_fitting_tab(self.tab_fitting)
        
        self.setup_network_tab(self.tab_network)
        self.setup_snn_tab(self.tab_snn)
        
        # Initialize network
        self.network = None
        
        # Initialize SNN
        self.snn = None
        self.snn_results = None
        self.snn_input_spikes = None
        
        # Initial update
        self.update_plots()
        self.update_info()
        
        self.custom_pattern = None  # Stores loaded Excel pattern
        self.custom_pattern_path = None  # Stores file path for reference
        
        
    def load_excel_pattern(self):
        """Load a custom pattern from an Excel file."""
        from tkinter.filedialog import askopenfilename
        from tkinter import messagebox as tk_messagebox
        
        # Open file dialog
        filepath = askopenfilename(
            title="Select Excel Pattern File",
            filetypes=[
                ("Excel Files", "*.xlsx *.xlsm"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return  # User cancelled
        
        # Validate file
        is_valid, message = validate_excel_file(filepath)
        
        if not is_valid:
            tk_messagebox.showerror("Invalid File", message)
            return
        
        try:
            # Load the pattern (raw, will be resized when applied)
            self.custom_pattern = ExcelPatternLoader.load_pattern_from_excel(
                filepath, max_size=500
            )
            self.custom_pattern_path = filepath
            
            # Update label
            filename = filepath.split('/')[-1].split('\\')[-1]  # Get filename
            self.custom_pattern_label.configure(
                text=f"✓ {filename} ({self.custom_pattern.shape[0]}×{self.custom_pattern.shape[1]})",
                text_color="green"
            )
            
            # Auto-select "Custom (from Excel)" in dropdown
            self.pattern_var.set("Custom (from Excel)")
            
            # Show success message with preview option
            response = tk_messagebox.askyesno(
                "Pattern Loaded Successfully",
                f"Loaded pattern from:\n{filename}\n\n"
                f"Pattern size: {self.custom_pattern.shape[0]}×{self.custom_pattern.shape[1]}\n"
                f"Intensity range: {self.custom_pattern.min():.2f} to {self.custom_pattern.max():.2f}\n\n"
                "Would you like to preview the pattern?"
            )
            
            if response:
                # Show preview
                fig = ExcelPatternLoader.preview_pattern(
                    self.custom_pattern,
                    title=f"Custom Pattern: {filename}"
                )
                            
        except Exception as e:
            tk_messagebox.showerror(
                "Error Loading Pattern",
                f"Failed to load Excel pattern:\n{str(e)}"
            )
            self.custom_pattern = None
            self.custom_pattern_path = None
            self.custom_pattern_label.configure(
                text="Failed to load pattern",
                text_color="red"
            )    
    
    def update_rest_label(self, value):
        """Update rest duration label."""
        self.rest_label.configure(text=f"{int(value)} ms")
    
    def setup_controls(self, parent):
        """Setup control panel."""
        
        # Title
        title = ctk.CTkLabel(parent, text="SYNAPSE CONTROLS", 
                            font=("Arial", 16, "bold"))
        title.pack(pady=(10, 20))
        
        # Mode Selection
        mode_frame = ctk.CTkFrame(parent)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(mode_frame, text="OPERATION MODE", 
                    font=("Arial", 12, "bold")).pack(pady=5)
        
        self.mode_var = ctk.StringVar(value="potentiation")
        
        self.mode_radio_frame = ctk.CTkFrame(mode_frame)
        self.mode_radio_frame.pack(pady=5)
        
        ctk.CTkRadioButton(self.mode_radio_frame, text="Potentiation (LTP)", 
                          variable=self.mode_var, value="potentiation",
                          command=self.on_mode_change).pack(anchor="w", padx=10)
        ctk.CTkRadioButton(self.mode_radio_frame, text="Depression (LTD)", 
                          variable=self.mode_var, value="depression",
                          command=self.on_mode_change).pack(anchor="w", padx=10)
        
        # Stimulus Type Selection (for depression)
        self.stimulus_type_frame = ctk.CTkFrame(mode_frame)
        
        ctk.CTkLabel(self.stimulus_type_frame, text="Depression Type:", 
                    anchor="w", font=("Arial", 10)).pack(anchor="w", padx=10, pady=(5,0))
        
        self.stimulus_type_var = ctk.StringVar(value="light")
        
        stim_radio_frame = ctk.CTkFrame(self.stimulus_type_frame)
        stim_radio_frame.pack(pady=5)
        
        ctk.CTkRadioButton(stim_radio_frame, text="Optical (wavelength-dependent)", 
                          variable=self.stimulus_type_var, value="light",
                          command=self.on_stimulus_type_change).pack(anchor="w", padx=10)
        ctk.CTkRadioButton(stim_radio_frame, text="Electrical (voltage pulses)", 
                          variable=self.stimulus_type_var, value="electrical",
                          command=self.on_stimulus_type_change).pack(anchor="w", padx=10)
        
        # Initially hidden (only show for depression mode)
        
        
        # Stimulus Parameters Section
        params_frame = ctk.CTkFrame(parent)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(params_frame, text="STIMULUS PARAMETERS", 
                    font=("Arial", 12, "bold")).pack(pady=5)
        
        # Intensity slider (also used for voltage in electrical depression mode)
        self.intensity_static_label = ctk.CTkLabel(params_frame, text="Intensity (mW/cm²):", 
                    anchor="w")
        self.intensity_static_label.pack(anchor="w", padx=10)
        self.intensity_slider = ctk.CTkSlider(params_frame, from_=0.1, to=100, 
                                              number_of_steps=999)
        self.intensity_slider.set(10)
        self.intensity_slider.pack(fill="x", padx=10, pady=5)
        self.intensity_label = ctk.CTkLabel(params_frame, text="10.0 mW/cm²")
        self.intensity_label.pack()
        self.intensity_slider.configure(command=self.update_intensity_label)
        
        # Wavelength slider (will be updated based on loaded data range)
        ctk.CTkLabel(params_frame, text="Wavelength (nm):", 
                    anchor="w").pack(anchor="w", padx=10, pady=(10, 0))
        # Wavelength slider with adaptive range
        wl_min = 200
        wl_max = 2000
        if self.using_experimental_params and hasattr(self.synapse, 'wavelength_range'):
            data_min, data_max = self.synapse.wavelength_range
            wl_min = max(200, data_min - 50)
            wl_max = min(2000, data_max + 50)
        
        self.wavelength_slider = ctk.CTkSlider(params_frame, from_=wl_min, to=wl_max, 
                                               number_of_steps=130)
        self.wavelength_slider.set(550)
        self.wavelength_slider.pack(fill="x", padx=10, pady=5)
        self.wavelength_label = ctk.CTkLabel(params_frame, text="550 nm (Green)")
        self.wavelength_label.pack()
        self.wavelength_slider.configure(command=self.update_wavelength_label)
        
        # Pulse width slider
        ctk.CTkLabel(params_frame, text="Pulse Width (ms):", 
                    anchor="w").pack(anchor="w", padx=10, pady=(10, 0))
        self.duration_slider = ctk.CTkSlider(params_frame, from_=10, to=500, 
                                             number_of_steps=49)
        self.duration_slider.set(100)
        self.duration_slider.pack(fill="x", padx=10, pady=5)
        self.duration_label = ctk.CTkLabel(params_frame, text="100 ms")
        self.duration_label.pack()
        self.duration_slider.configure(command=self.update_duration_label)
        
        # Rest duration slider
        ctk.CTkLabel(params_frame, text="Rest Between Pulses (ms):", anchor="w").pack(anchor="w", padx=10, pady=(10, 0))

        
        self.rest_slider = ctk.CTkSlider(params_frame, from_=0, to=500, 
                                          command=self.update_rest_label)
        self.rest_slider.set(100)
        self.rest_slider.pack(fill="x", padx=10, pady=5)
        
        self.rest_label = ctk.CTkLabel(params_frame, text="100 ms")
        self.rest_label.pack()
        
        # Number of pulses slider
        ctk.CTkLabel(params_frame, text="Number of Pulses:", 
                    anchor="w").pack(anchor="w", padx=10, pady=(10, 0))
        self.n_pulses_slider = ctk.CTkSlider(params_frame, from_=1, to=50, 
                                             number_of_steps=49)
        self.n_pulses_slider.set(10)
        self.n_pulses_slider.pack(fill="x", padx=10, pady=5)
        self.n_pulses_label = ctk.CTkLabel(params_frame, text="10 pulses")
        self.n_pulses_label.pack()
        self.n_pulses_slider.configure(command=self.update_n_pulses_label)
        
        # TARGET SELECTION
        target_frame = ctk.CTkFrame(params_frame)
        target_frame.pack(fill="x", padx=10, pady=(15, 5))
        
        ctk.CTkLabel(
            target_frame,
            text="TARGET",
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        self.target_var = ctk.StringVar(value="single")
        
        target_radio_frame = ctk.CTkFrame(target_frame)
        target_radio_frame.pack(pady=5)
        
        ctk.CTkRadioButton(
            target_radio_frame,
            text="Single Synapse",
            variable=self.target_var,
            value="single",
            command=self.on_target_change
        ).pack(anchor="w", padx=10)
        
        ctk.CTkRadioButton(
            target_radio_frame,
            text="Network",
            variable=self.target_var,
            value="network",
            command=self.on_target_change
        ).pack(anchor="w", padx=10)
        
        
        # NETWORK SIZE SELECTION (only visible for network target)
        self.network_size_frame = ctk.CTkFrame(target_frame)
        
        ctk.CTkLabel(
            self.network_size_frame,
            text="Network Size:",
            font=("Arial", 10)
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        # Size selection frame
        size_selection_frame = ctk.CTkFrame(self.network_size_frame)
        size_selection_frame.pack(padx=10, pady=5)
        
        # Rows input
        ctk.CTkLabel(size_selection_frame, text="Rows:", font=("Arial", 9)).grid(
            row=0, column=0, padx=5, pady=2, sticky="e"
        )
        self.network_rows_entry = ctk.CTkEntry(
            size_selection_frame, width=50, placeholder_text="3"
        )
        self.network_rows_entry.grid(row=0, column=1, padx=5, pady=2)
        self.network_rows_entry.insert(0, "3")
        
        # Columns input
        ctk.CTkLabel(size_selection_frame, text="Cols:", font=("Arial", 9)).grid(
            row=0, column=2, padx=5, pady=2, sticky="e"
        )
        self.network_cols_entry = ctk.CTkEntry(
            size_selection_frame, width=50, placeholder_text="3"
        )
        self.network_cols_entry.grid(row=0, column=3, padx=5, pady=2)
        self.network_cols_entry.insert(0, "3")
        
        # Quick size buttons
        quick_size_frame = ctk.CTkFrame(self.network_size_frame)
        quick_size_frame.pack(padx=10, pady=(0, 5))
        
        ctk.CTkLabel(quick_size_frame, text="Quick:", font=("Arial", 9)).pack(
            side="left", padx=5
        )
        
        # Variability input
        ctk.CTkLabel(
            self.network_size_frame,
            text="Device Variability (%):",
            font=("Arial", 9)
        ).pack(anchor="w", padx=10, pady=(10, 0))
        
        self.variability_entry = ctk.CTkEntry(
            self.network_size_frame, width=80, placeholder_text="2.0"
        )
        self.variability_entry.pack(padx=10, pady=2)
        self.variability_entry.insert(0, "2.0")
        
        def set_network_size(rows, cols):
            """Helper to set network size."""
            self.network_rows_entry.delete(0, "end")
            self.network_rows_entry.insert(0, str(rows))
            self.network_cols_entry.delete(0, "end")
            self.network_cols_entry.insert(0, str(cols))
        
        ctk.CTkButton(
            quick_size_frame, text="3×3", width=40, height=24,
            command=lambda: set_network_size(3, 3)
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            quick_size_frame, text="5×5", width=40, height=24,
            command=lambda: set_network_size(5, 5)
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            quick_size_frame, text="7×7", width=40, height=24,
            command=lambda: set_network_size(7, 7)
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            quick_size_frame, text="10×10", width=50, height=24,
            command=lambda: set_network_size(10, 10)
        ).pack(side="left", padx=2)
        
        # Initially hidden (single synapse mode)
        # Will be shown when network target selected
        
        
        # Pattern selection (only visible for network target)
        self.pattern_selection_frame = ctk.CTkFrame(target_frame)
        
        ctk.CTkLabel(
            self.pattern_selection_frame,
            text="Spatial Pattern:",
            font=("Arial", 10)
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        from network import SpatialPatterns
        pattern_names = list(SpatialPatterns.get_all_patterns().keys())
        self.pattern_var = ctk.StringVar(value=pattern_names[0])
        
        self.pattern_dropdown = ctk.CTkOptionMenu(
            self.pattern_selection_frame,
            variable=self.pattern_var,
            values=pattern_names,
            width=260
        )
        self.pattern_dropdown.pack(padx=10, pady=5)
        
        # Button to load custom Excel pattern
        self.btn_load_pattern = ctk.CTkButton(
            self.pattern_selection_frame,
            text="📁 Load Excel Pattern",
            command=self.load_excel_pattern,
            fg_color="#1976D2",
            hover_color="#0D47A1",
            height=32
        )
        self.btn_load_pattern.pack(side="left", padx=(10, 0))
        
        # Label to show loaded file
        self.custom_pattern_label = ctk.CTkLabel(
            self.pattern_selection_frame,
            text="No custom pattern loaded",
            font=("Arial", 10),
            text_color="gray"
        )
        self.custom_pattern_label.pack(side="left", padx=(10, 0))
        
        # Initially hidden (single synapse mode)
        # Will be shown when network target selected
        
        # Control buttons
        button_frame = ctk.CTkFrame(parent)
        button_frame.pack(fill="x", padx=10, pady=20)
        
        self.btn_stimulate = ctk.CTkButton(button_frame, text="Apply Potentiation",
                                           command=self.on_stimulate,
                                           fg_color="#2E7D32", hover_color="#1B5E20",
                                           height=40, font=("Arial", 14, "bold"))
        self.btn_stimulate.pack(fill="x", pady=5)
        
        self.btn_auto = ctk.CTkButton(button_frame, text="Auto Demo",
                                      command=self.on_auto_demo,
                                      fg_color="#1565C0", hover_color="#0D47A1",
                                      height=35)
        self.btn_auto.pack(fill="x", pady=5)
        
        # Fitting Demo button
        self.btn_fitting = ctk.CTkButton(
            button_frame,
            text="Run Fitting Demo",
            command=self.on_fitting_demo,
            fg_color="#9C27B0",
            hover_color="#7B1FA2",
            height=35
        )
        self.btn_fitting.pack(fill="x", pady=5)
        
        # Network Demo button
        self.btn_network = ctk.CTkButton(
            button_frame,
            text="Run Network Demo",
            command=self.on_network_demo,
            fg_color="#FF6F00",
            hover_color="#E65100",
            height=35
        )
        self.btn_network.pack(fill="x", pady=5)
        
        self.btn_reset = ctk.CTkButton(button_frame, text="Reset",
                                       command=self.on_reset,
                                       fg_color="#C62828", hover_color="#8E0000",
                                       height=35)
        self.btn_reset.pack(fill="x", pady=5)
        
        # Info display
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(info_frame, text="SYNAPSE STATE", 
                    font=("Arial", 12, "bold")).pack(pady=5)
        
        self.info_text = ctk.CTkTextbox(info_frame, height=350, 
                                        font=("Courier", 11))
        self.info_text.pack(fill="x", padx=5, pady=5)
    
    def setup_plots(self, parent):
        """Setup matplotlib plots."""
        
        # Create figure with 3 subplots
        self.fig = plt.Figure(figsize=(10, 8), facecolor='white')
        
        # Main plot: Conductance vs Time
        self.ax_main = self.fig.add_subplot(2, 1, 1)
        self.line_G, = self.ax_main.plot([], [], 'b-', linewidth=2, 
                                         label='Conductance G(t)')
        self.ax_main.set_xlabel('Time (s)', fontsize=11)
        self.ax_main.set_ylabel('Conductance (µS)', fontsize=11, color='b')
        self.ax_main.tick_params(axis='y', labelcolor='b')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper left')
        self.ax_main.set_title('Synapse Dynamics', fontsize=12, fontweight='bold')
        
        # Secondary y-axis for stimulus
        self.ax_stim = self.ax_main.twinx()
        self.line_stim, = self.ax_stim.plot([], [], 'r--', alpha=0.5, linewidth=1.5,
                                            label='Light Stimulus')
        self.ax_stim.set_ylabel('Intensity (mW/cm²)', color='r', fontsize=11)
        self.ax_stim.tick_params(axis='y', labelcolor='r')
        self.ax_stim.legend(loc='upper right')
        
        # Wavelength response curve
        self.ax_spectrum = self.fig.add_subplot(2, 1, 2)
        wl_range = self.synapse.wavelength_range
        wavelengths = np.linspace(wl_range[0], wl_range[1], 200)
        sensitivities = [self.synapse.wavelength_sensitivity(w) for w in wavelengths]
        self.spectrum_line, = self.ax_spectrum.plot(wavelengths, sensitivities, 'g-', linewidth=2, label='Sensitivity')
        self.wavelength_marker = self.ax_spectrum.axvline(550, color='orange', 
                                                          linestyle='--', linewidth=2,
                                                          label='Current λ')
        self.ax_spectrum.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax_spectrum.set_ylabel('Sensitivity', fontsize=11)
        self.ax_spectrum.set_title('Wavelength Response Curve', fontsize=12, fontweight='bold')
        self.ax_spectrum.grid(True, alpha=0.3)
        self.ax_spectrum.set_xlim(wl_range)
        self.ax_spectrum.set_ylim([-0.1, 1.1])
        self.ax_spectrum.legend()
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def setup_fitting_tab(self, parent):
        """Setup the model fitting demonstration tab."""
        
        # Main container
        container = ctk.CTkFrame(parent)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            container,
            text="PHASE 2: MODEL FITTING DEMONSTRATION",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Info frame
        info_frame = ctk.CTkFrame(container)
        info_frame.pack(fill="x", padx=5, pady=5)
        
        info_text = (
            "This tab demonstrates the characterization → model → simulation pipeline.\n\n"
            "Process:\n"
            "  1. Generate synthetic characterization data (like hardware measurements)\n"
            "  2. Fit mathematical models to extract parameters\n"
            "  3. Compare fitted vs. ground truth parameters\n\n"
            "Click 'Run Fitting Demo' to see the full process."
        )
        
        info_label = ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=("Arial", 11),
            justify="left"
        )
        info_label.pack(padx=10, pady=10)
        
        # Results text area
        self.fitting_results_text = ctk.CTkTextbox(
            container,
            font=("Courier", 10),
            wrap="none"
        )
        self.fitting_results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initial message
        self.fitting_results_text.insert("1.0", 
            "Click 'Run Fitting Demo' button to start the fitting demonstration.\n\n"
            "This will:\n"
            "  • Generate synthetic characterization data\n"
            "  • Fit synapse models to the data\n"
            "  • Display comparison plots\n"
            "  • Show parameter accuracy\n\n"
            "OR\n\n"
            "Click 'Load Experimental Model' to import parameters from hardware measurements."
        )
        
        # Button frame for experimental model loading
        button_frame = ctk.CTkFrame(container)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        # Load experimental model button
        load_exp_button = ctk.CTkButton(
            button_frame,
            text="📁 Load Experimental Model",
            command=self.load_experimental_synapse_model,
            fg_color="#1565C0",
            hover_color="#0D47A1",
            height=40,
            font=("Arial", 12, "bold")
        )
        load_exp_button.pack(fill="x", padx=10)
        
        # Info label for button
        info_label = ctk.CTkLabel(
            button_frame,
            text="Load fitted parameters from Keithley characterization",
            font=("Arial", 9),
            text_color="gray"
        )
        info_label.pack(pady=(5, 0))
    
    
    
    def load_experimental_synapse_model(self):
        """
        Load fitted synapse parameters from experimental characterization.
        
        This allows users to import parameters fitted from real hardware measurements
        and use them to initialize network simulations.
        
        Supported file types:
        - Fitted model JSON (output from fitting.extract_synapse_model)
        - Characterization suite JSON (will extract parameters first)
        
        Returns:
            dict: Parameters ready for VisualSynapse or SynapseNetwork, or None if cancelled
        """
        # Open file dialog
        json_path = askopenfilename(
            title="Select Fitted Model or Characterization Suite JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="."
        )
        
        if not json_path:
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a characterization suite or fitted model
            if 'dynamic_range' in data or 'wavelength_response' in data or 'nonlinearity' in data:
                # This is a v1.0 characterization suite - need to fit it first
                self.fitting_results_text.insert("end", 
                    "\n" + "=" * 60 + "\n"
                    + "LOADING CHARACTERIZATION SUITE (v1.0 format)\n"
                    + "=" * 60 + "\n"
                    "Detected v1.0 characterization suite format.\n"
                    "Extracting fitted parameters...\n\n"
                )
                
                # Import fitting module
                try:
                    from fitting import extract_synapse_model
                    fitted_model = extract_synapse_model(data)
                    
                    self.fitting_results_text.insert("end", 
                        "✓ Successfully extracted synapse model from characterization data.\n\n"
                    )
                    
                except Exception as e:
                    tk_messagebox.showerror(
                        "Fitting Error",
                        f"Failed to extract synapse model:\n{str(e)}\n\n"
                        "Make sure the characterization suite has required datasets."
                    )
                    return None
            elif 'datasets' in data:
                # This is a v2.0 format with raw datasets - need to fit it first
                self.fitting_results_text.insert("end", 
                    "\n" + "=" * 60 + "\n"
                    + "LOADING CHARACTERIZATION SUITE (v2.0 format)\n"
                    + "=" * 60 + "\n"
                    "Detected v2.0 dataset format.\n"
                    "Extracting fitted parameters...\n\n"
                )
                
                # Import fitting module and parse the data
                try:
                    from fitting import extract_synapse_model
                    fitted_model = extract_synapse_model(data)
                    
                    self.fitting_results_text.insert("end", 
                        "✓ Successfully extracted synapse model from characterization data.\n\n"
                    )
                    
                except Exception as e:
                    tk_messagebox.showerror(
                        "Fitting Error",
                        f"Failed to extract synapse model:\n{str(e)}\n\n"
                        "Make sure the characterization suite has required datasets."
                    )
                    return None
            elif 'fitted_model_format' in data:
                # A saved fitted model. Its curves are rebuilt from the stored
                # Gaussian coefficients — see fitting.deserialise_fitted_model.
                #
                # C9: this branch previously did `fitted_model = data`, so the
                # wavelength curve was simply absent (nothing could ever write
                # one, because json.dump cannot serialise a closure). The
                # synapse then fell back to a positive single Gaussian, which
                # is exactly the state that makes "Apply Depression"
                # potentiate.
                from fitting import deserialise_fitted_model
                fitted_model = deserialise_fitted_model(data)
                self.fitting_results_text.insert(
                    "end",
                    "\n" + "=" * 60 + "\n"
                    "LOADING SAVED FITTED MODEL\n"
                    + "=" * 60 + "\n"
                    f"Format version {data['fitted_model_format']}.\n"
                    f"Spectral curve rebuilt from "
                    f"{len(data.get('wavelength_peaks', []))} fitted Gaussian(s).\n\n"
                )
            else:
                tk_messagebox.showerror(
                    "Unrecognised File",
                    "This JSON is neither a characterization suite (no "
                    "'datasets' array, no v1.0 measurement keys) nor a saved "
                    "fitted model (no 'fitted_model_format' key).\n\n"
                    "Loading it as a fitted model would silently produce a "
                    "synapse with no spectral response."
                )
                return None
            
            # Convert to base_params format (standardized parameter names).
            #
            # C8: every key used to be read with .get(key, <hardcoded default>)
            # and no presence check, after which the GUI printed
            # "Experimental parameters applied to synapse model" regardless. A
            # decay_tau that was never measured was displayed identically to a
            # genuinely fitted G_min. `fitting.py` produces exactly the
            # diagnostics needed to prevent this — per-fit R² values and an
            # extraction_report whose entire purpose is to flag fallbacks — and
            # none of them were read.
            #
            # Each parameter is now resolved through _resolve_param, which
            # records where the value came from so the display can say so.
            provenance = {}

            # The extraction report itself, bound once here. Every consumer
            # below reads THIS name; it was previously referenced without ever
            # being assigned, which raised NameError on every single load and
            # skipped the whole tail of this method — including the line that
            # actually applies the fitted parameters to the synapse.
            report = fitted_model.get('extraction_report') or {}

            def _resolve_param(key, default, report_names=()):
                value, state = resolve_fitted_parameter(
                    fitted_model, key, default, report_names)
                provenance[key] = state
                return value

            base_params = {
                'G_min': _resolve_param('G_min', 1e-6, ('G_min', 'dynamic_range')),
                'G_max': _resolve_param('G_max', 1e-4, ('G_max', 'dynamic_range')),
                'alpha': _resolve_param('alpha', 0.8, ('alpha', 'nonlinearity')),
                'beta': _resolve_param('beta', 0.8, ('beta',)),
                'lambda_peak': _resolve_param('lambda_peak', 550,
                                              ('lambda_peak', 'wavelength_response')),
                'lambda_width': _resolve_param('lambda_width', 100,
                                               ('lambda_width', 'wavelength_response')),
                'A_peak': _resolve_param('A_peak', 8e-4),
                'B_peak': _resolve_param('B_peak', 6e-4),
                'decay_tau': _resolve_param('decay_tau', 100, ('decay_tau', 'retention')),
                'wavelength_curve': fitted_model.get('wavelength_curve'),
                'wavelength_range': fitted_model.get('wavelength_range', [200, 2000])
            }

            # Display loaded parameters
            self.fitting_results_text.insert("end",
                "\n" + "=" * 60 + "\n"
                + "EXPERIMENTAL SYNAPSE PARAMETERS LOADED\n"
                + "=" * 60 + "\n"
            )

            def _tag(key):
                state = provenance.get(key, 'fitted')
                if state == 'fitted':
                    return ''
                if state == 'MISSING':
                    return '   <-- NOT IN FILE, hardcoded default substituted'
                if state == 'FAILED':
                    return '   <-- FIT FAILED, value is not trustworthy'
                return '   <-- DEFAULT, not measured'

            self.fitting_results_text.insert("end",
                f"Source file: {json_path.split('/')[-1]}\n\n"
                f"Loaded Parameters:\n"
                f"  G_min:        {base_params['G_min']*1e6:.2f} µS{_tag('G_min')}\n"
                f"  G_max:        {base_params['G_max']*1e6:.2f} µS{_tag('G_max')}\n"
                f"  On/Off Ratio: {base_params['G_max']/base_params['G_min']:.1f}:1\n"
                f"  alpha:        {base_params['alpha']:.3f}{_tag('alpha')}\n"
                f"  beta:         {base_params['beta']:.3f}{_tag('beta')}\n"
                f"  A_peak:       {base_params['A_peak']:.3e}{_tag('A_peak')}\n"
                f"  B_peak:       {base_params['B_peak']:.3e}{_tag('B_peak')}\n"
                f"  lambda_peak:  {base_params['lambda_peak']:.1f} nm{_tag('lambda_peak')}\n"
                f"  lambda_width: {base_params['lambda_width']:.1f} nm{_tag('lambda_width')}\n"
                f"  decay_tau:    {base_params['decay_tau']:.1f} s{_tag('decay_tau')}\n\n"
            )

            # Fit quality, which the GUI previously discarded entirely.
            r2_keys = [
                ('wavelength_fit_R2', 'Wavelength response'),
                ('nonlinearity_fit_R2', 'Nonlinearity (alpha)'),
                ('depression_nonlinearity_fit_R2', 'Depression nonlinearity (beta)'),
                ('retention_fit_R2', 'Retention (decay_tau)'),
                ('stdp_fit_R2', 'STDP window'),
                ('srdp_fit_R2', 'SRDP response'),
            ]
            reported_r2 = [(label, fitted_model[k]) for k, label in r2_keys
                           if fitted_model.get(k) is not None]
            if reported_r2:
                self.fitting_results_text.insert("end", "Fit quality (R²):\n")
                for label, value in reported_r2:
                    if value != value:  # NaN
                        note = '   <-- UNDEFINED (constant data)'
                    elif value < 0.8:
                        note = '   <-- POOR'
                    else:
                        note = ''
                    self.fitting_results_text.insert(
                        "end", f"  {label:32s} {value:.4f}{note}\n"
                    )
                self.fitting_results_text.insert("end", "\n")

            # The extraction report, surfaced rather than discarded.
            substituted = sorted(k for k, v in provenance.items() if v != 'fitted')
            if substituted or report.get('failed') or report.get('warnings'):
                self.fitting_results_text.insert(
                    "end",
                    "-" * 60 + "\n"
                    + "NOT EVERYTHING IN THIS MODEL WAS MEASURED\n"
                    + "-" * 60 + "\n"
                )
                if substituted:
                    self.fitting_results_text.insert(
                        "end",
                        f"  Substituted or unfitted: {', '.join(substituted)}\n"
                    )
                for item in report.get('failed', []):
                    self.fitting_results_text.insert("end", f"  FAILED: {item}\n")
                for warning in report.get('warnings', []):
                    self.fitting_results_text.insert("end", f"  ! {warning}\n")
                self.fitting_results_text.insert("end", "\n")

            self.loaded_param_provenance = provenance
            # Remembered so every tab's status line can name the source file
            # rather than just asserting "experimental".
            try:
                self.loaded_model_source = os.path.basename(json_path)
            except Exception:
                self.loaded_model_source = 'loaded model'

            # Always update the synapse with experimental parameters
            self.synapse = VisualSynapse(
                G_min=base_params['G_min'],
                G_max=base_params['G_max'],
                alpha=base_params['alpha'],
                beta=base_params['beta'],
                lambda_peak=base_params['lambda_peak'],
                lambda_width=base_params['lambda_width'],
                decay_tau=base_params['decay_tau'],
                A_peak=base_params['A_peak'],
                B_peak=base_params['B_peak'],
                wavelength_curve=fitted_model.get('wavelength_curve'),
                wavelength_range=base_params['wavelength_range']
            )
            
            # Mark that we're using experimental parameters
            self.using_experimental_params = True
            
            # Store the loaded data for fitting visualization
            self.loaded_experimental_data = data
            self.loaded_experimental_params = base_params
            self.loaded_fitted_model = fitted_model  # Store full fitted model with all metrics
            
            # Reports what was actually applied. This used to be an
            # unconditional "✓ Experimental parameters applied", printed
            # identically whether every parameter had been fitted or every one
            # had been silently defaulted.
            n_fitted = sum(1 for v in provenance.values() if v == 'fitted')
            n_total = len(provenance)
            if n_fitted == n_total:
                headline = f"All {n_total} parameters applied from measured data.\n"
            else:
                headline = (
                    f"{n_fitted} of {n_total} parameters came from measured "
                    f"data; the remaining {n_total - n_fitted} are substituted "
                    "values, marked above.\n"
                )

            self.fitting_results_text.insert("end",
                headline +
                "  - Synapse calculations will now use these parameters\n"
                "  - Network initialization will use these parameters\n"
                "  - Click 'Run Fitting' to visualize the fit\n\n"
            )
            
            # Update wavelength slider range based on loaded data
            wl_range = base_params['wavelength_range']
            self.wavelength_slider.configure(from_=wl_range[0], to=wl_range[1])
            # Set to middle of range if current value is outside
            current_wl = self.wavelength_slider.get()
            if current_wl < wl_range[0] or current_wl > wl_range[1]:
                self.wavelength_slider.set((wl_range[0] + wl_range[1]) / 2)
                self.update_wavelength_label(self.wavelength_slider.get())
            
            # Update plots
            self.update_plots()
            self.update_info()
            
            # Update the fitting button text
            self.btn_fitting.configure(text="Run Fitting")
            
            return base_params
            
        except json.JSONDecodeError:
            tk_messagebox.showerror(
                "File Error",
                "Invalid JSON file. Please select a valid characterization suite or fitted model JSON file."
            )
            return None
        except Exception as e:
            tk_messagebox.showerror(
                "Load Error",
                f"Failed to load experimental model:\n{str(e)}"
            )
            self.fitting_results_text.insert("end", f"\n✗ Error loading model: {str(e)}\n\n")
            return None
    
    
    def setup_network_tab(self, parent):
        """Setup the 3×3 network simulation tab."""
        
        # Main container with two columns
        main_frame = ctk.CTkFrame(parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left column: Controls
        left_column = ctk.CTkFrame(main_frame, width=300)
        left_column.pack(side="left", fill="both", padx=(0, 10))
        left_column.pack_propagate(False)
        
        # Right column: Output
        right_column = ctk.CTkFrame(main_frame)
        right_column.pack(side="left", fill="both", expand=True)
        
        # === LEFT COLUMN: Controls ===
        
        # Title
        title_label = ctk.CTkLabel(
            left_column,
            text="NETWORK CONTROLS",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(10, 15))
        
        # Info section
        info_frame = ctk.CTkFrame(left_column)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            info_frame,
            text="3×3 Synapse Array",
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        info_text = (
            "Demonstrates emergent\n"
            "network behavior from\n"
            "device physics.\n\n"
            "• Device variability\n"
            "• Spatial patterns\n"
            "• Weight evolution"
        )
        
        ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=("Arial", 10),
            justify="left"
        ).pack(padx=10, pady=5)
        
        # Network mode selector
        mode_frame = ctk.CTkFrame(left_column)
        mode_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            mode_frame,
            text="NETWORK MODE",
            font=("Arial", 11, "bold")
        ).pack(pady=5)
        
        self.network_mode = ctk.StringVar(value="rate")
        
        ctk.CTkRadioButton(
            mode_frame,
            text="Rate-based (Spatial Patterns)",
            variable=self.network_mode,
            value="rate",
            command=self.on_network_mode_change
        ).pack(anchor="w", padx=10, pady=2)
        
        ctk.CTkRadioButton(
            mode_frame,
            text="Spiking (STDP Learning)",
            variable=self.network_mode,
            value="spiking",
            command=self.on_network_mode_change
        ).pack(anchor="w", padx=10, pady=2)

        action_frame = ctk.CTkFrame(left_column)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            action_frame,
            text="NETWORK ACTIONS",
            font=("Arial", 11, "bold")
        ).pack(pady=5)
        
        ctk.CTkButton(
            action_frame,
            text="Initialize Network",
            command=self.on_network_init,
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            height=35
        ).pack(fill="x", pady=3)
        
        ctk.CTkButton(
            action_frame,
            text="Rest (500ms)",
            command=self.on_network_rest,
            fg_color="#F57C00",
            hover_color="#E65100",
            height=30
        ).pack(fill="x", pady=3)
        
        ctk.CTkButton(
            action_frame,
            text="Show Evolution",
            command=self.on_network_show_evolution,
            fg_color="#7B1FA2",
            hover_color="#4A148C",
            height=30
        ).pack(fill="x", pady=3)
        
        ctk.CTkButton(
            action_frame,
            text="Reset Network",
            command=self.on_network_reset,
            fg_color="#C62828",
            hover_color="#8E0000",
            height=30
        ).pack(fill="x", pady=3)
        
        # Info label
        ctk.CTkLabel(
            action_frame,
            text="Use Synapse Controls (left panel)\nto apply patterns",
            font=("Arial", 9),
            text_color="gray"
        ).pack(pady=10)
        
        # === RIGHT COLUMN: Output ===
        
        # Title
        output_title = ctk.CTkLabel(
            right_column,
            text="NETWORK VISUALIZATION",
            font=("Arial", 14, "bold")
        )
        output_title.pack(pady=(0, 10))
        
        # Create matplotlib figure for embedding
        self.network_fig = Figure(figsize=(12, 5))
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, right_column)
        self.network_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initial placeholder
        ax = self.network_fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Initialize network to see visualization', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        self.network_canvas.draw()
        
        # Text output area below
        self.network_output_text = ctk.CTkTextbox(
            right_column,
            font=("Courier", 9),
            height=120
        )
        self.network_output_text.pack(fill="x", padx=5, pady=(5, 0))
        
        # Initial message
        self.network_output_text.insert("1.0",
            "=" * 60 + "\n"
            + "PHASE 3: SYNAPSE NETWORK SIMULATION (3×3 Array)\n"
            + "=" * 60 + "\n\n"
            + "Instructions:\n"
            "  1. Click 'Initialize Network' to create the synapse array\n"
            "  2. Select a spatial pattern from the dropdown\n"
            "  3. Click 'Apply Pattern' to stimulate the network\n"
            "  4. Click 'Show Evolution' to see the full history\n"
            "  5. Use 'Rest' to allow decay between patterns\n\n"
            "Experiment:\n"
            "  • Try different patterns and see how weights evolve\n"
            "  • Apply the same pattern multiple times (reinforcement)\n"
            "  • Mix different patterns to create complex weight maps\n"
            "  • Observe device variability effects\n\n"
            "Ready to start!\n"
        )
    
    
    def on_mode_change(self):
        """Update button text and show/hide stimulus type selector based on mode."""
        mode = self.mode_var.get()
        if mode == "potentiation":
            self.btn_stimulate.configure(text="Apply Potentiation", 
                                        fg_color="#2E7D32", hover_color="#1B5E20")
            # Hide stimulus type selector
            self.stimulus_type_frame.pack_forget()
            # Ensure wavelength controls are visible
            self.wavelength_slider.configure(state="normal")
            self.wavelength_label.configure(text_color=("gray10", "gray90"))
        else:
            self.btn_stimulate.configure(text="Apply Depression", 
                                        fg_color="#FF6F00", hover_color="#E65100")
            # Show stimulus type selector after radio buttons
            self.stimulus_type_frame.pack(fill="x", pady=(10,5), after=self.mode_radio_frame)
            # Update visibility based on current stimulus type
            self.on_stimulus_type_change()
    
    def update_intensity_label(self, value):
        """Update intensity label based on current stimulus type."""
        # Check if we're in depression mode with electrical stimulus
        if self.mode_var.get() == "depression" and self.stimulus_type_var.get() == "electrical":
            self.intensity_label.configure(text=f"{value:.1f} V")
        else:
            self.intensity_label.configure(text=f"{value:.1f} mW/cm²")
    
    def update_wavelength_label(self, value):
        """Update wavelength label and marker."""
        wavelength = int(value)
        
        # Color names for different ranges
        if wavelength < 380:
            color_name = "UV"
        elif wavelength < 450:
            color_name = "Violet"
        elif wavelength < 495:
            color_name = "Blue"
        elif wavelength < 570:
            color_name = "Green"
        elif wavelength < 590:
            color_name = "Yellow"
        elif wavelength < 620:
            color_name = "Orange"
        elif wavelength < 750:
            color_name = "Red"
        else:
            color_name = "IR"
        
        self.wavelength_label.configure(text=f"{wavelength} nm ({color_name})")
        
        # Update marker on spectrum plot
        self.wavelength_marker.set_xdata([wavelength, wavelength])
        self.canvas.draw_idle()
    
    def update_spectrum_curve(self):
        """Update the wavelength response curve plot with current synapse model."""
        # Use fitted model if available (for multi-Gaussian)
        if hasattr(self, 'loaded_fitted_model') and self.loaded_fitted_model is not None:
            fitted_model = self.loaded_fitted_model
            if 'wavelength_curve' in fitted_model and fitted_model['wavelength_curve'] is not None:
                # Use the SAME wavelength range as the fitting plot for consistency
                if 'wavelength_range' in fitted_model:
                    wl_min, wl_max = fitted_model['wavelength_range']
                    wavelengths = np.linspace(wl_min, wl_max, 200)
                else:
                    # Fallback to default range if not stored
                    wavelengths = np.linspace(200, 2000, 200)
                
                # M13: plot exactly what the PHYSICS uses.
                #
                # This used to evaluate the raw fitted curve and normalise it
                # by a single GLOBAL maximum, while `wavelength_sensitivity`
                # normalises PER BRANCH. Whenever the two lobes are unequal —
                # which is the usual case — the curve drawn to the user was not
                # the curve driving the simulation: the weaker branch appeared
                # smaller on screen than the value the ODE actually applied.
                #
                # Going through `wavelength_sensitivity` makes the plot and the
                # integration the same function by construction, so they cannot
                # drift apart again.
                sensitivities = np.array([
                    self.synapse.wavelength_sensitivity(w) for w in wavelengths
                ])
            else:
                # Fall back to synapse's wavelength_sensitivity method
                wl_range = self.synapse.wavelength_range
                wavelengths = np.linspace(wl_range[0], wl_range[1], 200)
                sensitivities = [self.synapse.wavelength_sensitivity(w) for w in wavelengths]
        else:
            # Use synapse's wavelength_sensitivity method
            wl_range = self.synapse.wavelength_range
            wavelengths = np.linspace(wl_range[0], wl_range[1], 200)
            sensitivities = [self.synapse.wavelength_sensitivity(w) for w in wavelengths]
        
        # Update the plot line
        self.spectrum_line.set_data(wavelengths, sensitivities)
        
        # Update x-axis limits to match wavelength range
        self.ax_spectrum.set_xlim([wavelengths[0], wavelengths[-1]])
        
        # Adjust y-limits to show negative values if present
        y_min = min(sensitivities)
        y_max = max(sensitivities)
        y_range = y_max - y_min
        self.ax_spectrum.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
        
        self.canvas.draw_idle()
    
    def update_duration_label(self, value):
        """Update duration label."""
        self.duration_label.configure(text=f"{int(value)} ms")
    
    def update_n_pulses_label(self, value):
        """Update number of pulses label."""
        n = int(value)
        self.n_pulses_label.configure(text=f"{n} pulse{'s' if n > 1 else ''}")
        
    
    def on_stimulus_type_change(self):
        """Handle stimulus type change between light and electrical."""
        stim_type = self.stimulus_type_var.get()
        
        if stim_type == "light":
            # Enable wavelength control
            self.wavelength_slider.configure(state="normal")
            self.wavelength_label.configure(text_color=("gray10", "gray90"))
            # Reconfigure slider for light intensity (0.1 to 100 mW/cm²)
            self.intensity_slider.configure(from_=0.1, to=100)
            self.intensity_slider.set(10)
            self.intensity_static_label.configure(text="Intensity (mW/cm²):")
            self.intensity_label.configure(text=f"{self.intensity_slider.get():.1f} mW/cm²")
        else:
            # Disable wavelength control (not used for electrical)
            self.wavelength_slider.configure(state="disabled")
            self.wavelength_label.configure(text_color="gray")
            # Reconfigure slider for depression voltage (-10V to +10V)
            self.intensity_slider.configure(from_=-10, to=10)
            self.intensity_slider.set(-5)
            self.intensity_static_label.configure(text="Depression Voltage (V):")
            # Change intensity label to voltage
            self.intensity_label.configure(text=f"{self.intensity_slider.get():.1f} V")
    
    def on_target_change(self):
        """Handle target mode change between single synapse and network."""
        target = self.target_var.get()
        
        if target == "single":
            # Hide network-specific controls
            self.pattern_selection_frame.pack_forget()
            self.network_size_frame.pack_forget()
            
            # Update button text
            mode = self.mode_var.get()
            if mode == "potentiation":
                self.btn_stimulate.configure(text="Apply Potentiation")
            else:
                self.btn_stimulate.configure(text="Apply Depression")
        
        else:  # network
            # Show network-specific controls
            self.network_size_frame.pack(fill="x", padx=10, pady=(5, 0))
            self.pattern_selection_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            # Update button text
            self.btn_stimulate.configure(text="Apply Pattern to Network")
    
    def on_stimulate(self, animate=True):
        """Apply stimulus to either single synapse or network based on target."""
        target = self.target_var.get()
        
        # Get unified parameters
        intensity = self.intensity_slider.get()
        wavelength = int(self.wavelength_slider.get())
        duration_ms = self.duration_slider.get()
        n_pulses = int(self.n_pulses_slider.get())
        mode = self.mode_var.get()
        
        # Get stimulus type (only relevant for depression)
        if mode == 'depression':
            stimulus_type = self.stimulus_type_var.get()
        else:
            stimulus_type = 'light'  # Potentiation is always light-based
        
        # Print verification of parameter source
        if self.using_experimental_params:
            print(f"[APPLYING {mode.upper()}] Using EXPERIMENTAL parameters from loaded JSON file")
            print(f"  → Synapse params: G_min={self.synapse.G_min:.2e} S, G_max={self.synapse.G_max:.2e} S, "
                  f"α={self.synapse.alpha:.2f}, β={self.synapse.beta:.2f}")
        else:
            print(f"[APPLYING {mode.upper()}] Using DEFAULT (synthetic) parameters")
            print(f"  → Synapse params: G_min={self.synapse.G_min:.2e} S, G_max={self.synapse.G_max:.2e} S, "
                  f"α={self.synapse.alpha:.2f}, β={self.synapse.beta:.2f}")
        
        if target == "single":
            # Apply to single synapse with rest periods and animation
            
            # Disable button during stimulation
            self.btn_stimulate.configure(state="disabled", text="Stimulating...")
            self.update()
            
            for i in range(n_pulses):
                # Apply pulse
                self.synapse.apply_stimulus(
                    intensity,         # arg 1: intensity_mW_cm2
                    wavelength,        # arg 2: wavelength_nm
                    duration_ms,       # arg 3: duration_ms
                    mode,              # arg 4: mode
                    stimulus_type      # arg 5: stimulus_type
                )
                
                # Rest period between pulses (shows decay!)
                rest_ms = self.rest_slider.get()
                if i < n_pulses - 1:
                    self.synapse.rest(rest_ms)
                
                # Update plot every few pulses for animation effect
                if animate and (i % 2 == 0 or i == n_pulses - 1):
                    self.update_plots()
                    self.update()
            
            # Final update
            self.update_plots()
            self.update_info()
            
            # Re-enable button
            self.on_mode_change()  # Restore proper text
            self.btn_stimulate.configure(state="normal")
            
        else:  # network
            # Auto-initialize network if not present
            if self.network is None:
                response = tk_messagebox.askyesno(
                    "Initialize Network?",
                    "No network detected. Would you like to initialize a 3×3 network now?\n\n"
                    "Default parameters will be used."
                )
                if response:
                    self.on_network_init()  # Initialize network
                else:
                    return  # User cancelled
            
            # Switch to network tab to see results
            self.notebook.set("Network")
            
            # Get selected pattern - now with dynamic shape
            pattern_name = self.pattern_var.get()
            from network import SpatialPatterns
            
            # Get current network shape
            network_shape = self.network.shape
            
            # Generate pattern with correct shape
            if pattern_name == "Vertical Bar (Left)":
                pattern = SpatialPatterns.vertical_bar(network_shape, 0)
            elif pattern_name == "Vertical Bar (Center)":
                pattern = SpatialPatterns.vertical_bar(network_shape, network_shape[1]//2)
            elif pattern_name == "Vertical Bar (Right)":
                pattern = SpatialPatterns.vertical_bar(network_shape, network_shape[1]-1)
            elif pattern_name == "Horizontal Bar (Top)":
                pattern = SpatialPatterns.horizontal_bar(network_shape, 0)
            elif pattern_name == "Horizontal Bar (Center)":
                pattern = SpatialPatterns.horizontal_bar(network_shape, network_shape[0]//2)
            elif pattern_name == "Horizontal Bar (Bottom)":
                pattern = SpatialPatterns.horizontal_bar(network_shape, network_shape[0]-1)
            elif pattern_name == "Diagonal (Main)":
                pattern = SpatialPatterns.diagonal_main(network_shape)
            elif pattern_name == "Diagonal (Anti)":
                pattern = SpatialPatterns.diagonal_anti(network_shape)
            elif pattern_name == "Cross":
                pattern = SpatialPatterns.cross(network_shape)
            elif pattern_name == "Corners":
                pattern = SpatialPatterns.corners(network_shape)
            elif pattern_name == "Center":
                pattern = SpatialPatterns.center(network_shape)
            elif pattern_name == "Uniform":
                pattern = SpatialPatterns.uniform(network_shape)
            elif pattern_name == "Checkerboard":
                pattern = SpatialPatterns.checkerboard(network_shape)
            elif pattern_name == "Left Half":
                pattern = SpatialPatterns.left_half(network_shape)
            elif pattern_name == "Right Half":
                pattern = SpatialPatterns.right_half(network_shape)
            elif pattern_name == "Top Half":
                # M18: the dropdown offered "Top Half" and the dispatcher had
                # no branch for it, so it fell through to the uniform fallback
                # while the dialog went on reporting the requested name — the
                # whole array was illuminated and labelled "Top Half".
                pattern = SpatialPatterns.top_half(network_shape)
            elif pattern_name == "Bottom Half":
                pattern = SpatialPatterns.bottom_half(network_shape)
            elif pattern_name == "Border":
                pattern = SpatialPatterns.border(network_shape)
            elif pattern_name == "Random":
                pattern = SpatialPatterns.random(network_shape)
            elif pattern_name == "Circle":
                pattern = SpatialPatterns.circle(network_shape)
            elif pattern_name == "Duck":
                pattern = SpatialPatterns.duck(network_shape)
            elif pattern_name == "Thumbs Up":
                pattern = SpatialPatterns.thumbs_up(network_shape)    
            elif pattern_name == "Raised Fist":
                pattern = SpatialPatterns.raised_fist(network_shape) 
            elif pattern_name == "Custom (from Excel)":  # <-- ADD THIS BLOCK
                if self.custom_pattern is not None:
                    pattern = ExcelPatternLoader.resize_pattern(
                        self.custom_pattern, 
                        network_shape
                    )
                else:
                    tk_messagebox.showerror(
                        "No Custom Pattern",
                        "Please load an Excel pattern first using the '📁 Load Excel Pattern' button."
                    )
                    self.btn_stimulate.configure(state="normal", text="Apply Pattern to Network")
                    return                

            else:
                # No silent substitution. A dropdown entry with no dispatcher
                # branch used to become a uniform pattern while the GUI kept
                # reporting the name the user chose, so the array was
                # illuminated everywhere and labelled something else entirely.
                tk_messagebox.showerror(
                    "Unknown Pattern",
                    f"'{pattern_name}' has no implementation in the pattern "
                    "dispatcher.\n\nThis is a bug: the dropdown offers a "
                    "pattern the code cannot build. Applying a different "
                    "pattern under this name would silently corrupt the "
                    "result, so nothing has been applied."
                )
                self.btn_stimulate.configure(state="normal",
                                             text="Apply Pattern to Network")
                return


            # Disable button during stimulation
            self.btn_stimulate.configure(state="disabled", text="Applying Pattern...")
            self.update()
            
            # Apply pattern n_pulses times with rest periods
            for i in range(n_pulses):
                self.network.apply_spatial_pattern(
                    pattern,           # arg 1: pattern
                    intensity,         # arg 2: intensity_mW_cm2
                    wavelength,        # arg 3: wavelength_nm
                    duration_ms,       # arg 4: duration_ms
                    mode,              # arg 5: mode
                    stimulus_type,     # arg 6: stimulus_type
                    pattern_name       # arg 7: pattern_name
                )
                
                # Rest period between pulses
                if i < n_pulses - 1:
                    rest_ms = self.rest_slider.get()
                    self.network.rest(rest_ms)  # Rest/decay
                
                # Update visualization every few pulses
                if i % 2 == 0 or i == n_pulses - 1:
                    self.update_network_plot()
                    self.update()
            
            # Final update
            self.update_network_plot()
            
            # Auto-show evolution window after pattern application
            if len(self.network.history['G_matrices']) >= 2:  # Only if we have evolution to show
                try:
                    self.on_network_show_evolution()
                except Exception as e:
                    print(f"Could not show evolution: {e}")
            
            # Re-enable button
            self.on_mode_change()
            self.btn_stimulate.configure(state="normal")
            
            tk_messagebox.showinfo(
                "Pattern Applied",
                f"Applied '{pattern_name}' pattern\n"
                f"Wavelength: {wavelength} nm\n"
                f"Intensity: {intensity:.1f} mW/cm²\n"
                f"Duration: {duration_ms:.0f} ms\n"
                f"Pulses: {n_pulses}\n"
                f"Mode: {mode}"
            )
    
    def on_reset(self):
        """Reset synapse to initial state."""
        self.synapse.reset()
        self.update_plots()
        self.update_info()
    
    def on_auto_demo(self):
        """Run automated demonstration sequence."""
        self.synapse.reset()
        
        # Disable controls
        self.btn_auto.configure(state="disabled", text="Running Demo...")
        self.update()
        
        # Demo sequence: Potentiation then Depression
        self.info_text.delete("1.0", "end")
        self.info_text.insert("end", "\n\nAuto Demo Running...\n")
        self.info_text.insert("end", "=" * 35 + "\n\n")
        
        # Part 1: Potentiation at different wavelengths
        self.info_text.insert("end", "PART 1: Potentiation\n")
        self.info_text.insert("end", "-" * 35 + "\n")
        
        demo_wavelengths = [
            (450, "Blue"),
            (550, "Green"),
            (650, "Red")
        ]
        
        for wavelength, color_name in demo_wavelengths:
            self.info_text.insert("end", f"Testing {color_name} ({wavelength} nm)...\n")
            self.wavelength_slider.set(wavelength)
            self.update_wavelength_label(wavelength)
            self.mode_var.set("potentiation")
            
            for i in range(5):
                self.synapse.apply_stimulus(20, wavelength, 100, 'potentiation')
                self.synapse.rest(100)
                self.update_plots()
                self.update()
            
            self.synapse.rest(300)
            self.update_plots()
            self.update()
        
        # Part 2: Depression
        self.info_text.insert("end", "\nPART 2: Depression\n")
        self.info_text.insert("end", "-" * 35 + "\n")
        self.info_text.insert("end", "Applying depression pulses...\n")
        
        self.mode_var.set("depression")
        self.wavelength_slider.set(550)
        self.update_wavelength_label(550)
        
        for i in range(10):
            self.synapse.apply_stimulus(20, 550, 100, 'depression')
            self.synapse.rest(100)
            self.update_plots()
            self.update()
        
        self.info_text.insert("end", "\nDemo Complete!\n")
        self.update_info()
        self.btn_auto.configure(state="normal", text="Auto Demo")
    
    
    def on_fitting_demo(self):
        """Run the model fitting demonstration or show experimental data fitting."""
        # Disable button
        if self.using_experimental_params:
            self.btn_fitting.configure(state="disabled", text="Running Fitting...")
        else:
            self.btn_fitting.configure(state="disabled", text="Running Fitting Demo...")
        self.update()
        
        # Clear results
        self.fitting_results_text.delete("1.0", "end")
        
        if self.using_experimental_params and self.loaded_experimental_data is not None:
            # Show fitting for experimental data
            self.fitting_results_text.insert("end", "="*60 + "\n")
            self.fitting_results_text.insert("end", "EXPERIMENTAL DATA FITTING VISUALIZATION\n")
            self.fitting_results_text.insert("end", "="*60 + "\n\n")
            
            self.fitting_results_text.insert("end", "Visualizing fitted parameters on experimental data...\n\n")
            self.update()
            
            # Check if we have a characterization suite (v1.0 or v2.0 format)
            data = self.loaded_experimental_data
            has_characterization = False
            
            # Check for v1.0 format
            if 'dynamic_range' in data or 'wavelength_response' in data or 'nonlinearity' in data:
                has_characterization = True
                characterization_suite = data
            # Check for v2.0 format
            elif 'datasets' in data:
                has_characterization = True
                characterization_suite = data
            
            if has_characterization:
                # We have characterization data - can show fitting visualization
                # Use full fitted_model if available, otherwise use base_params
                fitted_model = getattr(self, 'loaded_fitted_model', self.loaded_experimental_params)
                
                # Show peak override dialog if wavelength data exists
                if fitted_model.get('wavelength_n_peaks') is not None:
                    self.fitting_results_text.insert("end", 
                        f"Detected {fitted_model['wavelength_n_peaks']} wavelength peak(s).\n"
                        "Opening override dialog...\n\n"
                    )
                    self.update()
                    
                    # Show dialog
                    dialog = PeakOverrideDialog(self, fitted_model)
                    self.wait_window(dialog)
                    
                    # Check if user wants to override
                    if dialog.result is not None:
                        self.fitting_results_text.insert("end", 
                            f"Re-fitting with {dialog.result} peak(s)...\n"
                        )
                        self.update()
                        
                        # Re-fit with forced number of peaks
                        fitted_model = extract_synapse_model(characterization_suite, wavelength_n_peaks=dialog.result)
                        
                        # Update stored model
                        self.loaded_fitted_model = fitted_model
                        
                        self.fitting_results_text.insert("end", 
                            f"✓ Re-fitted with {dialog.result} peak(s).\n\n"
                        )
                    else:
                        self.fitting_results_text.insert("end", 
                            "Using automatic peak detection.\n\n"
                        )
                
                # Display parameters
                self.fitting_results_text.insert("end", "="*60 + "\n")
                self.fitting_results_text.insert("end", "FITTED PARAMETERS\n")
                self.fitting_results_text.insert("end", "="*60 + "\n\n")
                
                self.fitting_results_text.insert("end", 
                    f"  G_min:        {fitted_model['G_min']*1e6:.2f} µS\n"
                    f"  G_max:        {fitted_model['G_max']*1e6:.2f} µS\n"
                    f"  alpha:        {fitted_model['alpha']:.3f}\n"
                    f"  beta:         {fitted_model['beta']:.3f}\n"
                    f"  lambda_peak:  {fitted_model['lambda_peak']:.1f} nm\n"
                    f"  lambda_width: {fitted_model['lambda_width']:.1f} nm\n"
                    f"  decay_tau:    {fitted_model['decay_tau']:.1f} s\n\n"
                )
                
                self.fitting_results_text.insert("end", "Generating visualization plots...\n")
                self.update()
                
                # Generate plots (without ground truth comparison)
                fig = plot_fitting_results(characterization_suite, fitted_model, None)
                self.create_plot_window(fig, "Experimental Data Fitting Results")
                
                self.fitting_results_text.insert("end", "  ✓ Plots displayed in window\n\n")
                self.fitting_results_text.insert("end", "="*60 + "\n")
                self.fitting_results_text.insert("end", "✓ FITTING VISUALIZATION COMPLETE!\n")
                self.fitting_results_text.insert("end", "="*60 + "\n")
            else:
                # Just fitted parameters without characterization suite
                self.fitting_results_text.insert("end", 
                    "Note: Characterization suite data not available.\n"
                    "Only fitted parameters can be displayed.\n\n"
                    "To see fitting visualization, load a characterization suite JSON file.\n"
                )
        else:
            # Run synthetic fitting demo
            self.fitting_results_text.insert("end", "="*60 + "\n")
            self.fitting_results_text.insert("end", "PHASE 2: MODEL FITTING DEMONSTRATION\n")
            self.fitting_results_text.insert("end", "="*60 + "\n\n")
            
            # Use current synapse parameters as ground truth
            true_params = {
                'G_min': self.synapse.G_min,
                'G_max': self.synapse.G_max,
                'alpha': self.synapse.alpha,
                'beta': self.synapse.beta,
                'lambda_peak': self.synapse.lambda_peak,
                'lambda_width': self.synapse.lambda_width,
                'A_peak': self.synapse.A_peak,
                'B_peak': self.synapse.B_peak,
                'decay_tau': self.synapse.decay_tau
            }
            
            self.fitting_results_text.insert("end", "Step 1: Generating synthetic characterization data...\n")
            self.update()
            
            # Generate synthetic characterization data
            char_gen = SyntheticCharacterization(true_params)
            characterization_suite = {
                'wavelength_response': char_gen.measure_wavelength_response(),
                'nonlinearity': char_gen.measure_nonlinearity(),
                'dynamic_range': char_gen.measure_dynamic_range(),
                'retention': char_gen.measure_retention()
            }
            
            self.fitting_results_text.insert("end", "  ✓ Wavelength sweep complete\n")
            self.fitting_results_text.insert("end", "  ✓ Nonlinearity measurement complete\n")
            self.fitting_results_text.insert("end", "  ✓ Dynamic range measurement complete\n")
            self.fitting_results_text.insert("end", "  ✓ Retention measurement complete\n\n")
            self.update()
            
            self.fitting_results_text.insert("end", "Step 2: Fitting mathematical models...\n")
            self.update()
            
            # Fit models (automatic first)
            fitted_model = extract_synapse_model(characterization_suite)
            
            self.fitting_results_text.insert("end", "  ✓ Extracted synapse parameters from data\n\n")
            self.update()
            
            # Show peak override dialog if wavelength data exists
            if fitted_model.get('wavelength_n_peaks') is not None:
                self.fitting_results_text.insert("end", 
                    f"Detected {fitted_model['wavelength_n_peaks']} wavelength peak(s).\n"
                    "Opening override dialog...\n\n"
                )
                self.update()
                
                # Show dialog
                dialog = PeakOverrideDialog(self, fitted_model)
                self.wait_window(dialog)
                
                # Check if user wants to override
                if dialog.result is not None:
                    self.fitting_results_text.insert("end", 
                        f"Re-fitting with {dialog.result} peak(s)...\n"
                    )
                    self.update()
                    
                    # Re-fit with forced number of peaks
                    fitted_model = extract_synapse_model(characterization_suite, wavelength_n_peaks=dialog.result)
                    
                    self.fitting_results_text.insert("end", 
                        f"✓ Re-fitted with {dialog.result} peak(s).\n\n"
                    )
                else:
                    self.fitting_results_text.insert("end", 
                        "Using automatic peak detection.\n\n"
                    )
            
            self.fitting_results_text.insert("end", "Step 3: Comparing fitted vs. ground truth...\n\n")
            self.update()
            
            # Compare models
            comparison = compare_models(true_params, fitted_model)
            
            self.fitting_results_text.insert("end", "="*60 + "\n")
            self.fitting_results_text.insert("end", "FITTING RESULTS\n")
            self.fitting_results_text.insert("end", "="*60 + "\n\n")
            
            for param, values in comparison.items():
                self.fitting_results_text.insert("end", f"{param}:\n")
                self.fitting_results_text.insert("end", f"  Ground Truth:  {values['true']:.6g}\n")
                self.fitting_results_text.insert("end", f"  Fitted Value:  {values['fitted']:.6g}\n")
                self.fitting_results_text.insert("end", f"  Error:         {values['percent_error']:.2f}%\n\n")
            
            self.fitting_results_text.insert("end", "="*60 + "\n")
            self.fitting_results_text.insert("end", "\nStep 4: Generating visualization plots...\n")
            self.update()
            
            # Generate plots
            fig = plot_fitting_results(characterization_suite, fitted_model, true_params)
            self.create_plot_window(fig, "Model Fitting Results - Phase 2")
            
            self.fitting_results_text.insert("end", "  ✓ Plots displayed in custom window\n\n")
            self.fitting_results_text.insert("end", "="*60 + "\n")
            self.fitting_results_text.insert("end", "✓ FITTING DEMONSTRATION COMPLETE!\n")
            self.fitting_results_text.insert("end", "="*60 + "\n\n")
            self.fitting_results_text.insert("end", 
                "Interpretation:\n"
                "  • Errors < 5%: Excellent fit\n"
                "  • Errors 5-15%: Good fit (typical with noise)\n"
                "  • Errors > 15%: May need more data or refined model\n\n"
                "This validates the characterization → model pipeline!"
            )
        
        # Re-enable button
        if self.using_experimental_params:
            self.btn_fitting.configure(state="normal", text="Run Fitting")
        else:
            self.btn_fitting.configure(state="normal", text="Run Fitting Demo")
        
        # Switch to fitting tab to show results
        self.notebook.set("Model Fitting")
    
    def create_plot_window(self, fig, title="Plot Window"):
        """Create a separate tkinter window with the plot and save button."""
        plot_window = ctk.CTkToplevel(self)
        plot_window.title(title)
        plot_window.geometry("1200x900")
        
        # Button row is packed FIRST, against the bottom edge, so it always
        # gets its space. Packing the canvas first with expand=True let a
        # large figure (plot_network_evolution is 14x10 in, and the multi-panel
        # fitting figure is 6*ncols x 5*nrows) claim the whole window and push
        # the buttons off the TOP — measured at y = -187 in the default
        # 1200x900 geometry, so "Save Plot" was unreachable at every window
        # size and enlarging the window did not help.
        button_frame = ctk.CTkFrame(plot_window)
        button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Create frame for plot
        plot_frame = ctk.CTkFrame(plot_window)
        plot_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 0))

        # Embed matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        def save_plot():
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filepath:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {filepath}")
        
        save_btn = ctk.CTkButton(button_frame, text="Save Plot", 
                                command=save_plot, fg_color="#1565C0",
                                hover_color="#0D47A1", height=35, width=150)
        save_btn.pack(side="left", padx=5)
        
        close_btn = ctk.CTkButton(button_frame, text="Close Window",
                                 command=plot_window.destroy, fg_color="#C62828",
                                 hover_color="#8E0000", height=35, width=150)
        close_btn.pack(side="right", padx=5)
        
        return plot_window
    
    
    
    def on_network_mode_change(self):
        """Handle network mode change (rate vs spiking)."""
        mode = self.network_mode.get()
        if mode == "spiking":
            tk_messagebox.showinfo(
                "Spiking Network Mode",
                "Spiking mode requires spike trains as input.\n\n"
                "Use 'Generate Poisson Spikes' button to create input patterns.\n\n"
                "STDP learning will adjust synaptic weights based on spike timing."
            )
        # Reset network when mode changes
        if hasattr(self, 'network') and self.network is not None:
            self.network = None
            print(f"Network mode changed to: {mode}")
    
    def on_network_init(self):
        """Initialize the synapse network with user-specified size."""
        # Get network mode
        mode = self.network_mode.get()
        
        # Get network size from GUI
        try:
            rows = int(self.network_rows_entry.get())
            cols = int(self.network_cols_entry.get())
            
            # Validate size
            if rows < 1 or cols < 1:
                raise ValueError("Size must be at least 1×1")
            if rows > 50 or cols > 50:
                response = tk_messagebox.askyesno(
                    "Large Network Warning",
                    f"Creating a {rows}×{cols} network with {rows*cols} synapses.\n"
                    f"This may be slow to simulate and visualize.\n\n"
                    f"Continue?"
                )
                if not response:
                    return
            
            shape = (rows, cols)
        except ValueError as e:
            tk_messagebox.showerror(
                "Invalid Size",
                f"Please enter valid integers for network size.\n\n{e}"
            )
            return
        
        base_params = {
            'G_min': self.synapse.G_min,
            'G_max': self.synapse.G_max,
            'alpha': self.synapse.alpha,
            'beta': self.synapse.beta,
            'A_peak': self.synapse.A_peak,
            'B_peak': self.synapse.B_peak,
            'lambda_peak': self.synapse.lambda_peak,
            'lambda_width': self.synapse.lambda_width,
            'decay_tau': self.synapse.decay_tau,
            'wavelength_curve': self.synapse.wavelength_curve,
            'wavelength_range': self.synapse.wavelength_range
        }
        
        # Print what parameters are being used
        if self.using_experimental_params:
            print(f"[NETWORK INIT] Using EXPERIMENTAL parameters from loaded JSON file")
            print(f"  → Base params: G_min={base_params['G_min']:.2e} S, G_max={base_params['G_max']:.2e} S, "
                  f"α={base_params['alpha']:.2f}, β={base_params['beta']:.2f}")
        else:
            print(f"[NETWORK INIT] Using DEFAULT (synthetic) parameters")
            print(f"  → Base params: G_min={base_params['G_min']:.2e} S, G_max={base_params['G_max']:.2e} S, "
                  f"α={base_params['alpha']:.2f}, β={base_params['beta']:.2f}")

        # Get variability from GUI
        try:
            variability_percent = float(self.variability_entry.get())
            variability = variability_percent / 100.0
            if variability < 0 or variability > 1:
                raise ValueError("Variability must be between 0 and 100%")
        except ValueError as e:
            tk_messagebox.showerror(
                "Invalid Variability",
                f"Please enter a valid variability percentage (0-100).\n\n{e}"
            )
            return
        
        # Create network based on mode
        if mode == "rate":
            # Rate-based network (existing)
            self.network = SynapseNetwork(base_params, shape=shape, variability=variability)
            print(f"✓ Created rate-based network ({rows}×{cols})")
        else:
            # Spiking network (new)
            from network import SpikingSynapseNetwork
            
            # Get STDP parameters from loaded model or use defaults
            stdp_params = {}
            if self.loaded_fitted_model and hasattr(self, 'loaded_fitted_model'):
                stdp_params = {
                    'stdp_A_plus': self.loaded_fitted_model.get('stdp_A_plus', 0.5),
                    'stdp_A_minus': self.loaded_fitted_model.get('stdp_A_minus', 0.3),
                    'stdp_tau_plus_ms': self.loaded_fitted_model.get('stdp_tau_plus_ms', 20.0),
                    'stdp_tau_minus_ms': self.loaded_fitted_model.get('stdp_tau_minus_ms', 20.0)
                }
            else:
                stdp_params = {
                    'stdp_A_plus': 0.5,
                    'stdp_A_minus': 0.3,
                    'stdp_tau_plus_ms': 20.0,
                    'stdp_tau_minus_ms': 20.0
                }
            
            n_input = rows
            n_hidden = cols
            # The GUI's variability control applies here too — the spiking
            # constructor used to ignore it and hardcode 0.1.
            spiking_params = dict(base_params)
            spiking_params['variability'] = variability
            self.network = SpikingSynapseNetwork(
                base_synapse_params=spiking_params,
                stdp_params=stdp_params,
                n_input=n_input,
                n_hidden=n_hidden,
                dt_ms=1.0
            )
            print(f"✓ Created spiking network ({n_input} input → {n_hidden} hidden neurons)")
            print(f"  STDP: A+={stdp_params['stdp_A_plus']:.2f}, A-={stdp_params['stdp_A_minus']:.2f}, "
                  f"τ+={stdp_params['stdp_tau_plus_ms']:.1f}ms")

        
        # Create network with specified shape
        self.network_output_text.delete("1.0", "end")
        self.network_output_text.insert("end", "="*60 + "\n")
        self.network_output_text.insert("end", "NETWORK INITIALIZED\n")
        self.network_output_text.insert("end", "="*60 + "\n\n")
        # Say which topology was actually built. The Rows/Cols widgets mean
        # different things in the two modes — a spatial grid in rate mode, and
        # input/hidden layer sizes in spiking mode — and reporting "10×10" for
        # both read as a 10×10 spatial array in either case.
        if mode == "rate":
            self.network_output_text.insert(
                "end", f"{rows}×{cols} spatial synapse array created\n")
        else:
            self.network_output_text.insert(
                "end",
                f"Spiking network created: {rows} input neurons → {cols} hidden "
                f"neurons (fully connected)\n")
        self.network_output_text.insert("end", f"Total synapses: {self.network.n_synapses}\n")
        self.network_output_text.insert("end", f"Device variability: {variability*100:.1f}%\n\n")
        
        stats = self.network.get_statistics()
        self.network_output_text.insert("end", f"Initial Statistics:\n")
        self.network_output_text.insert("end", f"  Mean G: {stats['mean_G']*1e6:.2f} µS\n")
        self.network_output_text.insert("end", f"  Std G:  {stats['std_G']*1e6:.2f} µS\n")
        self.network_output_text.insert("end", f"  Min G:  {stats['min_G']*1e6:.2f} µS\n")
        self.network_output_text.insert("end", f"  Max G:  {stats['max_G']*1e6:.2f} µS\n\n")
        self.network_output_text.insert("end", "Ready to apply patterns!\n")
        
        # Update tab title to show current size
        # (Note: This requires recreating the tab - optional enhancement)
        
        # Show initial state in embedded plot
        self.update_network_plot()
        
    def on_network_rest(self):
        """Apply rest period to network."""
        if self.network is None:
            self.network_output_text.insert("end", "\n⚠ Please initialize network first!\n")
            return
        
        self.network.rest(500)
        self.network_output_text.insert("end", "\n→ Rest period (500ms) applied\n")
        
        stats = self.network.get_statistics()
        self.network_output_text.insert("end", f"  Mean G: {stats['mean_G']*1e6:.2f} µS\n")
    
    def on_network_show_evolution(self):
        """Show network evolution over time."""
        if self.network is None:
            self.network_output_text.insert("end", "\n⚠ Please initialize network first!\n")
            return
        
        self.network_output_text.insert("end", f"\n{'='*60}\n")
        self.network_output_text.insert("end", "Displaying network evolution...\n")
        self.network_output_text.insert("end", f"Total states recorded: {len(self.network.history['G_matrices'])}\n")
        
        fig = plot_network_evolution(self.network)
        self.create_plot_window(fig, "Network Evolution Over Time")
        
        self.network_output_text.insert("end", "✓ Evolution plot displayed in custom window!\n")
    
    def on_network_reset(self):
        """Reset the network."""
        if self.network is None:
            self.network_output_text.insert("end", "\n⚠ Please initialize network first!\n")
            return
        
        self.network.reset()
        self.network_output_text.insert("end", f"\n{'='*60}\n")
        self.network_output_text.insert("end", "Network reset to initial state\n")
        
        # Show reset state in embedded plot
        self.update_network_plot()
    
    def on_network_demo(self):
        """Run automated network demonstration."""
        self.btn_network.configure(state="disabled", text="Running Demo...")
        self.update()
        
        # Switch to network tab
        # M17: the tab is created as "Network", not "Network (3×3)". This was
        # the demo handler's FIRST statement, so it raised immediately and left
        # the button permanently disabled reading "Running Demo...".
        self.notebook.set("Network")
        
        self.network_output_text.delete("1.0", "end")
        self.network_output_text.insert("end", "="*60 + "\n")
        self.network_output_text.insert("end", "AUTOMATED NETWORK DEMONSTRATION\n")
        self.network_output_text.insert("end", "="*60 + "\n\n")
        
        # Initialize network
        self.network_output_text.insert("end", "Step 1: Initializing network...\n")
        self.on_network_init()
        self.update()
        
        # Demo sequence
        demo_patterns = [
            ('Vertical Bar (Center)', 5),
            ('Horizontal Bar (Center)', 5),
            ('Diagonal (Main)', 5),
            ('Cross', 5),
        ]
        
        self.network_output_text.insert("end", "\nStep 2: Applying pattern sequence...\n")
        
        for pattern_name, n_pulses in demo_patterns:
            self.pattern_var.set(pattern_name)
            self.n_pulses_slider.set(n_pulses)
            self.network_output_text.insert("end", f"\n→ Applying {pattern_name}...\n")
            self.update()
            
            # Generate pattern with network shape
            network_shape = self.network.shape
            if pattern_name == "Vertical Bar (Center)":
                pattern = SpatialPatterns.vertical_bar(network_shape, network_shape[1]//2)
            elif pattern_name == "Horizontal Bar (Center)":
                pattern = SpatialPatterns.horizontal_bar(network_shape, network_shape[0]//2)
            elif pattern_name == "Diagonal (Main)":
                pattern = SpatialPatterns.diagonal_main(network_shape)
            elif pattern_name == "Cross":
                pattern = SpatialPatterns.cross(network_shape)
            else:
                pattern = SpatialPatterns.uniform(network_shape)
            for i in range(n_pulses):
                self.network.apply_spatial_pattern(
                    pattern, 20, 550, 100, 'potentiation', pattern_name
                )
                self.network.rest(100)
            
            self.network_output_text.insert("end", f"  ✓ Complete\n")
            self.update()
        
        self.network_output_text.insert("end", "\nStep 3: Showing evolution...\n")
        self.on_network_show_evolution()
        
        self.network_output_text.insert("end", "\n" + "="*60 + "\n")
        self.network_output_text.insert("end", "✓ NETWORK DEMO COMPLETE!\n")
        self.network_output_text.insert("end", "="*60 + "\n\n")
        
        stats = self.network.get_statistics()
        self.network_output_text.insert("end", f"Final Network State:\n")
        self.network_output_text.insert("end", f"  Mean G: {stats['mean_G']*1e6:.2f} µS\n")
        self.network_output_text.insert("end", f"  Range: {stats['min_G']*1e6:.2f} - {stats['max_G']*1e6:.2f} µS\n")
        self.network_output_text.insert("end", f"  Total patterns applied: {len(demo_patterns)}\n")
        self.network_output_text.insert("end", "\nObserve how different patterns create different weight distributions!\n")
        
        self.btn_network.configure(state="normal", text="Run Network Demo")
    
    
    
    def update_network_plot(self):
        """Update the embedded network visualization plot with improved layout."""
        if self.network is None:
            return
        
        # Clear previous plot
        self.network_fig.clear()
        
        # Get current state
        G_matrix_uS = self.network.G_matrix * 1e6
        last_pattern = self.network.history['patterns'][-1]
        pattern_name = self.network.history['pattern_names'][-1]
        
        # Create subplots with better spacing - 2 plots instead of 3
        gs = self.network_fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax1 = self.network_fig.add_subplot(gs[0])
        ax2 = self.network_fig.add_subplot(gs[1])
        
        # 1. Conductance Heatmap (larger)
        im1 = ax1.imshow(G_matrix_uS, cmap='viridis', aspect='auto', interpolation='nearest')
        ax1.set_title('Conductance State (µS)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Column', fontsize=10)
        ax1.set_ylabel('Row', fontsize=10)
        
        cbar1 = self.network_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('G (µS)', fontsize=9)
        
        # 2. Last Applied Pattern with statistics overlay
        # Don't show Rest pattern (all zeros) - show previous non-zero pattern instead
        display_pattern = last_pattern
        display_name = pattern_name
        
        if pattern_name == 'Rest' and len(self.network.history['patterns']) > 1:
            # Find last non-rest pattern
            for i in range(len(self.network.history['patterns']) - 2, -1, -1):
                if self.network.history['pattern_names'][i] != 'Rest':
                    display_pattern = self.network.history['patterns'][i]
                    display_name = self.network.history['pattern_names'][i] + ' (last)'
                    break
        
        im2 = ax2.imshow(display_pattern, cmap='RdYlGn', aspect='auto', 
                        interpolation='nearest', vmin=0, vmax=1)
        
        # Add pattern name and statistics as title
        stats = self.network.get_statistics()
        title_text = (f'Last Pattern: {display_name}\n'
                     f'Mean: {stats["mean_G"]*1e6:.1f}µS | '
                     f'Range: {stats["min_G"]*1e6:.1f}-{stats["max_G"]*1e6:.1f}µS')
        ax2.set_title(title_text, fontsize=10, fontweight='bold')
        ax2.set_xlabel('Column', fontsize=10)
        ax2.set_ylabel('Row', fontsize=10)
        
        cbar2 = self.network_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Intensity', fontsize=9)
        
        # Add compact statistics footer
        footer_text = (f'Time: {self.network.current_time:.1f}s | '
                      f'Total: {stats["total_weight"]*1e6:.1f}µS | '
                      f'Std: {stats["std_G"]*1e6:.2f}µS | '
                      f'States: {len(self.network.history["G_matrices"])}')
        
        self.network_fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.network_fig.tight_layout(rect=[0, 0.05, 1, 1])
        self.network_canvas.draw()
    
    
    def update_plots(self):
        """Update all plots with proper scaling."""
        # Convert conductance to microsiemens for better readability
        G_microsiemens = [g * 1e6 for g in self.synapse.history_G]
        
        # Update main conductance plot
        self.line_G.set_data(self.synapse.history_t, G_microsiemens)
        self.line_stim.set_data(self.synapse.history_t, self.synapse.history_stimulus)
        
        # Auto-scale axes
        if len(self.synapse.history_t) > 1:
            self.ax_main.set_xlim([0, max(self.synapse.history_t) * 1.05])
            
            # Set y-axis limits for conductance with proper margins
            G_min_plot = min(G_microsiemens)
            G_max_plot = max(G_microsiemens)
            G_range = G_max_plot - G_min_plot
            margin = max(0.1 * G_range, 0.1)  # At least 0.1 µS margin
            
            self.ax_main.set_ylim([G_min_plot - margin, G_max_plot + margin])
            
            # Set y-axis limits for stimulus
            max_stim = max([abs(s) for s in self.synapse.history_stimulus] + [10])
            self.ax_stim.set_ylim([-max_stim * 1.2, max_stim * 1.2])
        
        # Update wavelength response curve (for experimental data)
        self.update_spectrum_curve()
        
        self.canvas.draw_idle()
            
    
    def update_info(self):
        """Update info panel with current state."""
        G_current = self.synapse.G
        G_percent = (G_current - self.synapse.G_min) / (self.synapse.G_max - self.synapse.G_min) * 100
        I_read = self.synapse.read_current(0.1)
        n_pulses = len([s for s in self.synapse.history_stimulus if s != 0])
        
        info = f"""
╔════════════════════════════════╗
║     SYNAPSE STATE              ║
╚════════════════════════════════╝

Current Conductance:
  G = {G_current*1e6:.2f} µS
  G = {G_current:.2e} S
  
Relative State:
  {G_percent:.1f}% of dynamic range
  
Read Current @ 0.1V:
  I = {I_read*1e6:.2f} µA
  I = {I_read:.2e} A
  
Physical Parameters:
  G_min = {self.synapse.G_min*1e6:.2f} µS
  G_max = {self.synapse.G_max*1e6:.2f} µS
  α (pot. nonlin.) = {self.synapse.alpha:.2f}
  β (dep. nonlin.) = {self.synapse.beta:.2f}
  λ_peak = {self.synapse.lambda_peak} nm
  τ_decay = {self.synapse.decay_tau} s

Total pulses applied: {n_pulses}
        """

        # Where these parameters came from. Without this the panel showed a
        # fitted device and a library default identically, and the only signal
        # was a print() to a shared console.
        if self.using_experimental_params:
            src = getattr(self, 'loaded_model_source', None) or 'loaded model'
            prov = getattr(self, 'loaded_param_provenance', None) or {}
            n_fit = sum(1 for v in prov.values() if v == 'fitted')
            info += (f"\nParameters: EXPERIMENTAL — {src}\n"
                     + (f"  {n_fit}/{len(prov)} fitted from your data\n" if prov else ""))
        else:
            info += ("\nParameters: DEFAULT (synthetic)\n"
                     "  Not measured from a device. Load a characterisation\n"
                     "  suite in the Model Fitting tab to use your own.\n")

        # The branch the device physics ACTUALLY took, whenever it differed
        # from what was requested. VisualSynapse records this on every
        # stimulus and nothing outside the test suite ever read it — so a
        # wavelength whose fitted spectral response inverts the requested mode
        # silently potentiated while the GUI still said "Depression".
        rpt = getattr(self.synapse, 'last_stimulus_report', None)
        if rpt and rpt.get('requested_mode') != rpt.get('actual_mode'):
            info += (f"\n! BRANCH OVERRIDE\n"
                     f"  requested: {rpt.get('requested_mode')}\n"
                     f"  actual:    {rpt.get('actual_mode')}\n"
                     f"  reason: {rpt.get('reason', '')}\n")
        elif rpt and rpt.get('unsigned_fallback'):
            info += ("\n! No fitted spectral response is loaded, so the\n"
                     "  wavelength carries magnitude only and the requested\n"
                     "  mode selects the branch.\n")

        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", info)
    
    def setup_snn_tab(self, parent):
        """Setup SNN simulation tab with controls and visualization."""
        # Left side: Controls
        controls_frame = ctk.CTkFrame(parent, width=300)
        controls_frame.pack(side="left", fill="y", padx=10, pady=10)
        controls_frame.pack_propagate(False)
        
        ctk.CTkLabel(controls_frame, text="SNN Simulation", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Network size.
        #
        # The default MUST exceed 1/SINGLE_SYNAPSE_DV_FRACTION (= 10 at the
        # documented 0.10), or the shipped configuration cannot fire a neuron
        # from a coincident volley and every first run produces zero hidden
        # spikes. It used to default to 8 against a threshold of 10. Derived
        # from the constant rather than restated, so the two cannot drift.
        _snn_default_inputs = max(32, int(np.ceil(1.0 / SINGLE_SYNAPSE_DV_FRACTION)) * 2)
        ctk.CTkLabel(controls_frame, text="Input Neurons:").pack(pady=(10,0))
        self.snn_n_input = ctk.CTkEntry(controls_frame, width=100)
        self.snn_n_input.insert(0, str(_snn_default_inputs))
        self.snn_n_input.pack()
        
        ctk.CTkLabel(controls_frame, text="Hidden Neurons:").pack(pady=(10,0))
        self.snn_n_hidden = ctk.CTkEntry(controls_frame, width=100)
        self.snn_n_hidden.insert(0, "4")
        self.snn_n_hidden.pack()
        
        # Poisson parameters
        ctk.CTkLabel(controls_frame, text="Input Firing Rate (Hz):").pack(pady=(10,0))
        self.snn_rate = ctk.CTkEntry(controls_frame, width=100)
        self.snn_rate.insert(0, "15")
        self.snn_rate.pack()
        
        ctk.CTkLabel(controls_frame, text="Duration (ms):").pack(pady=(10,0))
        self.snn_duration = ctk.CTkEntry(controls_frame, width=100)
        self.snn_duration.insert(0, "1000")
        self.snn_duration.pack()
        
        # STDP learning toggle
        self.snn_learning_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(controls_frame, text="Enable STDP Learning",
                       variable=self.snn_learning_var).pack(pady=10)
        
        # Buttons
        self.btn_snn_run = ctk.CTkButton(controls_frame, text="Run SNN Simulation",
                                        command=self.on_snn_run)
        self.btn_snn_run.pack(pady=10)
        
        self.btn_snn_reset = ctk.CTkButton(controls_frame, text="Reset SNN",
                                          command=self.on_snn_reset)
        self.btn_snn_reset.pack(pady=5)
        
        # Output text
        ctk.CTkLabel(controls_frame, text="Output:").pack(pady=(10,0))
        self.snn_output_text = ctk.CTkTextbox(controls_frame, height=200)
        self.snn_output_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right side: Visualization
        viz_frame = ctk.CTkFrame(parent)
        viz_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.snn_fig = Figure(figsize=(10, 8))
        self.snn_canvas = FigureCanvasTkAgg(self.snn_fig, viz_frame)
        self.snn_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def on_snn_run(self):
        """Run SNN simulation with Poisson inputs."""
        self.btn_snn_run.configure(state="disabled", text="Running...")
        self.snn_output_text.delete("1.0", "end")
        self.snn_output_text.insert("end", "Starting SNN simulation...\n")
        
        try:
            n_input = int(self.snn_n_input.get())
            n_hidden = int(self.snn_n_hidden.get())
            rate_hz = float(self.snn_rate.get())
            duration_ms = float(self.snn_duration.get())
            learning_enabled = self.snn_learning_var.get()

            # Device variability, validated the same way the Network tab does.
            snn_variability = float(self.variability_entry.get()) / 100.0
            if not (0.0 <= snn_variability <= 1.0):
                raise ValueError(
                    "Device Variability must be between 0 and 100 %, got "
                    f"{self.variability_entry.get()!r}."
                )

            # Can this configuration fire a neuron at all?
            #
            # SpikingSynapseNetwork scales its synaptic gain so ONE mid-range
            # synapse depolarises a resting neuron by
            # SINGLE_SYNAPSE_DV_FRACTION of the threshold gap per timestep
            # (network.py). With the default 0.10 that needs ~10 coincident
            # mid-range inputs, so an n_input below that cannot reach threshold
            # from a single volley however well timed it is -- and because the
            # devices also relax toward G_min between spikes, the network gets
            # FURTHER from firing the longer it runs. That combination used to
            # produce a silent "0 hidden spikes" with no explanation.
            min_inputs_to_fire = int(np.ceil(1.0 / SINGLE_SYNAPSE_DV_FRACTION))
            if n_input < min_inputs_to_fire:
                self.snn_output_text.insert(
                    "end",
                    "\n! WARNING: this configuration cannot fire a neuron from a\n"
                    f"  coincident volley. The synaptic gain is set so that one\n"
                    f"  mid-range synapse contributes "
                    f"{SINGLE_SYNAPSE_DV_FRACTION * 100:.0f}% of the threshold\n"
                    f"  gap, so at least {min_inputs_to_fire} simultaneous inputs are\n"
                    f"  needed; you have {n_input}. Even all {n_input} firing together\n"
                    f"  reaches only {n_input * SINGLE_SYNAPSE_DV_FRACTION * 100:.0f}% of threshold.\n"
                    f"  Expect few or no hidden spikes, and no STDP learning.\n"
                    f"  Fix: raise 'Input Neurons' to {min_inputs_to_fire} or more, or raise the\n"
                    "  firing rate so inputs summate within the membrane time constant.\n\n"
                )


            # Get synapse parameters
            # A_plus/A_minus are PERCENTAGES of the dynamic range: the learning
            # rule in network.py carries the `* 0.01` itself (CLAUDE.md), so a
            # fitted 0.4517 means 0.4517% and must be passed through UNCHANGED.
            # This site used to divide by 100, making every weight update 100x
            # too small while the simulation still reported success.
            if (self.loaded_fitted_model
                    and self.loaded_experimental_data
                    and 'stdp' in self.loaded_experimental_data):
                stdp_data = self.loaded_experimental_data['stdp']
                from fitting import fit_stdp_window
                fitted_stdp = fit_stdp_window(stdp_data)

                stdp_params = {
                    'stdp_A_plus': fitted_stdp['A_plus'],
                    'stdp_A_minus': fitted_stdp['A_minus'],
                    'stdp_tau_plus_ms': fitted_stdp['tau_plus_ms'],
                    'stdp_tau_minus_ms': fitted_stdp['tau_minus_ms']
                }
                self.snn_output_text.insert(
                    "end",
                    "Using fitted STDP parameters "
                    f"(A+={fitted_stdp['A_plus']:.3f}%, "
                    f"A-={fitted_stdp['A_minus']:.3f}% of dynamic range, "
                    f"tau+={fitted_stdp['tau_plus_ms']:.1f} ms, "
                    f"tau-={fitted_stdp['tau_minus_ms']:.1f} ms)\n"
                )
            else:
                # The documented defaults from CLAUDE.md. This used to be
                # 0.15/0.09 - undocumented values 3.3x weaker than spec,
                # labelled to the user as "default".
                stdp_params = {
                    'stdp_A_plus': 0.5,
                    'stdp_A_minus': 0.3,
                    'stdp_tau_plus_ms': 20.0,
                    'stdp_tau_minus_ms': 20.0
                }
                reason = ("no fitted model is loaded"
                          if not self.loaded_fitted_model else
                          "the loaded model carries no STDP measurement")
                self.snn_output_text.insert(
                    "end",
                    f"Using DEFAULT STDP parameters ({reason}): "
                    "A+=0.5%, A-=0.3% of dynamic range, tau=20 ms.\n"
                    "These are library defaults, NOT measured from your device.\n"
                )
            
            base_params = {
                'G_min': self.synapse.G_min,
                'G_max': self.synapse.G_max,
                'alpha': self.synapse.alpha,
                'beta': self.synapse.beta,
                'lambda_peak': self.synapse.lambda_peak,
                'lambda_width': self.synapse.lambda_width,
                'decay_tau': self.synapse.decay_tau,
                'A_peak': self.synapse.A_peak,
                'B_peak': self.synapse.B_peak,
                # Honour the GUI's Device Variability control. Omitting it made
                # network.py fall back to a hardcoded 0.1, so the SNN always ran
                # at 10% while the GUI displayed whatever the user had typed.
                'variability': snn_variability,
            }

            # Create SNN
            self.snn_output_text.insert("end", f"\nCreating {n_input}→{n_hidden} SNN...\n")
            self.snn = SpikingSynapseNetwork(
                base_synapse_params=base_params,
                stdp_params=stdp_params,
                n_input=n_input,
                n_hidden=n_hidden,
                dt_ms=1.0
            )
            
            # Generate Poisson inputs
            self.snn_output_text.insert("end", f"Generating Poisson spikes at {rate_hz} Hz...\n")
            self.snn_input_spikes = generate_poisson_spike_trains(
                n_neurons=n_input,
                rates_hz=rate_hz,
                duration_ms=duration_ms,
                seed=42
            )
            
            total_input = sum(len(s) for s in self.snn_input_spikes)
            self.snn_output_text.insert("end", f"Generated {total_input} input spikes\n")
            
            # Run simulation
            self.snn_output_text.insert("end", f"\nRunning simulation ({duration_ms} ms)...\n")
            self.snn_results = self.snn.simulate(
                input_spike_trains=self.snn_input_spikes,
                duration_ms=duration_ms,
                learning_enabled=learning_enabled
            )
            
            total_output = sum(len(s) for s in self.snn_results['hidden_spikes'])
            self.snn_output_text.insert("end", f"Hidden layer: {total_output} spikes\n")
            
            # Visualize
            self.snn_output_text.insert("end", "\nCreating visualization...\n")
            self.update_snn_plot()
            
            self.snn_output_text.insert("end", "\n✓ Simulation complete!\n")
            
        except Exception as e:
            self.snn_output_text.insert("end", f"\n✗ Error: {str(e)}\n")
        finally:
            self.btn_snn_run.configure(state="normal", text="Run SNN Simulation")
    
    def on_snn_reset(self):
        """Reset SNN simulation."""
        self.snn = None
        self.snn_results = None
        self.snn_input_spikes = None
        self.snn_fig.clear()
        self.snn_canvas.draw()
        self.snn_output_text.delete("1.0", "end")
        self.snn_output_text.insert("end", "SNN reset.\n")
    
    def update_snn_plot(self):
        """Update SNN visualization plot."""
        if self.snn_results is None:
            return
        
        self.snn_fig.clear()
        
        duration_ms = self.snn_results['time_points'][-1]
        n_input = len(self.snn_input_spikes)
        n_hidden = len(self.snn_results['hidden_spikes'])
        
        # Create subplots
        gs = self.snn_fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Input raster
        ax1 = self.snn_fig.add_subplot(gs[0, :])
        for i, spikes in enumerate(self.snn_input_spikes):
            if spikes:
                ax1.scatter(spikes, [i]*len(spikes), marker='|', s=100, c='blue', linewidths=2)
        ax1.set_ylabel('Input Neuron')
        ax1.set_title('Input Layer Spikes')
        ax1.set_xlim([0, duration_ms])
        ax1.set_ylim([-0.5, n_input-0.5])
        ax1.grid(True, alpha=0.3)
        
        # Hidden raster
        ax2 = self.snn_fig.add_subplot(gs[1, :])
        for i, spikes in enumerate(self.snn_results['hidden_spikes']):
            if spikes:
                ax2.scatter(spikes, [i]*len(spikes), marker='|', s=100, c='red', linewidths=2)
        ax2.set_ylabel('Hidden Neuron')
        ax2.set_xlabel('Time (ms)')
        ax2.set_title('Hidden Layer Spikes')
        ax2.set_xlim([0, duration_ms])
        ax2.set_ylim([-0.5, n_hidden-0.5])
        ax2.grid(True, alpha=0.3)
        
        # Initial weights
        ax3 = self.snn_fig.add_subplot(gs[2, 0])
        W0 = self.snn_results['weight_history'][0] * 1e6
        im1 = ax3.imshow(W0, cmap='RdBu_r', aspect='auto')
        ax3.set_title('Initial Weights (µS)')
        ax3.set_xlabel('Hidden')
        ax3.set_ylabel('Input')
        self.snn_fig.colorbar(im1, ax=ax3)
        
        # Final weights
        ax4 = self.snn_fig.add_subplot(gs[2, 1])
        Wf = self.snn_results['weight_history'][-1] * 1e6
        im2 = ax4.imshow(Wf, cmap='RdBu_r', aspect='auto')
        ax4.set_title('Final Weights (µS)')
        ax4.set_xlabel('Hidden')
        ax4.set_ylabel('Input')
        self.snn_fig.colorbar(im2, ax=ax4)
        
        self.snn_canvas.draw()

    # Buttons that disable themselves for the duration of a long operation,
    # with the label to restore. Each handler re-enables its own button on the
    # normal path; this is the recovery path for when one raises instead.
    _ACTION_BUTTONS = (
        ('btn_stimulate', 'Apply Stimulus'),
        ('btn_auto', 'Auto Demo'),
        ('btn_fitting', 'Run Fitting Demo'),
        ('btn_network', 'Run Network Demo'),
        ('btn_snn_run', 'Run SNN Simulation'),
    )

    def _restore_action_buttons(self):
        """Re-enable every long-operation button after an unhandled error.

        Without this, an exception inside a handler left its button greyed and
        reading "Running Demo..." forever, with the only recovery being to
        restart the application. Documented in-repo as having already happened
        once (a wrong tab name raised on a handler's first statement); the tab
        name was fixed but the structural cause was not.
        """
        for attr, label in self._ACTION_BUTTONS:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            try:
                widget.configure(state="normal", text=label)
            except Exception:
                pass

    def on_closing(self):
        """Properly cleanup matplotlib figures and close application."""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Destroy canvases explicitly
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
            if hasattr(self, 'fitting_canvas'):
                self.fitting_canvas.get_tk_widget().destroy()
            if hasattr(self, 'network_canvas'):
                self.network_canvas.get_tk_widget().destroy()
            if hasattr(self, 'snn_canvas'):
                self.snn_canvas.get_tk_widget().destroy()
            
            # Clear figure references
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            if hasattr(self, 'fitting_fig'):
                plt.close(self.fitting_fig)
            if hasattr(self, 'network_fig'):
                plt.close(self.network_fig)
            if hasattr(self, 'snn_fig'):
                plt.close(self.snn_fig)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Destroy the window
            self.quit()
            self.destroy()

# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Visual Synapse Simulator - Phase 1 Proof of Concept")
    print("="*60)
    print("\nThis tool demonstrates:")
    print("  • How light stimuli change synapse conductance")
    print("  • Wavelength-dependent learning rates")
    print("  • Nonlinear potentiation (α parameter)")
    print("  • Nonlinear depression (β parameter)")
    print("  • Conductance saturation (G_min → G_max)")
    print("\nControls:")
    print("  • Select Potentiation or Depression mode")
    print("  • Adjust sliders to set stimulus parameters")
    print("  • Click 'Apply Potentiation/Depression' to stimulate")
    print("  • Click 'Auto Demo' to see full cycle")
    print("  • Click 'Reset' to return to initial state")
    print("\nTry this experiment:")
    print("  1. Apply 20 potentiation pulses at 550nm")
    print("  2. Switch to Depression mode")
    print("  3. Apply 20 depression pulses at 550nm")
    print("  4. Watch conductance decrease back down!")
    print("\n" + "="*60 + "\n")
    
    # Create and run app
    app = SynapseSimulatorApp()
    app.mainloop()