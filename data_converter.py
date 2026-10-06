"""
Synapse Data Preparation Tool v2.0
===================================
Accepts time series data (pulse#/time vs conductance/current) with metadata.
Extracts parameters and creates JSON for network simulator.
"""

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# =============================================================================
# METADATA DIALOG
# =============================================================================

class MetadataDialog(ctk.CTkToplevel):
    """Dialog to collect metadata for each loaded file."""
    
    def __init__(self, parent, filename):
        super().__init__(parent)
        
        self.result = None
        self.filename = filename
        
        self.title(f"Metadata: {os.path.basename(filename)}")
        self.geometry("700x900")
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create metadata entry form."""
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self, width=650, height=800)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            scroll_frame,
            text=f"Metadata for: {os.path.basename(self.filename)}",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # === DATA COLUMNS ===
        frame = ctk.CTkFrame(scroll_frame)
        frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(frame, text="DATA COLUMNS", font=("Arial", 12, "bold")).pack(pady=5)
        
        ctk.CTkLabel(frame, text="X-axis represents:", anchor="w").pack(padx=10, anchor="w")
        self.x_type = ctk.CTkComboBox(
            frame,
            values=["time", "pulse_number", "frequency", "delta_t", "cycle_number"],
            width=250
        )
        self.x_type.pack(padx=10, pady=2)
        self.x_type.set("pulse_number")
        
        ctk.CTkLabel(frame, text="X-axis units:", anchor="w").pack(padx=10, anchor="w")
        self.x_units = ctk.CTkEntry(frame, placeholder_text="e.g., s, ms, Hz, #")
        self.x_units.pack(padx=10, pady=2, fill="x")
        self.x_units.insert(0, "#")
        
        ctk.CTkLabel(frame, text="Y-axis represents:", anchor="w").pack(padx=10, anchor="w")
        self.y_type = ctk.CTkComboBox(
            frame,
            values=["conductance", "current", "current_density", "delta_g", "ppf_ratio"],
            width=250
        )
        self.y_type.pack(padx=10, pady=2)
        self.y_type.set("conductance")
        
        ctk.CTkLabel(frame, text="Y-axis units:", anchor="w").pack(padx=10, anchor="w")
        self.y_units = ctk.CTkEntry(frame, placeholder_text="e.g., S, A, A/cm2")
        self.y_units.pack(padx=10, pady=2, fill="x")
        self.y_units.insert(0, "S")
        
        # === EXPERIMENT TYPE ===
        frame = ctk.CTkFrame(scroll_frame)
        frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(frame, text="EXPERIMENT TYPE", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.exp_type = ctk.CTkComboBox(
            frame,
            values=["potentiation", "depression", "potentiation_depression_cycle", 
                    "retention", "SRDP", "STDP"],
            width=300,
            command=self.on_exp_type_change
        )
        self.exp_type.pack(padx=10, pady=5)
        self.exp_type.set("potentiation")
        
        # === STIMULUS PARAMETERS (conditional based on experiment type) ===
        self.stim_frame = ctk.CTkFrame(scroll_frame)
        self.stim_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.stim_frame, text="STIMULUS PARAMETERS", font=("Arial", 12, "bold")).pack(pady=5)
        
        # POTENTIATION STIMULUS (for potentiation, potentiation_depression_cycle)
        self.pot_stim_frame = ctk.CTkFrame(self.stim_frame)
        self.pot_stim_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(self.pot_stim_frame, text="--- POTENTIATION STIMULUS ---", 
                    font=("Arial", 10, "bold"), text_color="#2196F3").pack(pady=2)
        
        ctk.CTkLabel(self.pot_stim_frame, text="Potentiation stimulus type:", anchor="w").pack(padx=10, anchor="w")
        self.pot_stim_type = ctk.CTkComboBox(self.pot_stim_frame, values=["light", "electrical"], 
                                             width=200, command=self.on_pot_stim_change)
        self.pot_stim_type.pack(padx=10, pady=2)
        self.pot_stim_type.set("light")
        
        # Wavelength (only for light)
        self.pot_wavelength_frame = ctk.CTkFrame(self.pot_stim_frame, fg_color="transparent")
        self.pot_wavelength_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.pot_wavelength_frame, text="Wavelength (nm):", anchor="w").pack(padx=10, anchor="w")
        self.wavelength_pot = ctk.CTkEntry(self.pot_wavelength_frame, placeholder_text="e.g., 365, 450, 550")
        self.wavelength_pot.pack(padx=10, pady=2, fill="x")
        
        # Light intensity (only for light)
        self.pot_intensity_frame = ctk.CTkFrame(self.pot_stim_frame, fg_color="transparent")
        self.pot_intensity_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.pot_intensity_frame, text="Light intensity (mW/cm²):", anchor="w").pack(padx=10, anchor="w")
        self.pot_intensity = ctk.CTkEntry(self.pot_intensity_frame, placeholder_text="e.g., 20")
        self.pot_intensity.pack(padx=10, pady=2, fill="x")
        
        # Voltage (only for electrical)
        self.pot_voltage_frame = ctk.CTkFrame(self.pot_stim_frame, fg_color="transparent")
        self.pot_voltage_frame.pack(fill="x", padx=5, pady=2)
        self.pot_voltage_frame.pack_forget()  # Hidden by default
        ctk.CTkLabel(self.pot_voltage_frame, text="Voltage (V):", anchor="w").pack(padx=10, anchor="w")
        self.pot_voltage = ctk.CTkEntry(self.pot_voltage_frame, placeholder_text="e.g., -2.0")
        self.pot_voltage.pack(padx=10, pady=2, fill="x")
        
        # DEPRESSION STIMULUS (only for depression, potentiation_depression_cycle)
        self.dep_stim_frame = ctk.CTkFrame(self.stim_frame)
        self.dep_stim_frame.pack(fill="x", padx=5, pady=5)
        self.dep_stim_frame.pack_forget()  # Hidden by default
        
        ctk.CTkLabel(self.dep_stim_frame, text="--- DEPRESSION STIMULUS ---", 
                    font=("Arial", 10, "bold"), text_color="#F44336").pack(pady=2)
        
        ctk.CTkLabel(self.dep_stim_frame, text="Depression stimulus type:", anchor="w").pack(padx=10, anchor="w")
        self.dep_stim_type = ctk.CTkComboBox(self.dep_stim_frame, values=["light", "electrical"], 
                                             width=200, command=self.on_dep_stim_change)
        self.dep_stim_type.pack(padx=10, pady=2)
        self.dep_stim_type.set("light")
        
        # Wavelength (only for light)
        self.dep_wavelength_frame = ctk.CTkFrame(self.dep_stim_frame, fg_color="transparent")
        self.dep_wavelength_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.dep_wavelength_frame, text="Wavelength (nm):", anchor="w").pack(padx=10, anchor="w")
        self.wavelength_dep = ctk.CTkEntry(self.dep_wavelength_frame, placeholder_text="e.g., 365")
        self.wavelength_dep.pack(padx=10, pady=2, fill="x")
        
        # Light intensity (only for light)
        self.dep_intensity_frame = ctk.CTkFrame(self.dep_stim_frame, fg_color="transparent")
        self.dep_intensity_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.dep_intensity_frame, text="Light intensity (mW/cm²):", anchor="w").pack(padx=10, anchor="w")
        self.dep_intensity = ctk.CTkEntry(self.dep_intensity_frame, placeholder_text="e.g., 20")
        self.dep_intensity.pack(padx=10, pady=2, fill="x")
        
        # Voltage (only for electrical)
        self.dep_voltage_frame = ctk.CTkFrame(self.dep_stim_frame, fg_color="transparent")
        self.dep_voltage_frame.pack(fill="x", padx=5, pady=2)
        self.dep_voltage_frame.pack_forget()  # Hidden by default
        ctk.CTkLabel(self.dep_voltage_frame, text="Voltage (V):", anchor="w").pack(padx=10, anchor="w")
        self.dep_voltage = ctk.CTkEntry(self.dep_voltage_frame, placeholder_text="e.g., 2.0")
        self.dep_voltage.pack(padx=10, pady=2, fill="x")
        
        # COMMON PULSE PARAMETERS
        self.pulse_params_frame = ctk.CTkFrame(self.stim_frame)
        self.pulse_params_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(self.pulse_params_frame, text="Pulse width (ms):", anchor="w").pack(padx=10, anchor="w")
        self.pulse_width = ctk.CTkEntry(self.pulse_params_frame, placeholder_text="e.g., 100")
        self.pulse_width.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.pulse_params_frame, text="Pulse frequency (Hz) or period (ms):", anchor="w").pack(padx=10, anchor="w")
        self.frequency = ctk.CTkEntry(self.pulse_params_frame, placeholder_text="e.g., 10 Hz or 100 ms")
        self.frequency.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.pulse_params_frame, text="Number of pulses per train:", anchor="w").pack(padx=10, anchor="w")
        self.n_pulses = ctk.CTkEntry(self.pulse_params_frame, placeholder_text="e.g., 50")
        self.n_pulses.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.pulse_params_frame, text="Read pulse timing relative to write (ms):", anchor="w").pack(padx=10, anchor="w")
        self.read_timing = ctk.CTkEntry(self.pulse_params_frame, placeholder_text="e.g., 10")
        self.read_timing.pack(padx=10, pady=2, fill="x")
        
        # RETENTION-SPECIFIC PARAMETERS
        self.retention_frame = ctk.CTkFrame(scroll_frame)
        self.retention_frame.pack(fill="x", padx=5, pady=5)
        self.retention_frame.pack_forget()  # Hidden by default
        
        ctk.CTkLabel(self.retention_frame, text="RETENTION MEASUREMENT", font=("Arial", 12, "bold")).pack(pady=5)
        ctk.CTkLabel(self.retention_frame, 
                    text="For retention, only the initial conductance state matters.\n"
                         "Specify how the device was prepared before decay measurement:",
                    font=("Arial", 9), justify="left", text_color="gray").pack(padx=10, pady=2)
        
        ctk.CTkLabel(self.retention_frame, text="Initial state preparation:", anchor="w").pack(padx=10, anchor="w")
        self.retention_prep = ctk.CTkComboBox(self.retention_frame, 
                                              values=["After potentiation", "After depression", "Custom state"],
                                              width=250)
        self.retention_prep.pack(padx=10, pady=2)
        self.retention_prep.set("After potentiation")
        
        ctk.CTkLabel(self.retention_frame, text="Initial conductance state (S):", anchor="w").pack(padx=10, anchor="w")
        self.retention_g_initial = ctk.CTkEntry(self.retention_frame, placeholder_text="e.g., 8e-5")
        self.retention_g_initial.pack(padx=10, pady=2, fill="x")
        
        # === CYCLE PARAMETERS (conditional) ===
        self.cycle_frame = ctk.CTkFrame(scroll_frame)
        self.cycle_frame.pack(fill="x", padx=5, pady=5)
        self.cycle_frame.pack_forget()  # Hidden by default
        
        ctk.CTkLabel(self.cycle_frame, text="CYCLE PARAMETERS", font=("Arial", 12, "bold")).pack(pady=5)
        
        ctk.CTkLabel(self.cycle_frame, text="Number of cycles:", anchor="w").pack(padx=10, anchor="w")
        self.n_cycles = ctk.CTkEntry(self.cycle_frame, placeholder_text="e.g., 10")
        self.n_cycles.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.cycle_frame, text="Delay between cycles (s):", anchor="w").pack(padx=10, anchor="w")
        self.cycle_delay = ctk.CTkEntry(self.cycle_frame, placeholder_text="e.g., 1")
        self.cycle_delay.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.cycle_frame, text="Rest period between cycles (s):", anchor="w").pack(padx=10, anchor="w")
        self.cycle_rest = ctk.CTkEntry(self.cycle_frame, placeholder_text="e.g., 0")
        self.cycle_rest.pack(padx=10, pady=2, fill="x")
        
        # === TRAIN PARAMETERS (conditional) ===
        self.train_frame = ctk.CTkFrame(scroll_frame)
        self.train_frame.pack(fill="x", padx=5, pady=5)
        self.train_frame.pack_forget()  # Hidden by default
        
        ctk.CTkLabel(self.train_frame, text="TRAIN PARAMETERS (SRDP)", font=("Arial", 12, "bold")).pack(pady=5)
        
        ctk.CTkLabel(self.train_frame, text="Number of trains:", anchor="w").pack(padx=10, anchor="w")
        self.n_trains = ctk.CTkEntry(self.train_frame, placeholder_text="e.g., 5")
        self.n_trains.pack(padx=10, pady=2, fill="x")
        
        ctk.CTkLabel(self.train_frame, text="Inter-train duration (s):", anchor="w").pack(padx=10, anchor="w")
        self.train_spacing = ctk.CTkEntry(self.train_frame, placeholder_text="e.g., 10")
        self.train_spacing.pack(padx=10, pady=2, fill="x")
        
        # === NOTES ===
        frame = ctk.CTkFrame(scroll_frame)
        frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(frame, text="NOTES (optional)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.notes = ctk.CTkTextbox(frame, height=60)
        self.notes.pack(padx=10, pady=5, fill="x")
        
        # === BUTTONS ===
        btn_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="OK",
            command=self.on_ok,
            width=150,
            fg_color="#2E7D32",
            hover_color="#1B5E20"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.on_cancel,
            width=150
        ).pack(side="left", padx=5)
        
    def on_exp_type_change(self, choice):
        """Show/hide conditional frames based on experiment type."""
        # Hide all conditional frames first
        self.cycle_frame.pack_forget()
        self.train_frame.pack_forget()
        self.dep_stim_frame.pack_forget()
        self.stim_frame.pack_forget()
        self.retention_frame.pack_forget()
        
        if choice == "potentiation":
            # Show only potentiation stimulus
            self.stim_frame.pack(fill="x", padx=5, pady=5)
            
        elif choice == "depression":
            # Show only depression stimulus (reuse the frames but relabel)
            self.stim_frame.pack(fill="x", padx=5, pady=5)
            
        elif choice == "potentiation_depression_cycle":
            # Show both potentiation and depression stimulus
            self.stim_frame.pack(fill="x", padx=5, pady=5)
            self.dep_stim_frame.pack(fill="x", padx=5, pady=5)
            self.cycle_frame.pack(fill="x", padx=5, pady=5)
            
        elif choice == "retention":
            # Show retention-specific parameters ONLY
            self.retention_frame.pack(fill="x", padx=5, pady=5)
            
        elif choice == "SRDP":
            self.stim_frame.pack(fill="x", padx=5, pady=5)
            self.train_frame.pack(fill="x", padx=5, pady=5)
            
        elif choice == "STDP":
            self.stim_frame.pack(fill="x", padx=5, pady=5)
    
    def on_pot_stim_change(self, choice):
        """Show/hide potentiation stimulus fields based on type."""
        if choice == "light":
            self.pot_wavelength_frame.pack(fill="x", padx=5, pady=2)
            self.pot_intensity_frame.pack(fill="x", padx=5, pady=2)
            self.pot_voltage_frame.pack_forget()
        else:  # electrical
            self.pot_wavelength_frame.pack_forget()
            self.pot_intensity_frame.pack_forget()
            self.pot_voltage_frame.pack(fill="x", padx=5, pady=2)
    
    def on_dep_stim_change(self, choice):
        """Show/hide depression stimulus fields based on type."""
        if choice == "light":
            self.dep_wavelength_frame.pack(fill="x", padx=5, pady=2)
            self.dep_intensity_frame.pack(fill="x", padx=5, pady=2)
            self.dep_voltage_frame.pack_forget()
        else:  # electrical
            self.dep_wavelength_frame.pack_forget()
            self.dep_intensity_frame.pack_forget()
            self.dep_voltage_frame.pack(fill="x", padx=5, pady=2)
    
    def on_ok(self):
        """Collect metadata and close."""
        try:
            exp_type = self.exp_type.get()
            
            metadata = {
                'x_type': self.x_type.get(),
                'x_units': self.x_units.get(),
                'y_type': self.y_type.get(),
                'y_units': self.y_units.get(),
                'experiment_type': exp_type,
                'notes': self.notes.get("1.0", "end").strip()
            }
            
            # Handle different experiment types
            if exp_type == "retention":
                # Retention: minimal metadata
                metadata['retention_preparation'] = self.retention_prep.get()
                if self.retention_g_initial.get():
                    metadata['g_initial_target'] = float(self.retention_g_initial.get())
                
            else:
                # All other experiments: collect stimulus parameters
                
                # Potentiation stimulus (or single stimulus for pot/dep)
                if exp_type in ["potentiation", "depression", "SRDP", "STDP"]:
                    pot_stim = self.pot_stim_type.get()
                    metadata['stimulus_type'] = pot_stim
                    
                    if pot_stim == "light":
                        if self.wavelength_pot.get():
                            metadata['wavelength_pot'] = float(self.wavelength_pot.get())
                        if self.pot_intensity.get():
                            metadata['light_intensity_mW_cm2'] = float(self.pot_intensity.get())
                    else:  # electrical
                        if self.pot_voltage.get():
                            metadata['voltage_V'] = float(self.pot_voltage.get())
                
                # Cycling: both potentiation AND depression stimuli
                elif exp_type == "potentiation_depression_cycle":
                    # Potentiation
                    pot_stim = self.pot_stim_type.get()
                    metadata['stimulus_type_pot'] = pot_stim
                    
                    if pot_stim == "light":
                        if self.wavelength_pot.get():
                            metadata['wavelength_pot'] = float(self.wavelength_pot.get())
                        if self.pot_intensity.get():
                            metadata['light_intensity_pot_mW_cm2'] = float(self.pot_intensity.get())
                    else:  # electrical
                        if self.pot_voltage.get():
                            metadata['voltage_pot_V'] = float(self.pot_voltage.get())
                    
                    # Depression
                    dep_stim = self.dep_stim_type.get()
                    metadata['stimulus_type_dep'] = dep_stim
                    
                    if dep_stim == "light":
                        if self.wavelength_dep.get():
                            metadata['wavelength_dep'] = float(self.wavelength_dep.get())
                        if self.dep_intensity.get():
                            metadata['light_intensity_dep_mW_cm2'] = float(self.dep_intensity.get())
                    else:  # electrical
                        if self.dep_voltage.get():
                            metadata['voltage_dep_V'] = float(self.dep_voltage.get())
                
                # Common pulse parameters (not for retention)
                if self.pulse_width.get():
                    metadata['pulse_width_ms'] = float(self.pulse_width.get())
                if self.frequency.get():
                    metadata['frequency_or_period'] = self.frequency.get()
                if self.n_pulses.get():
                    metadata['n_pulses_per_train'] = int(self.n_pulses.get())
                if self.read_timing.get():
                    metadata['read_pulse_timing_ms'] = float(self.read_timing.get())
            
            # Add cycle parameters if applicable
            if exp_type == "potentiation_depression_cycle":
                if self.n_cycles.get():
                    metadata['n_cycles'] = int(self.n_cycles.get())
                if self.cycle_delay.get():
                    metadata['cycle_delay_s'] = float(self.cycle_delay.get())
                if self.cycle_rest.get():
                    metadata['cycle_rest_s'] = float(self.cycle_rest.get())
            
            # Add train parameters if applicable
            if exp_type == "SRDP":
                if self.n_trains.get():
                    metadata['n_trains'] = int(self.n_trains.get())
                if self.train_spacing.get():
                    metadata['train_spacing_s'] = float(self.train_spacing.get())
            
            self.result = metadata
            self.destroy()
        
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Error parsing metadata:\n{str(e)}")
    
    def on_cancel(self):
        """Cancel metadata entry."""
        self.result = None
        self.destroy()


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DataProcessor:
    """Process time series data and compute derived metrics."""
    
    @staticmethod
    def compute_ppf(y_data):
        """Compute paired-pulse facilitation ratio."""
        if len(y_data) < 2:
            return None
        i1 = y_data[0]
        i2 = y_data[1]
        if i1 == 0:
            return None
        return (i2 - i1) / i1
    
    @staticmethod
    def compute_delta_g(y_data):
        """Compute total conductance change."""
        if len(y_data) < 2:
            return None
        return y_data[-1] - y_data[0]
    
    @staticmethod
    def compute_percent_change(y_data):
        """Compute percent change."""
        if len(y_data) < 2 or y_data[0] == 0:
            return None
        delta = y_data[-1] - y_data[0]
        return (delta / y_data[0]) * 100
    
    @staticmethod
    def compute_on_off_ratio(y_data):
        """Compute ON/OFF ratio."""
        g_max = np.max(y_data)
        g_min = np.min(y_data)
        if g_min == 0:
            return None
        return g_max / g_min
    
    @staticmethod
    def fit_retention(x_data, y_data):
        """Fit exponential decay to retention data."""
        try:
            # Exponential decay: G(t) = G_min + (G_0 - G_min) * exp(-t/tau)
            g_initial = y_data[0]
            g_min_est = np.min(y_data) * 0.95
            
            def exp_decay(t, tau):
                return g_min_est + (g_initial - g_min_est) * np.exp(-t / tau)
            
            popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[100], maxfev=5000)
            tau_fit = popt[0]
            
            # Calculate R²
            y_pred = exp_decay(x_data, tau_fit)
            residuals = y_data - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {'tau_retention_s': float(tau_fit), 'fit_R2': float(r_squared)}

        except Exception as e:
            # A bare `except: return None` here meant a FAILED fit and ABSENT
            # data were indistinguishable downstream: process_dataset tested
            # `if retention_fit:` and simply omitted the key, after which
            # fitting.py reported "No retention data - using default
            # decay_tau" — telling the user their data was missing when in
            # fact it was present and the fit had failed.
            #
            # The failure now travels WITH the dataset instead of vanishing.
            return {
                'tau_retention_s': None,
                'fit_R2': None,
                'fit_status': 'failed',
                'fit_error': f"{type(e).__name__}: {e}",
            }
    
    @staticmethod
    def process_dataset(x_data, y_data, metadata):
        """Process a single dataset and compute all derived metrics."""
        
        derived_metrics = {
            'G_initial': float(y_data[0]),
            'G_final': float(y_data[-1]),
            'delta_G': DataProcessor.compute_delta_g(y_data),
            'percent_change': DataProcessor.compute_percent_change(y_data),
            'G_min': float(np.min(y_data)),
            'G_max': float(np.max(y_data)),
            'on_off_ratio': DataProcessor.compute_on_off_ratio(y_data)
        }
        
        # Experiment-specific metrics
        exp_type = metadata.get('experiment_type')
        
        if exp_type in ['potentiation', 'depression']:
            ppf = DataProcessor.compute_ppf(y_data)
            if ppf is not None:
                derived_metrics['PPF'] = float(ppf)
        
        if exp_type == 'retention':
            retention_fit = DataProcessor.fit_retention(x_data, y_data)
            if retention_fit:
                derived_metrics.update(retention_fit)
                if retention_fit.get('fit_status') == 'failed':
                    print(f"WARNING: retention fit FAILED for this dataset: "
                          f"{retention_fit.get('fit_error')}\n"
                          f"         tau_retention_s is recorded as None with "
                          f"fit_status='failed'. The data is present; the fit "
                          f"is not trustworthy.")
        
        return derived_metrics


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class SynapseDataPrepApp(ctk.CTk):
    """Main application."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Synapse Data Preparation Tool v2.0")
        self.geometry("1400x800")
        
        # Storage
        self.datasets = []  # List of {filename, x_data, y_data, metadata, derived_metrics}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create UI."""
        
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Synapse Data Preparation Tool v2.0",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)
        
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Load time series data (pulse#/time vs conductance) with metadata",
            font=("Arial", 12),
            text_color="gray"
        )
        subtitle_label.pack(pady=5)
        
        # Content area (two columns)
        content_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, pady=10)
        
        # Left panel
        left_panel = ctk.CTkFrame(content_frame, width=400)
        left_panel.pack(side="left", fill="both", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="left", fill="both", expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup left control panel."""
        
        ctk.CTkLabel(
            parent,
            text="LOADED DATASETS",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Dataset list
        self.dataset_listbox = ctk.CTkTextbox(parent, height=400, font=("Courier", 9))
        self.dataset_listbox.pack(padx=10, pady=5, fill="both", expand=True)
        self.update_dataset_list()
        
        # Dataset selection
        ctk.CTkLabel(parent, text="Select dataset index to edit/remove:", anchor="w").pack(padx=10, anchor="w")
        self.dataset_index_entry = ctk.CTkEntry(parent, placeholder_text="e.g., 1", width=100)
        self.dataset_index_entry.pack(padx=10, pady=5)
        
        # Buttons
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="📁 Load Data File",
            command=self.load_file,
            height=40,
            font=("Arial", 12, "bold"),
            fg_color="#1976D2",
            hover_color="#1565C0"
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="📂 Batch Load (with metadata file)",
            command=self.batch_load_with_metadata,
            height=40,
            font=("Arial", 11, "bold"),
            fg_color="#00897B",
            hover_color="#00695C"
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="✏️ Edit Metadata",
            command=self.edit_metadata,
            height=35,
            fg_color="#FF9800",
            hover_color="#F57C00"
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="🗑️ Remove Dataset",
            command=self.remove_dataset,
            height=35,
            fg_color="#F44336",
            hover_color="#D32F2F"
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="💾 Save JSON",
            command=self.save_json,
            height=45,
            font=("Arial", 13, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20"
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="🔄 Clear All",
            command=self.clear_all,
            height=35,
            fg_color="#C62828",
            hover_color="#8E0000"
        ).pack(fill="x", pady=5)
        
    def setup_right_panel(self, parent):
        """Setup right preview panel."""
        
        ctk.CTkLabel(
            parent,
            text="DATA PREVIEW",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Matplotlib figure
        self.preview_fig = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, parent)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initial placeholder
        ax = self.preview_fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Load data files to see preview',
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        self.preview_canvas.draw()
        
    def load_file(self):
        """Load a data file and prompt for metadata."""
        
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
                ("Text Files", "*.txt *.dat"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Load file
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, delim_whitespace=True)
            
            if df.shape[1] < 2:
                messagebox.showerror("Error", "File must have at least 2 columns (X, Y)")
                return
            
            # Get first two columns
            x_data = df.iloc[:, 0].values
            y_data = df.iloc[:, 1].values
            
            # Open metadata dialog
            dialog = MetadataDialog(self, filepath)
            self.wait_window(dialog)
            
            if dialog.result is None:
                return  # User cancelled
            
            metadata = dialog.result
            
            # Compute derived metrics
            derived_metrics = DataProcessor.process_dataset(x_data, y_data, metadata)
            
            # Store dataset
            dataset = {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'x_data': x_data.tolist(),
                'y_data': y_data.tolist(),
                'metadata': metadata,
                'derived_metrics': derived_metrics
            }
            
            self.datasets.append(dataset)
            
            self.update_dataset_list()
            self.update_preview()
            
            messagebox.showinfo("Success", f"Loaded: {os.path.basename(filepath)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def batch_load_with_metadata(self):
        """
        Batch load multiple CSV files using a metadata instruction file.
        
        Workflow:
        1. Select metadata instructions file (e.g., METADATA_INSTRUCTIONS.txt)
        2. Select directory containing CSV files
        3. Automatically load all files with their metadata
        """
        # Step 1: Select metadata file
        metadata_filepath = filedialog.askopenfilename(
            title="Select Metadata Instructions File (e.g., METADATA_INSTRUCTIONS.txt)",
            filetypes=[
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        
        if not metadata_filepath:
            return
        
        # Step 2: Select directory containing CSV files
        csv_directory = filedialog.askdirectory(
            title="Select Directory Containing CSV Files"
        )
        
        if not csv_directory:
            return
        
        try:
            # Parse metadata file
            file_metadata_dict = self.parse_metadata_file(metadata_filepath)
            
            if not file_metadata_dict:
                messagebox.showerror("Error", "No valid metadata found in file")
                return
            
            # Load all CSV files
            loaded_count = 0
            failed_files = []
            
            for filename, metadata in file_metadata_dict.items():
                filepath = os.path.join(csv_directory, filename)
                
                if not os.path.exists(filepath):
                    failed_files.append(f"{filename} (not found)")
                    continue
                
                try:
                    # Load CSV
                    df = pd.read_csv(filepath)
                    
                    if df.shape[1] < 2:
                        failed_files.append(f"{filename} (< 2 columns)")
                        continue
                    
                    # Get first two columns
                    x_data = df.iloc[:, 0].values
                    y_data = df.iloc[:, 1].values
                    
                    # Compute derived metrics
                    derived_metrics = DataProcessor.process_dataset(x_data, y_data, metadata)
                    
                    # Store dataset
                    dataset = {
                        'filename': filename,
                        'filepath': filepath,
                        'x_data': x_data.tolist(),
                        'y_data': y_data.tolist(),
                        'metadata': metadata,
                        'derived_metrics': derived_metrics
                    }
                    
                    self.datasets.append(dataset)
                    loaded_count += 1
                    
                except Exception as e:
                    failed_files.append(f"{filename} ({str(e)})")
            
            # Update UI
            self.update_dataset_list()
            self.update_preview()
            
            # Show results
            message = f"Successfully loaded {loaded_count} dataset(s)"
            if failed_files:
                message += f"\n\nFailed to load {len(failed_files)} file(s):\n"
                message += "\n".join(failed_files[:5])  # Show first 5 failures
                if len(failed_files) > 5:
                    message += f"\n... and {len(failed_files) - 5} more"
            
            messagebox.showinfo("Batch Load Complete", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch load failed:\n{str(e)}")
    
    def parse_metadata_file(self, filepath):
        """Parse metadata instruction file and return dict of {filename: metadata}."""
        file_metadata = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        current_metadata = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for FILE: marker
            if line.startswith("FILE:"):
                # Save previous file's metadata
                if current_file and current_metadata:
                    file_metadata[current_file] = current_metadata.copy()
                
                # Start new file
                current_file = line.split("FILE:")[1].strip()
                current_metadata = {'notes': ''}
                i += 1
                continue
            
            # Skip separator lines and empty lines
            if line.startswith("---") or line.startswith("===") or not line:
                i += 1
                continue
            
            # Parse metadata fields
            if ":" in line and current_file:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                # Skip if value is empty or "(leave empty)"
                if not value or "(leave empty)" in value.lower():
                    i += 1
                    continue
                
                # Map keys to metadata dict keys
                if key == "X-axis represents":
                    current_metadata['x_type'] = value
                elif key == "X-axis units":
                    current_metadata['x_units'] = value
                elif key == "Y-axis represents":
                    current_metadata['y_type'] = value
                elif key == "Y-axis units":
                    current_metadata['y_units'] = value
                elif key == "Experiment type":
                    current_metadata['experiment_type'] = value
                elif key == "Stimulus type":
                    current_metadata['stimulus_type'] = value
                elif key == "Wavelength (nm) - potentiation":
                    try:
                        current_metadata['wavelength_pot'] = float(value)
                    except ValueError:
                        pass
                elif key == "Wavelength (nm) - depression":
                    try:
                        current_metadata['wavelength_dep'] = float(value)
                    except ValueError:
                        pass
                elif key == "Light intensity (mW/cm²)":
                    try:
                        current_metadata['light_intensity_mW_cm2'] = float(value.split()[0])
                    except ValueError:
                        pass
                elif "Voltage (V)" in key:
                    try:
                        current_metadata['voltage_V'] = float(value.split()[0])
                    except ValueError:
                        pass
                elif key == "Pulse width (ms)":
                    try:
                        current_metadata['pulse_width_ms'] = float(value)
                    except ValueError:
                        pass
                elif key == "Pulse frequency (Hz)" or key == "Frequency (Hz)":
                    try:
                        current_metadata['frequency_or_period'] = float(value)
                    except ValueError:
                        pass
                elif key == "Number of pulses" or key == "Number of pulses per train":
                    try:
                        current_metadata['n_pulses_per_train'] = int(value)
                    except ValueError:
                        pass
                elif key == "Read pulse timing relative to write (ms)":
                    try:
                        current_metadata['read_pulse_timing_ms'] = float(value)
                    except ValueError:
                        pass
                elif key == "Number of cycles":
                    try:
                        current_metadata['n_cycles'] = int(value)
                    except ValueError:
                        pass
                elif key == "Delay between cycles (s)":
                    try:
                        current_metadata['cycle_delay_s'] = float(value)
                    except ValueError:
                        pass
                elif key == "Rest period between cycles (s)":
                    try:
                        current_metadata['cycle_rest_s'] = float(value)
                    except ValueError:
                        pass
                elif key == "Notes":
                    current_metadata['notes'] = value
            
            i += 1
        
        # Don't forget last file
        if current_file and current_metadata:
            file_metadata[current_file] = current_metadata
        
        return file_metadata
    
    def update_dataset_list(self):
        """Update the dataset list display."""
        
        self.dataset_listbox.delete("1.0", "end")
        
        if not self.datasets:
            self.dataset_listbox.insert("1.0", "No datasets loaded.\n\nClick 'Load Data File' to begin.")
            return
        
        text = f"Loaded {len(self.datasets)} dataset(s):\n\n"
        
        for idx, ds in enumerate(self.datasets, 1):
            text += f"{idx}. {ds['filename']}\n"
            text += f"   Type: {ds['metadata']['experiment_type']}\n"
            text += f"   X: {ds['metadata']['x_type']} ({ds['metadata']['x_units']})\n"
            text += f"   Y: {ds['metadata']['y_type']} ({ds['metadata']['y_units']})\n"
            text += f"   Points: {len(ds['x_data'])}\n"
            
            if 'delta_G' in ds['derived_metrics'] and ds['derived_metrics']['delta_G'] is not None:
                text += f"   ΔG: {ds['derived_metrics']['delta_G']:.2e}\n"
            
            text += "\n"
        
        self.dataset_listbox.insert("1.0", text)
    
    def update_preview(self):
        """Update preview plot."""
        
        self.preview_fig.clear()
        
        if not self.datasets:
            ax = self.preview_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data loaded',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
        else:
            # Plot last loaded dataset
            ds = self.datasets[-1]
            ax = self.preview_fig.add_subplot(111)
            
            ax.plot(ds['x_data'], ds['y_data'], 'o-', linewidth=2, markersize=4)
            ax.set_xlabel(f"{ds['metadata']['x_type']} ({ds['metadata']['x_units']})")
            ax.set_ylabel(f"{ds['metadata']['y_type']} ({ds['metadata']['y_units']})")
            ax.set_title(f"{ds['filename']} - {ds['metadata']['experiment_type']}")
            ax.grid(True, alpha=0.3)
        
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()
    
    def save_json(self):
        """Save all datasets as JSON."""
        
        if not self.datasets:
            messagebox.showerror("Error", "No datasets loaded")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialfile=f"synapse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not filepath:
            return
        
        try:
            output = {
                'datasets': self.datasets,
                'n_datasets': len(self.datasets),
                'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'tool_version': '2.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2)
            
            messagebox.showinfo("Success", f"Saved {len(self.datasets)} dataset(s) to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
    
    def clear_all(self):
        """Clear all loaded datasets."""
        
        if not self.datasets:
            return
        
        if messagebox.askyesno("Confirm", "Clear all loaded datasets?"):
            self.datasets = []
            self.update_dataset_list()
            self.update_preview()
    
    def remove_dataset(self):
        """Remove a specific dataset by index."""
        
        if not self.datasets:
            messagebox.showerror("Error", "No datasets loaded")
            return
        
        try:
            index = int(self.dataset_index_entry.get())
            if index < 1 or index > len(self.datasets):
                messagebox.showerror("Error", f"Invalid index. Must be between 1 and {len(self.datasets)}")
                return
            
            dataset = self.datasets[index - 1]
            if messagebox.askyesno("Confirm", f"Remove dataset:\n{dataset['filename']}?"):
                self.datasets.pop(index - 1)
                self.dataset_index_entry.delete(0, "end")
                self.update_dataset_list()
                self.update_preview()
                messagebox.showinfo("Success", "Dataset removed")
        
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid dataset index")
    
    def edit_metadata(self):
        """Edit metadata of a specific dataset."""
        
        if not self.datasets:
            messagebox.showerror("Error", "No datasets loaded")
            return
        
        try:
            index = int(self.dataset_index_entry.get())
            if index < 1 or index > len(self.datasets):
                messagebox.showerror("Error", f"Invalid index. Must be between 1 and {len(self.datasets)}")
                return
            
            dataset = self.datasets[index - 1]
            
            # Open metadata dialog with existing metadata pre-filled
            dialog = MetadataDialog(self, dataset['filename'])
            
            # Pre-fill existing metadata
            old_metadata = dataset['metadata']
            
            # Set basic fields
            if 'x_type' in old_metadata:
                dialog.x_type.set(old_metadata['x_type'])
            if 'x_units' in old_metadata:
                dialog.x_units.delete(0, "end")
                dialog.x_units.insert(0, old_metadata['x_units'])
            if 'y_type' in old_metadata:
                dialog.y_type.set(old_metadata['y_type'])
            if 'y_units' in old_metadata:
                dialog.y_units.delete(0, "end")
                dialog.y_units.insert(0, old_metadata['y_units'])
            if 'experiment_type' in old_metadata:
                dialog.exp_type.set(old_metadata['experiment_type'])
                dialog.on_exp_type_change(old_metadata['experiment_type'])
            
            # Set stimulus fields based on experiment type
            exp_type = old_metadata.get('experiment_type', '')
            
            if exp_type == 'retention':
                if 'retention_preparation' in old_metadata:
                    dialog.retention_prep.set(old_metadata['retention_preparation'])
                if 'g_initial_target' in old_metadata:
                    dialog.retention_g_initial.delete(0, "end")
                    dialog.retention_g_initial.insert(0, str(old_metadata['g_initial_target']))
            else:
                # Potentiation stimulus
                if 'stimulus_type' in old_metadata:
                    dialog.pot_stim_type.set(old_metadata['stimulus_type'])
                    dialog.on_pot_stim_change(old_metadata['stimulus_type'])
                elif 'stimulus_type_pot' in old_metadata:
                    dialog.pot_stim_type.set(old_metadata['stimulus_type_pot'])
                    dialog.on_pot_stim_change(old_metadata['stimulus_type_pot'])
                
                if 'wavelength_pot' in old_metadata:
                    dialog.wavelength_pot.delete(0, "end")
                    dialog.wavelength_pot.insert(0, str(old_metadata['wavelength_pot']))
                if 'light_intensity_mW_cm2' in old_metadata:
                    dialog.pot_intensity.delete(0, "end")
                    dialog.pot_intensity.insert(0, str(old_metadata['light_intensity_mW_cm2']))
                elif 'light_intensity_pot_mW_cm2' in old_metadata:
                    dialog.pot_intensity.delete(0, "end")
                    dialog.pot_intensity.insert(0, str(old_metadata['light_intensity_pot_mW_cm2']))
                if 'voltage_V' in old_metadata:
                    dialog.pot_voltage.delete(0, "end")
                    dialog.pot_voltage.insert(0, str(old_metadata['voltage_V']))
                elif 'voltage_pot_V' in old_metadata:
                    dialog.pot_voltage.delete(0, "end")
                    dialog.pot_voltage.insert(0, str(old_metadata['voltage_pot_V']))
                
                # Depression stimulus (for cycling)
                if 'stimulus_type_dep' in old_metadata:
                    dialog.dep_stim_type.set(old_metadata['stimulus_type_dep'])
                    dialog.on_dep_stim_change(old_metadata['stimulus_type_dep'])
                
                if 'wavelength_dep' in old_metadata:
                    dialog.wavelength_dep.delete(0, "end")
                    dialog.wavelength_dep.insert(0, str(old_metadata['wavelength_dep']))
                if 'light_intensity_dep_mW_cm2' in old_metadata:
                    dialog.dep_intensity.delete(0, "end")
                    dialog.dep_intensity.insert(0, str(old_metadata['light_intensity_dep_mW_cm2']))
                if 'voltage_dep_V' in old_metadata:
                    dialog.dep_voltage.delete(0, "end")
                    dialog.dep_voltage.insert(0, str(old_metadata['voltage_dep_V']))
                
                # Common pulse parameters
                if 'pulse_width_ms' in old_metadata:
                    dialog.pulse_width.delete(0, "end")
                    dialog.pulse_width.insert(0, str(old_metadata['pulse_width_ms']))
                if 'frequency_or_period' in old_metadata:
                    dialog.frequency.delete(0, "end")
                    dialog.frequency.insert(0, str(old_metadata['frequency_or_period']))
                if 'n_pulses_per_train' in old_metadata:
                    dialog.n_pulses.delete(0, "end")
                    dialog.n_pulses.insert(0, str(old_metadata['n_pulses_per_train']))
                if 'read_pulse_timing_ms' in old_metadata:
                    dialog.read_timing.delete(0, "end")
                    dialog.read_timing.insert(0, str(old_metadata['read_pulse_timing_ms']))
                
                # Cycle parameters
                if 'n_cycles' in old_metadata:
                    dialog.n_cycles.delete(0, "end")
                    dialog.n_cycles.insert(0, str(old_metadata['n_cycles']))
                if 'cycle_delay_s' in old_metadata:
                    dialog.cycle_delay.delete(0, "end")
                    dialog.cycle_delay.insert(0, str(old_metadata['cycle_delay_s']))
                if 'cycle_rest_s' in old_metadata:
                    dialog.cycle_rest.delete(0, "end")
                    dialog.cycle_rest.insert(0, str(old_metadata['cycle_rest_s']))
                
                # Train parameters
                if 'n_trains' in old_metadata:
                    dialog.n_trains.delete(0, "end")
                    dialog.n_trains.insert(0, str(old_metadata['n_trains']))
                if 'train_spacing_s' in old_metadata:
                    dialog.train_spacing.delete(0, "end")
                    dialog.train_spacing.insert(0, str(old_metadata['train_spacing_s']))
            
            # Notes
            if 'notes' in old_metadata:
                dialog.notes.delete("1.0", "end")
                dialog.notes.insert("1.0", old_metadata['notes'])
            
            self.wait_window(dialog)
            
            if dialog.result is not None:
                # Update metadata
                dataset['metadata'] = dialog.result
                
                # Recompute derived metrics
                x_data = np.array(dataset['x_data'])
                y_data = np.array(dataset['y_data'])
                dataset['derived_metrics'] = DataProcessor.process_dataset(x_data, y_data, dialog.result)
                
                self.update_dataset_list()
                self.update_preview()
                messagebox.showinfo("Success", "Metadata updated")
        
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid dataset index")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app = SynapseDataPrepApp()
    # Unhandled Tk callback exceptions become a visible dialog rather than a
    # stderr traceback nobody reads.
    from gui_errors import install_tk_error_reporter
    install_tk_error_reporter(app, "SYNAPSYS Data Converter")
    app.mainloop()