"""
assembly_helpers.py (v2.0)
Converts raw Keithley measurement results into standardized characterization datasets
compatible with data_converter.py JSON format for fitting.py model extraction.

New features:
- Matches data_converter.py JSON structure with datasets array
- Comprehensive metadata for each measurement
- Raw time series + derived metrics
- Flexible for various experiment types

Usage:
    from assembly_helpers import build_characterization_suite
    
    suite = build_characterization_suite(
        wavelength_results=wl_list,
        wavelengths=[450, 500, 550, 600, 650],
        ltp_result=potentiation_data,
        cycle_results=cycling_data,
        retention_result=retention_data,
        metadata={'sample_id': 'Device_001', 'operator': 'John'}
    )
    
    json.dump(suite, open("device_characterization.json", "w"), indent=2)
"""

import numpy as np
import json
from typing import List, Dict, Optional
from datetime import datetime
from scipy.optimize import curve_fit


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_derived_metrics(x_data: np.ndarray, y_data: np.ndarray, 
                            metadata: Dict) -> Dict:
    """
    Compute all derived metrics from raw time series data.
    
    Args:
        x_data: X-axis data (time, pulse number, etc.)
        y_data: Y-axis data (conductance, current, etc.)
        metadata: Experiment metadata
    
    Returns:
        dict: Derived metrics including G_min, G_max, delta_G, etc.
    """
    derived = {
        'G_initial': float(y_data[0]) if len(y_data) > 0 else None,
        'G_final': float(y_data[-1]) if len(y_data) > 0 else None,
        'delta_G': float(y_data[-1] - y_data[0]) if len(y_data) > 1 else None,
        'percent_change': float((y_data[-1] - y_data[0]) / y_data[0] * 100) if len(y_data) > 1 and y_data[0] != 0 else None,
        'G_min': float(np.min(y_data)),
        'G_max': float(np.max(y_data)),
        'on_off_ratio': float(np.max(y_data) / np.min(y_data)) if np.min(y_data) > 0 else None
    }
    
    # Experiment-specific metrics
    exp_type = metadata.get('experiment_type', '')
    
    # PPF for potentiation/depression
    if exp_type in ['potentiation', 'depression'] and len(y_data) >= 2:
        i1 = y_data[0]
        i2 = y_data[1]
        if i1 != 0:
            derived['PPF'] = float((i2 - i1) / i1)
    
    # Retention decay fitting
    if exp_type == 'retention' and len(y_data) >= 3:
        try:
            g_initial = y_data[0]
            g_min_est = np.min(y_data) * 0.95
            
            def exp_decay(t, tau):
                return g_min_est + (g_initial - g_min_est) * np.exp(-t / tau)
            
            popt, pcov = curve_fit(exp_decay, x_data, y_data, p0=[100],
                                   maxfev=5000)
            tau_fit = popt[0]

            # A non-finite covariance means tau is NOT identifiable from this
            # data — SciPy emits "Covariance of the parameters could not be
            # estimated" and still returns a number. On flat or near-flat
            # retention that number is arbitrary (any large tau fits a
            # horizontal line equally well), so reporting it as a measured
            # retention time would be inventing a measurement. R^2 does not
            # catch this: on constant data ss_tot is 0 and R^2 is reported
            # as 0, which reads as "poor fit" rather than "no information".
            if not np.all(np.isfinite(pcov)):
                raise ValueError(
                    "retention time is not identifiable from this data "
                    "(covariance could not be estimated; the trace is flat or "
                    "too short to constrain a decay constant)"
                )
            tau_stderr = float(np.sqrt(pcov[0][0]))
            derived['tau_retention_stderr_s'] = tau_stderr
            if tau_fit > 0 and tau_stderr > abs(tau_fit):
                raise ValueError(
                    f"retention time is not constrained: tau = {tau_fit:.3g} s "
                    f"+/- {tau_stderr:.3g} s, an uncertainty larger than the "
                    f"value itself"
                )

            # R²
            y_pred = exp_decay(x_data, tau_fit)
            residuals = y_data - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            derived['tau_retention_s'] = float(tau_fit)
            derived['fit_R2'] = float(r_squared)
        except Exception as e:
            # A bare `except: pass` here left tau_retention_s simply absent,
            # with no record that a fit had been attempted and failed — the
            # downstream consumer could not tell "not measured" from "measured
            # and unfittable". The flag is explicit now.
            derived['tau_retention_s'] = None
            derived['retention_fit_failed'] = True
            derived['retention_fit_error'] = f"{type(e).__name__}: {e}"

    return derived


# =============================================================================
# DATASET CONVERSION FUNCTIONS
# =============================================================================

def convert_pulse_sequence_to_dataset(result: Dict, experiment_type: str, 
                                     stimulus_params: Dict,
                                     filename: str = "keithley_measurement") -> Dict:
    """
    Convert a single pulse_read_sequence result into the new dataset format.
    
    Args:
        result: Output from synapse_engine.pulse_read_sequence()
        experiment_type: One of ['potentiation', 'depression', 'retention']
        stimulus_params: Dict containing wavelength_nm, intensity, pulse_width, etc.
        filename: Name to identify this measurement
    
    Returns:
        dict: Dataset in data_converter.py format
    """
    # Extract data arrays.
    #
    # 'pulse_number' (SINGULAR) is what synapse_engine.pulse_read_sequence
    # actually returns. This branch used to test only the plural
    # 'pulse_numbers', which no production result ever carries — it matched
    # only this module's own __main__ fixture, which is why the self-test
    # passed while every real potentiation dataset was exported against a TIME
    # axis and labelled x_type='time'. The fitter's nonlinearity extraction
    # then sampled a time series as if it were a pulse index.
    # Both spellings are accepted; singular first, since that is the real one.
    if 'pulse_number' in result:
        x_data = result['pulse_number']
        x_type = 'pulse_number'
        x_units = '#'
    elif 'pulse_numbers' in result:
        x_data = result['pulse_numbers']
        x_type = 'pulse_number'
        x_units = '#'
    elif 'time_s' in result:
        x_data = result['time_s']
        x_type = 'time'
        x_units = 's'
    else:
        x_data = list(range(len(result['conductance_S'])))
        x_type = 'pulse_number'
        x_units = '#'

    # Visual/self-powered measurements store short-circuit current under
    # 'Jsc_A' and have no 'conductance_S' at all. An unguarded lookup here
    # raised a bare KeyError that the GUI surfaced as an opaque
    # "Failed to export suite" with no indication of which measurement or why.
    if 'conductance_S' not in result:
        available = ', '.join(sorted(k for k in result if not k.startswith('_')))
        raise ValueError(
            f"Measurement '{filename}' has no 'conductance_S' array, so it "
            f"cannot be converted to a {experiment_type} dataset.\n"
            f"Keys present: {available}\n"
            "Visual/self-powered measurements record 'Jsc_A' (short-circuit "
            "current) rather than conductance and are not convertible to a "
            "conductance dataset — exclude them from the export selection."
        )

    y_data = result['conductance_S']
    y_type = 'conductance'
    y_units = 'S'
    
    # Build metadata
    metadata = {
        'x_type': x_type,
        'x_units': x_units,
        'y_type': y_type,
        'y_units': y_units,
        'experiment_type': experiment_type,
        'stimulus_type': stimulus_params.get('stimulus_type', 'light'),
        'notes': stimulus_params.get('notes', '')
    }
    
    # Add stimulus parameters.
    #
    # The wavelength is filed under the key matching the branch it drove.
    # Writing a depression run's wavelength under 'wavelength_pot' — which is
    # what happened unconditionally before — mislabels the band, and the
    # spectral fit has no way to detect it.
    #
    # Legacy spellings are accepted so suites exported by an older build still
    # convert, but the canonical producer key is 'wavelength_nm'.
    _wl = stimulus_params.get('wavelength_nm',
                              stimulus_params.get('wavelength_pot',
                                                  stimulus_params.get('wavelength_dep')))
    if _wl is not None:
        wl_key = 'wavelength_dep' if experiment_type == 'depression' else 'wavelength_pot'
        metadata[wl_key] = _wl
    # No wavelength is left ABSENT, never defaulted: an electrical cycle
    # legitimately has none, and stamping 550/365 would invent a measurement.
    # fitting.py already reports the absence explicitly in its extraction
    # report, so the omission is flagged rather than silent.

    # Intensity, under the unit-bearing key fitting.py actually reads. The old
    # unit-less 'intensity' was written and never consumed, so A_peak/B_peak
    # extraction aborted for want of a value the user had supplied.
    _intensity = stimulus_params.get('light_intensity_mW_cm2',
                                     stimulus_params.get('intensity'))
    if _intensity is not None:
        metadata['light_intensity_mW_cm2'] = _intensity
        # Retained for backward compatibility with older readers.
        metadata['intensity'] = _intensity
    if 'pulse_width_ms' in stimulus_params:
        metadata['pulse_width_ms'] = stimulus_params['pulse_width_ms']
    if 'frequency_Hz' in stimulus_params:
        metadata['frequency_Hz'] = stimulus_params['frequency_Hz']
    if 'n_pulses' in stimulus_params:
        metadata['n_pulses'] = stimulus_params['n_pulses']
    if 'read_timing_ms' in stimulus_params:
        metadata['read_timing_ms'] = stimulus_params['read_timing_ms']
    
    # Compute derived metrics
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    derived_metrics = compute_derived_metrics(x_array, y_array, metadata)
    
    # Assemble dataset
    dataset = {
        'filename': filename,
        'filepath': f"keithley_measurement_{experiment_type}",
        'x_data': x_data if isinstance(x_data, list) else x_data.tolist(),
        'y_data': y_data if isinstance(y_data, list) else y_data.tolist(),
        'metadata': metadata,
        'derived_metrics': derived_metrics
    }
    
    return dataset


def convert_wavelength_sweep(results_list: List[Dict], wavelengths: List[float],
                            stimulus_params: Dict) -> List[Dict]:
    """
    Convert wavelength sweep results into multiple datasets (one per wavelength).
    
    Args:
        results_list: List of pulse_read_sequence results, one per wavelength
        wavelengths: Corresponding wavelength values (nm)
        stimulus_params: Base stimulus parameters (intensity, pulse_width, etc.)
    
    Returns:
        list: List of datasets, one per wavelength
    """
    if len(results_list) != len(wavelengths):
        raise ValueError("Number of results must match number of wavelengths")
    
    datasets = []
    for result, wl in zip(results_list, wavelengths):
        params = stimulus_params.copy()
        params['wavelength_nm'] = wl
        
        dataset = convert_pulse_sequence_to_dataset(
            result, 
            'potentiation',
            params,
            f"wavelength_{wl}nm"
        )
        
        datasets.append(dataset)
    
    return datasets


def convert_cycle_to_datasets(cycle_results: Dict, stimulus_params: Dict) -> List[Dict]:
    """
    Convert potentiation-depression cycling data into separate datasets.
    
    Args:
        cycle_results: Output from synapse_cycle.run_potentiation_depression_cycle()
        stimulus_params: Dict with pot and dep wavelengths, intensities, etc.
    
    Returns:
        list: List of datasets (separate for each potentiation and depression phase)
    """
    datasets = []
    
    for cycle_idx, cycle in enumerate(cycle_results['cycles']):
        # Potentiation phase
        if 'potentiation' in cycle and cycle['potentiation'] is not None:
            pot_data = cycle['potentiation']
            
            # No hardcoded wavelength fallback. A cycle exported without a
            # recorded potentiation wavelength used to be silently stamped
            # 550 nm, which then entered the spectral fit as though it had been
            # measured. Absent means absent: the key is omitted and the fitter
            # excludes the dataset from the wavelength response rather than
            # fitting a fabricated point.
            pot_params = {
                'stimulus_type': 'light',
                'intensity': stimulus_params.get('intensity_pot', 20),
                'pulse_width_ms': stimulus_params.get('pulse_width_ms', 100),
                'n_pulses': len(pot_data.get('conductance_S', [])),
                'notes': f'Potentiation cycle {cycle_idx + 1}'
            }
            if stimulus_params.get('wavelength_pot') is not None:
                pot_params['wavelength_nm'] = stimulus_params['wavelength_pot']


            dataset = convert_pulse_sequence_to_dataset(
                pot_data,
                'potentiation',
                pot_params,
                f"cycle{cycle_idx + 1}_potentiation"
            )
            dataset['metadata']['cycle_index'] = cycle_idx + 1
            dataset['metadata']['cycle_phase'] = 'potentiation'
            datasets.append(dataset)
        
        # Depression phase
        if 'depression' in cycle and cycle['depression'] is not None:
            dep_data = cycle['depression']
            
            # As above: no 365 nm fallback, and no falling back to the
            # POTENTIATION wavelength when the depression one is absent — that
            # substitution labelled a depression run with the potentiation
            # band, which is exactly the kind of error the spectral fit cannot
            # detect and cannot recover from.
            dep_params = {
                'stimulus_type': 'light',
                'intensity': stimulus_params.get('intensity_dep', 20),
                'pulse_width_ms': stimulus_params.get('pulse_width_ms', 100),
                'n_pulses': len(dep_data.get('conductance_S', [])),
                'notes': f'Depression cycle {cycle_idx + 1}'
            }
            if stimulus_params.get('wavelength_dep') is not None:
                dep_params['wavelength_nm'] = stimulus_params['wavelength_dep']


            dataset = convert_pulse_sequence_to_dataset(
                dep_data,
                'depression',
                dep_params,
                f"cycle{cycle_idx + 1}_depression"
            )
            dataset['metadata']['cycle_index'] = cycle_idx + 1
            dataset['metadata']['cycle_phase'] = 'depression'
            datasets.append(dataset)
    
    # Add cycle-level metadata to all datasets.
    #
    # This loop used to overwrite every dataset's experiment_type with
    # 'potentiation_depression_cycle', one line after the pot/dep distinction
    # had been computed and the derived_metrics calculated against it. The
    # consequences were total: fitting.py skips that type for the spectral fit
    # and filters it out of the depression-nonlinearity fit, so cycle data
    # contributed to neither, and beta could not be extracted from it.
    #
    # The phase label now survives. Cycle membership is recorded alongside it
    # in dedicated keys, which is what the overwrite was presumably reaching
    # for — the two pieces of information are orthogonal and both are needed.
    for ds in datasets:
        ds['metadata']['n_cycles'] = len(cycle_results['cycles'])
        ds['metadata']['from_cycling_measurement'] = True

    return datasets


def convert_stdp_to_dataset(stdp_result: Dict, stimulus_params: Dict,
                            filename: str = "stdp_timing_window") -> Dict:
    """
    Convert an STDP timing-window measurement into a dataset.

    There was previously no STDP converter anywhere in the suite, and
    build_characterization_suite had no STDP parameter, so
    `fitting.py`'s STDP extractor could never be reached from hardware data —
    the STDP window always came from defaults no matter what was measured.

    The fitter reads STDP datasets as a plain (x, y) pair: x is Δt in ms and y
    is the conductance after the pairing (`fitting.py::DataParser._build_stdp`
    derives the baseline itself, from the median of the largest-|Δt| points).
    So the conductance array is what must be exported, not the pre-computed
    percentage — the fitter recomputes that against its own baseline.

    Args:
        stdp_result: output of synapse_engine.measure_stdp / simulate_stdp,
            carrying 'delta_t_ms' and 'g_final_S'.
        stimulus_params: base stimulus conditions.
        filename: identifier for this measurement.

    Returns:
        dict: dataset in data_converter.py format.
    """
    if 'delta_t_ms' not in stdp_result:
        raise ValueError(
            f"'{filename}' is not an STDP measurement: no 'delta_t_ms' array. "
            f"Keys present: {', '.join(sorted(stdp_result))}"
        )
    if 'g_final_S' not in stdp_result:
        raise ValueError(
            f"STDP measurement '{filename}' has no 'g_final_S' array, so the "
            "post-pairing conductance cannot be exported. "
            f"Keys present: {', '.join(sorted(stdp_result))}"
        )

    x_data = list(stdp_result['delta_t_ms'])
    y_data = list(stdp_result['g_final_S'])
    if len(x_data) != len(y_data):
        raise ValueError(
            f"STDP measurement '{filename}' is inconsistent: "
            f"{len(x_data)} Δt values against {len(y_data)} conductance values."
        )

    metadata = {
        'x_type': 'delta_t',
        'x_units': 'ms',
        'y_type': 'conductance',
        'y_units': 'S',
        'experiment_type': 'stdp',
        'stimulus_type': stimulus_params.get('stimulus_type', 'electrical'),
        'notes': stimulus_params.get('notes', 'STDP timing window'),
    }
    for key in ('wavelength_nm', 'intensity', 'pulse_width_ms', 'n_pulses'):
        if key in stimulus_params:
            metadata['wavelength_pot' if key == 'wavelength_nm' else key] = \
                stimulus_params[key]

    derived_metrics = compute_derived_metrics(
        np.array(x_data, dtype=float), np.array(y_data, dtype=float), metadata
    )

    return {
        'filename': filename,
        'filepath': 'keithley_measurement_stdp',
        'x_data': x_data,
        'y_data': y_data,
        'metadata': metadata,
        'derived_metrics': derived_metrics,
    }


def convert_srdp_to_dataset(srdp_result: Dict, stimulus_params: Dict,
                            filename: str = "srdp_frequency_response") -> Dict:
    """
    Convert an SRDP frequency-response measurement into a dataset.

    Counterpart to convert_stdp_to_dataset; see that docstring for why these
    did not previously exist. The fitter reads x as frequency in Hz and y as
    the conductance after the train.

    Note the frequency axis is the *nominal* requested frequency. Where the
    instrument could not deliver it, the true rate is recoverable from the
    buffer timestamps — see audit H14, which is fixed separately in package P3.
    """
    if 'frequencies_hz' not in srdp_result:
        raise ValueError(
            f"'{filename}' is not an SRDP measurement: no 'frequencies_hz' "
            f"array. Keys present: {', '.join(sorted(srdp_result))}"
        )
    if 'g_final_S' not in srdp_result:
        raise ValueError(
            f"SRDP measurement '{filename}' has no 'g_final_S' array. "
            f"Keys present: {', '.join(sorted(srdp_result))}"
        )

    x_data = list(srdp_result['frequencies_hz'])
    y_data = list(srdp_result['g_final_S'])
    if len(x_data) != len(y_data):
        raise ValueError(
            f"SRDP measurement '{filename}' is inconsistent: "
            f"{len(x_data)} frequencies against {len(y_data)} conductance values."
        )

    metadata = {
        'x_type': 'frequency',
        'x_units': 'Hz',
        'y_type': 'conductance',
        'y_units': 'S',
        'experiment_type': 'srdp',
        'stimulus_type': stimulus_params.get('stimulus_type', 'electrical'),
        'notes': stimulus_params.get('notes', 'SRDP frequency response'),
    }
    for key in ('wavelength_nm', 'intensity', 'pulse_width_ms', 'n_pulses'):
        if key in stimulus_params:
            metadata['wavelength_pot' if key == 'wavelength_nm' else key] = \
                stimulus_params[key]

    derived_metrics = compute_derived_metrics(
        np.array(x_data, dtype=float), np.array(y_data, dtype=float), metadata
    )

    return {
        'filename': filename,
        'filepath': 'keithley_measurement_srdp',
        'x_data': x_data,
        'y_data': y_data,
        'metadata': metadata,
        'derived_metrics': derived_metrics,
    }


def convert_retention_to_dataset(retention_result: Dict,
                                stimulus_params: Dict) -> Dict:
    """
    Convert retention measurement to dataset format.
    
    Args:
        retention_result: Output from measure_retention() function
        stimulus_params: Dict with measurement details
    
    Returns:
        dict: Dataset in data_converter.py format
    """
    params = stimulus_params.copy()
    params['notes'] = params.get('notes', 'Retention/decay measurement')
    
    # Build result dict if not already in right format
    if 'time_s' not in retention_result:
        # Assume it's already in the right format
        result = retention_result
    else:
        result = {
            'time_s': retention_result['time_s'],
            'conductance_S': retention_result['conductance_S']
        }
    
    dataset = convert_pulse_sequence_to_dataset(
        result,
        'retention',
        params,
        'retention_measurement'
    )
    
    return dataset


# =============================================================================
# MASTER ASSEMBLY FUNCTION
# =============================================================================

def build_characterization_suite(
    wavelength_results: Optional[List[Dict]] = None,
    wavelengths: Optional[List[float]] = None,
    ltp_result: Optional[Dict] = None,
    ltd_result: Optional[Dict] = None,
    cycle_results: Optional[Dict] = None,
    retention_result: Optional[Dict] = None,
    stdp_result: Optional[Dict] = None,
    srdp_result: Optional[Dict] = None,
    stimulus_params: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Build complete characterization suite from raw Keithley measurement results.
    Now generates JSON format compatible with data_converter.py output.
    
    Args:
        wavelength_results: List of results at different wavelengths
        wavelengths: Corresponding wavelength values (nm)
        ltp_result: Single LTP measurement for nonlinearity
        ltd_result: Single LTD measurement (optional)
        cycle_results: Potentiation-depression cycling data
        retention_result: Decay measurement
        stdp_result: STDP timing-window sweep (for the STDP window fit)
        srdp_result: SRDP frequency sweep (for the SRDP sigmoid fit)
        stimulus_params: Dict with measurement conditions (intensity, pulse_width, etc.)
        metadata: Additional information (sample_id, date, operator, etc.)
    
    Returns:
        dict: Complete characterization_suite in data_converter.py format
    """
    datasets = []
    
    if stimulus_params is None:
        stimulus_params = {
            'stimulus_type': 'light',
            'intensity': 20,
            'pulse_width_ms': 100,
            'frequency_Hz': 10
        }
    
    # Convert wavelength sweep
    if wavelength_results is not None and wavelengths is not None:
        wl_datasets = convert_wavelength_sweep(wavelength_results, wavelengths, stimulus_params)
        datasets.extend(wl_datasets)
    
    # Convert LTP (nonlinearity)
    if ltp_result is not None:
        params = stimulus_params.copy()
        params['notes'] = 'Long-term potentiation for nonlinearity extraction'
        ltp_dataset = convert_pulse_sequence_to_dataset(
            ltp_result, 'potentiation', params, 'ltp_nonlinearity'
        )
        datasets.append(ltp_dataset)
    
    # Convert LTD
    if ltd_result is not None:
        params = stimulus_params.copy()
        params['notes'] = 'Long-term depression'
        ltd_dataset = convert_pulse_sequence_to_dataset(
            ltd_result, 'depression', params, 'ltd_measurement'
        )
        datasets.append(ltd_dataset)
    
    # Convert cycling data
    if cycle_results is not None:
        cycle_datasets = convert_cycle_to_datasets(cycle_results, stimulus_params)
        datasets.extend(cycle_datasets)
    
    # Convert retention
    if retention_result is not None:
        retention_dataset = convert_retention_to_dataset(retention_result, stimulus_params)
        datasets.append(retention_dataset)

    # Convert STDP timing window
    if stdp_result is not None:
        params = stimulus_params.copy()
        params['notes'] = 'STDP timing window'
        datasets.append(convert_stdp_to_dataset(stdp_result, params))

    # Convert SRDP frequency response
    if srdp_result is not None:
        params = stimulus_params.copy()
        params['notes'] = 'SRDP frequency response'
        datasets.append(convert_srdp_to_dataset(srdp_result, params))
    
    # Build final suite in data_converter.py format
    suite = {
        'datasets': datasets,
        'n_datasets': len(datasets),
        'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tool_version': '2.0',
        'source': 'keithley_measurement'
    }
    
    # Add user metadata
    if metadata is not None:
        suite['user_metadata'] = metadata
    else:
        suite['user_metadata'] = {'note': 'No metadata provided'}
    
    return suite


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def make_wavelength_response(results_list: List[Dict], wavelengths: List[float]) -> Dict:
    """
    Legacy function for backward compatibility.
    Generates old-style wavelength_response dict.
    """
    if len(results_list) != len(wavelengths):
        raise ValueError("Number of results must match number of wavelengths")
    
    dataset = {
        'wavelengths_nm': [],
        'delta_g_S': [],
        'delta_g_percent': [],
        'g_initial_S': [],
        'g_final_S': [],
        'measurement_type': 'wavelength_sweep'
    }
    
    for result, wl in zip(results_list, wavelengths):
        G_array = np.array(result['conductance_S'])
        
        if len(G_array) < 2:
            raise ValueError(f"Not enough data points in result for wavelength {wl}nm")
        
        g_initial = G_array[0]
        g_final = G_array[-1]
        delta_g = g_final - g_initial
        delta_g_percent = (delta_g / g_initial * 100) if g_initial != 0 else 0.0
        
        dataset['wavelengths_nm'].append(wl)
        dataset['delta_g_S'].append(float(delta_g))
        dataset['delta_g_percent'].append(float(delta_g_percent))
        dataset['g_initial_S'].append(float(g_initial))
        dataset['g_final_S'].append(float(g_final))
    
    return dataset


# =============================================================================
# FILE I/O
# =============================================================================

def save_characterization_suite(suite: Dict, filename: str):
    """
    Save characterization suite to JSON file.
    
    Args:
        suite: Output from build_characterization_suite()
        filename: Path to save JSON (e.g., "device_001_characterization.json")
    """
    with open(filename, 'w') as f:
        json.dump(suite, f, indent=2)
    
    print(f"✓ Characterization suite saved to {filename}")
    print(f"  Total datasets: {suite['n_datasets']}")
    print(f"  Experiment types: {set(ds['metadata']['experiment_type'] for ds in suite['datasets'])}")


def load_characterization_suite(filename: str) -> Dict:
    """
    Load characterization suite from JSON file.
    Handles both old and new formats.
    
    Args:
        filename: Path to JSON file
    
    Returns:
        dict: characterization_suite
    """
    with open(filename, 'r') as f:
        suite = json.load(f)
    
    print(f"✓ Loaded characterization suite from {filename}")
    
    # Check format
    if 'datasets' in suite:
        print(f"  Format: v2.0 (data_converter compatible)")
        print(f"  Total datasets: {suite['n_datasets']}")
    else:
        print(f"  Format: v1.0 (legacy)")
        print(f"  Datasets found: {list(suite.keys())}")
    
    return suite


# =============================================================================
# VALIDATION
# =============================================================================

def validate_suite(suite: Dict) -> Dict:
    """
    Check characterization suite and report what's available.
    Returns validation report with available measurements.
    
    Returns:
        dict: Validation report with status and available measurements
    """
    report = {
        'valid': True,
        'format': 'unknown',
        'available_measurements': [],
        'missing_critical': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Detect format
    if 'datasets' in suite:
        report['format'] = 'v2.0'
        datasets = suite['datasets']
        
        # Categorize by experiment type
        exp_types = {}
        for ds in datasets:
            exp_type = ds['metadata'].get('experiment_type', 'unknown')
            if exp_type not in exp_types:
                exp_types[exp_type] = []
            exp_types[exp_type].append(ds)
        
        report['available_measurements'] = list(exp_types.keys())
        
        # Check for minimum requirements
        has_dynamic_range = False
        for exp_type in ['potentiation_depression_cycle', 'potentiation', 'depression']:
            if exp_type in exp_types:
                has_dynamic_range = True
                break
        
        if not has_dynamic_range:
            report['missing_critical'].append('dynamic_range')
            report['warnings'].append("No potentiation/depression data found - cannot determine G_min/G_max")
            report['valid'] = False
        
        # Recommendations
        if 'retention' not in exp_types:
            report['recommendations'].append("Add retention measurement for decay time constant (tau)")
        
        if len([ds for ds in datasets if 'wavelength' in ds['metadata'].get('notes', '').lower()]) < 3:
            report['recommendations'].append("Add wavelength sweep (3+ wavelengths) for spectral sensitivity")
        
        # Counts DATASETS, not data points. The old wording ("current: N,
        # recommended: 10+") read as a point count, so a single 200-pulse
        # potentiation trace — ample for a nonlinearity fit — reported as
        # "current: 1, recommended: 10+".
        pot_datasets = exp_types.get('potentiation', [])
        longest_pot = max((len(ds['y_data']) for ds in pot_datasets), default=0)
        if longest_pot < 10:
            report['recommendations'].append(
                f"Add a longer potentiation trace for nonlinearity fitting "
                f"(longest available: {longest_pot} points across "
                f"{len(pot_datasets)} dataset(s); 10+ points required, "
                f"30+ recommended)"
            )

        if 'stdp' not in exp_types:
            report['recommendations'].append(
                "Add an STDP timing-window sweep, or the STDP parameters will "
                "come from defaults rather than from this device"
            )
        if 'srdp' not in exp_types:
            report['recommendations'].append(
                "Add an SRDP frequency sweep, or the SRDP parameters will come "
                "from defaults rather than from this device"
            )
        if 'depression' not in exp_types:
            report['recommendations'].append(
                "Add a depression (LTD) measurement — beta cannot be fitted "
                "without one"
            )
    
    else:
        report['format'] = 'v1.0'
        report['available_measurements'] = [k for k in suite.keys() if k != 'metadata']
        
        if 'dynamic_range' not in suite:
            report['missing_critical'].append('dynamic_range')
            report['warnings'].append("Missing dynamic_range measurement - cannot determine G_min/G_max")
            report['valid'] = False
    
    # Print report
    print("\n" + "="*60)
    print("CHARACTERIZATION SUITE VALIDATION")
    print("="*60)
    print(f"Format: {report['format']}")
    print(f"Status: {'✓ VALID' if report['valid'] else '✗ INVALID - Missing critical measurements'}")
    print(f"\nAvailable measurements: {', '.join(report['available_measurements'])}")
    
    if report['missing_critical']:
        print(f"\n✗ Missing critical: {', '.join(report['missing_critical'])}")
    
    if report['warnings']:
        print("\n⚠ Warnings:")
        for w in report['warnings']:
            print(f"  - {w}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for r in report['recommendations']:
            print(f"  - {r}")
    
    print("="*60)
    
    return report


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the updated assembly helpers"""
    print("Testing assembly_helpers.py v2.0...")
    
    # Test 1: Build suite with new format
    print("\n1. Testing new format conversion...")
    
    # Fake wavelength results
    fake_wl_results = []
    wavelengths = [450, 500, 550, 600, 650]
    for wl in wavelengths:
        fake_wl_results.append({
            'pulse_number': list(range(50)),
            'conductance_S': list(np.linspace(1e-6, 1e-5, 50))
        })
    
    # Fake LTP
    fake_ltp = {
        'pulse_number': list(range(100)),
        'conductance_S': list(np.logspace(-6, -5, 100))
    }
    
    # Fake cycles
    fake_cycles = {
        'cycles': [
            {
                'potentiation': {
                    'pulse_number': list(range(50)),
                    'conductance_S': list(np.linspace(1e-6, 8e-5, 50))
                },
                'depression': {
                    'pulse_number': list(range(50)),
                    'conductance_S': list(np.linspace(8e-5, 2e-6, 50))
                }
            }
            for _ in range(3)
        ]
    }
    
    # Build suite
    suite = build_characterization_suite(
        wavelength_results=fake_wl_results,
        wavelengths=wavelengths,
        ltp_result=fake_ltp,
        cycle_results=fake_cycles,
        stimulus_params={
            'stimulus_type': 'light',
            'intensity': 20,
            'pulse_width_ms': 100,
            'wavelength_pot': 550,
            'wavelength_dep': 365
        },
        metadata={'sample_id': 'TEST_001', 'test': True}
    )
    
    print(f"   ✓ Built suite with {suite['n_datasets']} datasets")
    
    # Test 2: Save/load
    print("\n2. Testing save/load...")
    save_characterization_suite(suite, "test_characterization_v2.json")
    loaded = load_characterization_suite("test_characterization_v2.json")
    print(f"   ✓ Save/load successful")
    
    # Test 3: Validation
    print("\n3. Testing validation...")
    report = validate_suite(suite)
    
    print("\n✓ All tests passed!")
