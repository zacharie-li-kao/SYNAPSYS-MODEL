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
# PER-RUN METADATA AND RUN CLASSIFICATION
# =============================================================================
#
# Every stored run carries the parameters it was measured with. A suite used to
# be stamped with ONE stimulus block taken from the first run and applied to all
# of them, so a sweep measured at 10/20/30 mW/cm2 exported at 30 throughout and
# an electrical LTD run was exported as light at 650 nm. Metadata is now derived
# from each run itself; a shared `stimulus_params` is only a base for keys the
# run does not carry.

_LIGHT_MODES = ('visual',)                                   # SMU drives an LED
_ELECTRICAL_MODES = ('electrical', 'memristor_pulse', 'srdp', 'stdp')
_PULSE_MODES = ('electrical', 'visual', 'memristor_pulse')
_NON_STIMULUS_MODES = ('retention',)                         # read-only sequence


def run_params(result: Dict) -> Dict:
    """The parameter dict a stored run carries ({} when it carries none)."""
    for key in ('params', 'base_params'):
        if isinstance(result.get(key), dict):
            return result[key]
    return {}


def stimulus_params_from_params(p: Dict, stimulus_type: Optional[str] = None) -> Dict:
    """Stimulus metadata for ONE run, from the parameters it was measured with.

    Only what the run recorded is returned; an absent value stays absent so the
    fitter can report it, rather than being filled with a plausible number.
    Wavelength and intensity are written only for an optical stimulus: for an
    electrical run they would be a "manual light" note that the spectral fit
    (which groups every potentiation/depression dataset by wavelength) would
    take for the driving wavelength.

    Raises ValueError on a mode that is not a canonical engine mode.
    """
    out: Dict = {}
    mode = p.get('mode')
    if stimulus_type is None and mode:
        if mode in _LIGHT_MODES:
            stimulus_type = 'light'
        elif mode in _ELECTRICAL_MODES:
            stimulus_type = 'electrical'
        elif mode not in _NON_STIMULUS_MODES:
            raise ValueError(
                f"Unrecognised measurement mode {mode!r}; expected one of "
                f"{_PULSE_MODES + ('srdp', 'stdp', 'retention')}."
            )
    if stimulus_type is not None:
        out['stimulus_type'] = stimulus_type
    if stimulus_type == 'light':
        if p.get('wavelength_nm') is not None:
            out['wavelength_nm'] = p['wavelength_nm']
        if p.get('intensity_mW_cm2') is not None:
            out['light_intensity_mW_cm2'] = p['intensity_mW_cm2']
    width = p.get('stim_width_ms', p.get('pulse_width_ms'))
    if width is not None:
        out['pulse_width_ms'] = width
    period = p.get('stim_period_ms', p.get('pulse_period_ms'))
    if period:
        out['pulse_period_ms'] = period
        out['frequency_Hz'] = 1000.0 / period
    if p.get('n_pulses') is not None:
        out['n_pulses'] = p['n_pulses']
    return out


def stimulus_params_from_run(result: Dict) -> Dict:
    return stimulus_params_from_params(run_params(result))


def _resolve_params(shared: Optional[Dict], result: Dict,
                    explicit: Optional[Dict] = None) -> Dict:
    """shared base  <  the run's own parameters  <  explicit per-result override."""
    merged = dict(shared or {})
    merged.update(stimulus_params_from_run(result))
    merged.update(explicit or {})
    return merged


def run_sample_id(result: Dict) -> Optional[str]:
    """The sample ID a run was measured under, or None when it recorded none."""
    for holder in (result.get('metadata'), run_params(result), result.get('pot_params')):
        sid = holder.get('sample_id') if isinstance(holder, dict) else None
        if sid and sid != 'unknown':
            return sid
    return None


def classify_run(result: Dict) -> str:
    """Kind of a stored run, from what it actually contains (not from its length)."""
    if 'devices' in result:
        return 'multi_device'
    if 'cycles' in result:
        return 'cycles'
    if 'frequencies_hz' in result:
        return 'srdp'
    if 'delta_t_ms' in result:
        return 'stdp'
    if 'Jsc_A' in result:
        return 'visual_jsc'
    mode = run_params(result).get('mode')
    if mode == 'retention' or result.get('metadata', {}).get('measurement_type') == 'retention':
        return 'retention'
    if 'conductance_S' in result and ('pulse_number' in result or mode in _PULSE_MODES):
        return 'pulse'
    return 'unknown'


# Which kinds of run each suite slot can take. A Visual (Jsc) run has no
# conductance, so it fits no slot; a retention trace is not a pulse train.
ROLE_KINDS = {
    'ltp': ('pulse',), 'ltd': ('pulse',), 'cycles': ('cycles',),
    'retention': ('retention',), 'stdp': ('stdp',), 'srdp': ('srdp',),
}


def check_role(result: Dict, role: str) -> Optional[str]:
    """None if `result` can fill suite slot `role`, else the reason it cannot."""
    kind = classify_run(result)
    if kind in ROLE_KINDS[role]:
        return None
    reason = {
        'visual_jsc': "a Visual (self-powered) run records photocurrent Jsc, not conductance",
        'multi_device': "a multi-device run holds several devices; a suite describes one device",
        'unknown': "its contents are not recognised as a measurement",
    }.get(kind, f"it is a {kind} run")
    return (f"This run cannot be used as {role.upper()}: {reason}. "
            f"{role.upper()} needs a {' / '.join(ROLE_KINDS[role])} run.")


def run_branch(result: Dict) -> Optional[str]:
    """'potentiation', 'depression' or 'flat' from the run's own conductance
    (None when it has too few finite points to say)."""
    g = np.asarray(result.get('conductance_S', []), dtype=float)
    g = g[np.isfinite(g)]
    if g.size < 2:
        return None
    return {1: 'potentiation', -1: 'depression', 0: 'flat'}[int(np.sign(g[-1] - g[0]))]


def describe_run(result: Dict) -> str:
    """One-line description of a run by what it is, for selection lists."""
    kind = classify_run(result)
    p = run_params(result)
    sid = run_sample_id(result)
    tail = f" | sample {sid}" if sid else ""
    if kind == 'cycles':
        return f"Pot-Dep cycles ({len(result['cycles'])} cycle(s)){tail}"
    if kind == 'multi_device':
        return f"Multi-device cycles ({len(result['devices'])} devices){tail}"
    if kind == 'srdp':
        return f"SRDP ({len(result['frequencies_hz'])} frequency points){tail}"
    if kind == 'stdp':
        return f"STDP ({len(result['delta_t_ms'])} timing points){tail}"
    if kind == 'visual_jsc':
        return f"Visual Jsc ({len(result['Jsc_A'])} points) - not exportable to a suite{tail}"
    if kind == 'retention':
        return f"Retention ({len(result['conductance_S'])} time points){tail}"
    if kind == 'pulse':
        mode = p.get('mode', 'pulse-read')
        n = len(result['conductance_S'])
        branch = run_branch(result)
        level = f" | {p['stim_level']:g} {p.get('stim_drive_type', 'V')}" \
            if p.get('stim_level') is not None else ""
        wl = f" | {p['wavelength_nm']:g} nm @ {p['intensity_mW_cm2']:g} mW/cm2" \
            if mode in _LIGHT_MODES and p.get('wavelength_nm') is not None \
            and p.get('intensity_mW_cm2') is not None else ""
        return f"{mode} pulse-read ({n} points, net {branch or 'n/a'}){level}{wl}{tail}"
    return f"Unrecognised measurement{tail}"


def detect_wavelength_sweep(runs: List[Dict]):
    """Find a wavelength sweep among stored runs.

    Only optical pulse-read runs qualify (an electrical run's "manual light"
    wavelength is not a stimulus). Results and wavelengths are returned sorted
    jointly by wavelength so the pairing is independent of measurement order.
    One run is kept per wavelength.

    Returns (results, wavelengths, skipped_duplicates); (None, None, skipped)
    when fewer than 3 distinct wavelengths were measured.
    """
    by_wl: Dict[float, Dict] = {}
    skipped = 0
    for r in runs:
        if classify_run(r) != 'pulse':
            continue
        sp = stimulus_params_from_run(r)
        if sp.get('stimulus_type') != 'light' or 'wavelength_nm' not in sp:
            continue
        if sp['wavelength_nm'] in by_wl:
            skipped += 1
        else:
            by_wl[sp['wavelength_nm']] = r
    if len(by_wl) < 3:
        return None, None, skipped
    ordered = sorted(by_wl.items(), key=lambda kv: kv[0])
    return [r for _, r in ordered], [wl for wl, _ in ordered], skipped


# =============================================================================
# DATASET CONVERSION FUNCTIONS
# =============================================================================

def _add_stimulus_metadata(metadata: Dict, sp: Dict, wavelength_key: str) -> None:
    """Copy the stimulus conditions present in `sp` into dataset metadata."""
    # The wavelength is filed under the key matching the branch it drove; a
    # depression run's wavelength under 'wavelength_pot' mislabels the band and
    # the spectral fit cannot detect it. Legacy spellings are accepted so suites
    # from an older build still convert.
    _wl = sp.get('wavelength_nm', sp.get('wavelength_pot', sp.get('wavelength_dep')))
    if _wl is not None:
        metadata[wavelength_key] = _wl
    # Absent stays absent: an electrical run legitimately has no wavelength, and
    # fitting.py reports the absence in its extraction report.

    # Under the unit-bearing key fitting.py reads; the unit-less 'intensity' is
    # accepted on input and kept on output for older readers.
    _intensity = sp.get('light_intensity_mW_cm2', sp.get('intensity'))
    if _intensity is not None:
        metadata['light_intensity_mW_cm2'] = _intensity
        metadata['intensity'] = _intensity
    for key in ('pulse_width_ms', 'pulse_period_ms', 'frequency_Hz', 'n_pulses',
                'read_timing_ms'):
        if key in sp:
            metadata[key] = sp[key]


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
    
    # Build metadata. stimulus_type is recorded only when known: defaulting to
    # 'light' stamped optical provenance on runs that never had any.
    metadata = {
        'x_type': x_type,
        'x_units': x_units,
        'y_type': y_type,
        'y_units': y_units,
        'experiment_type': experiment_type,
        'notes': stimulus_params.get('notes', '')
    }
    if 'stimulus_type' in stimulus_params:
        metadata['stimulus_type'] = stimulus_params['stimulus_type']
    _add_stimulus_metadata(
        metadata, stimulus_params,
        'wavelength_dep' if experiment_type == 'depression' else 'wavelength_pot')

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
                            stimulus_params: Dict,
                            per_result_params: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Convert wavelength sweep results into multiple datasets (one per wavelength).

    Args:
        results_list: List of pulse_read_sequence results, one per wavelength
        wavelengths: Corresponding wavelength values (nm)
        stimulus_params: Shared base parameters; each run's own recorded
            parameters (intensity, pulse width, ...) take precedence, so a
            sweep measured at several intensities keeps each one.
        per_result_params: Optional explicit overrides, one dict per result.

    Returns:
        list: List of datasets, one per wavelength
    """
    if len(results_list) != len(wavelengths):
        raise ValueError("Number of results must match number of wavelengths")
    if per_result_params is not None and len(per_result_params) != len(results_list):
        raise ValueError("per_result_params must have one entry per result")

    datasets = []
    for i, (result, wl) in enumerate(zip(results_list, wavelengths)):
        params = _resolve_params(stimulus_params, result,
                                 per_result_params[i] if per_result_params else None)
        params['wavelength_nm'] = wl

        dataset = convert_pulse_sequence_to_dataset(
            result,
            'potentiation',
            params,
            f"wavelength_{wl}nm"
        )

        datasets.append(dataset)

    return datasets


def _cycle_phase_params(shared: Dict, cfg: Optional[Dict], phase_data: Dict,
                        short: str, notes: str) -> Dict:
    """Stimulus metadata for one cycle phase (short = 'pot' or 'dep').

    The train's own config is the source: its real pulse width and pulse count,
    an electrical stimulus type (a cycle train is a voltage write pulse), and no
    intensity unless one was actually recorded. The legacy shared keys
    (`wavelength_pot`, `intensity_dep`, ...) are honoured when the caller
    supplies them; nothing is invented when they are absent.
    """
    out: Dict = {'notes': notes}
    for src, dst in (('stimulus_type', 'stimulus_type'),
                     (f'wavelength_{short}', 'wavelength_nm'),
                     (f'intensity_{short}', 'light_intensity_mW_cm2'),
                     ('pulse_width_ms', 'pulse_width_ms')):
        if shared.get(src) is not None:
            out[dst] = shared[src]
    if cfg:
        out.update(stimulus_params_from_params(cfg, stimulus_type='electrical'))
    if 'n_pulses' not in out:
        # pulse_number index 0 is the pre-stimulus baseline, so the pulse count
        # is one less than the number of points.
        pn = phase_data.get('pulse_number')
        if pn is not None and len(pn) and pn[0] == 0:
            out['n_pulses'] = len(pn) - 1
    return out


def convert_cycle_to_datasets(cycle_results: Dict, stimulus_params: Dict) -> List[Dict]:
    """
    Convert potentiation-depression cycling data into separate datasets.

    Args:
        cycle_results: Output from synapse_cycle.cycle_sequence(); its
            `pot_params` / `dep_params` (Train A / Train B configs) supply each
            phase's real pulse width, pulse count, period and stimulus type.
        stimulus_params: Optional shared values (`wavelength_pot`,
            `intensity_dep`, ...) for results that carry no train config.

    Returns:
        list: List of datasets (separate for each potentiation and depression phase)
    """
    datasets = []
    shared = stimulus_params or {}

    for cycle_idx, cycle in enumerate(cycle_results['cycles']):
        for phase, short in (('potentiation', 'pot'), ('depression', 'dep')):
            phase_data = cycle.get(phase)
            if phase_data is None:
                continue
            # No wavelength or intensity fallback of any kind. An unrecorded
            # one is omitted, and the depression run is never labelled with the
            # potentiation band (the spectral fit cannot detect that).
            cfg = cycle_results.get(f'{short}_params') or phase_data.get('params')
            params = _cycle_phase_params(
                shared, cfg, phase_data, short,
                f'{phase.capitalize()} cycle {cycle_idx + 1}')
            dataset = convert_pulse_sequence_to_dataset(
                phase_data, phase, params, f"cycle{cycle_idx + 1}_{phase}")
            dataset['metadata']['cycle_index'] = cycle_idx + 1
            dataset['metadata']['cycle_phase'] = phase
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
    _add_stimulus_metadata(metadata, stimulus_params, 'wavelength_pot')

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
    _add_stimulus_metadata(metadata, stimulus_params, 'wavelength_pot')

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
        retention_result: Output from measure_retention(): needs 'time_s' and
            'conductance_S'.
        stimulus_params: Dict with measurement details

    Returns:
        dict: Dataset in data_converter.py format
    """
    params = stimulus_params.copy()
    params['notes'] = params.get('notes', 'Retention/decay measurement')

    # An explicit check, not a pass-through: a result without these arrays used
    # to be handed on to the pulse converter, which then guessed an x-axis.
    for key in ('time_s', 'conductance_S'):
        if key not in retention_result:
            raise ValueError(
                f"Retention measurement has no '{key}' array. "
                f"Keys present: {', '.join(sorted(retention_result))}"
            )
    result = {'time_s': retention_result['time_s'],
              'conductance_S': retention_result['conductance_S']}

    return convert_pulse_sequence_to_dataset(
        result, 'retention', params, 'retention_measurement')


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
    metadata: Optional[Dict] = None,
    result_params: Optional[Dict] = None
) -> Dict:
    """
    Build complete characterization suite from raw Keithley measurement results.
    Now generates JSON format compatible with data_converter.py output.

    Each dataset carries the metadata of the run it came from (stimulus type,
    wavelength, intensity, pulse width, period, n_pulses), read from that run's
    own recorded parameters. `stimulus_params` is only a shared base for keys a
    run does not carry; nothing is defaulted when it is None.

    Args:
        wavelength_results: List of results at different wavelengths
        wavelengths: Corresponding wavelength values (nm); when omitted, taken
            from each result's own recorded wavelength (an error if absent)
        ltp_result: Single LTP measurement for nonlinearity
        ltd_result: Single LTD measurement (optional)
        cycle_results: Potentiation-depression cycling data
        retention_result: Decay measurement
        stdp_result: STDP timing-window sweep (for the STDP window fit)
        srdp_result: SRDP frequency sweep (for the SRDP sigmoid fit)
        stimulus_params: Shared base conditions (lowest precedence)
        metadata: Additional information (sample_id, date, operator, etc.)
        result_params: Explicit per-result overrides (highest precedence), for
            results that carry no parameters of their own. Keys 'ltp', 'ltd',
            'retention', 'stdp', 'srdp' map to a dict; 'wavelength' maps to a
            list of dicts, one per wavelength result.

    Returns:
        dict: Complete characterization_suite in data_converter.py format
    """
    datasets = []
    shared = dict(stimulus_params or {})
    explicit = result_params or {}

    # Convert wavelength sweep
    if wavelength_results is not None:
        if wavelengths is None:
            wavelengths = []
            for i, r in enumerate(wavelength_results):
                wl = _resolve_params(None, r, (explicit.get('wavelength') or [None] * (i + 1))[i]
                                     ).get('wavelength_nm')
                if wl is None:
                    raise ValueError(
                        f"Wavelength result {i} records no wavelength_nm and no "
                        "`wavelengths` list was given.")
                wavelengths.append(wl)
        datasets.extend(convert_wavelength_sweep(
            wavelength_results, wavelengths, shared, explicit.get('wavelength')))

    # Convert LTP (nonlinearity)
    if ltp_result is not None:
        params = _resolve_params(shared, ltp_result, explicit.get('ltp'))
        params['notes'] = 'Long-term potentiation for nonlinearity extraction'
        datasets.append(convert_pulse_sequence_to_dataset(
            ltp_result, 'potentiation', params, 'ltp_nonlinearity'))

    # Convert LTD
    if ltd_result is not None:
        params = _resolve_params(shared, ltd_result, explicit.get('ltd'))
        params['notes'] = 'Long-term depression'
        datasets.append(convert_pulse_sequence_to_dataset(
            ltd_result, 'depression', params, 'ltd_measurement'))

    # Convert cycling data
    if cycle_results is not None:
        datasets.extend(convert_cycle_to_datasets(cycle_results, shared))

    # Convert retention
    if retention_result is not None:
        datasets.append(convert_retention_to_dataset(
            retention_result,
            _resolve_params(shared, retention_result, explicit.get('retention'))))

    # Convert STDP timing window
    if stdp_result is not None:
        params = _resolve_params(shared, stdp_result, explicit.get('stdp'))
        params['notes'] = 'STDP timing window'
        datasets.append(convert_stdp_to_dataset(stdp_result, params))

    # Convert SRDP frequency response
    if srdp_result is not None:
        params = _resolve_params(shared, srdp_result, explicit.get('srdp'))
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
# PURE PARAMETER VALIDATION (GUI-free, so it can be tested without Tk)
# =============================================================================
#
# Each raises ValueError with a message the GUI shows verbatim, and each runs
# before any instrument is opened.

def require_count(value: float, label: str, minimum: int = 1,
                  maximum: Optional[int] = None, limit_name: str = "") -> int:
    """A whole-number count within [minimum, maximum].

    `int(3.7)` silently truncated a typed 3.7 to 3 pulses, and a count of 0 or
    below "completed" with empty arrays (SRDP) or raised a bare IndexError.
    """
    if not float(value).is_integer():
        raise ValueError(
            f"{label}: {value:g} is not a whole number. Enter an integer "
            f"(the value would otherwise be silently truncated to {int(value)}).")
    n = int(value)
    if n < minimum:
        raise ValueError(f"{label}: {n} is below the minimum of {minimum}.")
    if maximum is not None and n > maximum:
        raise ValueError(
            f"{label}: {n} exceeds the maximum of {maximum}"
            + (f" ({limit_name})." if limit_name else "."))
    return n


def check_stdp_plan(delta_t_ms, pulse_width_ms: float, pair_period_ms: float) -> None:
    """Validate a whole STDP sweep before any hardware activity.

    Delta_t is onset-to-onset, so every |Delta_t| must be at least one pulse
    width (otherwise the spikes would overlap, which the sequential generator
    cannot produce and which used to fail mid-sweep after hardware activity);
    Delta_t = 0 has no pre/post order; duplicate points add no information; and
    the pair period (an absolute period on the on-instrument path) must contain
    the longest pair, |Delta_t|max + one pulse width.
    """
    dt = np.asarray(delta_t_ms, dtype=float)
    if dt.size < 1 or not np.all(np.isfinite(dt)):
        raise ValueError("STDP: the delta-t list is empty or contains non-finite values.")
    problems = []
    if np.any(dt == 0):
        problems.append(
            "delta-t = 0 ms has no pre/post order; choose a point count that "
            "does not place a point at 0 (e.g. an even count for a symmetric range).")
    if np.unique(dt).size != dt.size:
        problems.append(
            "the delta-t list contains duplicate points (start equals end, or "
            "repeated values); each timing must be measured once.")
    too_short = sorted({float(round(abs(x), 4)) for x in dt if 0 < abs(x) < pulse_width_ms})
    if too_short:
        problems.append(
            f"|delta-t| = {', '.join(f'{x:g}' for x in too_short[:4])} ms is shorter than "
            f"the pulse width ({pulse_width_ms:g} ms). Delta-t is onset-to-onset, so the "
            f"spikes would overlap. Use |delta-t| >= {pulse_width_ms:g} ms, or a shorter pulse.")
    if pair_period_ms < pulse_width_ms:
        problems.append(
            f"the pair period ({pair_period_ms:g} ms) is shorter than the pulse width "
            f"({pulse_width_ms:g} ms).")
    longest = float(np.max(np.abs(dt))) + pulse_width_ms
    if pair_period_ms < longest:
        problems.append(
            f"the pair period ({pair_period_ms:g} ms) cannot contain the longest pair "
            f"({longest:g} ms = max |delta-t| + one pulse width).")
    if problems:
        raise ValueError("STDP sweep is not runnable:\n- " + "\n- ".join(problems)
                         + "\nNothing was measured.")


def check_srdp_range(freq_start_hz: float, freq_end_hz: float, n_points: int) -> None:
    """The sweep must rise: end below start was accepted and ran backwards."""
    if freq_end_hz < freq_start_hz:
        raise ValueError(
            f"SRDP: end frequency ({freq_end_hz:g} Hz) is below the start frequency "
            f"({freq_start_hz:g} Hz). Enter the lower frequency first.")
    if freq_end_hz == freq_start_hz and n_points > 1:
        raise ValueError(
            f"SRDP: start and end are both {freq_start_hz:g} Hz, so all {n_points} "
            "points would be the same frequency.")


def check_visual_window(pulse_width_ms: float, start_delay_ms: float,
                        end_margin_ms: float, nplc: float, line_freq_hz: float) -> None:
    """The ADC integration must complete inside the light pulse.

    Optical dose is intensity x pulse width only if the whole integration
    (nplc / line frequency) happens while the LED is on, so a window shorter
    than one integration is a configuration error, not something to run.
    """
    window = pulse_width_ms - start_delay_ms - end_margin_ms
    if window <= 0:
        raise ValueError(
            f"Invalid measurement window: pulse width ({pulse_width_ms:g} ms) must "
            f"exceed start delay ({start_delay_ms:g} ms) + end margin ({end_margin_ms:g} ms).")
    integration = nplc * 1000.0 / line_freq_hz
    if integration > window:
        raise ValueError(
            f"One ADC integration ({integration:.2f} ms at NPLC={nplc:g}, "
            f"{line_freq_hz:g} Hz) does not fit the measurement window "
            f"({window:.2f} ms = pulse width {pulse_width_ms:g} - start delay "
            f"{start_delay_ms:g} - end margin {end_margin_ms:g}).\n"
            "Increase the pulse width, lower NPLC, or shorten the start delay / end margin.")


def retention_waits_from_times(times_s) -> List[float]:
    """Waits before each reading from absolute sample times (s since start)."""
    t = [float(x) for x in times_s]
    if not t or t[0] <= 0 or any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError(
            "Retention sample times must be positive and strictly increasing "
            f"(got {t}).")
    return [t[0]] + [b - a for a, b in zip(t, t[1:])]


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
        
        # Judged from the wavelengths the datasets actually carry, not from a
        # free-text "notes" field that nothing guarantees mentions one.
        distinct_wl = {
            ds['metadata'].get('wavelength_pot', ds['metadata'].get('wavelength_dep'))
            for ds in datasets
            if ds['metadata'].get('experiment_type') in ('potentiation', 'depression')
        } - {None}
        if len(distinct_wl) < 3:
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
