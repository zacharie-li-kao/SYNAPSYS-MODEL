"""
fitting.py (v3.0) - Unified Model Fitting
==========================================

Standardized to use ONLY v2.0 data format (data_converter.py JSON with 'datasets' array).
Flexible fitting that works with incomplete characterization datasets.

Key features:
- Works with minimal data (just dynamic range) up to complete characterization
- Clear feedback about what parameters can/cannot be extracted
- Graceful degradation with defaults for missing measurements
"""

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA PARSER - v2.0 Format Only
# =============================================================================

# =============================================================================
# UNIT CONTRACT (audit M11)
# =============================================================================
#
# `data_converter.py` collects `x_units`, `y_units` and `y_type` for every
# dataset and writes them into the v2.0 JSON. `fitting.py` contained ZERO
# references to any of them: every axis was assumed to be in SI already.
#
# The consequence is silent and severe. `decay_tau` is fitted in whatever unit
# the time axis happens to carry, then reported in seconds and compared against
# a 100 s default — so a retention curve recorded in milliseconds yielded a tau
# a THOUSAND times wrong, with no warning anywhere and an excellent R², because
# the fit itself is perfectly good in the wrong unit.
#
# The units fields are free text in the GUI (placeholder "e.g., s, ms, Hz, #"),
# so the vocabulary below is deliberately generous about spelling. An
# unrecognised unit RAISES rather than being assumed to be SI: guessing is
# exactly what caused the problem.

_TIME_UNITS_TO_SECONDS = {
    's': 1.0, 'sec': 1.0, 'secs': 1.0, 'second': 1.0, 'seconds': 1.0,
    'ms': 1e-3, 'msec': 1e-3, 'millisecond': 1e-3, 'milliseconds': 1e-3,
    'us': 1e-6, 'µs': 1e-6, 'μs': 1e-6, 'usec': 1e-6, 'microsecond': 1e-6,
    'microseconds': 1e-6,
    'ns': 1e-9, 'nanosecond': 1e-9, 'nanoseconds': 1e-9,
    'min': 60.0, 'minute': 60.0, 'minutes': 60.0,
    'h': 3600.0, 'hr': 3600.0, 'hour': 3600.0, 'hours': 3600.0,
}

_CONDUCTANCE_UNITS_TO_SIEMENS = {
    's': 1.0, 'siemens': 1.0, 'siemen': 1.0,
    'ms': 1e-3, 'millisiemens': 1e-3,      # NOTE: ambiguous with milliseconds
    'us': 1e-6, 'µs': 1e-6, 'μs': 1e-6, 'microsiemens': 1e-6,
    'ns': 1e-9, 'nanosiemens': 1e-9,
    'ps': 1e-12, 'picosiemens': 1e-12,
    'mho': 1.0, 'mhos': 1.0,
}

_DIMENSIONLESS_UNITS = {'#', '', 'none', 'n/a', 'count', 'counts', 'index',
                        'pulse', 'pulses', 'pulse_number', 'a.u.', 'au',
                        'arb', 'arb.', 'arbitrary'}


def convert_axis_to_si(values, metadata, axis, expected_quantity):
    """Convert one axis to SI using the units the dataset declares.

    Args:
        values: the raw axis data.
        metadata: the dataset's metadata dict (carries `x_units` / `y_units`).
        axis: 'x' or 'y'.
        expected_quantity: 'time' or 'conductance' — what this extractor needs
            the axis to be.

    Returns:
        (numpy array in SI, note) where `note` describes the conversion applied,
        or None when the axis was already SI.

    Raises:
        ValueError on a unit that cannot be interpreted. Assuming SI is what
        produced a 1000x error in decay_tau without warning.
    """
    array = np.asarray(values, dtype=float)
    raw_unit = (metadata or {}).get(f'{axis}_units')

    if raw_unit is None:
        # No declaration at all — legacy data. SI is assumed, as before, but
        # the assumption is now recorded rather than invisible.
        return array, (
            f"{axis}-axis has no declared units; assumed SI "
            f"({'s' if expected_quantity == 'time' else 'S'})"
        )

    unit = str(raw_unit).strip()
    key = unit.lower()

    if key in _DIMENSIONLESS_UNITS:
        if expected_quantity == 'time':
            raise ValueError(
                f"This extractor needs a TIME axis, but the dataset declares "
                f"{axis}_units = {unit!r}, which is dimensionless. A pulse "
                "index cannot be fitted as a decay time."
            )
        return array, None

    table = (_TIME_UNITS_TO_SECONDS if expected_quantity == 'time'
             else _CONDUCTANCE_UNITS_TO_SIEMENS)

    if key not in table:
        raise ValueError(
            f"Unrecognised {axis}_units {unit!r} for a {expected_quantity} "
            f"axis. Known: {', '.join(sorted(set(table)))}. The unit is not "
            "assumed to be SI, because assuming it is exactly what made a "
            "millisecond retention axis produce a decay_tau 1000x wrong."
        )

    # 'ms' and 'ns' mean different things on a time axis and a conductance
    # axis, and both tables contain them. `expected_quantity` disambiguates,
    # which is why it is a required argument rather than inferred.
    factor = table[key]
    if factor == 1.0:
        return array, None

    si_name = 's' if expected_quantity == 'time' else 'S'
    return array * factor, (
        f"{axis}-axis converted from {unit} to {si_name} (x{factor:g})"
    )


class DataParser:
    """Parse characterization data from v2.0 format (data_converter.py JSON)."""
    
    @staticmethod
    def parse_suite(suite: Dict) -> Dict:
        """
        Parse v2.0 suite into standardized internal format for fitting.
        Also accepts legacy format from SyntheticCharacterization for backward compatibility.
        
        Args:
            suite: JSON with 'datasets' array (v2.0) OR legacy dict with direct keys
        
        Returns:
            dict: {
                'dynamic_range': {...},
                'wavelength_response': {...},
                'nonlinearity': {...},
                'retention': {...},
                'potentiation_data': [...],
                'depression_data': [...]
            }
        """
        # Check if it's v2.0 format (from data_converter.py/assembly_helpers.py)
        if 'datasets' in suite:
            return DataParser._parse_v2(suite)
        
        # Check if it's legacy format from SyntheticCharacterization (for Main.py compatibility)
        elif 'wavelength_response' in suite or 'dynamic_range' in suite:
            return DataParser._parse_legacy_synthetic(suite)
        
        # Unknown format
        else:
            raise ValueError(
                "ERROR: Invalid data format.\n"
                "Expected either:\n"
                "  - v2.0 format with 'datasets' array (from data_converter.py)\n"
                "  - Legacy synthetic format with direct keys (from Main.py SyntheticCharacterization)\n"
                f"Received keys: {list(suite.keys())}"
            )
    
    @staticmethod
    def _parse_legacy_synthetic(suite: Dict) -> Dict:
        """Parse legacy format from SyntheticCharacterization (for Main.py compatibility)."""
        parsed = {}
        
        # Direct copy of sections
        if 'dynamic_range' in suite:
            parsed['dynamic_range'] = suite['dynamic_range']
        
        if 'wavelength_response' in suite:
            parsed['wavelength_response'] = suite['wavelength_response']
        
        if 'nonlinearity' in suite:
            parsed['nonlinearity'] = suite['nonlinearity']
        
        if 'retention' in suite:
            parsed['retention'] = suite['retention']
        
        return parsed
    
    @staticmethod
    def _parse_v2(suite: Dict) -> Dict:
        """Parse v2.0 format (data_converter.py output)."""
        parsed = {
            'potentiation_data': [],
            'depression_data': [],
            'retention_data': [],
            'wavelength_data': [],
            'stdp_data': [],
            'srdp_data': []
        }
        
        datasets = suite.get('datasets', [])
        
        for ds in datasets:
            exp_type = ds['metadata'].get('experiment_type', '').lower()
            
            # Collect potentiation data
            if exp_type == 'potentiation' or exp_type == 'potentiation_depression_cycle':
                parsed['potentiation_data'].append(ds)
            
            # Collect depression data  
            if exp_type == 'depression' or exp_type == 'potentiation_depression_cycle':
                parsed['depression_data'].append(ds)
            
            # Collect retention data
            if exp_type == 'retention':
                parsed['retention_data'].append(ds)
            
            # Collect STDP data
            if exp_type == 'stdp':
                parsed['stdp_data'].append(ds)
            
            # Collect SRDP data
            if exp_type == 'srdp':
                parsed['srdp_data'].append(ds)
        
        # Extract dynamic range from potentiation/depression data
        if parsed['potentiation_data'] or parsed['depression_data']:
            parsed['dynamic_range'] = DataParser._extract_dynamic_range(
                parsed['potentiation_data'] + parsed['depression_data']
            )
        
        # Extract wavelength response from both potentiation AND depression data with varying wavelengths
        all_wavelength_data = parsed['potentiation_data'] + parsed['depression_data']
        wavelength_datasets = DataParser._group_by_wavelength(all_wavelength_data)
        if len(wavelength_datasets) >= 3:
            parsed['wavelength_response'] = DataParser._build_wavelength_response(wavelength_datasets)
        
        # Extract nonlinearity from the longest SINGLE-PHASE potentiation
        # dataset.
        #
        # Cycling traces are filed into potentiation_data above (a cycle does
        # contain potentiation), and simply taking the longest meant a
        # 300-point cycling trace beat a 101-point dedicated nonlinearity
        # trace. _build_nonlinearity then sampled ACROSS the
        # potentiation/depression phase boundary, where ΔG changes sign, and
        # survived the `len < 5` guard by exactly one point. alpha came out
        # plausible but was fitted from data that is not a potentiation curve.
        #
        # A dedicated measurement is always preferred; a cycling trace is used
        # only when there is nothing else, and that is recorded.
        pot_candidates = [ds for ds in parsed['potentiation_data']
                          if len(ds['y_data']) >= 10]
        if pot_candidates:
            def _is_single_phase(ds):
                md = ds.get('metadata', {})
                exp_type = str(md.get('experiment_type', '')).lower()
                return (exp_type != 'potentiation_depression_cycle'
                        and not md.get('from_cycling_measurement'))

            single_phase = [ds for ds in pot_candidates if _is_single_phase(ds)]
            preferred = single_phase or pot_candidates

            longest_pot = max(preferred, key=lambda ds: len(ds['y_data']))
            # Returns None when too few samples survive selection — the caller
            # treats a missing nonlinearity as "not measured", which is correct.
            nonlinearity = DataParser._build_nonlinearity(longest_pot)
            if nonlinearity is not None:
                nonlinearity['source_filename'] = longest_pot.get('filename')
                nonlinearity['source_is_cycling_trace'] = not single_phase
                parsed['nonlinearity'] = nonlinearity
        
        # Extract depression nonlinearity from longest PURE depression dataset (not cycles)
        if parsed['depression_data']:
            # Filter to only pure depression experiments (exclude cycles)
            pure_depression = [ds for ds in parsed['depression_data'] 
                             if ds['metadata'].get('experiment_type') == 'depression']
            if pure_depression:
                longest_dep = max(pure_depression, key=lambda ds: len(ds['y_data']))
                if len(longest_dep['y_data']) >= 20:
                    parsed['depression_nonlinearity'] = DataParser._build_depression_nonlinearity(longest_dep)
        
        # Extract retention from retention data
        if parsed['retention_data']:
            # Use first retention measurement
            parsed['retention'] = DataParser._build_retention(parsed['retention_data'][0])
        
        # Extract STDP from STDP data. The dynamic range is passed in because
        # the STDP amplitude is defined against it, not against the operating
        # point — see _build_stdp.
        if parsed['stdp_data']:
            parsed['stdp'] = DataParser._build_stdp(
                parsed['stdp_data'][0], parsed.get('dynamic_range')
            )
        
        # Extract SRDP from SRDP data. As with STDP, the amplitude is defined
        # against the dynamic range rather than the operating point.
        if parsed['srdp_data']:
            parsed['srdp'] = DataParser._build_srdp(
                parsed['srdp_data'][0], parsed.get('dynamic_range')
            )
        
        return parsed
    
    @staticmethod
    def _extract_dynamic_range(datasets: List[Dict]) -> Dict:
        """Extract G_min and G_max from multiple datasets.

        Uses a ROBUST plateau estimate per trace rather than its extreme value.

        A saturating trace spends its last many pulses near the bound, so the
        bound is best read off that plateau. Taking the raw maximum instead
        reads off the largest NOISE EXCURSION in the plateau: with 1.5%
        multiplicative noise over ~50 plateau samples, the expected maximum
        sits a few percent above the true asymptote, and taking the maximum
        again across datasets compounds it. Measured on the synthetic suite:
        G_max came out 2.4% high, which propagated into an alpha 8% high and,
        because the power-law base is ~1e-4, an A_peak 83% high.

        The median of the top decile is used instead — an estimator of the
        plateau LEVEL rather than of its largest fluctuation.
        """
        G_min = float('inf')
        G_max = float('-inf')

        for ds in datasets:
            y_data = np.asarray(ds.get('y_data', []), dtype=float)
            metrics = ds.get('derived_metrics', {})

            if y_data.size >= 10:
                y_sorted = np.sort(y_data)
                n_tail = max(2, int(round(0.1 * y_sorted.size)))

                # The plateau estimator applies only at the end the trace is
                # actually saturating towards. A rising trace has a plateau at
                # its top and a rising START at its bottom — the bottom decile
                # there is the early part of the climb, not a bound, and
                # averaging it overstates G_min. The other end keeps the raw
                # extreme, which is the best available statement about how far
                # the device was driven.
                rising = y_data[-1] >= y_data[0]
                if rising:
                    ds_max = float(np.median(y_sorted[-n_tail:]))
                    ds_min = float(np.min(y_data))
                else:
                    ds_max = float(np.max(y_data))
                    ds_min = float(np.median(y_sorted[:n_tail]))
            else:
                # Too short for a plateau to be meaningful; fall back to the
                # per-dataset metrics the converter already computed.
                ds_max = metrics.get('G_max')
                ds_min = metrics.get('G_min')

            if ds_min is not None:
                G_min = min(G_min, ds_min)
            if ds_max is not None:
                G_max = max(G_max, ds_max)

        if G_min == float('inf') or G_max == float('-inf'):
            return None
        
        return {
            'G_min_S': G_min,
            'G_max_S': G_max,
            'dynamic_range_S': G_max - G_min,
            'on_off_ratio': G_max / G_min if G_min > 0 else None,
            'measurement_type': 'dynamic_range'
        }
    
    @staticmethod
    def _group_by_wavelength(datasets: List[Dict]) -> Dict[float, Dict]:
        """
        Group datasets by wavelength for wavelength sweep fitting.
        Includes both potentiation AND depression experiments.
        If multiple datasets exist for same wavelength, keeps the one with most pulses.
        """
        wavelength_groups = {}
        
        for ds in datasets:
            # Check both potentiation and depression wavelength fields
            wl_pot = ds['metadata'].get('wavelength_pot')
            wl_dep = ds['metadata'].get('wavelength_dep')
            exp_type = ds['metadata'].get('experiment_type', '')
            
            # Determine which wavelength to use
            # For single-wavelength experiments (pure potentiation or depression),
            # the wavelength is typically stored in wavelength_pot
            # For cycles, wavelength_pot is for pot phase, wavelength_dep is for dep phase
            wl = None
            if exp_type == 'potentiation':
                # Pure potentiation: use wavelength_pot
                wl = wl_pot
            elif exp_type == 'depression':
                # Pure depression: try wavelength_dep first, then wavelength_pot
                # (some experiments put the stimulus wavelength in wavelength_pot even for depression)
                wl = wl_dep if wl_dep is not None else wl_pot
            # Skip cycles (potentiation_depression_cycle) as they may have net zero delta_G
            
            if wl is None:
                continue  # Skip experiments without wavelength info
            
            # If wavelength already exists, keep the dataset with more data points
            if wl in wavelength_groups:
                if len(ds['y_data']) > len(wavelength_groups[wl]['y_data']):
                    wavelength_groups[wl] = ds
            else:
                wavelength_groups[wl] = ds
        
        return wavelength_groups
    
    @staticmethod
    def _initial_rate(y_data) -> float:
        """Initial rate of conductance change per pulse, dG/dn at n = 0.

        Spectral responsivity lives in the INITIAL RATE, not in the endpoint
        after a fixed number of pulses (audit C11). The soft-bound update
        saturates: dG per pulse falls to zero as G approaches its bound, so by
        pulse 50 a strongly-driven wavelength and a very strongly-driven one
        have both arrived at the same plateau and are indistinguishable. On the
        synthetic suite the measured fraction of range remaining at 540 nm was
        0.0004 — the "spectral peak" was a clipped plateau reading
        -0.98/-1.00/-1.00/-0.98 at 520/540/560/580 nm, and the Gaussian fitted
        to it came out 48.9 nm wide against a true 30 nm. R² could not detect
        this, because the plateau was being fitted faithfully.

        The rate is estimated by least squares over the leading portion of the
        trace, restricted to where the response is still within 20% of its
        total excursion — i.e. where it is still approximately linear and has
        not yet begun to saturate.
        """
        y = np.asarray(y_data, dtype=float)
        if len(y) < 3:
            return 0.0

        g0 = y[0]
        total_excursion = y[-1] - g0
        if not np.isfinite(total_excursion) or total_excursion == 0:
            return 0.0

        # Leading window: points still within 20% of the full excursion.
        within = np.flatnonzero(np.abs(y - g0) <= abs(total_excursion) * 0.2)
        n_window = int(within[-1]) + 1 if len(within) else 3
        # At least 3 points to fit a slope through noise; at most 15, beyond
        # which even the linear portion starts to bend.
        n_window = int(np.clip(n_window, 3, min(15, len(y))))

        pulses = np.arange(n_window, dtype=float)
        slope = float(np.polyfit(pulses, y[:n_window], 1)[0])
        return slope

    @staticmethod
    def _build_wavelength_response(wavelength_datasets: Dict[float, Dict]) -> Dict:
        """
        Build wavelength_response dict from wavelength-grouped datasets.

        The response at each wavelength is the INITIAL RATE dG/dn|₀ — see
        _initial_rate for why the previous endpoint-at-fixed-pulse-count
        measurement could not recover the spectral width.

        Normalisation is PER BRANCH, as CLAUDE.md specifies: the strongest
        potentiation response becomes +1 and the strongest depression response
        becomes -1, independently. A single global maximum (the previous
        behaviour) squashes the weaker branch and, combined with the Gaussian
        fit's freedom, produced stored peak amplitudes of -1.15 and +0.66 —
        outside the [-1, +1] the convention promises.
        """

        wavelengths = sorted(wavelength_datasets.keys())

        raw_responses = {}
        for wl in wavelengths:
            ds = wavelength_datasets[wl]
            metrics = ds['derived_metrics']
            md = ds.get('metadata', {})
            y_data = ds.get('y_data', [])

            rate = DataParser._initial_rate(y_data)

            # Divide out the drive, so what remains is a RESPONSIVITY — rate
            # per unit intensity per unit pulse width — rather than a raw rate.
            #
            # The initial rate is proportional to A_peak · S(λ) · I · dt. If the
            # sweep points were not all measured at the same intensity and
            # pulse width, pooling raw rates yields a curve of S(λ)·I(λ), not
            # S(λ), and the spectral shape is corrupted by the drive schedule.
            # Nothing warns about this because the fit still succeeds — it just
            # fits the wrong function.
            intensity = md.get('light_intensity_mW_cm2')
            pulse_width_ms = md.get('pulse_width_ms')
            drive = 1.0
            if intensity:
                drive *= float(intensity)
            if pulse_width_ms:
                drive *= float(pulse_width_ms) / 1000.0
            if drive > 0:
                rate = rate / drive

            raw_responses[wl] = {
                'rate': rate,
                'drive': drive,
                'metrics': metrics,
                'dataset': ds
            }

        rates = np.array([raw_responses[wl]['rate'] for wl in wavelengths], dtype=float)

        # Per-branch normalisation constants.
        positive = rates[rates > 0]
        negative = rates[rates < 0]
        max_positive = float(np.max(positive)) if positive.size else 0.0
        max_negative = float(np.abs(np.min(negative))) if negative.size else 0.0

        response = {
            'wavelengths_nm': wavelengths,
            'delta_g_S': [],
            'delta_g_percent': [],
            'delta_g_normalized': [],
            'initial_rate_S_per_pulse': [],
            'g_initial_S': [],
            'g_final_S': [],
            'normalisation': 'per_branch',
            'max_positive_rate': max_positive,
            'max_negative_rate': max_negative,
            # Retained for display code that scales plots by it. It is the
            # larger of the two branch maxima, i.e. the old global constant.
            'max_abs_response': max(max_positive, max_negative),
            'measurement_type': 'wavelength_sweep'
        }

        for wl in wavelengths:
            data = raw_responses[wl]
            rate = data['rate']
            metrics = data['metrics']

            if rate > 0:
                normalized = rate / max_positive if max_positive > 0 else 0.0
            elif rate < 0:
                normalized = rate / max_negative if max_negative > 0 else 0.0
            else:
                normalized = 0.0

            response['initial_rate_S_per_pulse'].append(rate)
            response['delta_g_normalized'].append(normalized)
            response['delta_g_S'].append(metrics.get('delta_G', 0))
            response['delta_g_percent'].append(metrics.get('percent_change', 0))
            response['g_initial_S'].append(metrics.get('G_initial', 0))
            response['g_final_S'].append(metrics.get('G_final', 0))

        return response
    
    @staticmethod
    def _estimate_noise_sigma(y_data) -> float:
        """Estimate per-sample noise from the data's own scatter.

        For a smooth underlying curve, the second difference
        d_i = y_{i-1} − 2y_i + y_{i+1} annihilates any locally linear trend and
        has variance 6σ². The median absolute deviation is used rather than the
        standard deviation so that a few saturated or clipped samples cannot
        inflate the estimate.
        """
        y = np.asarray(y_data, dtype=float)
        if len(y) < 4:
            return 0.0
        d = y[:-2] - 2 * y[1:-1] + y[2:]
        mad = float(np.median(np.abs(d - np.median(d))))
        # 1.4826 converts MAD to a standard deviation for Gaussian noise.
        return mad * 1.4826 / np.sqrt(6.0)

    @staticmethod
    def _build_nonlinearity(dataset: Dict) -> Dict:
        """
        Build nonlinearity dict from a single potentiation dataset.

        Samples the potentiation curve at regular intervals to approximate
        different starting conductance states for alpha fitting.
        """
        y_data = np.array(dataset['y_data'])

        if len(y_data) < 20:
            return None

        # Use larger sample interval to reduce noise correlation
        # Each sample represents: G_initial -> G_after_N_pulses
        sample_interval = max(10, len(y_data) // 10)  # ~10 points, minimum 10 pulses apart

        G_initial_S = []
        delta_g_S = []

        # --- Sample selection (audit M4, and the bias underneath it) ---
        #
        # The old criterion was `delta_G > 0.001 * G_range` — a threshold an
        # order of magnitude BELOW the 1.5% multiplicative noise the data
        # actually carries, so saturated samples survived whenever their noise
        # happened to be positive. That is not merely a threshold that is too
        # low: filtering on delta_G at all conditions on the DEPENDENT variable,
        # which biases the fit upward at any threshold. In the saturated region
        # the true delta is ~0, so the surviving points are exactly those whose
        # noise was positive, and the fitted exponent rises to accommodate them.
        #
        # Selection is now on the INDEPENDENT variable: a sample is used when
        # its starting conductance still has real headroom below the maximum
        # attained, which is a statement about where on the curve the sample
        # sits and says nothing about its measured increment. Nothing is
        # discarded for having an inconveniently small or negative delta.
        sigma = DataParser._estimate_noise_sigma(y_data)
        G_ceiling = float(np.max(y_data))
        # Three sigma of headroom, so (G_max_est − G_init) is meaningfully
        # positive rather than dominated by noise.
        min_headroom = 3.0 * sigma

        for i in range(0, len(y_data) - sample_interval, sample_interval):
            G_init = y_data[i]
            G_final = y_data[i + sample_interval]

            if (G_ceiling - G_init) <= min_headroom:
                continue

            G_initial_S.append(G_init)
            delta_g_S.append(G_final - G_init)

        if len(G_initial_S) < 5:
            return None

        return {
            'G_initial_S': G_initial_S,
            'delta_g_S': delta_g_S,
            'noise_sigma_S': sigma,
            # Needed to subtract the relaxation contribution from each sampled
            # increment — see fit_nonlinearity_exponent.
            'sample_interval_pulses': sample_interval,
            'pulse_width_ms': (dataset.get('metadata') or {}).get('pulse_width_ms'),
            # Pulse PERIOD, so relaxation can be subtracted over elapsed
            # time rather than stimulus-on time (see _relaxation_factor).
            # The Keithley exporter records the period as frequency_Hz.
            'pulse_period_ms': (dataset.get('metadata') or {}).get('pulse_period_ms'),
            'frequency_Hz': (dataset.get('metadata') or {}).get('frequency_Hz'),
            'measurement_type': 'nonlinearity'
        }
    
    @staticmethod
    def _build_depression_nonlinearity(dataset: Dict) -> Dict:
        """
        Build depression nonlinearity dict from a single depression dataset.
        
        Samples the depression curve at regular intervals to approximate
        different starting conductance states for beta fitting.
        """
        y_data = np.array(dataset['y_data'])
        
        if len(y_data) < 20:
            return None
        
        # Use larger sample interval to reduce noise correlation
        sample_interval = max(10, len(y_data) // 10)
        
        G_initial_S = []
        delta_g_S = []

        # Mirror image of the potentiation selection — see _build_nonlinearity
        # for why this conditions on the starting conductance rather than on
        # the measured increment. Here the relevant headroom is the distance
        # ABOVE the floor the trace reached, since the depression rate goes as
        # (G − G_min)^beta.
        sigma = DataParser._estimate_noise_sigma(y_data)
        G_floor = float(np.min(y_data))
        min_headroom = 3.0 * sigma

        for i in range(0, len(y_data) - sample_interval, sample_interval):
            G_init = y_data[i]
            G_final = y_data[i + sample_interval]

            if (G_init - G_floor) <= min_headroom:
                continue

            G_initial_S.append(G_init)
            delta_g_S.append(G_final - G_init)

        if len(G_initial_S) < 5:
            return None

        return {
            'G_initial_S': G_initial_S,
            'delta_g_S': delta_g_S,
            'noise_sigma_S': sigma,
            'sample_interval_pulses': sample_interval,
            'pulse_width_ms': (dataset.get('metadata') or {}).get('pulse_width_ms'),
            # Pulse PERIOD, so relaxation can be subtracted over elapsed
            # time rather than stimulus-on time (see _relaxation_factor).
            # The Keithley exporter records the period as frequency_Hz.
            'pulse_period_ms': (dataset.get('metadata') or {}).get('pulse_period_ms'),
            'frequency_Hz': (dataset.get('metadata') or {}).get('frequency_Hz'),
            'measurement_type': 'depression_nonlinearity'
        }

    
    @staticmethod
    def _build_retention(dataset: Dict) -> Dict:
        """Build retention dict from retention dataset.

        The time axis is converted to SECONDS from whatever unit the dataset
        declares — see `convert_axis_to_si`. `decay_tau` is reported in seconds
        and compared against a 100 s default, so a retention curve recorded in
        milliseconds previously produced a tau a THOUSAND times wrong, with
        nothing anywhere warning about it. `data_converter` records `x_units`
        faithfully; `fitting.py` simply never read it.
        """
        times, note = convert_axis_to_si(
            dataset['x_data'], dataset.get('metadata', {}), axis='x',
            expected_quantity='time',
        )
        conductances, y_note = convert_axis_to_si(
            dataset['y_data'], dataset.get('metadata', {}), axis='y',
            expected_quantity='conductance',
        )

        return {
            'time_s': times,
            'conductance_S': conductances,
            'g_initial_S': conductances[0] if len(conductances) else 0,
            'unit_conversions': [n for n in (note, y_note) if n],
            'measurement_type': 'retention'
        }
    
    @staticmethod
    def _build_stdp(dataset: Dict, dynamic_range: Optional[Dict] = None) -> Dict:
        """Build STDP dict from STDP dataset."""
        x_data = np.array(dataset['x_data'])  # delta_t in ms
        y_data = np.array(dataset['y_data'])  # conductance in S
        
        # STDP: G_final = G_initial + A*exp(-|Δt|/τ) * scale
        # Baseline is at large |Δt| where STDP effect vanishes
        
        # Take 20% of points with largest |Δt| as baseline
        abs_delta_t = np.abs(x_data)
        sorted_indices = np.argsort(abs_delta_t)[::-1]
        n_baseline = max(3, len(y_data) // 5)
        baseline_indices = sorted_indices[:n_baseline]
        
        # Median is robust to noise and clipping artifacts
        G_initial = np.median([y_data[i] for i in baseline_indices])

        # Calculate delta_g from baseline
        delta_g_S = [g - G_initial for g in y_data]

        # --- The normalisation denominator (audit C7) ---
        #
        # The learning rule this fit must feed is, per CLAUDE.md:
        #
        #     ΔG = A_plus · exp(−dt/τ_plus) · (G_max − G_min) · 0.01
        #
        # The 0.01 is ALREADY INSIDE THE RULE. So A_plus = 0.5 does not mean
        # "half the dynamic range" — it means 0.5 PERCENT of it. A_plus is
        # numerically a percentage of the dynamic range.
        #
        # Rearranged, the quantity to fit is therefore
        #
        #     A_plus·exp(−dt/τ) = 100 · ΔG / (G_max − G_min)
        #
        # and NEITHER consumer divides by 100 afterwards (network.py's STDP
        # rule and Main.py's two call sites all apply the 0.01 themselves).
        # Reading "fraction of dynamic range" intuitively, and dividing by 100
        # somewhere, lands you exactly 100× off.
        #
        # This used to divide by G_initial — the BASELINE CONDUCTANCE, i.e.
        # the operating point. That is a different denominator entirely
        # (50.4 µS against a 99 µS dynamic range on the synthetic suite), it
        # is operating-point dependent, and no constant rescaling reconciles
        # the two: the fitted amplitude could never equal the ground truth
        # whatever factor was applied.
        dynamic_range_S = None
        if dynamic_range and dynamic_range.get('dynamic_range_S'):
            dynamic_range_S = float(dynamic_range['dynamic_range_S'])

        if dynamic_range_S is None or dynamic_range_S <= 0:
            # No dynamic range means the amplitude cannot be expressed in the
            # units the learning rule requires. Fabricating a denominator here
            # would produce an amplitude that looks like a percentage of range
            # but is not one, which is precisely the defect being fixed.
            raise ValueError(
                "STDP amplitudes are defined as a percentage of the dynamic "
                "range (G_max - G_min), but no dynamic range could be "
                "determined from this suite. Include a potentiation or "
                "depression measurement so G_min and G_max can be extracted."
            )

        delta_g_percent = [(dg / dynamic_range_S) * 100 for dg in delta_g_S]

        return {
            'delta_t_ms': x_data.tolist(),
            'conductance_S': y_data.tolist(),
            'delta_g_S': delta_g_S,
            # Percent OF DYNAMIC RANGE — this is what fit_stdp_window fits and
            # what A_plus/A_minus are denominated in.
            'delta_g_percent': delta_g_percent,
            'normalisation': 'percent_of_dynamic_range',
            'dynamic_range_S': dynamic_range_S,
            'g_initial_S': [G_initial] * len(x_data),
            'g_final_S': y_data.tolist(),
            'measurement_type': 'stdp'
        }
    
    @staticmethod
    def _build_srdp(dataset: Dict, dynamic_range: Optional[Dict] = None) -> Dict:
        """Build SRDP dict from SRDP dataset.

        Normalised by the DYNAMIC RANGE, matching STDP — see _build_stdp for
        why the baseline conductance is the wrong denominator. `max_change` is
        therefore a percentage of (G_max - G_min).

        M7 also noted a 140x unit discontinuity in this field: the FITTED value
        came out in percent (~28.6) while the DEFAULT substituted when no SRDP
        data exists was 0.2, a fraction. Two different units for one field,
        depending on whether the extractor ran. The default is now 20.0, i.e.
        the same 0.2 expressed in the same units as everything else.
        """
        x_data = dataset['x_data']  # frequency in Hz
        y_data = dataset['y_data']  # conductance in S

        # The lowest-frequency point is NOT assumed to be responseless — the
        # sigmoid's lower asymptote is a free parameter of the fit (M8). This
        # baseline only sets the zero of the reported delta.
        G_initial = y_data[0] if len(y_data) > 0 else 1e-6
        delta_g_S = [g - G_initial for g in y_data]

        dynamic_range_S = None
        if dynamic_range and dynamic_range.get('dynamic_range_S'):
            dynamic_range_S = float(dynamic_range['dynamic_range_S'])

        if dynamic_range_S is None or dynamic_range_S <= 0:
            raise ValueError(
                "SRDP amplitudes are expressed as a percentage of the dynamic "
                "range (G_max - G_min), but no dynamic range could be "
                "determined from this suite. Include a potentiation or "
                "depression measurement so G_min and G_max can be extracted."
            )

        delta_g_percent = [(dg / dynamic_range_S) * 100 for dg in delta_g_S]

        return {
            'frequencies_hz': x_data,
            'conductance_S': y_data,
            'delta_g_S': delta_g_S,
            'delta_g_percent': delta_g_percent,
            'normalisation': 'percent_of_dynamic_range',
            'dynamic_range_S': dynamic_range_S,
            'g_initial_S': [G_initial] * len(x_data),
            'g_final_S': y_data,
            'measurement_type': 'srdp'
        }


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def _fit_n_gaussians(wavelengths, delta_g, n_peaks):
    """
    Fit sum of n Gaussians with signed amplitudes.
    
    Physical basis: Each Gaussian represents an absorption band/electronic transition.
    Positive amplitude = potentiation mechanism
    Negative amplitude = depression mechanism
    
    Args:
        wavelengths: Array of wavelength values (nm)
        delta_g: Array of conductance changes (can be positive or negative)
        n_peaks: Number of Gaussians to fit (1-3)
    
    Returns:
        dict with fit parameters and quality metrics
    """
    
    def multi_gaussian(x, *params):
        """Sum of Gaussians: Σ A_i × exp(-((x - λ_i) / w_i)²)"""
        result = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            A = params[i*3]
            peak = params[i*3 + 1]
            width = params[i*3 + 2]
            result += A * np.exp(-((x - peak) / width)**2)
        return result
    
    # Initial guess: intelligently place peaks
    p0 = []
    
    # Strategy: Look for both positive AND negative peaks separately
    positive_indices = np.where(delta_g > 0)[0]
    negative_indices = np.where(delta_g < 0)[0]
    
    used_wavelengths = []
    
    if n_peaks == 1:
        # Single peak: use strongest absolute response
        idx_peak = np.argmax(np.abs(delta_g))
        A_init = delta_g[idx_peak]
        peak_init = wavelengths[idx_peak]
        width_init = 80.0
        p0.extend([A_init, peak_init, width_init])
        used_wavelengths.append(peak_init)
        
    elif n_peaks == 2:
        # Two peaks: try to find one positive and one negative if both exist
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            # Mixed scenario: one pot, one dep
            idx_pos = positive_indices[np.argmax(delta_g[positive_indices])]
            idx_neg = negative_indices[np.argmin(delta_g[negative_indices])]
            
            # Order by wavelength
            if wavelengths[idx_pos] < wavelengths[idx_neg]:
                indices = [idx_pos, idx_neg]
            else:
                indices = [idx_neg, idx_pos]
                
            for idx in indices:
                A_init = delta_g[idx]
                peak_init = wavelengths[idx]
                width_init = 80.0
                p0.extend([A_init, peak_init, width_init])
                used_wavelengths.append(peak_init)
        else:
            # All same sign: use two largest absolute values
            sorted_indices = np.argsort(np.abs(delta_g))[::-1]
            for i in range(2):
                idx = sorted_indices[i]
                A_init = delta_g[idx]
                peak_init = wavelengths[idx]
                width_init = 80.0
                p0.extend([A_init, peak_init, width_init])
                used_wavelengths.append(peak_init)
                
    else:  # n_peaks >= 3
        # Three peaks: sort by absolute magnitude
        sorted_indices = np.argsort(np.abs(delta_g))[::-1]
        for i in range(n_peaks):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                A_init = delta_g[idx]
                peak_init = wavelengths[idx]
            else:
                # Spread additional peaks
                peak_init = wavelengths[len(wavelengths) * i // n_peaks]
                A_init = np.mean(delta_g) / n_peaks
            
            width_init = 80.0
            p0.extend([A_init, peak_init, width_init])
            used_wavelengths.append(peak_init)
    
    # Set physical bounds
    bounds_lower = []
    bounds_upper = []
    for i in range(n_peaks):
        bounds_lower.extend([-np.inf, 300, 20])   # A can be ±, λ ≥ 300nm (UV), w ≥ 20nm
        bounds_upper.extend([np.inf, 900, 250])   # λ ≤ 900nm (near-IR), w ≤ 250nm
    
    try:
        popt, pcov = curve_fit(multi_gaussian, wavelengths, delta_g, 
                              p0=p0, bounds=(bounds_lower, bounds_upper), 
                              maxfev=10000)
        
        # Calculate R²
        fitted = multi_gaussian(wavelengths, *popt)
        ss_res = np.sum((delta_g - fitted)**2)
        ss_tot = np.sum((delta_g - np.mean(delta_g))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Extract parameters for each peak
        peaks_info = []
        for i in range(n_peaks):
            amplitude = popt[i*3]
            wavelength = popt[i*3 + 1]
            width = popt[i*3 + 2]
            
            peaks_info.append({
                'amplitude': float(amplitude),
                'wavelength_nm': float(wavelength),
                'width_nm': float(width),
                'type': 'potentiation' if amplitude > 0 else 'depression'
            })
        
        # Sort peaks by wavelength for clarity
        peaks_info.sort(key=lambda p: p['wavelength_nm'])
        
        return {
            'n_peaks': n_peaks,
            'peaks': peaks_info,
            'fit_quality_R2': float(r_squared),
            'fitted_params': popt.tolist(),
            'success': True
        }
    
    except Exception as e:
        # Fit failed
        return {
            'n_peaks': n_peaks,
            'peaks': [],
            'fit_quality_R2': -1.0,
            'success': False,
            'error': str(e)
        }


def fit_wavelength_sensitivity(wavelength_data, max_gaussians=3, force_n_peaks=None):
    """
    Fit multi-Gaussian spectral response to handle complex wavelength dependence.
    
    Physical basis:
    - Each Gaussian represents an absorption band or electronic transition
    - Positive amplitude = potentiation mechanism (e.g., trap filling)
    - Negative amplitude = depression mechanism (e.g., trap emptying)
    
    Automatically tries 1-3 Gaussians and selects best fit based on:
    1. R² improvement
    2. Akaike Information Criterion (penalizes overfitting)
    
    Args:
        wavelength_data: dict with 'wavelengths_nm' and 'delta_g_percent' or 'delta_g_S'
        max_gaussians: Maximum number of Gaussians to try (1-3)
        force_n_peaks: If specified (1, 2, or 3), force exactly this many Gaussians.
                      Overrides automatic detection. Use for manual override.
    
    Returns:
        dict with fit parameters, including backwards-compatible lambda_peak and lambda_width
    """
    wavelengths = np.array(wavelength_data['wavelengths_nm'])
    
    # Use normalized delta_g if available (accounts for different pulse counts and initial states)
    # Otherwise fall back to delta_g_S or delta_g_percent
    if 'delta_g_normalized' in wavelength_data:
        delta_g = np.array(wavelength_data['delta_g_normalized'])
    elif 'delta_g_S' in wavelength_data:
        delta_g = np.array(wavelength_data['delta_g_S'])
    else:
        delta_g = np.array(wavelength_data['delta_g_percent'])
    
    if len(wavelengths) < 3:
        # Not enough data points for fitting
        return {
            'lambda_peak_nm': float(wavelengths[np.argmax(np.abs(delta_g))]),
            'lambda_width_nm': 100.0,
            'fit_quality_R2': 0.0,
            'fitted_curve': None,
            'n_peaks': 1,
            'peaks': [],
            'fit_type': 'insufficient_data'
        }
    
    # If force_n_peaks specified, only fit that number
    if force_n_peaks is not None:
        if force_n_peaks < 1 or force_n_peaks > 3:
            raise ValueError(f"force_n_peaks must be 1, 2, or 3, got {force_n_peaks}")
        
        # Only fit the requested number of peaks
        result = _fit_n_gaussians(wavelengths, delta_g, force_n_peaks)
        
        if not result['success']:
            # Forced fit failed - return fallback
            print(f"Warning: Forced {force_n_peaks}-peak fit failed, using simple peak detection")
            return {
                'lambda_peak_nm': float(wavelengths[np.argmax(np.abs(delta_g))]),
                'lambda_width_nm': 100.0,
                'fit_quality_R2': 0.0,
                'fitted_curve': None,
                'n_peaks': force_n_peaks,
                'peaks': [],
                'fit_type': 'forced_failed'
            }
        
        best_fit = result
        
    else:
        # Automatic detection: Try fitting 1, 2, and 3 Gaussians
        max_gaussians = min(max_gaussians, len(wavelengths) - 1, 3)  # Can't fit more peaks than data points
        
        fits = []
        for n_peaks in range(1, max_gaussians + 1):
            result = _fit_n_gaussians(wavelengths, delta_g, n_peaks)
            if result['success']:
                # Calculate AIC: penalizes additional parameters
                n_params = n_peaks * 3  # Each Gaussian has 3 parameters
                n_data = len(wavelengths)
                # Prevent log(0) when R² = 1.0
                r2_for_aic = min(result['fit_quality_R2'], 0.9999)
                aic = n_data * np.log(1 - r2_for_aic) + 2 * n_params
                result['aic'] = aic
                fits.append(result)
        
        if not fits:
            # All fits failed - return simple fallback
            print("Warning: All multi-Gaussian fits failed, using simple peak detection")
            return {
                'lambda_peak_nm': float(wavelengths[np.argmax(np.abs(delta_g))]),
                'lambda_width_nm': 100.0,
                'fit_quality_R2': 0.0,
                'fitted_curve': None,
                'n_peaks': 1,
                'peaks': [],
                'fit_type': 'failed'
            }
        
        # Select best fit: prioritize R² improvement, then penalize with AIC
        # Use n=1 if R² is good enough, otherwise allow more peaks if R² improves significantly
        best_fit = fits[0]  # Start with single Gaussian
        
        for fit in fits[1:]:
            # Accept more complex model only if:
            # 1. R² improves by at least 0.10 (significant improvement), OR
            # 2. R² improves by at least 0.05 AND AIC improves (better model selection)
            r2_improvement = fit['fit_quality_R2'] - best_fit['fit_quality_R2']
            
            if r2_improvement > 0.10:
                best_fit = fit
            elif r2_improvement > 0.05 and fit.get('aic', np.inf) < best_fit.get('aic', np.inf):
                best_fit = fit
    
    # Build response curve function
    def response_curve(x):
        """Evaluate the fitted multi-Gaussian at wavelength x"""
        result = np.zeros_like(x, dtype=float)
        for i in range(best_fit['n_peaks']):
            A = best_fit['fitted_params'][i*3]
            peak = best_fit['fitted_params'][i*3 + 1]
            width = best_fit['fitted_params'][i*3 + 2]
            result += A * np.exp(-((x - peak) / width)**2)
        return result
    
    # Identify each branch's own peak.
    #
    # M1: `lambda_peak` and `lambda_width` used to be the largest-|amplitude|
    # peak, which for any device with both a potentiation and a depression band
    # is the DEPRESSION band — reported under names every consumer reads as
    # potentiation parameters. Main.py reads only these top-level scalars, so
    # the potentiation branch was invisible to it even though it survived
    # inside `peaks`.
    #
    # The top-level scalars are now the POTENTIATION peak, matching what their
    # names claim and what VisualSynapse's single-Gaussian fallback needs. Both
    # branches are also reported explicitly, so nothing has to be inferred from
    # a sign convention.
    pot_peaks = [p for p in best_fit['peaks'] if p['amplitude'] > 0]
    dep_peaks = [p for p in best_fit['peaks'] if p['amplitude'] < 0]

    strongest_pot = max(pot_peaks, key=lambda p: p['amplitude']) if pot_peaks else None
    strongest_dep = min(dep_peaks, key=lambda p: p['amplitude']) if dep_peaks else None

    # If there is no potentiation lobe at all, the device only depresses under
    # light; fall back to the depression peak rather than inventing one, and
    # say so in the result.
    primary_peak = strongest_pot or strongest_dep
    primary_branch = 'potentiation' if strongest_pot else 'depression'

    result = {
        'lambda_peak_nm': primary_peak['wavelength_nm'],
        'lambda_width_nm': primary_peak['width_nm'],
        'lambda_peak_branch': primary_branch,
        'fit_quality_R2': best_fit['fit_quality_R2'],
        'fitted_curve': response_curve,
        'n_peaks': best_fit['n_peaks'],
        'peaks': best_fit['peaks'],
        'fit_type': f'{best_fit["n_peaks"]}_gaussian'
    }

    if strongest_pot:
        result['lambda_pot_peak_nm'] = strongest_pot['wavelength_nm']
        result['lambda_pot_width_nm'] = strongest_pot['width_nm']
    if strongest_dep:
        result['lambda_dep_peak_nm'] = strongest_dep['wavelength_nm']
        result['lambda_dep_width_nm'] = strongest_dep['width_nm']

    return result


def _relaxation_factor(data, decay_tau):
    """Fraction of (G − G_min) lost to relaxation over one sampling interval.

    Spontaneous relaxation runs during stimulation (CLAUDE.md), so each sampled
    increment is drive MINUS relaxation. Ignoring that makes the exponent
    absorb it: relaxation grows with G, so the measured increment is suppressed
    more at high G than at low G, the curve looks steeper than it is, and alpha
    comes out high — measured at 0.86 against a true 0.75 before this was
    subtracted, which then pushed A_peak to nearly 3x its true value.

    Returns 0.0 when the information needed is unavailable, in which case the
    fit reduces to the previous drive-only model.
    """
    if not decay_tau or decay_tau <= 0:
        return 0.0
    n_pulses = data.get('sample_interval_pulses')
    if not n_pulses:
        return 0.0

    # ELAPSED time per sampling interval, which is n_pulses x the pulse
    # PERIOD — not the pulse width.
    #
    # Relaxation is always on (CLAUDE.md): it runs during the dead time between
    # pulses just as it does during them. Using the width counted only the
    # stimulus-ON time, so on a duty-cycled measurement the correction was too
    # small by exactly the duty cycle and the missing relaxation was absorbed
    # into alpha. Measured on constructed data: alpha rose from 0.82 to 1.33
    # against a true 0.75 as the duty cycle fell from 1.0 to 0.01.
    #
    # The synthetic generator advances its clock only during pulses (duty = 1),
    # so width and period coincide there and the harness could never see this.
    period_ms = data.get('pulse_period_ms')
    if not period_ms:
        freq_hz = data.get('frequency_Hz')
        if freq_hz:
            try:
                period_ms = 1000.0 / float(freq_hz)
            except (TypeError, ZeroDivisionError):
                period_ms = None
    if not period_ms:
        # Fall back to the pulse width, i.e. assume a 100% duty cycle. This is
        # the old behaviour and it is correct only when the pulses abut. It is
        # recorded in the returned diagnostics rather than applied silently.
        period_ms = data.get('pulse_width_ms')
        if not period_ms:
            return 0.0

    return float(n_pulses) * (float(period_ms) / 1000.0) / float(decay_tau)


def _relaxation_basis(data):
    """Which timing quantity `_relaxation_factor` used, for the report."""
    if data.get('pulse_period_ms'):
        return 'pulse_period_ms'
    if data.get('frequency_Hz'):
        return 'frequency_Hz'
    if data.get('pulse_width_ms'):
        return 'pulse_width_ms (ASSUMED 100% duty cycle - no period recorded)'
    return 'none (relaxation not subtracted)'


def fit_nonlinearity_exponent(nonlinearity_data, G_max=None, G_min=0.0,
                              decay_tau=None):
    """
    Fit power-law nonlinearity: ΔG = A × (G_max - G_initial)^α − relaxation

    Args:
        nonlinearity_data: from DataParser._build_nonlinearity
        G_max: the device's upper soft bound. Pass the suite's dynamic-range
            G_max when available — see below.
        G_min: the device's lower soft bound, for the relaxation term.
        decay_tau: the device's relaxation time constant (s). Pass the fitted
            value so relaxation is subtracted rather than absorbed into alpha.
    """
    G_initial = np.array(nonlinearity_data['G_initial_S'])
    delta_g = np.array(nonlinearity_data['delta_g_S'])
    relax = _relaxation_factor(nonlinearity_data, decay_tau)

    # Largest conductance actually attained.
    #
    # M5: this was np.max(G_initial) + np.max(delta_g), which adds the largest
    # starting conductance of one sample to the largest increment of a
    # DIFFERENT sample — a quantity no single measurement ever reached. The
    # largest conductance actually attained is the largest per-sample sum.
    max_G_achieved = float(np.max(G_initial + delta_g))

    # M6: which G_max to use, and why not to fit it here.
    #
    # The upper soft bound is an ASYMPTOTE: the device approaches it and never
    # arrives, so the largest value any finite pulse train reached always
    # understates it. The old code used (this trace's maximum × 1.05), which
    # forced alpha to absorb the shortfall — and because the base (G_max − G)
    # is ~1e-4, a few percent of exponent error becomes tens of percent in the
    # prefactor A, which is what made A_peak 60% high.
    #
    # Fitting G_max as a third free parameter is worse, not better: on ~10
    # points, A, alpha and G_max are strongly degenerate, and the optimum
    # drifts to a G_max well below the truth (verified: alpha fell to 0.65
    # against a true 0.75). The suite's dynamic-range G_max is a far better
    # estimate because it is drawn from EVERY dataset rather than this one
    # trace, so it is used directly when supplied.
    #
    # alpha is sensitive to it — on the synthetic suite, a G_max 2.4% high
    # moves alpha from 0.77 to 0.81 — so the value used is reported alongside
    # the result rather than left implicit.
    G_max_used = float(G_max) if G_max else max_G_achieved * 1.05
    if G_max_used <= max_G_achieved:
        # A bound below data that was actually measured is not a bound.
        G_max_used = max_G_achieved * 1.0001

    def power_law(G_init, A, alpha):
        drive = A * np.maximum(G_max_used - G_init, 1e-30) ** alpha
        return drive - relax * np.maximum(G_init - G_min, 0.0)

    alpha_start = 0.8
    p0 = [1e-5, alpha_start]

    # Bounds keep the optimiser in physically meaningful territory: A > 0 (a
    # potentiation rate) and alpha in (0, 4]. x_scale='jac' conditions the two
    # parameters against each other (A ~ 1e-5 against alpha ~ 1), and _FIT_TOL
    # stops trf from terminating on the initial guess.
    bounds = ([0.0, 1e-6], [np.inf, 4.0])

    try:
        popt, pcov = curve_fit(power_law, G_initial, delta_g, p0=p0,
                               bounds=bounds, x_scale='jac', maxfev=50000,
                               **_FIT_TOL)
        A_fit, alpha_fit = popt
        _assert_moved_from_start('alpha', alpha_fit, alpha_start)

        r_squared = _r_squared(delta_g, power_law(G_initial, *popt))

        return {
            'alpha': float(alpha_fit),
            'A_coefficient': float(A_fit),
            'G_max_est': G_max_used,
            'G_max_attained': max_G_achieved,
            'fit_quality_R2': r_squared
        }
    except Exception as e:
        # RAISE, do not substitute. This used to return alpha = 0.8 — a
        # hardcoded physical default — and the caller in extract_synapse_model
        # then unconditionally recorded 'nonlinearity' as successfully
        # extracted, because no exception had reached it. A user reading the
        # extraction report could not tell a fit from a constant.
        raise RuntimeError(
            f"Nonlinearity fit failed ({type(e).__name__}: {e}). "
            f"{len(G_initial)} points, maximum attained {max_G_achieved:.3e} S."
        ) from e


# Convergence tolerances for every bounded curve_fit in this module.
#
# Supplying `bounds` makes curve_fit switch from 'lm' to 'trf', whose stopping
# tests are far looser by default. On conductance data the residuals are ~1e-6
# in absolute terms, and trf reads that as "already converged" — it returned
# the INITIAL GUESS, unchanged, and reported success. That is how alpha came
# back as exactly 0.7999 (from p0 = 0.8) and decay_tau as exactly 100.0 s (from
# p0 = 100, and numerically identical to the documented default). Both looked
# like measurements. Verified: with these tolerances, two different initial
# guesses for alpha (0.5 and 0.8) converge to the same value; with the
# defaults, each returned its own starting point.
_FIT_TOL = dict(ftol=1e-15, xtol=1e-15, gtol=1e-15)


def _assert_moved_from_start(name, fitted, start, rtol=1e-6):
    """Raise if a fitted parameter is still sitting on its initial guess.

    A backstop for the failure above: a parameter that has not moved at all is
    not evidence of a good fit, it is evidence that no fit happened.
    """
    if start == 0:
        return
    if abs(fitted - start) / abs(start) < rtol:
        raise RuntimeError(
            f"{name} = {fitted:.6g} is identical to its initial guess "
            f"{start:.6g}; the optimiser terminated without moving it, so this "
            "is not a fitted value."
        )


def _r_squared(observed, predicted):
    """Coefficient of determination, guarded against a degenerate total sum.

    ss_tot == 0 means the observations are constant, so R² is undefined rather
    than zero or one. Several call sites computed this unguarded and raised
    ZeroDivisionError or produced -inf on flat data; returning NaN says
    "undefined" without fabricating a quality figure.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot <= 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def fit_depression_nonlinearity(depression_data, G_min, decay_tau=None):
    """
    Fit power-law nonlinearity for depression: ΔG = -B × (G_initial - G_min)^β

    Relaxation pulls in the SAME direction as depression, so it inflates the
    measured decrease and, unsubtracted, is absorbed into B.
    """
    G_initial = np.array(depression_data['G_initial_S'])
    delta_g = np.array(depression_data['delta_g_S'])
    relax = _relaxation_factor(depression_data, decay_tau)

    # For depression, delta_g should be negative
    # Use absolute values for fitting
    abs_delta_g = np.abs(delta_g)
    
    min_G_achieved = float(np.min(G_initial + delta_g))

    # G_min comes from the suite's dynamic range, for the same reasons the
    # potentiation fit takes G_max from there — see fit_nonlinearity_exponent.
    # A bound above data that was actually measured is not a bound.
    G_min_used = float(G_min)
    if G_min_used >= min_G_achieved:
        G_min_used = min_G_achieved * 0.9999

    def power_law_dep(G_init, B, beta):
        drive = B * np.maximum(G_init - G_min_used, 1e-30) ** beta
        return drive + relax * np.maximum(G_init - G_min_used, 0.0)

    beta_start = 0.8
    p0 = [1e-5, beta_start]

    # M9: power_law_dep had no bounds, so curve_fit could explore beta < 0
    # against a base approaching zero as G_initial → G_min. That overflows
    # (an actual "RuntimeWarning: overflow encountered in power" was observed
    # on the synthetic suite), the fit fails, and the failure then landed in
    # the silent beta = 0.8 fallback below. Bounding beta to a physical range
    # removes both the overflow and the fallback that hid it.
    # x_scale='jac' for the same reason as in fit_nonlinearity_exponent: with
    # bounds present, trf's absolute convergence test would otherwise stop at
    # the initial guess given B ~ 1e-5 against beta ~ 1.
    bounds = ([0.0, 1e-6], [np.inf, 4.0])

    try:
        popt, pcov = curve_fit(power_law_dep, G_initial, abs_delta_g, p0=p0,
                               bounds=bounds, x_scale='jac', maxfev=50000,
                               **_FIT_TOL)
        B_fit, beta_fit = popt
        _assert_moved_from_start('beta', beta_fit, beta_start)

        r_squared = _r_squared(abs_delta_g, power_law_dep(G_initial, *popt))

        return {
            'beta': float(beta_fit),
            'B_coefficient': float(B_fit),
            'G_min_est': G_min_used,
            'G_min_attained': min_G_achieved,
            'fit_quality_R2': r_squared
        }
    except Exception as e:
        # Raise rather than substituting beta = 0.8 — see fit_nonlinearity_exponent.
        raise RuntimeError(
            f"Depression nonlinearity fit failed ({type(e).__name__}: {e}). "
            f"{len(G_initial)} points, minimum attained {min_G_achieved:.3e} S."
        ) from e



def fit_retention_decay(retention_data):
    """
    Fit exponential decay: G(t) = G_min + (G_initial - G_min) × exp(-t/tau)
    """
    times = np.array(retention_data['time_s'])
    conductances = np.array(retention_data['conductance_S'])
    G_initial = retention_data['g_initial_S']
    
    # Estimate G_min
    G_min_est = np.min(conductances) * 0.95
    
    # Exponential decay fit
    def exp_decay(t, tau):
        return G_min_est + (G_initial - G_min_est) * np.exp(-t / tau)
    
    p0 = [100.0]
    
    # Fitted on NORMALISED conductance. Raw conductances are ~1e-5 S, so the
    # residuals are ~1e-5 in absolute terms; a bounded solver (trf, selected
    # automatically whenever bounds are given) reads that as "already
    # converged" and returns the initial guess untouched — which here is
    # tau = 100 s, numerically identical to the documented default and
    # therefore indistinguishable from a device that never got fitted at all.
    # Normalising makes the residuals O(1) so the convergence test is
    # meaningful. tau is unaffected by the rescaling: it is a property of the
    # exponential's shape, not its amplitude.
    scale = float(G_initial - G_min_est)
    if scale <= 0:
        raise ValueError(
            f"Retention data has no decay range: G_initial = {G_initial:.3e} S "
            f"is not above the estimated floor {G_min_est:.3e} S."
        )

    def exp_decay_norm(t, tau):
        return np.exp(-t / tau)

    conductances_norm = (conductances - G_min_est) / scale

    try:
        popt, pcov = curve_fit(exp_decay_norm, times, conductances_norm, p0=p0,
                               bounds=([1e-6], [np.inf]), maxfev=50000,
                               **_FIT_TOL)
        tau_fit = popt[0]
        _assert_moved_from_start('decay_tau', tau_fit, p0[0])

        # R² reported against the data as measured, not the normalised proxy.
        r_squared = _r_squared(conductances, exp_decay(times, tau_fit))

        # Non-decaying data otherwise yield tau ~1e10 s with negative R2.
        k = max(3, len(conductances) // 10)
        if np.mean(conductances[-k:]) >= np.mean(conductances[:k]):
            raise RuntimeError(
                "Retention data do not decay: the final readings are not below the "
                "initial ones, so a relaxation time constant is not defined."
            )
        if not np.isfinite(r_squared) or r_squared <= 0:
            raise RuntimeError(
                f"Retention decay fit explains none of the variance (R2 = {r_squared:.3g}); "
                f"tau = {tau_fit:.3g} s is not a measurement."
            )

        return {
            'decay_tau_s': float(tau_fit),
            'fit_quality_R2': r_squared,
            'fitted_curve': lambda t: exp_decay(t, tau_fit)
        }
    except Exception as e:
        # Raise rather than substituting decay_tau = 100 s, which is the
        # documented default and therefore indistinguishable from a device
        # that genuinely measured 100 s.
        raise RuntimeError(
            f"Retention decay fit failed ({type(e).__name__}: {e}). "
            f"{len(times)} points spanning {times[0]:.3g}-{times[-1]:.3g} "
            f"{retention_data.get('time_units', 's')}."
        ) from e


def _dataset_wavelength(ds):
    """The wavelength that drove this dataset, chosen by its experiment type.

    `metadata.get('wavelength_pot') or metadata.get('wavelength_dep')` returned
    the POTENTIATION wavelength for a depression dataset whenever both keys
    were present, so a cycling measurement's depression phase was attributed to
    the potentiation band.
    """
    md = ds.get('metadata', {})
    exp_type = str(md.get('experiment_type', '')).lower()
    if exp_type == 'depression':
        return md.get('wavelength_dep', md.get('wavelength_pot'))
    if exp_type == 'potentiation':
        return md.get('wavelength_pot', md.get('wavelength_dep'))
    return md.get('wavelength_pot') if md.get('wavelength_pot') is not None \
        else md.get('wavelength_dep')


# Search bounds for the rate constants, in decades. Wide enough to contain any
# physically plausible device without the optimiser having to be told where to
# look.
_RATE_LOG_BOUNDS = (-8.0, 0.0)   # 1e-8 .. 1e0


def _fit_rate_constant(y_data, spectral_weight, intensity, dt_s,
                       G_min, G_max, exponent, depressing, decay_tau=None):
    """Fit one rate constant to a measured conductance trajectory.

    Implements the ODE exactly as CLAUDE.md documents it:

        potentiation:  dG = +A_peak · |S(λ)| · I · (G_max − G)^alpha · dt_s
        depression:    dG = −B_peak · |S(λ)| · I · (G − G_min)^beta  · dt_s

    Two defects are addressed here.

    First, the forward model omitted |S(λ)| entirely, so the fitted constant
    absorbed the spectral factor: it was a rate at the measured wavelength, not
    the PEAK rate its name claims, and the two differ by however far off-peak
    the measurement sat.

    Second, the search was `minimize_scalar(..., bounds=(1e-6, 1e-2),
    method='bounded')` on the constant itself. Brent's bounded method assumes a
    reasonably conditioned objective; this one is needle-shaped and spans eight
    decades, so the optimiser railed at the upper bound and — because
    `result.success` is True for a boundary solution — the railed value was
    recorded as a successful fit. Widening the bounds made it worse, not
    better. Searching over log10 of the constant makes the objective smooth on
    the scale the parameter actually varies over.

    Returns (value, diagnostics). `value` is None when the fit could not be
    trusted; diagnostics always explains why.
    """
    from scipy.optimize import minimize_scalar

    y_data = np.asarray(y_data, dtype=float)
    n_fit = min(len(y_data), 30)

    def sse(log10_rate):
        rate = 10.0 ** log10_rate
        G = y_data[0]
        error = 0.0
        for i in range(1, n_fit):
            drive = rate * spectral_weight * intensity
            if depressing:
                dG = -drive * max(G - G_min, 0.0) ** exponent * dt_s
            else:
                dG = drive * max(G_max - G, 0.0) ** exponent * dt_s
            # Spontaneous relaxation runs during stimulation too (CLAUDE.md).
            # Omitting it here would make the drive constant absorb the decay:
            # for potentiation the fitted A_peak would come out high to
            # compensate for the relaxation pulling the other way, and for
            # depression low. At the default parameters this term is ~6% of the
            # drive early in a train and grows as G rises.
            if decay_tau and decay_tau > 0:
                dG -= (G - G_min) / decay_tau * dt_s
            G = np.clip(G + dG, G_min, G_max)
            error += (G - y_data[i]) ** 2
        return error

    result = minimize_scalar(sse, bounds=_RATE_LOG_BOUNDS, method='bounded',
                             options={'xatol': 1e-4})

    diagnostics = {
        'converged': bool(result.success),
        'log10_value': float(result.x),
        'sse': float(result.fun),
        'spectral_weight': float(spectral_weight),
        'intensity': float(intensity),
        'n_points_fitted': int(n_fit),
    }

    if not result.success:
        diagnostics['reason'] = 'optimiser did not converge'
        return None, diagnostics

    # A solution sitting on either search bound is not a fit — it means the
    # true value is outside the searched range, or the objective is degenerate.
    # Reporting it as a measurement is what let B_peak come back 16.7x wrong
    # while the extraction report said everything succeeded.
    lo, hi = _RATE_LOG_BOUNDS
    margin = 1e-3 * (hi - lo)
    if result.x <= lo + margin or result.x >= hi - margin:
        diagnostics['reason'] = (
            f"solution railed at the search bound (log10 = {result.x:.3f}, "
            f"bounds = {lo} .. {hi}); the true value lies outside the searched "
            "range or the objective is degenerate"
        )
        diagnostics['railed'] = True
        return None, diagnostics

    return float(10.0 ** result.x), diagnostics


def extract_rate_constants(parsed_data, fitted_model):
    """
    Extract A_peak and B_peak by fitting to the actual conductance trajectories.

    Returns the two constants plus a `diagnostics` dict recording, per
    constant, whether it was fitted and — when it was not — why. The caller
    uses that to populate the extraction report, so a substituted default is
    never presented as a measurement.
    """
    A_peak = None
    B_peak = None
    diagnostics = {'A_peak': {'reason': 'no suitable dataset found'},
                   'B_peak': {'reason': 'no suitable dataset found'}}

    G_min = fitted_model.get('G_min', 1e-6)
    G_max = fitted_model.get('G_max', 1e-4)
    alpha = fitted_model.get('alpha', 0.8)
    beta = fitted_model.get('beta', 0.8)
    wavelength_curve = fitted_model.get('wavelength_curve')

    def spectral_weight_at(wl):
        """|S(λ)| normalised so the branch peak is 1, per the documented ODE.

        With no fitted curve there is no spectral information, so the weight is
        1 and the constant is a rate AT THE MEASURED WAVELENGTH. That is
        recorded in the diagnostics rather than passed off as a peak rate.
        """
        if wavelength_curve is None or wl is None:
            return 1.0, False
        value = float(np.atleast_1d(wavelength_curve(np.array([float(wl)])))[0])
        return abs(value), True

    if 'wavelength_response' in parsed_data and parsed_data['wavelength_response'] is not None:
        wl_data = parsed_data['wavelength_response']
        # The initial rate is the spectral quantity (see _build_wavelength_response);
        # fall back to the endpoint delta only if rates are unavailable.
        response_list = wl_data.get('initial_rate_S_per_pulse') or wl_data.get('delta_g_S', [])
        wavelengths = wl_data.get('wavelengths_nm', [])

        if response_list and wavelengths:
            all_datasets = (parsed_data.get('potentiation_data', [])
                            + parsed_data.get('depression_data', []))

            for label, sign in (('A_peak', +1), ('B_peak', -1)):
                indices = [i for i, dg in enumerate(response_list)
                           if (dg > 0 if sign > 0 else dg < 0)]
                if not indices:
                    diagnostics[label] = {
                        'reason': f'no wavelength produced a '
                                  f'{"potentiating" if sign > 0 else "depressing"} response'
                    }
                    continue

                best_idx = (max(indices, key=lambda i: response_list[i]) if sign > 0
                            else min(indices, key=lambda i: response_list[i]))
                target_wl = wavelengths[best_idx]

                for ds in all_datasets:
                    wl = _dataset_wavelength(ds)
                    if wl is None or abs(wl - target_wl) >= 1:
                        continue

                    y_data = np.asarray(ds['y_data'], dtype=float)
                    if len(y_data) < 10:
                        continue

                    md = ds['metadata']
                    pulse_width_ms = md.get('pulse_width_ms')
                    intensity = md.get('light_intensity_mW_cm2')
                    if pulse_width_ms is None or intensity is None:
                        diagnostics[label] = {
                            'reason': (
                                f'dataset at {wl} nm is missing '
                                f'{"pulse_width_ms" if pulse_width_ms is None else "light_intensity_mW_cm2"}, '
                                'so the rate constant cannot be denominated'
                            )
                        }
                        break

                    weight, have_spectrum = spectral_weight_at(wl)
                    if weight <= 0:
                        diagnostics[label] = {
                            'reason': f'fitted spectral response is zero at {wl} nm'
                        }
                        break

                    value, diag = _fit_rate_constant(
                        y_data, weight, intensity, pulse_width_ms / 1000.0,
                        G_min, G_max, beta if sign < 0 else alpha,
                        depressing=(sign < 0),
                        decay_tau=fitted_model.get('decay_tau'),
                    )
                    diag['wavelength_nm'] = wl
                    diag['spectral_normalisation_applied'] = have_spectrum
                    if not have_spectrum:
                        diag['caveat'] = (
                            'no fitted spectral curve; this is the rate at the '
                            'measured wavelength, not the peak rate'
                        )
                    diagnostics[label] = diag

                    if value is not None:
                        if sign > 0:
                            A_peak = value
                        else:
                            B_peak = value
                    break

    # Defaults, explicitly flagged. These used to be substituted silently, and
    # the caller then compared the result against the literal default to guess
    # whether a fit had happened — which cannot distinguish "defaulted" from
    # "fitted, and happened to land on the default".
    if A_peak is None or A_peak <= 0 or not np.isfinite(A_peak):
        A_peak = 8e-4
        diagnostics['A_peak']['defaulted'] = True
    else:
        diagnostics['A_peak']['defaulted'] = False

    if B_peak is None or B_peak <= 0 or not np.isfinite(B_peak):
        B_peak = 6e-4
        diagnostics['B_peak']['defaulted'] = True
    else:
        diagnostics['B_peak']['defaulted'] = False

    return {
        'A_peak': float(A_peak),
        'B_peak': float(B_peak),
        'diagnostics': diagnostics,
    }


def fit_stdp_window(stdp_data):
    """
    Fit STDP window parameters from spike timing-dependent plasticity data.
    
    STDP model:
        ΔW = A_+ * exp(-Δt/τ_+)  for Δt > 0 (LTP)
        ΔW = -A_- * exp(Δt/τ_-)  for Δt < 0 (LTD)
    
    Args:
        stdp_data: dict with 'delta_t_ms' and 'delta_g_percent' arrays
    
    Returns:
        dict: Fitted STDP parameters {A_plus, A_minus, tau_plus_ms, tau_minus_ms, fit_quality_R2}
    """
    # float dtype is enforced on both axes. A hardware sweep at whole-ms steps
    # yields an INTEGER delta_t array (audit M3), and an integer array here
    # propagates into np.zeros_like below, where the float exponentials then
    # truncate on assignment. That was harmless only while amplitudes were
    # ~19 (the old baseline-normalised scale); at the correct ~0.5 percent-of-
    # range scale every value truncates to zero and the whole LTP branch
    # silently vanishes while every fitted parameter still looks plausible.
    delta_t = np.asarray(stdp_data['delta_t_ms'], dtype=float)
    delta_g_percent = np.asarray(stdp_data['delta_g_percent'], dtype=float)

    # Split into LTP (Δt > 0) and LTD (Δt < 0)
    ltp_mask = delta_t > 0
    ltd_mask = delta_t < 0

    def _fit_branch(t, dg, branch):
        """Fit A·exp(-t/tau) to one branch, with t and dg already positive-signed.

        No sign-selective filtering. The previous implementation discarded
        every point with dg <= 0, which sounds like noise rejection but is a
        BIASED estimator: at large |Δt| the true signal is ~0 and noise
        dominates symmetrically, so the filter removes only the low half of
        that scatter and leaves the high half. The retained tail sits above
        the true curve, which drags tau. Measured effect on the synthetic
        suite: tau_plus = 14.0 ms and tau_minus = 42.0 ms against a true
        20 ms for both, at a respectable-looking R² of 0.914.

        Least-squares on all points, including the negative ones, is
        unbiased — the noise averages out as it should.
        """
        if len(t) < 3:
            raise ValueError(
                f"STDP {branch} branch has only {len(t)} points; at least 3 "
                "are needed to fit an amplitude and a time constant."
            )

        def exp_model(tt, A, tau):
            return A * np.exp(-tt / tau)

        # A ≥ 0 because the branch sign is already factored out; tau bounded
        # to a physically plausible window for synaptic plasticity.
        #
        # A is bounded ABOVE as well. A·exp(−t/τ) is degenerate against
        # noise-dominated data: the optimiser can drive τ onto its lower bound
        # and compensate with an arbitrarily large A, "fitting" a flat noisy
        # branch with a spike at the origin. Measured on seed 3 of the
        # synthetic suite: A_minus = 38.7 (129x the true 0.3) with
        # tau_minus = 1.0 ms exactly at the bound — reported at R² = 0.92 with
        # an extraction report claiming success. An amplitude above 100% of the
        # dynamic range per pairing is not a synapse, so it is not in the
        # search space.
        A_MAX_PERCENT_OF_RANGE = 100.0
        TAU_LO, TAU_HI = 1.0, 500.0
        A_guess = max(float(np.max(dg)), 1e-6)
        popt, _ = curve_fit(
            exp_model, t, dg,
            p0=[min(A_guess, A_MAX_PERCENT_OF_RANGE), 20.0],
            bounds=([0.0, TAU_LO], [A_MAX_PERCENT_OF_RANGE, TAU_HI]),
            x_scale='jac', maxfev=50000, **_FIT_TOL,
        )
        A_fit, tau_fit = float(popt[0]), float(popt[1])

        # Railing detection. A solution sitting on a bound is not a
        # measurement of that parameter — it is the optimiser reporting that
        # the data does not constrain it. The B_peak fit already detects and
        # reports this; _fit_branch did not, which is how a 129x error reached
        # the caller unflagged.
        rail = []
        if tau_fit <= TAU_LO * 1.001 or tau_fit >= TAU_HI * 0.999:
            rail.append(f"tau_{branch.lower()} = {tau_fit:.4g} ms is at its "
                        f"[{TAU_LO}, {TAU_HI}] ms bound")
        if A_fit >= A_MAX_PERCENT_OF_RANGE * 0.999:
            rail.append(f"A_{branch.lower()} = {A_fit:.4g} % is at its "
                        f"{A_MAX_PERCENT_OF_RANGE} % bound")
        if rail:
            raise ValueError(
                f"STDP {branch} fit did not converge to an interior solution: "
                + "; ".join(rail) + ".\n"
                "A railed exponential means the branch is noise-dominated and "
                "its amplitude and time constant are not jointly identifiable "
                "from this sweep — a large amplitude paired with a minimal tau "
                "can fit flat data at a high R², so R² will not reveal it.\n"
                "Average more pairings per Delta_t, or widen the Delta_t range "
                "so the exponential decay is actually resolved."
            )
        return A_fit, tau_fit

    # Fit LTP branch (Δt > 0)
    if not np.any(ltp_mask):
        raise ValueError(
            "STDP data contains no points with Δt > 0, so the LTP branch "
            "cannot be fitted. A_plus and tau_plus are unmeasurable from this "
            "sweep."
        )
    A_plus, tau_plus = _fit_branch(delta_t[ltp_mask], delta_g_percent[ltp_mask], "LTP")

    # Fit LTD branch (Δt < 0). Both axes are negated so the same positive
    # decaying exponential is fitted.
    if not np.any(ltd_mask):
        raise ValueError(
            "STDP data contains no points with Δt < 0, so the LTD branch "
            "cannot be fitted. A_minus and tau_minus are unmeasurable from "
            "this sweep."
        )
    A_minus, tau_minus = _fit_branch(
        -delta_t[ltd_mask], -delta_g_percent[ltd_mask], "LTD"
    )

    # Calculate R² for full STDP curve
    def stdp_model(dt):
        dt = np.asarray(dt, dtype=float)
        # dtype=float explicitly, not zeros_like's inherited dtype — see the
        # note on delta_t above (M3).
        result = np.zeros(dt.shape, dtype=float)
        result[dt > 0] = A_plus * np.exp(-dt[dt > 0] / tau_plus)
        result[dt < 0] = -A_minus * np.exp(dt[dt < 0] / tau_minus)
        # Δt == 0 is a simultaneous pre/post pair. The spec defines the window
        # by its two one-sided limits, which disagree at the origin, so the
        # value there is genuinely undefined rather than A_plus - A_minus (a
        # discontinuity that appears nowhere in the spec). Left at 0, and
        # excluded from R² below.
        return result

    predicted = stdp_model(delta_t)
    scored = delta_t != 0
    r2 = _r_squared(delta_g_percent[scored], predicted[scored])

    return {
        'A_plus': float(A_plus),
        'A_minus': float(A_minus),
        'tau_plus_ms': float(tau_plus),
        'tau_minus_ms': float(tau_minus),
        # Reported as computed. This used to be clamped to [0, 1], which
        # discards the one thing a negative R² tells you: that the fitted
        # model tracks the data worse than a flat line through its mean.
        'fit_quality_R2': r2,
        'amplitude_units': 'percent_of_dynamic_range',
    }


def fit_srdp_curve(srdp_data):
    """
    Fit SRDP (Spike-Rate-Dependent Plasticity) curve.
    
    Fits a sigmoid in LOG10 FREQUENCY (audit M7):

        ΔG(f) = baseline + max_change / (1 + exp(-(log10 f - log10 f0) / slope))

    `slope` is therefore in DECADES of frequency, not Hz, and is returned
    tagged as such ('slope_units': 'decades'). This docstring previously
    described the old LINEAR-frequency model, which is a different function —
    exactly the kind of stale claim that produced the generator/fitter
    mismatch M7 had to fix.

    `transition_freq_hz` is the sigmoid's INFLECTION point. For a monotonic
    (potentiation-only) response that is the whole story, but for a biphasic
    device — low-rate depression turning into high-rate potentiation, the
    BCM-like behaviour rate-dependent plasticity is named for — the inflection
    is NOT the zero crossing. `zero_crossing_freq_hz` is reported alongside for
    that case, and is the quantity to compare against a literature θ_m.

    Args:
        srdp_data: dict with 'frequencies_hz' and 'delta_g_percent' arrays
    
    Returns:
        dict: Fitted SRDP parameters {transition_freq_hz, slope, max_change, fit_quality_R2}
    """
    frequencies = np.asarray(srdp_data['frequencies_hz'], dtype=float)
    delta_g_percent = np.asarray(srdp_data['delta_g_percent'], dtype=float)

    if np.any(frequencies <= 0):
        raise ValueError(
            "SRDP frequencies must be positive; the sigmoid is fitted in "
            f"log-frequency. Got minimum {frequencies.min()} Hz."
        )

    # M7: the physics is log-frequency-dependent, and SRDP sweeps are
    # log-spaced by construction. Fitting a sigmoid in LINEAR frequency over
    # log-spaced data weights the top decade overwhelmingly — the fit is
    # dominated by a handful of high-frequency points and f0 is pulled toward
    # them. The transition is fitted in log10(f) and converted back, so f0
    # remains a frequency in Hz for the caller.
    log_f = np.log10(frequencies)

    def sigmoid_log(lf, max_change, log_f0, slope_decades):
        return max_change / (1 + np.exp(-(lf - log_f0) / slope_decades))

    # M8: the baseline used to be y_data[0] — the LOWEST-frequency point,
    # which on a typical sweep already carries ~27% of the full sigmoid. Using
    # it as "zero response" subtracts a real part of the signal and biases f0
    # upward (measured: +37% on the synthetic suite). The sigmoid's own lower
    # asymptote is fitted instead, as a free parameter, so no point has to be
    # assumed responseless.
    def sigmoid_log_offset(lf, baseline, max_change, log_f0, slope_decades):
        return baseline + sigmoid_log(lf, max_change, log_f0, slope_decades)

    span = float(np.max(delta_g_percent) - np.min(delta_g_percent))
    p0 = [
        float(np.min(delta_g_percent)),   # baseline (lower asymptote)
        span if span > 0 else 1.0,        # max_change (amplitude above it)
        float(np.median(log_f)),          # log10(f0)
        0.3,                              # slope, in decades
    ]

    try:
        popt, _ = curve_fit(
            sigmoid_log_offset, log_f, delta_g_percent,
            p0=p0,
            bounds=(
                [-np.inf, 0.0, log_f.min(), 0.01],
                [np.inf, np.inf, log_f.max(), 5.0],
            ),
            x_scale='jac', maxfev=50000, **_FIT_TOL,
        )
        baseline, max_change, log_f0, slope_decades = popt
        r2 = _r_squared(delta_g_percent, sigmoid_log_offset(log_f, *popt))
    except Exception as e:
        # Raise rather than returning f0 = 10 Hz / slope = 10, which are the
        # documented defaults and so indistinguishable from a device that
        # genuinely measured them.
        raise RuntimeError(
            f"SRDP sigmoid fit failed ({type(e).__name__}: {e}). "
            f"{len(frequencies)} points spanning "
            f"{frequencies.min():.3g}-{frequencies.max():.3g} Hz."
        ) from e

    f0 = float(10.0 ** log_f0)

    # The LTD -> LTP crossover, when the fitted response actually spans zero.
    #
    # `transition_freq_hz` is the sigmoid inflection, which coincides with the
    # sign change only for a curve centred on zero. On a biphasic device the
    # two differ materially — on a constructed curve crossing zero at 10.0 Hz
    # the inflection fitted to 10.08 Hz while the true crossing was 7.55 Hz,
    # a 34% discrepancy — and it is the CROSSING that corresponds to the
    # modification threshold the plasticity literature reports.
    # Solve  baseline + max_change / (1 + exp(-(x - log_f0)/s)) = 0  for x:
    #   1 + E = max_change / (-baseline),  E = exp(-(x - log_f0)/s)
    #   x = log_f0 - s * ln(E)
    lo, hi = float(baseline), float(baseline + max_change)
    zero_crossing_hz = None
    if lo < 0.0 < hi:
        E = (max_change / (-baseline)) - 1.0
        if E > 0:
            x = log_f0 - slope_decades * np.log(E)
            candidate = float(10.0 ** x)
            # Only report a crossing that lies inside the swept range. A
            # response that merely dips a hair below zero on noise extrapolates
            # to an absurd frequency, and reporting that would be inventing a
            # measurement rather than making one.
            if frequencies.min() <= candidate <= frequencies.max():
                zero_crossing_hz = candidate

    return {
        'max_change': float(max_change),
        'baseline': float(baseline),
        'transition_freq_hz': f0,
        # Present (not None) only for a response that genuinely changes sign.
        'zero_crossing_freq_hz': zero_crossing_hz,
        'transition_freq_definition': 'sigmoid_inflection',
        # Slope is reported in decades of frequency, which is the unit it is
        # now fitted in. The old value was a linear-Hz slope and is not
        # comparable.
        'slope': float(slope_decades),
        'slope_units': 'decades',
        'fit_quality_R2': r2,
        'change_units': 'percent_of_dynamic_range',
    }


# =============================================================================
# SERIALISATION (audit C9)
# =============================================================================
#
# `fitted_model['wavelength_curve']` and `['retention_curve']` are CLOSURES, so
# `json.dump` raises "Object of type function is not JSON serializable" — and
# leaves a truncated file behind, because the exception fires partway through
# writing. No module in the suite ever wrote a fitted-model JSON, so Main.py's
# "assume this is already a fitted model" load branch could only ever be
# reached with a file that necessarily LACKED the callable: the synapse then
# silently reverted to a positive single Gaussian, which is precisely the
# condition that made "Apply Depression" potentiate (C1).
#
# The fix is to store the curves' COEFFICIENTS — which is all a Gaussian sum or
# an exponential decay needs — and rebuild the callables on load.

FITTED_MODEL_FORMAT_VERSION = "1.0"


def serialise_fitted_model(fitted_model: Dict) -> Dict:
    """Return a JSON-safe copy of a fitted model.

    Callables are dropped; the coefficients needed to rebuild them are kept.
    Everything else is converted to plain Python types.
    """
    def _plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, dict):
            return {k: _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        return value

    out = {'fitted_model_format': FITTED_MODEL_FORMAT_VERSION}
    for key, value in fitted_model.items():
        if callable(value):
            # Rebuildable from wavelength_peaks / decay_tau on load.
            continue
        out[key] = _plain(value)
    return out


def deserialise_fitted_model(data: Dict) -> Dict:
    """Rebuild a fitted model, including its callable curves.

    The spectral curve is rebuilt as the SIGNED sum of the stored Gaussians —
    every peak, with its own sign — not as a single positive Gaussian at
    `lambda_peak`. Rebuilding it that way (audit M2) discarded the depression
    band entirely, so a loaded model made green light potentiate and UV do
    nothing.
    """
    model = dict(data)

    peaks = model.get('wavelength_peaks')
    if peaks:
        coefficients = [
            (float(p['amplitude']), float(p['wavelength_nm']), float(p['width_nm']))
            for p in peaks
        ]

        def wavelength_curve(x, _c=coefficients):
            x = np.asarray(x, dtype=float)
            result = np.zeros(x.shape, dtype=float)
            for amplitude, centre, width in _c:
                result += amplitude * np.exp(-((x - centre) / width) ** 2)
            return result

        model['wavelength_curve'] = wavelength_curve

    decay_tau = model.get('decay_tau')
    g_min = model.get('G_min')
    if decay_tau and g_min is not None:
        def retention_curve(t, _tau=float(decay_tau), _gmin=float(g_min),
                            _g0=float(model.get('G_max', g_min))):
            t = np.asarray(t, dtype=float)
            return _gmin + (_g0 - _gmin) * np.exp(-t / _tau)

        model['retention_curve'] = retention_curve

    return model


def save_fitted_model(fitted_model: Dict, filename: str):
    """Write a fitted model to JSON.

    Written to a temporary file and moved into place, so a failure partway
    through cannot leave a truncated model behind — which is what the previous
    (always-failing) json.dump did.
    """
    import json
    import os
    import tempfile

    payload = serialise_fitted_model(fitted_model)
    directory = os.path.dirname(os.path.abspath(filename)) or '.'

    handle, temp_path = tempfile.mkstemp(suffix='.json', dir=directory)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, filename)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    return filename


def load_fitted_model(filename: str) -> Dict:
    """Read a fitted model from JSON and rebuild its callable curves."""
    import json

    with open(filename, encoding='utf-8') as f:
        data = json.load(f)

    if 'fitted_model_format' not in data:
        raise ValueError(
            f"'{filename}' is not a saved fitted model (no "
            "'fitted_model_format' key). Characterization suites carry a "
            "'datasets' array and should be passed to extract_synapse_model "
            "instead."
        )

    return deserialise_fitted_model(data)


# =============================================================================
# FLEXIBLE MODEL EXTRACTION
# =============================================================================

def extract_synapse_model(characterization_suite: Dict, verbose: bool = True, wavelength_n_peaks: int = None) -> Dict:
    """
    Master function: Extract complete synapse model from characterization data.
    
    Args:
        characterization_suite: v2.0 format characterization data with 'datasets' array
        verbose: Print extraction progress and warnings
        wavelength_n_peaks: Force specific number of wavelength peaks (1, 2, or 3).
                           If None, use automatic detection.
    
    Returns:
        dict: Fitted synapse model parameters with extraction_report
    """
    # Parse data
    parsed_data = DataParser.parse_suite(characterization_suite)
    
    fitted_model = {}
    extraction_report = {
        'format': 'v2.0',
        'extracted': [],
        'failed': [],
        'defaults_used': [],
        'warnings': []
    }
    
    # 1. Dynamic range (CRITICAL - required)
    if 'dynamic_range' in parsed_data and parsed_data['dynamic_range'] is not None:
        dr_data = parsed_data['dynamic_range']
        fitted_model['G_min'] = dr_data['G_min_S']
        fitted_model['G_max'] = dr_data['G_max_S']
        # Calculate on_off_ratio from G_min and G_max
        if fitted_model['G_min'] > 0:
            fitted_model['on_off_ratio'] = fitted_model['G_max'] / fitted_model['G_min']
        else:
            fitted_model['on_off_ratio'] = None
        extraction_report['extracted'].append('dynamic_range')
    else:
        extraction_report['failed'].append('dynamic_range')
        extraction_report['warnings'].append("CRITICAL: No dynamic range data - using default G_min/G_max")
        fitted_model['G_min'] = 1e-6
        fitted_model['G_max'] = 1e-4
        fitted_model['on_off_ratio'] = 100.0
        extraction_report['defaults_used'].append('G_min')
        extraction_report['defaults_used'].append('G_max')
    
    # 2. Wavelength sensitivity (optional)
    if 'wavelength_response' in parsed_data and parsed_data['wavelength_response'] is not None:
        try:
            wl_fit = fit_wavelength_sensitivity(parsed_data['wavelength_response'], force_n_peaks=wavelength_n_peaks)
            fitted_model['lambda_peak'] = wl_fit['lambda_peak_nm']
            fitted_model['lambda_width'] = wl_fit['lambda_width_nm']
            fitted_model['lambda_peak_branch'] = wl_fit.get('lambda_peak_branch')
            # Both branches, so a consumer never has to guess which one the
            # top-level scalars describe (audit M1).
            for key in ('lambda_pot_peak_nm', 'lambda_pot_width_nm',
                        'lambda_dep_peak_nm', 'lambda_dep_width_nm'):
                if key in wl_fit:
                    fitted_model[key] = wl_fit[key]
            fitted_model['wavelength_fit_R2'] = wl_fit['fit_quality_R2']
            fitted_model['wavelength_curve'] = wl_fit['fitted_curve']
            fitted_model['wavelength_n_peaks'] = wl_fit['n_peaks']
            fitted_model['wavelength_peaks'] = wl_fit['peaks']
            # Store the wavelength range from the actual data for consistent plotting
            wavelengths = np.array(parsed_data['wavelength_response']['wavelengths_nm'])
            fitted_model['wavelength_range'] = [float(wavelengths.min()), float(wavelengths.max())]
            # Store the normalization factor for consistent display
            if 'max_abs_response' in parsed_data['wavelength_response']:
                fitted_model['max_abs_response'] = parsed_data['wavelength_response']['max_abs_response']
            extraction_report['extracted'].append('wavelength_response')
        except Exception as e:
            extraction_report['failed'].append('wavelength_response')
            extraction_report['warnings'].append(f"Wavelength fitting failed: {e}")
    else:
        extraction_report['warnings'].append("No wavelength data - using default spectral response")
        fitted_model['lambda_peak'] = 550.0
        fitted_model['lambda_width'] = 100.0
        extraction_report['defaults_used'].append('lambda_peak')
        extraction_report['defaults_used'].append('lambda_width')
    
    # 3. Retention/decay.
    #
    # Fitted BEFORE the nonlinearity exponents, which need decay_tau: spontaneous
    # relaxation runs during stimulation, so each sampled increment is drive
    # minus relaxation, and an exponent fitted without subtracting it absorbs it
    # (measured: alpha 0.86 against a true 0.75, which then pushed A_peak to
    # nearly 3x). Retention is independent of both exponents, so nothing is lost
    # by fitting it first.
    if 'retention' in parsed_data and parsed_data['retention'] is not None:
        try:
            ret_fit = fit_retention_decay(parsed_data['retention'])
            fitted_model['decay_tau'] = ret_fit['decay_tau_s']
            fitted_model['retention_fit_R2'] = ret_fit['fit_quality_R2']
            fitted_model['retention_curve'] = ret_fit['fitted_curve']
            extraction_report['extracted'].append('retention')
            # Any unit conversion applied on the way in is reported, so a
            # decay_tau derived from a millisecond axis is visibly derived
            # from a millisecond axis (M11).
            for note in parsed_data['retention'].get('unit_conversions', []):
                extraction_report['warnings'].append(f"retention: {note}")
        except Exception as e:
            extraction_report['failed'].append('retention')
            extraction_report['warnings'].append(f"Retention fitting failed: {e}")
            fitted_model['decay_tau'] = 100.0
            extraction_report['defaults_used'].append('decay_tau')
    else:
        extraction_report['warnings'].append("No retention data - using default decay_tau")
        fitted_model['decay_tau'] = 100.0
        extraction_report['defaults_used'].append('decay_tau')

    # 4. Nonlinearity exponent (optional)
    if 'nonlinearity' in parsed_data and parsed_data['nonlinearity'] is not None:
        try:
            # G_max from the suite's dynamic range: drawn from every dataset,
            # so a much better estimate of the asymptote than this one trace's
            # maximum. alpha is sensitive to it — see fit_nonlinearity_exponent.
            nl_fit = fit_nonlinearity_exponent(
                parsed_data['nonlinearity'],
                G_max=fitted_model.get('G_max'),
                G_min=fitted_model.get('G_min', 0.0),
                decay_tau=fitted_model.get('decay_tau'),
            )
            fitted_model['alpha'] = nl_fit['alpha']
            fitted_model['nonlinearity_fit_R2'] = nl_fit['fit_quality_R2']
            extraction_report['extracted'].append('nonlinearity')
        except Exception as e:
            extraction_report['failed'].append('nonlinearity')
            extraction_report['warnings'].append(f"Nonlinearity fitting failed: {e}")
            # Value still supplied so downstream consumers do not KeyError;
            # flagged as a default so nothing reads it as measured.
            fitted_model['alpha'] = 0.8
            extraction_report['defaults_used'].append('alpha')
    else:
        extraction_report['warnings'].append("No nonlinearity data - using default alpha")
        fitted_model['alpha'] = 0.8
        extraction_report['defaults_used'].append('alpha')

    # 4b. Depression nonlinearity exponent (beta) - optional
    if 'depression_nonlinearity' in parsed_data and parsed_data['depression_nonlinearity'] is not None:
        try:
            G_min = fitted_model.get('G_min', 1e-6)
            dep_fit = fit_depression_nonlinearity(
                parsed_data['depression_nonlinearity'], G_min,
                decay_tau=fitted_model.get('decay_tau'),
            )
            fitted_model['beta'] = dep_fit['beta']
            fitted_model['depression_nonlinearity_fit_R2'] = dep_fit['fit_quality_R2']
            extraction_report['extracted'].append('beta')
        except Exception as e:
            extraction_report['failed'].append('beta')
            extraction_report['warnings'].append(f"Depression nonlinearity fitting failed: {e}")
            # Values still supplied so downstream consumers do not KeyError,
            # and flagged as defaults so nothing reads them as measured.
            fitted_model.setdefault('beta', fitted_model.get('alpha', 0.8))
            extraction_report['defaults_used'].append('beta')
    else:
        extraction_report['warnings'].append("No depression nonlinearity data - using alpha as beta")
        fitted_model.setdefault('beta', fitted_model.get('alpha', 0.8))
        extraction_report['defaults_used'].append('beta')


    # 5. STDP parameters (optional but important for SNN simulation)
    if 'stdp' in parsed_data and parsed_data['stdp'] is not None:
        try:
            stdp_fit = fit_stdp_window(parsed_data['stdp'])
            fitted_model['stdp_A_plus'] = stdp_fit['A_plus']
            fitted_model['stdp_A_minus'] = stdp_fit['A_minus']
            fitted_model['stdp_tau_plus_ms'] = stdp_fit['tau_plus_ms']
            fitted_model['stdp_tau_minus_ms'] = stdp_fit['tau_minus_ms']
            fitted_model['stdp_fit_R2'] = stdp_fit['fit_quality_R2']
            extraction_report['extracted'].append('stdp')
        except Exception as e:
            extraction_report['failed'].append('stdp')
            extraction_report['warnings'].append(f"STDP fitting failed: {e}")
            # Values still supplied so downstream consumers do not KeyError,
            # and flagged as defaults so nothing reads them as measured.
            fitted_model['stdp_A_plus'] = 0.5
            fitted_model['stdp_A_minus'] = 0.3
            fitted_model['stdp_tau_plus_ms'] = 20.0
            fitted_model['stdp_tau_minus_ms'] = 20.0
            extraction_report['defaults_used'].append('stdp_params')
    else:
        extraction_report['warnings'].append("No STDP data - using default STDP parameters")
        fitted_model['stdp_A_plus'] = 0.5
        fitted_model['stdp_A_minus'] = 0.3
        fitted_model['stdp_tau_plus_ms'] = 20.0
        fitted_model['stdp_tau_minus_ms'] = 20.0
        extraction_report['defaults_used'].append('stdp_params')
    
    # 6. SRDP parameters (optional)
    if 'srdp' in parsed_data and parsed_data['srdp'] is not None:
        try:
            srdp_fit = fit_srdp_curve(parsed_data['srdp'])
            fitted_model['srdp_transition_freq_hz'] = srdp_fit['transition_freq_hz']
            fitted_model['srdp_slope'] = srdp_fit['slope']
            fitted_model['srdp_max_change'] = srdp_fit['max_change']
            fitted_model['srdp_fit_R2'] = srdp_fit['fit_quality_R2']
            extraction_report['extracted'].append('srdp')
        except Exception as e:
            extraction_report['failed'].append('srdp')
            extraction_report['warnings'].append(f"SRDP fitting failed: {e}")
            # Values still supplied so downstream consumers do not KeyError,
            # and flagged as defaults so nothing reads them as measured.
            fitted_model['srdp_transition_freq_hz'] = 10.0
            fitted_model['srdp_slope'] = 0.3
            fitted_model['srdp_max_change'] = 20.0
            extraction_report['defaults_used'].append('srdp_params')
    else:
        extraction_report['warnings'].append("No SRDP data - using default SRDP parameters")
        fitted_model['srdp_transition_freq_hz'] = 10.0
        # Slope in DECADES of frequency, matching the fitted units. The old
        # default of 10.0 was a linear-Hz slope and is not comparable.
        fitted_model['srdp_slope'] = 0.3
        # PERCENT of dynamic range, matching the fitted units. The old default
        # of 0.2 was a fraction against a fitted value in percent — the same
        # field carrying two units 140x apart depending on which path ran (M7).
        fitted_model['srdp_max_change'] = 20.0
        extraction_report['defaults_used'].append('srdp_params')
    
    # 7. Extract A_peak and B_peak from experimental data
    rate_constants = extract_rate_constants(parsed_data, fitted_model)
    fitted_model['A_peak'] = rate_constants['A_peak']
    fitted_model['B_peak'] = rate_constants['B_peak']
    fitted_model['rate_constant_diagnostics'] = rate_constants['diagnostics']

    # Provenance comes from the extractor's own diagnostics, not from comparing
    # the result against the literal default. That comparison could not
    # distinguish "defaulted" from "fitted, and landed on the default", and it
    # said nothing at all about a solution that had railed at its search bound
    # — which is exactly how B_peak came back 16.7x wrong while the report
    # claimed a clean extraction.
    for name in ('A_peak', 'B_peak'):
        diag = rate_constants['diagnostics'].get(name, {})
        if diag.get('defaulted'):
            extraction_report['defaults_used'].append(name)
            reason = diag.get('reason', 'no reason recorded')
            extraction_report['warnings'].append(
                f"{name} could not be fitted ({reason}); using the default "
                f"{fitted_model[name]:.3e}"
            )
            if diag.get('railed'):
                extraction_report['failed'].append(name)
        else:
            extraction_report['extracted'].append(name)
            if diag.get('caveat'):
                extraction_report['warnings'].append(f"{name}: {diag['caveat']}")


    # Add report to model
    fitted_model['extraction_report'] = extraction_report
    
    # Print report if verbose
    if verbose:
        print_extraction_report(extraction_report, fitted_model)
    
    return fitted_model


def print_extraction_report(report: Dict, fitted_model: Dict):
    """Print extraction report to console."""
    print("\n" + "="*70)
    print("MODEL EXTRACTION REPORT")
    print("="*70)
    print(f"Data format: {report['format']}")
    print(f"\n✓ Successfully extracted: {', '.join(report['extracted']) if report['extracted'] else 'None'}")
    
    if report['failed']:
        print(f"✗ Failed to extract: {', '.join(report['failed'])}")
    
    if report['defaults_used']:
        print(f"\n⚙ Using defaults for: {', '.join(set(report['defaults_used']))}")
    
    if report['warnings']:
        print("\n⚠ Warnings:")
        for w in report['warnings']:
            print(f"  - {w}")
    
    print("\n" + "-"*70)
    print("FITTED PARAMETERS:")
    print("-"*70)
    print(f"G_min:        {fitted_model.get('G_min', 0)*1e6:.2f} µS")
    print(f"G_max:        {fitted_model.get('G_max', 0)*1e6:.2f} µS")
    print(f"ON/OFF ratio: {fitted_model.get('on_off_ratio', 0):.1f}")
    print(f"alpha:        {fitted_model.get('alpha', 0):.3f}")
    print(f"A_peak:       {fitted_model.get('A_peak', 0):.3e}")
    print(f"B_peak:       {fitted_model.get('B_peak', 0):.3e}")
    print(f"λ_peak:       {fitted_model.get('lambda_peak', 0):.1f} nm")
    print(f"λ_width:      {fitted_model.get('lambda_width', 0):.1f} nm")
    print(f"decay_tau:    {fitted_model.get('decay_tau', 0):.1f} s")
    
    print("\n" + "-"*70)
    print("FIT QUALITY (R²):")
    print("-"*70)
    if 'wavelength_fit_R2' in fitted_model:
        print(f"Wavelength:    {fitted_model['wavelength_fit_R2']:.3f}")
    if 'nonlinearity_fit_R2' in fitted_model:
        print(f"Nonlinearity:  {fitted_model['nonlinearity_fit_R2']:.3f}")
    if 'retention_fit_R2' in fitted_model:
        print(f"Retention:     {fitted_model['retention_fit_R2']:.3f}")
    
    print("="*70 + "\n")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def reconstruct_fitted_curves(fitted_model: Dict, parsed_data: Dict = None) -> Dict:
    """
    Reconstruct fitted curve functions from stored parameters.
    This is needed when loading models from JSON (lambda functions can't be serialized).
    
    Args:
        fitted_model: Dictionary with fitted parameters
        parsed_data: Optional parsed characterization data (for retention curve reconstruction)
    
    Returns:
        dict: Updated fitted_model with reconstructed curve functions
    """
    model = fitted_model.copy()
    
    # Reconstruct the wavelength curve from the STORED PEAKS (audit M2).
    #
    # This used to rebuild a strictly positive SINGLE Gaussian at lambda_peak,
    # discarding every other peak and every sign. Combined with M1 — where
    # lambda_peak was the DEPRESSION band's centre — that meant a reconstructed
    # model had green light potentiating and UV doing nothing: the exact
    # inverse of the device. `wavelength_peaks` was stored all along and never
    # used.
    if model.get('wavelength_curve') is None:
        peaks = model.get('wavelength_peaks')
        if peaks:
            coefficients = [
                (float(p['amplitude']), float(p['wavelength_nm']), float(p['width_nm']))
                for p in peaks
            ]

            def wavelength_curve(x, _c=coefficients):
                x = np.asarray(x, dtype=float)
                result = np.zeros(x.shape, dtype=float)
                for amplitude, centre, width in _c:
                    result += amplitude * np.exp(-((x - centre) / width) ** 2)
                return result

            model['wavelength_curve'] = wavelength_curve

        elif 'lambda_peak' in model and 'lambda_width' in model:
            # No peak list at all — a legacy model. A single Gaussian is the
            # only thing reconstructible, and its SIGN is taken from the
            # recorded branch rather than assumed positive.
            lambda_peak = float(model['lambda_peak'])
            lambda_width = float(model['lambda_width'])
            sign = -1.0 if model.get('lambda_peak_branch') == 'depression' else 1.0

            def wavelength_curve(x, _p=lambda_peak, _w=lambda_width, _s=sign):
                x = np.asarray(x, dtype=float)
                return _s * np.exp(-((x - _p) / _w) ** 2)

            model['wavelength_curve'] = wavelength_curve
            model['wavelength_curve_is_single_gaussian_fallback'] = True
    
    # Reconstruct retention curve if parameters exist but curve doesn't
    if ('decay_tau' in model and 'G_min' in model and 
        model.get('retention_curve') is None and 
        parsed_data is not None and 
        'retention' in parsed_data and 
        parsed_data['retention'] is not None):
        
        decay_tau = model['decay_tau']
        G_min = model['G_min']
        G_initial = parsed_data['retention']['g_initial_S']
        
        # Create retention curve function
        model['retention_curve'] = lambda t: G_min + (G_initial - G_min) * np.exp(-t / decay_tau)
    
    return model


def plot_fitting_results(characterization_suite: Dict, fitted_model: Dict, true_params: Dict = None):
    """
    Plot fitting results showing data and fitted curves.
    
    Args:
        characterization_suite: v2.0 format data
        fitted_model: Output from extract_synapse_model()
        true_params: Optional dict of ground truth parameters for comparison
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    parsed_data = DataParser.parse_suite(characterization_suite)
    
    # Reconstruct fitted curves from parameters (needed after loading from JSON)
    fitted_model = reconstruct_fitted_curves(fitted_model, parsed_data)
    
    # Determine which plots to show
    has_wavelength = 'wavelength_response' in parsed_data and parsed_data['wavelength_response'] is not None
    has_nonlinearity = 'nonlinearity' in parsed_data and parsed_data['nonlinearity'] is not None
    has_retention = 'retention' in parsed_data and parsed_data['retention'] is not None
    has_dynamic_range = 'dynamic_range' in parsed_data and parsed_data['dynamic_range'] is not None
    has_stdp = 'stdp' in parsed_data and parsed_data['stdp'] is not None
    has_srdp = 'srdp' in parsed_data and parsed_data['srdp'] is not None
    
    # Count total plots
    n_plots = sum([has_wavelength, has_nonlinearity, has_retention, has_dynamic_range, has_stdp, has_srdp])
    
    if n_plots == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', fontsize=16)
        return fig
    
    # Create grid: try to make it roughly square
    if n_plots <= 4:
        n_rows, n_cols = 2, 2
    elif n_plots <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    
    # Add comparison info to title if ground truth provided
    if true_params:
        title = "Synapse Model Fitting Results (with Ground Truth Comparison)"
    else:
        title = "Synapse Model Fitting Results"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plot_idx = 1
    
    # Plot 1: Wavelength sensitivity
    if has_wavelength:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        wl_data = parsed_data['wavelength_response']
        wavelengths = np.array(wl_data['wavelengths_nm'])
        
        # Use normalized delta_g if available
        if 'delta_g_normalized' in wl_data:
            delta_g_values = np.array(wl_data['delta_g_normalized'])
        else:
            delta_g_values = np.array(wl_data.get('delta_g_S', wl_data.get('delta_g_percent', [])))
        
        # Normalize to [-1, 1] range for plotting
        max_abs = np.max(np.abs(delta_g_values))
        if max_abs > 0:
            delta_g_plot = delta_g_values / max_abs
        else:
            delta_g_plot = delta_g_values
        
        ax.plot(wavelengths, delta_g_plot, 'o', markersize=8, label='Data')
        
        if fitted_model.get('wavelength_curve') is not None:
            wl_fit = np.linspace(wavelengths.min(), wavelengths.max(), 100)
            fitted_values = fitted_model['wavelength_curve'](wl_fit)
            if max_abs > 0:
                fitted_norm = fitted_values / max_abs
            else:
                fitted_norm = fitted_values
            
            ax.plot(wl_fit, fitted_norm, '-', 
                   label=f"Fit: λ_peak={fitted_model['lambda_peak']:.1f}nm (R²={fitted_model.get('wavelength_fit_R2', 0):.3f})")
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Response")
        ax.set_title("Wavelength Sensitivity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        wl_range = wavelengths.max() - wavelengths.min()
        margin = wl_range * 0.05
        ax.set_xlim([wavelengths.min() - margin, wavelengths.max() + margin])
    
    # Plot 2: Nonlinearity
    if has_nonlinearity:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        nl_data = parsed_data['nonlinearity']
        G_initial = np.array(nl_data['G_initial_S']) * 1e6
        delta_g = np.array(nl_data['delta_g_S']) * 1e6
        
        ax.plot(G_initial, delta_g, 'o', markersize=6, label='Data')
        ax.set_xlabel("G_initial (µS)")
        ax.set_ylabel("ΔG (µS)")
        ax.set_title(f"Nonlinearity: α={fitted_model.get('alpha', 0):.3f}")
        ax.grid(True, alpha=0.3)
        if 'nonlinearity_fit_R2' in fitted_model:
            ax.text(0.05, 0.95, f"R² = {fitted_model['nonlinearity_fit_R2']:.3f}", 
                   transform=ax.transAxes, va='top')
    
    # Plot 3: Retention
    if has_retention:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        ret_data = parsed_data['retention']
        times = np.array(ret_data['time_s'])
        conductances = np.array(ret_data['conductance_S']) * 1e6
        
        ax.plot(times, conductances, 'o', markersize=6, label='Data')
        
        if fitted_model.get('retention_curve') is not None:
            t_fit = np.linspace(times.min(), times.max(), 100)
            ax.plot(t_fit, fitted_model['retention_curve'](t_fit) * 1e6, '-',
                   label=f"Fit: τ={fitted_model.get('decay_tau', 0):.1f}s (R²={fitted_model.get('retention_fit_R2', 0):.3f})")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Conductance (µS)")
        ax.set_title("Retention/Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Dynamic range
    if has_dynamic_range:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        G_min = fitted_model['G_min'] * 1e6
        G_max = fitted_model['G_max'] * 1e6
        
        ax.barh(['G_min', 'G_max'], [G_min, G_max], color=['blue', 'red'], alpha=0.7)
        ax.set_xlabel("Conductance (µS)")
        
        ratio = fitted_model.get('on_off_ratio')
        if ratio is not None and ratio > 0:
            ratio_str = f"{ratio:.1f}×"
        else:
            ratio_str = "N/A"
        
        ax.set_title(f"Dynamic Range: {ratio_str}")
        ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: STDP
    if has_stdp:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        stdp_data = parsed_data['stdp']
        delta_t = np.array(stdp_data['delta_t_ms'])
        delta_g_percent = np.array(stdp_data['delta_g_percent'])
        
        ax.plot(delta_t, delta_g_percent, 'o', markersize=6, label='Data', color='purple')
        
        # Plot fitted STDP curve if parameters available
        if all(k in fitted_model for k in ['stdp_A_plus', 'stdp_A_minus', 'stdp_tau_plus_ms', 'stdp_tau_minus_ms']):
            t_fit = np.linspace(delta_t.min(), delta_t.max(), 200)
            
            # Reconstruct STDP curve from fitted parameters
            A_plus = fitted_model['stdp_A_plus']
            A_minus = fitted_model['stdp_A_minus']
            tau_plus = fitted_model['stdp_tau_plus_ms']
            tau_minus = fitted_model['stdp_tau_minus_ms']
            
            # Approximate scaling to match data (delta_g_percent scale)
            G_range = (fitted_model['G_max'] - fitted_model['G_min'])
            scale = 10.0  # Approximate scale factor
            
            stdp_fit = np.zeros_like(t_fit)
            stdp_fit[t_fit > 0] = A_plus * np.exp(-t_fit[t_fit > 0] / tau_plus) * scale
            stdp_fit[t_fit < 0] = -A_minus * np.exp(t_fit[t_fit < 0] / tau_minus) * scale
            stdp_fit[t_fit == 0] = (A_plus - A_minus) * scale
            
            ax.plot(t_fit, stdp_fit, '-', color='darkviolet',
                   label=f"Fit: A+={A_plus:.2f}, τ+={tau_plus:.1f}ms (R²={fitted_model.get('stdp_fit_R2', 0):.3f})")
        
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Δt (ms) [t_post - t_pre]")
        ax.set_ylabel("ΔG (%)")
        ax.set_title("STDP Window")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: SRDP
    if has_srdp:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_idx += 1
        
        srdp_data = parsed_data['srdp']
        frequencies = np.array(srdp_data['frequencies_hz'])
        delta_g_percent = np.array(srdp_data['delta_g_percent'])
        
        ax.semilogx(frequencies, delta_g_percent, 'o', markersize=6, label='Data', color='darkorange')
        
        # Plot fitted SRDP curve if parameters available
        if all(k in fitted_model for k in ['srdp_transition_freq_hz', 'srdp_slope', 'srdp_max_change']):
            f_fit = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), 100)
            
            # Reconstruct sigmoid curve
            f0 = fitted_model['srdp_transition_freq_hz']
            slope = fitted_model['srdp_slope']
            max_change = fitted_model['srdp_max_change']
            
            srdp_fit = max_change / (1 + np.exp(-(f_fit - f0) / slope))
            
            ax.semilogx(f_fit, srdp_fit, '-', color='orangered',
                       label=f"Fit: f0={f0:.1f}Hz (R²={fitted_model.get('srdp_fit_R2', 0):.3f})")
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ΔG (%)")
        ax.set_title("SRDP (Spike-Rate-Dependent Plasticity)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add ground truth comparison text if provided
    if true_params:
        comparison_text = "Ground Truth vs Fitted:\n"
        if 'lambda_peak' in true_params and 'lambda_peak' in fitted_model:
            error = abs(fitted_model['lambda_peak'] - true_params['lambda_peak'])
            comparison_text += f"λ_peak: {true_params['lambda_peak']:.0f}nm → {fitted_model['lambda_peak']:.0f}nm (Δ={error:.0f}nm)\n"
        if 'alpha' in true_params and 'alpha' in fitted_model:
            error = abs(fitted_model['alpha'] - true_params['alpha'])
            comparison_text += f"α: {true_params['alpha']:.2f} → {fitted_model['alpha']:.2f} (Δ={error:.3f})\n"
        
        fig.text(0.02, 0.02, comparison_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig


# =============================================================================
# LEGACY COMPATIBILITY - Keep SyntheticCharacterization for Main.py
# =============================================================================

class SyntheticCharacterization:
    """
    Generates fake characterization data in v2.0 format for testing.
    This maintains backward compatibility with Main.py.
    """
    
    def __init__(self, true_params):
        self.params = true_params
        self.noise_level = 0.03
    
    def _add_noise(self, data):
        noise = np.random.normal(0, self.noise_level, len(data))
        return data * (1 + noise)
    
    def wavelength_sensitivity(self, wavelength_nm):
        return np.exp(-((wavelength_nm - self.params['lambda_peak']) / 
                       self.params['lambda_width'])**2)
    
    def measure_wavelength_response(self, wavelength_range_nm=(400, 700, 15)):
        """Generate wavelength sweep data."""
        wavelengths = np.linspace(*wavelength_range_nm)
        delta_g_list = []
        g_initial_list = []
        g_final_list = []
        
        base_intensity = 20
        n_pulses = 50
        pulse_duration_s = 0.1
        
        for wavelength in wavelengths:
            G_initial = self.params['G_min'] * 1.5
            G_current = G_initial
            
            A_eff = (self.params['A_peak'] * 
                    self.wavelength_sensitivity(wavelength) * 
                    base_intensity)
            
            for _ in range(n_pulses):
                dG = A_eff * (self.params['G_max'] - G_current)**self.params['alpha'] * pulse_duration_s
                G_current += dG
                G_current = np.clip(G_current, self.params['G_min'], self.params['G_max'])
            
            G_final = G_current
            delta_g = G_final - G_initial
            
            G_final_noisy = G_final * (1 + np.random.normal(0, self.noise_level))
            delta_g_noisy = G_final_noisy - G_initial
            
            delta_g_list.append(delta_g_noisy)
            g_initial_list.append(G_initial)
            g_final_list.append(G_final_noisy)
        
        return {
            'wavelengths_nm': wavelengths.tolist(),
            'delta_g_S': delta_g_list,
            'delta_g_percent': [(dg/gi)*100 for dg, gi in zip(delta_g_list, g_initial_list)],
            'g_initial_S': g_initial_list,
            'g_final_S': g_final_list,
            'measurement_type': 'wavelength_sweep'
        }
    
    def measure_nonlinearity(self, n_measurements=20):
        """Generate nonlinearity data."""
        G_states = np.linspace(self.params['G_min'] * 1.2, 
                              self.params['G_max'] * 0.8, 
                              n_measurements)
        G_initial_list = []
        delta_g_list = []
        
        base_intensity = 20
        wavelength = self.params['lambda_peak']
        n_pulses = 10
        pulse_duration_s = 0.1
        
        A_eff = (self.params['A_peak'] * 
                self.wavelength_sensitivity(wavelength) * 
                base_intensity)
        
        for G_initial in G_states:
            G_current = G_initial
            
            for _ in range(n_pulses):
                dG = A_eff * (self.params['G_max'] - G_current)**self.params['alpha'] * pulse_duration_s
                G_current += dG
                G_current = np.clip(G_current, self.params['G_min'], self.params['G_max'])
            
            delta_g = G_current - G_initial
            delta_g_noisy = delta_g * (1 + np.random.normal(0, self.noise_level))
            
            G_initial_list.append(G_initial)
            delta_g_list.append(delta_g_noisy)
        
        return {
            'G_initial_S': G_initial_list,
            'delta_g_S': delta_g_list,
            'measurement_type': 'nonlinearity'
        }
    
    def measure_retention(self, max_time_s=1000, n_points=10):
        """Generate retention/decay data."""
        # Potentiate first
        G_initial = self.params['G_min'] * 1.5
        base_intensity = 30
        wavelength = self.params['lambda_peak']
        n_pulses = 50
        pulse_duration_s = 0.1
        
        A_eff = (self.params['A_peak'] * 
                self.wavelength_sensitivity(wavelength) * 
                base_intensity)
        
        G_current = G_initial
        for _ in range(n_pulses):
            dG = A_eff * (self.params['G_max'] - G_current)**self.params['alpha'] * pulse_duration_s
            G_current += dG
            G_current = np.clip(G_current, self.params['G_min'], self.params['G_max'])
        
        G_potentiated = G_current
        
        # Measure decay
        times = np.logspace(0, np.log10(max_time_s), n_points)
        conductances = []
        
        for t in times:
            G_t = self.params['G_min'] + (G_potentiated - self.params['G_min']) * \
                  np.exp(-t / self.params['decay_tau'])
            G_t_noisy = G_t * (1 + np.random.normal(0, self.noise_level))
            conductances.append(G_t_noisy)
        
        return {
            'time_s': times.tolist(),
            'conductance_S': conductances,
            'g_initial_S': G_potentiated,
            'measurement_type': 'retention'
        }
    
    def measure_dynamic_range(self):
        """Generate dynamic range data."""
        return {
            'G_min_S': self.params['G_min'],
            'G_max_S': self.params['G_max'],
            'dynamic_range_S': self.params['G_max'] - self.params['G_min'],
            'on_off_ratio': self.params['G_max'] / self.params['G_min'],
            'measurement_type': 'dynamic_range'
        }
    
    def measure_stdp(self, delta_t_range_ms=(-100, 100), n_points=20):
        """
        Generate STDP (Spike-Timing-Dependent Plasticity) data.
        
        Args:
            delta_t_range_ms: Tuple of (min, max) timing differences in ms
            n_points: Number of timing points to measure
        
        Returns:
            dict: STDP measurements
        """
        # STDP window parameters (realistic biological values)
        A_plus = 0.5 * self.params.get('A_peak', 8e-4) / 8e-4  # LTP amplitude (normalized)
        A_minus = 0.3 * self.params.get('B_peak', 6e-4) / 6e-4  # LTD amplitude (normalized)
        tau_plus = 20.0  # LTP time constant (ms)
        tau_minus = 20.0  # LTD time constant (ms)
        
        # Generate timing differences
        delta_t_list = np.linspace(delta_t_range_ms[0], delta_t_range_ms[1], n_points)
        
        G_initial = (self.params['G_min'] + self.params['G_max']) / 2
        
        delta_g_list = []
        g_initial_list = []
        g_final_list = []
        
        for delta_t_ms in delta_t_list:
            # Calculate STDP weight change
            if delta_t_ms > 0:
                # Pre before Post → LTP
                weight_change = A_plus * np.exp(-delta_t_ms / tau_plus)
            elif delta_t_ms < 0:
                # Post before Pre → LTD
                weight_change = -A_minus * np.exp(delta_t_ms / tau_minus)
            else:
                weight_change = A_plus - A_minus  # At t=0, both contributions
            
            # Apply change (scaled by dynamic range)
            delta_g = weight_change * (self.params['G_max'] - self.params['G_min']) * 0.1
            G_final = G_initial + delta_g
            
            # Add noise
            G_final_noisy = G_final * (1 + np.random.normal(0, self.noise_level))
            delta_g_noisy = G_final_noisy - G_initial
            
            # Clip to physical bounds
            G_final_noisy = np.clip(G_final_noisy, self.params['G_min'], self.params['G_max'])
            delta_g_noisy = G_final_noisy - G_initial
            
            delta_g_list.append(delta_g_noisy)
            g_initial_list.append(G_initial)
            g_final_list.append(G_final_noisy)
        
        return {
            'delta_t_ms': delta_t_list.tolist(),
            'delta_g_S': delta_g_list,
            'delta_g_percent': [(dg/gi)*100 for dg, gi in zip(delta_g_list, g_initial_list)],
            'g_initial_S': g_initial_list,
            'g_final_S': g_final_list,
            'measurement_type': 'stdp',
            'ground_truth_params': {
                'A_plus': A_plus,
                'A_minus': A_minus,
                'tau_plus_ms': tau_plus,
                'tau_minus_ms': tau_minus
            }
        }
    
    def measure_srdp(self, freq_range_hz=(0.1, 100), n_points=15):
        """
        Generate SRDP (Spike-Rate-Dependent Plasticity) data.
        
        Args:
            freq_range_hz: Tuple of (min, max) frequencies in Hz
            n_points: Number of frequency points to measure
        
        Returns:
            dict: SRDP measurements
        """
        frequencies = np.logspace(np.log10(freq_range_hz[0]), 
                                  np.log10(freq_range_hz[1]), 
                                  n_points)
        
        G_initial = (self.params['G_min'] + self.params['G_max']) / 2
        
        delta_g_list = []
        g_initial_list = []
        g_final_list = []
        
        # Frequency dependence: higher frequency → stronger plasticity
        for freq_hz in frequencies:
            # Sigmoid-like frequency response
            freq_factor = 1 / (1 + np.exp(-(freq_hz - 10) / 10))  # Transition around 10 Hz
            
            # Total conductance change increases with frequency
            base_change = 0.2 * freq_factor  # Up to 20% change at high frequencies
            
            # Add some variability
            noise_factor = 1 + np.random.uniform(-0.05, 0.05)
            delta_g = base_change * (self.params['G_max'] - self.params['G_min']) * noise_factor
            
            G_final = G_initial + delta_g
            
            # Add measurement noise
            G_final_noisy = G_final * (1 + np.random.normal(0, self.noise_level))
            delta_g_noisy = G_final_noisy - G_initial
            
            # Clip to bounds
            G_final_noisy = np.clip(G_final_noisy, self.params['G_min'], self.params['G_max'])
            delta_g_noisy = G_final_noisy - G_initial
            
            delta_g_list.append(delta_g_noisy)
            g_initial_list.append(G_initial)
            g_final_list.append(G_final_noisy)
        
        return {
            'frequencies_hz': frequencies.tolist(),
            'delta_g_S': delta_g_list,
            'delta_g_percent': [(dg/gi)*100 for dg, gi in zip(delta_g_list, g_initial_list)],
            'g_initial_S': g_initial_list,
            'g_final_S': g_final_list,
            'measurement_type': 'srdp'
        }
    
    def generate_complete_suite(self):
        """Generate complete v2.0 format characterization suite."""
        # For compatibility with Main.py which expects old format
        return {
            'wavelength_response': self.measure_wavelength_response(),
            'nonlinearity': self.measure_nonlinearity(),
            'retention': self.measure_retention(),
            'dynamic_range': self.measure_dynamic_range(),
            'stdp': self.measure_stdp(),
            'srdp': self.measure_srdp()
        }


def compare_models(true_params: Dict, fitted_model: Dict):
    """Compare fitted parameters against ground truth."""
    print("\n" + "="*70)
    print("MODEL COMPARISON (Fitted vs Ground Truth)")
    print("="*70)
    
    comparisons = [
        ('G_min (µS)', fitted_model.get('G_min', 0)*1e6, true_params.get('G_min', 0)*1e6),
        ('G_max (µS)', fitted_model.get('G_max', 0)*1e6, true_params.get('G_max', 0)*1e6),
        ('alpha', fitted_model.get('alpha', 0), true_params.get('alpha', 0)),
        ('λ_peak (nm)', fitted_model.get('lambda_peak', 0), true_params.get('lambda_peak', 0)),
        ('λ_width (nm)', fitted_model.get('lambda_width', 0), true_params.get('lambda_width', 0)),
        ('decay_tau (s)', fitted_model.get('decay_tau', 0), true_params.get('decay_tau', 0)),
    ]
    
    print(f"{'Parameter':<15} {'Fitted':<12} {'True':<12} {'Error (%)':<12}")
    print("-"*70)
    
    comparison_dict = {}
    for param_name, fitted, true in comparisons:
        if true != 0:
            error = abs((fitted - true) / true) * 100
            print(f"{param_name:<15} {fitted:<12.3f} {true:<12.3f} {error:<12.2f}")
        else:
            error = 0
            print(f"{param_name:<15} {fitted:<12.3f} {true:<12.3f} {'N/A':<12}")
        
        comparison_dict[param_name] = {
            'fitted': fitted,
            'true': true,
            'percent_error': error
        }
    
    print("="*70 + "\n")
    
    return comparison_dict


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing fitting.py v3.0...")
    
    # Test parameters
    test_params = {
        'G_min': 1e-6,
        'G_max': 1e-4,
        'alpha': 0.75,
        'beta': 0.75,
        'lambda_peak': 550,
        'lambda_width': 100,
        'A_peak': 8e-4,
        'B_peak': 6e-4,
        'decay_tau': 100
    }
    
    # Generate synthetic data
    print("\n1. Generating synthetic characterization data...")
    synth = SyntheticCharacterization(test_params)
    suite_old_format = synth.generate_complete_suite()
    
    # Convert to v2.0 format for testing
    suite_v2 = {
        'datasets': [],
        'metadata': {'test': True}
    }
    
    # Note: This is a simplified conversion just for testing
    # Real v2.0 format would have proper dataset structure
    print("   WARNING: Using simplified v2.0 format conversion for testing")
    
    print("\n2. Testing model extraction...")
    try:
        fitted_model = extract_synapse_model(suite_old_format, verbose=True)
        print("   ✓ Model extraction successful")
    except ValueError as e:
        print(f"   Expected error for old format: {e}")
        print("   This is correct - v3.0 only accepts v2.0 format")
    
    print("\n✓ fitting.py v3.0 test complete")