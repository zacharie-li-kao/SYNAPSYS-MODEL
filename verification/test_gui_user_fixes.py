"""Headless tests for the GUI-independent logic behind the Main.py user-facing fixes.

No Tk app is constructed: everything here is a module-level function or the
VisualSynapse class.
"""
import math

import numpy as np
import pytest

import Main
from Main import (VisualSynapse, resolve_stdp_parameters, pick_branch_wavelengths,
                  describe_branch, summarise_branch_overrides, parse_snn_run_inputs)


def _signed_curve(wl):
    """+ lobe at 365 nm, - lobe at 550 nm (the synthetic suite's shape)."""
    wl = np.asarray(wl, dtype=float)
    return (1.0 * np.exp(-((wl - 365) / 30) ** 2)
            - 0.6 * np.exp(-((wl - 550) / 30) ** 2))


def _signed_synapse(curve=_signed_curve):
    return VisualSynapse(wavelength_curve=curve, wavelength_range=[300, 700])


# --- item 1: apply_stimulus vocabulary ---------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(stimulus_type='optical'),
    dict(stimulus_type='Light'),
    dict(mode='inhibition'),
    dict(mode='Potentiation'),
    dict(intensity_mW_cm2=float('nan')),
    dict(intensity_mW_cm2=float('inf')),
    dict(intensity_mW_cm2=-5.0),  # light, negative
])
def test_apply_stimulus_rejects_invalid_requests(kwargs):
    syn = VisualSynapse()
    args = dict(intensity_mW_cm2=10.0, wavelength_nm=550, duration_ms=10,
                mode='potentiation', stimulus_type='light')
    args.update(kwargs)
    G0 = syn.G
    with pytest.raises(ValueError):
        syn.apply_stimulus(**args)
    assert syn.G == G0


def test_non_finite_electrical_intensity_rejected():
    with pytest.raises(ValueError):
        VisualSynapse().apply_stimulus(float('nan'), 0, 10, 'potentiation', 'electrical')


def test_valid_calls_unchanged():
    syn = VisualSynapse()
    G0 = syn.G
    syn.apply_stimulus(20, 550, 50, 'potentiation', 'light')
    assert syn.G > G0
    G1 = syn.G
    syn.apply_stimulus(20, 550, 50, 'depression', 'light')
    assert syn.G < G1
    # Electrical keeps |V| with the mode choosing the branch; negative V allowed.
    G2 = syn.G
    syn.apply_stimulus(-5, 0, 50, 'depression', 'electrical')
    assert syn.G < G2
    G3 = syn.G
    syn.apply_stimulus(-5, 0, 50, 'potentiation', 'electrical')
    assert syn.G > G3
    # Zero light intensity and rest() remain legal.
    syn.apply_stimulus(0, 550, 10, 'potentiation', 'light')
    syn.rest(45)


# --- items 2 / 7: branch wavelengths -----------------------------------------

def test_pick_branch_wavelengths_finds_both_extrema():
    pot, dep = pick_branch_wavelengths(_signed_synapse())
    assert pot[0] == pytest.approx(365, abs=1) and pot[1] == pytest.approx(1.0, abs=1e-6)
    assert dep[0] == pytest.approx(550, abs=1) and dep[1] == pytest.approx(-1.0, abs=1e-6)


def test_pick_branch_wavelengths_reports_missing_band():
    only_dep = _signed_synapse(lambda wl: -np.exp(-((np.asarray(wl, float) - 500) / 40) ** 2))
    pot, dep = pick_branch_wavelengths(only_dep)
    assert pot is None and dep[0] == pytest.approx(500, abs=1)
    only_pot = _signed_synapse(lambda wl: np.exp(-((np.asarray(wl, float) - 500) / 40) ** 2))
    pot, dep = pick_branch_wavelengths(only_pot)
    assert dep is None and pot[0] == pytest.approx(500, abs=1)


def test_pick_branch_wavelengths_refuses_unfitted_synapse():
    with pytest.raises(ValueError):
        pick_branch_wavelengths(VisualSynapse())


def test_picked_wavelengths_really_run_their_branch():
    syn = _signed_synapse()
    pot, dep = pick_branch_wavelengths(syn)
    G0 = syn.G
    syn.apply_stimulus(20, pot[0], 100, 'potentiation', 'light')
    assert syn.G > G0 and syn.last_stimulus_report is None
    syn.apply_stimulus(20, dep[0], 100, 'depression', 'light')
    assert syn.last_stimulus_report is None


# --- items 7 / 8: branch reporting -------------------------------------------

def test_describe_branch_and_override_summary():
    syn = _signed_synapse()
    syn.apply_stimulus(20, 550, 10, 'potentiation', 'light')  # 550 nm depresses
    rep = syn.last_stimulus_report
    assert rep['requested_mode'] == 'potentiation' and rep['actual_mode'] == 'depression'
    assert describe_branch('potentiation', rep).startswith('OVERRIDE')
    assert describe_branch('potentiation', None) == 'ran as requested (potentiation)'

    syns = [[_signed_synapse(), _signed_synapse()]]
    pattern = np.array([[1.0, 0.0]])
    for j, row_syn in enumerate(syns[0]):
        row_syn.apply_stimulus(20 * pattern[0, j], 550, 10, 'potentiation', 'light')
    n_lit, overrides = summarise_branch_overrides(syns, pattern)
    assert n_lit == 1  # the unlit pixel is ignored
    assert len(overrides) == 1 and overrides[0]['count'] == 1
    assert overrides[0]['requested'] == 'potentiation' and overrides[0]['actual'] == 'depression'


def test_no_override_when_branch_matches():
    syns = [[_signed_synapse()]]
    syns[0][0].apply_stimulus(20, 365, 10, 'potentiation', 'light')
    assert summarise_branch_overrides(syns, np.ones((1, 1))) == (1, [])


# --- item 4: STDP parameters from one source ---------------------------------

def _model(report, **stdp):
    m = {'stdp_A_plus': 0.4517, 'stdp_A_minus': 0.21, 'stdp_tau_plus_ms': 18.0,
         'stdp_tau_minus_ms': 22.0, 'extraction_report': report}
    m.update(stdp)
    return m


def test_stdp_uses_fitted_values_undivided_when_extracted():
    params, text = resolve_stdp_parameters(_model({'extracted': ['stdp']}))
    assert params == {'stdp_A_plus': 0.4517, 'stdp_A_minus': 0.21,
                      'stdp_tau_plus_ms': 18.0, 'stdp_tau_minus_ms': 22.0}
    assert 'fitted' in text and 'DEFAULT' not in text


def test_stdp_defaults_with_honest_reason():
    d = Main.DEFAULT_STDP_PARAMS
    assert d == {'stdp_A_plus': 0.5, 'stdp_A_minus': 0.3,
                 'stdp_tau_plus_ms': 20.0, 'stdp_tau_minus_ms': 20.0}

    params, text = resolve_stdp_parameters(None)
    assert params == d and 'no fitted model' in text

    # fitting.py stores default values under the stdp_* keys: presence != measured.
    params, text = resolve_stdp_parameters(
        _model({'extracted': ['dynamic_range'], 'defaults_used': ['stdp_params']}))
    assert params == d and 'no STDP measurement' in text

    params, text = resolve_stdp_parameters(
        _model({'extracted': [], 'failed': ['stdp'], 'defaults_used': ['stdp_params']}))
    assert params == d and 'failed' in text

    params, text = resolve_stdp_parameters(_model({}))
    assert params == d and 'DEFAULT' in text


def test_stdp_extracted_but_keys_missing_fails_loudly():
    with pytest.raises(KeyError):
        resolve_stdp_parameters({'extraction_report': {'extracted': ['stdp']}})


# --- item 10: SNN input validation -------------------------------------------

def test_snn_inputs_valid():
    assert parse_snn_run_inputs("10", " 5 ", "20", "100.5") == (10, 5, 20.0, 100.5)
    assert parse_snn_run_inputs("1", "1", "0", "0.001")[2] == 0.0


@pytest.mark.parametrize("args", [
    ("0", "5", "20", "100"),
    ("5", "0", "20", "100"),
    ("-3", "5", "20", "100"),
    ("2.5", "5", "20", "100"),
    ("abc", "5", "20", "100"),
    ("5", "5", "-1", "100"),
    ("5", "5", "nan", "100"),
    ("5", "5", "inf", "100"),
    ("5", "5", "20", "0"),
    ("5", "5", "20", "-10"),
    ("5", "5", "20", "nan"),
    ("5", "5", "20", ""),
])
def test_snn_inputs_invalid(args):
    with pytest.raises(ValueError):
        parse_snn_run_inputs(*args)
