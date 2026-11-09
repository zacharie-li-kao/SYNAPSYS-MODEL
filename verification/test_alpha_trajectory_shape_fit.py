"""alpha from the shared shape of every potentiation trace.

A_peak carries units S^(1-alpha), so any error in alpha reaches it multiplied
by |ln W| ~ 9. With alpha fitted from ~10 increments of one trace, A_peak
scattered +-49% across generator seeds while the physical rate
k = A_peak * W^(alpha-1) was right to +-3%. The shape fit uses every reading
of every trace with a free amplitude per trace, so trace-to-trace gain (spectral
weight, intensity, device variability) cannot leak into alpha.
"""

import numpy as np
import pytest

import baseline as bl
import fitting
import pipeline

G_MIN, G_MAX, TAU, ALPHA = 1e-6, 1e-4, 100.0, 0.75
W = G_MAX - G_MIN


def _trace(amplitude, n=60, g0=1.5e-6, t_pulse=0.1, noise=0.015, rng=None):
    G, out = g0, [g0]
    for _ in range(n):
        G += amplitude * W * ((G_MAX - G) / W) ** ALPHA * t_pulse - (G - G_MIN) * t_pulse / TAU
        G = min(max(G, G_MIN), G_MAX)
        out.append(G * (1 + noise * rng.standard_normal()))
    return np.array(out)


def _as_traces(ys):
    return [{'name': f't{i}', 'y': y, 'pulse_width_s': 0.1,
             'timing': {'pulse_period_ms': None, 'frequency_Hz': None, 'pulse_width_ms': 100.0}}
            for i, y in enumerate(ys)]


def test_alpha_is_recovered_whatever_the_per_trace_gain():
    rng = np.random.default_rng(11)
    gains = [0.02, 0.05, 0.1, 0.3, 0.8]          # 40x spread of drive between traces
    fit = fitting.fit_exponent_from_trajectories(
        _as_traces([_trace(a, rng=rng) for a in gains]), G_MIN, G_MAX, TAU)
    assert fit['alpha'] == pytest.approx(ALPHA, abs=3 * fit['alpha_stderr'])
    assert fit['alpha_stderr'] < 0.02
    # The amplitudes carry the gains, not alpha.
    assert np.allclose(fit['trace_amplitudes_per_s'], gains, rtol=0.1)


def test_flat_traces_are_skipped_with_a_reason():
    flat = {'filename': 'flat.csv', 'y_data': [5e-5] * 30,
            'metadata': {'experiment_type': 'potentiation', 'pulse_width_ms': 100.0}}
    traces, skipped = fitting._potentiation_traces_for_shape_fit({'potentiation_data': [flat]})
    assert traces == [] and 'flat.csv' in skipped[0]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_suite_alpha")
    return pipeline.run(str(out), seed=bl.PIPELINE_SEED, verbose=False)[1]


def test_the_pipeline_uses_the_shape_fit_and_reports_its_uncertainty(model):
    assert model['alpha_method'].startswith('trajectory shape fit')
    assert model['alpha_stderr'] < 0.02
    W_fit = model['G_max'] - model['G_min']
    assert model['A_peak_rel_uncertainty_from_alpha'] == pytest.approx(
        abs(np.log(W_fit)) * model['alpha_stderr'])
    # The increment method survives as an independent cross-check.
    assert model['alpha_increment_estimate'] == pytest.approx(model['alpha'], abs=0.05)


def test_rate_constants_are_the_unit_independent_form(model):
    W_fit = model['G_max'] - model['G_min']
    assert model['k_A_per_s_per_mW_cm2'] == pytest.approx(
        model['A_peak'] * W_fit ** (model['alpha'] - 1))
    assert model['k_B_per_s_per_mW_cm2'] == pytest.approx(
        model['B_peak'] * W_fit ** (model['beta'] - 1))
