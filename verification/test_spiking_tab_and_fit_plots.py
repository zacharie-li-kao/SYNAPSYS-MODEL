"""Spiking-mode Network tab interface, and the STDP/SRDP curves drawn by
`plot_fitting_results`.

The plot used to scale the STDP window by a hand-set 10.0 and redraw SRDP with
the old linear-frequency formula and no baseline, so neither drawn line was the
model whose R² sat in the legend.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import baseline as bl
import fitting
import pipeline
from network import (SpikingSynapseNetwork, SynapseNetwork,
                     plot_network_evolution, plot_network_state)

BASE_PARAMS = dict(G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                   lambda_peak=550, lambda_width=100, decay_tau=100,
                   A_peak=8e-4, B_peak=6e-4)
STDP_PARAMS = dict(stdp_A_plus=0.5, stdp_A_minus=0.3,
                   stdp_tau_plus_ms=20.0, stdp_tau_minus_ms=20.0)


def _net(seed=1):
    np.random.seed(seed)
    return SpikingSynapseNetwork(BASE_PARAMS, STDP_PARAMS, n_input=6, n_hidden=4)


# --- spiking network: spatial-array interface --------------------------------


def test_spiking_pattern_raises_lit_pixels_and_only_relaxes_dark_ones():
    net = _net()
    G0 = net.G_matrix.copy()
    pattern = np.zeros(net.shape)
    pattern[:3, :] = 1.0

    net.apply_spatial_pattern(pattern, 20, 550, 20, pattern_name="top half")
    G1 = net.G_matrix

    assert np.all(G1[:3] > G0[:3])
    assert np.all(G1[3:] <= G0[3:])          # decay toward G_min only
    assert np.all(G1[3:] < G0[3:])
    assert net.current_time == pytest.approx(0.020)
    assert len(net.history['G_matrices']) == 2
    assert net.history['pattern_names'] == ['Initial State', 'top half']
    assert net.history['time_points'] == [0, pytest.approx(0.020)]


def test_spiking_pattern_shape_is_validated():
    with pytest.raises(ValueError):
        _net().apply_spatial_pattern(np.ones((2, 2)), 1, 550, 10)


def test_spiking_reset_restores_initial_state_and_history():
    net = _net()
    G0 = net.G_matrix.copy()
    net.apply_spatial_pattern(np.ones(net.shape), 0.5, 550, 20)
    net.rest(100)
    trains = [net.generate_poisson_spikes(40, 200, seed=i) for i in range(net.n_input)]
    net.simulate(trains, 200)
    assert len(net.history['G_matrices']) == 3
    assert not np.array_equal(net.G_matrix, G0)

    net.reset()

    np.testing.assert_array_equal(net.G_matrix, G0)
    assert net.current_time == 0
    assert net.history['pattern_names'] == ['Initial State']
    assert len(net.history['G_matrices']) == 1
    assert all(n.V == n.V_rest and not n.spike_times for n in net.hidden_neurons)
    assert all(not h for h in net.hidden_spike_history)


def test_spiking_evolution_and_state_plots_run():
    net = _net()
    net.apply_spatial_pattern(np.ones(net.shape), 0.5, 550, 20)
    net.rest(50)
    assert plot_network_evolution(net) is not None
    assert plot_network_state(net) is not None
    plt.close('all')


def test_simulate_is_deterministic_and_leaves_the_spatial_clock_alone():
    """The refactor must not change simulate(); its ms clock is a local."""
    results = []
    for _ in range(2):
        net = _net(seed=3)
        trains = [net.generate_poisson_spikes(40, 300, seed=i)
                  for i in range(net.n_input)]
        results.append(net.simulate(trains, 300)['final_weights'])
        assert net.current_time == 0
        assert len(net.history['G_matrices']) == 1
    np.testing.assert_array_equal(*results)


def test_both_networks_share_one_spatial_implementation():
    assert (SpikingSynapseNetwork.apply_spatial_pattern
            is SynapseNetwork.apply_spatial_pattern)


# --- fit plots draw the fitted model -----------------------------------------


@pytest.fixture(scope="module")
def suite_and_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("plot_suite")
    return pipeline.run(str(out), seed=bl.PIPELINE_SEED, verbose=False)


def _line(fig, colour):
    target = mcolors.to_rgba(colour)
    for ax in fig.axes:
        for ln in ax.lines:
            if mcolors.to_rgba(ln.get_color()) == target:
                return ln
    raise AssertionError(f"no {colour} line in figure")


def test_stdp_and_srdp_lines_are_the_module_level_models(suite_and_model):
    suite, model = suite_and_model
    fig = fitting.plot_fitting_results(suite, model)

    ln = _line(fig, 'darkviolet')
    x = np.asarray(ln.get_xdata(), dtype=float)
    expected = fitting.stdp_window_percent(
        x, model['stdp_A_plus'], model['stdp_A_minus'],
        model['stdp_tau_plus_ms'], model['stdp_tau_minus_ms'])
    np.testing.assert_allclose(ln.get_ydata(), expected, equal_nan=True)

    ln = _line(fig, 'orangered')
    f = np.asarray(ln.get_xdata(), dtype=float)
    expected = fitting.srdp_sigmoid_percent(
        f, model['srdp_baseline'], model['srdp_max_change'],
        model['srdp_transition_freq_hz'], model['srdp_slope'])
    np.testing.assert_allclose(ln.get_ydata(), expected)
    plt.close(fig)


def test_stdp_window_is_float_and_undefined_at_zero():
    out = fitting.stdp_window_percent(np.array([-10, 0, 10]), 0.5, 0.3, 20.0, 20.0)
    assert out.dtype == float
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(0.5 * np.exp(-0.5))
    assert out[0] == pytest.approx(-0.3 * np.exp(-0.5))


def test_srdp_fit_reproduces_its_own_sigmoid_and_round_trips(suite_and_model):
    suite, model = suite_and_model
    assert 'srdp_baseline' in model
    restored = fitting.deserialise_fitted_model(fitting.serialise_fitted_model(model))
    assert restored['srdp_baseline'] == model['srdp_baseline']

    parsed = fitting.DataParser.parse_suite(suite)['srdp']
    f = np.asarray(parsed['frequencies_hz'], dtype=float)
    y = np.asarray(parsed['delta_g_percent'], dtype=float)
    pred = fitting.srdp_sigmoid_percent(
        f, model['srdp_baseline'], model['srdp_max_change'],
        model['srdp_transition_freq_hz'], model['srdp_slope'])
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 == pytest.approx(model['srdp_fit_R2'], abs=1e-6)
