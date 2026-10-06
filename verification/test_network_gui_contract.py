"""The Network tab's contract with network.py — C12, M17, M18.

These are GUI-adjacent but none of them need a running GUI: each is a mismatch
between what `Main.py` calls and what `network.py` provides.
"""

import inspect
import re

import numpy as np
import pytest

import Main
from network import SpatialPatterns, SpikingSynapseNetwork, SynapseNetwork

BASE_PARAMS = dict(G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                   lambda_peak=550, lambda_width=100, decay_tau=100,
                   A_peak=8e-4, B_peak=6e-4)
STDP_PARAMS = dict(stdp_A_plus=0.5, stdp_A_minus=0.3,
                   stdp_tau_plus_ms=20.0, stdp_tau_minus_ms=20.0)


# --- C12 ------------------------------------------------------------------


def test_spiking_network_provides_everything_the_network_tab_reads():
    """C12 — "Initialize Network" in spiking mode raised AttributeError.

    `Main.py` reads `n_synapses` unconditionally on the statement right after
    constructing the network, then `get_statistics()`, `G_matrix` and `rest()`.
    None existed on SpikingSynapseNetwork: the tab was dead on arrival in that
    mode.
    """
    net = SpikingSynapseNetwork(BASE_PARAMS, STDP_PARAMS,
                                n_input=6, n_hidden=4, dt_ms=1.0)

    assert net.n_synapses == 24
    assert net.shape == (6, 4)

    stats = net.get_statistics()
    for key in ("mean_G", "std_G", "min_G", "max_G"):
        assert key in stats
        assert np.isfinite(stats[key])

    assert net.G_matrix.shape == (6, 4)

    net.rest(100)   # must not raise


def test_the_two_network_classes_agree_on_units():
    """G_matrix is siemens on both; get_weight_matrix is µS.

    Confusing the two is what made plot_weight_evolution draw values 1e6 too
    large.
    """
    spiking = SpikingSynapseNetwork(BASE_PARAMS, STDP_PARAMS,
                                    n_input=4, n_hidden=3, dt_ms=1.0)
    rate = SynapseNetwork(BASE_PARAMS, shape=(4, 3), variability=0.0)

    assert 1e-7 < np.mean(spiking.G_matrix) < 1e-3
    assert 1e-7 < np.mean(rate.G_matrix) < 1e-3
    assert np.mean(spiking.get_weight_matrix()) == pytest.approx(
        np.mean(spiking.G_matrix) * 1e6, rel=1e-9
    )


def test_the_spiking_network_honours_the_variability_setting():
    """The constructor hardcoded 0.1 and ignored the GUI control."""
    params = dict(BASE_PARAMS, variability=0.0)
    net = SpikingSynapseNetwork(params, STDP_PARAMS, n_input=5, n_hidden=5)
    assert np.std(net.G_matrix) == pytest.approx(0.0, abs=1e-18)

    params = dict(BASE_PARAMS, variability=0.2)
    net = SpikingSynapseNetwork(params, STDP_PARAMS, n_input=8, n_hidden=8)
    assert np.std(net.G_matrix) > 0


def test_initial_conductances_start_mid_range_and_inside_bounds():
    """G_init_mean was (G_min + G_max)/3 — undocumented, and reading like a
    typo for /2 — and the variability draw was never clipped."""
    net = SpikingSynapseNetwork(dict(BASE_PARAMS, variability=0.5),
                                STDP_PARAMS, n_input=10, n_hidden=10)
    G = net.G_matrix
    assert np.all(G >= BASE_PARAMS['G_min'])
    assert np.all(G <= BASE_PARAMS['G_max'])

    net = SpikingSynapseNetwork(dict(BASE_PARAMS, variability=0.0),
                                STDP_PARAMS, n_input=4, n_hidden=4)
    expected_mid = (BASE_PARAMS['G_min'] + BASE_PARAMS['G_max']) / 2
    assert np.mean(net.G_matrix) == pytest.approx(expected_mid, rel=1e-9)


# --- M17 ------------------------------------------------------------------


def test_the_network_demo_targets_a_tab_that_exists():
    """M17 — `notebook.set("Network (3×3)")` against a tab named "Network".

    It was the handler's first statement, so it raised immediately and left the
    button permanently disabled reading "Running Demo...".
    """
    source = inspect.getsource(Main)
    assert 'notebook.set("Network (3×3)")' not in source

    tab_names = set(re.findall(r'notebook\.add\("([^"]+)"\)', source))
    targets = set(re.findall(r'notebook\.set\("([^"]+)"\)', source))
    unknown = targets - tab_names
    assert not unknown, f"notebook.set targets non-existent tabs: {unknown}"


# --- M18 ------------------------------------------------------------------


def test_every_offered_pattern_can_actually_be_built():
    """M18 — "Top Half" was in the dropdown with no dispatcher branch.

    It fell through to the uniform fallback while the dialog went on reporting
    the requested name, so the whole array was illuminated and labelled
    "Top Half".
    """
    shape = (5, 5)
    for name, builder in SpatialPatterns.get_all_patterns().items():
        if builder is None:
            # Only the Excel pattern may have no builder — its content comes
            # from a user file.
            assert name == 'Custom (from Excel)'
            continue
        pattern = builder(shape)
        assert pattern.shape == shape, f"{name} produced the wrong shape"
        assert np.any(pattern != 0), f"{name} produced an all-zero pattern"


def test_the_gui_dispatcher_covers_every_offered_pattern():
    """The dropdown and the dispatcher must not drift apart again."""
    source = inspect.getsource(Main)
    dispatched = set(re.findall(r'pattern_name == "([^"]+)"', source))

    offered = set(SpatialPatterns.get_all_patterns())
    missing = offered - dispatched
    assert not missing, (
        f"offered in the pattern list but absent from the dispatcher: {missing}"
    )


def test_an_unknown_pattern_name_does_not_silently_become_uniform():
    """The fallback substituted a different pattern under the chosen name."""
    source = inspect.getsource(Main)
    # The uniform fallback in the final `else` is gone.
    assert "# Fallback\n                pattern = SpatialPatterns.uniform" not in source
    assert "Unknown Pattern" in source, (
        "the dispatcher must report an unimplemented pattern rather than "
        "substituting one"
    )


def test_top_half_is_the_top_half():
    pattern = SpatialPatterns.top_half((6, 4))
    assert np.all(pattern[:3, :] == 1.0)
    assert np.all(pattern[3:, :] == 0.0)

