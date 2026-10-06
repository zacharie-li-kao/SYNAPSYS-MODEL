"""P4 cycle integrity (H8, H9, H11, H12) and P6 device variability (M14).

H9 and M14 are the two remaining findings that silently produce physically
impossible or misaligned DATA, rather than merely misreporting it.
"""

import inspect

import numpy as np
import pytest

import synapse_cycle as sc
import synapse_engine as se
from network import SynapseNetwork

BASE_PARAMS = dict(G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                   lambda_peak=550, lambda_width=100, decay_tau=100,
                   A_peak=8e-4, B_peak=6e-4)


def _train(n_pulses=5, **overrides):
    cfg = {
        'topology': 'single',
        'write_ch': 'smua',
        'read_ch': 'smua',
        'stim_drive_type': 'V',
        'stim_level': 1.0,
        'stim_width_ms': 1.0,
        'stim_period_ms': 10.0,
        'n_pulses': n_pulses,
        'read_voltage': 0.1,
        'read_delay_ms': 2.0,
        'compliance_A': 1e-3,
        'nplc': 0.1,
        'settle_ms': 1.0,
        'measure_avg': 1,
    }
    cfg.update(overrides)
    return cfg


# --- M14: device variability ----------------------------------------------


@pytest.mark.parametrize("variability", [0.1, 0.5, 1.0])
def test_variability_never_produces_an_unphysical_device(variability):
    """M14 — `normal(1.0, variability)` is unbounded below.

    At the GUI-permitted 100%, out of 25 devices: 5 had negative G_min, 8 had
    negative A_peak (potentiation running backwards), 5 had negative decay_tau
    (runaway rather than relaxation), and 7 held NEGATIVE CONDUCTANCE after one
    pulse.
    """
    np.random.seed(20260731)
    net = SynapseNetwork(BASE_PARAMS, shape=(6, 6), variability=variability)

    for row in net.synapses:
        for syn in row:
            assert syn.G_min > 0, "negative or zero G_min"
            assert syn.G_max > 0, "negative or zero G_max"
            assert syn.G_min < syn.G_max, "inverted conductance bounds"
            assert syn.A_peak > 0, "negative A_peak — potentiation runs backwards"
            assert syn.B_peak > 0, "negative B_peak"
            assert syn.decay_tau > 0, "negative decay_tau — relaxation runs away"
            assert syn.alpha > 0 and syn.beta > 0
            assert syn.G_min <= syn.G <= syn.G_max


@pytest.mark.parametrize("variability", [0.1, 0.5, 1.0])
def test_conductance_stays_physical_after_stimulation(variability):
    """The consequence the audit actually measured."""
    np.random.seed(20260731)
    net = SynapseNetwork(BASE_PARAMS, shape=(5, 5), variability=variability)

    net.apply_spatial_pattern(np.ones((5, 5)), 20, 365, 100,
                              mode='potentiation', stimulus_type='light')
    assert np.all(net.G_matrix > 0), "negative conductance after one pulse"

    net.apply_spatial_pattern(np.ones((5, 5)), 20, 550, 100,
                              mode='depression', stimulus_type='light')
    assert np.all(net.G_matrix > 0), "negative conductance after depression"


def test_the_variation_factor_has_unit_mean_and_the_requested_spread():
    """The GUI control says "percent variability"; it must mean that."""
    np.random.seed(20260731)
    for variability in (0.1, 0.3):
        draws = np.array([SynapseNetwork._variation_factor(variability)
                          for _ in range(20000)])
        assert np.all(draws > 0)
        assert np.mean(draws) == pytest.approx(1.0, abs=0.02)
        assert np.std(draws) == pytest.approx(variability, rel=0.10)


def test_zero_variability_is_exactly_uniform():
    net = SynapseNetwork(BASE_PARAMS, shape=(4, 4), variability=0.0)
    g_mins = [s.G_min for row in net.synapses for s in row]
    assert len(set(g_mins)) == 1


# --- H9: cycle de-interleaving --------------------------------------------


def _raw(n_points, same_buffer=True):
    ts = [i * 0.01 for i in range(n_points)]
    curr = [1e-6 * (i + 1) for i in range(n_points)]
    data = {'train_a': {'timestamps': ts, 'currents': curr}}
    if not same_buffer:
        data['train_b'] = {'timestamps': list(ts), 'currents': list(curr)}
    return data


def test_a_complete_buffer_yields_every_cycle():
    train_a = _train(n_pulses=4)     # 5 points per train
    train_b = _train(n_pulses=4)
    n_cycles = 3
    stride = 5 + 5

    results = sc._parse_cycle_raw_data(
        _raw(n_cycles * stride), {}, train_a, train_b, n_cycles,
        inter_train_delay_ms=0, inter_cycle_delay_ms=0,
    )
    assert results["n_cycles_parsed"] == 3
    assert results["truncated"] is False
    assert len(results["cycles"]) == 3


def test_a_short_buffer_does_not_yield_misaligned_cycles():
    """H9 — slicing by a fixed stride with no length check.

    Python slicing silently returns a short list past the end, so a truncated
    buffer produced cycles parsed at a progressively wrong offset, each built
    partly from its neighbour's data. `_retrieve_buffer` already detects a
    short buffer, warns, and returns it anyway, and the summary then reported
    all cycles as successful.
    """
    train_a = _train(n_pulses=4)
    train_b = _train(n_pulses=4)
    n_cycles = 4
    stride = 10

    # Only 2.5 cycles' worth of data arrived.
    results = sc._parse_cycle_raw_data(
        _raw(25), {}, train_a, train_b, n_cycles,
        inter_train_delay_ms=0, inter_cycle_delay_ms=0,
    )

    assert results["n_cycles_requested"] == 4
    assert results["n_cycles_parsed"] == 2, (
        "a partial cycle was parsed — its data straddles a cycle boundary"
    )
    assert results["truncated"] is True

    # Every cycle that IS reported must be complete.
    for cycle in results["cycles"]:
        assert len(cycle["potentiation"]["conductance_S"]) == 5
        assert len(cycle["depression"]["conductance_S"]) == 5


def test_a_short_second_buffer_limits_the_dual_channel_parse():
    train_a = _train(n_pulses=4)
    train_b = _train(n_pulses=4, write_ch='smub', read_ch='smub')

    raw = _raw(20, same_buffer=False)
    raw['train_b']['timestamps'] = raw['train_b']['timestamps'][:12]
    raw['train_b']['currents'] = raw['train_b']['currents'][:12]

    results = sc._parse_cycle_raw_data(
        raw, {}, train_a, train_b, 4,
        inter_train_delay_ms=0, inter_cycle_delay_ms=0,
    )
    # Train A holds 4 cycles, Train B only 2 — the smaller governs.
    assert results["n_cycles_parsed"] == 2
    assert results["truncated"] is True


# --- H8 / H12: instrument state must not leak between runs -----------------


def test_the_cycle_script_always_sets_the_filter_state():
    """H8 — the filter was written only when measure_avg > 1.

    Instrument settings persist, so an aborted run that left
    `filter.enable = ON, count = 8` leaked 8x averaging into the next run, with
    the timing budget blown and nothing in the data recording it.
    """
    lines = se._emit_measurement_state("smua", 1)
    joined = "\n".join(lines)
    assert "FILTER_OFF" in joined, "the filter is not explicitly disabled"
    assert "measure.count = 1" in joined, "measure.count is never written"

    lines = se._emit_measurement_state("smua", 8)
    joined = "\n".join(lines)
    assert "FILTER_ON" in joined
    assert "filter.count = 8" in joined


def test_the_averaging_ceiling_is_the_hardware_limit():
    """H12 — the generator clamped to 50 while the warning shown to the user
    was computed from the unclamped value, so the user was warned about a
    constraint that did not exist and was not told their averaging was halved.
    """
    lines = se._emit_measurement_state("smua", 80)
    assert "filter.count = 80" in "\n".join(lines), (
        "80 samples was silently reduced — the hardware limit is 100"
    )

    lines = se._emit_measurement_state("smua", 500)
    assert f"filter.count = {se.FILTER_COUNT_MAX}" in "\n".join(lines)


# --- H11: shared channels with incompatible settings -----------------------


def test_two_trains_sharing_a_channel_with_different_ranges_is_refused():
    """H11 — channel config is emitted once, so the second train's ranges,
    compliance and filter were silently discarded. Train A at 1.0 V then
    Train B at 15 V on the same SMU left B clipping at the 2 V range with no
    error anywhere."""
    train_a = _train(stim_level=1.0)
    train_b = _train(stim_level=15.0)     # same channel, very different range

    with pytest.raises(ValueError, match="share SMU channel"):
        se.generate_cycle_lua_script(train_a, train_b, n_cycles=2,
                                     inter_train_delay_ms=0,
                                     inter_cycle_delay_ms=0)


def test_two_trains_sharing_a_channel_with_matching_settings_is_fine():
    train_a = _train(stim_level=1.0)
    train_b = _train(stim_level=1.0)
    # Returns (lua_script, buffer_map).
    script, buffer_map = se.generate_cycle_lua_script(
        train_a, train_b, n_cycles=2,
        inter_train_delay_ms=0, inter_cycle_delay_ms=0)
    assert buffer_map, "no buffer map produced"

    # Both trains must actually appear, and the timing architecture CLAUDE.md
    # mandates must be intact: one timer reset, a single monotonic accumulator,
    # and no named script.
    assert "Train A" in script and "Train B" in script
    assert script.count("timer.reset()") == 1, (
        "the cycle script must reset the timer exactly once, at the start"
    )
    assert "t_abs" in script, "the monotonic time accumulator is missing"
    assert "loadscript " not in script, "named scripts are forbidden"
    assert "script.delete" not in script
