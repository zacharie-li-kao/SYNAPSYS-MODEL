"""The SNN learning loop — C2, C3, C4.

The audit's finding was that this network could not learn at all: LTD was
unreachable dead code, every causally-related spike pair was discarded, and the
synaptic drive was ~84x over, so all hidden neurons fired identically and there
was no timing structure for STDP to read even in principle.
"""

import numpy as np
import pytest

from network import LIFNeuron, SpikingSynapseNetwork

BASE_PARAMS = dict(G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                   lambda_peak=550, lambda_width=100, decay_tau=100,
                   A_peak=8e-4, B_peak=6e-4, variability=0.0)
STDP_PARAMS = dict(stdp_A_plus=0.5, stdp_A_minus=0.3,
                   stdp_tau_plus_ms=20.0, stdp_tau_minus_ms=20.0)


def _net(n_input=4, n_hidden=3):
    return SpikingSynapseNetwork(BASE_PARAMS, STDP_PARAMS,
                                 n_input=n_input, n_hidden=n_hidden, dt_ms=1.0)


# --- C2: LTD must be reachable --------------------------------------------


def test_ltd_decreases_a_weight():
    """C2 — the `delta_t < 0` branch was dead code and A_minus was never used.

    STDP was applied only inside the post-synaptic `if spiked:` block, pairing
    each new post-spike against PAST pre-spikes, so delta_t was never negative.
    """
    net = _net()
    syn = net.synapses[0][0]
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    net.apply_stdp(0, 0, -10.0)   # post 10 ms before pre

    assert syn.G < before, "a negative Δt must depress the synapse"


def test_ltp_increases_a_weight():
    net = _net()
    syn = net.synapses[0][0]
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    net.apply_stdp(0, 0, +10.0)

    assert syn.G > before


def test_the_stdp_rule_matches_the_spec_magnitude():
    """A_plus = 0.5 means 0.5 PERCENT of the dynamic range at Δt → 0.

    The 0.01 is part of the rule as CLAUDE.md states it. A second division by
    100 anywhere makes every weight update 100x too small; omitting the 0.01
    makes it 100x too large and rails every synapse in a handful of pairings.
    """
    net = _net()
    syn = net.synapses[0][0]
    syn.G = (syn.G_min + syn.G_max) / 2
    G_range = syn.G_max - syn.G_min

    delta_G = net.apply_stdp(0, 0, 0.0)

    expected = 0.5 * G_range * 0.01     # 0.5% of range
    assert delta_G == pytest.approx(expected, rel=1e-9)
    # Sanity: that is a small fraction of the range, not a sizeable chunk of it.
    assert delta_G / G_range < 0.01


def test_a_simulation_produces_both_potentiation_and_depression():
    """End to end: the live run that produced 4370 positive pairs and 0 negative.

    Input 0 fires early and input 1 late, so some pairings are causal and some
    are anti-causal, and both signs must appear.
    """
    np.random.seed(20260731)
    net = _net(n_input=12, n_hidden=3)

    # Inputs 0-10 drive the layer during the first 200 ms, so their pairings
    # are predominantly causal and must potentiate.
    #
    # Input 11 fires ONLY after 205 ms, by which time the drive has stopped so
    # no further post-spikes occur. Every pairing it takes part in is therefore
    # post-before-pre, still inside the 100 ms eligibility window, and can only
    # depress. Mixing the two in one train would not isolate LTD: with
    # A_plus = 0.5 against A_minus = 0.3, repeated volleys produce a net
    # increase even for a slightly-late input, which is correct STDP rather
    # than a defect.
    trains = [list(range(5, 200, 6)) for _ in range(11)]
    trains.append(list(range(205, 280, 6)))

    before = net.get_weight_matrix().copy()
    net.simulate(trains, duration_ms=400, learning_enabled=True)
    after = net.get_weight_matrix()

    increased = np.sum(after[:11] > before[:11] + 1e-15)
    decreased_late = np.sum(after[11] < before[11] - 1e-15)

    assert increased > 0, "no causally-driven synapse was potentiated"
    assert decreased_late > 0, (
        "the exclusively anti-causal input's synapses were not depressed — "
        "LTD is still unreachable in the simulation loop even though "
        "apply_stdp supports it.\n"
        f"row 11 before {before[11]}, after {after[11]}"
    )


# --- C3: the causal pair must count ---------------------------------------


def test_a_same_bin_causal_pair_potentiates():
    """C3 — Δt = 0 fell through both branches and was discarded.

    The input spike is appended to history before the neuron update in the same
    timestep, so the pre-spike that actually CAUSED the post-spike is already
    in history with t_pre == t. Verified on a live run: 465 zero-Δt pairs
    against 455 post-spikes — every genuinely causal pair was thrown away, and
    all 4370 weight updates came from acausal older spikes.

    Δt = 0 is now the dt → 0+ limit of the LTP branch, which is what a
    pre-spike that drove the neuron to threshold within one bin deserves.
    """
    net = _net()
    syn = net.synapses[0][0]
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    net.apply_stdp(0, 0, 0.0)

    assert syn.G > before, "the causal same-bin pair contributed nothing"


def test_the_eligibility_window_is_symmetric():
    """The spec's window is (−100, +100) ms for tau = 20 ms, not [0, +100)."""
    net = _net()
    assert net.stdp_window_ms == pytest.approx(100.0)


# --- C4: the synaptic drive ------------------------------------------------


def test_one_synapse_does_not_fire_a_neuron_on_its_own():
    """C4 — `I_syn = G*1e9*0.1` gave dV = 1683 mV against a 20 mV threshold.

    Every hidden neuron fired on the first timestep any single input fired, so
    the post layer was a deterministic OR of the input layer.
    """
    net = _net()
    neuron = LIFNeuron()
    threshold_gap = neuron.V_thresh - neuron.V_rest

    G_mid = (BASE_PARAMS['G_min'] + BASE_PARAMS['G_max']) / 2
    I_syn = G_mid * net.synaptic_gain_nA_per_S

    dV = (-(neuron.V - neuron.V_rest) + neuron.R * I_syn) / neuron.tau_m * net.dt_ms

    assert dV < threshold_gap, (
        f"a single mid-range synapse depolarises by {dV:.1f} mV against a "
        f"{threshold_gap:.1f} mV threshold gap — it fires the neuron alone"
    )
    # And it matches the stated design target rather than being merely "less".
    assert dV == pytest.approx(net.single_synapse_dv_fraction * threshold_gap,
                               rel=1e-9)


def test_enough_coincident_inputs_do_fire_a_neuron():
    """The regime has to be excitable, not merely quiet."""
    net = _net()
    neuron = LIFNeuron()
    G_mid = (BASE_PARAMS['G_min'] + BASE_PARAMS['G_max']) / 2

    n_needed = int(round(1.0 / net.single_synapse_dv_fraction))
    I_syn = n_needed * G_mid * net.synaptic_gain_nA_per_S

    spiked = neuron.update(I_syn, net.dt_ms, 1.0)
    assert spiked, (
        f"{n_needed} coincident mid-range inputs failed to fire the neuron"
    )


def test_hidden_neurons_do_not_all_emit_identical_trains():
    """The consequence of C4 that made STDP pointless.

    With the drive 84x over, all five hidden neurons emitted *identical* spike
    trains (455 spikes = 5 x 91 exactly), so there was no neuron-specific
    timing structure for STDP to read.
    """
    rng = np.random.default_rng(20260731)
    net = _net(n_input=20, n_hidden=4)

    # Dense enough to drive the layer: ~10 coincident mid-range inputs are
    # needed to fire a neuron by design (see the C4 note), so at 20 inputs this
    # is a realistic Poisson regime rather than a starved one.
    trains = [sorted(rng.choice(np.arange(1, 400), 90, replace=False).tolist())
              for _ in range(20)]

    # Give the synapses different weights so the neurons see different drive —
    # with identical weights and shared inputs, identical outputs would be
    # correct behaviour rather than a bug.
    for i in range(net.n_input):
        for j in range(net.n_hidden):
            s = net.synapses[i][j]
            s.G = float(rng.uniform(s.G_min, s.G_max))

    net.simulate(trains, duration_ms=400, learning_enabled=True)

    counts = [len(h) for h in net.hidden_spike_history]
    assert sum(counts) > 0, f"no hidden neuron fired at all (counts {counts})"

    trains_out = [tuple(h) for h in net.hidden_spike_history]
    assert len(set(trains_out)) > 1, (
        f"every hidden neuron emitted an identical spike train (counts {counts}) "
        "— the post layer is still a deterministic function of the input layer"
    )


def test_weights_do_not_saturate_within_a_few_pairings():
    """The Network-tab half of C7.

    With ΔG ≈ 1.6e-5 S per pairing — 17% of the full dynamic range — every
    synapse railed in about six spike pairs.
    """
    net = _net()
    syn = net.synapses[0][0]
    syn.G = (syn.G_min + syn.G_max) / 2

    for _ in range(6):
        net.apply_stdp(0, 0, 1.0)

    assert syn.G < syn.G_max * 0.99, (
        "six pairings saturated the synapse — the update is far too large"
    )


# --- the multi-layer variant carries the same defects ----------------------


def test_multilayer_network_has_a_symmetric_window():
    """network.py:1335-1350 hardcoded the eligibility window as 100 ms."""
    from network import MultiLayerSpikingNetwork

    net = MultiLayerSpikingNetwork([4, 3, 2], BASE_PARAMS, dict(STDP_PARAMS),
                                   dt_ms=1.0)
    # tau = 50 ms would make the correct window 250 ms, not 100.
    net.tau_plus = 50.0
    net.tau_minus = 50.0
    assert net.stdp_window_ms == pytest.approx(250.0), (
        "the eligibility window is hardcoded rather than derived from 5*max(tau)"
    )
