"""VisualSynapse core physics — the ODE, the branch selection, and rest.

C1 is the most user-visible defect in the audit: with no fitted model loaded —
the out-of-the-box state — selecting Depression and clicking the button raised
conductance toward G_max while the GUI reported depression. These tests pin the
behaviour that replaced it.
"""

import numpy as np
import pytest

import fitting
from Main import VisualSynapse


# --- C1: the mode argument must mean something ----------------------------


def test_depression_lowers_conductance_in_the_default_state():
    """C1 — the headline defect.

    The light branch inferred potentiation vs depression purely from the SIGN
    of the spectral response. With no fitted model the fallback response is a
    plain positive Gaussian, so the sign was never negative: "Apply Depression"
    potentiated. `stimulus_type` defaults to 'light', so this was the
    out-of-the-box behaviour, and it made the startup banner's own advertised
    demo ("watch conductance decrease back down") impossible.
    """
    syn = VisualSynapse()
    assert syn.wavelength_curve is None, "default state must have no fitted curve"

    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    syn.apply_stimulus(20, 550, 100, mode='depression', stimulus_type='light')

    assert syn.G < before, (
        f"depression raised G from {before:.3e} to {syn.G:.3e} S"
    )


def test_potentiation_still_raises_conductance_in_the_default_state():
    syn = VisualSynapse()
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    syn.apply_stimulus(20, 550, 100, mode='potentiation', stimulus_type='light')

    assert syn.G > before


def test_the_unsigned_fallback_is_reported_not_hidden():
    """The branch came from the requested mode, not from the wavelength.

    That is a real caveat about what the simulation means, so it is recorded
    rather than left for the user to infer.
    """
    syn = VisualSynapse()
    syn.G = (syn.G_min + syn.G_max) / 2
    syn.apply_stimulus(20, 550, 100, mode='depression', stimulus_type='light')

    report = syn.last_stimulus_report
    assert report is not None
    assert report['unsigned_fallback'] is True
    assert report['actual_mode'] == 'depression'
    assert report['requested_mode'] == 'depression'


def test_a_fitted_signed_curve_lets_the_wavelength_decide():
    """With a real spectral response, the wavelength determines the branch.

    That IS the physics — a device does not depress under its potentiation band
    because the operator asked it to. The conflict is surfaced instead.
    """
    # Positive lobe at 365 nm, negative lobe at 550 nm.
    def curve(wl):
        wl = np.asarray(wl, dtype=float)
        return (np.exp(-((wl - 365) / 30) ** 2)
                - np.exp(-((wl - 550) / 30) ** 2))

    syn = VisualSynapse(wavelength_curve=curve, wavelength_range=[300, 700])
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    # Ask for depression at the POTENTIATION wavelength.
    syn.apply_stimulus(20, 365, 100, mode='depression', stimulus_type='light')

    assert syn.G > before, "the fitted potentiation band must potentiate"
    report = syn.last_stimulus_report
    assert report is not None, "the conflict between intent and physics must be reported"
    assert report['requested_mode'] == 'depression'
    assert report['actual_mode'] == 'potentiation'
    assert report['response'] > 0


def test_a_fitted_curve_depresses_on_its_negative_lobe():
    def curve(wl):
        wl = np.asarray(wl, dtype=float)
        return (np.exp(-((wl - 365) / 30) ** 2)
                - np.exp(-((wl - 550) / 30) ** 2))

    syn = VisualSynapse(wavelength_curve=curve, wavelength_range=[300, 700])
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    syn.apply_stimulus(20, 550, 100, mode='depression', stimulus_type='light')

    assert syn.G < before
    # Requested and actual agree, so there is nothing to report.
    assert syn.last_stimulus_report is None


# --- the network consumer of the same branch ------------------------------


def test_a_spatial_pattern_under_depression_depresses():
    """network.py drives VisualSynapse objects that have no fitted curve.

    Before C1, a whole array under "depression" potentiated.
    """
    from network import SynapseNetwork

    base_params = dict(G_min=1e-6, G_max=1e-4, alpha=0.8, beta=0.8,
                       lambda_peak=550, lambda_width=100, decay_tau=100,
                       A_peak=8e-4, B_peak=6e-4)
    net = SynapseNetwork(base_params, shape=(3, 3), variability=0.0)
    for row in net.synapses:
        for syn in row:
            syn.G = (syn.G_min + syn.G_max) / 2
    net.update_conductance_matrix()
    before = net.G_matrix.copy()

    pattern = np.ones((3, 3))
    net.apply_spatial_pattern(pattern, 20, 550, 100,
                              mode='depression', stimulus_type='light')

    assert np.all(net.G_matrix < before), (
        "a fully-lit depression pattern raised conductance"
    )


# --- rest ------------------------------------------------------------------


def test_rest_decays_toward_g_min():
    """`rest()` passed dt_ms into the stimulus_type slot.

    Every rest therefore routed through the ELECTRICAL branch and integrated at
    the 1 ms default rather than the spec's 10 ms, so dt_ms was dead code and
    each rest generated ten times the history inside a loop that re-renders it.
    """
    syn = VisualSynapse()
    syn.G = (syn.G_min + syn.G_max) / 2
    before = syn.G

    syn.rest(1000)

    assert syn.G < before, "rest must relax the device toward G_min"
    assert syn.G >= syn.G_min


def test_rest_integrates_at_the_spec_timestep():
    """100 ms of rest at the spec's 10 ms step is 10 samples, not 100."""
    syn = VisualSynapse()
    n_before = len(syn.history_G)

    syn.rest(100)

    assert len(syn.history_G) - n_before == 10


def test_rest_drives_no_potentiation():
    """The electrical branch at zero drive must contribute exactly nothing."""
    syn = VisualSynapse()
    syn.G = syn.G_min * 1.01
    before = syn.G
    syn.rest(50)
    assert syn.G <= before


# --- integration bookkeeping ----------------------------------------------


def test_a_sub_timestep_stimulus_raises_rather_than_doing_nothing():
    """`steps = int(duration/dt)` made any sub-dt stimulus a silent no-op."""
    syn = VisualSynapse()
    with pytest.raises(ValueError, match="shorter than the integration step"):
        syn.apply_stimulus(20, 550, 0.5, mode='potentiation',
                           stimulus_type='light', dt_ms=1.0)


def test_history_timestamps_are_end_of_step():
    """Each stimulus finished one dt short and duplicated the previous timestamp."""
    syn = VisualSynapse()
    syn.apply_stimulus(20, 550, 10, mode='potentiation',
                       stimulus_type='light', dt_ms=1.0)

    times = syn.history_t
    assert times[0] == 0
    assert times[-1] == pytest.approx(0.010), (
        f"10 ms of stimulus ended at t = {times[-1]:.4f} s"
    )
    assert len(set(times)) == len(times), "duplicate timestamps in history"


def test_initial_state_is_inside_the_physical_bounds():
    """G = G_min * 1.5 lies outside [G_min, G_max] for an ON/OFF ratio below 1.5.

    The first clip then silently pinned such a device to G_max.
    """
    syn = VisualSynapse(G_min=1e-5, G_max=1.2e-5)   # ON/OFF ratio 1.2
    assert syn.G_min <= syn.G <= syn.G_max

    syn.reset()
    assert syn.G_min <= syn.G <= syn.G_max


def test_clipping_keeps_conductance_within_bounds_under_hard_drive():
    syn = VisualSynapse()
    syn.apply_stimulus(1000, 550, 5000, mode='potentiation', stimulus_type='light')
    assert syn.G <= syn.G_max
    assert syn.G >= syn.G_min

    syn.apply_stimulus(1000, 550, 5000, mode='depression', stimulus_type='light')
    assert syn.G >= syn.G_min


# --- C8: the GUI must not present a substituted value as a measurement -----


def test_a_fitted_parameter_is_reported_as_fitted():
    from Main import resolve_fitted_parameter

    model = {
        'alpha': 0.75,
        'extraction_report': {'extracted': ['nonlinearity'],
                              'defaults_used': [], 'failed': []},
    }
    value, state = resolve_fitted_parameter(model, 'alpha', 0.8, ('alpha', 'nonlinearity'))
    assert value == 0.75
    assert state == 'fitted'


def test_a_defaulted_parameter_is_flagged_even_though_it_is_present():
    """C8 — this is the case `.get(key, default)` could not detect.

    The key IS in the model, carrying the extractor's hardcoded fallback. Only
    the extraction report distinguishes it from a measurement, and the loader
    read the report not at all.
    """
    from Main import resolve_fitted_parameter

    model = {
        'decay_tau': 100.0,     # numerically identical to the documented default
        'extraction_report': {'extracted': [], 'defaults_used': ['decay_tau'],
                              'failed': []},
    }
    value, state = resolve_fitted_parameter(model, 'decay_tau', 100.0,
                                            ('decay_tau', 'retention'))
    assert value == 100.0
    assert state == 'DEFAULT', (
        "a substituted default is indistinguishable from a measurement"
    )


def test_a_failed_extractor_marks_its_parameter_untrustworthy():
    from Main import resolve_fitted_parameter

    model = {
        'B_peak': 9.9e-3,
        'extraction_report': {'extracted': [], 'defaults_used': [],
                              'failed': ['B_peak']},
    }
    _, state = resolve_fitted_parameter(model, 'B_peak', 6e-4)
    assert state == 'FAILED'


def test_an_absent_parameter_is_flagged_not_silently_substituted():
    from Main import resolve_fitted_parameter

    value, state = resolve_fitted_parameter({}, 'alpha', 0.8)
    assert value == 0.8
    assert state == 'MISSING'


def test_an_extractor_governing_several_parameters_flags_all_of_them():
    """The wavelength fit produces both lambda_peak and lambda_width, but the
    report names the EXTRACTOR, not each output."""
    from Main import resolve_fitted_parameter

    model = {
        'lambda_peak': 550.0, 'lambda_width': 100.0,
        'extraction_report': {'extracted': [], 'failed': [],
                              'defaults_used': ['lambda_peak', 'lambda_width']},
    }
    for key in ('lambda_peak', 'lambda_width'):
        _, state = resolve_fitted_parameter(model, key, 0.0,
                                            (key, 'wavelength_response'))
        assert state == 'DEFAULT'


def test_the_loader_reads_the_report_rather_than_defaulting_silently():
    """Guards the display side, which needs a file dialog to reach directly."""
    import inspect

    import Main

    source = inspect.getsource(
        Main.SynapseSimulatorApp.load_experimental_synapse_model)

    assert "resolve_fitted_parameter" in source, (
        "the loader no longer resolves provenance"
    )
    # The silent pattern must not come back.
    for silent in ("fitted_model.get('alpha', 0.8)",
                   "fitted_model.get('decay_tau', 100)",
                   "fitted_model.get('B_peak', 6e-4)"):
        assert silent not in source, f"silent default reinstated: {silent}"

    # And the fit-quality figures the audit found discarded must be displayed.
    for r2_key in ("wavelength_fit_R2", "stdp_fit_R2", "srdp_fit_R2"):
        assert r2_key in source, f"{r2_key} is discarded again"

    assert "NOT EVERYTHING IN THIS MODEL WAS MEASURED" in source


# --- the M4 selection rule must not starve a short trace -------------------


def test_nonlinearity_survives_a_short_noisy_saturating_trace():
    """The headroom-based sample selection must not drop below the 5-point guard.

    Selection now excludes samples whose starting conductance is within 3σ of
    the maximum attained. On a short, noisy, strongly-saturating run that could
    in principle leave too few samples — in which case the extractor returns
    None and alpha silently becomes unavailable.
    """
    rng = np.random.default_rng(20260731)
    G_min, G_max = 1e-6, 1e-4
    G = G_min * 1.5
    trace = [G]
    for _ in range(60):
        G = min(G + 8e-4 * 20 * (G_max - G) ** 0.75 * 0.1, G_max)
        trace.append(G * (1 + rng.normal(0, 0.015)))

    dataset = {
        'filename': 'short_saturating.csv',
        'x_data': list(range(len(trace))),
        'y_data': trace,
        'metadata': {'experiment_type': 'potentiation'},
        'derived_metrics': {},
    }

    nonlinearity = fitting.DataParser._build_nonlinearity(dataset)
    assert nonlinearity is not None, (
        "a 60-pulse saturating trace yielded too few samples to fit alpha"
    )
    assert len(nonlinearity['G_initial_S']) >= 5

    result = fitting.fit_nonlinearity_exponent(nonlinearity, G_max=G_max)
    assert 0.4 < result['alpha'] < 1.2


def test_rest_integrates_the_full_duration_even_when_not_a_multiple_of_dt():
    """rest(45, dt=10) used to advance 40 ms and a 1.9 ms stimulus took 1 ms."""
    syn = VisualSynapse(decay_tau=10.0)
    syn.rest(45, dt_ms=10.0)
    assert syn.history_t[-1] == pytest.approx(0.045, abs=1e-12)
    syn2 = VisualSynapse()
    syn2.apply_stimulus(1.0, 550, 1.9, dt_ms=1.0)
    assert syn2.history_t[-1] == pytest.approx(0.0019, abs=1e-12)


def test_retention_fit_raises_on_data_that_do_not_decay():
    """A rising readout must raise, not return tau ~ 4e10 s with negative R2."""
    t = np.arange(0.0, 7200.0, 5.0)
    rising = 8e-7 + 7e-8 * (1 - np.exp(-t / 2000.0))
    with pytest.raises(RuntimeError):
        fitting.fit_retention_decay(dict(time_s=t, conductance_S=rising, g_initial_S=rising[0]))
    decaying = 1e-6 + 5e-7 * np.exp(-t / 1500.0)
    out = fitting.fit_retention_decay(dict(time_s=t, conductance_S=decaying, g_initial_S=decaying[0]))
    assert out['fit_quality_R2'] > 0.9 and out['decay_tau_s'] > 0
