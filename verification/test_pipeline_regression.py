"""End-to-end synthetic pipeline: generate → convert → fit.

Guards the parameters that are currently correct, and pins the ones that are
not, so that landing a fix from AUDIT_2026-07-31.md §9 shows up as a specific,
named test failure rather than a number drifting unnoticed.

The generator runs once per session — it writes 26 files and takes a few
seconds, and every test reads the same fitted model.
"""

import math

import pytest

import baseline as bl
import pipeline


@pytest.fixture(scope="session")
def fitted(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_suite")
    suite, model = pipeline.run(str(out), seed=bl.PIPELINE_SEED, verbose=False)
    return suite, model


# --- C5 / H22: the generator must write everything, on a stock console -----


def test_generator_writes_the_complete_suite(fitted):
    """All 26 files, including the three that a console-encoding fault once ate.

    `pipeline.generate` raises if METADATA_INSTRUCTIONS.txt, the STDP CSV or the
    SRDP CSV are missing, so reaching this fixture at all proves C5 is fixed.
    Asserted explicitly anyway, because it is the precondition for everything
    else in this file.
    """
    suite, _ = fitted
    assert suite["n_datasets"] >= 24, f"only {suite['n_datasets']} datasets converted"
    assert suite["tool_version"] == "2.0"


def test_metadata_survives_conversion(fitted):
    """The light-intensity key must not silently vanish.

    `data_converter.parse_metadata_file` opens the file with no encoding and
    matches "Light intensity (mW/cm²)" by exact string. If that key stops
    matching — re-saving METADATA_INSTRUCTIONS.txt as UTF-8 is enough — the
    intensity is dropped and the fitter substitutes 20 mW/cm² with no warning,
    quietly corrupting A_peak and B_peak.
    """
    suite, _ = fitted

    # Pinned counts, verified against an actual run. A count rather than a
    # non-empty check: if the "mW/cm²" key stops matching, every intensity
    # disappears at once and the count goes to zero, but a filter that itself
    # matched nothing would also produce an empty list and pass a weaker test.
    stim_types = [d["metadata"].get("stimulus_type", "<absent>")
                  for d in suite["datasets"]]
    assert stim_types.count("light") == 21, (
        f"expected 21 purely light-driven datasets, got {stim_types.count('light')}. "
        f"Distinct values present: {sorted(set(stim_types))}"
    )

    with_intensity = [d["filename"] for d in suite["datasets"]
                      if "light_intensity_mW_cm2" in d["metadata"]]
    assert len(with_intensity) == 21, (
        f"expected 21 datasets carrying light_intensity_mW_cm2, got "
        f"{len(with_intensity)}. A drop to 0 means the "
        '"Light intensity (mW/cm²)" key stopped matching — re-saving '
        "METADATA_INSTRUCTIONS.txt as UTF-8 is enough to cause this, and the "
        "fitter then substitutes 20 mW/cm² with no warning."
    )

    # Experiment types the fitter dispatches on must all be present, or a
    # passing fit is only evidence that the extractor was never reached.
    exp_types = {d["metadata"].get("experiment_type") for d in suite["datasets"]}
    for required in ("potentiation", "depression", "retention", "stdp", "srdp"):
        assert required in exp_types, (
            f"no dataset of experiment_type {required!r} survived conversion"
        )


# --- parameters that are correct today ------------------------------------


@pytest.mark.parametrize(
    "key", [k for k, v in bl.BASELINE.items() if v["status"] == bl.PINNED]
)
def test_pinned_parameters_stay_correct(fitted, key):
    _, model = fitted
    spec = bl.BASELINE[key]
    value = model.get(key)
    assert value is not None, f"{key} missing from the fitted model"

    if "min_value" in spec:
        assert value >= spec["min_value"], (
            f"{key} = {value:.4f} fell below {spec['min_value']}"
        )
    else:
        truth = spec["truth"]
        rel = abs(value - truth) / abs(truth)
        assert rel <= spec["rel_tol"], (
            f"{key} = {value:.6g} is {rel:.1%} from ground truth {truth:.6g} "
            f"(tolerance {spec['rel_tol']:.0%})"
        )


# --- live audit findings, pinned at their current wrong values ------------


_KNOWN_BAD_KEYS = [k for k, v in bl.BASELINE.items() if v["status"] == bl.KNOWN_BAD]


def test_every_baseline_parameter_has_a_status():
    """No entry may sit outside the PINNED / KNOWN_BAD classification.

    Also makes the "no known-bad parameters left" state VISIBLE. An empty
    parametrize list silently skips, which reads in the output as though the
    check had run — the opposite of what an empty list actually means here.
    """
    for key, spec in bl.BASELINE.items():
        assert spec["status"] in (bl.PINNED, bl.KNOWN_BAD), (
            f"{key} has an unrecognised status {spec['status']!r}"
        )

    if not _KNOWN_BAD_KEYS:
        # Every parameter in the table is now correct. Kept as an explicit
        # assertion rather than a skip so the state is stated, not inferred.
        assert all(v["status"] == bl.PINNED for v in bl.BASELINE.values())


@pytest.mark.skipif(not _KNOWN_BAD_KEYS,
                    reason="no KNOWN_BAD parameters remain — every pinned "
                           "audit finding is fixed")
@pytest.mark.parametrize("key", _KNOWN_BAD_KEYS)
def test_known_bad_parameters_are_still_wrong(fitted, key):
    """Fails when the finding is fixed — that is the point.

    On failure: confirm the new value is right, then flip `status` to PINNED in
    baseline.py and set `baseline` to the new value.
    """
    _, model = fitted
    spec = bl.BASELINE[key]
    value = model.get(key)
    assert value is not None, f"{key} missing from the fitted model"

    recorded = spec["baseline"]
    rel = abs(value - recorded) / abs(recorded)
    assert rel <= spec["rel_tol"], (
        f"{key} moved from its recorded broken baseline {recorded:.6g} to "
        f"{value:.6g}. Target is {spec['target']:.6g}.\n"
        f"Finding: {spec['finding']}\n"
        f"If this is the fix landing, update baseline.py: status → PINNED."
    )


def test_alpha_is_fitted_from_the_dedicated_nonlinearity_trace(fitted):
    """§0.3 — FIXED, now guarded.

    `fitting.py` files `potentiation_depression_cycle` datasets into
    `potentiation_data` (a cycle does contain potentiation) and then selected
    the LONGEST one, so the 300-point cycling trace beat the 101-point
    dedicated nonlinearity trace. `_build_nonlinearity` then sampled across the
    potentiation/depression phase boundary, where ΔG changes sign, and survived
    the `len < 5` guard by exactly one point.

    A dedicated single-phase measurement must now win regardless of length. The
    longest potentiation-like dataset in this suite is still the cycling trace,
    which is what makes this test meaningful — selection is by kind, not size.
    """
    suite, _ = fitted
    import fitting

    parsed = fitting.DataParser.parse_suite(suite)
    nonlinearity = parsed["nonlinearity"]

    assert nonlinearity["source_is_cycling_trace"] is False, (
        "alpha is being fitted from a cycling trace again"
    )
    assert "nonlinearity" in nonlinearity["source_filename"], (
        f"alpha was fitted from {nonlinearity['source_filename']!r}, not the "
        "dedicated nonlinearity measurement"
    )

    # The preference must be by kind, not by luck of length.
    pot_like = [d for d in suite["datasets"]
                if d["metadata"].get("experiment_type", "").lower()
                in ("potentiation", "potentiation_depression_cycle")]
    longest = max(pot_like, key=lambda d: len(d["y_data"]))
    assert longest["filename"] != nonlinearity["source_filename"], (
        "this suite no longer has a longer cycling trace, so the test cannot "
        "distinguish 'preferred single-phase' from 'happened to be longest'"
    )


# --- STDP: scale, not value ------------------------------------------------


def test_stdp_amplitudes_match_the_spec_values(fitted):
    """C7 — FIXED. Amplitudes were 38× and 21.7× the spec values.

    They were normalised by the BASELINE CONDUCTANCE G_initial (50.4 µS)
    instead of the DYNAMIC RANGE G_max − G_min (99 µS). Those are different
    quantities, the former is operating-point dependent, and no constant
    divisor reconciles them — the fitted amplitude could never equal the ground
    truth whatever rescaling was applied afterwards.
    """
    _, model = fitted
    for key in ("stdp_A_plus", "stdp_A_minus"):
        value = model[key]
        truth = bl.STDP_TRUTH[key]
        assert abs(value - truth) / truth <= 0.30, (
            f"{key} = {value:.4g} is {abs(value - truth) / truth:.0%} from the "
            f"spec value {truth} (percent of dynamic range)"
        )


def test_stdp_time_constants_are_recovered(fitted):
    """C7, the τ half — FIXED, now guarded against regression.

    The sign-selective filter discarded every point with value ≤ 0. At large
    |Δt| the true signal is ~0 and noise dominates symmetrically, so the filter
    removed only the low half of that scatter and kept the high half — a biased
    estimator that produced τ₊ = 14.0 ms and τ₋ = 42.0 ms against a true 20 ms,
    at a respectable-looking R² of 0.914.

    Fitting all points by least squares is unbiased. Both τ now land near 20 ms
    and, importantly, near EACH OTHER — the generator uses the same 20 ms for
    both branches, so a large asymmetry is itself the signature of the bug.
    """
    _, model = fitted
    tau_plus = model["stdp_tau_plus_ms"]
    tau_minus = model["stdp_tau_minus_ms"]

    for name, tau in (("tau_plus", tau_plus), ("tau_minus", tau_minus)):
        assert abs(tau - 20.0) / 20.0 <= 0.30, (
            f"{name} = {tau:.1f} ms is more than 30% from the true 20 ms"
        )

    asymmetry = abs(tau_plus - tau_minus) / 20.0
    assert asymmetry <= 0.25, (
        f"τ₊ = {tau_plus:.1f} ms and τ₋ = {tau_minus:.1f} ms differ by "
        f"{asymmetry:.0%} of the true value, though the generator uses 20 ms "
        "for both. A large asymmetry is the signature of a sign-selective "
        "filter biasing one branch."
    )


def test_stdp_amplitudes_are_denominated_in_percent_of_dynamic_range(fitted):
    """C7, the denominator half — the unit contract, asserted structurally.

    The amplitude must be normalised against the DYNAMIC RANGE (G_max − G_min),
    not the baseline conductance G_initial. Those are different quantities
    (99 µS vs 50.4 µS on this suite), the latter is operating-point dependent,
    and no constant rescaling reconciles them — so the fitted amplitude could
    never equal the ground truth whatever factor was applied afterwards.
    """
    suite, model = fitted
    import fitting

    parsed = fitting.DataParser.parse_suite(suite)
    stdp = parsed["stdp"]

    assert stdp["normalisation"] == "percent_of_dynamic_range"

    dr = parsed["dynamic_range"]
    expected_dr = dr["G_max_S"] - dr["G_min_S"]
    assert stdp["dynamic_range_S"] == pytest.approx(expected_dr, rel=1e-9)

    # And the denominator is genuinely the range, not the operating point.
    g_initial = stdp["g_initial_S"][0]
    assert abs(stdp["dynamic_range_S"] - g_initial) / g_initial > 0.2, (
        "dynamic range and baseline conductance are too close for this test to "
        "distinguish them on this data"
    )


def test_stdp_normalisation_refuses_to_invent_a_denominator():
    """No dynamic range means the amplitude has no defined units.

    Fabricating a denominator would produce a number that looks like a
    percentage of range but is not one — exactly the defect being fixed.
    """
    import fitting

    dataset = {
        "x_data": [-50, -20, 20, 50],
        "y_data": [4.9e-5, 4.5e-5, 5.5e-5, 5.1e-5],
        "metadata": {"experiment_type": "stdp"},
    }
    with pytest.raises(ValueError, match="dynamic range"):
        fitting.DataParser._build_stdp(dataset, None)


def test_stdp_r2_survives_a_whole_millisecond_sweep_at_spec_amplitudes():
    """M3 — FIXED, now guarded. Hit directly rather than through the pipeline.

    A hardware sweep at whole-ms steps yields an *integer* Δt array.
    `stdp_model` builds its output with `np.zeros_like(dt)`, which then also
    has integer dtype, so the float exponentials truncate on assignment. At the
    current inflated amplitudes (~11) the truncation merely coarsens the curve;
    at the spec amplitudes C7 will restore (~0.5) every value truncates to
    zero, the predicted curve is flat, and R² collapses — while every fitted
    parameter still looks plausible.

    Built from the exact analytic STDP window, so a correct implementation
    fits it essentially perfectly.
    """
    import numpy as np

    import fitting

    # Whole-millisecond steps, as a real instrument sweep produces. Integer
    # dtype is the entire point — do not let this become a float array.
    delta_t = np.arange(-50, 51, 5)
    assert np.issubdtype(delta_t.dtype, np.integer), "test setup must use integer Δt"

    A_plus, A_minus, tau = 0.5, 0.3, 20.0  # spec values, post-C7 scale
    dg = np.where(
        delta_t > 0, A_plus * np.exp(-delta_t / tau),
        np.where(delta_t < 0, -A_minus * np.exp(delta_t / tau), 0.0),
    )

    result = fitting.fit_stdp_window(
        {"delta_t_ms": delta_t, "delta_g_percent": dg}
    )

    assert result["fit_quality_R2"] > 0.95, (
        f"R² = {result['fit_quality_R2']:.3f} on noiseless analytic data. "
        "The predicted curve was truncated to integers, so it cannot track a "
        "signal whose amplitude is below 1."
    )


# --- C10: the extraction report must not call a fallback a success ---------


def test_rate_constants_report_their_own_provenance(fitted):
    """C6/C10 — a railed boundary solution must not be reported as a fit.

    B_peak used to rail at the optimiser's 1e-2 upper bound. `minimize_scalar`
    returns success=True for a boundary solution, and the caller decided
    provenance by comparing the result against the literal default 6e-4 — a
    test that cannot distinguish "defaulted" from "fitted, and landed on the
    default", and says nothing at all about railing.

    Provenance now comes from the extractor's own diagnostics.
    """
    _, model = fitted
    diagnostics = model.get("rate_constant_diagnostics")
    assert diagnostics is not None, "rate constants report no diagnostics"

    for name in ("A_peak", "B_peak"):
        diag = diagnostics[name]
        assert diag["defaulted"] is False, (
            f"{name} was defaulted: {diag.get('reason')}"
        )
        assert not diag.get("railed"), f"{name} railed at its search bound"
        assert diag["converged"] is True
        # The forward model must include the spectral factor, per the
        # documented ODE — omitting it made the constant a rate at the measured
        # wavelength rather than the peak rate its name claims.
        assert diag["spectral_normalisation_applied"] is True
        assert 0 < diag["spectral_weight"] <= 1.0001


def test_a_railed_rate_constant_is_flagged_rather_than_returned():
    """The railing detector itself, exercised directly.

    Data that no plausible rate constant can reproduce must come back as
    'defaulted with a reason', never as a boundary value dressed as a fit.
    """
    import numpy as np

    import fitting

    # A trace that rises far faster than any rate in the search range allows.
    y = np.linspace(1e-6, 1e-4, 40)
    value, diag = fitting._fit_rate_constant(
        y, spectral_weight=1.0, intensity=1e-12, dt_s=1e-9,
        G_min=1e-6, G_max=1e-4, exponent=0.8, depressing=False,
    )
    assert value is None, "a solution at the search bound was returned as a fit"
    assert diag.get("railed") is True
    assert "railed" in diag["reason"]


def test_the_extraction_report_is_honest_about_this_suite(fitted):
    """Everything claimed as extracted must genuinely have been fitted.

    The synthetic suite feeds every extractor, so nothing should be defaulted.
    If something is, the report must say so — that is the whole contract.
    """
    _, model = fitted
    report = model["extraction_report"]

    for item in ("dynamic_range", "wavelength_response", "nonlinearity",
                 "beta", "retention", "stdp", "srdp", "A_peak", "B_peak"):
        assert item in report["extracted"], f"{item} was not extracted"

    assert not report["defaults_used"], (
        f"defaults were substituted: {report['defaults_used']}"
    )
    assert not report["failed"], f"extractors failed: {report['failed']}"
