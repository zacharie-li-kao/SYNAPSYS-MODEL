"""The Keithley → JSON export contract (audit §6, F1-F8).

Before these fixes, a hardware export was structurally incapable of feeding four
of the fitter's six extractors. Each test here corresponds to one of those
blockages, and asserts against `fitting.py`'s real dispatch rules rather than
against the exporter's own idea of what it produced.
"""

import numpy as np
import pytest

import assembly_helpers as ah
import fitting


def _pulse_result(n=50, g0=1e-6, g1=8e-5):
    """A result shaped exactly as synapse_engine.pulse_read_sequence returns it.

    Note the SINGULAR 'pulse_number' — the plural spelling appears nowhere in
    the engine, and testing against the plural is what let F5 survive.
    """
    return {
        "pulse_number": list(range(n)),
        "time_s": [i * 0.1 for i in range(n)],
        "conductance_S": list(np.linspace(g0, g1, n)),
        "I_A": [1e-7] * n,
        "V_read_V": [0.1] * n,
    }


# --- F5: the x-axis key ----------------------------------------------------


def test_pulse_trains_export_against_a_pulse_axis_not_time():
    """F5 — the branch tested 'pulse_numbers' while the engine returns 'pulse_number'.

    It therefore never fired on real data, only on this module's own __main__
    fixture, so every potentiation dataset was exported with a time x-axis and
    labelled x_type='time'.
    """
    ds = ah.convert_pulse_sequence_to_dataset(
        _pulse_result(), "potentiation", {"stimulus_type": "light"}, "ltp"
    )
    assert ds["metadata"]["x_type"] == "pulse_number"
    assert ds["metadata"]["x_units"] == "#"
    assert ds["x_data"] == list(range(50))


def test_a_genuine_time_series_still_exports_as_time():
    """Retention has no pulse axis and must keep its time axis."""
    result = {"time_s": [0.0, 1.0, 2.0], "conductance_S": [8e-5, 5e-5, 3e-5]}
    ds = ah.convert_pulse_sequence_to_dataset(result, "retention", {}, "ret")
    assert ds["metadata"]["x_type"] == "time"
    assert ds["metadata"]["x_units"] == "s"


# --- F2: cycle phase labels ------------------------------------------------


def test_cycle_phases_keep_their_potentiation_and_depression_labels():
    """F2 — the labels were computed, used for derived_metrics, then overwritten.

    `fitting.py:216` skips 'potentiation_depression_cycle' for the spectral fit
    and `:144` filters it out of the beta fit, so the overwrite meant cycle data
    contributed to neither.
    """
    cycles = {
        "cycles": [
            {
                "potentiation": _pulse_result(30, 1e-6, 8e-5),
                "depression": _pulse_result(30, 8e-5, 2e-6),
            }
            for _ in range(3)
        ]
    }
    datasets = ah.convert_cycle_to_datasets(
        cycles, {"wavelength_pot": 365, "wavelength_dep": 550}
    )

    types = [d["metadata"]["experiment_type"] for d in datasets]
    assert types.count("potentiation") == 3
    assert types.count("depression") == 3
    assert "potentiation_depression_cycle" not in types

    # Cycle membership must still be recorded — that is what the overwrite was
    # reaching for, and it is orthogonal to the phase.
    for d in datasets:
        assert d["metadata"]["from_cycling_measurement"] is True
        assert d["metadata"]["cycle_index"] in (1, 2, 3)
        assert d["metadata"]["n_cycles"] == 3


def test_cycle_depression_data_can_reach_the_beta_fit():
    """The end-to-end consequence of F2, asserted against fitting's own filter."""
    cycles = {
        "cycles": [
            {
                "potentiation": _pulse_result(40, 1e-6, 8e-5),
                "depression": _pulse_result(40, 8e-5, 2e-6),
            }
        ]
    }
    suite = ah.build_characterization_suite(
        cycle_results=cycles,
        stimulus_params={"wavelength_pot": 365, "wavelength_dep": 550,
                         "stimulus_type": "light", "intensity": 20},
        metadata={"sample_id": "T1"},
    )
    parsed = fitting.DataParser.parse_suite(suite)
    assert parsed["depression_data"], (
        "no depression data survived into the fitter — beta is unfittable"
    )


# --- F1: the LTD path ------------------------------------------------------


def test_an_ltd_measurement_produces_a_depression_dataset():
    """F1 — keithley_analyser never passed ltd_result, and the dialog had no
    way to mark one, so no dataset with experiment_type 'depression' was ever
    produced and beta could never be extracted from a Keithley export."""
    suite = ah.build_characterization_suite(
        ltp_result=_pulse_result(60, 1e-6, 8e-5),
        ltd_result=_pulse_result(60, 8e-5, 2e-6),
        stimulus_params={"stimulus_type": "light", "intensity": 20},
        metadata={"sample_id": "T1"},
    )
    types = [d["metadata"]["experiment_type"] for d in suite["datasets"]]
    assert "depression" in types
    assert "potentiation" in types


def test_export_dialog_offers_every_marking_the_fitter_needs():
    """The GUI must be able to name each experiment type the fitter dispatches on."""
    import inspect

    import keithley_analyser as ka

    src = inspect.getsource(ka.export_characterization_suite)
    for marker in ("mark_as_ltp", "mark_as_ltd", "mark_as_cycles",
                   "mark_as_retention", "mark_as_stdp", "mark_as_srdp"):
        assert f"def {marker}" in src, f"export dialog has no {marker} entry"
    for kwarg in ("ltd_result=", "stdp_result=", "srdp_result="):
        assert kwarg in src, f"export never passes {kwarg} to the suite builder"


# --- F3: STDP and SRDP converters ------------------------------------------


def test_stdp_measurement_reaches_the_stdp_fitter():
    """F3 — there was no STDP converter anywhere, so the STDP window always
    came from defaults no matter what had been measured."""
    delta_t = np.linspace(-100, 100, 21)
    g = 5e-5 + 1e-6 * np.where(delta_t > 0, np.exp(-delta_t / 20),
                               -np.exp(delta_t / 20))
    stdp_result = {
        "delta_t_ms": delta_t.tolist(),
        "g_final_S": g.tolist(),
        "g_initial_S": [5e-5] * len(delta_t),
        "delta_g_S": (g - 5e-5).tolist(),
        "delta_g_percent": ((g - 5e-5) / 5e-5 * 100).tolist(),
    }

    suite = ah.build_characterization_suite(
        ltp_result=_pulse_result(),
        stdp_result=stdp_result,
        stimulus_params={"stimulus_type": "electrical"},
        metadata={"sample_id": "T1"},
    )
    parsed = fitting.DataParser.parse_suite(suite)
    assert parsed["stdp_data"], "STDP dataset did not reach the fitter"
    assert parsed.get("stdp") is not None
    assert len(parsed["stdp"]["delta_t_ms"]) == 21


def test_srdp_measurement_reaches_the_srdp_fitter():
    """F3, SRDP half."""
    freqs = np.logspace(0, 2, 12)
    g = 5e-5 + 2e-5 / (1 + np.exp(-(freqs - 10) / 5))
    srdp_result = {
        "frequencies_hz": freqs.tolist(),
        "g_final_S": g.tolist(),
        "g_initial_S": [5e-5] * len(freqs),
        "delta_g_S": (g - 5e-5).tolist(),
        "delta_g_percent": ((g - 5e-5) / 5e-5 * 100).tolist(),
    }

    suite = ah.build_characterization_suite(
        ltp_result=_pulse_result(),
        srdp_result=srdp_result,
        stimulus_params={"stimulus_type": "electrical"},
        metadata={"sample_id": "T1"},
    )
    parsed = fitting.DataParser.parse_suite(suite)
    assert parsed["srdp_data"], "SRDP dataset did not reach the fitter"
    assert parsed.get("srdp") is not None


def test_a_malformed_stdp_result_raises_rather_than_exporting_nonsense():
    with pytest.raises(ValueError, match="delta_t_ms"):
        ah.convert_stdp_to_dataset({"conductance_S": [1, 2, 3]}, {})

    with pytest.raises(ValueError, match="inconsistent"):
        ah.convert_stdp_to_dataset(
            {"delta_t_ms": [0, 1, 2], "g_final_S": [1e-6, 2e-6]}, {}
        )


# --- F4: visual results are not conductance datasets -----------------------


def test_a_visual_measurement_fails_with_a_diagnostic_not_a_keyerror():
    """F4 — visual results store 'Jsc_A' and have no 'conductance_S'.

    The unguarded lookup raised a bare KeyError that the GUI surfaced as an
    opaque "Failed to export suite" with no indication of which measurement.
    """
    visual = {"time_s": [0, 1, 2], "Jsc_A": [1e-7, 2e-7, 3e-7]}
    with pytest.raises(ValueError, match="Jsc_A|conductance_S"):
        ah.convert_pulse_sequence_to_dataset(visual, "potentiation", {}, "visual_run")


# --- F6/F7 and H4: metadata correctness ------------------------------------


def test_wavelength_sweep_pairs_traces_with_their_own_wavelengths():
    """H4 — results were zipped in run order against sorted wavelengths.

    Measuring 550, then 450, then 650 nm tagged the 550 nm trace as 450 nm.
    Asserted through the exporter's own auto-detect ordering: the trace's
    distinguishing conductance must travel with its own wavelength.
    """
    # Distinct final conductance per wavelength, measured out of order.
    run_order = [(550, 5e-5), (450, 2e-5), (650, 9e-5)]
    wavelength_dict = {
        wl: {"pulse_number": list(range(10)),
             "conductance_S": list(np.linspace(1e-6, gf, 10)),
             "metadata": {"wavelength_nm": wl}}
        for wl, gf in run_order
    }

    ordered = sorted(wavelength_dict.items(), key=lambda kv: kv[0])
    wavelengths = [wl for wl, _ in ordered]
    results = [res for _, res in ordered]

    datasets = ah.convert_wavelength_sweep(results, wavelengths, {"intensity": 20})

    expected = dict(run_order)
    for ds in datasets:
        wl = ds["metadata"]["wavelength_pot"]
        assert ds["y_data"][-1] == pytest.approx(expected[wl]), (
            f"the {wl} nm dataset carries another wavelength's trace"
        )


def test_depression_wavelength_is_filed_under_the_depression_key():
    ds = ah.convert_pulse_sequence_to_dataset(
        _pulse_result(20, 8e-5, 2e-6), "depression",
        {"wavelength_nm": 550, "stimulus_type": "light"}, "ltd"
    )
    assert ds["metadata"].get("wavelength_dep") == 550
    assert "wavelength_pot" not in ds["metadata"]


def test_no_wavelength_is_invented_when_none_was_recorded():
    """A cycle exported without a recorded wavelength used to be stamped 550/365."""
    cycles = {
        "cycles": [{
            "potentiation": _pulse_result(20, 1e-6, 8e-5),
            "depression": _pulse_result(20, 8e-5, 2e-6),
        }]
    }
    datasets = ah.convert_cycle_to_datasets(cycles, {})
    for d in datasets:
        assert "wavelength_pot" not in d["metadata"]
        assert "wavelength_dep" not in d["metadata"]


def test_stimulus_voltage_is_never_written_into_the_intensity_field():
    """F6 — 'intensity': params.get('stim_level', 20) put volts into the field
    the fitter reads as optical intensity in mW/cm²."""
    import inspect

    import keithley_analyser as ka

    src = inspect.getsource(ka.export_characterization_suite)
    assert "'intensity': data['params'].get('stim_level'" not in src
    assert "'intensity': p.get('stim_level'" not in src
    assert "intensity_mW_cm2" in src, (
        "the export must read the real intensity key"
    )


# --- F8: retention fit failure is flagged ----------------------------------


def test_a_failed_retention_fit_is_flagged_not_silently_dropped():
    """F8 — a bare `except: pass` left tau_retention_s simply absent, so the
    consumer could not distinguish "not measured" from "measured, unfittable"."""
    # Constant data: the decay fit is degenerate and cannot converge.
    x = np.arange(10, dtype=float)
    y = np.full(10, 5e-5)
    derived = ah.compute_derived_metrics(x, y, {"experiment_type": "retention"})

    if derived.get("tau_retention_s") is None:
        assert derived.get("retention_fit_failed") is True
        assert "retention_fit_error" in derived
    else:
        # It converged; then it must report a quality figure, not nothing.
        assert "fit_R2" in derived


# --- the whole contract, end to end ---------------------------------------


def test_a_full_hardware_style_export_feeds_every_extractor():
    """The point of step 0: a realistic export must reach all six extractors."""
    wavelengths = [365, 450, 550, 650]
    wl_results = [_pulse_result(40, 1e-6, 8e-5) for _ in wavelengths]

    delta_t = np.linspace(-100, 100, 21)
    stdp_result = {
        "delta_t_ms": delta_t.tolist(),
        "g_final_S": (5e-5 + 1e-6 * np.where(
            delta_t > 0, np.exp(-delta_t / 20), -np.exp(delta_t / 20))).tolist(),
    }
    freqs = np.logspace(0, 2, 12)
    srdp_result = {
        "frequencies_hz": freqs.tolist(),
        "g_final_S": (5e-5 + 2e-5 / (1 + np.exp(-(freqs - 10) / 5))).tolist(),
    }

    suite = ah.build_characterization_suite(
        wavelength_results=wl_results,
        wavelengths=wavelengths,
        ltp_result=_pulse_result(100, 1e-6, 9e-5),
        ltd_result=_pulse_result(100, 9e-5, 2e-6),
        cycle_results={"cycles": [{
            "potentiation": _pulse_result(40, 1e-6, 8e-5),
            "depression": _pulse_result(40, 8e-5, 2e-6),
        }]},
        retention_result={"time_s": list(np.linspace(0, 500, 50)),
                          "conductance_S": list(8e-5 * np.exp(-np.linspace(0, 500, 50) / 100) + 1e-6)},
        stdp_result=stdp_result,
        srdp_result=srdp_result,
        stimulus_params={"stimulus_type": "light", "intensity": 20,
                         "wavelength_pot": 365, "wavelength_dep": 550},
        metadata={"sample_id": "FULL"},
    )

    parsed = fitting.DataParser.parse_suite(suite)
    for key in ("potentiation_data", "depression_data", "retention_data",
                "stdp_data", "srdp_data"):
        assert parsed[key], f"{key} is empty — that extractor cannot run"

    assert parsed.get("dynamic_range") is not None
    assert parsed.get("nonlinearity") is not None
