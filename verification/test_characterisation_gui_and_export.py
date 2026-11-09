"""Characterisation GUI + suite export: per-run metadata and pure validation.

GUI-free. The Tk paths (Save, Export dialog, Multi-Device, safety prompts,
Retention) are exercised through the simulated-instrument harness instead.
"""

import numpy as np
import pytest

import assembly_helpers as ah
import fitting


def _run(mode, wl=None, intensity=None, n=12, width=20.0, period=100.0, sid="S1",
         g0=1e-6, g1=8e-6, level=1.0):
    p = {"mode": mode, "stim_level": level, "stim_drive_type": "V",
         "stim_width_ms": width, "stim_period_ms": period, "n_pulses": n,
         "sample_id": sid}
    if wl is not None:
        p["wavelength_nm"] = wl
    if intensity is not None:
        p["intensity_mW_cm2"] = intensity
    return {"pulse_number": list(range(n + 1)),
            "time_s": [i * period / 1000 for i in range(n + 1)],
            "conductance_S": list(np.linspace(g0, g1, n + 1)), "params": p}


def _softbound_run(wl, intensity, peak=550.0, width=100.0, n=40, w_ms=20.0):
    """Initial-rate-limited soft-bound trace whose rate is S(wl) * I * pulse width."""
    s = np.exp(-((wl - peak) / width) ** 2)
    g, g_max, a_peak, alpha = 1e-6, 1e-4, 8e-4, 0.8
    out = [g]
    for _ in range(n):
        g += a_peak * s * intensity * (w_ms / 1000.0) * (g_max - g) ** alpha
        out.append(g)
    r = _run("visual", wl, intensity, n=n, width=w_ms)
    r["conductance_S"] = out
    return r


# --- per-run metadata ------------------------------------------------------


def test_each_sweep_dataset_keeps_its_own_intensity_and_wavelength():
    """The suite used to be stamped from the FIRST run: 10/20/30 mW/cm2 all became 30."""
    runs = [_run("visual", 450, 10), _run("visual", 550, 20), _run("visual", 650, 30)]
    results, wls, skipped = ah.detect_wavelength_sweep(runs)
    suite = ah.build_characterization_suite(wavelength_results=results, wavelengths=wls)
    got = [(d["metadata"]["wavelength_pot"], d["metadata"]["light_intensity_mW_cm2"])
           for d in suite["datasets"]]
    assert got == [(450, 10), (550, 20), (650, 30)]
    assert all(d["metadata"]["stimulus_type"] == "light" for d in suite["datasets"])
    assert skipped == 0


def test_an_electrical_ltd_run_is_not_exported_as_light():
    ltd = _run("electrical", wl=650, intensity=20, g0=8e-6, g1=1e-6, level=-1.0)
    suite = ah.build_characterization_suite(ltd_result=ltd)
    md = suite["datasets"][0]["metadata"]
    assert md["stimulus_type"] == "electrical"
    assert "wavelength_dep" not in md and "wavelength_pot" not in md
    assert "light_intensity_mW_cm2" not in md and "intensity" not in md
    assert md["experiment_type"] == "depression"


def test_electrical_runs_never_enter_the_wavelength_sweep():
    runs = [_run("visual", 450, 10), _run("visual", 550, 20),
            _run("electrical", 600, 20), _run("memristor_pulse", 650, 20)]
    assert ah.detect_wavelength_sweep(runs)[0] is None   # only 2 optical runs


def test_sweep_is_paired_by_wavelength_regardless_of_run_order():
    runs = [_run("visual", 650, 30, g1=3e-6), _run("visual", 450, 10, g1=2e-6),
            _run("visual", 550, 20, g1=9e-6)]
    results, wls, _ = ah.detect_wavelength_sweep(runs)
    assert wls == [450, 550, 650]
    assert [r["conductance_S"][-1] for r in results] == [2e-6, 9e-6, 3e-6]


def test_pulse_width_period_and_count_come_from_the_run():
    ltp = _run("electrical", n=12, width=20.0, period=100.0)
    md = ah.build_characterization_suite(ltp_result=ltp)["datasets"][0]["metadata"]
    assert (md["pulse_width_ms"], md["pulse_period_ms"], md["n_pulses"]) == (20.0, 100.0, 12)
    assert md["frequency_Hz"] == pytest.approx(10.0)


def test_nothing_is_invented_when_the_run_recorded_nothing():
    bare = {"pulse_number": [0, 1, 2], "conductance_S": [1e-6, 2e-6, 3e-6]}
    md = ah.build_characterization_suite(ltp_result=bare)["datasets"][0]["metadata"]
    for key in ("stimulus_type", "light_intensity_mW_cm2", "intensity", "pulse_width_ms",
                "n_pulses", "wavelength_pot"):
        assert key not in md


def test_unrecognised_mode_fails_loudly():
    with pytest.raises(ValueError, match="Unrecognised measurement mode"):
        ah.build_characterization_suite(ltp_result=_run("Memristor (Pulse)"))


def test_explicit_per_result_params_override_the_run():
    md = ah.build_characterization_suite(
        ltp_result=_run("visual", 550, 20),
        result_params={"ltp": {"light_intensity_mW_cm2": 5}},
    )["datasets"][0]["metadata"]
    assert md["light_intensity_mW_cm2"] == 5


# --- cycles ----------------------------------------------------------------


def _cycle_results():
    def phase(cfg, g0, g1):
        n = cfg["n_pulses"]
        return {"pulse_number": list(range(n + 1)),
                "conductance_S": list(np.linspace(g0, g1, n + 1)), "params": cfg}
    a = {"stim_level": 1.0, "stim_drive_type": "V", "stim_width_ms": 40.0,
         "stim_period_ms": 200.0, "n_pulses": 5, "sample_id": "C1"}
    b = {"stim_level": -1.0, "stim_drive_type": "V", "stim_width_ms": 100.0,
         "stim_period_ms": 300.0, "n_pulses": 5, "sample_id": "C1"}
    return {"cycles": [{"potentiation": phase(a, 1e-6, 8e-6),
                        "depression": phase(b, 8e-6, 1e-6)}],
            "pot_params": a, "dep_params": b}


def test_cycle_datasets_use_each_trains_real_width_count_and_no_invented_intensity():
    ds = {d["metadata"]["cycle_phase"]: d["metadata"]
          for d in ah.convert_cycle_to_datasets(_cycle_results(), {})}
    assert ds["potentiation"]["pulse_width_ms"] == 40.0
    assert ds["depression"]["pulse_width_ms"] == 100.0
    # 5 pulses is 6 points (index 0 is the baseline); the baseline is not a pulse.
    assert ds["potentiation"]["n_pulses"] == ds["depression"]["n_pulses"] == 5
    for md in ds.values():
        assert md["stimulus_type"] == "electrical"
        assert "light_intensity_mW_cm2" not in md and "intensity" not in md
        assert "wavelength_pot" not in md and "wavelength_dep" not in md


def test_cycle_count_without_a_config_excludes_the_baseline():
    r = _cycle_results()
    r.pop("pot_params"); r.pop("dep_params")
    for c in r["cycles"]:
        for ph in c.values():
            ph.pop("params")
    for d in ah.convert_cycle_to_datasets(r, {}):
        assert d["metadata"]["n_pulses"] == 5


# --- STDP / SRDP intensity key ---------------------------------------------


@pytest.mark.parametrize("conv,result", [
    (ah.convert_stdp_to_dataset, {"delta_t_ms": [-20, 20], "g_final_S": [1e-6, 2e-6]}),
    (ah.convert_srdp_to_dataset, {"frequencies_hz": [1, 10], "g_final_S": [1e-6, 2e-6]}),
])
def test_stdp_and_srdp_read_the_key_the_gui_emits(conv, result):
    md = conv(result, {"light_intensity_mW_cm2": 15, "wavelength_nm": 550})["metadata"]
    assert md["light_intensity_mW_cm2"] == 15
    assert md["wavelength_pot"] == 550


# --- fitting recovers per-wavelength intensity -----------------------------


def test_fitter_recovers_the_spectrum_from_a_mixed_intensity_sweep():
    wls = [400, 450, 500, 550, 600, 650, 700]
    intensities = [30, 10, 20, 30, 10, 20, 30]
    runs = [_softbound_run(w, i) for w, i in zip(wls, intensities)]
    results, ws, _ = ah.detect_wavelength_sweep(runs)
    suite = ah.build_characterization_suite(
        wavelength_results=results, wavelengths=ws,
        ltp_result=_softbound_run(550, 20, n=60),
        ltd_result=dict(_run("electrical", n=40, g0=8e-5, g1=2e-5, level=-1.0)),
        metadata={"sample_id": "S1"})
    for d in suite["datasets"][:7]:
        assert d["metadata"]["light_intensity_mW_cm2"] == intensities[wls.index(d["metadata"]["wavelength_pot"])]
    model = fitting.extract_synapse_model(suite, verbose=False)
    assert model["lambda_peak"] == pytest.approx(550.0, abs=25.0)
    # Had the first run's 30 mW/cm2 been stamped on all, the 10 mW/cm2 points
    # would read 3x too weak and the peak would shift.
    curve = model["wavelength_curve"]
    assert abs(float(curve(np.array([450.0]))[0]) - np.exp(-1.0)) < 0.2


# --- classification and role checks ----------------------------------------


def test_run_kinds_and_roles():
    visual_jsc = {"Jsc_A": [1e-6], "time_s": [0.0], "pulse_number": [0], "params": {}}
    retention = {"time_s": [1.0, 2.0], "conductance_S": [2e-6, 1e-6],
                 "params": {"mode": "retention"}}
    assert ah.classify_run(visual_jsc) == "visual_jsc"      # not "retention" because of time_s
    assert ah.classify_run(retention) == "retention"
    assert ah.classify_run(_run("electrical", n=50)) == "pulse"   # no n > 30 rule
    assert ah.classify_run({"devices": [], "cycles": []}) == "multi_device"
    assert ah.check_role(visual_jsc, "retention")
    assert ah.check_role(_run("electrical"), "retention")
    assert ah.check_role(retention, "retention") is None
    assert ah.check_role(_run("electrical"), "ltd") is None
    assert "Visual" in ah.check_role(visual_jsc, "ltp")


def test_branch_comes_from_the_data():
    assert ah.run_branch(_run("electrical", g0=1e-6, g1=5e-6)) == "potentiation"
    assert ah.run_branch(_run("electrical", g0=5e-6, g1=1e-6)) == "depression"


def test_retention_dataset_is_exportable_and_keeps_a_time_axis():
    ret = {"time_s": [1.0, 10.0, 30.0], "conductance_S": [4e-6, 3e-6, 2e-6],
           "I_A": [4e-7, 3e-7, 2e-7], "params": {"mode": "retention", "sample_id": "R"}}
    suite = ah.build_characterization_suite(retention_result=ret)
    d = suite["datasets"][0]
    assert d["metadata"]["experiment_type"] == "retention"
    assert d["metadata"]["x_type"] == "time"
    with pytest.raises(ValueError, match="no 'conductance_S'"):
        ah.convert_retention_to_dataset({"time_s": [1.0]}, {})


def test_sample_id_comes_from_the_run_itself():
    assert ah.run_sample_id(_run("electrical", sid="X7")) == "X7"
    assert ah.run_sample_id(_cycle_results()) == "C1"
    assert ah.run_sample_id({"conductance_S": []}) is None


def test_validate_suite_wavelength_hint_is_judged_from_content(capsys):
    swept = ah.build_characterization_suite(
        wavelength_results=[_run("visual", w, 20) for w in (450, 550, 650)],
        wavelengths=[450, 550, 650])
    assert not any("wavelength sweep" in r for r in ah.validate_suite(swept)["recommendations"])
    single = ah.build_characterization_suite(ltp_result=_run("visual", 550, 20))
    assert any("wavelength sweep" in r for r in ah.validate_suite(single)["recommendations"])


def test_suite_with_only_stdp_is_flagged_invalid():
    stdp = {"delta_t_ms": [-20.0, 20.0], "g_final_S": [1e-6, 2e-6]}
    assert ah.validate_suite(ah.build_characterization_suite(stdp_result=stdp))["valid"] is False


# --- pure validation -------------------------------------------------------


@pytest.mark.parametrize("value", [0, -4, 3.7])
def test_require_count_rejects_non_positive_and_fractional(value):
    with pytest.raises(ValueError):
        ah.require_count(value, "# Pulses")


def test_require_count_names_the_limit():
    assert ah.require_count(5.0, "n") == 5
    with pytest.raises(ValueError, match="49999.*buffer"):
        ah.require_count(60000, "# Pulses", maximum=49999, limit_name="the Keithley buffer")


def test_stdp_default_style_sweep_is_valid_and_the_old_one_is_not():
    ah.check_stdp_plan(np.linspace(-50, 50, 10), 5.0, 100.0)
    with pytest.raises(ValueError, match="shorter than the pulse width"):
        ah.check_stdp_plan(np.linspace(-50, 50, 15), 10.0, 100.0)   # old defaults


def test_stdp_refuses_zero_duplicates_and_a_period_that_does_not_fit():
    with pytest.raises(ValueError, match="delta-t = 0"):
        ah.check_stdp_plan([-20, 0, 20], 5.0, 100.0)
    with pytest.raises(ValueError, match="duplicate"):
        ah.check_stdp_plan(np.linspace(20, 20, 4), 5.0, 100.0)
    with pytest.raises(ValueError, match="cannot contain the longest pair"):
        ah.check_stdp_plan([-50, 50], 10.0, 55.0)
    with pytest.raises(ValueError, match="pair period.*shorter than the pulse width"):
        ah.check_stdp_plan([-50, 50], 10.0, 5.0)


def test_srdp_end_below_start_is_refused():
    ah.check_srdp_range(1, 100, 10)
    with pytest.raises(ValueError, match="below the start"):
        ah.check_srdp_range(50, 5, 10)
    with pytest.raises(ValueError, match="same frequency"):
        ah.check_srdp_range(5, 5, 10)


def test_visual_default_window_fits_one_adc_integration_and_the_old_one_did_not():
    ah.check_visual_window(50, 1, 1, 1.0, 50.0)       # new defaults
    ah.check_visual_window(50, 1, 1, 1.0, 60.0)
    with pytest.raises(ValueError, match="does not fit"):
        ah.check_visual_window(10, 1, 1, 1.0, 50.0)   # old defaults
    with pytest.raises(ValueError, match="Invalid measurement window"):
        ah.check_visual_window(2, 1, 1, 0.001, 50.0)


def test_retention_times_become_waits():
    assert ah.retention_waits_from_times([1, 10, 30]) == [1.0, 9.0, 20.0]
    for bad in ([], [0, 1], [5, 3]):
        with pytest.raises(ValueError):
            ah.retention_waits_from_times(bad)
