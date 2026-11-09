"""The pulse-read / visual / retention engine, executed end to end.

Every test here runs the REAL generated script (or the real standard path)
against `tsp_sim.SimulatedKeithley`, which interprets the TSP in Lua 5.1 with a
virtual clock, DELAY_AUTO, a delay(0) rejection and a finite ASCII precision.
Text-level lint cannot see any of the defects below; each was reproduced in the
simulator before it was fixed:

  * a hidden DELAY_AUTO settling before every read,
  * OUTPUT_ON/OFF toggled per pulse (OUTPUT_OFF does not open the relay),
  * delay(0) aborting the script,
  * a cleanup that left levels and the filter live,
  * a current-drive source range derived from the compliance,
  * continuous visual deriving pulse width/period from sample counts,
  * a silent re-run on a device the script had already stimulated,
  * conductance fabricated as 0 at zero read bias,
  * 6-digit timestamps beyond 100 s.
"""

import math
import re

import pytest

import synapse_engine as se
import tsp_sim


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(se.time, "sleep", lambda s: None)


def _params(**overrides):
    p = {
        "mode": "electrical", "stim_drive_type": "V", "stim_level": 1.0,
        "stim_width_ms": 20.0, "stim_period_ms": 100.0, "n_pulses": 5,
        "read_voltage": 0.1, "read_delay_ms": 10.0, "read_settle_delay_ms": 5.0,
        "compliance_A": 1e-6, "nplc": 0.1, "measure_avg": 1, "settle_ms": 5,
        "sample_id": "verification",
    }
    p.update(overrides)
    return p


def _visual(**overrides):
    p = {
        "light_pulse_voltage": 3.0, "pulse_width_ms": 20.0, "pulse_period_ms": 100.0,
        "n_pulses": 4, "compliance_A": 1e-3, "nplc": 0.1,
        "measure_start_delay_ms": 5.0, "measure_end_margin_ms": 5.0,
        "readings_per_pulse": 1, "sample_interval_ms": 1.0,
    }
    p.update(overrides)
    return p


def _levels(sim, ch, key="levelv"):
    return [(t, v) for (t, n, k, v) in sim.log if n == f"{ch}.source" and k == key]


def _outputs(sim, ch):
    return [v for (_, n, k, v) in sim.log if n == f"{ch}.source" and k == "output"]


def _edges(sim, ch, level):
    """(on_times, off_times) of the pulses that raised `ch` to `level` and back to 0."""
    ons, offs, high = [], [], False
    for t, v in _levels(sim, ch):
        if v == level and not high:
            ons.append(t)
            high = True
        elif v == 0 and high:
            offs.append(t)
            high = False
    return ons, offs


def _run_pulse_read(sim=None, stim_ch="smua", read_ch="smub", **overrides):
    sim = sim or tsp_sim.SimulatedKeithley()
    results = se.pulse_read_sequence(sim, stim_ch, read_ch, _params(**overrides))
    return sim, results


# --- 1. no hidden read delay ---------------------------------------------------


@pytest.mark.parametrize("compliance_A", [100e-9, 10e-9])
def test_read_follows_the_pulse_by_exactly_read_delay_plus_settle(compliance_A):
    """DELAY_AUTO added 15 ms at 100 nA and 90 ms at 10 nA to every read."""
    sim, results = _run_pulse_read(compliance_A=compliance_A)
    assert results["execution_path"] == "lua" and sim.errors == []
    _, offs = _edges(sim, "smua", 1.0)
    reads = [m for m in sim.measurements if m[1] == "smub"]
    assert len(reads) == 6 and len(offs) == 5
    want = (10.0 + 5.0) / 1000.0
    for k in range(5):
        gap = (reads[k + 1][0] + sim.t0) - offs[k]
        # two TSP commands (bias write, measure call) are all that may intervene
        assert gap == pytest.approx(want, abs=3 * sim.t_cmd), (k, gap)
    # Off while reading; the instrument's own default put back afterwards.
    assert _delay_during_reads(sim) == 0
    assert sim.state["smub.measure"]["delay"] == "DELAY_AUTO"


def test_a_settled_read_range_raises_no_warning():
    _, results = _run_pulse_read(compliance_A=100e-9, read_settle_delay_ms=20.0)
    assert results["timing"]["settle_warning"] is None


def test_an_unsettled_read_is_flagged_in_the_results_and_the_csv(tmp_path):
    """The user may trade accuracy for speed; it must be visible, never refused."""
    _, results = _run_pulse_read(compliance_A=100e-9, read_settle_delay_ms=5.0)
    warn = results["timing"]["settle_warning"]
    assert warn and "15" in warn
    out = tmp_path / "run.csv"
    se.process_and_save_synapse_data(results, str(out))
    header = out.read_text(encoding="utf-8").split("pulse,timestamp_s")[0]
    assert "timing_settle_warning=" in header and "SETTLE_WARNING" in header


# --- 3. delay(0) ---------------------------------------------------------------


def test_zero_read_delay_and_settle_run_instead_of_aborting_the_script():
    sim, results = _run_pulse_read(read_delay_ms=0.0, read_settle_delay_ms=0.0,
                                   settle_ms=0, compliance_A=1e-3)
    assert sim.errors == [], sim.errors
    assert results["execution_path"] == "lua"
    assert len(results["conductance_S"]) == 6


@pytest.mark.parametrize("field", ["read_delay_ms", "read_settle_delay_ms", "settle_ms",
                                   "stim_width_ms"])
def test_negative_times_are_refused_up_front(field):
    sim = tsp_sim.SimulatedKeithley()
    with pytest.raises(ValueError, match=field):
        se.pulse_read_sequence(sim, "smua", "smub", _params(**{field: -1.0}))
    assert sim.scripts_run == []


def test_negative_visual_times_are_refused():
    with pytest.raises(ValueError, match="measure_start_delay_ms"):
        se.generate_visual_synapse_lua_script("smua", "smub", _visual(measure_start_delay_ms=-1.0))
    with pytest.raises(ValueError, match="sample_interval_ms"):
        se.generate_visual_continuous_lua_script("smua", "smub", _visual(sample_interval_ms=0))


def test_a_zero_start_delay_visual_run_is_valid():
    sim = tsp_sim.SimulatedKeithley()
    se.visual_synapse_sequence(sim, "smua", "smub", _visual(measure_start_delay_ms=0.0))
    assert sim.errors == []


# --- 2. level changes, not output toggling ---------------------------------------


def test_pulse_read_switches_the_stim_output_exactly_twice():
    sim, _ = _run_pulse_read(n_pulses=8)
    assert _outputs(sim, "smua") == ["OUTPUT_ON", "OUTPUT_OFF"]
    ons, offs = _edges(sim, "smua", 1.0)
    assert len(ons) == len(offs) == 8


def test_current_drive_pulses_by_level_too():
    sim, _ = _run_pulse_read(stim_drive_type="I", stim_level=1e-3, compliance_A=1e-3)
    assert _outputs(sim, "smua") == ["OUTPUT_ON", "OUTPUT_OFF"]
    assert [v for _, v in _levels(sim, "smua", "leveli") if v] == [1e-3] * 5


def test_visual_modes_switch_the_led_output_exactly_twice():
    for continuous in (False, True):
        sim = tsp_sim.SimulatedKeithley()
        se.visual_synapse_sequence(sim, "smua", "smub", _visual(), continuous_mode=continuous)
        assert _outputs(sim, "smua") == ["OUTPUT_ON", "OUTPUT_OFF"], continuous


def test_standard_path_follows_the_same_protocol(monkeypatch):
    monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)
    sim, results = _run_pulse_read(measure_avg=1)
    assert results["execution_path"] == "standard"
    assert _outputs(sim, "smua") == ["OUTPUT_ON", "OUTPUT_OFF"]
    assert len(results["conductance_S"]) == 6


# --- 4. cleanup -------------------------------------------------------------------


def _assert_quiescent(sim, stim="smua", read="smub"):
    for ch in (stim, read):
        src = sim.state[f"{ch}.source"]
        assert src["output"] == "OUTPUT_OFF", ch
        assert src["levelv"] == 0 and src["leveli"] == 0, (ch, src["levelv"], src["leveli"])
    assert sim.state[f"{read}.measure.filter"]["enable"] == "FILTER_OFF"


def test_pulse_read_cleanup_leaves_nothing_live():
    """It used to leave filter ON/count 10 and the stim level at 1 V."""
    sim, _ = _run_pulse_read(measure_avg=10, nplc=0.01, compliance_A=1e-3)
    _assert_quiescent(sim)


def test_current_drive_cleanup_zeroes_the_current_level():
    sim, _ = _run_pulse_read(stim_drive_type="I", stim_level=2e-4, compliance_A=1e-3)
    _assert_quiescent(sim)


def test_visual_cleanup_leaves_nothing_live():
    for continuous in (False, True):
        sim = tsp_sim.SimulatedKeithley()
        se.visual_synapse_sequence(sim, "smua", "smub",
                                   _visual(readings_per_pulse=3, nplc=0.1),
                                   continuous_mode=continuous)
        _assert_quiescent(sim)


def test_standard_path_cleanup_and_full_adc_state(monkeypatch):
    """With averaging 1 no FILTER_OFF was written, so a dirty ON/8 state leaked in."""
    monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)
    sim = tsp_sim.SimulatedKeithley()
    sim.state["smub.measure.filter"].update(enable="FILTER_ON", count=8)
    sim.state["smub.measure"]["delay"] = "DELAY_AUTO"
    _run_pulse_read(sim=sim, measure_avg=1, compliance_A=1e-6)
    reads = [m for m in sim.measurements if m[1] == "smub"]
    assert len(reads) == 6
    # every read integrates once: nplc/linefreq, with no filter multiplier or DELAY_AUTO
    first_cfg = [w for w in sim.writes if "filter.enable" in w]
    assert first_cfg and "FILTER_OFF" in first_cfg[0]
    assert any("measure.delay = 0" in w for w in sim.writes)
    _assert_quiescent(sim)


# --- 5. current-drive source range --------------------------------------------------


@pytest.mark.parametrize("level, want", [(1e-3, 1e-3), (2e-4, 1e-3), (5e-6, 1e-5), (0.05, 0.1)])
def test_current_drive_range_contains_the_sourced_current_not_the_compliance(level, want):
    script = se.generate_pulse_read_lua_script(
        "smua", "smub", _params(stim_drive_type="I", stim_level=level,
                                compliance_A=1e-2, max_voltage=200.0))
    m = re.search(r"smua\.source\.rangei = (\S+)", script)
    assert m and float(m.group(1)) == want
    assert re.search(r"smua\.source\.limitv = 200\.0", script)


def test_standard_path_uses_the_same_current_source_range(monkeypatch):
    monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)
    sim, _ = _run_pulse_read(stim_drive_type="I", stim_level=2e-4, compliance_A=1e-2)
    assert sim.state["smua.source"]["rangei"] == 1e-3


# --- 6. visual source range -----------------------------------------------------------


def test_visual_read_range_is_native_on_both_voltage_families():
    for max_v, want in ((200.0, 0.2), (40.0, 0.1)):
        for gen in (se.generate_visual_synapse_lua_script,
                    se.generate_visual_continuous_lua_script):
            script = gen("smua", "smub", _visual(max_voltage=max_v, light_pulse_voltage=0.5))
            assert f"smub.source.rangev = {want}" in script, (gen.__name__, max_v)


def test_visual_runs_on_a_40v_family_instrument():
    sim = tsp_sim.SimulatedKeithley(model="2602B")
    se.visual_synapse_sequence(sim, "smua", "smub", _visual())
    assert sim.errors == []


# --- 7. continuous visual honours the requested pulse width and period ------------------


@pytest.mark.parametrize("width, period", [(20.0, 100.0), (1.0, 50.0), (10.0, 10.0)])
def test_continuous_pulses_have_the_requested_width_and_period(width, period):
    """nplc 0.1 at 1 ms spacing made every sample 2.1 ms: 20/100 ran as 42/210 ms,
    and a pulse shorter than one sample ran 0.05 ms."""
    sim = tsp_sim.SimulatedKeithley()
    results = se.visual_synapse_sequence(
        sim, "smua", "smub",
        _visual(pulse_width_ms=width, pulse_period_ms=period, n_pulses=4),
        continuous_mode=True)
    assert sim.errors == []
    ons, offs = _edges(sim, "smua", 3.0)
    assert len(ons) == len(offs) == 4
    for on, off in zip(ons, offs):
        assert (off - on) * 1e3 == pytest.approx(width, abs=0.2)
    for a, b in zip(ons, ons[1:]):
        assert (b - a) * 1e3 == pytest.approx(period, abs=0.2)
    if width < period:
        # With width == period the OFF write precedes the next ON write directly,
        # so that ON is late by the TSP command latency; that is a real, counted
        # overrun, not a scheduling error.
        assert results["timing"]["instrument_report"]["overruns"] == 0


def test_continuous_samples_on_the_requested_interval_and_flags_an_oversize_adc():
    sim = tsp_sim.SimulatedKeithley()
    ok = se.visual_synapse_sequence(
        sim, "smua", "smub", _visual(nplc=0.01, sample_interval_ms=1.0), continuous_mode=True)
    t = ok["time_s"]
    spacing = sorted((b - a) * 1e3 for a, b in zip(t, t[1:]))
    assert spacing[len(spacing) // 2] == pytest.approx(1.0, abs=0.01)
    assert ok["timing"]["sample_warning"] is None
    assert ok["timing"]["period_delivered"] is True

    slow = se.visual_synapse_sequence(
        tsp_sim.SimulatedKeithley(), "smua", "smub",
        _visual(nplc=0.1, sample_interval_ms=1.0), continuous_mode=True)
    assert slow["timing"]["sample_warning"] and "longer" in slow["timing"]["sample_warning"]
    assert slow["timing"]["period_delivered"] is False
    assert slow["timing"]["delivered_period_ms_median"] > 1.0


def test_continuous_buffer_bound_is_not_exceeded():
    plan = se.visual_continuous_plan(_visual(nplc=0.01))
    sim = tsp_sim.SimulatedKeithley()
    results = se.visual_synapse_sequence(sim, "smua", "smub", _visual(nplc=0.01),
                                         continuous_mode=True)
    assert len(results["Jsc_A"]) <= plan["max_samples"]
    assert len(results["Jsc_A"]) > 0.9 * plan["max_samples"] - 20


# --- 8. execute_lua_script_fast / _wait_lua_complete -----------------------------------


def test_a_stale_completion_token_does_not_end_the_wait_early():
    sim = tsp_sim.SimulatedKeithley()
    sim.out_queue.extend([se.LUA_DONE_TOKEN])
    _, results = _run_pulse_read(sim=sim)
    assert results["timing"]["instrument_report"]["n_pulses"] == 5.0


def test_timestamps_beyond_100_seconds_are_not_quantised():
    """Six digits turn 120.0004 s into 120.000 s and raise false drift alarms."""
    sim, results = _run_pulse_read(stim_period_ms=120000.4, n_pulses=3)
    diffs = [b - a for a, b in zip(results["time_s"][1:], results["time_s"][2:])]
    assert diffs == pytest.approx([120.0004] * 2, abs=1e-6)
    assert results["timing"]["period_delivered"] is True
    assert abs(results["timing"]["max_drift_ms"]) < 0.01
    assert sim.ascii_precision == se.ASCII_PRECISION


def test_wait_timeout_is_based_on_the_feasible_not_the_nominal_duration(monkeypatch):
    seen = {}
    real = se._wait_lua_complete

    def spy(instrument, expected_time_s=None):
        seen["t"] = expected_time_s
        return real(instrument, expected_time_s)

    monkeypatch.setattr(se, "_wait_lua_complete", spy)
    params = _params(stim_period_ms=5.0, n_pulses=50, compliance_A=1e-3)
    script = se.generate_pulse_read_lua_script("smua", "smub", dict(params, max_voltage=200.0))
    min_period = script.facts["min_period_ms"]
    assert min_period > 5.0
    _run_pulse_read(**params)
    assert seen["t"] == pytest.approx(min_period * 50 / 1000.0)


def test_visual_drift_analysis_counts_periods_correctly():
    """n_pulses was passed as n_expected_points = n while the analysis assumed n+1,
    reporting a spurious -25 ms/cycle every run."""
    sim = tsp_sim.SimulatedKeithley()
    results = se.visual_synapse_sequence(sim, "smua", "smub", _visual(n_pulses=5))
    t = results["timing"]
    assert t["status"] == "ok" and t["n_intervals"] == 4 and t["n_pulses"] == 5
    assert abs(t["mean_drift_ms_per_cycle"]) < 0.01
    assert t["period_delivered"] is True


# --- 9. no silent re-run on a stimulated device -------------------------------------------


class _FaultyKeithley(tsp_sim.SimulatedKeithley):
    """Raises inside the running script after `fail_after` delay() calls."""

    def __init__(self, *a, fail_after=40, **k):
        self._fail_after, self._n_delays = fail_after, 0
        super().__init__(*a, **k)

    def _delay(self, t):
        self._n_delays += 1
        if self._n_delays > self._fail_after:
            raise tsp_sim.TSPError("injected mid-run fault")
        super()._delay(t)


def test_a_fault_after_the_script_started_raises_instead_of_rerunning():
    sim = _FaultyKeithley(fail_after=20)
    with pytest.raises(se.LuaScriptStartedError, match=r"reached \d+ of 11 points") as exc:
        se.pulse_read_sequence(sim, "smua", "smub", _params(n_pulses=10))
    assert "injected mid-run fault" in str(exc.value) or "error queue" in str(exc.value)
    assert not any("*CLS" in w for w in sim.writes), "the standard path ran"
    assert len(sim.scripts_run) == 1
    # and the device was left safe
    assert sim.state["smua.source"]["output"] == "OUTPUT_OFF"
    assert sim.state["smua.source"]["levelv"] == 0


def test_a_failure_before_the_script_ran_still_falls_back(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(se, "execute_lua_script_fast", boom)
    sim = tsp_sim.SimulatedKeithley()
    results = se.pulse_read_sequence(sim, "smua", "smub", _params())
    assert results["execution_path"] == "standard"
    assert "Falling back to standard execution" in capsys.readouterr().out


def test_a_short_buffer_is_a_failed_run_not_partial_data():
    sim = tsp_sim.SimulatedKeithley(buffer_capacity=4)
    with pytest.raises(se.LuaScriptStartedError, match="4 of the 6"):
        se.pulse_read_sequence(sim, "smua", "smub", _params(n_pulses=5))


# --- 10. same stim and read channel ------------------------------------------------------------


def test_stim_equal_to_read_channel_is_refused_everywhere():
    sim = tsp_sim.SimulatedKeithley()
    with pytest.raises(ValueError, match="different SMU"):
        se.pulse_read_sequence(sim, "smua", "smua", _params())
    assert sim.writes == [] and sim.queries == []
    for gen, p in ((se.generate_pulse_read_lua_script, _params()),
                   (se.generate_visual_synapse_lua_script, _visual()),
                   (se.generate_visual_continuous_lua_script, _visual())):
        with pytest.raises(ValueError, match="different SMU"):
            gen("smub", "smub", p)
    with pytest.raises(ValueError, match="different SMU"):
        se.visual_synapse_sequence(sim, "smua", "smua", _visual())


def _delay_during_reads(sim, ch="smub"):
    """measure.delay value last set before the first read on `ch`."""
    t_first = min(m[0] for m in sim.measurements if m[1] == ch) + sim.t0
    sets = [v for (t, n, k, v) in sim.log
            if n == f"{ch}.measure" and k == "delay" and t <= t_first]
    return sets[-1] if sets else None


# --- 12. retention -------------------------------------------------------------------------------


def test_retention_gives_nan_at_zero_read_bias_and_does_not_crash(capsys):
    sim = tsp_sim.SimulatedKeithley()
    res = se.measure_retention(sim, "smub", {
        "read_voltage": 0.0, "retention_times_s": [1, 2, 3],
        "compliance_A": 1e-6, "nplc": 0.1})
    assert len(res["conductance_S"]) == 3
    assert all(math.isnan(g) for g in res["conductance_S"])
    assert "not defined" in capsys.readouterr().out
    # Autorange: the range can change right before a read, so retention keeps
    # the instrument's range settling (DELAY_AUTO) while reading.
    assert _delay_during_reads(sim) == "DELAY_AUTO"
    assert sim.state["smub.measure.filter"]["enable"] == "FILTER_OFF"
    assert sim.state["smub.source"]["levelv"] == 0
    assert sim.state["smub.source"]["output"] == "OUTPUT_OFF"


def test_retention_measures_conductance_at_a_real_bias():
    sim = tsp_sim.SimulatedKeithley(G0=2e-6)
    res = se.measure_retention(sim, "smub", {
        "read_voltage": 0.1, "retention_times_s": [1, 1],
        "compliance_A": 1e-6, "nplc": 0.1})
    assert res["conductance_S"] == pytest.approx([2e-6, 2e-6], rel=1e-3)
    zero_sets = [v for (t, n, k, v) in sim.log if n == "smub.measure" and k == "autozero"]
    assert "AUTOZERO_ONCE" in zero_sets
    assert sim.state["smub.measure"]["autozero"] == "AUTOZERO_AUTO"  # restored


def test_retention_refuses_negative_times():
    with pytest.raises(ValueError, match="retention_times_s"):
        se.measure_retention(tsp_sim.SimulatedKeithley(), "smub", {
            "read_voltage": 0.1, "retention_times_s": [1, -2],
            "compliance_A": 1e-6})


# --- 13. simulate_pulse_read ------------------------------------------------------------------------


def test_simulate_pulse_read_returns_a_baseline_plus_n_pulses():
    res = se.simulate_pulse_read(_params(n_pulses=7))
    assert len(res["conductance_S"]) == len(res["I_A"]) == len(res["time_s"]) == 8
    assert res["pulse_number"] == list(range(8))
    assert res["execution_path"] == "simulation"
    assert res["conductance_S"][0] == pytest.approx(1e-6)   # pre-stimulus state
    assert res["conductance_S"][1] > res["conductance_S"][0]


# --- 14/15. standard path parity and overrun visibility -------------------------------------------------


def test_lua_and_standard_paths_return_the_same_shape():
    _, lua = _run_pulse_read(compliance_A=1e-3)
    sim = tsp_sim.SimulatedKeithley()
    se_sup = se.supports_lua_execution
    try:
        se.supports_lua_execution = lambda inst: False
        std = se.pulse_read_sequence(sim, "smua", "smub", _params(compliance_A=1e-3))
    finally:
        se.supports_lua_execution = se_sup
    assert lua["pulse_number"] == std["pulse_number"] == list(range(6))
    for key in ("period_delivered", "settle_warning", "delivered_period_ms_median"):
        assert key in lua["timing"] and key in std["timing"], key


def test_an_infeasible_period_is_reported_as_not_delivered():
    """5 ms cannot hold 20 ms of stimulation: the results must say so."""
    _, results = _run_pulse_read(stim_period_ms=5.0, compliance_A=1e-3)
    t = results["timing"]
    assert t["period_delivered"] is False and t["period_verified"] is True
    assert t["delivered_period_ms_median"] > 5.0
    assert t["instrument_report"]["overruns"] > 0
    assert t["min_period_ms"] > 5.0


def test_a_feasible_period_is_reported_as_delivered():
    _, results = _run_pulse_read(compliance_A=1e-3)
    t = results["timing"]
    assert t["period_delivered"] is True and t["period_verified"] is True
    assert t["delivered_period_ms_median"] == pytest.approx(100.0, abs=0.1)


def test_an_unverifiable_period_is_never_reported_as_delivered():
    """No instrument report and no usable timestamps: unknown must not read as fine."""
    sim = tsp_sim.SimulatedKeithley()
    out = {}
    script = se.generate_pulse_read_lua_script("smua", "smub", _params(max_voltage=200.0, n_pulses=1))
    se.execute_lua_script_fast(sim, script.replace(se.LUA_TIMING_TOKEN, "SYN_NOTIMING"),
                               "smub", 2, expected_time_s=0.1, timing_out=out, n_is_exact=True)
    assert out["period_delivered"] is False and out["period_verified"] is False
