"""Step 9 hardware fixes — H3, H5, H6, H15, H17, H20.

Each is a local defect in the instrument stack that produced either wrong data
or a spurious failure. None needs hardware: the retrieval and safety logic is
exercised against FakeKeithley, and the PV extraction is pure numerics.
"""

import inspect

import numpy as np
import pytest

import keithley_analyser as ka
import synapse_engine as se
from fake_instrument import FakeKeithley


# --- H17: PV parameter extraction -----------------------------------------


def _ideal_jv(v_oc=0.6, j_sc=20.0, n_points=61, v_min=-0.2, v_max=1.0, area=1.0):
    """A well-behaved diode JV curve, in AMPS (the function converts to mA/cm²)."""
    v = np.linspace(v_min, v_max, n_points)
    # Simple exponential diode plus photocurrent, shaped to cross zero near v_oc.
    j = -j_sc + j_sc * np.exp((v - v_oc) * 12.0)
    return v, j * area / 1e3     # mA/cm² back to A


def test_voc_is_the_same_on_a_forward_and_a_reverse_sweep():
    """H17 — the crossing search assumed an ascending sweep.

    It took the LAST index with J < 0 and the FIRST with J > 0. On a reverse
    sweep those sit at opposite ends of the data, so the interpolation
    extrapolated far outside the measured range, and FF and PCE inherited it.
    """
    v, i = _ideal_jv()
    forward = ka.calculate_pv_parameters(v, i, area=1.0, light_power=1000.0)
    reverse = ka.calculate_pv_parameters(v[::-1], i[::-1], area=1.0, light_power=1000.0)

    assert forward["Voc"] is not None
    assert reverse["Voc"] is not None
    assert forward["Voc"] == pytest.approx(reverse["Voc"], abs=1e-6), (
        f"Voc depends on sweep direction: {forward['Voc']} vs {reverse['Voc']}"
    )
    assert forward["Voc"] == pytest.approx(0.6, abs=0.05)
    # And it must lie inside the measured range, not extrapolated beyond it.
    assert v.min() <= forward["Voc"] <= v.max()


def test_fill_factor_and_efficiency_agree_across_sweep_direction():
    v, i = _ideal_jv()
    forward = ka.calculate_pv_parameters(v, i, area=1.0, light_power=1000.0)
    reverse = ka.calculate_pv_parameters(v[::-1], i[::-1], area=1.0, light_power=1000.0)

    for key in ("Jsc", "FF", "PCE"):
        assert forward[key] == pytest.approx(reverse[key], rel=1e-6), (
            f"{key} depends on sweep direction"
        )
    assert 0 < forward["FF"] <= 100


def test_jsc_is_interpolated_to_zero_volts():
    """Jsc used to be the current at the nearest measured point."""
    # Deliberately offset so no sample sits exactly at V = 0.
    v = np.linspace(-0.205, 1.0, 60)
    j = -20.0 + 20.0 * np.exp((v - 0.6) * 12.0)
    assert not np.any(v == 0.0)

    result = ka.calculate_pv_parameters(v, j / 1e3, area=1.0, light_power=1000.0)
    assert result["Jsc"] == pytest.approx(20.0, rel=0.05)


def test_an_undeterminable_voc_is_reported_as_such():
    """The old fallback silently substituted the voltage of minimum |J|."""
    # A sweep that never crosses zero current.
    v = np.linspace(-0.2, 0.2, 21)
    j = np.full_like(v, -20.0)

    result = ka.calculate_pv_parameters(v, j / 1e3, area=1.0, light_power=1000.0)
    assert result["Voc"] is None
    assert result["FF"] is None
    assert any("sign" in w for w in result["warnings"])


def test_zero_light_power_does_not_produce_an_efficiency():
    v, i = _ideal_jv()
    result = ka.calculate_pv_parameters(v, i, area=1.0, light_power=0.0)
    assert result["PCE"] is None
    assert any("light power" in w for w in result["warnings"])


# --- H15: safety checks ----------------------------------------------------


def _synapse_params(**overrides):
    p = {
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "read_voltage": 0.1,
        "compliance_A": 1e-3,
    }
    p.update(overrides)
    return p


def test_a_voltage_stimulus_is_checked_against_the_voltage_limit(monkeypatch):
    """H15 — SRDP and STDP compared a VOLTAGE against the 1.5 A current limit.

    A 2 V pre-spike was refused on a 2612B as "2.0A", while no voltage check
    existed at all.
    """
    monkeypatch.setattr(se, "safety_check", lambda *a, **k: True)
    inst = FakeKeithley(model="2612B")   # 200 V, 1.5 A

    # 2 V is far below the 200 V ceiling and must be accepted.
    assert ka.check_synapse_safety(inst, _synapse_params(stim_level=2.0), "STDP") is True

    # 250 V exceeds it and must be refused.
    with pytest.raises(ValueError, match="exceeds"):
        ka.check_synapse_safety(inst, _synapse_params(stim_level=250.0), "STDP")


def test_a_current_stimulus_is_checked_against_the_current_limit(monkeypatch):
    monkeypatch.setattr(se, "safety_check", lambda *a, **k: True)
    inst = FakeKeithley(model="2612B")   # 1.5 A

    assert ka.check_synapse_safety(
        inst, _synapse_params(stim_drive_type="I", stim_level=1.0), "SRDP") is True

    with pytest.raises(ValueError, match="exceeds"):
        ka.check_synapse_safety(
            inst, _synapse_params(stim_drive_type="I", stim_level=5.0), "SRDP")


def test_the_read_voltage_is_checked_too(monkeypatch):
    """In some modes the read bias is applied for the whole inter-pulse period."""
    monkeypatch.setattr(se, "safety_check", lambda *a, **k: True)
    inst = FakeKeithley(model="2601B")   # 40 V ceiling

    with pytest.raises(ValueError, match="read voltage"):
        ka.check_synapse_safety(inst, _synapse_params(read_voltage=100.0), "Cycle")


def test_declining_the_device_safety_prompt_stops_the_run(monkeypatch):
    monkeypatch.setattr(se, "safety_check", lambda *a, **k: False)
    inst = FakeKeithley(model="2636A")
    assert ka.check_synapse_safety(inst, _synapse_params(stim_level=5.0), "Basic") is False


@pytest.mark.parametrize("mode_fn", ["run_srdp_characterization",
                                     "run_stdp_characterization",
                                     "run_cycle_characterization"])
def test_every_synapse_mode_performs_a_safety_check(mode_fn):
    """SRDP, STDP and Cycle all reached the instrument with no check.

    Basic and Visual both checked, so the same stimulus was accepted or refused
    depending only on which tab launched it.
    """
    fn = getattr(ka, mode_fn)
    source = inspect.getsource(fn)
    # SRDP and the transistor run are wrapped; follow through to the wrapped
    # implementation, which is where the measurement actually happens.
    for original_name in ("original_run_srdp", "original_run_transistor"):
        if original_name in source:
            source += inspect.getsource(getattr(ka, original_name))
    assert "check_synapse_safety" in source, (
        f"{mode_fn} performs no safety check"
    )


def test_the_hardcoded_2612b_current_comparison_is_gone():
    source = inspect.getsource(ka)
    assert '"2612B" in model and stim_current > 1.5' not in source
    assert '"2612B" in model and max_current > 1.5' not in source


# --- H18: the run buttons must call the wrappers ---------------------------


def test_the_run_buttons_are_bound_to_the_wrappers():
    """H18 — `command=` captured the function object at button-creation time.

    Both buttons are created hundreds of lines above the wrappers that rebind
    those names, so they held the originals and the wrappers never ran. The
    visible consequence: `sync_transistor_to_diode()` never executed, so the
    Transistor panel ignored its own voltage fields.
    """
    assert ka.run_transistor_button.cget("command"), "no command bound"
    assert ka.run_srdp_button.cget("command"), "no command bound"

    source = inspect.getsource(ka)
    assert "run_transistor_button.configure(command=run_transistor_measurement)" in source
    assert "run_srdp_button.configure(command=run_srdp_characterization)" in source

    # And the rebinding must come AFTER the wrappers are defined, or it binds
    # the originals again.
    wrapper_pos = source.index("original_run_transistor = run_transistor_measurement")
    rebind_pos = source.index("run_transistor_button.configure(command=")
    assert rebind_pos > wrapper_pos


def test_the_transistor_voltage_redirect_is_restored():
    """Enabling the H18 wrapper must not leak the redirect into diode mode.

    `sync_transistor_to_diode` rebinds the module-level `start_voltage` etc. to
    the TRANSISTOR widgets. Left permanent — which is what it did — a
    subsequent DIODE sweep would read the transistor panel's voltages. That was
    latent only because the wrapper was never called; fixing the binding
    without restoring would have made it live.
    """
    before = (ka.start_voltage, ka.end_voltage, ka.voltage_step)

    previous = ka.sync_transistor_to_diode()
    assert ka.start_voltage is ka.start_voltage_trans, "the redirect did not apply"

    ka.restore_voltage_entry_bindings(previous)
    after = (ka.start_voltage, ka.end_voltage, ka.voltage_step)
    assert after == before, "the voltage-entry bindings were not restored"

    # And the wrapper does the restore itself, in a finally.
    wrapper_source = inspect.getsource(ka.run_transistor_measurement)
    assert "finally:" in wrapper_source
    assert "restore_voltage_entry_bindings" in wrapper_source


# --- H14: the SRDP frequency ceiling ---------------------------------------


def test_the_srdp_frequency_ceiling_includes_settle_and_lua_overhead():
    """H14 — the ceiling omitted read_settle_delay_ms and the ~2 ms per-pulse
    LUA overhead that pulse-read's own check includes, so it advertised a
    maximum the instrument could not deliver. The run then proceeded at the
    achievable rate while recording the NOMINAL frequency, corrupting the
    x-axis exactly where f0 and the slope are determined."""
    source = inspect.getsource(ka.original_run_srdp)
    assert "read_settle_delay_entry" in source, (
        "the settle delay is still omitted from the achievable-frequency check"
    )
    assert "LUA_PER_PULSE_OVERHEAD_MS" in source


# --- H3: STDP protocol integrity -------------------------------------------


def test_a_transient_lua_failure_does_not_demote_the_whole_stdp_sweep():
    """H3 — `use_lua` was cleared inside the per-Δt handler and never restored.

    One transient failure silently demoted every remaining point to the
    standard path, and the resulting array mixed two protocols with nothing
    recording which point came from which.
    """
    source = inspect.getsource(se.measure_stdp)

    assert "lua_supported = supports_lua_execution" in source, (
        "the sweep-level capability flag is gone"
    )
    # The per-point decision must be re-taken from the sweep-level flag.
    assert "use_lua = lua_supported" in source
    assert "results[\"execution_path\"]" in source, (
        "each point must record which protocol produced it"
    )


def test_the_standard_stdp_path_uses_level_changes_not_relay_toggling():
    """CLAUDE.md forbids OUTPUT_ON/OFF for STDP spikes.

    Relay latency is milliseconds and non-deterministic, so at a 1 ms pulse
    width and Δt = 5 ms the toggling path was measuring relay jitter rather
    than controlling Δt.
    """
    source = inspect.getsource(se.measure_stdp)
    standard = source.split("=== STANDARD PATH")[1]

    # The pair loop must contain no relay activity at all.
    pair_loop = standard.split("for pair_idx in range(n_pairs):")[1].split(
        "# Measure final conductance")[0]
    assert "OUTPUT_ON" not in pair_loop, "relay toggling inside the STDP pair loop"
    assert "OUTPUT_OFF" not in pair_loop
    assert "source.levelv" in pair_loop


def test_the_standard_stdp_path_honours_the_requested_nplc():
    """It hardcoded nplc = 0.1, discarding the GUI value."""
    source = inspect.getsource(se.measure_stdp)
    assert "measure.nplc = 0.1" not in source
    assert "derive_measurement_averaging(base_params)" in source


# --- H5: buffer retrieval --------------------------------------------------


def test_lua_retrieval_asks_the_buffer_how_many_points_it_holds():
    """H5 — visual continuous I(t) failed at retrieval, every time.

    `n_expected_points` is deliberately inflated by 2*n_pulses "for safety",
    and printbuffer was then called with that upper bound and no `.n` guard.
    Requesting indices past `.n` raises instrument error 5038.
    """
    source = inspect.getsource(se.execute_lua_script_fast)
    assert "nvbuffer1.n" in source, "the retrieval never queries the buffer count"

    # The chunk loop must iterate over what the buffer holds, not the bound.
    assert "range(1, n_to_read + 1, CHUNK_SIZE)" in source
    assert "range(1, n_expected_points + 1, CHUNK_SIZE)" not in source


@pytest.mark.parametrize(
    "n_in_buffer, n_expected, n_should_read",
    [
        # Buffer shorter than the bound: continuous mode's deliberate
        # over-allocation. Reading to the bound is what raised error 5038.
        (6, 20, 6),
        # Buffer longer than the bound: must not read past what was asked for
        # either, or the extra points are silently appended as data.
        (30, 20, 20),
        (12, 12, 12),
    ],
)
def test_lua_retrieval_clamps_to_the_available_points(
        monkeypatch, n_in_buffer, n_expected, n_should_read):
    """The clamp is min(available, expected) — exercised in both directions."""
    inst = FakeKeithley(model="2636A")

    requested = []

    def fake_query(command):
        if "nvbuffer1.n" in command:
            return str(n_in_buffer)
        if "printbuffer" in command:
            requested.append(command)
            lo, hi = (int(t) for t in command.split("(")[1].split(",")[:2])
            return ", ".join(["1.0e-06"] * (hi - lo + 1))
        if "*IDN?" in command:
            return "Keithley Instruments Inc., Model 2636A, 1, 1.4"
        if "errorqueue.count" in command:
            return "0"
        raise AssertionError(f"unexpected query {command}")

    monkeypatch.setattr(inst, "query", fake_query)
    monkeypatch.setattr(se, "_wait_lua_complete", lambda *a, **k: None)

    timestamps, currents = se.execute_lua_script_fast(
        inst, "print('x')", "smub",
        n_expected_points=n_expected, expected_time_s=0.01,
    )

    assert requested, "printbuffer was never called"
    highest = max(
        int(command.split("(")[1].split(",")[1])
        for command in requested
    )
    assert highest == n_should_read, (
        f"read up to index {highest}, expected {n_should_read} "
        f"(buffer held {n_in_buffer}, bound was {n_expected})"
    )
    assert len(currents) == n_should_read


def test_an_empty_buffer_after_execution_is_an_error(monkeypatch):
    """A script that ran but recorded nothing must not return silently."""
    inst = FakeKeithley(model="2636A")

    def fake_query(command):
        if "nvbuffer1.n" in command:
            return "0"
        if "*IDN?" in command:
            return "Keithley Instruments Inc., Model 2636A, 1, 1.4"
        if "errorqueue.count" in command:
            return "0"
        raise AssertionError(f"unexpected query {command}")

    monkeypatch.setattr(inst, "query", fake_query)
    monkeypatch.setattr(se, "_wait_lua_complete", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="empty"):
        se.execute_lua_script_fast(inst, "print('x')", "smub",
                                   n_expected_points=10, expected_time_s=0.01)


# --- H6: JV buffer capacity ------------------------------------------------


def test_the_jv_sweep_sizes_its_buffer():
    """H6 — nvbuffer1 was cleared but its capacity never set.

    The 2600-series default is 100 readings; a −0.2 V → 1.0 V sweep at 0.01 V
    is 121 points, so the run overflowed with error 5038 and the oldest 21
    readings had already been overwritten.
    """
    source = inspect.getsource(ka.run_measurement_buffered)
    assert "nvbuffer1.capacity" in source, "the JV sweep never sets buffer capacity"
    assert "BUFFER_CAPACITY_MAX" in source, "no guard against exceeding the maximum"


def test_the_jv_sweep_synchronises_the_adc_to_mains():
    """H16 — run_measurement_buffered never wrote localnode.linefreq, unlike
    every synapse path, so a JV-only session integrated against whatever the
    node happened to hold and NPLC did not reject mains pickup."""
    source = inspect.getsource(ka.run_measurement_buffered)
    assert "localnode.linefreq" in source


# --- H20: the finally clause -----------------------------------------------


def test_the_jv_cleanup_cannot_raise_nameerror():
    """H20 — the `finally` referenced `channel_cmd`, first assigned partway
    through the try. Any earlier exception made the cleanup itself raise
    NameError, replacing the real error and leaving the output ON."""
    source = inspect.getsource(ka.run_measurement_buffered)
    finally_block = source.split("finally:")[-1]
    assert "'channel_cmd' in locals()" in finally_block, (
        "the cleanup still assumes channel_cmd is bound"
    )
