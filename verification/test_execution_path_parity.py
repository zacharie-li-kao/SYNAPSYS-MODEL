"""Execution-path parity and instrument-capability invariants.

These lock in the fixes recorded in AUDIT_2026-07-31.md §10 (H1/H2/H7/H22) and
the "Suite-Wide Invariants" section of CLAUDE.md. Each test corresponds to a
defect that was live in the codebase, so a failure here means an invariant has
been regressed, not that a test is being fussy.
"""

import math

import pytest

import synapse_engine as se
from fake_instrument import FakeKeithley, fake_lua_result


def _params(**overrides):
    p = {
        "mode": "electrical",
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "stim_width_ms": 1.0,
        "stim_period_ms": 10.0,
        "n_pulses": 5,
        "read_voltage": 0.1,
        "read_delay_ms": 2.0,
        "read_settle_delay_ms": 1.0,
        "compliance_A": 1e-3,
        "nplc": 0.1,
        "measure_avg": 1,
        "settle_ms": 1.0,
        "wire_mode": "2-Wire",
        "sample_id": "verification",
    }
    p.update(overrides)
    return p


# --- H2: the two paths must return the same thing ------------------------


@pytest.mark.parametrize("path", ["lua", "standard"])
def test_both_paths_return_baseline_plus_n_pulses(monkeypatch, path):
    """Index 0 is the pre-stimulus baseline on both paths.

    Before the fix the standard path had no baseline read, so index 0 meant
    "after the first pulse" there and "baseline" on the LUA path — and
    calculate_synapse_metrics computed PPF from indices 0 and 1 regardless.
    """
    inst = FakeKeithley(model="2636A")
    params = _params()

    if path == "lua":
        monkeypatch.setattr(
            se, "execute_lua_script_fast",
            lambda *a, **k: fake_lua_result(params["n_pulses"]),
        )
    else:
        monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)

    results = se.pulse_read_sequence(inst, "smua", "smub", params)

    assert results["execution_path"] == path
    assert len(results["conductance_S"]) == params["n_pulses"] + 1
    assert results["pulse_number"] == list(range(params["n_pulses"] + 1))


def test_failed_lua_is_recorded_as_standard(monkeypatch):
    """A LUA attempt that failed and fell through must not claim it ran on LUA."""
    inst = FakeKeithley(model="2636A")

    def boom(*a, **k):
        raise RuntimeError("simulated instrument fault")

    monkeypatch.setattr(se, "execute_lua_script_fast", boom)

    results = se.pulse_read_sequence(inst, "smua", "smub", _params())
    assert results["execution_path"] == "standard"
    assert len(results["conductance_S"]) == _params()["n_pulses"] + 1


@pytest.mark.parametrize("path", ["lua", "standard"])
def test_zero_read_bias_gives_nan_not_zero(monkeypatch, path):
    """Conductance is undefined at zero read bias — NaN, never a fabricated 0.

    A 0 here averages and plots as though it had been measured, which is the
    silent-degradation failure the project ground rules forbid.
    """
    inst = FakeKeithley(model="2636A")
    params = _params(read_voltage=0.0)

    if path == "lua":
        monkeypatch.setattr(
            se, "execute_lua_script_fast",
            lambda *a, **k: fake_lua_result(params["n_pulses"]),
        )
    else:
        monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)

    results = se.pulse_read_sequence(inst, "smua", "smub", params)
    assert all(math.isnan(g) for g in results["conductance_S"])


def test_standard_path_uses_fixed_ranging_and_instrument_compliance(monkeypatch):
    """Autorange makes per-pulse measurement time vary with the signal.

    The standard path must mirror the LUA script: autorange off, fixed voltage
    and current ranges, and voltage compliance taken from the instrument rather
    than a hardcoded 10 V.
    """
    inst = FakeKeithley(model="2636A")
    monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)

    se.pulse_read_sequence(inst, "smua", "smub", _params(stim_drive_type="I", stim_level=1e-4))

    assert inst.wrote(r"measure\.autorangei = smu[ab]\.AUTORANGE_OFF")
    assert inst.wrote(r"source\.rangev = ")
    assert inst.wrote(r"measure\.rangei = ")

    limitv = inst.writes_matching(r"source\.limitv = ")
    assert limitv, "current-driven stimulus must set a voltage compliance"

    # Asserted as a value, not as "not the string 10": a hardcoded 10.0 would
    # slip past a suffix check. The fake reports a 2636A, whose ceiling is
    # 200 V, so anything else means the compliance did not come from the
    # instrument.
    expected = se.MODEL_CAPABILITIES["2636A"]["max_voltage"]
    written = [float(w.split("=")[-1].strip()) for w in limitv]
    assert all(v == expected for v in written), (
        f"voltage compliance {written} must come from the instrument "
        f"({expected} V on a 2636A), not a hardcoded value"
    )


def test_both_paths_share_one_averaging_derivation():
    """`derive_measurement_averaging` is the single source of truth.

    The standard path once imposed its own ceiling of 50 against the hardware
    limit of 100, so the same "Samples per Pulse" setting averaged differently
    depending on which path ran.
    """
    params = _params(measure_avg=80, read_width_ms=200.0)
    nplc, measure_avg = se.derive_measurement_averaging(params)
    assert measure_avg <= se.FILTER_COUNT_MAX
    assert measure_avg > 50, (
        "a request of 80 must not be silently halved by a second, lower ceiling"
    )
    assert nplc >= se.MIN_NPLC


# --- H1: canonical mode vocabulary ---------------------------------------


def test_unregistered_mode_raises_rather_than_changing_path():
    """An unknown mode must raise, not fall through to the standard path."""
    inst = FakeKeithley(model="2636A")
    with pytest.raises(ValueError, match="unknown mode"):
        se.pulse_read_sequence(inst, "smua", "smub", _params(mode="Memristor (Pulse)"))


def test_gui_labels_round_trip_to_canonical_modes():
    import keithley_analyser as ka

    for label, mode in ka.SYNAPSE_MODE_BY_LABEL.items():
        assert ka.canonical_synapse_mode(label) == mode
        assert ka.synapse_mode_display(mode) == label

    with pytest.raises(ValueError):
        ka.canonical_synapse_mode("Not A Real Mode")


def test_every_mapped_mode_is_dispatched_by_the_engine():
    """The GUI must not be able to name a mode the engine will not dispatch."""
    import inspect

    import keithley_analyser as ka

    source = inspect.getsource(se.pulse_read_sequence)
    for mode in ka.SYNAPSE_MODE_BY_LABEL.values():
        assert f"'{mode}'" in source, f"mode {mode!r} is unreachable in the engine"


# --- H7: capability table is the single source of truth -------------------


@pytest.mark.parametrize(
    "model",
    ["2601B", "2604B", "2611B", "2614B", "2634B", "2635B", "2636A"],
)
def test_lua_capable_models_are_recognised(model):
    """Six of these were silently denied LUA before MODEL_CAPABILITIES existed."""
    inst = FakeKeithley(model=model)
    assert se.supports_lua_execution(inst) is True


def test_unidentifiable_instrument_raises():
    """Guessing a voltage ceiling is a safety decision, not a fallback."""
    inst = FakeKeithley(idn_raises=RuntimeError("VISA timeout"))
    with pytest.raises(se.InstrumentIdentificationError):
        se.validate_instrument_model(inst)

    inst2 = FakeKeithley(idn_raises=RuntimeError("VISA timeout"))
    with pytest.raises(se.InstrumentIdentificationError):
        se.supports_lua_execution(inst2)


def test_capability_table_is_consistent_with_its_consumers():
    """The four capability helpers must all read from MODEL_CAPABILITIES."""
    for model in se.MODEL_CAPABILITIES:
        inst = FakeKeithley(model=model)
        assert se.get_max_voltage(inst) == se.MODEL_CAPABILITIES[model]["max_voltage"]
        assert se.get_max_current(inst) == se.MODEL_CAPABILITIES[model]["max_current_dc"]
        assert se.supports_lua_execution(inst) == se.MODEL_CAPABILITIES[model]["lua"]


# --- H22: a software fault must not be reported as an instrument failure ---


def test_console_encoding_shim_is_wired_into_entry_points():
    """Non-ASCII physics notation in print() is safe only because of this."""
    import io
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entry_points = [
        "hub.py", "keithley_analyser.py", "Main.py", "fitting.py",
        "data_converter.py", "generate_test_data.py", "synapse_engine.py",
    ]
    for name in entry_points:
        with io.open(os.path.join(root, name), encoding="utf-8") as f:
            src = f.read()
        assert "enable_utf8_console()" in src, f"{name} does not enable UTF-8 console"


def test_lua_try_block_guards_only_the_instrument_call():
    """Result processing and console output must sit outside the try.

    When the try wrapped the whole LUA section, a `✓` that failed to encode on
    a cp1252 console was reported as "LUA execution failed" and the entire
    measurement was silently re-run on the slow path.
    """
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(se.pulse_read_sequence))
    lines = src.splitlines()

    try_lines = [i for i, ln in enumerate(lines) if ln.strip() == "try:"]
    assert try_lines, "expected a guarded LUA call"

    for i in try_lines:
        # Collect the body until the matching except.
        body = []
        indent = len(lines[i]) - len(lines[i].lstrip())
        for ln in lines[i + 1:]:
            if ln.strip().startswith("except") and (len(ln) - len(ln.lstrip())) == indent:
                break
            body.append(ln)
        joined = "\n".join(body)
        if "execute_lua_script_fast" in joined:
            assert "results[" not in joined, (
                "result processing must not sit inside the LUA try block"
            )
            assert "print(" not in joined, (
                "console output must not sit inside the LUA try block"
            )
