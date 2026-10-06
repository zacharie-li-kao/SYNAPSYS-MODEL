"""Mains line-frequency resolution — the installation-agnostic contract.

Motivated by a field report (2026-08-18, Shiga, Japan): every LUA script forced
`localnode.linefreq = 50` (the Barcelona default) because the GUI had no control
for it, so a user on a 60 Hz grid had their ADC synchronised to the wrong mains
frequency and no NPLC setting could reject line pickup. The result was a clean
aliased sinusoid riding on nanoamp-level conductance data.

The contract locked in here: when the caller does not specify a line frequency,
the ENGINE ASKS THE INSTRUMENT — the 2600 series auto-detects its mains
frequency at power-up and reports it via `localnode.linefreq` — so the software
is correct on any grid without configuration. An explicit
`params['line_freq_hz']` still wins, and anything other than 50/60 raises
rather than silently degrading rejection.
"""

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


# --- The resolver itself ----------------------------------------------------


def test_resolver_adopts_the_instruments_detected_frequency():
    inst = FakeKeithley(model="2636A", line_freq_hz=60)
    params = _params()
    assert "line_freq_hz" not in params

    freq = se.resolve_line_freq(inst, params)

    assert freq == 60.0
    assert params["line_freq_hz"] == 60.0, (
        "the detected value must be stamped into params so every consumer "
        "(generator, averaging derivation, drift analysis) uses one value"
    )


def test_resolver_lets_an_explicit_frequency_win():
    inst = FakeKeithley(model="2636A", line_freq_hz=60)
    params = _params(line_freq_hz=50)

    freq = se.resolve_line_freq(inst, params)

    assert freq == 50.0
    assert not any("localnode.linefreq" in q for q in inst.queries), (
        "an explicit override must not be second-guessed by querying the "
        "instrument"
    )


def test_resolver_treats_explicit_none_as_auto():
    """The GUI's Auto setting sends line_freq_hz=None — that means 'ask the
    instrument', exactly like an absent key."""
    inst = FakeKeithley(model="2636A", line_freq_hz=60)
    params = _params(line_freq_hz=None)

    assert se.resolve_line_freq(inst, params) == 60.0
    assert params["line_freq_hz"] == 60.0


def test_simulation_fallback_accepts_none():
    """Simulation paths have no instrument to ask; explicit None falls back to
    the documented default instead of crashing on float(None)."""
    assert se._line_freq_hz({"line_freq_hz": None}) == se.DEFAULT_LINE_FREQ_HZ


def test_resolver_raises_on_an_unusable_reported_frequency():
    inst = FakeKeithley(model="2636A", line_freq_hz=55)
    with pytest.raises(ValueError):
        se.resolve_line_freq(inst, _params())


# --- Through the measurement entry points -----------------------------------


def test_lua_script_carries_the_detected_frequency(monkeypatch):
    inst = FakeKeithley(model="2636A", line_freq_hz=60)
    params = _params()
    captured = {}

    def capture(instrument, lua_script, read_ch, n_points, **kwargs):
        captured["script"] = lua_script
        return fake_lua_result(params["n_pulses"])

    monkeypatch.setattr(se, "execute_lua_script_fast", capture)

    results = se.pulse_read_sequence(inst, "smua", "smub", params)

    assert "localnode.linefreq = 60" in captured["script"]
    assert results["params"]["line_freq_hz"] == 60.0


def test_standard_path_synchronises_to_the_detected_frequency(monkeypatch):
    inst = FakeKeithley(model="2636A", line_freq_hz=60)
    monkeypatch.setattr(se, "supports_lua_execution", lambda inst: False)

    se.pulse_read_sequence(inst, "smua", "smub", _params())

    assert inst.wrote(r"localnode\.linefreq = 60"), (
        "the PC-timed path must sync the ADC to the same detected mains "
        "frequency as the LUA path"
    )
    assert not inst.wrote(r"localnode\.linefreq = 50")
