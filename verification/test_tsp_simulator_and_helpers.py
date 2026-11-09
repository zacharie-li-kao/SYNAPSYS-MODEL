"""The TSP simulator's fidelity to the 2600B manual, and the shared ADC helpers.

`tsp_sim.SimulatedKeithley` executes generated scripts in a real Lua runtime.
Its value rests on modelling the instrument behaviours that matter for timing
and accuracy as the Reference Manual documents them; these tests pin that, so a
later edit to the simulator cannot quietly make every engine test meaningless.
"""

import numpy as np
import pytest

import synapse_engine as se
import tsp_sim


def _measure_once(sim, ch="smub", rangei=1e-7):
    sim.write(f"{ch}.source.output = {ch}.OUTPUT_ON")
    sim.write(f"{ch}.source.levelv = 0.1")
    sim.write(f"{ch}.measure.rangei = {rangei}")
    sim.write(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")
    sim.write(f"{ch}.measure.nplc = 0.01")
    t0 = sim.clock
    sim.write(f"{ch}.measure.i()")
    return sim.clock - t0


def test_default_measure_delay_is_auto_and_range_dependent():
    """RM 7-202: DELAY_AUTO is the 263xB default and applies to direct measure calls."""
    slow = _measure_once(tsp_sim.SimulatedKeithley(), rangei=1e-7)
    fast = _measure_once(tsp_sim.SimulatedKeithley(), rangei=1e-3)
    assert slow - fast == pytest.approx(0.015 - 1e-4, abs=1e-6)


def test_measure_delay_zero_removes_it():
    sim = tsp_sim.SimulatedKeithley()
    sim.write("smub.measure.delay = 0")
    assert _measure_once(sim, rangei=1e-7) < 1e-3


def test_delay_zero_is_rejected():
    """RM 7-47: 'You cannot set a delay for zero seconds.'"""
    sim = tsp_sim.SimulatedKeithley()
    sim.write("delay(0)")
    assert sim.errors and "zero seconds" in sim.errors[-1]


def test_unknown_attribute_is_an_instrument_error():
    sim = tsp_sim.SimulatedKeithley()
    sim.write("smua.source.levelvv = 1")
    assert sim.errors


def test_ascii_precision_governs_printbuffer():
    sim = tsp_sim.SimulatedKeithley(G0=1.234567890123e-6)
    sim.write("smub.nvbuffer1.clear()")
    sim.write("smub.source.output = smub.OUTPUT_ON")
    sim.write("smub.source.levelv = 1")
    sim.write("smub.measure.i(smub.nvbuffer1)")
    six = sim.query("printbuffer(1, 1, smub.nvbuffer1.readings)")
    sim.write("format.asciiprecision = 15")
    fifteen = sim.query("printbuffer(1, 1, smub.nvbuffer1.readings)")
    assert len(six.split("e")[0].replace(".", "").lstrip("-")) == 6
    assert float(fifteen) == pytest.approx(1.234567890123e-6, rel=1e-12)


# --- shared helpers ------------------------------------------------------

def test_range_settling_matches_the_manual_table():
    assert se.range_settling_s(1e-7) == 0.015
    assert se.range_settling_s(1e-8) == 0.090
    assert se.range_settling_s(1e-6) == 2.5e-3
    assert se.range_settling_s(5e-7) == 2.5e-3      # next range up
    with pytest.raises(ValueError):
        se.range_settling_s(20.0)


def test_check_read_settle_flags_only_short_settles():
    assert se.check_read_settle(0.020, 1e-7) is None
    msg = se.check_read_settle(0.001, 1e-7)
    assert msg and "15" in msg


def test_lua_delay_never_emits_zero_and_rejects_negative():
    assert se.lua_delay(0) == ""
    assert se.lua_delay(0.002) == "delay(0.002)"
    with pytest.raises(ValueError):
        se.lua_delay(-1e-3)
    with pytest.raises(ValueError):
        se.lua_delay(float("nan"))


def test_adc_setup_turns_delay_off_and_autozeros_after_nplc():
    lines = se._emit_read_adc_setup("smub", 0.1, 4)
    text = "\n".join(lines)
    assert "smub.measure.delay = 0" in text
    # The autozero cache is per aperture: ONCE must follow the nplc write.
    assert lines.index("smub.measure.nplc = 0.1") < lines.index(
        "smub.measure.autozero = smub.AUTOZERO_ONCE")
    assert "smub.measure.filter.count = 4" in text


def test_adc_setup_executes_on_the_simulated_instrument():
    sim = tsp_sim.SimulatedKeithley()
    for line in se._emit_read_adc_setup("smub", 0.1, 1):
        sim.write(line)
    assert sim.errors == []
    assert sim.state["smub.measure"]["delay"] == 0
    assert sim.state["smub.measure.filter"]["enable"] == "FILTER_OFF"


def test_drain_discards_a_stale_completion_token():
    sim = tsp_sim.SimulatedKeithley()
    sim.out_queue.extend([f"{se.LUA_TIMING_TOKEN} overruns=0", se.LUA_DONE_TOKEN])
    drained = se._drain_output(sim)
    assert len(drained) == 2 and sim.out_queue == []
