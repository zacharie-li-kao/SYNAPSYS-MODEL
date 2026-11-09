"""Cycle, STDP and SRDP execution, run for real on the simulated TSP instrument.

`tsp_sim.SimulatedKeithley` executes the generated scripts in a Lua runtime with
a virtual clock, so these tests observe what the instrument would do: delivered
intervals, level timelines, range registers, outputs after a fault.
"""

import re
import time

import numpy as np
import pytest

import synapse_cycle as sc
import synapse_engine as se
import tsp_sim


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _train(**kw):
    cfg = dict(topology='single', write_ch='smua', read_ch='smua', stim_drive_type='V',
               stim_level=1.0, stim_width_ms=10, stim_period_ms=50, n_pulses=6,
               read_voltage=0.1, read_delay_ms=5, compliance_A=1e-3, nplc=0.1,
               settle_ms=5, measure_avg=1)
    cfg.update(kw)
    return cfg


def _run(sim, a, b, n_cycles=2, inter_train=200, inter_cycle=300, script_edit=None):
    a, b = dict(a), (dict(b) if b else None)
    mv = se.get_max_voltage(sim)
    a['max_voltage'] = mv
    if b:
        b['max_voltage'] = mv
    a['line_freq_hz'] = sim.state['localnode']['linefreq']
    script, bmap = se.generate_cycle_lua_script(a, b, n_cycles, inter_train, inter_cycle)
    if script_edit:
        script = script_edit(script)
    return se.execute_cycle_lua(sim, script, bmap, 1.0), bmap


def _blocks(result, bmap, n_cycles):
    """Per-train, per-cycle timestamp blocks (shared buffer, A then B per cycle)."""
    ts = result['train_a']['timestamps']
    pa = bmap['train_a']['n_points_per_cycle']
    pb = bmap['train_b']['n_points_per_cycle']
    return [(ts[c * (pa + pb): c * (pa + pb) + pa],
             ts[c * (pa + pb) + pa: (c + 1) * (pa + pb)]) for c in range(n_cycles)]


# --- timing ----------------------------------------------------------------

def test_every_pulse_interval_equals_the_period_including_the_first():
    sim = tsp_sim.SimulatedKeithley()
    result, bmap = _run(sim, _train(), _train(stim_level=-1.0))
    for a_blk, b_blk in _blocks(result, bmap, 2):
        for blk in (a_blk, b_blk):
            iv = np.diff(blk[1:])     # pulse 1 -> 2 onward; baseline->1 is not a period
            assert np.allclose(iv, 0.05, atol=1e-6), iv
    rec = result['timing']['train_a']
    assert rec['status'] == 'ok' and rec['n_intervals'] == 10
    assert rec['n_intervals_exceeding_threshold'] == 0
    # Onsets of the stimulus itself (levelv == 1.0), not only the read stamps.
    onsets = [t for (t, n, k, v) in sim.log
              if n == 'smua.source' and k == 'levelv' and v == 1.0]
    assert np.allclose(np.diff(onsets[:6]), 0.05, atol=1e-4)


def test_an_overrun_is_rebaselined_reported_and_does_not_eat_the_inter_train_delay():
    sim = tsp_sim.SimulatedKeithley()
    # 12 ms period against ~17 ms of work per pulse: every pulse overruns.
    a = _train(stim_period_ms=12)
    result, bmap = _run(sim, a, _train(stim_level=-1.0, stim_period_ms=12), inter_train=400)
    stats = result['timing']['instrument_report']
    assert stats['a_overruns'] == 12 and stats['b_overruns'] == 12      # positive control
    assert stats['a_worst_ms'] > 1.0
    assert result['timing']['train_a']['n_intervals_exceeding_threshold'] > 0
    assert result['timing']['train_a']['instrument_overruns'] == 12
    # Re-baselined: pulses are spaced by the work, never fired back to back.
    iv = np.diff(_blocks(result, bmap, 2)[0][0][1:])
    assert np.all(iv > 0.012)
    # The 400 ms delay is measured from the ACTUAL end of Train A.
    a_blk, b_blk = _blocks(result, bmap, 2)[0]
    assert 0.400 <= b_blk[0] - a_blk[-1] < 0.400 + 0.06


def test_drift_analysis_reports_no_false_drift_across_trains_and_cycles():
    sim = tsp_sim.SimulatedKeithley()
    result, _ = _run(sim, _train(), _train(stim_level=-1.0), n_cycles=3, inter_train=500,
                     inter_cycle=1000)
    for key in ('train_a', 'train_b'):
        rec = result['timing'][key]
        assert rec['status'] == 'ok' and rec['n_blocks'] == 3
        assert rec['n_intervals_exceeding_threshold'] == 0
        assert rec['max_late_ms'] < 0.01 and rec['instrument_overruns'] == 0


def test_the_minimum_period_estimate_is_a_tight_lower_bound():
    sim = tsp_sim.SimulatedKeithley()
    result, bmap = _run(sim, _train(stim_width_ms=2, read_delay_ms=2, stim_period_ms=5), None, n_cycles=1)
    est = bmap['train_a']['min_period_ms']
    delivered = result['timing']['train_a']['delivered_period_ms_median']
    assert est <= delivered <= est + 2.0, (est, delivered)
    assert bmap['feasible_time_s'] > 6 * 0.005


def test_cycle_averaging_is_the_shared_rule_at_60_hz():
    cfg = _train(read_width_ms=50, nplc=0.01, line_freq_hz=60.0)
    nplc, avg = se.derive_measurement_averaging(cfg)
    script, _ = se.generate_cycle_lua_script(cfg, None, 1)
    assert f"smua.measure.filter.count = {avg}" in script and f"smua.measure.nplc = {nplc}" in script


# --- single-channel read disturb --------------------------------------------

def test_single_channel_read_line_returns_to_zero_between_pulses():
    sim = tsp_sim.SimulatedKeithley()
    _run(sim, _train(), None, n_cycles=1)
    seq = "".join({1.0: "S", 0.1: "R", 0.0: "0"}[float(v)] for (_, n, k, v) in sim.log
                  if n == 'smua.source' and k == 'levelv' and float(v) in (1.0, 0.1, 0.0))
    assert re.search(r"R0(SR0){6}", seq), seq   # baseline read, then stim/read/0 V per pulse
    assert "RR" not in seq and "SS" not in seq


# --- range handling ----------------------------------------------------------

def test_swapped_role_trains_with_incompatible_ranges_are_rejected():
    a = _train(topology='dual', write_ch='smua', read_ch='smub', stim_level=1.0)
    b = _train(topology='dual', write_ch='smub', read_ch='smua', stim_level=-1.0)
    with pytest.raises(ValueError, match="share SMU channel"):
        se.generate_cycle_lua_script(a, b, 1)
    # Control: compatible swapped roles (stimulus fits the read range) are fine.
    a = _train(topology='dual', write_ch='smua', read_ch='smub', stim_level=0.1)
    b = _train(topology='dual', write_ch='smub', read_ch='smua', stim_level=-0.1)
    script, _ = se.generate_cycle_lua_script(a, b, 1)
    assert "smub" in script


def test_current_drive_ranges_in_the_cycle_script():
    sim = tsp_sim.SimulatedKeithley()
    a = _train(topology='dual', write_ch='smua', read_ch='smub', stim_drive_type='I',
               stim_level=5e-3, compliance_A=1e-4)
    _run(sim, a, None, n_cycles=1)
    src = sim.state['smua.source']
    assert src['func'] == 'OUTPUT_DCAMPS' and src['rangei'] == 1e-2   # contains 5 mA
    assert src['limitv'] == 200.0                                      # a voltage compliance
    with pytest.raises(ValueError, match="voltage compliance"):
        se.generate_cycle_lua_script(a, None, 1)    # no max_voltage / voltage_limit_V


# --- faults --------------------------------------------------------------

def test_a_cycle_fault_raises_and_leaves_the_outputs_off(monkeypatch):
    sim = tsp_sim.SimulatedKeithley()
    seen = []
    orig = se._outputs_off

    def spy(inst):
        seen.append((inst.state['smua.source']['output'], inst.state['smua.source']['levelv']))
        return orig(inst)
    monkeypatch.setattr(se, "_outputs_off", spy)

    with pytest.raises(se.LuaExecutionError) as ei:
        _run(sim, _train(), _train(stim_level=-1.0),
             script_edit=lambda s: s.replace("local t_next", "smua.bogus = 1\nlocal t_next", 1))
    assert seen[0][0] == 'OUTPUT_ON'            # the fault really left them on
    assert sim.state['smua.source']['output'] == 'OUTPUT_OFF'
    assert sim.state['smub.source']['output'] == 'OUTPUT_OFF'
    assert 'smua' in ei.value.buffer_counts
    assert any('bogus' in e for e in ei.value.instrument_errors)
    assert 'NOT complete' in str(ei.value)


def test_cycle_sequence_never_reports_ok_after_a_fault(monkeypatch, capsys):
    sim = tsp_sim.SimulatedKeithley()
    real = se.generate_cycle_lua_script

    def faulty(*a, **k):
        script, bmap = real(*a, **k)
        return script.replace("local t_next", "smua.bogus = 1\nlocal t_next", 1), bmap
    monkeypatch.setattr(se, "generate_cycle_lua_script", faulty)
    with pytest.raises(se.LuaExecutionError):
        sc.cycle_sequence(sim, _train(), None, 1, 100, 100)
    assert "[OK]" not in capsys.readouterr().out


def test_retrieve_buffer_refuses_to_guess_when_the_count_query_fails():
    class Dead:
        def query(self, cmd):
            raise TimeoutError("no reply")
    with pytest.raises(se.LuaExecutionError, match="buffer count"):
        se._retrieve_buffer(Dead(), "smua", 10)


def test_adc_state_is_restored_after_a_cycle_run():
    sim = tsp_sim.SimulatedKeithley()
    _run(sim, _train(), None, n_cycles=1)
    assert sim.state['smua.measure']['delay'] == 'DELAY_AUTO'
    assert sim.state['smua.measure']['autozero'] == 'AUTOZERO_AUTO'


# --- STDP ------------------------------------------------------------------

def _stdp_params(**kw):
    p = dict(mode='stdp', stim_drive_type='V', stim_level=1.0, post_spike_level=1.0,
             stim_width_ms=2, stim_period_ms=20, n_pulses=3, read_voltage=0.1,
             read_delay_ms=1, compliance_A=1e-3, nplc=0.1, settle_ms=5, measure_avg=1)
    p.update(kw)
    return p


def test_stdp_current_drive_uses_leveli_and_a_current_range():
    sim = tsp_sim.SimulatedKeithley()
    res = se.measure_stdp(sim, 'smua', 'smub',
                          _stdp_params(stim_drive_type='I', stim_level=1e-3), [5, -5])
    assert res['execution_path'] == ['lua', 'lua'] and len(res['timing']) == 2
    src = sim.state['smua.source']
    assert src['rangei'] == 1e-3 and src['limitv'] == 200.0
    assert any(k == 'leveli' and v == 1e-3 for (_, n, k, v) in sim.log if n == 'smua.source')
    assert not any(k == 'levelv' and v == 1e-3 for (_, n, k, v) in sim.log if n == 'smua.source')
    with pytest.raises(ValueError, match="post_spike_level"):
        p = _stdp_params(stim_drive_type='I', stim_level=1e-3)
        del p['post_spike_level']
        se.measure_stdp(tsp_sim.SimulatedKeithley(), 'smua', 'smub', p, [5])


def test_stdp_restores_adc_state_and_sets_the_filter():
    sim = tsp_sim.SimulatedKeithley()
    sim.state['smub.measure.filter']['enable'] = 'FILTER_ON'     # dirty, as left by a prior run
    sim.state['smub.measure.filter']['count'] = 25
    se.measure_stdp(sim, 'smua', 'smub', _stdp_params(), [5])
    assert sim.state['smub.measure.filter']['enable'] == 'FILTER_OFF'
    assert sim.state['smub.measure']['delay'] == 'DELAY_AUTO'


def test_stdp_stale_globals_can_never_leak(monkeypatch):
    sim = tsp_sim.SimulatedKeithley()
    real = se.generate_stdp_lua_script
    calls = []

    def maybe_faulty(*a, **k):
        script = real(*a, **k)
        calls.append(1)
        if len(calls) == 2:     # second point dies right after the globals are cleared
            script = script.replace("_stdp_i_final = nil", "_stdp_i_final = nil\nsmua.bogus = 1", 1)
        return script
    monkeypatch.setattr(se, "generate_stdp_lua_script", maybe_faulty)
    with pytest.raises(se.LuaExecutionError) as ei:
        se.measure_stdp(sim, 'smua', 'smub', _stdp_params(), [5, 10, -5])
    partial = ei.value.partial_results
    assert partial['delta_t_ms'] == [5]                         # point 1 only; 2 NOT recorded
    assert sim.state['smua.source']['output'] == 'OUTPUT_OFF'
    assert sim.state['smub.source']['output'] == 'OUTPUT_OFF'
    assert any('bogus' in e for e in ei.value.instrument_errors)
    # The globals were cleared by the failed script, so a read can only fail.
    with pytest.raises(se.LuaExecutionError, match="did not produce"):
        se._read_stdp_results(sim)


@pytest.mark.parametrize("params, dts", [
    (dict(stim_width_ms=10, stim_period_ms=30), [20, 50]),     # 50 + 10 > 30
    (dict(), [5, 0]),                                           # delta_t = 0
    (dict(stim_width_ms=10), [20, 5]),                          # |dt| < width
    (dict(n_pulses=0), [5]),
    (dict(), []),
])
def test_stdp_validation_fires_before_any_instrument_activity(params, dts):
    sim = tsp_sim.SimulatedKeithley()
    with pytest.raises(ValueError, match="not producible"):
        se.measure_stdp(sim, 'smua', 'smub', _stdp_params(**params), dts)
    assert sim.writes == [] and sim.queries == []


def test_stdp_validation_accepts_a_valid_sweep():
    se._validate_stdp_sweep(_stdp_params(stim_width_ms=10, stim_period_ms=80), [-50, -20, 20, 50, 70])


# --- SRDP ------------------------------------------------------------------

def _srdp_params(**kw):
    p = dict(mode='srdp', stim_drive_type='V', stim_level=1.0, stim_width_ms=2,
             n_pulses=5, read_voltage=0.1, read_delay_ms=1, compliance_A=1e-3, nplc=0.1)
    p.update(kw)
    return p


@pytest.mark.parametrize("params, freqs, match", [
    (dict(n_pulses=-4), [1, 2], "n_pulses"),
    (dict(n_pulses=0), [1, 2], "n_pulses"),
    (dict(), [100, 10], "ascending"),
    (dict(), [], "empty"),
    (dict(), [0, 5], "> 0"),
])
def test_srdp_validation_fires_before_any_instrument_activity(params, freqs, match):
    sim = tsp_sim.SimulatedKeithley()
    with pytest.raises(ValueError, match=match):
        se.measure_srdp(sim, 'smua', 'smub', _srdp_params(**params), freqs)
    assert sim.writes == [] and sim.queries == []


def test_srdp_deduplicates_frequencies_beyond_the_ceiling(monkeypatch):
    ceiling = 60.0
    measured = []

    def stub(instrument, stim_ch, read_ch, params):
        f = min(1000.0 / params['stim_period_ms'], ceiling)
        measured.append(f)
        n = params['n_pulses']
        return {'conductance_S': [1e-6 + 1e-7 * i for i in range(n + 1)],
                'time_s': [0.0] + [0.05 + k / f for k in range(n)],
                'execution_path': 'lua', 'timing': {'overruns': 0.0}}
    monkeypatch.setattr(se, "pulse_read_sequence", stub)
    sim = tsp_sim.SimulatedKeithley()
    res = se.measure_srdp(sim, 'smua', 'smub', _srdp_params(), [10, 50, 100, 200, 400, 400])
    assert len(measured) == 3                       # 10, 50, 100(->60); the rest are the same point
    assert res['requested_frequencies_hz'] == [10, 50, 100]
    assert res['frequencies_hz'][2] == pytest.approx(60.0, rel=1e-6)
    assert [s['requested_hz'] for s in res['skipped_frequencies']] == [200, 400, 400]
    assert res['execution_path'] == ['lua'] * 3 and len(res['timing']) == 3
    assert sim.state['localnode']['linefreq'] == 50.0
