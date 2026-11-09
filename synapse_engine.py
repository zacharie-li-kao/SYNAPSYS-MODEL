"""
Synapse Engine Module for Keithley Sourcemeter
Implements pulse-read sequences for electrical, visual, and memristor synapse characterization.
Phase 1: Core engine and helper functions

NOTE: This module now includes fast LUA execution for synapse characterization.
When using 2600-series Keithley instruments in electrical mode, pulse sequences
execute directly on the instrument for 5-10x speed improvement. This is completely
transparent to the user and requires no API changes.
"""

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import pyvisa
import time
import numpy as np
from tkinter import messagebox

# Sentinel token emitted by every generated LUA script as its final statement
# (after all measurements and cleanup). The host blocks on instrument.read()
# until this token arrives — a TSP-native replacement for *OPC?, which is
# unreliable for anonymous loadandrunscript bodies on 2600-series instruments
# and causes 60-120 s stalls on repeated runs.
LUA_DONE_TOKEN = "SYN_DONE"

# Prefix for machine-readable timing statistics printed by a running script.
# The pulse loops run as fast as the hardware allows and never throttle to hit
# a requested period they cannot meet — but a period that was NOT achieved must
# be visible rather than inferred from the nominal value, so the loop counts its
# own overruns and reports them on a line of this form:
#     SYN_TIMING overruns=<n> worst_ms=<float> n_pulses=<n>
LUA_TIMING_TOKEN = "SYN_TIMING"

# Mains line frequency (Hz). Sets the ADC integration aperture (NPLC = N power
# line cycles) and the instrument's normal-mode rejection of 50/60 Hz mains
# pickup: integrating each reading over a whole number of line cycles makes the
# periodic interference average to zero. The 2600-series synchronises its ADC
# to this value, so it MUST match the actual mains — otherwise the aperture no
# longer spans an integer number of cycles and line-frequency noise leaks into
# every reading. Only 50 or 60 Hz are valid. Default 50 Hz (European mains, UPC
# Barcelona); callers may override per measurement via params['line_freq_hz'].
DEFAULT_LINE_FREQ_HZ = 50.0


def _line_freq_hz(params):
    """Resolve the mains line frequency (Hz) from a params/config dict.

    Falls back to DEFAULT_LINE_FREQ_HZ when the caller does not specify one
    (an explicit None counts as unspecified — the GUI's Auto setting) — never
    silently assumes 60 Hz. Hardware measurements resolve Auto through
    resolve_line_freq(), which asks the instrument; this fallback is only ever
    reached by simulation paths, where no mains exists to reject. Raises
    loudly on a value the 2600-series cannot synchronise to, rather than
    degrading mains rejection silently.
    """
    raw = (params.get('line_freq_hz') if isinstance(params, dict) else None)
    if raw is None:
        raw = DEFAULT_LINE_FREQ_HZ
    try:
        freq = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"line_freq_hz must be numeric 50 or 60, got {raw!r}")
    if freq not in (50.0, 60.0):
        raise ValueError(
            f"line_freq_hz must be 50 or 60 Hz (2600-series line sync), got {freq}"
        )
    return freq


def resolve_line_freq(instrument, params):
    """Resolve the mains frequency for a hardware measurement and stamp it.

    An explicit ``params['line_freq_hz']`` always wins. When the caller does
    not specify one, the INSTRUMENT is asked: the 2600 series auto-detects its
    mains frequency at power-up and reports it via ``localnode.linefreq``, so
    the detected value is correct on any grid (50 or 60 Hz) without any
    configuration. This replaced a hard default of 50 Hz that silently
    synchronised the ADC to the wrong mains on 60 Hz installations (field
    report 2026-08-18), where no NPLC setting could then reject line pickup.

    The resolved value is written back into ``params`` so every consumer —
    LUA generator, averaging derivation, drift analysis — uses one value.
    Anything other than 50/60 raises rather than degrading rejection silently.
    """
    if params.get('line_freq_hz') is not None:
        freq = _line_freq_hz(params)
    else:
        raw = instrument.query("print(localnode.linefreq)").strip()
        try:
            freq = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Instrument reported an unreadable mains frequency: {raw!r}. "
                "Set params['line_freq_hz'] to 50 or 60 explicitly."
            )
        if freq not in (50.0, 60.0):
            raise ValueError(
                f"Instrument reported a mains frequency of {freq} Hz; the "
                "2600 series can only synchronise to 50 or 60 Hz. Set "
                "params['line_freq_hz'] explicitly if the detection is wrong."
            )
    params['line_freq_hz'] = freq
    return freq


def _apply_wire_modes(instrument, params, channels):
    """
    Apply the per-SMU sense mode to each channel in `channels`.

    Reads `params['wire_modes']` (a dict keyed by 'smua'/'smub' with values
    '2-Wire' or '4-Wire'). Falls back to the legacy `params['wire_mode']` when
    no per-channel dict is provided, in which case the same mode is applied
    to every channel.

    Skips silently if neither key is present, leaving whatever sense mode the
    instrument already holds.
    """
    wire_modes = params.get('wire_modes')
    legacy = params.get('wire_mode')
    if wire_modes is None and legacy is None:
        return
    for ch in channels:
        if not ch:
            continue
        key = str(ch).lower()
        mode = None
        if isinstance(wire_modes, dict):
            mode = wire_modes.get(key) or wire_modes.get(str(ch))
        if mode is None:
            mode = legacy
        if mode is None:
            continue
        const = 'SENSE_REMOTE' if mode == '4-Wire' else 'SENSE_LOCAL'
        instrument.write(f"{ch}.sense = {ch}.{const}")


def _wait_lua_complete(instrument, expected_time_s=None):
    """
    Blocks until the running TSP script emits LUA_DONE_TOKEN.

    Generated LUA scripts call `print("<token>")` as their final statement,
    so this read() unblocks within milliseconds of true completion.

    Returns a dict of timing statistics reported by the script itself via
    LUA_TIMING_TOKEN lines (empty if the script emits none). The instrument is
    the only witness to whether a pulse loop actually kept its requested
    period, so it reports that directly rather than leaving the host to infer
    it. See the overrun accounting in the generated pulse loops.

    Any other lines printed by the script (unexpected but harmless) are
    forwarded to the PC console for visibility.

    Raises pyvisa.VisaIOError on timeout — caller is responsible for any
    sleep-based fallback or error propagation.
    """
    max_wait = max(60, (expected_time_s or 60) * 2.5)
    old_timeout = instrument.timeout
    instrument.timeout = int(max_wait * 1000)
    stats = {}
    try:
        while True:
            line = instrument.read().strip()
            if not line:
                continue
            if LUA_TIMING_TOKEN in line:
                # "SYN_TIMING overruns=3 worst_ms=1.2340 n_pulses=50"
                for field in line.split()[1:]:
                    if '=' not in field:
                        continue
                    key, _, raw = field.partition('=')
                    try:
                        stats[key] = float(raw)
                    except ValueError:
                        stats[key] = raw
                continue
            if LUA_DONE_TOKEN in line:
                return stats
            print(f"  [LUA stdout] {line}")
    finally:
        instrument.timeout = old_timeout


# =============================================================================
# LUA SCRIPT GENERATION FOR FAST SYNAPSE CHARACTERIZATION
# =============================================================================

# Hardware cap on `measure.filter.count` for the 2600 series. Writing beyond
# this is rejected with SCPI error -222 and leaves the filter at its previous
# value. This is the ONLY averaging ceiling in the codebase — the standard
# execution path used to impose a second, lower one (50), so the same GUI
# setting produced different averaging depending on which path ran.
FILTER_COUNT_MAX = 100
MIN_NPLC = 0.001  # Keithley 2600 series minimum


def _emit_measurement_state(channel, measure_avg):
    """LUA lines that set a channel's measurement state UNCONDITIONALLY.

    H8: the cycle generator wrote the filter state only when measure_avg > 1
    and never wrote `measure.count` at all. Instrument settings persist across
    runs, so an aborted run that had left `filter.enable = ON, count = 8`
    leaked that into the NEXT run — which then silently averaged 8 readings per
    point, blowing the timing budget the whole architecture is built around,
    with nothing in the data recording it.

    Both branches are emitted explicitly, so the channel's state is fully
    determined by this script and not by whatever ran before it.

    H12: the count is clamped to FILTER_COUNT_MAX (the hardware limit, 100),
    not to a local 50. The generator used 50 here while the timing warning
    shown to the user was computed from the unclamped value, so the user was
    warned about a constraint that did not exist and was not told that the
    averaging they asked for had been halved.
    """
    lines = [f"{channel}.measure.count = 1"]
    if measure_avg > 1:
        lines.append(f"{channel}.measure.filter.enable = {channel}.FILTER_ON")
        lines.append(f"{channel}.measure.filter.type = {channel}.FILTER_REPEAT_AVG")
        lines.append(f"{channel}.measure.filter.count = "
                     f"{max(2, min(measure_avg, FILTER_COUNT_MAX))}")
    else:
        lines.append(f"{channel}.measure.filter.enable = {channel}.FILTER_OFF")
    return lines


# Current-measurement settling time per fixed current range: "delays are
# enforced to account for approximately 6 time constants of a given range"
# (2600A Reference Manual 2600AS-901-01 Rev. D, Table 4-4 "Current Measure
# Settling Time"; the 2600B manual has no table and says only that the default
# measurement delay "varies by model"). These are the values the instrument's
# own DELAY_AUTO inserts before every measure.i() on the 2634B/2635B/2636B and
# 2635A/2636A, where measure.delay defaults to AUTO.
CURRENT_RANGE_SETTLING_S = (
    (100e-12, 0.360), (1e-9, 0.360), (10e-9, 0.090), (100e-9, 0.015),
    (1e-6, 2.5e-3), (10e-6, 5e-4), (100e-6, 1.5e-4), (1e-3, 1e-4),
    (10e-3, 8e-5), (100e-3, 8e-5), (1.0, 1.2e-4), (1.5, 1.2e-4), (3.0, 1.2e-4),
    (10.0, 1.2e-4),
)


def range_settling_s(range_A):
    """Settling time (s) of a fixed current-measurement range, from the table above."""
    for r, t in CURRENT_RANGE_SETTLING_S:
        if range_A <= r * (1 + 1e-9):
            return t
    raise ValueError(f"current range {range_A} A is beyond the 2600-series ranges")


def check_read_settle(read_settle_s, range_A):
    """Settling verdict for a read: None when settled, else a warning string.

    measure.delay is set OFF by every script (see _emit_read_adc_setup), so the
    script's own read_settle is the ONLY settling a read gets. Shorter than the
    range's settling time means the reading is taken on an unsettled range and
    is not accurate to specification; that is reported, never hidden, because
    the user may be deliberately trading accuracy for speed.
    """
    need = range_settling_s(range_A)
    if read_settle_s + 1e-12 < need:
        return (f"read settle {read_settle_s * 1e3:.3g} ms is shorter than the "
                f"{need * 1e3:.3g} ms the {range_A:g} A range needs to settle "
                "(~6 time constants); readings are not settled to specification")
    return None


def _emit_read_adc_setup(channel, nplc, measure_avg, measure_delay="off"):
    """LUA lines fixing a read channel's ADC state completely, for speed AND accuracy.

    - measure.delay = 0: the instrument default on the 2634B/2635B/2636B is
      DELAY_AUTO, which silently inserted the range settling time (2.5 ms on
      1 uA, 15 ms on 100 nA, 90 ms on 10 nA) before EVERY read, on top of the
      script's own read_settle — shifting every read later than requested and,
      on low ranges, halving the pulse rate. Settling is now the script's
      read_settle alone, checked against the range by check_read_settle().
    - nplc, then AUTOZERO_ONCE: OFF alone reuses the cached reference and zero
      "regardless of age", and accuracy "will drift with time and
      temperature"; the manual's remedy is ONCE just before the sequence. The
      cache is per aperture, so ONCE must follow the nplc write. It costs two
      conversions once and then behaves as OFF.
    - count and filter set unconditionally (_emit_measurement_state).
    - The channel's own measure.delay and autozero are saved first and put back
      by _emit_restore_adc at cleanup. Their defaults differ by model (delay is
      AUTO on 263x, OFF on 260x/261x), so no value can be hardcoded, and without
      the restore a J-V or transistor run later in the session inherited
      delay = 0. Saved only if not already saved: a second setup of the same
      channel, or a previous run that aborted before restoring, must not
      overwrite the original with this script's own value.

    measure_delay='auto' keeps DELAY_AUTO, for reads where the range may change
    (autorange) and timing does not matter — retention — because there the
    instrument's range settling is the only settling the read gets.
    """
    delay_value = {"off": "0", "auto": f"{channel}.DELAY_AUTO"}[measure_delay]
    lines = [f"if _syn_saved_delay_{channel} == nil then "
             f"_syn_saved_delay_{channel} = {channel}.measure.delay end",
             f"if _syn_saved_az_{channel} == nil then "
             f"_syn_saved_az_{channel} = {channel}.measure.autozero end",
             f"{channel}.measure.delay = {delay_value}",
             f"{channel}.measure.nplc = {nplc}"]
    lines.extend(_emit_measurement_state(channel, measure_avg))
    lines.append(f"{channel}.measure.autozero = {channel}.AUTOZERO_ONCE")
    return lines


def _emit_restore_adc(channel):
    """LUA lines putting back the measure.delay and autozero saved by
    _emit_read_adc_setup, so nothing this suite sets outlives its run."""
    return [f"if _syn_saved_delay_{channel} ~= nil then "
            f"{channel}.measure.delay = _syn_saved_delay_{channel} "
            f"_syn_saved_delay_{channel} = nil end",
            f"if _syn_saved_az_{channel} ~= nil then "
            f"{channel}.measure.autozero = _syn_saved_az_{channel} "
            f"_syn_saved_az_{channel} = nil end"]


def lua_delay(seconds, label="delay"):
    """A LUA delay() statement, or '' for zero. The 2600 rejects delay(0)
    ("You cannot set a delay for zero seconds", RM 7-47), which aborted a whole
    script whenever a user asked for a 0 ms read delay."""
    s = float(seconds)
    if not np.isfinite(s) or s < 0:
        raise ValueError(f"{label} must be a finite non-negative time, got {seconds!r} s")
    return f"delay({s!r})" if s > 0 else ""


def _drain_output(instrument, max_lines=1000):
    """Discard anything left in the instrument's output queue before a run.

    A completion token left over from an earlier run otherwise ends the next
    _wait_lua_complete immediately and drops that run's timing report.
    """
    old = instrument.timeout
    instrument.timeout = 100
    drained = []
    try:
        for _ in range(max_lines):
            try:
                line = instrument.read()
            except Exception:
                break
            drained.append(line)
    finally:
        instrument.timeout = old
    if drained:
        print(f"  [LUA] discarded {len(drained)} stale output line(s) before upload: "
              f"{[l.strip() for l in drained[:3]]}")
    return drained


# Significant digits for printbuffer(). The default is 6, which quantised
# timestamps to 1 ms beyond 100 s and 10 ms beyond 1000 s, so the drift
# analysis of a long run raised alarms on spacing that was exact.
ASCII_PRECISION = 15


def _set_ascii_precision(instrument):
    instrument.write(f"format.asciiprecision = {ASCII_PRECISION}")


def derive_measurement_averaging(params):
    """Resolve (nplc, measure_avg) for a pulse-read measurement.

    Single source of truth, shared by the LUA generator and the standard
    execution path so that both measure the same way. `read_width_ms` defines
    the total measurement window; one ADC integration lasts nplc / line_freq
    (nplc * 20 ms at 50 Hz, nplc * 16.67 ms at 60 Hz), and NPLC must fit inside
    the window.

    Returns:
        (nplc, measure_avg) — the integration time per reading and the number of
        hardware-averaged readings per pulse.
    """
    nplc = params.get('nplc', 1.0)
    read_width_ms = params.get('read_width_ms', None)
    NPLC_PERIOD_MS = 1000.0 / _line_freq_hz(params)

    user_override = params.get('measure_avg_user')
    if user_override is not None and int(user_override) > 0:
        # User explicitly specified a sample count. Respect it, clamped.
        measure_avg = max(1, min(int(user_override), FILTER_COUNT_MAX))
        if read_width_ms is not None and read_width_ms > 0:
            max_nplc_for_width = read_width_ms / NPLC_PERIOD_MS
            if nplc > max_nplc_for_width:
                nplc = max(MIN_NPLC, max_nplc_for_width)
    elif read_width_ms is not None and read_width_ms > 0:
        max_nplc_for_width = read_width_ms / NPLC_PERIOD_MS
        if nplc > max_nplc_for_width:
            nplc = max(MIN_NPLC, max_nplc_for_width)
        # Fit as many integrations as possible within the read width for averaging.
        # FP-tolerant floor: nudge by 1e-6 so exact integer ratios (e.g.
        # 10 ms / NPLC=0.01) don't lose a cycle to 1000/line_freq imprecision.
        measure_avg = max(1, int(read_width_ms / (nplc * NPLC_PERIOD_MS) + 1e-6))
        if measure_avg > FILTER_COUNT_MAX:
            print(f"ℹ️  Hardware averaging clamped from {measure_avg} to "
                  f"{FILTER_COUNT_MAX} (2600-series filter.count limit)")
            measure_avg = FILTER_COUNT_MAX
    else:
        measure_avg = max(1, min(int(params.get('measure_avg', 1)), FILTER_COUNT_MAX))

    return nplc, measure_avg


def get_fixed_current_range(compliance_A):
    """Smallest native 2600-series current range that contains the compliance.

    Shared by both execution paths. Fixed ranging (rather than autorange) keeps
    the per-pulse measurement time deterministic, which is what makes the pulse
    period reproducible.
    """
    ranges = [100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6,
              1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
    for r in ranges:
        if compliance_A <= r:
            return r
    return ranges[-1]


# Lateness (s) under which an absolute-timer edge is not counted as an overrun:
# TSP command latency, not a missed deadline.
EDGE_LATE_TOL_S = 1e-4

# Per-cycle TSP command overhead in the LUA pulse-read path (not physical relay
# settling — TSP writes return immediately). Measured on a 2636B: ~2 ms of
# cumulative command latency per pulse (level writes + measure.i call +
# timer.measure.t check + buffer append).
LUA_OVERHEAD_MS = 2.0


class LuaScriptStartedError(RuntimeError):
    """A failure after `endscript` was sent: the script had already started.

    The device may be partly stimulated, so the measurement must NOT be silently
    re-run on another path — that re-run would start from a modified device and
    report its altered state as the pre-stimulus baseline.
    """


class LuaScript(str):
    """Generated TSP source that also carries the feasibility facts computed
    while generating it (`.facts`), so callers need not recompute them."""

    def __new__(cls, text, **facts):
        script = super().__new__(cls, text)
        script.facts = facts
        return script


def _check_channels(stim_ch, read_ch):
    if str(stim_ch).lower() == str(read_ch).lower():
        raise ValueError(
            f"stim_ch and read_ch are both {stim_ch!r}: one SMU cannot be the "
            "stimulus source and the read source at once (the stimulus level "
            "overwrites the read bias and every current after pulse 1 is "
            "meaningless). Use two different SMU channels."
        )


def _check_times_ms(nonneg=None, positive=None):
    """Raise a clear ValueError for a negative (or non-finite) user time."""
    for label, v in (nonneg or {}).items():
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{label} must be a finite time >= 0 ms, got {v!r}")
    for label, v in (positive or {}).items():
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{label} must be a finite time > 0 ms, got {v!r}")


def _pulse_read_plan(stim_ch, read_ch, params):
    """Validated, resolved timing facts for one pulse-read sequence.

    Single source for the LUA generator and the standard path, so both resolve
    the same defaults, the same fixed range and the same feasibility verdicts.
    """
    _check_channels(stim_ch, read_ch)
    stim_width_ms = params['stim_width_ms']
    stim_period_ms = params['stim_period_ms']
    read_delay_ms = params.get('read_delay_ms', 10)
    read_settle_delay_ms = params.get('read_settle_delay_ms', 5)
    settle_ms = params.get('settle_ms', 5)
    _check_times_ms(
        nonneg={'read_delay_ms': read_delay_ms,
                'read_settle_delay_ms': read_settle_delay_ms,
                'settle_ms': settle_ms},
        positive={'stim_width_ms': stim_width_ms, 'stim_period_ms': stim_period_ms})

    nplc, measure_avg = derive_measurement_averaging(params)
    measurement_time_ms = nplc * (1000.0 / _line_freq_hz(params)) * measure_avg
    fixed_i_range = get_fixed_current_range(params['compliance_A'])
    return {
        'stim_width_ms': stim_width_ms, 'stim_period_ms': stim_period_ms,
        'read_delay_ms': read_delay_ms, 'read_settle_delay_ms': read_settle_delay_ms,
        'settle_ms': settle_ms, 'nplc': nplc, 'measure_avg': measure_avg,
        'measurement_time_ms': measurement_time_ms, 'fixed_i_range': fixed_i_range,
        'min_period_ms': (stim_width_ms + read_delay_ms + read_settle_delay_ms
                          + measurement_time_ms + LUA_OVERHEAD_MS),
        'settle_warning': check_read_settle(read_settle_delay_ms / 1000.0, fixed_i_range),
    }


def generate_pulse_read_lua_script(stim_ch, read_ch, params):
    """
    Generates LUA script for fast pulse-read sequences on Keithley.
    Optimized for 2636A compatibility.

    Returns a LuaScript (a str) whose `.facts` hold `min_period_ms` (the shortest
    period this sequence can actually deliver) and `settle_warning`.
    """

    # Extract parameters
    stim_drive_type = params['stim_drive_type']
    stim_level = params['stim_level']
    n_pulses = params['n_pulses']
    read_voltage = params['read_voltage']
    compliance_A = params['compliance_A']
    max_voltage = params.get('max_voltage', 10.0)

    line_freq_hz = _line_freq_hz(params)
    plan = _pulse_read_plan(stim_ch, read_ch, params)
    stim_width_ms = plan['stim_width_ms']
    stim_period_ms = plan['stim_period_ms']
    read_delay_ms = plan['read_delay_ms']
    read_settle_delay_ms = plan['read_settle_delay_ms']
    nplc, measure_avg = plan['nplc'], plan['measure_avg']
    measurement_time_ms = plan['measurement_time_ms']
    min_required_period = plan['min_period_ms']
    fixed_i_range = plan['fixed_i_range']

    # Validation checks
    KEITHLEY_MAX_BUFFER_SIZE = 50000
    if n_pulses + 1 > KEITHLEY_MAX_BUFFER_SIZE:
        raise ValueError(f"Pulse count ({n_pulses}) exceeds buffer capacity.")

    if stim_period_ms < min_required_period:
        print(f"⚠️  Requested period ({stim_period_ms:.2f} ms) shorter than minimum for zero-drift "
              f"({min_required_period:.2f} ms = stim {stim_width_ms:g} + read_delay {read_delay_ms:g} + "
              f"settle {read_settle_delay_ms:g} + meas {measurement_time_ms:.2f} + "
              f"LUA overhead {LUA_OVERHEAD_MS:g}). Instrument will run as fast as possible.")
    if plan['settle_warning']:
        print(f"⚠️  {plan['settle_warning']}")

    # Convert times to seconds for LUA
    stim_width_s = stim_width_ms / 1000.0
    stim_period_s = stim_period_ms / 1000.0
    read_delay_s = read_delay_ms / 1000.0
    read_settle_delay_s = read_settle_delay_ms / 1000.0
    settle_s = plan['settle_ms'] / 1000.0

    stim_level_attr = 'levelv' if stim_drive_type == 'V' else 'leveli'
    adc_setup = "\n".join(_emit_read_adc_setup(read_ch, nplc, measure_avg))

    # Generate LUA script
    lua_script = f"""
-- Fast Synapse Pulse-Read Sequence (2636A Compatible)
-- Read voltage is level-switched per pulse (0V idle, read_voltage during measure)
-- to match standard execution path behavior.
-- The stimulus output is switched ON once, at level 0, and only its LEVEL changes
-- afterwards (as in the cycle and STDP scripts). With the default offmode NORMAL,
-- OUTPUT_OFF does not open the relay: it holds 0 V at a 1 mA off-limit (RM 2-77),
-- so toggling it every pulse disconnected nothing and only added latency.

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local n_pulses = {n_pulses}
local stim_level = {stim_level}
local stim_period = {stim_period_s}
local read_voltage = {read_voltage}

-- Buffer Configuration
-- Buffer capacity is READ-ONLY on the 2600 series; it cannot be
-- enlarged from a script. Overflow behaviour is what we CAN set.
{read_ch}.nvbuffer1.fillmode = {read_ch}.FILL_ONCE

{read_ch}.nvbuffer1.clear()
{read_ch}.nvbuffer1.appendmode = 1
{read_ch}.nvbuffer1.collecttimestamps = 1
{read_ch}.nvbuffer1.collectsourcevalues = 0

-- Configure read channel (idle at 0V — NOT at read_voltage)
{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS
{read_ch}.source.rangev = {_get_fixed_voltage_range(read_voltage, max_voltage)}
{read_ch}.source.levelv = 0
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF

-- ADC state, fully determined here. Averaging is the 2600 repeat-average filter:
-- filter.count = N integrations averaged per measure.i() call, measure.count = 1,
-- so one (averaged) value is appended per pulse — the 1-pulse-1-entry contract
-- the host expects. measure.delay = 0 removes the hidden DELAY_AUTO settling.
{adc_setup}

{read_ch}.source.output = {read_ch}.OUTPUT_ON
{lua_delay(settle_s, 'settle_ms')}

-- Configure stimulus channel"""

    if stim_drive_type == 'V':
        fixed_v_range = _get_fixed_voltage_range(stim_level, max_voltage)
        lua_script += f"""
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}"""
    else:
        # The source range must contain the SOURCED current, not the compliance
        # (which limits the measured/sensed side).
        lua_script += f"""
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS
{stim_ch}.source.leveli = 0
{stim_ch}.source.rangei = {get_fixed_current_range(abs(stim_level))}
{stim_ch}.source.limitv = {max_voltage}"""

    lua_script += f"""
-- Stim output ON once, at level 0; pulses are level changes only.
{stim_ch}.source.output = {stim_ch}.OUTPUT_ON

-- Reset timer before the initial read so every buffer timestamp shares a
-- single reference. Otherwise t[0] would sit on whatever the timer held
-- before this script, which skews the first reported interval by tens of
-- milliseconds and contaminates the drift summary.
timer.reset()

-- Initial read (level-switch read voltage: 0V → read_voltage → measure → 0V)
{read_ch}.source.levelv = read_voltage
{lua_delay(read_settle_delay_s, 'read_settle_delay_ms')}
{read_ch}.measure.i({read_ch}.nvbuffer1)
{read_ch}.source.levelv = 0

-- Pulse-read loop (timer-based to prevent cumulative drift).
-- t_next tracks the *absolute* deadline for each pulse's end, measured
-- against the timer that started just before the initial read.
local t_next = timer.measure.t()
local overruns = 0
local worst_overrun = 0
for pulse = 1, n_pulses do
    t_next = t_next + stim_period
    -- Stimulus ON: raise the level, hold for the pulse width, return to 0
    {stim_ch}.source.{stim_level_attr} = stim_level
    {lua_delay(stim_width_s, 'stim_width_ms')}
    {stim_ch}.source.{stim_level_attr} = 0

    -- Read: apply read voltage, settle, measure, return to 0V
    {lua_delay(read_delay_s, 'read_delay_ms')}
    {read_ch}.source.levelv = read_voltage
    {lua_delay(read_settle_delay_s, 'read_settle_delay_ms')}
    {read_ch}.measure.i({read_ch}.nvbuffer1)
    {read_ch}.source.levelv = 0

    -- Wait remainder of period using absolute timer (prevents cumulative drift).
    --
    -- If the pulse's own work (stim + read delay + settle + ADC integration)
    -- took LONGER than the requested period, there is no remainder: the train
    -- runs at the hardware's limit, which is the intended behaviour. What must
    -- not happen is that the shortfall goes unrecorded, so it is counted here
    -- and reported to the host. t_next is re-baselined to NOW so that one slow
    -- pulse does not mark every subsequent pulse as an overrun and does not
    -- make the loop fire the following pulses early trying to catch up.
    local t_remain = t_next - timer.measure.t()
    if t_remain > 0 then
        delay(t_remain)
    else
        overruns = overruns + 1
        if -t_remain > worst_overrun then worst_overrun = -t_remain end
        t_next = timer.measure.t()
    end
end

-- Cleanup: levels to 0 BEFORE the outputs go off, filter off on the read channel
-- (a left-over ON/count state leaks into the next run's reads).
{stim_ch}.source.{stim_level_attr} = 0
{read_ch}.source.levelv = 0
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF
{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF
{chr(10).join(_emit_restore_adc(read_ch))}

-- Timing report: how often the requested period could NOT be met, and by how
-- much at worst. Printed before the completion token so the host records what
-- the instrument actually delivered rather than what was asked for.
print(string.format("{LUA_TIMING_TOKEN} overruns=%d worst_ms=%.4f n_pulses=%d",
                    overruns, worst_overrun * 1000, n_pulses))

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return LuaScript(lua_script, min_period_ms=min_required_period,
                     settle_warning=plan['settle_warning'],
                     fixed_i_range=fixed_i_range)


# =============================================================================
# LUA SCRIPT FOR VISUAL (SELF-POWERED) SYNAPSE CHARACTERIZATION
# =============================================================================

def generate_visual_synapse_lua_script(stim_ch, read_ch, params):
    """
    Generates LUA script for visual (self-powered) synapse characterization.

    In this mode:
    - Stim channel sends voltage pulses to LED/light source
    - Read channel measures photocurrent (Jsc) at 0V during illumination
    - Measurement occurs DURING the light pulse with configurable timing window

    Args:
        stim_ch (str): Stimulus channel for LED ('smua' or 'smub')
        read_ch (str): Read channel for Jsc measurement ('smua' or 'smub')
        params (dict): Measurement parameters containing:
            - light_pulse_voltage (float): Voltage to LED controller (V)
            - pulse_width_ms (float): Light pulse duration (ms)
            - pulse_period_ms (float): Time between pulse starts (ms)
            - n_pulses (int): Number of light pulses
            - measure_start_delay_ms (float): Delay after pulse start before measuring (ms)
            - measure_end_margin_ms (float): Stop measuring before pulse ends (ms)
            - readings_per_pulse (int): Number of readings to average per pulse
            - compliance_A (float): Current compliance (A)
            - nplc (float): Integration time in power line cycles

    Returns:
        LuaScript: LUA script for visual synapse measurement; `.facts` holds
        `settle_warning`.
    """

    _check_channels(stim_ch, read_ch)

    # Extract parameters
    light_voltage = params['light_pulse_voltage']
    pulse_width_ms = params['pulse_width_ms']
    pulse_period_ms = params['pulse_period_ms']
    n_pulses = params['n_pulses']
    measure_start_delay_ms = params.get('measure_start_delay_ms', 1.0)
    measure_end_margin_ms = params.get('measure_end_margin_ms', 1.0)
    readings_per_pulse = params.get('readings_per_pulse', 1)
    compliance_A = params['compliance_A']
    nplc = params.get('nplc', 1.0)
    line_freq_hz = _line_freq_hz(params)

    _check_times_ms(
        nonneg={'measure_start_delay_ms': measure_start_delay_ms,
                'measure_end_margin_ms': measure_end_margin_ms},
        positive={'pulse_width_ms': pulse_width_ms, 'pulse_period_ms': pulse_period_ms})

    # Validation
    measurement_window_ms = pulse_width_ms - measure_start_delay_ms - measure_end_margin_ms
    if measurement_window_ms <= 0:
        raise ValueError(
            f"Measurement window is invalid.\n"
            f"Pulse width ({pulse_width_ms}ms) must be > start delay ({measure_start_delay_ms}ms) + end margin ({measure_end_margin_ms}ms)"
        )

    # The ADC integration must complete strictly inside the light pulse, so the
    # optical dose equals intensity * pulse_width (the LED is held on for exactly
    # pulse_width in the loop below, regardless of how long the measurement
    # takes). One integration lasts nplc / line_freq; the repeat-average filter
    # runs `readings_per_pulse` of them back-to-back. Clamp that count down to
    # fit the window — keeping nplc (and thus mains rejection) untouched — and to
    # the 2600-series filter.count hardware ceiling. Fail loudly when not even a
    # single integration fits: that is a genuine configuration error, not
    # something to silently paper over.
    single_integration_ms = nplc * (1000.0 / line_freq_hz)
    if single_integration_ms > measurement_window_ms:
        raise ValueError(
            f"One ADC integration ({single_integration_ms:.2f} ms at NPLC={nplc}, "
            f"{int(line_freq_hz)} Hz) does not fit the measurement window "
            f"({measurement_window_ms:.2f} ms = pulse_width {pulse_width_ms:g} ms "
            f"- start delay {measure_start_delay_ms:g} ms - end margin "
            f"{measure_end_margin_ms:g} ms).\n"
            f"Increase pulse width, lower NPLC, or shorten the start delay / end margin."
        )
    max_readings_for_window = int(measurement_window_ms / single_integration_ms + 1e-6)
    readings_eff = max(1, min(readings_per_pulse, max_readings_for_window, FILTER_COUNT_MAX))
    if readings_eff != readings_per_pulse:
        print(f"ℹ️  Readings per pulse reduced from {readings_per_pulse} to {readings_eff} so "
              f"the measurement ({readings_eff} x {single_integration_ms:.2f} ms) finishes "
              f"within the {measurement_window_ms:.2f} ms window before the {pulse_width_ms:g} "
              f"ms light pulse ends.")
    readings_per_pulse = readings_eff

    # Check buffer capacity
    KEITHLEY_MAX_BUFFER_SIZE = 50000
    if n_pulses > KEITHLEY_MAX_BUFFER_SIZE:
        raise ValueError(f"Pulse count ({n_pulses}) exceeds buffer capacity ({KEITHLEY_MAX_BUFFER_SIZE}).")

    # Convert times to seconds for LUA
    pulse_width_s = pulse_width_ms / 1000.0
    pulse_period_s = pulse_period_ms / 1000.0
    measure_start_delay_s = measure_start_delay_ms / 1000.0

    # The pulse must contain its own light-on time.
    if pulse_period_s < pulse_width_s:
        raise ValueError(f"Pulse period ({pulse_period_ms}ms) must be >= pulse width ({pulse_width_ms}ms)")

    fixed_i_range = get_fixed_current_range(compliance_A)

    # Model-aware native voltage ranges (the 40 V and 200 V families differ).
    max_voltage = params.get('max_voltage', 200.0)
    fixed_v_range = _get_fixed_voltage_range(light_voltage, max_voltage)
    read_v_range = _get_fixed_voltage_range(0.0, max_voltage)

    # The photocurrent is read measure_start_delay after the LED turns on; with
    # measure.delay = 0 that interval is the range's only settling time.
    settle_warning = check_read_settle(measure_start_delay_s, fixed_i_range)
    if settle_warning:
        print(f"⚠️  {settle_warning}")
    adc_setup = "\n".join(_emit_read_adc_setup(read_ch, nplc, readings_per_pulse))

    # Generate LUA script
    lua_script = f"""
-- Visual (Self-Powered) Synapse Characterization
-- Measures photocurrent (Jsc) at 0V during light pulses

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local n_pulses = {n_pulses}
local light_voltage = {light_voltage}
local pulse_width = {pulse_width_s}
local measure_start_delay = {measure_start_delay_s}
local late_tol = {EDGE_LATE_TOL_S}

-- Buffer Configuration
-- Buffer capacity is READ-ONLY on the 2600 series; it cannot be
-- enlarged from a script. Overflow behaviour is what we CAN set.
{read_ch}.nvbuffer1.fillmode = {read_ch}.FILL_ONCE

{read_ch}.nvbuffer1.clear()
{read_ch}.nvbuffer1.appendmode = 1
{read_ch}.nvbuffer1.collecttimestamps = 1
{read_ch}.nvbuffer1.collectsourcevalues = 0

-- Configure read channel (Jsc measurement at 0V)
{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS
{read_ch}.source.levelv = 0
{read_ch}.source.rangev = {read_v_range}
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{adc_setup}

-- Configure stimulus channel (LED driver)
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}

-- Both outputs ON once, at 0 V. The LED is pulsed by LEVEL changes only: with
-- the default offmode NORMAL, OUTPUT_OFF holds 0 V at a 1 mA off-limit rather
-- than opening the relay (RM 2-77), so toggling it per pulse disconnected
-- nothing and only added latency.
{read_ch}.source.output = {read_ch}.OUTPUT_ON
{stim_ch}.source.output = {stim_ch}.OUTPUT_ON
delay(0.01)

-- Pulse-Measure loop (absolute-timer based to prevent cumulative drift).
-- The LED is held ON for exactly pulse_width every period, so the delivered
-- optical dose (intensity x pulse_width) is set by pulse_width alone and does
-- NOT depend on the ADC integration time. The photocurrent is measured
-- measure_start_delay into the pulse; host-side clamping guarantees the
-- integration finishes inside the pulse (>= measure_end_margin before LED off).
local pulse_period = {pulse_period_s}
local overruns = 0
local worst_overrun = 0
timer.reset()
local t_pulse_start = 0
for pulse = 1, n_pulses do
    -- Wait out the previous period's tail so this pulse starts on its absolute
    -- grid point (t_pulse_start), independent of accumulated measurement time.
    local t_wait_start = t_pulse_start - timer.measure.t()
    if t_wait_start > 0 then
        delay(t_wait_start)
    elseif -t_wait_start > late_tol then
        overruns = overruns + 1
        if -t_wait_start > worst_overrun then worst_overrun = -t_wait_start end
    end

    -- Light pulse ON
    {stim_ch}.source.levelv = light_voltage

    -- Open the measurement window measure_start_delay into the pulse, then
    -- measure (hardware averaging active; this call blocks for the ADC time).
    local t_meas = t_pulse_start + measure_start_delay - timer.measure.t()
    if t_meas > 0 then delay(t_meas) end
    {read_ch}.measure.i({read_ch}.nvbuffer1)

    -- Hold the light ON until the full pulse_width has elapsed, THEN drop the
    -- level. This absolute LED-off deadline is what makes the delivered dose
    -- equal intensity x pulse_width for every pulse.
    local t_off = t_pulse_start + pulse_width - timer.measure.t()
    if t_off > 0 then delay(t_off) end
    {stim_ch}.source.levelv = 0

    -- Advance to the next pulse's absolute start (no cumulative drift)
    t_pulse_start = t_pulse_start + pulse_period
end

-- Cleanup: levels to 0 BEFORE the outputs go off, filter off on the read channel
{stim_ch}.source.levelv = 0
{read_ch}.source.levelv = 0
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF
{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF
{chr(10).join(_emit_restore_adc(read_ch))}

print(string.format("{LUA_TIMING_TOKEN} overruns=%d worst_ms=%.4f n_pulses=%d",
                    overruns, worst_overrun * 1000, n_pulses))

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return LuaScript(lua_script, settle_warning=settle_warning,
                     fixed_i_range=fixed_i_range)


def visual_continuous_plan(params):
    """Validated timing facts for the continuous I(t) sequence.

    Shared by the generator and visual_synapse_sequence so the buffer-size
    bound is computed once. The bound is exact for the sampling grid the script
    uses (one sample per interval from t = 0 to the end of the post-pulse tail),
    not padded.
    """
    pulse_width_ms = params['pulse_width_ms']
    pulse_period_ms = params['pulse_period_ms']
    n_pulses = params['n_pulses']
    nplc = params.get('nplc', 0.1)  # Lower NPLC for faster sampling in continuous mode
    sample_interval_ms = params.get('sample_interval_ms', 1.0)
    _check_times_ms(positive={'pulse_width_ms': pulse_width_ms,
                              'pulse_period_ms': pulse_period_ms,
                              'sample_interval_ms': sample_interval_ms})
    if pulse_period_ms < pulse_width_ms:
        raise ValueError(f"Pulse period ({pulse_period_ms}ms) must be >= pulse width ({pulse_width_ms}ms)")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses!r}")

    sample_dur_ms = nplc * (1000.0 / _line_freq_hz(params))
    # Dark samples kept after the last pulse: up to 10 intervals, never past
    # the end of that pulse's period.
    tail_ms = min(10.0 * sample_interval_ms, pulse_period_ms - pulse_width_ms)
    span_ms = (n_pulses - 1) * pulse_period_ms + pulse_width_ms + tail_ms
    max_samples = int(span_ms / sample_interval_ms + 1e-9) + 1

    sample_warning = None
    if sample_dur_ms > sample_interval_ms:
        sample_warning = (
            f"one ADC sample ({sample_dur_ms:.3g} ms = nplc {nplc:g} / "
            f"{_line_freq_hz(params):g} Hz) is longer than the requested "
            f"{sample_interval_ms:g} ms sample interval; samples are taken back to "
            "back on the next free interval boundary, so the delivered spacing is "
            "longer than requested. Lower NPLC or raise the sample interval.")
    fixed_i_range = get_fixed_current_range(params['compliance_A'])
    # No settled reading is possible faster than the range settles (measure.delay = 0).
    settle_warning = check_read_settle(sample_interval_ms / 1000.0, fixed_i_range)
    return {
        'sample_dur_ms': sample_dur_ms, 'sample_interval_ms': sample_interval_ms,
        'tail_ms': tail_ms, 'max_samples': max_samples, 'nplc': nplc,
        'fixed_i_range': fixed_i_range, 'sample_warning': sample_warning,
        'settle_warning': settle_warning,
    }


def generate_visual_continuous_lua_script(stim_ch, read_ch, params):
    """
    Generates LUA script for continuous photocurrent monitoring during light pulses.

    This mode provides I(t) data - continuous current vs time, useful for
    analyzing transient photocurrent dynamics.

    The light pulse edges (ON at k*period, OFF at k*period + width) are scheduled
    on the absolute timer, independently of sampling. Samples are taken on the
    absolute grid j*sample_interval between edges, and a sample is only started
    if its whole ADC integration finishes before the next edge, so sampling can
    never stretch a pulse or delay the next one.

    Args:
        stim_ch (str): Stimulus channel for LED ('smua' or 'smub')
        read_ch (str): Read channel for current measurement ('smua' or 'smub')
        params (dict): Measurement parameters

    Returns:
        LuaScript: LUA script; `.facts` holds `max_samples` (buffer-size upper
        bound), `sample_warning` and `settle_warning`.
    """

    _check_channels(stim_ch, read_ch)

    # Extract parameters
    light_voltage = params['light_pulse_voltage']
    n_pulses = params['n_pulses']
    compliance_A = params['compliance_A']
    line_freq_hz = _line_freq_hz(params)
    plan = visual_continuous_plan(params)
    nplc = plan['nplc']

    # Convert times to seconds
    pulse_width_s = params['pulse_width_ms'] / 1000.0
    pulse_period_s = params['pulse_period_ms'] / 1000.0
    sample_interval_s = plan['sample_interval_ms'] / 1000.0

    # Check buffer capacity
    KEITHLEY_MAX_BUFFER_SIZE = 50000
    if plan['max_samples'] > KEITHLEY_MAX_BUFFER_SIZE:
        raise ValueError(
            f"Expected samples ({plan['max_samples']}) exceeds buffer capacity.\n"
            f"Reduce pulse count or increase sample interval."
        )

    for w in (plan['sample_warning'], plan['settle_warning']):
        if w:
            print(f"⚠️  {w}")

    fixed_i_range = plan['fixed_i_range']
    max_voltage = params.get('max_voltage', 200.0)
    fixed_v_range = _get_fixed_voltage_range(light_voltage, max_voltage)
    read_v_range = _get_fixed_voltage_range(0.0, max_voltage)
    adc_setup = "\n".join(_emit_read_adc_setup(read_ch, nplc, 1))

    lua_script = f"""
-- Visual Synapse Continuous Mode - I(t) Measurement
-- Continuously samples photocurrent during light pulse sequence

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local n_pulses = {n_pulses}
local light_voltage = {light_voltage}
local pulse_width = {pulse_width_s}
local pulse_period = {pulse_period_s}
local sample_interval = {sample_interval_s}
local sample_dur = {plan['sample_dur_ms'] / 1000.0}
local tail = {plan['tail_ms'] / 1000.0}
local late_tol = {EDGE_LATE_TOL_S}

-- Buffer Configuration.
-- Capacity is READ-ONLY on the 2600 series (Reference Manual: "Attribute (R)")
-- so it cannot be sized from here; only the overflow policy is settable.
{read_ch}.nvbuffer1.fillmode = {read_ch}.FILL_ONCE
{read_ch}.nvbuffer1.clear()
{read_ch}.nvbuffer1.appendmode = 1
{read_ch}.nvbuffer1.collecttimestamps = 1
{read_ch}.nvbuffer1.collectsourcevalues = 0

-- Configure read channel (0V for Jsc)
{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS
{read_ch}.source.levelv = 0
{read_ch}.source.rangev = {read_v_range}
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{adc_setup}

-- Configure stimulus channel
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}

-- Both outputs ON once, at 0 V; the LED is pulsed by LEVEL changes only (see the
-- note in the standard visual script on why OUTPUT_OFF is not toggled).
{read_ch}.source.output = {read_ch}.OUTPUT_ON
{stim_ch}.source.output = {stim_ch}.OUTPUT_ON
delay(0.01)

timer.reset()
local k = 0
local overruns = 0
local worst_overrun = 0

-- Block until the absolute time t_edge. An edge reached later than late_tol
-- is counted, so a pulse that did not start/stop on schedule is reported.
local function wait_until(t_edge)
    local w = t_edge - timer.measure.t()
    if w > 0 then
        delay(w)
    elseif -w > late_tol then
        overruns = overruns + 1
        if -w > worst_overrun then worst_overrun = -w end
    end
end

-- Sample on the absolute grid k*sample_interval until t_stop, then land on
-- t_stop. A sample whose integration would run past t_stop is not started, so
-- the next edge is never delayed by an ADC conversion. If the ADC is slower
-- than the interval, the next free grid point is used.
local function sample_until(t_stop)
    while true do
        local now = timer.measure.t()
        local kk = math.ceil(now / sample_interval - 1e-6)
        if kk < k then kk = k end
        local t_s = kk * sample_interval
        if t_s + sample_dur > t_stop then break end
        if t_s > now then delay(t_s - now) end
        {read_ch}.measure.i({read_ch}.nvbuffer1)
        k = kk + 1
    end
    wait_until(t_stop)
end

for pulse = 0, n_pulses - 1 do
    local t_on = pulse * pulse_period
    sample_until(t_on)
    {stim_ch}.source.levelv = light_voltage
    sample_until(t_on + pulse_width)
    {stim_ch}.source.levelv = 0
end

-- Dark samples after the last pulse
sample_until((n_pulses - 1) * pulse_period + pulse_width + tail)

-- Cleanup: levels to 0 BEFORE the outputs go off, filter off on the read channel
{stim_ch}.source.levelv = 0
{read_ch}.source.levelv = 0
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF
{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF
{chr(10).join(_emit_restore_adc(read_ch))}

print(string.format("{LUA_TIMING_TOKEN} overruns=%d worst_ms=%.4f n_pulses=%d",
                    overruns, worst_overrun * 1000, n_pulses))

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return LuaScript(lua_script, **plan)


# =============================================================================
# LUA SCRIPT FOR UNIFIED CYCLE MODE (SINGLE/DUAL CHANNEL, MULTI-TRAIN)
# =============================================================================

KEITHLEY_MAX_BUFFER_SIZE = 50000


class LuaExecutionError(RuntimeError):
    """A generated TSP script did not complete cleanly.

    Carries what the instrument still holds so the caller can report it:
    `buffer_counts` ({channel: "n/expected" or an 'unreadable' note}) and
    `instrument_errors` (the error-queue entries). Never raised for a script
    that finished and left a clean queue.
    """

    def __init__(self, message, buffer_counts=None, instrument_errors=None):
        super().__init__(message)
        self.buffer_counts = buffer_counts or {}
        self.instrument_errors = instrument_errors or []


def _current_drive_voltage_limit(cfg, label):
    """Voltage compliance (V) for a current-driven stimulus.

    Taken from an explicit `voltage_limit_V`, else the instrument ceiling
    `max_voltage` stamped by the caller. Raises when neither is present: a
    compliance voltage is a device-safety decision and is not guessed. (It used
    to be derived from the read/stim LEVEL, i.e. 0.1-0.2 V, which starved a
    current source of the voltage it needs to drive any real load.)
    """
    limit = cfg.get('voltage_limit_V', cfg.get('max_voltage'))
    if limit is None:
        raise ValueError(
            f"{label}: current drive needs a voltage compliance. Set "
            "'voltage_limit_V' (or 'max_voltage', the instrument ceiling) in the config."
        )
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(f"{label}: voltage compliance must be a positive finite voltage, got {limit!r}")
    return limit


def _validate_train_config(config, label="Train"):
    """
    Validates a train configuration dict.

    Args:
        config (dict): Train configuration
        label (str): Label for error messages (e.g. 'Train A')

    Raises:
        ValueError: On invalid configuration
    """
    required = ['topology', 'write_ch', 'read_ch', 'stim_level', 'read_voltage',
                'stim_width_ms', 'stim_period_ms', 'n_pulses', 'read_delay_ms',
                'compliance_A']
    for key in required:
        if key not in config:
            raise ValueError(f"{label}: Missing required parameter '{key}'")

    for key in ('write_ch', 'read_ch'):
        if config[key] not in ('smua', 'smub'):
            raise ValueError(f"{label}: {key} must be 'smua' or 'smub', got {config[key]!r}")
    drive = config.get('stim_drive_type', 'V')
    if drive not in ('V', 'I'):
        raise ValueError(f"{label}: stim_drive_type must be 'V' or 'I', got {drive!r}")
    if drive == 'I':
        _current_drive_voltage_limit(config, label)
    if not (np.isfinite(config['stim_width_ms']) and config['stim_width_ms'] > 0):
        raise ValueError(f"{label}: stim_width_ms must be > 0")
    if not (np.isfinite(config['read_delay_ms']) and config['read_delay_ms'] >= 0):
        raise ValueError(f"{label}: read_delay_ms must be >= 0")
    if not (np.isfinite(config['compliance_A']) and config['compliance_A'] > 0):
        raise ValueError(f"{label}: compliance_A must be > 0")
    if int(config['n_pulses']) != config['n_pulses']:
        raise ValueError(f"{label}: n_pulses must be a whole number, got {config['n_pulses']!r}")

    if config['topology'] == 'single':
        if config['write_ch'] != config['read_ch']:
            raise ValueError(f"{label}: Single-channel topology requires write_ch == read_ch")
        if config.get('stim_drive_type', 'V') != 'V':
            raise ValueError(
                f"{label}: Single-channel topology requires voltage drive (stim_drive_type='V'). "
                "Current drive would require relay-toggling source.func changes."
            )
    elif config['topology'] == 'dual':
        if config['write_ch'] == config['read_ch']:
            raise ValueError(f"{label}: Dual-channel topology requires write_ch != read_ch")
    else:
        raise ValueError(f"{label}: topology must be 'single' or 'dual', got '{config['topology']}'")

    if config['n_pulses'] <= 0:
        raise ValueError(f"{label}: n_pulses must be > 0")
    if config['stim_period_ms'] < config['stim_width_ms']:
        raise ValueError(f"{label}: stim_period_ms must be >= stim_width_ms")


def _get_fixed_current_range(compliance_A):
    """Returns the smallest fixed current range that accommodates the compliance."""
    ranges = [100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6,
              1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
    for r in ranges:
        if compliance_A <= r:
            return r
    return ranges[-1]


def _get_fixed_voltage_range(voltage, max_voltage=200.0):
    """
    Return the smallest native source.rangev that can source |voltage| on
    the given instrument family. Writing a range value that is not native
    triggers error -222 ("Parameter data out of range") on 2600-series SMUs.

    Native voltage ranges:
        40 V family  (2601/02/04):  100 mV, 1 V, 6 V, 40 V
        200 V family (2611/12/14, 2634/35/36):  200 mV, 2 V, 20 V, 200 V
    """
    v = abs(voltage)
    if max_voltage >= 200:
        ranges = [0.2, 2.0, 20.0, 200.0]
    else:
        ranges = [0.1, 1.0, 6.0, 40.0]
    for r in ranges:
        if v <= r:
            return r
    return ranges[-1]


def generate_cycle_lua_script(train_a_config, train_b_config, n_cycles,
                              inter_train_delay_ms=500, inter_cycle_delay_ms=1000):
    """
    Generates a single LUA script for on-instrument potentiation-depression cycling.

    Supports two channel topologies per train:
    - Single-channel: Same SMU writes and reads (voltage switching). The read
      line returns to 0 V after every read, as in pulse-read, to limit read
      disturb; it is raised to the read voltage only for read_delay + the ADC.
    - Dual-channel: One SMU writes, another reads at constant bias

    Args:
        train_a_config (dict): Train A configuration (see _validate_train_config)
        train_b_config (dict or None): Train B configuration, or None
        n_cycles (int): Number of A-B cycle repetitions
        inter_train_delay_ms (float): Delay between the ACTUAL end of Train A and
            the start of Train B
        inter_cycle_delay_ms (float): Delay between the ACTUAL end of the last
            train and the next cycle

    Returns:
        tuple: (lua_script: str, buffer_map: dict). Besides the buffer layout,
        each buffer_map['train_*'] entry carries the timing plan the host needs
        to judge the run (requested period, achievable lower bound, drift
        threshold, read-settle verdict) and buffer_map['feasible_time_s'] the
        duration the script can actually take, which governs the host timeout.
    """
    # --- Validate ---
    _validate_train_config(train_a_config, "Train A")
    if train_b_config is not None:
        _validate_train_config(train_b_config, "Train B")
    if n_cycles <= 0:
        raise ValueError("n_cycles must be > 0")
    for name, delay_ms in (("inter_train_delay_ms", inter_train_delay_ms),
                           ("inter_cycle_delay_ms", inter_cycle_delay_ms)):
        if not (np.isfinite(delay_ms) and delay_ms >= 0):
            raise ValueError(f"{name} must be a finite non-negative time, got {delay_ms!r}")

    # Mains line frequency drives the ADC integration aperture and rejection.
    # Both trains run on the same instrument/mains, so resolve once from Train A.
    line_freq_hz = _line_freq_hz(train_a_config)
    NPLC_PERIOD_MS = 1000.0 / line_freq_hz

    # --- Collect all unique channels ---
    channels_used = {train_a_config['write_ch'], train_a_config['read_ch']}
    if train_b_config:
        channels_used.update({train_b_config['write_ch'], train_b_config['read_ch']})

    # --- Compute derived values ---
    def train_lua_params(cfg, label):
        # The one averaging rule, shared with pulse-read. This function used its
        # own copy, without the FP nudge, and disagreed on 35 of 35 settings at
        # 60 Hz (59 vs 60 samples).
        nplc, measure_avg = derive_measurement_averaging(
            dict(cfg, line_freq_hz=line_freq_hz))
        max_v = cfg.get('max_voltage', 200.0)
        single = cfg['topology'] == 'single'
        drive = cfg.get('stim_drive_type', 'V')
        stim, read_v = abs(cfg['stim_level']), abs(cfg['read_voltage'])
        fixed_i_range = _get_fixed_current_range(cfg['compliance_A'])
        meas_time_ms = nplc * NPLC_PERIOD_MS * measure_avg
        settle_s = cfg.get('settle_ms', 5) / 1000.0
        read_delay_s = cfg['read_delay_ms'] / 1000.0
        # What precedes a read after a bias CHANGE: the settle for the first
        # read; on the single-channel line also read_delay on every pulse.
        # Dual-channel per-pulse reads sit on a constant bias.
        bias_change_settle_s = min(settle_s, read_delay_s) if single else settle_s
        events_ms = [e for e in (cfg['stim_width_ms'], cfg['read_delay_ms'], meas_time_ms) if e > 0]
        return {
            'nplc': nplc,
            'measure_avg': measure_avg,
            'fixed_i_range': fixed_i_range,
            # Source range the write role needs. Single-channel: the same SMU also
            # sources the read voltage. Dual: the write SMU only sources the stimulus.
            'write_v_range': (_get_fixed_voltage_range(max(stim, read_v), max_v) if single
                              else _get_fixed_voltage_range(stim, max_v) if drive == 'V' else None),
            # A current stimulus: the range must CONTAIN |stim_level| (it was
            # derived from the compliance, giving a 5 mA pulse a 100 uA range) and
            # the compliance is a VOLTAGE (it was the read/stim level, 0.1-0.2 V).
            'write_i_range': _get_fixed_current_range(stim) if drive == 'I' else None,
            'v_limit': _current_drive_voltage_limit(cfg, label) if drive == 'I' else None,
            'read_v_range': _get_fixed_voltage_range(read_v, max_v),
            'stim_width_s': cfg['stim_width_ms'] / 1000.0,
            'stim_period_s': cfg['stim_period_ms'] / 1000.0,
            'read_delay_s': read_delay_s,
            'settle_s': settle_s,
            'meas_time_ms': meas_time_ms,
            # Lower bound of the loop's own work per pulse: the waits and the ADC
            # integration it must perform. TSP command latency is on top and is
            # hardware-dependent, so it is reported by the overrun count instead
            # of being guessed here. (settle_ms is NOT per-pulse work: it only
            # precedes the baseline read, and its inclusion overstated the
            # minimum by 5 ms.)
            'min_period_ms': cfg['stim_width_ms'] + cfg['read_delay_ms'] + meas_time_ms,
            'drift_threshold_ms': max(0.1, min(1.0, 0.25 * min(events_ms))),
            'read_settle_warning': check_read_settle(bias_change_settle_s, fixed_i_range),
        }

    a_lp = train_lua_params(train_a_config, "Train A")
    b_lp = train_lua_params(train_b_config, "Train B") if train_b_config else None

    # --- Buffer map ---
    a_read = train_a_config['read_ch']
    b_read = train_b_config['read_ch'] if train_b_config else None
    same_buffer = (b_read is None) or (b_read == a_read)

    pts_a_per_cycle = train_a_config['n_pulses'] + 1   # initial read + pulses
    pts_b_per_cycle = (train_b_config['n_pulses'] + 1) if train_b_config else 0

    if same_buffer:
        total = (pts_a_per_cycle + pts_b_per_cycle) * n_cycles
        if total > KEITHLEY_MAX_BUFFER_SIZE:
            raise ValueError(
                f"Total buffer requirement ({total} points) exceeds instrument capacity "
                f"({KEITHLEY_MAX_BUFFER_SIZE}). Reduce n_pulses or n_cycles."
            )
    else:
        total_a = pts_a_per_cycle * n_cycles
        total_b = pts_b_per_cycle * n_cycles
        if total_a > KEITHLEY_MAX_BUFFER_SIZE:
            raise ValueError(f"Train A buffer ({total_a} pts) exceeds capacity ({KEITHLEY_MAX_BUFFER_SIZE}).")
        if total_b > KEITHLEY_MAX_BUFFER_SIZE:
            raise ValueError(f"Train B buffer ({total_b} pts) exceeds capacity ({KEITHLEY_MAX_BUFFER_SIZE}).")

    def plan(cfg, lp, pts_per_cycle):
        return {
            'channel': cfg['read_ch'], 'buffer': 'nvbuffer1',
            'n_points_per_cycle': pts_per_cycle,
            'n_total': pts_per_cycle * n_cycles,
            'n_pulses': cfg['n_pulses'],
            'period_ms': cfg['stim_period_ms'],
            'min_period_ms': lp['min_period_ms'],
            'drift_threshold_ms': lp['drift_threshold_ms'],
            'read_settle_warning': lp['read_settle_warning'],
        }

    buffer_map = {'train_a': plan(train_a_config, a_lp, pts_a_per_cycle)}
    if train_b_config:
        buffer_map['train_b'] = plan(train_b_config, b_lp, pts_b_per_cycle)

    # Duration the script can really take: a train whose period is below what it
    # can achieve runs at the hardware limit, not at the requested period.
    def train_feasible_s(cfg, lp):
        return (cfg['n_pulses'] * max(cfg['stim_period_ms'], lp['min_period_ms']) / 1000.0
                + lp['settle_s'])
    per_cycle_s = train_feasible_s(train_a_config, a_lp)
    if train_b_config:
        per_cycle_s += train_feasible_s(train_b_config, b_lp) + inter_train_delay_ms / 1000.0
    buffer_map['feasible_time_s'] = (per_cycle_s * n_cycles
                                     + inter_cycle_delay_ms / 1000.0 * (n_cycles - 1) + 0.1)

    inter_train_s = inter_train_delay_ms / 1000.0
    inter_cycle_s = inter_cycle_delay_ms / 1000.0

    # --- Build LUA script ---
    lines = []
    lines.append(f"-- Unified Cycle Sequence: {n_cycles} cycle(s)")
    lines.append(f"-- Train A: {train_a_config['topology']}, {train_a_config['n_pulses']} pulses, "
                 f"write={train_a_config['write_ch']} read={train_a_config['read_ch']}")
    if train_b_config:
        lines.append(f"-- Train B: {train_b_config['topology']}, {train_b_config['n_pulses']} pulses, "
                     f"write={train_b_config['write_ch']} read={train_b_config['read_ch']}")

    # --- Mains synchronisation ---
    # Sync the ADC to mains for normal-mode rejection of line-frequency pickup.
    lines.append("")
    lines.append(f"localnode.linefreq = {int(line_freq_hz)}")

    # --- Buffer configuration ---
    # Buffer capacity is read-only on the 2600 series; only the overflow
    # behaviour can be set from a script.
    lines.append("")
    lines.append("-- Buffer configuration")
    for ch in ([a_read] if same_buffer else [a_read, b_read]):
        lines.append(f"{ch}.nvbuffer1.fillmode = {ch}.FILL_ONCE")
        lines.append(f"{ch}.nvbuffer1.clear()")
        lines.append(f"{ch}.nvbuffer1.appendmode = 1")
        lines.append(f"{ch}.nvbuffer1.collecttimestamps = 1")
        lines.append(f"{ch}.nvbuffer1.collectsourcevalues = 0")

    # --- Channel configuration ---
    lines.append("")
    lines.append("-- Channel configuration")

    # H11: channel configuration is emitted ONCE per channel and role-group —
    # re-ranging mid-run would perturb the timing the whole architecture is
    # built to protect. That silently discarded the second train's ranges,
    # compliance and filter whenever the two trains shared a channel: Train A
    # at 1.0 V followed by Train B at 15 V on the same SMU left Train B
    # clipping at the 2 V range with no error anywhere.
    #
    # Sharing a channel is legitimate; sharing it with INCOMPATIBLE settings is
    # not, and cannot be honoured without re-ranging. So it is rejected here
    # rather than silently mis-measured. The comparison is per CHANNEL and per
    # setting the two trains both need — not only the write range: trains with
    # swapped roles (A writes smua / reads smub, B writes smub / reads smua)
    # configured smub's source range once for the 0.2 V read role and then had B
    # write -1 V into it.
    def requirements(cfg, lp):
        w, r = cfg['write_ch'], cfg['read_ch']
        read_req = {'measure.rangei': lp['fixed_i_range'], 'nplc': lp['nplc'],
                    'measure_avg': lp['measure_avg']}

        def volts(rng):
            return {'source.func': 'DCVOLTS', 'source.rangev': rng,
                    'source.limiti': cfg['compliance_A']}
        if cfg['topology'] == 'single':
            return {w: {**volts(lp['write_v_range']), **read_req}}
        if cfg.get('stim_drive_type', 'V') == 'V':
            write_req = volts(lp['write_v_range'])
        else:
            write_req = {'source.func': 'DCAMPS', 'source.rangei': lp['write_i_range'],
                         'source.limitv': lp['v_limit']}
        return {w: write_req, r: {**volts(lp['read_v_range']), **read_req}}

    if train_b_config:
        a_req = requirements(train_a_config, a_lp)
        b_req = requirements(train_b_config, b_lp)
        shared = sorted(set(a_req) & set(b_req))
        conflicts = [
            f"{ch} {key} ({a_req[ch][key]} vs {b_req[ch][key]})"
            for ch in shared
            for key in sorted(set(a_req[ch]) & set(b_req[ch]))
            if a_req[ch][key] != b_req[ch][key]
        ]
        if conflicts:
            raise ValueError(
                "Train A and Train B share SMU channel(s) "
                f"{', '.join(shared)} but require different "
                "settings:\n  - " + "\n  - ".join(conflicts) + "\n\n"
                "A channel is configured once, before the run, because "
                "re-ranging mid-sequence would corrupt the pulse timing. "
                "Either give the trains compatible settings, or put them "
                "on separate channels."
            )

    emitted = set()

    def emit_group(ch, group, group_lines):
        if (ch, group) not in emitted:
            emitted.add((ch, group))
            lines.extend(group_lines)

    def volts_group(ch, cfg, rng, level):
        return [f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS",
                f"{ch}.source.rangev = {rng}",
                f"{ch}.source.levelv = {level}",
                f"{ch}.source.limiti = {cfg['compliance_A']}"]

    def measure_group(ch, lp):
        # The ADC state is fixed completely here, once per read channel: delay
        # off, NPLC, count/filter and AUTOZERO_ONCE (see _emit_read_adc_setup).
        return [f"{ch}.measure.rangei = {lp['fixed_i_range']}",
                f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF"
                ] + _emit_read_adc_setup(ch, lp['nplc'], lp['measure_avg'])

    def emit_channel_config(cfg, lp, label):
        wch, rch = cfg['write_ch'], cfg['read_ch']
        if cfg['topology'] == 'single':
            lines.append(f"-- {label} single-channel: {wch}")
            emit_group(wch, 'volts', volts_group(wch, cfg, lp['write_v_range'], 0))
            emit_group(wch, 'measure', measure_group(wch, lp))
        else:
            lines.append(f"-- {label} write channel: {wch}, read channel: {rch}")
            if cfg.get('stim_drive_type', 'V') == 'V':
                emit_group(wch, 'volts', volts_group(wch, cfg, lp['write_v_range'], 0))
            else:
                emit_group(wch, 'amps', [
                    f"{wch}.source.func = {wch}.OUTPUT_DCAMPS",
                    f"{wch}.source.rangei = {lp['write_i_range']}",
                    f"{wch}.source.leveli = 0",
                    f"{wch}.source.limitv = {lp['v_limit']}"])
            emit_group(rch, 'volts', volts_group(rch, cfg, lp['read_v_range'], cfg['read_voltage']))
            emit_group(rch, 'measure', measure_group(rch, lp))

    emit_channel_config(train_a_config, a_lp, "Train A")
    if train_b_config:
        emit_channel_config(train_b_config, b_lp, "Train B")

    # --- Outputs ON ---
    lines.append("")
    lines.append("-- Turn on all outputs at idle levels (no relay toggling during execution)")
    for ch in sorted(channels_used):
        lines.append(f"{ch}.source.output = {ch}.OUTPUT_ON")
    lines.append("delay(0.01)")

    # --- Main cycle loop ---
    lines.append("")
    lines.append("-- Main cycle loop")
    lines.append("timer.reset()")
    # t_abs is the monotonic script-time deadline of the current gap. Each train
    # sets it to the time it ACTUALLY ended, so the inter-train and inter-cycle
    # delays are measured from there: carrying the nominal schedule forward let
    # an overrunning train eat its following 500 ms delay down to 74 ms.
    lines.append("local t_abs = 0")
    lines.append("local a_overruns, a_worst = 0, 0")
    if train_b_config:
        lines.append("local b_overruns, b_worst = 0, 0")
    lines.append("")
    lines.append(f"for cycle = 1, {n_cycles} do")

    def emit_train_block(cfg, lp, label, tag, indent="    "):
        rch = cfg['read_ch']
        wch = cfg['write_ch']
        n = cfg['n_pulses']
        single = cfg['topology'] == 'single'
        lvl = 'levelv' if cfg.get('stim_drive_type', 'V') == 'V' else 'leveli'

        def dly(seconds, name, extra=""):
            d = lua_delay(seconds, name)   # delay(0) aborts a 2600 script
            if d:
                lines.append(f"{indent}{extra}{d}")

        lines.append("")
        lines.append(f"{indent}-- === {label}: {cfg['topology']} channel, {n} pulses ===")

        # Initial (pre-stimulus baseline) read
        if single:
            lines.append(f"{indent}{wch}.source.levelv = {cfg['read_voltage']}")
        else:
            # Read channel at its constant bias, write channel idle
            lines.append(f"{indent}{rch}.source.levelv = {cfg['read_voltage']}")
            lines.append(f"{indent}{wch}.source.{lvl} = 0")
        dly(lp['settle_s'], "settle_ms")
        lines.append(f"{indent}{rch}.measure.i({rch}.nvbuffer1)")
        if single:
            lines.append(f"{indent}{wch}.source.levelv = 0")

        # The train's time base is taken AFTER the baseline read, as pulse-read
        # does. Taking it before the settle and the read made the first interval
        # of every train shorter by the duration of that read (~7 ms: 42.85 ms
        # delivered for a 50 ms period).
        lines.append(f"{indent}local t_next = timer.measure.t()")
        lines.append(f"{indent}for pulse = 1, {n} do")
        lines.append(f"{indent}    t_next = t_next + {lp['stim_period_s']}")

        if single:
            # Write, raise the line to the read voltage for read_delay + the ADC,
            # then back to 0 V. It used to sit at the read voltage for the rest
            # of the period, i.e. continuous read disturb between pulses.
            lines.append(f"{indent}    {wch}.source.levelv = {cfg['stim_level']}")
            dly(lp['stim_width_s'], "stim_width_ms", "    ")
            lines.append(f"{indent}    {wch}.source.levelv = {cfg['read_voltage']}")
            dly(lp['read_delay_s'], "read_delay_ms", "    ")
            lines.append(f"{indent}    {rch}.measure.i({rch}.nvbuffer1)")
            lines.append(f"{indent}    {wch}.source.levelv = 0")
        else:
            lines.append(f"{indent}    {wch}.source.{lvl} = {cfg['stim_level']}")
            dly(lp['stim_width_s'], "stim_width_ms", "    ")
            lines.append(f"{indent}    {wch}.source.{lvl} = 0")
            dly(lp['read_delay_s'], "read_delay_ms", "    ")
            lines.append(f"{indent}    {rch}.measure.i({rch}.nvbuffer1)")

        # Timer-based period enforcement. A pulse whose own work outlasted the
        # period has no remainder: it is counted and reported (SYN_TIMING), and
        # t_next is re-baselined to now so one slow pulse neither flags every
        # later pulse nor makes the loop fire back-to-back to catch up.
        lines.append(f"{indent}    local t_remain = t_next - timer.measure.t()")
        lines.append(f"{indent}    if t_remain > 0 then")
        lines.append(f"{indent}        delay(t_remain)")
        lines.append(f"{indent}    else")
        lines.append(f"{indent}        {tag}_overruns = {tag}_overruns + 1")
        lines.append(f"{indent}        if -t_remain > {tag}_worst then {tag}_worst = -t_remain end")
        lines.append(f"{indent}        t_next = timer.measure.t()")
        lines.append(f"{indent}    end")
        lines.append(f"{indent}end")

        # Return channel to safe idle after train
        lines.append(f"{indent}{wch}.source.{lvl} = 0")
        lines.append(f"{indent}t_abs = timer.measure.t()")

    # Train A
    emit_train_block(train_a_config, a_lp, "TRAIN A", "a")

    # Inter-train delay + Train B
    if train_b_config:
        lines.append("")
        lines.append("    -- Inter-train delay, from the actual end of Train A")
        lines.append(f"    t_abs = t_abs + {inter_train_s}")
        lines.append("    local t_wait_train = t_abs - timer.measure.t()")
        lines.append("    if t_wait_train > 0 then delay(t_wait_train) end")

        emit_train_block(train_b_config, b_lp, "TRAIN B", "b")

    # Inter-cycle delay
    lines.append("")
    lines.append("    -- Inter-cycle delay, from the actual end of the last train")
    lines.append(f"    if cycle < {n_cycles} then")
    lines.append(f"        t_abs = t_abs + {inter_cycle_s}")
    lines.append("        local t_wait_cycle = t_abs - timer.measure.t()")
    lines.append("        if t_wait_cycle > 0 then delay(t_wait_cycle) end")
    lines.append("    end")

    lines.append("end")

    # --- Cleanup ---
    lines.append("")
    lines.append("-- Cleanup")
    for ch in sorted(channels_used):
        lines.append(f"{ch}.source.output = {ch}.OUTPUT_OFF")
        lines.append(f"{ch}.measure.filter.enable = {ch}.FILTER_OFF")
        if (ch, 'measure') in emitted:
            lines.extend(_emit_restore_adc(ch))   # put back delay/autozero for later J-V runs

    # Timing report, one aggregated line with per-train fields (the host merges
    # every SYN_TIMING line into one dict, so per-train lines would overwrite
    # each other). Printed before the completion token.
    fields = "a_overruns=%d a_worst_ms=%.4f a_n_pulses=%d"
    args = f"a_overruns, a_worst * 1000, {train_a_config['n_pulses'] * n_cycles}"
    if train_b_config:
        fields += " b_overruns=%d b_worst_ms=%.4f b_n_pulses=%d"
        args += f", b_overruns, b_worst * 1000, {train_b_config['n_pulses'] * n_cycles}"
    lines.append("")
    lines.append(f'print(string.format("{LUA_TIMING_TOKEN} {fields}", {args}))')

    # Completion token (host reads this to detect script end)
    lines.append(f'print("{LUA_DONE_TOKEN}")')

    lua_script = '\n'.join(lines) + '\n'

    # --- Timing validation (warn but don't block) ---
    for cfg, lp, label in [(train_a_config, a_lp, "Train A")] + \
            ([(train_b_config, b_lp, "Train B")] if train_b_config else []):
        if cfg['stim_period_ms'] < lp['min_period_ms']:
            print(f"⚠️  {label}: Requested period ({cfg['stim_period_ms']:.1f} ms) shorter than minimum "
                  f"({lp['min_period_ms']:.1f} ms = write {cfg['stim_width_ms']:g} + read_delay "
                  f"{cfg['read_delay_ms']:g} + measure {lp['meas_time_ms']:.2f}, before TSP command "
                  f"latency). Instrument will run as fast as possible.")
        if lp['read_settle_warning']:
            print(f"⚠️  {label}: {lp['read_settle_warning']}")

    return lua_script, buffer_map

# =============================================================================
# SAFE STATE — CALLED BEFORE EVERY SCRIPT UPLOAD
# =============================================================================

def _safe_state(instrument):
    """
    Drive both SMU channels to 0 V and clear the error queue.

    Called before every loadandrunscript to ensure the instrument is not
    stimulating the DUT during script upload.

    IMPORTANT: Does NOT toggle OUTPUT_ON/OUTPUT_OFF.
    - With the default offmode NORMAL, OUTPUT_OFF does NOT open the relay: the
      output is held at 0 V behind a 1 mA off-limit (RM 2-77) — it is not high
      impedance (that needs offmode HIGH_Z). So switching it here neither
      isolates the DUT nor is it "safer" than a 0 V level; it is a pointless
      transition that the script's OUTPUT_ON would then reverse.
    - Instead, just drive the source level to 0 V.  If the output was ON,
      the DUT is actively driven to 0 V (safe).  If the output was OFF,
      levelv has no external effect (also safe).
    """
    instrument.write("errorqueue.clear()")
    instrument.write("smua.source.levelv = 0")
    instrument.write("smub.source.levelv = 0")


# =============================================================================
# CYCLE-SPECIFIC LUA EXECUTION AND BUFFER RETRIEVAL
# =============================================================================

def _upload_lua(instrument, lua_script):
    """
    Uploads a script via anonymous loadandrunscript, WITHOUT starting it.
    Nothing has run on the instrument when this returns or raises, so a failure
    here is the only point at which a caller may still choose another path.
    """
    # A token left from an earlier run would otherwise end the wait below at once.
    _drain_output(instrument)
    _safe_state(instrument)
    instrument.write("loadandrunscript")
    time.sleep(0.02)

    # Chunked upload (~512 bytes per VISA write)
    raw_lines = lua_script.strip().split('\n')
    code_lines = [line for line in raw_lines
                  if line.strip() and not line.strip().startswith('--')]
    chunk = []
    chunk_size = 0
    for line in code_lines:
        line_size = len(line) + 1
        if chunk_size + line_size > 512 and chunk:
            instrument.write('\n'.join(chunk))
            chunk = []
            chunk_size = 0
        chunk.append(line)
        chunk_size += line_size
    if chunk:
        instrument.write('\n'.join(chunk))


def _run_uploaded_lua(instrument, expected_time_s=None):
    """
    Starts the uploaded script (endscript) and waits for LUA_DONE_TOKEN, which
    every generated script emits as its final statement. Returns the dict of
    SYN_TIMING fields the script reported.

    No sleep-based fallback: a token that does not arrive means the script did
    not finish (a mid-script TSP error aborts it; the instrument's error queue
    then says why), and carrying on as if it had is how a dead run was reported
    as complete. Raises LuaExecutionError; the caller owns the cleanup.
    """
    instrument.write("endscript")
    try:
        return _wait_lua_complete(instrument, expected_time_s) or {}
    except Exception as e:
        raise LuaExecutionError(
            f"completion token not received within the timeout ({e!r})") from e


def _upload_and_execute_lua(instrument, lua_script, expected_time_s=None):
    """
    Uploads and executes a LUA script via anonymous loadandrunscript and waits
    for the completion token. Does NOT retrieve any buffer data.

    Returns the SYN_TIMING statistics the script printed; raises
    LuaExecutionError if the token never arrives.
    """
    _upload_lua(instrument, lua_script)
    return _run_uploaded_lua(instrument, expected_time_s)


def _outputs_off(instrument):
    """Both SMU outputs OFF. Returns None, or a description of why it failed."""
    try:
        instrument.write("smua.source.output = smua.OUTPUT_OFF")
        instrument.write("smub.source.output = smub.OUTPUT_OFF")
    except Exception as e:
        return f"COULD NOT SWITCH THE OUTPUTS OFF ({e!r}) - check the instrument front panel"
    return None


def _read_error_queue(instrument, limit=20):
    """Entries of the instrument error queue (cleared as they are read).

    An unreadable queue is returned as an entry saying so, never as an empty
    list: "we could not look" must not read as "there was nothing there".
    """
    try:
        count = int(float(instrument.query("print(errorqueue.count)").strip()))
        entries = [instrument.query('print(errorqueue.next())').strip()
                   for _ in range(min(limit, count))]
        if count:
            instrument.write("errorqueue.clear()")
        if count > limit:
            entries.append(f"... and {count - limit} more")
        return entries
    except Exception as e:
        return [f"error queue unreadable: {e!r}"]


def _cycle_fault(instrument, buffer_map, reason, errors=None):
    """Make a failed cycle run safe and build the exception that reports it.

    After a mid-script TSP error the outputs were left ON at the read voltage
    and the caller printed "[OK] Cycle sequence complete" with nothing parsed.
    The outputs go OFF first; the buffer counts and the error queue are then
    attached so the failure is diagnosable. The caller raises the result.
    """
    off_problem = _outputs_off(instrument)
    expected = {}
    for key in ('train_a', 'train_b'):
        info = buffer_map.get(key)
        if info:
            expected[info['channel']] = expected.get(info['channel'], 0) + info['n_total']
    counts = {}
    for ch, exp in expected.items():
        try:
            held = int(float(instrument.query(f"print({ch}.nvbuffer1.n)").strip()))
            counts[ch] = f"{held}/{exp}"
        except Exception as e:
            counts[ch] = f"unreadable ({e!r})/{exp}"
    if errors is None:
        errors = _read_error_queue(instrument)
    msg = (f"Cycle sequence FAILED: {reason}.\n"
           f"  Outputs: {off_problem or 'switched OFF'}\n"
           f"  Buffer points held/expected: {counts}\n"
           f"  Instrument error queue: {errors if errors else 'empty'}\n"
           "  No data was returned; the sequence is NOT complete.")
    return LuaExecutionError(msg, counts, errors)


def _retrieve_buffer(instrument, channel, n_points):
    """
    Retrieves timestamps and currents from a channel's nvbuffer1.
    Uses chunked printbuffer (500 pts/read) to avoid VISA overflow.
    Queries actual buffer count first to avoid error 5038 (index exceeds maximum).

    Raises LuaExecutionError if the count cannot be read: requesting the
    expected number of points from a buffer that may hold fewer is error 5038
    at best, and misaligned data at worst.

    Returns:
        tuple: (timestamps: list[float], currents: list[float])
    """
    CHUNK = 500
    buf = f"{channel}.nvbuffer1"

    try:
        actual_n = int(float(instrument.query(f"print({buf}.n)").strip()))
    except Exception as e:
        raise LuaExecutionError(
            f"could not read the {channel} buffer count ({e!r}); not assuming "
            f"the expected {n_points} points are there") from e
    if actual_n < n_points:
        print(f"WARNING: {channel} buffer has {actual_n}/{n_points} expected points")

    all_currents = []
    all_timestamps = []

    for start_idx in range(1, actual_n + 1, CHUNK):
        end_idx = min(start_idx + CHUNK - 1, actual_n)
        curr_str = instrument.query(f"printbuffer({start_idx}, {end_idx}, {buf}.readings)").strip()
        time_str = instrument.query(f"printbuffer({start_idx}, {end_idx}, {buf}.timestamps)").strip()
        all_currents.extend(float(x) for x in curr_str.split(','))
        all_timestamps.extend(float(x) for x in time_str.split(','))

    return all_timestamps, all_currents


def _analyse_cycle_timing(result, buffer_map, lua_stats):
    """Per-train timing record of a cycle run, from the buffer timestamps.

    Each train block of each cycle is analysed on its own, against that train's
    requested period. The baseline->pulse-1 interval is not a pulse period and
    the inter-train / inter-cycle gaps lie BETWEEN blocks, so neither enters.
    (The previous analysis took the median over every consecutive pair in the
    buffer and reported the gaps as "+3465 ms drift".) An analysis that cannot
    run is recorded as such; it never reads as "no drift".
    """
    timing = {'instrument_report': dict(lua_stats) if lua_stats else
              {'status': 'unavailable', 'reason': 'script emitted no SYN_TIMING line'}}
    a_info = buffer_map['train_a']
    b_info = buffer_map.get('train_b')
    shared = b_info is not None and b_info['channel'] == a_info['channel']
    stride_shared = a_info['n_points_per_cycle'] + (b_info['n_points_per_cycle'] if shared else 0)
    layout = [('train_a', 'Train A', 'a', a_info, 'train_a', 0,
               stride_shared if shared else a_info['n_points_per_cycle'])]
    if b_info is not None:
        layout.append(('train_b', 'Train B', 'b', b_info,
                       'train_a' if shared else 'train_b',
                       a_info['n_points_per_cycle'] if shared else 0,
                       stride_shared if shared else b_info['n_points_per_cycle']))

    for key, label, tag, info, data_key, base, stride in layout:
        try:
            ts = result[data_key]['timestamps']
            n = info['n_pulses']
            req_ms = info['period_ms']
            thr_ms = info['drift_threshold_ms']
            periods_ms = []
            n_blocks = 0
            while base + n_blocks * stride + n + 1 <= len(ts):
                blk = ts[base + n_blocks * stride: base + n_blocks * stride + n + 1]
                periods_ms.extend((blk[k + 1] - blk[k]) * 1000.0 for k in range(1, n))
                n_blocks += 1
            if not periods_ms:
                rec = {'status': 'no_intervals', 'n_blocks': n_blocks,
                       'reason': 'fewer than two pulses per block or no complete block'}
            else:
                dev = np.asarray(periods_ms) - req_ms
                rec = {
                    'status': 'ok',
                    'requested_period_ms': req_ms,
                    'delivered_period_ms_median': float(np.median(periods_ms)),
                    'max_late_ms': float(max(dev.max(), 0.0)),
                    'max_early_ms': float(max(-dev.min(), 0.0)),
                    'threshold_ms': thr_ms,
                    'n_intervals': len(periods_ms),
                    'n_intervals_exceeding_threshold': int(np.sum(np.abs(dev) > thr_ms)),
                    'n_blocks': n_blocks,
                }
                if rec['n_intervals_exceeding_threshold']:
                    print(f"⚠️  {label}: {rec['n_intervals_exceeding_threshold']}/{len(periods_ms)} "
                          f"pulse intervals off the requested {req_ms:g} ms by more than "
                          f"{thr_ms:.2f} ms (worst: +{rec['max_late_ms']:.2f} / "
                          f"-{rec['max_early_ms']:.2f} ms)")
        except Exception as e:
            rec = {'status': 'analysis_failed', 'error': repr(e)}
            print(f"⚠️  {label}: timing analysis FAILED ({e!r}). Timing quality is "
                  "UNKNOWN for this train - the absence of a drift warning does NOT "
                  "mean the timing was good.")
        n_over = lua_stats.get(f'{tag}_overruns') if lua_stats else None
        if isinstance(n_over, float):
            rec['instrument_overruns'] = int(n_over)
            rec['instrument_worst_overrun_ms'] = lua_stats.get(f'{tag}_worst_ms')
            if n_over > 0:
                print(f"⚠️  {label}: {int(n_over)} pulse(s) could NOT meet the requested period "
                      f"(worst shortfall {lua_stats.get(f'{tag}_worst_ms', float('nan')):.3f} ms). "
                      "The train ran at the hardware limit there; use the buffer timestamps, "
                      "not the nominal period.")
        rec['min_period_ms'] = info['min_period_ms']
        # Same contract as pulse-read's timing record, which the GUI reads:
        # `settle_warning`, and `period_delivered` True / False / None
        # (None = could not be verified, never assumed True).
        rec['settle_warning'] = info['read_settle_warning']
        if 'instrument_overruns' in rec and rec.get('status') == 'ok':
            rec['period_delivered'] = (rec['instrument_overruns'] == 0
                                       and rec['n_intervals_exceeding_threshold'] == 0)
        elif rec.get('instrument_overruns'):
            rec['period_delivered'] = False
        else:
            rec['period_delivered'] = None
        timing[key] = rec
    return timing


def execute_cycle_lua(instrument, lua_script, buffer_map, expected_time_s):
    """
    Executes a cycle LUA script and retrieves data from one or two buffers.

    Args:
        instrument: PyVISA resource
        lua_script (str): The cycle LUA script
        buffer_map (dict): Output from generate_cycle_lua_script
        expected_time_s (float): Expected execution time at the requested periods

    Returns:
        dict: {'train_a': {'timestamps': [...], 'currents': [...]},
               'train_b': {...},
               'timing': per-train timing record (see _analyse_cycle_timing)}

    Raises:
        LuaExecutionError: the completion token did not arrive, or the
            instrument reported any error. The outputs are switched OFF first.
    """
    try:
        # The timeout follows the duration the script can really take: trains
        # whose period is below what they can achieve run longer than requested.
        timeout_basis_s = max(expected_time_s or 0.0, buffer_map.get('feasible_time_s') or 0.0)
        try:
            lua_stats = _upload_and_execute_lua(instrument, lua_script, timeout_basis_s)
        except LuaExecutionError as e:
            raise _cycle_fault(instrument, buffer_map, str(e)) from e

        # Errors first, before any data is trusted: a range or attribute write
        # the instrument rejected leaves the script RUNNING on the wrong setting.
        errors = _read_error_queue(instrument)
        if errors:
            raise _cycle_fault(instrument, buffer_map,
                               "the instrument reported error(s) during the script",
                               errors=errors)

        # 15 digits: the default 6 quantises timestamps beyond 100 s.
        _set_ascii_precision(instrument)

        result = {}

        # Retrieve Train A data
        a_info = buffer_map['train_a']
        a_ts, a_curr = _retrieve_buffer(instrument, a_info['channel'], a_info['n_total'])
        result['train_a'] = {'timestamps': a_ts, 'currents': a_curr}

        # Retrieve Train B data (if on a separate buffer)
        if 'train_b' in buffer_map:
            b_info = buffer_map['train_b']
            if b_info['channel'] != a_info['channel']:
                b_ts, b_curr = _retrieve_buffer(instrument, b_info['channel'], b_info['n_total'])
                result['train_b'] = {'timestamps': b_ts, 'currents': b_curr}
            # If same channel, Train B data is interleaved in train_a's buffer (parsed later)

        result['timing'] = _analyse_cycle_timing(result, buffer_map, lua_stats)

        # Clear buffers
        instrument.write(f"{a_info['channel']}.nvbuffer1.clear()")
        if 'train_b' in buffer_map and buffer_map['train_b']['channel'] != a_info['channel']:
            instrument.write(f"{buffer_map['train_b']['channel']}.nvbuffer1.clear()")

        return result

    except Exception:
        problem = _outputs_off(instrument)
        if problem:
            print(f"⚠️  {problem}")
        raise

# =============================================================================
# FAST LUA EXECUTION FOR SYNAPSE CHARACTERIZATION (EXISTING)
# =============================================================================

def _abort_outputs(instrument):
    """Best-effort safe shutdown: levels to 0 BEFORE the outputs go off."""
    try:
        for ch in ('smua', 'smub'):
            instrument.write(f"{ch}.source.levelv = 0")
            instrument.write(f"{ch}.source.leveli = 0")
        for ch in ('smua', 'smub'):
            instrument.write(f"{ch}.source.output = {ch}.OUTPUT_OFF")
    except Exception:
        pass


def _script_fault_report(instrument, read_ch):
    """Buffer point count reached and the instrument error queue, after a fault."""
    try:
        reached = int(float(instrument.query(f"print({read_ch}.nvbuffer1.n)").strip()))
    except Exception as e:
        reached = f"unknown ({e!r})"
    errs = []
    try:
        count = int(float(instrument.query("print(errorqueue.count)").strip()))
        for _ in range(min(10, count)):
            errs.append(instrument.query("print(errorqueue.next())").strip())
    except Exception as e:
        errs.append(f"error queue unreadable: {e!r}")
    return reached, errs


def execute_lua_script_fast(instrument, lua_script, read_ch, n_expected_points,
                            expected_time_s=None, min_event_ms=None,
                            timing_out=None, analysis='pulse_read',
                            requested_period_s=None, feasible_time_s=None,
                            n_is_exact=False):
    """
    Uploads and executes LUA script with 'Safe Delete' to prevent Error -292.

    Uses concatenated upload (fewer VISA writes) and non-intrusive OPC waiting
    to avoid interfering with real-time script execution on the instrument.

    Args:
        instrument: PyVISA instrument resource
        lua_script (str): LUA script to upload and execute
        read_ch (str): Read channel name for buffer retrieval
        n_expected_points (int): Expected number of buffer data points
        expected_time_s (float, optional): REQUESTED duration in seconds. The
            requested period for drift analysis is derived from it.
        feasible_time_s (float, optional): duration the instrument can actually
            deliver (>= expected_time_s when the requested period is infeasible).
            Sets the completion timeout and the sleep fallback; defaults to
            expected_time_s. An infeasible 5 ms x 5000 run takes ~77 s, which a
            timeout based on the requested 25 s would cut off.
        analysis: how buffer timestamps map to periods. 'pulse_read': index 0 is
            a baseline read, followed by n_expected_points-1 pulses. 'pulses':
            one reading per pulse, no baseline. 'samples': a uniform sampling
            grid whose requested spacing is `requested_period_s`.
        n_is_exact (bool): the buffer must hold exactly n_expected_points
            (pulse-read, visual per pulse). False when it is only an upper bound
            (continuous sampling). A short buffer is then a failed run.

    Raises:
        LuaScriptStartedError: any failure after the script was started. The
            device may already be partly stimulated, so callers must not re-run
            the measurement elsewhere.
    """
    started = False
    try:
        # A completion token left over from an earlier run would end the wait
        # below immediately and drop this run's timing report.
        _drain_output(instrument)

        # Use loadandrunscript (anonymous) to avoid naming entirely.
        # No name → no script.delete needed → no error -292 on front panel.
        # endscript triggers immediate execution of the anonymous script.
        _safe_state(instrument)
        instrument.write("loadandrunscript")
        time.sleep(0.02)

        # Upload script — concatenate lines to minimize VISA write count
        # Strip comments and empty lines, join with newlines, send in chunks
        lines = []
        for line in lua_script.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)

        # Send in concatenated chunks (~512 bytes each to stay within VISA buffer limits)
        chunk = []
        chunk_size = 0
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if chunk_size + line_size > 512 and chunk:
                instrument.write('\n'.join(chunk))
                chunk = []
                chunk_size = 0
            chunk.append(line)
            chunk_size += line_size
        if chunk:
            instrument.write('\n'.join(chunk))

        # endscript triggers immediate execution for loadandrunscript
        print("Executing LUA script on instrument...")
        instrument.write("endscript")
        started = True

        wait_s = feasible_time_s if feasible_time_s else expected_time_s

        # Wait for completion via LUA_DONE_TOKEN emitted as the script's
        # final statement. Deterministic and TSP-native — avoids the *OPC?
        # latching pathology that stalls repeated anonymous-script runs on
        # the 2636B (60-120 s VISA timeouts between measurements).
        # CRITICAL: Do NOT poll nvbuffer1.n during execution — that causes
        # TSP context switches which inject timing jitter into the running
        # script. The token-based wait does zero bus traffic during execution.
        lua_timing = {}
        try:
            lua_timing = _wait_lua_complete(instrument, wait_s) or {}
        except Exception:
            # Token never arrived (VISA disruption or LUA error). Sleep the
            # expected duration, then check buffer count ONCE as a final
            # integrity probe before giving up.
            #
            # The instrument's own timing report is lost on this path, so the
            # caller must not read its absence as "no overruns occurred".
            lua_timing = {'status': 'unavailable',
                          'reason': 'completion token not received'}
            wait_time = wait_s if wait_s else 60
            print(f"  [LUA] completion token not received, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            try:
                cnt_str = instrument.query(f"print({read_ch}.nvbuffer1.n)").strip()
                count = int(float(cnt_str))
                if count < n_expected_points:
                    time.sleep(wait_time * 0.5)
                    cnt_str = instrument.query(f"print({read_ch}.nvbuffer1.n)").strip()
                    count = int(float(cnt_str))
                    if count < n_expected_points:
                        raise TimeoutError(
                            f"Script did not complete: got {count}/{n_expected_points} points "
                            f"after {wait_time*1.5:.0f}s"
                        )
            except TimeoutError:
                raise
            except Exception:
                pass  # If we can't check, assume completion after sleep

        # H5: ask the buffer how many readings it actually holds before reading
        # from it. `n_expected_points` is an UPPER BOUND for continuous sampling,
        # not a count, and requesting indices past `.n` makes the instrument
        # raise error 5038. `_retrieve_buffer` already queries `.n` for exactly
        # this reason; this path did not.
        #
        # The query happens AFTER execution has completed, so it does not
        # introduce the TSP context switches the timing architecture forbids
        # during a run.
        n_available = n_expected_points
        try:
            n_available = int(float(
                instrument.query(f"print({read_ch}.nvbuffer1.n)").strip()
            ))
        except Exception as e:
            print(f"  [LUA] could not read buffer count ({e}); "
                  f"requesting the expected {n_expected_points} points")

        if n_available < 1:
            raise RuntimeError(
                f"Instrument buffer is empty after execution "
                f"(expected up to {n_expected_points} points). The script ran "
                "but recorded nothing."
            )

        if n_is_exact and n_available < n_expected_points:
            raise RuntimeError(
                f"Instrument buffer holds {n_available} of the {n_expected_points} "
                "readings this sequence must produce (buffer full or readings "
                "missed); the data set is incomplete."
            )

        n_to_read = min(n_available, n_expected_points)
        if n_available < n_expected_points:
            # Not necessarily a fault: continuous mode sizes the buffer for the
            # most it could need. Reported so a genuinely short buffer is still visible.
            print(f"  [LUA] buffer holds {n_available} points, "
                  f"upper bound was {n_expected_points}")

        # Full-precision readback: the default 6 digits quantise timestamps to
        # 1 ms beyond 100 s (10 ms beyond 1000 s) and raise false drift alarms.
        _set_ascii_precision(instrument)

        print(f"✓ LUA script completed, retrieving {n_to_read} points...")

        # Retrieve Data — use chunked reads for large datasets to avoid
        # VISA buffer overflow (instrument output buffer ~64 KB)
        CHUNK_SIZE = 500  # Points per read — safe for all transfer modes

        all_currents = []
        all_timestamps = []

        for start_idx in range(1, n_to_read + 1, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE - 1, n_to_read)
            curr_str = instrument.query(
                f"printbuffer({start_idx}, {end_idx}, {read_ch}.nvbuffer1.readings)"
            ).strip()
            time_str = instrument.query(
                f"printbuffer({start_idx}, {end_idx}, {read_ch}.nvbuffer1.timestamps)"
            ).strip()
            all_currents.extend(float(x) for x in curr_str.split(','))
            all_timestamps.extend(float(x) for x in time_str.split(','))

        currents = all_currents
        timestamps = all_timestamps

        # Per-pulse timing analysis from buffer timestamps (zero execution
        # overhead — timestamps are already collected by the instrument and
        # already retrieved above).
        #
        # Two reporting tiers:
        #
        #   1. ALWAYS print a one-line summary of mean / max / total drift
        #      whenever the total drift exceeds the noise floor (0.02 ms).
        #      This way short-pulse experiments see *any* slip, not just
        #      drift large enough to trip the per-pulse threshold.
        #
        #   2. Per-pulse breakdown when individual cycles exceed an
        #      event-scaled threshold = clamp(0.25 × min_event_ms, 0.02 ms,
        #      1.0 ms). The 0.02 ms floor is the realistic 2636B TSP
        #      timestamp noise floor; the 1.0 ms cap prevents false fires
        #      on long-period experiments.
        # Machine-readable timing record, returned with the data. Previously
        # every number below was printed and then discarded, so nothing
        # downstream — not the results dict, not the CSV header, not the GUI —
        # could say whether the requested period had actually been delivered.
        # period_delivered is only ever True on evidence: a run whose period
        # could not be checked is reported as NOT delivered and unverified.
        drift_report = {'status': 'not_analysed', 'period_delivered': False,
                        'period_verified': False}
        median_ok = None
        try:
            if analysis == 'samples':
                if not requested_period_s:
                    raise ValueError("analysis='samples' needs requested_period_s")
                n_pulses = None
                first = 0
                expected_period_s = requested_period_s
            elif analysis == 'pulses':
                n_pulses = n_expected_points
                first = 0
                expected_period_s = (expected_time_s / n_pulses) if expected_time_s else None
            elif analysis == 'pulse_read':
                n_pulses = n_expected_points - 1
                # Skip the first interval. timestamps[0] is the initial read,
                # which happens a few ms after timer.reset() and is NOT a full
                # stim_period away from timestamps[1]. Including it pulls the
                # reported mean toward a large negative value that does not
                # reflect actual per-cycle timing.
                first = 1
                expected_period_s = (expected_time_s / n_pulses) if (expected_time_s and n_pulses > 0) else None
            else:
                raise ValueError(f"unknown timing analysis {analysis!r}")

            if expected_period_s and len(timestamps) - first >= 2:
                # Noise floor 0.1 ms matches typical 2636B TSP cycle jitter
                # (command dispatch, filter state updates). A threshold below
                # that reports instrument overhead, not real drift.
                if analysis == 'samples':
                    drift_threshold_ms = 0.1
                elif min_event_ms and min_event_ms > 0:
                    drift_threshold_ms = max(0.1, min(1.0, 0.25 * min_event_ms))
                else:
                    drift_threshold_ms = 0.25

                measured_periods_ms = [
                    (timestamps[i+1] - timestamps[i]) * 1000.0
                    for i in range(first, len(timestamps) - 1)
                ]
                drifts_ms = [p - expected_period_s * 1000.0 for p in measured_periods_ms]

                total_drift_ms = sum(drifts_ms)
                max_drift_ms = max(drifts_ms)
                mean_drift_ms = total_drift_ms / len(drifts_ms)
                drifted = [(i+1, d) for i, d in enumerate(drifts_ms)
                           if d > drift_threshold_ms]

                # Delivered timing, measured from the instrument's own
                # timestamps — the authority on what the device actually saw.
                sorted_ms = sorted(measured_periods_ms)
                _mid = len(sorted_ms) // 2
                median_period_ms = (
                    sorted_ms[_mid] if len(sorted_ms) % 2
                    else 0.5 * (sorted_ms[_mid - 1] + sorted_ms[_mid])
                )
                median_ok = median_period_ms <= expected_period_s * 1000.0 + drift_threshold_ms
                drift_report = {
                    'status': 'ok',
                    'requested_period_ms': expected_period_s * 1000.0,
                    'delivered_period_ms_median': median_period_ms,
                    'mean_drift_ms_per_cycle': mean_drift_ms,
                    'max_drift_ms': max_drift_ms,
                    'total_drift_ms': total_drift_ms,
                    'threshold_ms': drift_threshold_ms,
                    'n_pulses_exceeding_threshold': len(drifted),
                    'n_pulses': n_pulses if n_pulses is not None else len(measured_periods_ms),
                    'n_intervals': len(measured_periods_ms),
                    'period_delivered': False, 'period_verified': False,
                }

                if analysis == 'samples':
                    # Gaps next to pulse edges (a sample that would straddle an
                    # edge is skipped) are expected; the MEDIAN spacing is the
                    # delivered sampling interval.
                    if not median_ok:
                        print(f"⚠️  Delivered sample spacing {median_period_ms:.3f} ms "
                              f"(median) is longer than the requested "
                              f"{expected_period_s * 1000.0:.3f} ms.")
                else:
                    # Tier 1: always-on summary if any meaningful slip occurred
                    if abs(total_drift_ms) > 0.02 or max_drift_ms > 0.02:
                        print(f"ℹ️  Drift summary over {len(drifts_ms)} periods: "
                              f"mean = {mean_drift_ms:+.3f} ms/cycle, "
                              f"max = {max_drift_ms:+.3f} ms, "
                              f"total = {total_drift_ms:+.2f} ms "
                              f"(alert threshold = {drift_threshold_ms:.3f} ms)")

                    # Tier 2: per-pulse breakdown when any cycle crosses threshold
                    if drifted:
                        worst_pulse, worst_ms = max(drifted, key=lambda x: x[1])
                        print(f"⚠️  {len(drifted)}/{len(drifts_ms)} periods exceeded "
                              f"{drift_threshold_ms:.3f} ms drift threshold:")
                        if len(drifted) <= 5:
                            for pulse_num, d in drifted:
                                print(f"      Pulse {pulse_num}: +{d:.3f} ms")
                        else:
                            print(f"      worst: pulse {worst_pulse} at +{worst_ms:.3f} ms")
        except Exception as e:
            # "We could not check" must never render as "we checked and it was
            # fine". This block used to swallow its own deliberate guard, so a
            # failed drift analysis produced total silence and the user read
            # the absence of a warning as confirmation of good timing.
            drift_report = {'status': 'analysis_failed', 'error': repr(e),
                            'period_delivered': False, 'period_verified': False}
            median_ok = None
            print(f"⚠️  Drift analysis FAILED: {e!r}\n"
                  f"    Timing quality is UNKNOWN for this run — the absence of a "
                  f"drift warning below does NOT mean the timing was good.")

        # The instrument's own overrun count, if the script reported one.
        instrument_ok = None
        if lua_timing:
            drift_report['instrument_report'] = dict(lua_timing)
            n_over = lua_timing.get('overruns')
            if isinstance(n_over, float):
                instrument_ok = (n_over == 0)
            if isinstance(n_over, float) and n_over > 0:
                print(f"⚠️  {int(n_over)}/{int(lua_timing.get('n_pulses', 0))} pulses could "
                      f"NOT meet the requested period (worst shortfall "
                      f"{lua_timing.get('worst_ms', float('nan')):.3f} ms).\n"
                      f"    The train ran at the hardware limit. The delivered period is "
                      f"longer than requested; use the buffer timestamps, not the nominal "
                      f"value, when reporting this measurement.")

        verdicts = [v for v in (instrument_ok, median_ok) if v is not None]
        drift_report['period_verified'] = bool(verdicts)
        drift_report['period_delivered'] = bool(verdicts) and all(verdicts)
        if median_ok is False and instrument_ok is not False and analysis != 'samples':
            print("⚠️  The requested period was NOT delivered (median delivered "
                  f"period {drift_report['delivered_period_ms_median']:.3f} ms vs "
                  f"{drift_report['requested_period_ms']:.3f} ms requested).")

        # Check for errors (ignoring benign ones if any remain)
        try:
            err_cnt_str = instrument.query("print(errorqueue.count)").strip()
            err_count = int(float(err_cnt_str))
            if err_count > 0:
                print(f"⚠️  Warning: {err_count} instrument errors:")
                errs = []
                for _ in range(min(5, err_count)):
                    errs.append(instrument.query('print(errorqueue.next())').strip())
                    print(f"   {errs[-1]}")
                instrument.write("errorqueue.clear()")
                drift_report['instrument_errors'] = errs
        except Exception as e:
            # Same rule: an unreadable error queue is not an empty error queue.
            drift_report['instrument_errors'] = {'status': 'unreadable', 'error': repr(e)}
            print(f"⚠️  Could not read the instrument error queue: {e!r} — "
                  f"instrument error state is UNKNOWN for this run.")

        # Cleanup (anonymous script is auto-discarded, just clear buffer)
        instrument.write(f"{read_ch}.nvbuffer1.clear()")

        # The timing record travels via the caller-supplied out-dict rather
        # than the return tuple, so the (timestamps, currents) contract every
        # existing caller relies on is unchanged.
        if timing_out is not None:
            timing_out.clear()
            timing_out.update(drift_report)

        return timestamps, currents

    except Exception as e:
        _abort_outputs(instrument)
        if started and not isinstance(e, LuaScriptStartedError):
            reached, errs = _script_fault_report(instrument, read_ch)
            raise LuaScriptStartedError(
                f"The LUA script was already running when it failed ({e!r}); the "
                f"read buffer had reached {reached} of {n_expected_points} points. "
                f"Instrument error queue: {errs if errs else 'empty'}. The device may "
                "be partly stimulated, so the measurement was NOT re-run on another "
                "path; check the device state and repeat the run deliberately."
            ) from e
        raise


# =============================================================================
# INSTRUMENT MODEL VALIDATION
# =============================================================================
#
# SINGLE SOURCE OF TRUTH for instrument capabilities.
#
# Every capability query below reads this one table. Previously the same
# knowledge was spelled out four times, in four inconsistent hardcoded lists,
# and they disagreed: the LUA allowlist recognised only 2602/2612/2636 while the
# voltage and current tables recognised ten models. On a 2635B that mismatch
# silently routed pulse-read onto the slower PC-timed path (which measures a
# different quantity — see pulse_read_sequence) and hard-failed visual and cycle
# modes. Adding a model here now updates every capability at once.
#
# Sources: Keithley 2600A/2600B series specifications.
#   Low voltage  (26x1/26x2/26x4 ...B): 40 V,  3.0 A DC
#   High voltage (2611/2612/2614):      200 V, 1.5 A DC
#   High power   (2634/2635/2636):      200 V, 1.5 A DC
# All 2600-series models embed the TSP engine and therefore run LUA scripts.
# All support a 10 A pulse range.

_LOW_VOLTAGE = {'max_voltage': 40.0, 'max_current_dc': 3.0, 'pulse_10a': True, 'lua': True}
_HIGH_VOLTAGE = {'max_voltage': 200.0, 'max_current_dc': 1.5, 'pulse_10a': True, 'lua': True}

MODEL_CAPABILITIES = {
    # Low-voltage family (40 V, 3 A DC)
    '2601A': _LOW_VOLTAGE, '2601B': _LOW_VOLTAGE,
    '2602A': _LOW_VOLTAGE, '2602B': _LOW_VOLTAGE,
    '2604A': _LOW_VOLTAGE, '2604B': _LOW_VOLTAGE,
    # High-voltage family (200 V, 1.5 A DC)
    '2611A': _HIGH_VOLTAGE, '2611B': _HIGH_VOLTAGE,
    '2612A': _HIGH_VOLTAGE, '2612B': _HIGH_VOLTAGE,
    '2614A': _HIGH_VOLTAGE, '2614B': _HIGH_VOLTAGE,
    # High-power family (200 V, 1.5 A DC)
    '2634A': _HIGH_VOLTAGE, '2634B': _HIGH_VOLTAGE,
    '2635A': _HIGH_VOLTAGE, '2635B': _HIGH_VOLTAGE,
    '2636A': _HIGH_VOLTAGE, '2636B': _HIGH_VOLTAGE,
}


def _highest_dc_current_A():
    """Highest DC current ceiling in the capability table."""
    return max(c['max_current_dc'] for c in MODEL_CAPABILITIES.values())


def _models_at_highest_dc_current():
    """Model names achieving that ceiling, sorted."""
    top = _highest_dc_current_A()
    return sorted(m for m, c in MODEL_CAPABILITIES.items()
                  if c['max_current_dc'] == top)


class InstrumentIdentificationError(RuntimeError):
    """Raised when the connected instrument cannot be identified.

    This is deliberately fatal. Guessing a capability is a safety decision:
    assuming the 200 V family on a 40 V instrument invites damage, and assuming
    the 40 V family on a 200 V instrument writes non-native ranges (SCPI -222).
    Silently substituting a "conservative default" also violates the project
    ground rule that nothing may be quietly substituted for a value that could
    not be determined.
    """


def validate_instrument_model(instrument):
    """Return the raw *IDN? string of the connected instrument.

    Raises InstrumentIdentificationError if the instrument cannot be queried.
    """
    try:
        return instrument.query("*IDN?")
    except Exception as e:
        raise InstrumentIdentificationError(
            f"Could not query *IDN? from the instrument: {e}\n"
            "The model determines the voltage and current limits and whether "
            "LUA execution is available, so the measurement cannot safely "
            "continue without it. Check the connection and address."
        )


def get_model_capabilities(instrument):
    """Resolve the capability record for the connected instrument.

    Returns a dict with keys: model, idn, max_voltage, max_current_dc,
    pulse_10a, lua.

    The *IDN? result is cached on the instrument object, so the several
    capability queries a single measurement performs cost one VISA round trip
    rather than four.

    Raises InstrumentIdentificationError if the model is not in
    MODEL_CAPABILITIES.
    """
    cached = getattr(instrument, '_synapsys_capabilities', None)
    if cached is not None:
        return cached

    idn = validate_instrument_model(instrument)
    idn_upper = idn.upper()

    for model, caps in MODEL_CAPABILITIES.items():
        if model in idn_upper:
            resolved = dict(caps, model=model, idn=idn.strip())
            try:
                instrument._synapsys_capabilities = resolved
            except Exception:
                pass  # Resource forbids attribute assignment; query each time.
            return resolved

    raise InstrumentIdentificationError(
        f"Unrecognised instrument: {idn.strip()}\n"
        f"Known models: {', '.join(sorted(MODEL_CAPABILITIES))}.\n"
        "Voltage and current limits cannot be assumed for an unknown model. "
        "Add it to synapse_engine.MODEL_CAPABILITIES with its specifications "
        "before using it."
    )


def supports_lua_execution(instrument):
    """True if the instrument runs on-instrument LUA (TSP) scripts.

    Raises InstrumentIdentificationError for an unidentifiable instrument
    rather than returning False: a comms failure previously disabled LUA
    silently, which changed what the measurement recorded (see
    pulse_read_sequence) with no indication in the data.
    """
    return get_model_capabilities(instrument)['lua']


def check_10a_range_support(instrument):
    """True if the instrument supports the 10 A pulse range."""
    return get_model_capabilities(instrument)['pulse_10a']


def get_max_current(instrument):
    """Maximum DC current capability in amperes (10 A pulse is separate)."""
    return get_model_capabilities(instrument)['max_current_dc']


def get_max_voltage(instrument):
    """Maximum voltage capability in volts."""
    return get_model_capabilities(instrument)['max_voltage']

# =============================================================================
# STEP 1: HELPER FUNCTIONS
# =============================================================================

def _assert_gpib_ifc(rm):
    """
    Asserts Interface Clear on every available GPIB board.

    IFC must be issued through an interface-level session (the resource
    string "GPIB{n}::INTFC"), not through the default ResourceManager
    session — that's what produces VI_ERROR_INV_OBJECT. After IFC the
    board rescans the bus, which rebuilds enumeration for devices that
    weren't present or responsive at driver startup.

    Returns the list of boards on which IFC was successfully asserted.
    """
    touched = []
    try:
        interfaces = list(rm.list_resources('GPIB?*::INTFC'))
    except Exception:
        interfaces = []

    # Fallback: if enumeration of INTFC resources fails (some NI-VISA
    # versions hide them until first use), try GPIB0 and GPIB1 explicitly.
    if not interfaces:
        interfaces = ['GPIB0::INTFC', 'GPIB1::INTFC']

    for intf in interfaces:
        iface = None
        try:
            iface = rm.open_resource(intf)
            iface.send_ifc()
            touched.append(intf)
            print(f"  [GPIB recovery] IFC asserted on {intf}")
        except Exception as e:
            # Non-existent boards raise here; that's fine, skip them.
            print(f"  [GPIB recovery] IFC on {intf} skipped: {type(e).__name__}: {e}")
        finally:
            if iface is not None:
                try:
                    iface.close()
                except Exception:
                    pass
    return touched


def _open_gpib_with_recovery(port_str, max_attempts=4, probe_timeout_ms=2000):
    """
    Opens a GPIB resource with active recovery from stale bus enumeration.

    Symptom this fixes: after the Keithley was powered off, reset, or
    otherwise absent when the NI-488.2 driver enumerated the bus at
    startup, the VISA resource list does not contain GPIB::N::INSTR and
    open_resource() fails with VI_ERROR_RSRC_NFOUND — until the USB-GPIB
    adapter is unplugged and replugged, which re-enumerates.

    Recovery strategy per attempt: open the GPIB interface resource,
    call send_ifc() on it (real software replug), wait for the controller
    to rescan, then re-list resources and try to open the device. On
    success, issue SDC via instrument.clear() and probe *IDN?. A fresh
    ResourceManager is used each attempt.

    Assumes the GPIB bus is dedicated to the Keithley (IFC resets every
    device on the bus).

    Raises ValueError on final failure with the full attempt history.
    """
    resource_string = f"GPIB::{port_str}::INSTR"
    failures = []

    for attempt in range(1, max_attempts + 1):
        rm = None
        instrument = None
        try:
            rm = pyvisa.ResourceManager()

            # If the device is already enumerated, skip IFC on the first
            # attempt — IFC briefly disrupts the bus and is unnecessary
            # when enumeration is already healthy.
            try:
                enumerated = resource_string in rm.list_resources()
            except Exception:
                enumerated = False

            if not enumerated or attempt > 1:
                _assert_gpib_ifc(rm)
                # NI-488.2 needs ~300-500 ms after IFC to rescan the bus
                # and rebuild the device list. Scale with attempt count.
                time.sleep(0.3 + 0.2 * attempt)

            instrument = rm.open_resource(resource_string)
            instrument.timeout = probe_timeout_ms

            # SDC — Selected Device Clear. Resets the device's interface
            # state without resetting its source state.
            try:
                instrument.clear()
            except Exception as clr_err:
                print(f"  [GPIB recovery] device clear failed: {clr_err}")

            instrument.write("*CLS")
            idn = instrument.query("*IDN?").strip()
            if not idn:
                raise ValueError("empty *IDN? response")

            print(f"[GPIB] connected on attempt {attempt}: {idn}")
            return instrument

        except Exception as e:
            failures.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            print(f"  [GPIB recovery] {failures[-1]}")
            if instrument is not None:
                try:
                    instrument.close()
                except Exception:
                    pass
            if rm is not None:
                try:
                    rm.close()
                except Exception:
                    pass
            # Back off before the next attempt so the driver can settle.
            time.sleep(0.5 * attempt)

    raise ValueError(
        f"Could not establish GPIB communication on {resource_string} "
        f"after {max_attempts} attempts. History:\n  " + "\n  ".join(failures)
    )


def open_instrument(port_str, connection_type):
    """
    Opens the pyvisa instrument resource based on connection type.

    For GPIB, uses active recovery (interface-level IFC + bus rescan +
    SDC + *IDN? probe with retries) to survive stale NI-488.2 bus
    enumeration that otherwise requires unplugging the USB-GPIB adapter.

    Args:
        port_str (str): Port address/identifier
        connection_type (str): One of "GPIB", "RS232", or "LAN"

    Returns:
        pyvisa.Resource: Opened, probed instrument resource

    Raises:
        ValueError: If connection cannot be established
    """
    if connection_type == "GPIB":
        return _open_gpib_with_recovery(port_str)

    rm = pyvisa.ResourceManager()
    instrument = None

    if connection_type == "RS232":
        instrument = rm.open_resource(
            port_str,
            baud_rate=9600,
            data_bits=8,
            parity=pyvisa.constants.Parity.none,
            stop_bits=pyvisa.constants.StopBits.one,
            flow_control=pyvisa.constants.VI_ASRL_FLOW_NONE
        )
    elif connection_type == "LAN":
        instrument = rm.open_resource(f"TCPIP::{port_str}::INSTR")
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

    if instrument is None:
        raise ValueError("Could not establish communication with the instrument.")

    return instrument


def validate_float(entry_value, field_name):
    """
    Validates and converts entry to float.
    
    Args:
        entry_value: Value to validate
        field_name (str): Name of field for error messages
    
    Returns:
        float: Validated float value
    
    Raises:
        ValueError: If conversion fails
    """
    try:
        return float(entry_value)
    except ValueError:
        raise ValueError(f"Invalid value for {field_name}. Please enter a valid number.")


# Default confirmation thresholds, per drive type. These are NOT instrument
# ceilings (those live in MODEL_CAPABILITIES) — they are the levels above which
# a typical memristive test device is at risk and the user is asked to confirm.
MAX_SAFE_STIM_VOLTAGE_V = 2.5
MAX_SAFE_STIM_CURRENT_A = 1e-3


def safety_check(stim_level, max_safe_voltage=MAX_SAFE_STIM_VOLTAGE_V,
                 drive_type='V', max_safe_current=MAX_SAFE_STIM_CURRENT_A):
    """
    Performs safety check on stimulus level, in the units actually being driven.

    Args:
        stim_level (float): Stimulus level, in V when drive_type='V' and in
            AMPERES when drive_type='I'.
        max_safe_voltage (float): Confirmation threshold for voltage drive (V)
        drive_type (str): 'V' (voltage source) or 'I' (current source)
        max_safe_current (float): Confirmation threshold for current drive (A)

    Returns:
        bool: True if safe to proceed, False if user cancels

    The drive type matters, and getting it wrong is a device-destroying bug.
    This function used to compare every level against the 2.5 V threshold and
    phrase the warning in volts REGARDLESS of drive type. In current-source
    mode the GUI's default stimulus level of 1.0 therefore meant one AMPERE,
    which passed a 2.5 "V" check unchallenged — and the global "Compliance (A)"
    field is a current limit that does nothing when current is the source
    quantity (the engine sets `limitv` instead). The only remaining guard was
    the instrument ceiling of 1.5 A, which 1 A clears comfortably.
    """
    drive = str(drive_type).upper()
    if drive == 'I':
        if abs(stim_level) > max_safe_current:
            return messagebox.askyesno(
                "Safety Warning — CURRENT drive",
                f"Stimulus CURRENT ({stim_level:g} A = {stim_level*1e3:g} mA) "
                f"exceeds the recommended safe value of "
                f"{max_safe_current:g} A ({max_safe_current*1e3:g} mA).\n\n"
                "The instrument is sourcing CURRENT, so the 'Compliance (A)' "
                "setting does not limit it — compliance bounds the VOLTAGE in "
                "this mode. This level can destroy a memristive device "
                "immediately.\n\n"
                "Do you want to proceed?"
            )
        return True

    if abs(stim_level) > max_safe_voltage:
        return messagebox.askyesno(
            "Safety Warning — VOLTAGE drive",
            f"Stimulus VOLTAGE ({stim_level:g} V) exceeds the recommended safe "
            f"value ({max_safe_voltage:g} V).\n"
            "This may damage sensitive devices.\n\n"
            "Do you want to proceed?"
        )
    return True


# =============================================================================
# STEP 3: CORE ENGINE IMPLEMENTATION
# =============================================================================

def pulse_read_sequence(instrument, stim_ch, read_ch, params):
    """
    Core engine for electrical, visual, and memristor pulse measurements.

    Performs a sequence of stimulus pulses followed by read measurements to
    characterize synaptic devices.

    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Stimulus channel ('smua' or 'smub')
        read_ch (str): Read channel ('smua' or 'smub')
        params (dict): Measurement parameters containing:
            - mode (str): 'electrical', 'visual', or 'memristor_pulse'
            - stim_drive_type (str): 'V' or 'I'
            - stim_level (float): Stimulus amplitude in V or A
            - stim_width_ms (float): Pulse width in milliseconds
            - stim_period_ms (float): Pulse period in milliseconds
            - n_pulses (int): Number of pulses
            - read_voltage (float): Read bias voltage in V
            - read_delay_ms (float): Delay after pulse before read (ms)
            - compliance_A (float): Current compliance in A
            - settle_ms (float, optional): Extra settling time (default: 5)
            - measure_avg (int, optional): Number of measurements to average (default: 1)
            - wire_mode (str, optional): '2-Wire' or '4-Wire' (default: None)

    Returns:
        dict: Results containing:
            - pulse_number: List of pulse indices
            - time_s: List of timestamps in seconds
            - I_A: List of measured currents in A
            - V_read_V: List of read voltages in V
            - conductance_S: List of conductance values in S
            - params: Copy of input parameters
            - timing: what the run actually delivered, including
              `period_delivered` (bool), the delivered median period and
              `settle_warning` (None, or why the readings are not settled)

    Raises:
        LuaScriptStartedError: the on-instrument script failed after it had
            started; the measurement is deliberately NOT re-run.
    """
    _check_channels(stim_ch, read_ch)

    results = {
        "pulse_number": [],
        "time_s": [],
        "I_A": [],
        "V_read_V": [],
        "conductance_S": [],
        "params": params.copy(),
        "metadata": {
            "wavelength_nm": params.get('wavelength_nm', None),
            "intensity_mW_cm2": params.get('intensity_mW_cm2', None),
            "sample_id": params.get('sample_id', 'unknown'),
            "measurement_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "operator": params.get('operator', 'unknown'),
            "device_area_cm2": params.get('device_area_cm2', None)
        }
    }

    # Extract parameters with defaults
    nplc = params.get('nplc', 1.0)

    # === INSTRUMENT CAPABILITY CHECK ===
    # Check if instrument supports requested current level
    if params['stim_drive_type'] == 'I':  # Current sourcing mode
        requested_current = abs(params['stim_level'])
        max_current = get_max_current(instrument)

        if requested_current > max_current:
            model = validate_instrument_model(instrument)
            raise ValueError(
                f"ERROR: Requested current {requested_current}A exceeds instrument capability.\n"
                f"Connected instrument: {model}\n"
                f"Maximum current: {max_current}A\n"
                f"Models reaching {_highest_dc_current_A():g} A DC: "
                f"{', '.join(_models_at_highest_dc_current())}.\n"
                f"Note the 10 A range is PULSE-only and is not used by this "
                f"measurement path. Reduce the current level, or use one of the "
                f"models listed above."
            )

    # Record instrument voltage ceiling for the LUA range picker before either
    # execution path runs. Without this the generator falls back to its default
    # (10 V) and picks 40 V family ranges on a 200 V model, triggering -222.
    params['max_voltage'] = get_max_voltage(instrument)

    # Adopt the instrument's auto-detected mains frequency (unless the caller
    # specified one) BEFORE any path derives NPLC timing from it. results
    # carries a copy of params taken above, so stamp the resolved value there
    # too — the exported header must record the frequency actually used.
    resolve_line_freq(instrument, params)
    results['params']['line_freq_hz'] = params['line_freq_hz']

    # === FAST LUA EXECUTION PATH (TRANSPARENT TO USER) ===
    # On-instrument LUA execution eliminates PC communication delays during the
    # pulse sequence. Every mode handled by this function is a stimulus pulse
    # followed by a read, which is exactly what generate_pulse_read_lua_script
    # emits, so the same script serves all of them:
    #   electrical      — stim channel drives the device electrode
    #   visual          — stim channel drives an LED; electrically identical to
    #                     the above from the SMU's point of view
    #   memristor_pulse — same pulse-read sequence, memristive device
    #   srdp            — same sequence repeated at different frequencies
    # Keep this tuple in sync with keithley_analyser.SYNAPSE_MODE_BY_LABEL.
    LUA_PULSE_READ_MODES = ('electrical', 'visual', 'memristor_pulse', 'srdp')

    if params['mode'] not in LUA_PULSE_READ_MODES:
        raise ValueError(
            f"pulse_read_sequence received unknown mode '{params['mode']}'.\n"
            f"Expected one of: {', '.join(LUA_PULSE_READ_MODES)}.\n"
            "A mode that is not dispatched here would fall through to the "
            "standard execution path and be measured differently without any "
            "record of it. Register the mode in LUA_PULSE_READ_MODES (and in "
            "keithley_analyser.SYNAPSE_MODE_BY_LABEL) before using it."
        )

    # Both paths record the same quantity — see the standard path below, which
    # mirrors the LUA script's baseline read, fixed ranging and averaging. The
    # path that actually ran is still stamped into the results and exported with
    # the data: the timing source differs, so provenance matters even though the
    # measured quantity does not change.
    use_lua = supports_lua_execution(instrument)

    if use_lua:
        print("Using fast LUA execution on instrument...")

        # Wire mode must match physical cabling for voltage to be sourced.
        _apply_wire_modes(instrument, params, [read_ch, stim_ch])

        # Generate and execute LUA script
        lua_script = generate_pulse_read_lua_script(stim_ch, read_ch, params)
        n_expected_points = params['n_pulses'] + 1  # Initial read + pulses
        expected_time_s = params['stim_period_ms'] * params['n_pulses'] / 1000.0
        # A period below the generator's minimum cannot be delivered; the run
        # then takes n x minimum, and waiting only for the nominal time would
        # time out a healthy run (5 ms x 5000 takes ~77 s against 62.5 s).
        feasible_time_s = (max(params['stim_period_ms'], lua_script.facts['min_period_ms'])
                           * params['n_pulses'] / 1000.0)

        # Smallest event in the cycle — used to scale the post-hoc drift
        # threshold so a 0.3 ms read pulse alerts at ~0.15 ms drift instead
        # of waiting for a blanket 0.5 ms.
        NPLC_PERIOD_MS_LOCAL = 1000.0 / _line_freq_hz(params)
        rw_ms_local = params.get('read_width_ms')
        meas_event_ms = (
            rw_ms_local if (rw_ms_local and rw_ms_local > 0)
            else nplc * NPLC_PERIOD_MS_LOCAL
        )
        candidate_events = [
            params.get('stim_width_ms', 0),
            params.get('read_delay_ms', 0),
            params.get('read_settle_delay_ms', 0),
            meas_event_ms,
        ]
        positive_events = [e for e in candidate_events if e and e > 0]
        min_event_ms_local = min(positive_events) if positive_events else None

        # Only the instrument interaction is guarded. Result processing and
        # console output are deliberately OUTSIDE this try: an exception there
        # is a software fault, not an instrument fault, and silently converting
        # it into "LUA failed → fall back" would re-run the whole measurement
        # on the other path and blame the instrument.
        #
        # Fallback is only for failures BEFORE the script ran (upload,
        # capability). Once `endscript` was sent the device may already be
        # partly stimulated, and a re-run would start from that altered state
        # and report it as the baseline: LuaScriptStartedError propagates.
        lua_data = None
        lua_timing_report = {}
        try:
            lua_data = execute_lua_script_fast(
                instrument, lua_script, read_ch, n_expected_points,
                expected_time_s=expected_time_s,
                feasible_time_s=feasible_time_s,
                min_event_ms=min_event_ms_local,
                timing_out=lua_timing_report,
                n_is_exact=True,
            )
        except LuaScriptStartedError:
            raise
        except Exception as e:
            print(f"WARNING: LUA execution failed: {e}")
            print("Falling back to standard execution...")
            # Fall through to standard execution below

        if lua_data is not None:
            timestamps, currents = lua_data

            # Process results into standard format
            t_start = timestamps[0]
            read_voltage = params['read_voltage']

            for i, (timestamp, current) in enumerate(zip(timestamps, currents)):
                pulse_num = i  # 0 is the pre-stimulus baseline, then pulses 1..N
                t_elapsed = timestamp - t_start
                # Conductance is undefined at zero read bias. NaN says so;
                # 0 would be a fabricated measurement that plots and averages
                # as if it were real.
                G_measured = current / read_voltage if read_voltage != 0 else float('nan')

                results['pulse_number'].append(pulse_num)
                results['time_s'].append(t_elapsed)
                results['I_A'].append(current)
                results['V_read_V'].append(read_voltage)
                results['conductance_S'].append(G_measured)

                # Display progress for key points
                if i == 0:
                    print(f"Initial G = {G_measured*1e6:.2f} uS")
                elif i == len(timestamps) - 1:
                    print(f"Final G = {G_measured*1e6:.2f} uS (pulse {pulse_num})")

            print(f"Completed {params['n_pulses']} pulses via LUA")
            results['execution_path'] = 'lua'
            # What the instrument actually delivered, alongside what was asked
            # for. Previously every one of these numbers was printed and then
            # thrown away, so a train that could not hold its requested period
            # exported under that period with nothing recording the shortfall.
            results['timing'] = dict(lua_timing_report)
            results['timing']['min_period_ms'] = lua_script.facts['min_period_ms']
            results['timing']['settle_warning'] = lua_script.facts['settle_warning']
            return results


    # --- STANDARD EXECUTION PATH ---
    #
    # Reached when the instrument cannot run LUA, or when LUA failed BEFORE the
    # script ran (see above). It is deliberately a faithful software mirror of
    # generate_pulse_read_lua_script: same baseline read, same fixed ranging,
    # same ADC state, same averaging derivation, same conductance convention.
    # Only the timing source differs (host clock instead of the instrument's
    # timer), which is unavoidable off-instrument and is the reason LUA is
    # preferred.
    #
    # These two paths MUST stay equivalent. When they diverged, the same GUI
    # settings produced different physical quantities depending on which one
    # ran — pulse index 0 meant "baseline" on one and "after the first pulse"
    # on the other, so calculate_synapse_metrics reported PPF as G1/G0 in one
    # case and G2/G1 in the other, with nothing in the data recording which.

    # Stamped here rather than before the LUA attempt, so that a LUA run which
    # failed and fell through is recorded as what it actually was.
    results['execution_path'] = 'standard'

    # Same validation and defaults the LUA generator applies.
    plan = _pulse_read_plan(stim_ch, read_ch, params)
    if plan['settle_warning']:
        print(f"⚠️  {plan['settle_warning']}")

    # --- CONFIGURATION PHASE ---

    # 0. Force safe state before configuration
    _safe_state(instrument)

    # 0a. Synchronise ADC to mains for normal-mode rejection of line-frequency
    #     pickup (matches the LUA path's localnode.linefreq).
    instrument.write(f"localnode.linefreq = {int(_line_freq_hz(params))}")

    # 0b. Configure sense mode (MUST be before channel configuration)
    _apply_wire_modes(instrument, params, [read_ch, stim_ch])

    # Same derivation the LUA generator uses, so the GUI's NPLC / Read Pulse
    # Width / Samples per Pulse controls mean the same thing on both paths.
    nplc, n_samples = plan['nplc'], plan['measure_avg']
    fixed_i_range = plan['fixed_i_range']
    read_voltage = params['read_voltage']
    stim_level_attr = 'levelv' if params['stim_drive_type'] == 'V' else 'leveli'
    read_delay_s = plan['read_delay_ms'] / 1000.0
    read_settle_s = plan['read_settle_delay_ms'] / 1000.0

    # 1. Configure Read Channel (for low-level DC bias and measurement).
    #    Fixed ranging, matching the LUA path: autorange would make the
    #    per-pulse measurement time vary with the signal, which destroys the
    #    period reproducibility this measurement depends on.
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.rangev = {_get_fixed_voltage_range(read_voltage, params['max_voltage'])}")
    instrument.write(f"{read_ch}.source.limiti = {params['compliance_A']}")
    instrument.write(f"{read_ch}.measure.rangei = {fixed_i_range}")
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF")
    # The complete ADC state — delay, nplc, count, filter, autozero — exactly as
    # the LUA script sets it. Writing the filter only when averaging > 1 let a
    # dirty ON/8 state from an earlier run leak into this one.
    for line in _emit_read_adc_setup(read_ch, nplc, n_samples):
        instrument.write(line)
    if n_samples > 1:
        # Ceiling is the hardware limit, shared with the LUA path.
        n_samples_clamped = max(2, min(n_samples, FILTER_COUNT_MAX))
        print(f"ℹ️  Hardware averaging: {n_samples_clamped} samples")

    # 2. Configure Stimulus Channel (fixed ranges, as in the LUA path)
    if params['stim_drive_type'] == 'V':
        instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS")
        instrument.write(f"{stim_ch}.source.levelv = 0")
        instrument.write(f"{stim_ch}.source.rangev = {_get_fixed_voltage_range(params['stim_level'], params['max_voltage'])}")
        instrument.write(f"{stim_ch}.source.limiti = {params['compliance_A']}")
    else:
        instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS")
        instrument.write(f"{stim_ch}.source.leveli = 0")
        # The sourced current's range, not the compliance's.
        instrument.write(f"{stim_ch}.source.rangei = {get_fixed_current_range(abs(params['stim_level']))}")
        # Voltage compliance from the instrument's actual ceiling, as the LUA
        # path does. A hardcoded 10 V here silently clamped current-driven
        # stimuli on a 200 V instrument.
        instrument.write(f"{stim_ch}.source.limitv = {params['max_voltage']}")

    # 3. Both outputs ON once, at level 0; pulses are level changes only (as in
    #    the LUA script — see there for why OUTPUT_OFF is not toggled per pulse).
    instrument.write(f"{read_ch}.source.levelv = 0")
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
    time.sleep(plan['settle_ms'] / 1000.0)
    instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")

    # === BUFFER CLEARING: Clear only what's necessary ===
    instrument.write("*CLS")  # Clear instrument error queue
    time.sleep(0.1)  # Give instrument time to process

    # --- MEASUREMENT LOOP ---

    def _read_conductance():
        """Apply the read bias, measure, return to 0 V. Returns (I_A, G_S).

        Mirrors the LUA script's read block: bias, settle, one averaged
        measurement, back to 0 V idle.
        """
        instrument.write(f"{read_ch}.source.levelv = {read_voltage}")
        time.sleep(read_settle_s)
        I = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        instrument.write(f"{read_ch}.source.levelv = 0")
        # Conductance is undefined at zero read bias; NaN says so rather than
        # fabricating a value that would average and plot as if measured.
        G = I / read_voltage if read_voltage != 0 else float('nan')
        return I, G

    t0 = time.time()
    pulse_starts = []
    n_overruns = 0
    worst_overrun_ms = 0.0

    try:
        # --- INITIAL BASELINE READ (pulse_number 0) ---
        # The LUA script performs this before its pulse loop, so the buffer
        # holds n_pulses + 1 points with index 0 = the pre-stimulus state.
        # Without it here the two paths disagreed about what index 0 means,
        # which silently changed the meaning of PPF and shifted every
        # delta_G-versus-pulse-index fit downstream by one pulse.
        I_meas, G_meas = _read_conductance()
        results["pulse_number"].append(0)
        results["time_s"].append(time.time() - t0)
        results["I_A"].append(I_meas)
        results["V_read_V"].append(read_voltage)
        results["conductance_S"].append(G_meas)
        print(f"Initial G = {G_meas*1e6:.2f} µS")

        for i in range(params['n_pulses']):
            pulse_start = time.time()
            pulse_starts.append(pulse_start)

            # --- STIMULUS ON (level change; the output stays ON) ---
            instrument.write(f"{stim_ch}.source.{stim_level_attr} = {params['stim_level']}")

            # Wait for pulse duration
            time.sleep(plan['stim_width_ms'] / 1000.0)

            # --- STIMULUS OFF ---
            instrument.write(f"{stim_ch}.source.{stim_level_attr} = 0")

            # --- WAIT READ DELAY ---
            time.sleep(read_delay_s)

            # --- APPLY READ BIAS AND MEASURE CURRENT ---
            I_meas, G_meas = _read_conductance()

            # --- STORE RESULTS ---
            # Index i+1, because index 0 is the pre-stimulus baseline above.
            results["pulse_number"].append(i + 1)
            results["time_s"].append(time.time() - t0)
            results["I_A"].append(I_meas)
            results["V_read_V"].append(read_voltage)
            results["conductance_S"].append(G_meas)

            # --- WAIT REMAINDER OF PERIOD ---
            pulse_elapsed = time.time() - pulse_start
            remaining_time = (params['stim_period_ms'] / 1000.0) - pulse_elapsed

            if remaining_time > 0:
                time.sleep(remaining_time)
            else:
                n_overruns += 1
                worst_overrun_ms = max(worst_overrun_ms, -remaining_time * 1000.0)
                if remaining_time < -0.010:  # Warn if >10ms over budget
                    print(f"⚠️  Pulse {i}: Period exceeded by {-remaining_time*1000:.1f} ms")

    finally:
        # Levels to 0 BEFORE the outputs go off, and the filter off, so nothing
        # live (a 1 V stim level, filter ON/count 10) leaks into the next run.
        instrument.write(f"{stim_ch}.source.{stim_level_attr} = 0")
        instrument.write(f"{read_ch}.source.levelv = 0")
        instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        instrument.write(f"{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF")
        for line in _emit_restore_adc(read_ch):
            instrument.write(line)

    delivered_ms = [(b - a) * 1000.0 for a, b in zip(pulse_starts, pulse_starts[1:])]
    delivered_ms.sort()
    if delivered_ms:
        _mid = len(delivered_ms) // 2
        median_ms = (delivered_ms[_mid] if len(delivered_ms) % 2
                     else 0.5 * (delivered_ms[_mid - 1] + delivered_ms[_mid]))
    else:
        median_ms = float('nan')
    # Host-timed: "delivered" means no pulse's own work outlasted the period.
    results['timing'] = {
        'status': 'host_timed',
        'requested_period_ms': params['stim_period_ms'],
        'delivered_period_ms_median': median_ms,
        'n_overruns': n_overruns,
        'worst_overrun_ms': worst_overrun_ms,
        'period_delivered': n_overruns == 0,
        'period_verified': True,
        'min_period_ms': plan['min_period_ms'],
        'settle_warning': plan['settle_warning'],
    }
    if n_overruns:
        print(f"⚠️  {n_overruns}/{params['n_pulses']} pulses could NOT meet the requested "
              f"{params['stim_period_ms']:g} ms period (worst shortfall {worst_overrun_ms:.1f} ms).")

    # Check for instrument errors before returning
    try:
        error_check = instrument.query("print(errorqueue.count)")
        if int(float(error_check.strip())) > 0:
            print(f"⚠️ Warning: {error_check.strip()} errors in instrument queue")
            # Read and print errors
            for _ in range(min(5, int(float(error_check.strip())))):
                error_msg = instrument.query("print(errorqueue.next())")
                print(f"   Error: {error_msg.strip()}")
    except:
        pass  # Don't fail measurement if error check fails

    return results

# =============================================================================
# STEP 4: SIMULATION ENGINE
# =============================================================================

def simulate_pulse_read(params):
    """
    Simulates a pulse-read sequence for testing without hardware.
    Models simple exponential conductance change for LTP/LTD behavior.

    Args:
        params (dict): Same parameters as pulse_read_sequence

    Returns:
        dict: Results in same format as pulse_read_sequence: n_pulses + 1
        points, index 0 being the pre-stimulus baseline (as on the instrument).
    """

    # === SIMULATION CURRENT LIMIT CHECK ===
    # Warn if simulating beyond 2612B capability
    if params['stim_drive_type'] == 'I':
        requested_current = abs(params['stim_level'])
        if requested_current > 1.5:
            print(f"⚠️  WARNING: Simulating {requested_current}A current.")
            print(f"    This exceeds the 200 V family's 1.5 A DC capability.")
            print(f"    Models reaching {_highest_dc_current_A():g} A DC: "
                  f"{', '.join(_models_at_highest_dc_current())}.")

    results = {
        "pulse_number": [],
        "time_s": [],
        "I_A": [],
        "V_read_V": [],
        "conductance_S": [],
        "params": params.copy()
    }

    # Simulation parameters
    G_initial = 1e-6  # 1 µS initial conductance
    G_max = 1e-4      # 100 µS maximum conductance
    G_min = 1e-7      # 0.1 µS minimum conductance
    step_size = 0.05  # Learning rate (5% change per pulse)

    G_current = G_initial
    V_read = params['read_voltage']
    t0 = time.time()

    # Index 0: the pre-stimulus baseline, as the instrument paths record it
    # (the CSV header says pulse_index_0=pre_stimulus_baseline). Measurement
    # noise only; no pulse has been applied yet.
    results["pulse_number"].append(0)
    results["time_s"].append(0.0)
    results["I_A"].append(G_current * V_read * (1 + np.random.uniform(-0.01, 0.01)))
    results["V_read_V"].append(V_read)
    results["conductance_S"].append(G_current)

    for i in range(params['n_pulses']):
        # Apply conductance change based on stimulus polarity
        if params['stim_level'] > 0:
            # LTP (Potentiation): Move toward G_max
            G_current += (G_max - G_current) * step_size
        else:
            # LTD (Depression): Move toward G_min
            G_current += (G_min - G_current) * step_size

        # Clip to physical limits
        G_current = np.clip(G_current, G_min, G_max)

        # Apply small random noise (±2%)
        G_current *= (1 + np.random.uniform(-0.02, 0.02))

        # Calculate measured current
        I_meas = G_current * V_read

        # Add measurement noise
        I_meas *= (1 + np.random.uniform(-0.01, 0.01))

        # Store results
        t_rel = time.time() - t0
        results["pulse_number"].append(i + 1)
        results["time_s"].append(t_rel)
        results["I_A"].append(I_meas)
        results["V_read_V"].append(V_read)
        results["conductance_S"].append(G_current)

        # Simulate timing
        time.sleep(params['stim_period_ms'] / 1000.0)

    # Provenance: this data came from the SIMULATION model, not from an
    # instrument. Without this flag the returned dict, the exported CSV and
    # the plots are indistinguishable from a real measurement, and synthetic
    # numbers can travel the whole CSV -> JSON -> fit pipeline unchallenged.
    results['simulated'] = True
    results['data_source'] = 'synthetic_model'
    results['execution_path'] = 'simulation'

    return results


# =============================================================================
# VISUAL (SELF-POWERED) SYNAPSE CHARACTERIZATION
# =============================================================================

def visual_synapse_sequence(instrument, stim_ch, read_ch, params, continuous_mode=False):
    """
    Core engine for visual (self-powered) synapse measurements.

    Measures photocurrent (Jsc) at 0V during light pulses. The stim channel
    controls the LED/light source while the read channel measures the
    photocurrent from the synapse device in short-circuit conditions.

    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): LED control channel ('smua' or 'smub')
        read_ch (str): Photocurrent measurement channel ('smua' or 'smub')
        params (dict): Measurement parameters containing:
            - light_pulse_voltage (float): Voltage to LED controller (V)
            - pulse_width_ms (float): Light pulse duration (ms)
            - pulse_period_ms (float): Time between pulse starts (ms)
            - n_pulses (int): Number of light pulses
            - measure_start_delay_ms (float): Delay after pulse start before measuring
            - measure_end_margin_ms (float): Stop measuring before pulse ends
            - readings_per_pulse (int): Number of readings to average per pulse
            - compliance_A (float): Current compliance (A)
            - nplc (float): Integration time in power line cycles
            - sample_interval_ms (float): For continuous mode, sampling interval
        continuous_mode (bool): If True, use continuous I(t) measurement

    Returns:
        dict: Results containing:
            - pulse_number: List of pulse indices (standard mode) or None (continuous)
            - time_s: List of timestamps in seconds
            - Jsc_A: List of measured photocurrents in A
            - params: Copy of input parameters
            - mode: 'visual_standard' or 'visual_continuous'
    """
    _check_channels(stim_ch, read_ch)

    # Initialize results structure
    results = {
        "pulse_number": [],
        "time_s": [],
        "Jsc_A": [],
        "params": params.copy(),
        "mode": "visual_continuous" if continuous_mode else "visual_standard",
        "metadata": {
            "wavelength_nm": params.get('wavelength_nm', None),
            "intensity_mW_cm2": params.get('intensity_mW_cm2', None),
            "sample_id": params.get('sample_id', 'unknown'),
            "measurement_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "light_pulse_voltage": params.get('light_pulse_voltage'),
            "device_area_cm2": params.get('device_area_cm2', None)
        }
    }

    # Check if instrument supports LUA
    if not supports_lua_execution(instrument):
        raise ValueError(
            "Visual synapse mode requires LUA-capable instrument (2600 series).\n"
            "Connected instrument does not support LUA execution."
        )

    # Configure sense mode (must be set before LUA script runs)
    _apply_wire_modes(instrument, params, [read_ch, stim_ch])

    # Record instrument voltage ceiling for the LUA generator's range picker.
    params['max_voltage'] = get_max_voltage(instrument)

    # Adopt the instrument's auto-detected mains frequency unless overridden.
    # results['params'] was copied before this point, so stamp it there too.
    resolve_line_freq(instrument, params)
    results['params']['line_freq_hz'] = params['line_freq_hz']

    # Generate appropriate LUA script
    if continuous_mode:
        lua_script = generate_visual_continuous_lua_script(stim_ch, read_ch, params)
        # Exact upper bound on the samples the script can take
        n_expected_points = lua_script.facts['max_samples']
        analysis = 'samples'
        sample_interval_s = lua_script.facts['sample_interval_ms'] / 1000.0
    else:
        lua_script = generate_visual_synapse_lua_script(stim_ch, read_ch, params)
        # One reading per pulse, no baseline read: every interval is a period.
        n_expected_points = params['n_pulses']
        analysis = 'pulses'
        sample_interval_s = None

    # pulse_period >= pulse_width is enforced by the generators, so the
    # requested duration is always deliverable.
    expected_time_s = params['pulse_period_ms'] * params['n_pulses'] / 1000.0

    print(f"Using LUA execution for visual synapse measurement...")
    print(f"  Mode: {'Continuous I(t)' if continuous_mode else 'Jsc per pulse'}")
    print(f"  Pulses: {params['n_pulses']}")
    print(f"  Light voltage: {params['light_pulse_voltage']}V")

    try:
        # Execute LUA script
        timing_report = {}
        timestamps, currents = execute_lua_script_fast(
            instrument, lua_script, read_ch, n_expected_points,
            expected_time_s=expected_time_s,
            timing_out=timing_report,
            analysis=analysis,
            requested_period_s=sample_interval_s,
            n_is_exact=not continuous_mode,
        )

        # Process results
        t_start = timestamps[0] if timestamps else 0

        for i, (timestamp, current) in enumerate(zip(timestamps, currents)):
            t_elapsed = timestamp - t_start
            results['time_s'].append(t_elapsed)
            results['Jsc_A'].append(current)

            if not continuous_mode:
                results['pulse_number'].append(i)

        results['execution_path'] = 'lua'
        # What the instrument delivered, plus the settling / sampling caveats
        # computed while generating the script (None when there are none).
        results['timing'] = dict(timing_report)
        results['timing']['settle_warning'] = lua_script.facts['settle_warning']
        if continuous_mode:
            results['timing']['sample_warning'] = lua_script.facts['sample_warning']
            results['timing']['sample_interval_ms'] = lua_script.facts['sample_interval_ms']

        # Display summary
        if results['Jsc_A']:
            jsc_array = np.array(results['Jsc_A'])
            print(f"✓ Completed {len(results['Jsc_A'])} measurements")
            print(f"  Initial Jsc: {jsc_array[0]*1e6:.3f} µA")
            print(f"  Final Jsc: {jsc_array[-1]*1e6:.3f} µA")
            if len(jsc_array) > 1:
                delta_jsc = (jsc_array[-1] - jsc_array[0]) / abs(jsc_array[0]) * 100 if jsc_array[0] != 0 else 0
                print(f"  ΔJsc: {delta_jsc:.2f}%")

        return results

    except Exception as e:
        # Ensure outputs are off on error (levels to 0 first)
        _abort_outputs(instrument)
        raise


def simulate_visual_synapse(params, continuous_mode=False):
    """
    Simulates visual synapse measurement for testing without hardware.

    Models photocurrent response with light-induced plasticity behavior:
    - Photocurrent increases with repeated light pulses (potentiation)
    - Includes realistic noise and transient dynamics

    Args:
        params (dict): Same parameters as visual_synapse_sequence
        continuous_mode (bool): If True, simulate continuous I(t) data

    Returns:
        dict: Results in same format as visual_synapse_sequence
    """
    results = {
        "pulse_number": [],
        "time_s": [],
        "Jsc_A": [],
        "params": params.copy(),
        "mode": "visual_continuous" if continuous_mode else "visual_standard",
        "metadata": {
            "wavelength_nm": params.get('wavelength_nm', None),
            "intensity_mW_cm2": params.get('intensity_mW_cm2', None),
            "sample_id": params.get('sample_id', 'unknown'),
            "measurement_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "light_pulse_voltage": params.get('light_pulse_voltage'),
            "simulated": True
        }
    }

    n_pulses = params['n_pulses']
    pulse_width_ms = params['pulse_width_ms']
    pulse_period_ms = params['pulse_period_ms']

    # Simulation parameters
    Jsc_initial = -50e-6  # Initial photocurrent: -50 µA (negative for photocurrent)
    Jsc_max = -200e-6     # Maximum photocurrent: -200 µA
    learning_rate = 0.03  # Photoplasticity rate

    Jsc_current = Jsc_initial
    t0 = time.time()

    if continuous_mode:
        # Continuous mode: sample throughout pulse sequence
        sample_interval_ms = params.get('sample_interval_ms', 1.0)
        samples_per_period = int(pulse_period_ms / sample_interval_ms)
        samples_during_pulse = int(pulse_width_ms / sample_interval_ms)

        for pulse in range(n_pulses):
            # During light pulse
            for s in range(samples_during_pulse):
                t_rel = (pulse * pulse_period_ms + s * sample_interval_ms) / 1000.0
                # Photocurrent during illumination
                Jsc_measured = Jsc_current * (1 + np.random.uniform(-0.02, 0.02))
                results['time_s'].append(t_rel)
                results['Jsc_A'].append(Jsc_measured)

            # After light pulse (dark period)
            for s in range(samples_per_period - samples_during_pulse):
                t_rel = (pulse * pulse_period_ms + pulse_width_ms + s * sample_interval_ms) / 1000.0
                # Dark current (much smaller)
                dark_current = Jsc_current * 0.01 * (1 + np.random.uniform(-0.1, 0.1))
                results['time_s'].append(t_rel)
                results['Jsc_A'].append(dark_current)

            # Update photoplasticity state
            Jsc_current += (Jsc_max - Jsc_current) * learning_rate

        # Simulate measurement time
        time.sleep(0.1)

    else:
        # Standard mode: one measurement per pulse
        for i in range(n_pulses):
            # Update photoplasticity state
            Jsc_current += (Jsc_max - Jsc_current) * learning_rate

            # Add measurement noise
            Jsc_measured = Jsc_current * (1 + np.random.uniform(-0.02, 0.02))

            # Store results
            t_rel = i * pulse_period_ms / 1000.0
            results['pulse_number'].append(i)
            results['time_s'].append(t_rel)
            results['Jsc_A'].append(Jsc_measured)

            # Simulate timing (fast for simulation)
            time.sleep(0.01)

    # Provenance: this data came from the SIMULATION model, not from an
    # instrument. Without this flag the returned dict, the exported CSV and
    # the plots are indistinguishable from a real measurement, and synthetic
    # numbers can travel the whole CSV -> JSON -> fit pipeline unchallenged.
    results['simulated'] = True
    results['data_source'] = 'synthetic_model'

    return results


def calculate_visual_synapse_metrics(results):
    """
    Calculates metrics specific to visual synapse measurements.

    Args:
        results (dict): Results from visual_synapse_sequence

    Returns:
        dict: Metrics including:
            - Initial Jsc (A): First measured photocurrent
            - Final Jsc (A): Last measured photocurrent
            - Delta Jsc (%): Relative change in photocurrent
            - Mean Jsc (A): Average photocurrent
            - PPF (%): Paired-pulse facilitation
    """
    metrics = {}

    Jsc_list = np.array(results.get("Jsc_A", []))

    if len(Jsc_list) == 0:
        return metrics

    # Basic statistics
    metrics["Initial Jsc (A)"] = f"{Jsc_list[0]:.4e}"
    metrics["Final Jsc (A)"] = f"{Jsc_list[-1]:.4e}"
    metrics["Mean Jsc (A)"] = f"{np.mean(Jsc_list):.4e}"
    metrics["Max |Jsc| (A)"] = f"{np.max(np.abs(Jsc_list)):.4e}"
    metrics["Min |Jsc| (A)"] = f"{np.min(np.abs(Jsc_list)):.4e}"

    # Delta Jsc
    if Jsc_list[0] != 0:
        delta_jsc_percent = (Jsc_list[-1] - Jsc_list[0]) / abs(Jsc_list[0]) * 100
        metrics["Delta Jsc (%)"] = round(delta_jsc_percent, 2)
    else:
        metrics["Delta Jsc (%)"] = "N/A"

    # Paired-pulse facilitation (for standard mode)
    if len(Jsc_list) >= 2 and Jsc_list[0] != 0:
        ppf = abs(Jsc_list[1]) / abs(Jsc_list[0])
        metrics["PPF (%)"] = round(ppf * 100, 2)

    return metrics


def save_visual_synapse_data(results, filename):
    """
    Saves visual synapse data to CSV with metadata.

    Args:
        results (dict): Results from visual_synapse_sequence
        filename (str): Output file path

    Returns:
        dict: Calculated metrics
    """
    metrics = calculate_visual_synapse_metrics(results)

    params = results.get("params", {})
    metadata = results.get("metadata", {})

    csv_content = []

    # Header with metadata
    csv_content.append("# Visual (Self-Powered) Synapse Characterization")
    csv_content.append(f"# mode={results.get('mode', 'visual_standard')}")
    csv_content.append(f"# execution_path={results.get('execution_path', 'unknown')}")

    # Delivered timing and settling verdict, exactly as for the pulse-read export:
    # a run that did not hold its period or whose readings were not settled must
    # say so in the file, not only on the console.
    _timing = results.get('timing') or {}
    if _timing:
        csv_content.append(
            "# " + " # ".join(f"timing_{k}={v}" for k, v in _timing.items())
        )
        if _timing.get('settle_warning'):
            csv_content.append(f"# SETTLE_WARNING: {_timing['settle_warning']}")
    else:
        csv_content.append("# timing_status=not_recorded")

    # Parameters
    param_str = " # ".join(f"{k}={v}" for k, v in params.items() if not isinstance(v, dict))
    csv_content.append(f"# {param_str}")

    # Metadata
    meta_str = " # ".join(f"{k}={v}" for k, v in metadata.items() if v is not None)
    csv_content.append(f"# {meta_str}")

    # Metrics
    metrics_str = " # ".join(f"{k}={v}" for k, v in metrics.items())
    csv_content.append(f"# METRICS: {metrics_str}")

    csv_content.append("#")

    # Column headers and data
    if results.get('mode') == 'visual_continuous':
        csv_content.append("timestamp_s,Jsc_A")
        for t, jsc in zip(results['time_s'], results['Jsc_A']):
            csv_content.append(f"{t:.6f},{jsc:.6e}")
    else:
        csv_content.append("pulse,timestamp_s,Jsc_A")
        for pulse, t, jsc in zip(results['pulse_number'], results['time_s'], results['Jsc_A']):
            csv_content.append(f"{pulse},{t:.6f},{jsc:.6e}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))

    print(f"Visual synapse data saved to {filename}")

    return metrics


# =============================================================================
# STEP 5: METRICS CALCULATION AND DATA SAVING
# =============================================================================

def calculate_synapse_metrics(results):
    """
    Calculates derived synaptic metrics from pulse-read results.
    
    Args:
        results (dict): Results from pulse_read_sequence or simulate_pulse_read
    
    Returns:
        dict: Calculated metrics including:
            - PPF (%): Paired-pulse facilitation
            - Mean Conductance (S): Average conductance
            - Delta G (S): Change in conductance (final - initial)
            - Delta G (%): Relative change in conductance
            - Max Conductance (S): Maximum observed conductance
            - Min Conductance (S): Minimum observed conductance
    """
    metrics = {}
    
    G_list = np.array(results["conductance_S"])
    I_list = np.array(results["I_A"])
    
    # Filter out NaN values
    G_valid = G_list[~np.isnan(G_list)]
    
    if len(G_valid) >= 2:
        # Paired-Pulse Facilitation (PPF) - ratio of 2nd to 1st pulse
        if G_valid[0] != 0:
            ppf = G_valid[1] / G_valid[0]
            metrics["PPF (%)"] = round(ppf * 100, 2)
        else:
            metrics["PPF (%)"] = "N/A"
        
        # Delta G (absolute and relative)
        delta_G = G_valid[-1] - G_valid[0]
        metrics["Delta G (S)"] = f"{delta_G:.4e}"
        
        if G_valid[0] != 0:
            delta_G_percent = (delta_G / G_valid[0]) * 100
            metrics["Delta G (%)"] = round(delta_G_percent, 2)
        else:
            metrics["Delta G (%)"] = "N/A"
    
    # Statistical measures
    if len(G_valid) > 0:
        metrics["Mean Conductance (S)"] = f"{np.mean(G_valid):.4e}"
        metrics["Max Conductance (S)"] = f"{np.max(G_valid):.4e}"
        metrics["Min Conductance (S)"] = f"{np.min(G_valid):.4e}"
        metrics["Std Conductance (S)"] = f"{np.std(G_valid):.4e}"
    
    # Current statistics
    if len(I_list) > 0:
        metrics["Mean Current (A)"] = f"{np.mean(I_list):.4e}"
        metrics["Max Current (A)"] = f"{np.max(I_list):.4e}"
        metrics["Min Current (A)"] = f"{np.min(I_list):.4e}"
    
    return metrics


def process_and_save_synapse_data(results_dict, filename):
    """
    Calculates derived metrics and saves data to CSV with metadata.
    
    Args:
        results_dict (dict): Results from pulse_read_sequence
        filename (str): Path to save CSV file
    
    Returns:
        dict: Calculated metrics
    """
    # Calculate metrics
    metrics = calculate_synapse_metrics(results_dict)
    
    # Prepare CSV content
    params = results_dict["params"]
    csv_content = []
    
    # Metadata header (key=value format)
    metadata_line = "# " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)

    # Acquisition provenance. Both execution paths measure the same quantity,
    # but the timing source differs (instrument timer vs host clock), so which
    # one ran belongs with the data rather than only in the console.
    csv_content.append(
        f"# execution_path={results_dict.get('execution_path', 'unknown')}"
        f" # pulse_index_0=pre_stimulus_baseline"
        f" # simulated={bool(results_dict.get('simulated', False))}"
        f" # data_source={'synthetic_model' if results_dict.get('simulated') else 'instrument'}"
    )

    # Delivered timing. The requested period is already in the params header
    # above; this records what the instrument ACTUALLY achieved, so a train
    # that ran at the hardware limit is not silently filed under a period it
    # never delivered. `status` is written even when analysis failed —
    # "we could not check" must not read as "we checked and it was fine".
    _timing = results_dict.get('timing') or {}
    if _timing:
        csv_content.append(
            "# " + " # ".join(f"timing_{k}={v}" for k, v in _timing.items())
        )
        if _timing.get('settle_warning'):
            # Spelled out on its own line: the readings are not settled to
            # specification, which affects how far every conductance can be trusted.
            csv_content.append(f"# SETTLE_WARNING: {_timing['settle_warning']}")
    else:
        csv_content.append("# timing_status=not_recorded")

    # Add calculated metrics as metadata
    metrics_line = "# " + " # ".join(f"{k}={v}" for k, v in metrics.items())
    csv_content.append(metrics_line)
    
    # Column headers
    csv_content.append("pulse,timestamp_s,I_A,V_read_V,conductance_S")
    
    # Data rows
    for i in range(len(results_dict["pulse_number"])):
        row = [
            str(results_dict["pulse_number"][i]),
            f"{results_dict['time_s'][i]:.6f}",
            f"{results_dict['I_A'][i]:.6e}",
            f"{results_dict['V_read_V'][i]:.4f}",
            f"{results_dict['conductance_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))

    return metrics



# =============================================================================
# STEP 6: SRDP AND STDP CHARACTERIZATION
# =============================================================================

def _validate_srdp_sweep(base_params, freq_list_hz, require_read_bias=True):
    """Whole-sweep SRDP validation. Touches no instrument.

    Returns the frequencies as floats. Everything that can be wrong with the
    request is decided here, before the first pulse: n_pulses = -4 used to
    reach the instrument and fail with a raw IndexError after a script had run,
    and n_pulses = 0 "completed" with empty arrays.
    """
    n = base_params.get('n_pulses')
    if n is None or int(n) != n or int(n) < 1:
        raise ValueError(f"SRDP n_pulses must be a whole number >= 1, got {n!r}")
    freqs = [float(f) for f in freq_list_hz]
    if not freqs:
        raise ValueError("SRDP frequency list is empty")
    if not all(np.isfinite(f) and f > 0 for f in freqs):
        raise ValueError(f"SRDP frequencies must be finite and > 0 Hz, got {list(freq_list_hz)!r}")
    # The device keeps its state from one frequency to the next, so the order is
    # part of the measurement: a descending sweep measures each rate against the
    # history of the faster ones and is not comparable with an ascending one.
    if any(b < a for a, b in zip(freqs, freqs[1:])):
        raise ValueError(
            "SRDP frequencies must be in ascending order (end frequency >= start "
            f"frequency); got {freqs!r}. Re-order them, or swap Start and End.")
    if require_read_bias and base_params.get('read_voltage', 0) == 0:
        raise ValueError("SRDP needs a non-zero read voltage: conductance is undefined at 0 V")
    return freqs


# A point whose delivered rate falls this far below the requested one has hit
# the instrument's ceiling (the loop runs flat out), so any faster request would
# deliver the same rate.
SRDP_CLIP_TOLERANCE_PERCENT = 5.0
# Two delivered rates this close are the same operating point.
SRDP_SAME_RATE_TOLERANCE = 0.02


def _validate_srdp_sweep(base_params, freq_list_hz, require_read_bias=True):
    """Whole-sweep SRDP validation. Touches no instrument.

    Returns the frequencies as floats. Everything that can be wrong with the
    request is decided here, before the first pulse: n_pulses = -4 used to
    reach the instrument and fail with a raw IndexError after a script had run,
    and n_pulses = 0 "completed" with empty arrays.
    """
    n = base_params.get('n_pulses')
    if n is None or int(n) != n or int(n) < 1:
        raise ValueError(f"SRDP n_pulses must be a whole number >= 1, got {n!r}")
    freqs = [float(f) for f in freq_list_hz]
    if not freqs:
        raise ValueError("SRDP frequency list is empty")
    if not all(np.isfinite(f) and f > 0 for f in freqs):
        raise ValueError(f"SRDP frequencies must be finite and > 0 Hz, got {list(freq_list_hz)!r}")
    # The device keeps its state from one frequency to the next, so the order is
    # part of the measurement: a descending sweep measures each rate against the
    # history of the faster ones and is not comparable with an ascending one.
    if any(b < a for a, b in zip(freqs, freqs[1:])):
        raise ValueError(
            "SRDP frequencies must be in ascending order (end frequency >= start "
            f"frequency); got {freqs!r}. Re-order them, or swap Start and End.")
    if require_read_bias and base_params.get('read_voltage', 0) == 0:
        raise ValueError("SRDP needs a non-zero read voltage: conductance is undefined at 0 V")
    return freqs


# A point whose delivered rate falls this far below the requested one has hit
# the instrument's ceiling (the loop runs flat out), so any faster request would
# deliver the same rate.
SRDP_CLIP_TOLERANCE_PERCENT = 5.0
# Two delivered rates this close are the same operating point.
SRDP_SAME_RATE_TOLERANCE = 0.02


def measure_srdp(instrument, stim_ch, read_ch, base_params, freq_list_hz):
    """
    Measures Spike-Rate-Dependent Plasticity (SRDP).

    Sweeps through different spike frequencies and measures the resulting
    conductance change (ΔG) to characterize rate-dependent learning.

    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Stimulus channel ('smua' or 'smub')
        read_ch (str): Read channel ('smua' or 'smub')
        base_params (dict): Base measurement parameters (stim_level, read_voltage, etc.)
        freq_list_hz (list): Frequencies to test in Hz, ascending

    Returns:
        dict: SRDP results containing:
            - frequencies_hz: DELIVERED frequencies of the measured points
            - requested_frequencies_hz: what was asked for at those points
            - skipped_frequencies: requests not measured, each with the reason
              (repeated, or beyond the rate the instrument had already saturated at)
            - delta_g_S: List of conductance changes
            - delta_g_percent: List of relative conductance changes
            - g_initial_S: List of initial conductances
            - g_final_S: List of final conductances
            - execution_path / timing / frequency_source: one entry per point
            - base_params: Copy of base parameters
    """
    if stim_ch == read_ch:
        raise ValueError("Stim and Read channels must be different for SRDP measurements")

    # Validate the whole request before any instrument access (queries included).
    freq_list_hz = _validate_srdp_sweep(base_params, freq_list_hz)

    # Adopt the instrument's auto-detected mains frequency unless overridden.
    resolve_line_freq(instrument, base_params)

    # Check instrument capability for stimulus level
    if base_params.get('stim_drive_type') == 'I':
        max_current = get_max_current(instrument)
        stim_current = abs(base_params.get('stim_level', 0))
        if stim_current > max_current:
            model = validate_instrument_model(instrument)
            raise ValueError(
                f"ERROR: SRDP stimulus level {stim_current}A exceeds instrument capability.\n"
                f"Connected: {model} (Max: {max_current}A)"
            )
    elif base_params.get('stim_drive_type') == 'V':
        max_voltage = get_max_voltage(instrument)
        stim_voltage = abs(base_params.get('stim_level', 0))
        if stim_voltage > max_voltage:
            model = validate_instrument_model(instrument)
            raise ValueError(
                f"ERROR: SRDP stimulus level {stim_voltage}V exceeds instrument capability.\n"
                f"Connected: {model} (Max: {max_voltage}V)"
            )

    results = {
        # DELIVERED frequencies, measured from the instrument's own timestamps
        # — see the note where they are computed. This is the axis every
        # downstream consumer should use.
        "frequencies_hz": [],
        # What the caller asked for, kept for reference and for the record of
        # how far the instrument fell short.
        "requested_frequencies_hz": [],
        "frequency_shortfall_percent": [],
        "frequency_source": [],
        "skipped_frequencies": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        # Per point, as pulse-read stamps them: which path produced it and what
        # timing the instrument reported.
        "execution_path": [],
        "timing": [],
        "base_params": base_params.copy()
    }

    # Frequencies the instrument is known to deliver identically are measured
    # once. Requests repeated verbatim go first; beyond that, once a point shows
    # the loop running flat out, every faster request would only re-measure it.
    seen = set()
    saturated_at_hz = None
    last_delivered_hz = None

    # The line frequency here is the RESOLVED one (resolve_line_freq above),
    # never the module default.
    NPLC_PERIOD_MS = 1000.0 / _line_freq_hz(base_params)
    nplc_eff, avg_eff = derive_measurement_averaging(base_params)
    meas_time_ms = nplc_eff * NPLC_PERIOD_MS * avg_eff

    for freq_hz in freq_list_hz:
        if freq_hz in seen:
            results["skipped_frequencies"].append(
                {"requested_hz": freq_hz, "reason": "repeated request"})
            continue
        seen.add(freq_hz)
        if saturated_at_hz is not None:
            results["skipped_frequencies"].append(
                {"requested_hz": freq_hz,
                 "reason": f"instrument already saturated at {saturated_at_hz:.4g} Hz; "
                           "this request would deliver the same rate"})
            continue

        # Calculate period from frequency
        period_ms = 1000.0 / freq_hz

        # Warn if period is shorter than physical minimum (instrument will run as fast as it can)
        min_period = base_params['stim_width_ms'] + base_params['read_delay_ms'] + meas_time_ms + 0.5
        if period_ms < min_period:
            print(f"⚠️  Frequency {freq_hz:.1f} Hz: requested period {period_ms:.2f} ms < minimum {min_period:.2f} ms. "
                  f"Instrument will run as fast as possible.")

        # Update parameters for this frequency
        freq_params = base_params.copy()
        freq_params['stim_period_ms'] = period_ms

        # Run pulse-read sequence
        pulse_results = pulse_read_sequence(instrument, stim_ch, read_ch, freq_params)

        # Extract initial and final conductance
        g_list = np.array(pulse_results["conductance_S"])
        g_valid = g_list[~np.isnan(g_list)]

        if len(g_valid) < 2:
            raise RuntimeError(
                f"SRDP at {freq_hz:g} Hz returned {len(g_valid)} valid conductance "
                "reading(s); at least the baseline and one pulse are needed. The "
                "point is not recorded.")

        g_initial = g_valid[0]
        g_final = g_valid[-1]
        delta_g = g_final - g_initial

        if g_initial != 0:
            delta_g_percent = (delta_g / g_initial) * 100
        else:
            delta_g_percent = float('nan')

        # --- The DELIVERED frequency (audit H14) ---
        #
        # The requested frequency is not necessarily the one the device
        # experienced. Whenever the requested period is shorter than the
        # instrument can service — stimulus width, read delay, settle and
        # ADC aperture all have to fit inside it — the LUA loop simply runs
        # as fast as it can, and the warning above says so.
        #
        # This function used to record the NOMINAL frequency regardless. A
        # user requesting 100 Hz against a true 62 Hz ceiling got a run at
        # 62 Hz filed under 100 Hz, and since f0 and the slope are
        # determined precisely where the ceiling starts to bite, the SRDP
        # x-axis was wrong exactly where it mattered most.
        #
        # The instrument's own timestamps were being retrieved and thrown
        # away. The mean inter-pulse interval over the train recovers the
        # rate the device actually saw.
        timestamps = np.asarray(pulse_results.get("time_s", []), dtype=float)
        delivered_hz = freq_hz
        frequency_source = "requested (fewer than 2 pulse intervals: delivered rate not measurable)"
        if timestamps.size >= 3:
            # Skip index 0: it is the pre-stimulus baseline read, not a
            # pulse, so the 0->1 interval is not a pulse period.
            intervals = np.diff(timestamps[1:])
            intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
            if intervals.size:
                # Median rather than mean: robust to a single stalled
                # interval from a VISA hiccup.
                delivered_hz = 1.0 / float(np.median(intervals))
                frequency_source = "timestamps"

        shortfall = (freq_hz - delivered_hz) / freq_hz * 100 if freq_hz else 0.0
        if abs(shortfall) > 2.0:
            print(f"  {freq_hz:.2f} Hz requested, {delivered_hz:.2f} Hz "
                  f"delivered ({shortfall:+.1f}%). The DELIVERED rate is "
                  "what is recorded.")

        clipped = shortfall > SRDP_CLIP_TOLERANCE_PERCENT
        if clipped:
            saturated_at_hz = delivered_hz
        if (clipped and last_delivered_hz is not None
                and abs(delivered_hz - last_delivered_hz) / last_delivered_hz
                < SRDP_SAME_RATE_TOLERANCE):
            # Same delivered rate as the previous point: one operating point.
            results["skipped_frequencies"].append(
                {"requested_hz": freq_hz,
                 "reason": f"delivered {delivered_hz:.4g} Hz, the same rate as the "
                           "previous point"})
        else:
            results["frequencies_hz"].append(delivered_hz)
            results["requested_frequencies_hz"].append(freq_hz)
            results["frequency_shortfall_percent"].append(shortfall)
            results["frequency_source"].append(frequency_source)
            results["delta_g_S"].append(delta_g)
            results["delta_g_percent"].append(delta_g_percent)
            results["g_initial_S"].append(g_initial)
            results["g_final_S"].append(g_final)
            results["execution_path"].append(pulse_results["execution_path"])
            results["timing"].append(pulse_results.get("timing") or {"status": "not_recorded"})
        last_delivered_hz = delivered_hz

        # Small delay between frequency measurements
        time.sleep(0.5)

    if results["skipped_frequencies"]:
        print("  SRDP: not measured -> "
              + "; ".join(f"{s['requested_hz']:g} Hz ({s['reason']})"
                          for s in results["skipped_frequencies"]))

    return results


def _validate_stdp_sweep(params, delta_t_list_ms):
    """Whole-sweep STDP validation. Touches no instrument.

    Every spike pair must be physically producible, and that is decided for the
    entire Δt list BEFORE the first pulse: the GUI defaults used to fail only at
    the 7th point, after 6 sweeps had already stimulated the device. Collects
    every violation so the user sees them together.
    """
    n_pairs = params.get('n_pulses', 50)
    width = params['stim_width_ms']
    period = params.get('stim_period_ms', 200)
    problems = []
    if int(n_pairs) != n_pairs or int(n_pairs) < 1:
        problems.append(f"n_pulses (spike pairs) must be a whole number >= 1, got {n_pairs!r}")
    if not (np.isfinite(width) and width > 0):
        problems.append(f"stim_width_ms must be > 0, got {width!r}")
    if not np.isfinite(period) or period <= 0:
        problems.append(f"stim_period_ms must be > 0, got {period!r}")
    if params.get('stim_drive_type', 'V') == 'I' and 'post_spike_level' not in params:
        problems.append(
            "stim_drive_type='I' needs an explicit post_spike_level in VOLTS: the "
            "post-synaptic spike is always a voltage on the read channel, and "
            "defaulting it to the stimulus level would apply amperes as volts")
    if not len(delta_t_list_ms):
        problems.append("the delta_t list is empty")
    for dt in delta_t_list_ms:
        if not np.isfinite(dt):
            problems.append(f"delta_t = {dt!r} ms is not a finite time")
        elif dt == 0:
            problems.append("delta_t = 0 ms has no order (pre-before-post vs post-before-pre is undefined)")
        elif abs(dt) < width:
            problems.append(
                f"|delta_t| = {abs(dt):g} ms is below the pulse width ({width:g} ms); delta_t "
                "is onset-to-onset, so the spikes would have to overlap")
        elif abs(dt) + width > period * (1 + 1e-12):
            problems.append(
                f"delta_t = {dt:g} ms with a {width:g} ms pulse needs {abs(dt) + width:g} ms per "
                f"pair, longer than the pair period ({period:g} ms); the pair would run at "
                f"{abs(dt) + width:g} ms, not {period:g} ms")
    if problems:
        raise ValueError("STDP sweep is not producible:\n  - " + "\n  - ".join(problems))


def generate_stdp_lua_script(stim_ch, read_ch, params, delta_t_ms):
    """
    Generates LUA script for fast STDP spike-pair sequences.
    Executes entirely on instrument for deterministic millisecond-level timing.

    The pre-synaptic spike is on stim_ch and honours stim_drive_type ('V' volts,
    'I' amperes). The post-synaptic spike is always a VOLTAGE on read_ch.

    Args:
        stim_ch (str): Pre-synaptic channel ('smua' or 'smub')
        read_ch (str): Post-synaptic channel ('smua' or 'smub')
        params (dict): Measurement parameters
        delta_t_ms (float): Time difference between pre and post spikes (ms)

    Returns:
        str: LUA script for STDP measurement
    """
    _validate_stdp_sweep(params, [delta_t_ms])

    # Extract parameters
    drive = params.get('stim_drive_type', 'V')
    stim_level = params['stim_level']
    post_level = params.get('post_spike_level', stim_level)
    pulse_width_ms = params['stim_width_ms']
    n_pairs = params.get('n_pulses', 50)
    pair_period_ms = params.get('stim_period_ms', 200)
    read_voltage = params['read_voltage']
    compliance_A = params['compliance_A']
    settle_ms = params.get('settle_ms', 5)
    line_freq_hz = _line_freq_hz(params)
    # The shared averaging rule. This script used the raw nplc (default 0.1) and
    # never wrote the filter, so it inherited whatever the previous run left:
    # 25 ms reads instead of 7 ms.
    nplc, measure_avg = derive_measurement_averaging(params)

    # Convert to seconds
    pulse_width_s = pulse_width_ms / 1000.0
    delta_t_s = delta_t_ms / 1000.0
    pair_period_s = pair_period_ms / 1000.0
    settle_s = settle_ms / 1000.0

    fixed_i_range = get_fixed_current_range(compliance_A)
    max_voltage = params.get('max_voltage', 200.0)
    stim_v_range = _get_fixed_voltage_range(
        max(abs(stim_level), abs(post_level), abs(read_voltage)), max_voltage
    )
    # The read channel also sources the post-synaptic spike, so its source range
    # must hold post_level, not only the (typically 0.1 V) read voltage.
    read_v_range = _get_fixed_voltage_range(max(abs(read_voltage), abs(post_level)), max_voltage)

    if drive == 'V':
        stim_attr = 'levelv'
        stim_config = f"""{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.rangev = {stim_v_range}
{stim_ch}.source.levelv = 0
{stim_ch}.source.limiti = {compliance_A}"""
    else:
        # A current spike: the range must contain |stim_level| and the
        # compliance is a voltage. Writing levelv here applied 1 mA as 1 mV.
        stim_attr = 'leveli'
        stim_config = f"""{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS
{stim_ch}.source.rangei = {_get_fixed_current_range(abs(stim_level))}
{stim_ch}.source.leveli = 0
{stim_ch}.source.limitv = {_current_drive_voltage_limit(params, 'STDP')}"""

    settle_delay = lua_delay(settle_s, "settle_ms")   # delay(0) aborts a 2600 script
    adc_setup = "\n".join(_emit_read_adc_setup(read_ch, nplc, measure_avg))
    restore_adc = "\n".join(_emit_restore_adc(read_ch))

    # --- Delta_t convention (audit M28) ---
    #
    # CLAUDE.md, network.py's STDP rule and fitting.fit_stdp_window all define
    # Delta_t ONSET-TO-ONSET: the interval between the two spikes' leading
    # edges. This generator emitted `delay(delta_t)` between the END of the
    # pre-pulse and the START of the post-pulse, which is EDGE-TO-EDGE and
    # differs from the spec by exactly one pulse width — a ~25% error in every
    # fitted tau at a 5 ms pulse width against a 20 ms tau.
    #
    # Resolved by making the INSTRUMENT honour the spec rather than by
    # converting afterwards: the requested Delta_t stays onset-to-onset
    # everywhere, so the fitter, the learning rule and the exported data all
    # keep the one convention and nothing downstream needed re-validating.
    # (|delta_t| >= pulse width is enforced by _validate_stdp_sweep.)
    inter_pulse_gap_s = abs(delta_t_s) - pulse_width_s

    # Generate LUA script
    lua_script = f"""
-- Fast STDP Spike-Pair Sequence
-- Deterministic millisecond-level timing

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- The result globals are cleared FIRST. They persist on the instrument between
-- scripts, so a run that dies before assigning them would otherwise leave the
-- PREVIOUS delta_t point's readings for the host to collect as this point's.
_stdp_i_initial = nil
_stdp_i_final = nil

-- Configuration
local n_pairs = {n_pairs}
local stim_level = {stim_level}
local post_level = {post_level}
local pulse_width = {pulse_width_s}
-- Gap between the two pulses. delta_t is ONSET-TO-ONSET per the spec, so the
-- silent interval between them is delta_t minus one pulse width.
local inter_pulse_gap = {inter_pulse_gap_s}
local pair_period = {pair_period_s}
local read_voltage = {read_voltage}

-- Configure channels with fixed ranges for speed
{stim_config}
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF

{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS
{read_ch}.source.rangev = {read_v_range}
{read_ch}.source.levelv = 0
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{adc_setup}
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Turn on outputs at 0V (avoid relay toggling during spike pairs)
{stim_ch}.source.output = {stim_ch}.OUTPUT_ON
{read_ch}.source.output = {read_ch}.OUTPUT_ON
delay(0.01)

-- Measure initial conductance
{read_ch}.source.levelv = read_voltage
{settle_delay}
local I_initial = {read_ch}.measure.i()
{read_ch}.source.levelv = 0

-- STDP spike-pair loop (timer-based to prevent cumulative drift)
timer.reset()
local t_next = 0
local overruns = 0
local worst_overrun = 0
for pair = 1, n_pairs do
    t_next = t_next + pair_period
"""

    if delta_t_ms >= 0:
        # Pre before Post (LTP)
        lua_script += f"""    -- Pre-spike (level change, no relay toggle)
    {stim_ch}.source.{stim_attr} = stim_level
    delay(pulse_width)
    {stim_ch}.source.{stim_attr} = 0

    -- Gap, so that post ONSET falls delta_t after pre ONSET
    if inter_pulse_gap > 0 then
        delay(inter_pulse_gap)
    end

    -- Post-spike (level change, no relay toggle)
    {read_ch}.source.levelv = post_level
    delay(pulse_width)
    {read_ch}.source.levelv = 0
"""
    else:
        # Post before Pre (LTD)
        lua_script += f"""    -- Post-spike (level change, no relay toggle)
    {read_ch}.source.levelv = post_level
    delay(pulse_width)
    {read_ch}.source.levelv = 0

    -- Gap, so that pre ONSET falls |delta_t| after post ONSET
    if inter_pulse_gap > 0 then
        delay(inter_pulse_gap)
    end

    -- Pre-spike (level change, no relay toggle)
    {stim_ch}.source.{stim_attr} = stim_level
    delay(pulse_width)
    {stim_ch}.source.{stim_attr} = 0
"""

    lua_script += f"""
    -- Wait for next pair using absolute timer (prevents cumulative drift).
    -- A pair that outlasted its period has no remainder: it is counted and
    -- reported, and t_next is re-baselined to now so the loop neither flags
    -- every later pair nor fires them back to back to catch up.
    local t_remain = t_next - timer.measure.t()
    if t_remain > 0 then
        delay(t_remain)
    else
        overruns = overruns + 1
        if -t_remain > worst_overrun then worst_overrun = -t_remain end
        t_next = timer.measure.t()
    end
end

-- Measure final conductance
{read_ch}.source.levelv = read_voltage
{settle_delay}
local I_final = {read_ch}.measure.i()

-- Cleanup
{stim_ch}.source.{stim_attr} = 0
{read_ch}.source.levelv = 0
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

{restore_adc}

-- Return results (stored in globals for retrieval)
_stdp_i_initial = I_initial
_stdp_i_final = I_final

print(string.format("{LUA_TIMING_TOKEN} overruns=%d worst_ms=%.4f n_pulses=%d",
                    overruns, worst_overrun * 1000, n_pairs))

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return lua_script


def _read_stdp_results(instrument):
    """(I_initial, I_final) from the script's globals, or LuaExecutionError.

    The script clears both to nil before it does anything, so "nil" here means
    this run did not produce the reading; it is never a previous point's value.
    """
    values = []
    for name in ('_stdp_i_initial', '_stdp_i_final'):
        raw = instrument.query(f"print({name})").strip()
        try:
            value = float(raw)
        except ValueError:
            raise LuaExecutionError(f"{name} is {raw!r}: the script did not produce this reading") from None
        # 9.91e37 is the 2600's overflow / not-a-number marker.
        if not np.isfinite(value) or abs(value) >= 9.9e37:
            raise LuaExecutionError(f"{name} = {raw}: the instrument returned an invalid reading")
        values.append(value)
    return tuple(values)


def measure_stdp(instrument, stim_ch, read_ch, base_params, delta_t_list_ms):
    """
    Measures Spike-Timing-Dependent Plasticity (STDP).

    Sweeps through different pre-post spike time differences (Δt) and measures
    the resulting conductance change to characterize timing-dependent learning.

    Convention:
        - Δt > 0: Pre-spike before post-spike → typically LTP
        - Δt < 0: Post-spike before pre-spike → typically LTD

    Implementation:
        - Pre-spike: Applied on stim_ch
        - Post-spike: Applied on read_ch (dual role: post-spike + readout)
        - Δt is controlled by delay between pre and post pulses

    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Pre-synaptic stimulus channel ('smua' or 'smub')
        read_ch (str): Post-synaptic channel ('smua' or 'smub')
        base_params (dict): Base measurement parameters
        delta_t_list_ms (list): List of Δt values to test in milliseconds

    Returns:
        dict: STDP results containing:
            - delta_t_ms: List of time differences
            - delta_g_S: List of conductance changes
            - delta_g_percent: List of relative conductance changes
            - g_initial_S: List of initial conductances
            - g_final_S: List of final conductances
            - execution_path / timing: one entry per point
            - base_params: Copy of base parameters

    Raises:
        ValueError: the sweep is not producible (nothing was touched).
        LuaExecutionError: an on-instrument point failed. The outputs are OFF,
            the points completed so far are on the exception as
            `partial_results`, and the failed point is NOT recorded.
    """
    if stim_ch == read_ch:
        raise ValueError("Stim and Read channels must be different for STDP measurements")

    # The whole Δt list is validated before any instrument access, queries
    # included — see _validate_stdp_sweep.
    _validate_stdp_sweep(base_params, delta_t_list_ms)

    # Adopt the instrument's auto-detected mains frequency unless overridden.
    resolve_line_freq(instrument, base_params)

    # Check instrument capability for stimulus level
    if base_params.get('stim_drive_type') == 'I':
        max_current = get_max_current(instrument)
        stim_current = abs(base_params.get('stim_level', 0))
        if stim_current > max_current:
            model = validate_instrument_model(instrument)
            raise ValueError(
                f"ERROR: STDP stimulus level {stim_current}A exceeds instrument capability.\n"
                f"Connected: {model} (Max: {max_current}A)"
            )
    elif base_params.get('stim_drive_type') == 'V':
        max_voltage = get_max_voltage(instrument)
        stim_voltage = abs(base_params.get('stim_level', 0))
        if stim_voltage > max_voltage:
            model = validate_instrument_model(instrument)
            raise ValueError(
                f"ERROR: STDP stimulus level {stim_voltage}V exceeds instrument capability.\n"
                f"Connected: {model} (Max: {max_voltage}V)"
            )

    results = {
        "delta_t_ms": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy(),
        # The one convention used by the spec, the learning rule, the fitter
        # and now the instrument (M28). Stamped into the results so exported
        # data is self-describing.
        "delta_t_convention": "onset_to_onset",
    }

    # Configure sense mode (4-wire/2-wire) if specified
    _apply_wire_modes(instrument, base_params, [stim_ch, read_ch])

    # Record instrument voltage ceiling so STDP LUA generation picks native ranges.
    base_params['max_voltage'] = get_max_voltage(instrument)

    # The post-synaptic spike is a voltage on the read channel whatever the
    # pre-synaptic drive is, so it is checked against the voltage ceiling too.
    post_voltage = abs(base_params.get('post_spike_level', base_params['stim_level']))
    if post_voltage > base_params['max_voltage']:
        raise ValueError(
            f"ERROR: STDP post-spike level {post_voltage}V exceeds the instrument "
            f"ceiling ({base_params['max_voltage']}V)")

    # Check if instrument supports LUA for fast deterministic timing.
    #
    # H3: this used to be the loop's only gate, and the per-Δt exception
    # handler CLEARED it without ever restoring it. One transient failure at
    # any Δt therefore silently demoted every remaining point in the sweep to
    # the standard path — and for STDP the standard path is not merely slower,
    # it is a different protocol (see below). The resulting array mixed two
    # protocols with nothing recording which point came from which.
    #
    # `lua_supported` is now fixed for the sweep and a per-point decision is
    # taken from it, so a failure at one Δt cannot affect the next. There is
    # no fall-back from a failed LUA point at all: once its script has been
    # sent the device may have been stimulated, and repeating the point on the
    # host-timed path would both double-stimulate it and mix two protocols.
    lua_supported = supports_lua_execution(instrument)
    results["execution_path"] = []
    results["timing"] = []

    # measure.delay is 0 in the script, so the settle is the ONLY settling the
    # read gets; whether it suffices for the range is recorded, not assumed.
    settle_warning = check_read_settle(
        base_params.get('settle_ms', 5) / 1000.0,
        get_fixed_current_range(base_params['compliance_A']))
    results["read_settle_warning"] = settle_warning
    if settle_warning:
        print(f"⚠️  STDP: {settle_warning}")

    # print(number) honours format.asciiprecision (default 6 digits).
    if lua_supported:
        _set_ascii_precision(instrument)

    for delta_t_ms in delta_t_list_ms:
        use_lua = lua_supported
        if use_lua:
            # === FAST LUA PATH: Deterministic millisecond-level timing ===
            print(f"STDP Δt={delta_t_ms}ms: Using LUA for deterministic timing...")

            # Generation only raises on a configuration error, which the
            # standard path would hit too, so it is not guarded.
            lua_script = generate_stdp_lua_script(stim_ch, read_ch, base_params, delta_t_ms)
            n_pairs = base_params.get('n_pulses', 50)
            pair_period_ms = base_params.get('stim_period_ms', 200)
            estimated_time_s = (n_pairs * pair_period_ms) / 1000.0 + 2.0

            try:
                print("Executing STDP LUA script on instrument...")
                lua_timing = _upload_and_execute_lua(instrument, lua_script, estimated_time_s)

                errors = _read_error_queue(instrument)
                if errors:
                    raise LuaExecutionError(
                        "the instrument reported error(s) during the script", [], errors)
                I_initial, I_final = _read_stdp_results(instrument)
            except Exception as e:
                problem = _outputs_off(instrument)
                queue = e.instrument_errors if isinstance(e, LuaExecutionError) and e.instrument_errors \
                    else _read_error_queue(instrument)
                fault = LuaExecutionError(
                    f"STDP FAILED at delta_t = {delta_t_ms} ms ({e}).\n"
                    f"  Outputs: {problem or 'switched OFF'}\n"
                    f"  Instrument error queue: {queue if queue else 'empty'}\n"
                    f"  {len(results['delta_t_ms'])} earlier point(s) are on this exception as "
                    "`partial_results`; this point is NOT recorded and was NOT repeated on "
                    "another path, because the device may already have been stimulated.",
                    instrument_errors=queue)
                fault.partial_results = results
                raise fault from e

            lua_timing = dict(lua_timing)
            lua_timing['requested_pair_period_ms'] = pair_period_ms

            # Calculate conductances. NaN at zero read bias, never 0 —
            # see the execution-path parity invariant in CLAUDE.md.
            read_voltage = base_params['read_voltage']
            g_initial = I_initial / read_voltage if read_voltage != 0 else float('nan')
            g_final = I_final / read_voltage if read_voltage != 0 else float('nan')

            if g_initial and np.isfinite(g_initial) and g_initial != 0:
                print(f"  LUA execution complete: dG = "
                      f"{((g_final - g_initial) / g_initial * 100):.2f}%")
            else:
                print("  LUA execution complete")
            if lua_timing.get('overruns'):
                print(f"⚠️  {int(lua_timing['overruns'])}/{n_pairs} spike pairs could NOT keep the "
                      f"{pair_period_ms:g} ms pair period (worst shortfall "
                      f"{lua_timing.get('worst_ms', float('nan')):.3f} ms).")
            point_timing = lua_timing

        if not use_lua:
            # === STANDARD PATH: Python-controlled timing ===
            #
            # Both spikes are driven by LEVEL CHANGES (`source.levelv`), never
            # by OUTPUT_ON/OUTPUT_OFF. CLAUDE.md forbids relay toggling for
            # STDP spikes and the LUA generator correctly avoids it: relay
            # latency is milliseconds and non-deterministic, so at a 1 ms pulse
            # width and Delta_t = 5 ms the toggling path was not controlling
            # Delta_t at all — it was measuring relay jitter. The outputs are
            # enabled once, before the pair loop, and stay on.
            #
            # Host-clock timing remains inherently less precise than the
            # instrument timer, which is why LUA is preferred and why every
            # point records which path produced it.
            pulse_width_s = base_params['stim_width_ms'] / 1000.0
            settle_ms = base_params.get('settle_ms', 5)
            n_pairs = base_params.get('n_pulses', 50)
            pair_period_ms = base_params.get('stim_period_ms', 200)
            delta_t_s = delta_t_ms / 1000.0
            read_voltage = base_params['read_voltage']
            post_level = base_params.get('post_spike_level', base_params['stim_level'])
            stim_attr = 'levelv' if base_params.get('stim_drive_type', 'V') == 'V' else 'leveli'

            # NPLC and averaging from the shared derivation, not a hardcoded 0.1
            # that discarded whatever the GUI asked for.
            nplc_std, avg_std = derive_measurement_averaging(base_params)

            # Configure channels
            instrument.write(f"localnode.linefreq = {int(_line_freq_hz(base_params))}")
            if stim_attr == 'levelv':
                instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS")
                instrument.write(f"{stim_ch}.source.limiti = {base_params['compliance_A']}")
            else:
                instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS")
                instrument.write(f"{stim_ch}.source.rangei = "
                                 f"{_get_fixed_current_range(abs(base_params['stim_level']))}")
                instrument.write(f"{stim_ch}.source.limitv = "
                                 f"{_current_drive_voltage_limit(base_params, 'STDP')}")
            instrument.write(f"{stim_ch}.source.{stim_attr} = 0")
            instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
            instrument.write(f"{read_ch}.source.limiti = {base_params['compliance_A']}")
            instrument.write(f"{read_ch}.source.levelv = 0")
            try:
                for line in _emit_read_adc_setup(read_ch, nplc_std, avg_std):
                    instrument.write(line)

                # Outputs on once, for the whole point — no relay activity inside
                # the timing-critical pair loop.
                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")

                def _read_g():
                    instrument.write(f"{read_ch}.source.levelv = {read_voltage}")
                    time.sleep(settle_ms / 1000.0)
                    current = float(instrument.query(f"print({read_ch}.measure.i())").strip())
                    instrument.write(f"{read_ch}.source.levelv = 0")
                    # NaN at zero read bias, never a fabricated 0.
                    return (current / read_voltage) if read_voltage != 0 else float('nan')

                # Measure initial conductance
                g_initial = _read_g()

                # Delta_t is ONSET-TO-ONSET, matching the LUA path and the spec —
                # see the note in generate_stdp_lua_script (M28). The silent gap
                # between the pulses is therefore |Delta_t| minus one pulse width.
                inter_pulse_gap_s = abs(delta_t_s) - pulse_width_s
                if inter_pulse_gap_s < 0:
                    raise ValueError(
                        f"STDP Delta_t = {delta_t_ms} ms is smaller than the pulse "
                        f"width ({base_params['stim_width_ms']} ms). Delta_t is "
                        "defined onset-to-onset, so the spikes would have to "
                        "overlap, which this sequential generator cannot produce."
                    )

                # Apply spike pairs
                for pair_idx in range(n_pairs):
                    if delta_t_ms >= 0:
                        # Pre before post
                        instrument.write(f"{stim_ch}.source.{stim_attr} = {base_params['stim_level']}")
                        time.sleep(pulse_width_s)
                        instrument.write(f"{stim_ch}.source.{stim_attr} = 0")

                        if inter_pulse_gap_s > 0:
                            time.sleep(inter_pulse_gap_s)

                        instrument.write(f"{read_ch}.source.levelv = {post_level}")
                        time.sleep(pulse_width_s)
                        instrument.write(f"{read_ch}.source.levelv = 0")
                    else:
                        # Post before pre
                        instrument.write(f"{read_ch}.source.levelv = {post_level}")
                        time.sleep(pulse_width_s)
                        instrument.write(f"{read_ch}.source.levelv = 0")

                        if inter_pulse_gap_s > 0:
                            time.sleep(inter_pulse_gap_s)

                        instrument.write(f"{stim_ch}.source.{stim_attr} = {base_params['stim_level']}")
                        time.sleep(pulse_width_s)
                        instrument.write(f"{stim_ch}.source.{stim_attr} = 0")

                    time.sleep(pair_period_ms / 1000.0)

                # Measure final conductance
                g_final = _read_g()

                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
                point_timing = {'status': 'host_clock',
                                'reason': 'PC-timed path: spike timing is not measured by the instrument'}
            finally:
                # put back the measure.delay / autozero the setup saved
                for line in _emit_restore_adc(read_ch):
                    instrument.write(line)

        # Provenance for this point: LUA timing and host timing are not the
        # same measurement, and an array that mixes them must say so.
        results["execution_path"].append('lua' if use_lua else 'standard')
        results["timing"].append(point_timing)

        # Calculate and store results
        delta_g = g_final - g_initial
        delta_g_percent = (delta_g / g_initial) * 100 if g_initial != 0 else float('nan')

        results["delta_t_ms"].append(delta_t_ms)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(g_initial)
        results["g_final_S"].append(g_final)

        # Recovery time between delta_t measurements
        time.sleep(1.0)

    return results

def measure_retention(instrument, read_ch, params):
    """
    Measure conductance decay over time after potentiation.
    
    This is crucial for extracting decay_tau parameter.
    
    IMPORTANT USAGE:
        1. First potentiate device to high state (run LTP measurement)
        2. THEN immediately call this function to measure decay
        3. Do NOT turn off instrument between potentiation and retention measurement
    
    Args:
        instrument: PyVISA instrument resource
        read_ch (str): Read channel ('smua' or 'smub')
        params (dict): Measurement parameters containing:
            - read_voltage (float): Read bias voltage in V
            - retention_times_s (list): Wait times in seconds [1, 10, 30, 100, 300, 1000]
            - compliance_A (float): Current compliance in A
            - wavelength_nm (float, optional): Wavelength for metadata
            - intensity_mW_cm2 (float, optional): Light intensity for metadata
            - sample_id (str, optional): Sample identifier
            - nplc (float, optional): Integration time (default: 1.0)
            - wire_mode (str, optional): '2-Wire' or '4-Wire'
    
    Returns:
        dict: {
            'time_s': [t0, t1, t2, ...],
            'conductance_S': [G0, G1, G2, ...],
            'I_A': [I0, I1, I2, ...],
            'params': {...},
            'metadata': {...}
        }
    
    Example:
        # Step 1: Potentiate device to high state
        pot_params = {
            'mode': 'electrical',
            'stim_drive_type': 'V',
            'stim_level': 1.0,
            'stim_width_ms': 100,
            'stim_period_ms': 200,
            'n_pulses': 50,
            'read_voltage': 0.1,
            'read_delay_ms': 10,
            'compliance_A': 100e-6,
            'wavelength_nm': 550,
            'intensity_mW_cm2': 20,
            'sample_id': 'Device_001'
        }
        ltp_result = pulse_read_sequence(instrument, stim_ch, read_ch, pot_params)
        
        # Step 2: Immediately measure retention (device still potentiated)
        retention_params = {
            'read_voltage': 0.1,
            'retention_times_s': [1, 10, 30, 100, 300, 1000],
            'compliance_A': 100e-6,
            'wavelength_nm': 550,
            'intensity_mW_cm2': 20,
            'sample_id': 'Device_001',
            'nplc': 1.0,
            'wire_mode': '2-Wire'
        }
        retention_result = measure_retention(instrument, read_ch, retention_params)
        
        # Step 3: Use with assembly_helpers
        from assembly_helpers import make_retention_trace
        retention_dataset = make_retention_trace(retention_result)
    """
    results = {
        'time_s': [],
        'conductance_S': [],
        'I_A': [],
        'params': params.copy(),
        'metadata': {
            'wavelength_nm': params.get('wavelength_nm', None),
            'intensity_mW_cm2': params.get('intensity_mW_cm2', None),
            'sample_id': params.get('sample_id', 'unknown'),
            'measurement_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'measurement_type': 'retention'
        }
    }
    
    # Extract parameters
    read_voltage = params['read_voltage']
    retention_times = params.get('retention_times_s', [1, 10, 30, 100, 300, 1000])
    compliance_A = params['compliance_A']
    settle_ms = params.get('settle_ms', 5)
    nplc = params.get('nplc', 1.0)
    
    _check_times_ms(nonneg={'settle_ms': settle_ms,
                            **{f'retention_times_s[{i}] (ms)': t * 1000.0
                               for i, t in enumerate(retention_times)}})

    # Configure sense mode if specified
    _apply_wire_modes(instrument, params, [read_ch])

    # Synchronise ADC to mains for normal-mode rejection of line-frequency
    # pickup, adopting the instrument's auto-detected frequency unless the
    # caller specified one.
    instrument.write(f"localnode.linefreq = {int(resolve_line_freq(instrument, params))}")

    # Configure read channel for retention measurement
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.limiti = {compliance_A}")
    # Autorange is kept deliberately: the signal decays over decades, so no
    # single fixed range suits every point. With autorange the range can change
    # right before a read, so the instrument's own range-dependent DELAY_AUTO
    # is the settling that read needs; retention reads are seconds apart, so
    # its cost is irrelevant here (unlike the pulse paths, which set delay 0).
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_ON")
    for line in _emit_read_adc_setup(read_ch, nplc, 1, measure_delay="auto"):
        instrument.write(line)
    
    print(f"\n=== Retention Measurement Started ===")
    print(f"Sample: {params.get('sample_id', 'unknown')}")
    print(f"Read voltage: {read_voltage} V")
    print(f"Measurement points: {len(retention_times)}")
    print(f"Total duration: ~{sum(retention_times)} seconds ({sum(retention_times)/60:.1f} minutes)")
    print("\nMeasuring conductance decay...")
    
    t_start = time.time()
    
    for i, t_wait in enumerate(retention_times):
        # Wait for specified time
        print(f"  Waiting {t_wait}s...", end=" ", flush=True)
        time.sleep(t_wait)
        
        # Measure conductance. The reads are minutes apart, so the zero
        # reference is refreshed before each one (AUTOZERO_ONCE costs two
        # conversions; a cached zero drifts with temperature over that time).
        instrument.write(f"{read_ch}.measure.autozero = {read_ch}.AUTOZERO_ONCE")
        instrument.write(f"{read_ch}.source.levelv = {read_voltage}")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
        time.sleep(settle_ms / 1000.0)
        
        # Measure current
        I_measured = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        
        # Conductance is undefined at zero read bias: NaN, never a fabricated 0
        G_measured = I_measured / read_voltage if read_voltage != 0 else float('nan')
        
        # Level to 0 BEFORE the output goes off
        instrument.write(f"{read_ch}.source.levelv = 0")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        
        # Calculate elapsed time from start
        t_elapsed = time.time() - t_start
        
        # Store results
        results['time_s'].append(t_elapsed)
        results['conductance_S'].append(G_measured)
        results['I_A'].append(I_measured)
        
        # Display progress
        print(f"G = {G_measured*1e6:.2f} µS (t = {t_elapsed:.1f}s)")

    for line in _emit_restore_adc(read_ch):
        instrument.write(line)

    print(f"\n✓ Retention measurement complete!")
    G_first, G_last = results['conductance_S'][0], results['conductance_S'][-1]
    print(f"  Initial G: {G_first*1e6:.2f} µS")
    print(f"  Final G: {G_last*1e6:.2f} µS")
    if np.isfinite(G_first) and np.isfinite(G_last) and G_first != 0:
        print(f"  Decay: {(1 - G_last/G_first)*100:.1f}%")
    else:
        print("  Decay: not defined (initial conductance is NaN or zero)")
    
    return results


def simulate_srdp(base_params, freq_list_hz):
    """
    Simulates SRDP behavior with frequency-dependent plasticity.
    Uses LINEAR conductance change model for realistic behavior.
    """
    results = {
        "frequencies_hz": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    # Calculate minimum period based on actual parameters
    NPLC_PERIOD_MS = 1000.0 / _line_freq_hz(base_params)
    nplc = base_params.get('nplc', 1.0)
    read_width_ms = base_params.get('read_width_ms', None)
    if read_width_ms is not None and read_width_ms > 0:
        actual_nplc = min(nplc, read_width_ms / NPLC_PERIOD_MS)
        actual_nplc = max(0.001, actual_nplc)
        meas_time_ms = actual_nplc * NPLC_PERIOD_MS
    else:
        meas_time_ms = nplc * NPLC_PERIOD_MS
    min_period = base_params['stim_width_ms'] + base_params['read_delay_ms'] + meas_time_ms + 0.5
    max_freq = 1000.0 / min_period

    print(f"\nSRDP Simulation Info:")
    print(f"  Min period: {min_period:.1f} ms")
    print(f"  Max theoretical freq: {max_freq:.2f} Hz")
    print(f"  Requested range: {min(freq_list_hz):.1f} - {max(freq_list_hz):.1f} Hz")

    for freq_hz in freq_list_hz:
        # Calculate period from frequency
        period_ms = 1000.0 / freq_hz

        # Warn if too fast (simulation will still run, just noting the constraint)
        if period_ms < min_period:
            print(f"  ⚠️  {freq_hz:.1f} Hz: period {period_ms:.1f} ms < instrument min {min_period:.1f} ms")
        
        print(f"  Testing {freq_hz:.1f} Hz (period {period_ms:.1f} ms)...", end=" ")
        
        # Initial conductance
        G_initial = 1e-6  # 1 µS
        
        # Frequency-dependent total change (SRDP behavior)
        # Higher frequency → stronger total plasticity
        freq_factor = np.log10(freq_hz + 1) / np.log10(max_freq + 1)
        
        # Total conductance change as percentage (realistic range)
        if base_params['stim_level'] > 0:
            # LTP: 10% to 100% total change depending on frequency
            base_change_percent = 10 + freq_factor * 90
        else:
            # LTD: -10% to -70% total change
            base_change_percent = -(10 + freq_factor * 60)
        
        # Add variability (±5%)
        noise_factor = 1 + np.random.uniform(-0.05, 0.05)
        total_change_percent = base_change_percent * noise_factor
        
        # Calculate final conductance
        g_final = G_initial * (1 + total_change_percent / 100)
        
        # Clip to physical limits
        G_max = 1e-4  # 100 µS
        G_min = 1e-7  # 0.1 µS
        g_final = np.clip(g_final, G_min, G_max)
        
        # Calculate actual change
        delta_g = g_final - G_initial
        delta_g_percent = (delta_g / G_initial) * 100
        
        # Store results
        results["frequencies_hz"].append(freq_hz)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(G_initial)
        results["g_final_S"].append(g_final)
        
        print(f"ΔG = {delta_g_percent:.2f}%")
        
        # Simulate measurement time (0.1s per frequency)
        time.sleep(0.1)
    
    print(f"\n✓ Completed {len(results['frequencies_hz'])}/{len(freq_list_hz)} frequency points")
    
    # Provenance: this data came from the SIMULATION model, not from an
    # instrument. Without this flag the returned dict, the exported CSV and
    # the plots are indistinguishable from a real measurement, and synthetic
    # numbers can travel the whole CSV -> JSON -> fit pipeline unchallenged.
    results['simulated'] = True
    results['data_source'] = 'synthetic_model'

    return results


def simulate_stdp(base_params, delta_t_list_ms):
    """
    Simulates STDP by modeling pre-post spike pairs with realistic timing.
    Uses the pulse simulation engine to generate honest data.
    """
    results = {
        "delta_t_ms": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    n_pairs = base_params.get('n_pulses', 50)
    pair_period_ms = base_params.get('stim_period_ms', 200)
    
    # Initial conductance baseline
    G_baseline = 1e-6  # 1 µS
    
    for delta_t_ms in delta_t_list_ms:
        # Measure initial conductance (simulate read)
        g_initial = G_baseline * (1 + np.random.uniform(-0.02, 0.02))
        
        # Simulate STDP effect based on timing
        # Classic exponential STDP window
        A_plus = 0.5   # LTP amplitude  
        A_minus = 0.3  # LTD amplitude
        tau_plus = 20  # LTP time constant (ms)
        tau_minus = 20 # LTD time constant (ms)
        
        if delta_t_ms > 0:
            # Pre before Post → LTP
            weight_change = A_plus * np.exp(-delta_t_ms / tau_plus)
        else:
            # Post before Pre → LTD  
            weight_change = -A_minus * np.exp(delta_t_ms / tau_minus)
        
        # Apply n_pairs of spike pairs (cumulative effect)
        # Model: Each pair contributes a fraction of the total change
        total_change = weight_change * n_pairs * 0.01  # Scale by number of pairs
        
        # Calculate final conductance
        g_final = g_initial * (1 + total_change)
        
        # Add measurement noise
        g_final *= (1 + np.random.uniform(-0.02, 0.02))
        
        # Clip to physical limits
        G_max = 1e-4  # 100 µS
        G_min = 1e-7  # 0.1 µS
        g_final = np.clip(g_final, G_min, G_max)
        
        # Calculate changes
        delta_g = g_final - g_initial
        delta_g_percent = (delta_g / g_initial) * 100 if g_initial != 0 else float('nan')
        
        # Store results
        results["delta_t_ms"].append(delta_t_ms)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(g_initial)
        results["g_final_S"].append(g_final)
        
        # Simulate measurement time
        time.sleep(0.01)
    
    # Provenance: this data came from the SIMULATION model, not from an
    # instrument. Without this flag the returned dict, the exported CSV and
    # the plots are indistinguishable from a real measurement, and synthetic
    # numbers can travel the whole CSV -> JSON -> fit pipeline unchallenged.
    results['simulated'] = True
    results['data_source'] = 'synthetic_model'

    return results


# =============================================================================
# MODULE TEST (optional - can be removed in production)
# =============================================================================

if __name__ == "__main__":
    """Test the module with simulation"""
    print("Testing synapse_engine module...")
    
    # Test parameters
    test_params = {
        "mode": "electrical",
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 10,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "settle_ms": 5,
        "measure_avg": 1
    }
    
    # Run simulation
    print("\nRunning simulation...")
    results = simulate_pulse_read(test_params)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_synapse_metrics(results)
    
    # Display results
    print("\n=== Synapse Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save to file
    print("\nSaving to test_synapse_data.csv...")
    process_and_save_synapse_data(results, "test_synapse_data.csv")
    print("✓ Test complete!")