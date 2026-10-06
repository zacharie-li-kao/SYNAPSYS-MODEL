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


def generate_pulse_read_lua_script(stim_ch, read_ch, params):
    """
    Generates LUA script for fast pulse-read sequences on Keithley.
    Optimized for 2636A compatibility.
    """

    # Extract parameters
    stim_drive_type = params['stim_drive_type']
    stim_level = params['stim_level']
    stim_width_ms = params['stim_width_ms']
    stim_period_ms = params['stim_period_ms']
    n_pulses = params['n_pulses']
    read_voltage = params['read_voltage']
    read_delay_ms = params.get('read_delay_ms', 10)
    read_settle_delay_ms = params.get('read_settle_delay_ms', 5)
    compliance_A = params['compliance_A']
    settle_ms = params.get('settle_ms', 5)
    max_voltage = params.get('max_voltage', 10.0)

    line_freq_hz = _line_freq_hz(params)
    NPLC_PERIOD_MS = 1000.0 / line_freq_hz
    nplc, measure_avg = derive_measurement_averaging(params)

    # Validation checks
    KEITHLEY_MAX_BUFFER_SIZE = 50000
    if n_pulses + 1 > KEITHLEY_MAX_BUFFER_SIZE:
        raise ValueError(f"Pulse count ({n_pulses}) exceeds buffer capacity.")

    measurement_time_ms = nplc * NPLC_PERIOD_MS * measure_avg
    # Per-cycle TSP command overhead in the LUA path (not physical relay
    # settling — TSP source.output writes return immediately and the
    # relay closes asynchronously while delay() runs). Measured on a
    # 2636B: ~2 ms of cumulative command latency per pulse (two
    # source.output writes + two source.levelv writes + measure.i call
    # + timer.measure.t check + buffer append).
    LUA_OVERHEAD_MS = 2.0
    min_required_period = (stim_width_ms + read_delay_ms + read_settle_delay_ms
                           + measurement_time_ms + LUA_OVERHEAD_MS)

    if stim_period_ms < min_required_period:
        print(f"⚠️  Requested period ({stim_period_ms:.2f} ms) shorter than minimum for zero-drift "
              f"({min_required_period:.2f} ms = stim {stim_width_ms:g} + read_delay {read_delay_ms:g} + "
              f"settle {read_settle_delay_ms:g} + meas {measurement_time_ms:.2f} + "
              f"LUA overhead {LUA_OVERHEAD_MS:g}). Instrument will run as fast as possible.")
    
    # Convert times to seconds for LUA
    stim_width_s = stim_width_ms / 1000.0
    stim_period_s = stim_period_ms / 1000.0
    read_delay_s = read_delay_ms / 1000.0
    read_settle_delay_s = read_settle_delay_ms / 1000.0
    settle_s = settle_ms / 1000.0

    fixed_i_range = get_fixed_current_range(compliance_A)

    # Generate LUA script
    lua_script = f"""
-- Fast Synapse Pulse-Read Sequence (2636A Compatible)
-- Read voltage is TOGGLED per pulse (0V idle, read_voltage during measure)
-- to match standard execution path behavior.

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local n_pulses = {n_pulses}
local stim_level = {stim_level}
local stim_width = {stim_width_s}
local stim_period = {stim_period_s}
local read_voltage = {read_voltage}
local read_delay = {read_delay_s}
local read_settle = {read_settle_delay_s}
local settle_time = {settle_s}
local measure_avg = {measure_avg}

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
{read_ch}.measure.autozero = {read_ch}.AUTOZERO_OFF
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
-- (Filter state is set explicitly later, below the stim configuration —
--  no need to pre-disable it here, which only created a transient OFF.)

{read_ch}.source.output = {read_ch}.OUTPUT_ON
delay(settle_time)

-- Configure stimulus channel"""

    if stim_drive_type == 'V':
        fixed_v_range = _get_fixed_voltage_range(stim_level, max_voltage)
        lua_script += f"""
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}"""
    else:
        lua_script += f"""
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS
{stim_ch}.source.leveli = 0
{stim_ch}.source.rangei = {fixed_i_range}
{stim_ch}.source.limitv = {max_voltage}"""

    lua_script += f"""
-- Stim channel starts OFF (relay open) — turned ON/OFF per pulse
-- to physically disconnect the load between pulses.
-- Level changes alone cannot discharge capacitive loads fast enough
-- when current is compliance-limited.

-- Configure measurement.
-- HARDWARE AVERAGING via the 2600 repeat-average filter:
--   filter.count = N integrations averaged per measure.i() call
--   filter.type  = REPEAT_AVG (true arithmetic mean of N consecutive ADCs)
--   measure.count = 1   →  one (averaged) value appended to buffer per pulse
-- This preserves the 1-pulse-1-buffer-entry contract the host expects.
-- Setting measure.count = N instead writes N raw readings to the buffer
-- per pulse, which the retrieval loop then mis-aligns.
{read_ch}.measure.count = 1
{read_ch}.measure.nplc = {nplc}
{read_ch}.measure.filter.count = {measure_avg if measure_avg > 1 else 1}
{read_ch}.measure.filter.type = {read_ch}.FILTER_REPEAT_AVG
{read_ch}.measure.filter.enable = {read_ch}.{'FILTER_ON' if measure_avg > 1 else 'FILTER_OFF'}

-- Reset timer before the initial read so every buffer timestamp shares a
-- single reference. Otherwise t[0] would sit on whatever the timer held
-- before this script, which skews the first reported interval by tens of
-- milliseconds and contaminates the drift summary.
timer.reset()

-- Initial read (toggle read voltage: 0V → read_voltage → measure → 0V)
{read_ch}.source.levelv = read_voltage
delay(read_settle)
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
    -- Stimulus ON: set level then close relay"""

    if stim_drive_type == 'V':
        lua_script += f"""
    {stim_ch}.source.levelv = stim_level"""
    else:
        lua_script += f"""
    {stim_ch}.source.leveli = stim_level"""

    lua_script += f"""
    {stim_ch}.source.output = {stim_ch}.OUTPUT_ON
    delay(stim_width)
    -- Stimulus OFF: open relay (physically disconnects load)
    {stim_ch}.source.output = {stim_ch}.OUTPUT_OFF

    -- Read: apply read voltage, settle, measure, return to 0V
    delay(read_delay)
    {read_ch}.source.levelv = read_voltage
    delay(read_settle)
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

-- Cleanup
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Timing report: how often the requested period could NOT be met, and by how
-- much at worst. Printed before the completion token so the host records what
-- the instrument actually delivered rather than what was asked for.
print(string.format("{LUA_TIMING_TOKEN} overruns=%d worst_ms=%.4f n_pulses=%d",
                    overruns, worst_overrun * 1000, n_pulses))

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return lua_script


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
        str: LUA script for visual synapse measurement
    """

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
    FILTER_COUNT_MAX = 100  # 2600-series measure.filter.count hardware limit
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
    measure_end_margin_s = measure_end_margin_ms / 1000.0

    # The pulse must contain its own light-on time.
    if pulse_period_s < pulse_width_s:
        raise ValueError(f"Pulse period ({pulse_period_ms}ms) must be >= pulse width ({pulse_width_ms}ms)")

    # Determine optimal fixed range for current measurement
    def get_fixed_current_range(compliance_A):
        ranges = [100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
        for r in ranges:
            if compliance_A <= r:
                return r
        return ranges[-1]

    fixed_i_range = get_fixed_current_range(compliance_A)

    # Determine voltage range for LED driver (model-aware: native ranges differ
    # between the 40 V and 200 V families).
    max_voltage = params.get('max_voltage', 200.0)
    fixed_v_range = _get_fixed_voltage_range(light_voltage, max_voltage)

    # Generate LUA script
    lua_script = f"""
-- Visual (Self-Powered) Synapse Characterization
-- Measures photocurrent (Jsc) at 0V during light pulses

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local stim_ch = {stim_ch}
local read_ch = {read_ch}
local n_pulses = {n_pulses}
local light_voltage = {light_voltage}
local pulse_width = {pulse_width_s}
local measure_start_delay = {measure_start_delay_s}
local readings_per_pulse = {readings_per_pulse}

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
{read_ch}.source.rangev = 0.2
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autozero = {read_ch}.AUTOZERO_OFF
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{read_ch}.measure.nplc = {nplc}
{read_ch}.measure.count = 1

-- Configure hardware averaging filter
{read_ch}.measure.filter.enable = {read_ch}.FILTER_ON
{read_ch}.measure.filter.type = {read_ch}.FILTER_REPEAT_AVG
{read_ch}.measure.filter.count = {readings_per_pulse}

-- Configure stimulus channel (LED driver)
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}

-- Turn on read channel (0V for Jsc); stim starts OFF (relay open)
{read_ch}.source.output = {read_ch}.OUTPUT_ON
delay(0.01)

-- Pulse-Measure loop (absolute-timer based to prevent cumulative drift).
-- The LED is held ON for exactly pulse_width every period, so the delivered
-- optical dose (intensity x pulse_width) is set by pulse_width alone and does
-- NOT depend on the ADC integration time. The photocurrent is measured
-- measure_start_delay into the pulse; host-side clamping guarantees the
-- integration finishes inside the pulse (>= measure_end_margin before LED off).
-- Stim uses OUTPUT_ON/OUTPUT_OFF relay toggling to physically disconnect the
-- LED between pulses — level changes alone cannot discharge capacitive loads
-- fast enough when current is compliance-limited.
local pulse_period = {pulse_period_s}
timer.reset()
local t_pulse_start = 0
for pulse = 1, n_pulses do
    -- Wait out the previous period's tail so this pulse starts on its absolute
    -- grid point (t_pulse_start), independent of accumulated measurement time.
    local t_wait_start = t_pulse_start - timer.measure.t()
    if t_wait_start > 0 then delay(t_wait_start) end

    -- Light pulse ON: set voltage then close relay
    {stim_ch}.source.levelv = light_voltage
    {stim_ch}.source.output = {stim_ch}.OUTPUT_ON

    -- Open the measurement window measure_start_delay into the pulse, then
    -- measure (hardware averaging active; this call blocks for the ADC time).
    local t_meas = t_pulse_start + measure_start_delay - timer.measure.t()
    if t_meas > 0 then delay(t_meas) end
    {read_ch}.measure.i({read_ch}.nvbuffer1)

    -- Hold the light ON until the full pulse_width has elapsed, THEN open the
    -- relay. This absolute LED-off deadline is what makes the delivered dose
    -- equal intensity x pulse_width for every pulse.
    local t_off = t_pulse_start + pulse_width - timer.measure.t()
    if t_off > 0 then delay(t_off) end
    {stim_ch}.source.output = {stim_ch}.OUTPUT_OFF

    -- Advance to the next pulse's absolute start (no cumulative drift)
    t_pulse_start = t_pulse_start + pulse_period
end

-- Cleanup
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Disable filter
{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return lua_script


def generate_visual_continuous_lua_script(stim_ch, read_ch, params):
    """
    Generates LUA script for continuous photocurrent monitoring during light pulses.

    This mode provides I(t) data - continuous current vs time, useful for
    analyzing transient photocurrent dynamics.

    Args:
        stim_ch (str): Stimulus channel for LED ('smua' or 'smub')
        read_ch (str): Read channel for current measurement ('smua' or 'smub')
        params (dict): Measurement parameters

    Returns:
        str: LUA script for continuous visual synapse measurement
    """

    # Extract parameters
    light_voltage = params['light_pulse_voltage']
    pulse_width_ms = params['pulse_width_ms']
    pulse_period_ms = params['pulse_period_ms']
    n_pulses = params['n_pulses']
    compliance_A = params['compliance_A']
    nplc = params.get('nplc', 0.1)  # Lower NPLC for faster sampling in continuous mode
    sample_interval_ms = params.get('sample_interval_ms', 1.0)  # How often to sample
    line_freq_hz = _line_freq_hz(params)

    # Convert times to seconds
    pulse_width_s = pulse_width_ms / 1000.0
    pulse_period_s = pulse_period_ms / 1000.0
    sample_interval_s = sample_interval_ms / 1000.0

    # Calculate total measurement time and expected samples
    total_time_s = pulse_period_s * n_pulses
    expected_samples = int(total_time_s / sample_interval_s) + n_pulses * 2  # Extra for safety

    # Check buffer capacity
    KEITHLEY_MAX_BUFFER_SIZE = 50000
    if expected_samples > KEITHLEY_MAX_BUFFER_SIZE:
        raise ValueError(
            f"Expected samples ({expected_samples}) exceeds buffer capacity.\n"
            f"Reduce pulse count or increase sample interval."
        )

    # Determine optimal fixed range
    def get_fixed_current_range(compliance_A):
        ranges = [100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
        for r in ranges:
            if compliance_A <= r:
                return r
        return ranges[-1]

    fixed_i_range = get_fixed_current_range(compliance_A)
    max_voltage = params.get('max_voltage', 200.0)
    fixed_v_range = _get_fixed_voltage_range(light_voltage, max_voltage)

    # Calculate samples per pulse period
    samples_per_period = int(pulse_period_s / sample_interval_s)
    samples_during_pulse = int(pulse_width_s / sample_interval_s)
    samples_between_pulses = samples_per_period - samples_during_pulse

    lua_script = f"""
-- Visual Synapse Continuous Mode - I(t) Measurement
-- Continuously samples photocurrent during light pulse sequence

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local stim_ch = {stim_ch}
local read_ch = {read_ch}
local n_pulses = {n_pulses}
local light_voltage = {light_voltage}
local pulse_width = {pulse_width_s}
local sample_interval = {sample_interval_s}
local samples_during_pulse = {samples_during_pulse}
local samples_between_pulses = {samples_between_pulses}

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
{read_ch}.source.rangev = 0.2
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autozero = {read_ch}.AUTOZERO_OFF
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{read_ch}.measure.nplc = {nplc}
{read_ch}.measure.count = 1
{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF

-- Configure stimulus channel
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.levelv = 0
{stim_ch}.source.rangev = {fixed_v_range}
{stim_ch}.source.limiti = {compliance_A}

-- Turn on read channel; stim starts OFF (relay open)
{read_ch}.source.output = {read_ch}.OUTPUT_ON
delay(0.01)

-- Continuous sampling during pulse sequence (timer-based to prevent cumulative drift)
-- Stim uses relay toggling to physically disconnect LED between pulses.
timer.reset()
local sample_count = 0
for pulse = 1, n_pulses do
    -- Light pulse ON: set voltage then close relay
    {stim_ch}.source.levelv = light_voltage
    {stim_ch}.source.output = {stim_ch}.OUTPUT_ON

    -- Sample during pulse
    for s = 1, samples_during_pulse do
        sample_count = sample_count + 1
        local t_target = sample_count * sample_interval
        local t_wait = t_target - timer.measure.t()
        if t_wait > 0 then delay(t_wait) end
        {read_ch}.measure.i({read_ch}.nvbuffer1)
    end

    -- Light pulse OFF: open relay
    {stim_ch}.source.output = {stim_ch}.OUTPUT_OFF

    -- Sample between pulses
    if pulse < n_pulses and samples_between_pulses > 0 then
        for s = 1, samples_between_pulses do
            sample_count = sample_count + 1
            local t_target = sample_count * sample_interval
            local t_wait = t_target - timer.measure.t()
            if t_wait > 0 then delay(t_wait) end
            {read_ch}.measure.i({read_ch}.nvbuffer1)
        end
    end
end

-- Final samples after last pulse
for s = 1, math.min(10, samples_between_pulses) do
    sample_count = sample_count + 1
    local t_target = sample_count * sample_interval
    local t_wait = t_target - timer.measure.t()
    if t_wait > 0 then delay(t_wait) end
    {read_ch}.measure.i({read_ch}.nvbuffer1)
end

-- Cleanup
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return lua_script


# =============================================================================
# LUA SCRIPT FOR UNIFIED CYCLE MODE (SINGLE/DUAL CHANNEL, MULTI-TRAIN)
# =============================================================================

KEITHLEY_MAX_BUFFER_SIZE = 50000


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
    - Single-channel: Same SMU writes and reads (voltage switching)
    - Dual-channel: One SMU writes, another reads at constant bias

    Args:
        train_a_config (dict): Train A configuration (see _validate_train_config)
        train_b_config (dict or None): Train B configuration, or None
        n_cycles (int): Number of A-B cycle repetitions
        inter_train_delay_ms (float): Delay between Train A end and Train B start
        inter_cycle_delay_ms (float): Delay between end of last train and next cycle

    Returns:
        tuple: (lua_script: str, buffer_map: dict)
    """
    # --- Validate ---
    _validate_train_config(train_a_config, "Train A")
    if train_b_config is not None:
        _validate_train_config(train_b_config, "Train B")
    if n_cycles <= 0:
        raise ValueError("n_cycles must be > 0")

    # Mains line frequency drives the ADC integration aperture and rejection.
    # Both trains run on the same instrument/mains, so resolve once from Train A.
    line_freq_hz = _line_freq_hz(train_a_config)

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

    buffer_map = {
        'train_a': {
            'channel': a_read, 'buffer': 'nvbuffer1',
            'n_points_per_cycle': pts_a_per_cycle,
            'n_total': pts_a_per_cycle * n_cycles
        }
    }
    if train_b_config:
        buffer_map['train_b'] = {
            'channel': b_read, 'buffer': 'nvbuffer1',
            'n_points_per_cycle': pts_b_per_cycle,
            'n_total': pts_b_per_cycle * n_cycles
        }

    # --- Collect all unique channels ---
    channels_used = {train_a_config['write_ch'], train_a_config['read_ch']}
    if train_b_config:
        channels_used.update({train_b_config['write_ch'], train_b_config['read_ch']})

    # --- Compute derived values ---
    def train_lua_params(cfg):
        nplc = cfg.get('nplc', 1.0)
        NPLC_PERIOD_MS = 1000.0 / line_freq_hz
        FILTER_COUNT_MAX = 100
        read_width_ms = cfg.get('read_width_ms', None)
        user_override = cfg.get('measure_avg_user')
        measure_avg = 1
        if read_width_ms is not None and read_width_ms > 0:
            max_nplc = read_width_ms / NPLC_PERIOD_MS
            if nplc > max_nplc:
                nplc = max(0.001, max_nplc)
        if user_override is not None and int(user_override) > 0:
            measure_avg = max(1, min(int(user_override), FILTER_COUNT_MAX))
        elif read_width_ms is not None and read_width_ms > 0:
            measure_avg = max(1, min(
                int(read_width_ms / (nplc * NPLC_PERIOD_MS)),
                FILTER_COUNT_MAX,
            ))
        return {
            'nplc': nplc,
            'measure_avg': measure_avg,
            'fixed_i_range': _get_fixed_current_range(cfg['compliance_A']),
            'fixed_v_range': _get_fixed_voltage_range(
                max(abs(cfg['stim_level']), abs(cfg['read_voltage'])),
                cfg.get('max_voltage', 200.0)
            ),
            'fixed_read_v_range': _get_fixed_voltage_range(
                cfg['read_voltage'],
                cfg.get('max_voltage', 200.0)
            ),
            'stim_width_s': cfg['stim_width_ms'] / 1000.0,
            'stim_period_s': cfg['stim_period_ms'] / 1000.0,
            'read_delay_s': cfg['read_delay_ms'] / 1000.0,
            'settle_s': cfg.get('settle_ms', 5) / 1000.0,
        }

    a_lp = train_lua_params(train_a_config)
    b_lp = train_lua_params(train_b_config) if train_b_config else None

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
    # Uses conditional capacity (matches pulse-read pattern): avoids writing
    # capacity when buffer is already large enough or in append mode from a
    # previous run — unconditional writes trigger error -286 on the 2636A.
    lines.append("")
    lines.append("-- Buffer configuration")
    if same_buffer:
        cap = min((pts_a_per_cycle + pts_b_per_cycle) * n_cycles + 100, KEITHLEY_MAX_BUFFER_SIZE)
        lines.append(f"{a_read}.nvbuffer1.fillmode = {a_read}.FILL_ONCE")
        lines.append(f"{a_read}.nvbuffer1.clear()")
        lines.append(f"{a_read}.nvbuffer1.appendmode = 1")
        lines.append(f"{a_read}.nvbuffer1.collecttimestamps = 1")
        lines.append(f"{a_read}.nvbuffer1.collectsourcevalues = 0")
    else:
        for ch, pts in [(a_read, pts_a_per_cycle * n_cycles), (b_read, pts_b_per_cycle * n_cycles)]:
            cap = min(pts + 100, KEITHLEY_MAX_BUFFER_SIZE)
            lines.append(f"{ch}.nvbuffer1.fillmode = {ch}.FILL_ONCE")
            lines.append(f"{ch}.nvbuffer1.clear()")
            lines.append(f"{ch}.nvbuffer1.appendmode = 1")
            lines.append(f"{ch}.nvbuffer1.collecttimestamps = 1")
            lines.append(f"{ch}.nvbuffer1.collectsourcevalues = 0")

    # --- Channel configuration ---
    lines.append("")
    lines.append("-- Channel configuration")

    configured_channels = set()

    def emit_channel_config(cfg, lp, label):
        wch = cfg['write_ch']
        rch = cfg['read_ch']

        if cfg['topology'] == 'single':
            # Single channel: one SMU does both, configured as voltage source
            ch = wch  # == rch
            if ch not in configured_channels:
                configured_channels.add(ch)
                lines.append(f"-- {label} single-channel: {ch}")
                lines.append(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
                lines.append(f"{ch}.source.rangev = {lp['fixed_v_range']}")
                lines.append(f"{ch}.source.levelv = 0")
                lines.append(f"{ch}.source.limiti = {cfg['compliance_A']}")
                lines.append(f"{ch}.measure.rangei = {lp['fixed_i_range']}")
                lines.append(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")
                lines.append(f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF")
                lines.append(f"{ch}.measure.nplc = {lp['nplc']}")
                lines.extend(_emit_measurement_state(ch, lp['measure_avg']))
        else:
            # Dual channel
            if wch not in configured_channels:
                configured_channels.add(wch)
                lines.append(f"-- {label} write channel: {wch}")
                if cfg.get('stim_drive_type', 'V') == 'V':
                    lines.append(f"{wch}.source.func = {wch}.OUTPUT_DCVOLTS")
                    lines.append(f"{wch}.source.rangev = {lp['fixed_v_range']}")
                    lines.append(f"{wch}.source.levelv = 0")
                    lines.append(f"{wch}.source.limiti = {cfg['compliance_A']}")
                else:
                    lines.append(f"{wch}.source.func = {wch}.OUTPUT_DCAMPS")
                    lines.append(f"{wch}.source.rangei = {lp['fixed_i_range']}")
                    lines.append(f"{wch}.source.leveli = 0")
                    lines.append(f"{wch}.source.limitv = {lp['fixed_v_range']}")
            if rch not in configured_channels:
                configured_channels.add(rch)
                lines.append(f"-- {label} read channel: {rch}")
                lines.append(f"{rch}.source.func = {rch}.OUTPUT_DCVOLTS")
                lines.append(f"{rch}.source.rangev = {lp['fixed_read_v_range']}")
                lines.append(f"{rch}.source.levelv = {cfg['read_voltage']}")
                lines.append(f"{rch}.source.limiti = {cfg['compliance_A']}")
                lines.append(f"{rch}.measure.rangei = {lp['fixed_i_range']}")
                lines.append(f"{rch}.measure.autozero = {rch}.AUTOZERO_OFF")
                lines.append(f"{rch}.measure.autorangei = {rch}.AUTORANGE_OFF")
                lines.append(f"{rch}.measure.nplc = {lp['nplc']}")
                lines.extend(_emit_measurement_state(rch, lp['measure_avg']))

    # H11: channel configuration is emitted ONCE per channel — the
    # `configured_channels` guard above — because re-ranging mid-run would
    # perturb the timing the whole architecture is built to protect. That is
    # the right call, but it silently discarded the second train's ranges,
    # compliance and filter whenever the two trains shared a channel: Train A
    # at 1.0 V followed by Train B at 15 V on the same SMU left Train B
    # clipping at the 2 V range with no error anywhere.
    #
    # Sharing a channel is legitimate; sharing it with INCOMPATIBLE settings is
    # not, and cannot be honoured without re-ranging. So it is rejected here
    # rather than silently mis-measured.
    if train_b_config:
        shared = {train_a_config.get('write_ch'), train_a_config.get('read_ch')} & \
                 {train_b_config.get('write_ch'), train_b_config.get('read_ch')}
        shared.discard(None)
        if shared:
            conflicts = []
            if a_lp['fixed_v_range'] != b_lp['fixed_v_range']:
                conflicts.append(
                    f"source voltage range ({a_lp['fixed_v_range']} V vs "
                    f"{b_lp['fixed_v_range']} V)")
            if a_lp['fixed_i_range'] != b_lp['fixed_i_range']:
                conflicts.append(
                    f"current measure range ({a_lp['fixed_i_range']} A vs "
                    f"{b_lp['fixed_i_range']} A)")
            if train_a_config['compliance_A'] != train_b_config['compliance_A']:
                conflicts.append(
                    f"compliance ({train_a_config['compliance_A']} A vs "
                    f"{train_b_config['compliance_A']} A)")
            if a_lp['nplc'] != b_lp['nplc']:
                conflicts.append(f"NPLC ({a_lp['nplc']} vs {b_lp['nplc']})")
            if a_lp['measure_avg'] != b_lp['measure_avg']:
                conflicts.append(
                    f"averaging ({a_lp['measure_avg']} vs {b_lp['measure_avg']})")

            if conflicts:
                raise ValueError(
                    "Train A and Train B share SMU channel(s) "
                    f"{', '.join(sorted(shared))} but require different "
                    "settings:\n  - " + "\n  - ".join(conflicts) + "\n\n"
                    "A channel is configured once, before the run, because "
                    "re-ranging mid-sequence would corrupt the pulse timing. "
                    "Either give the trains compatible settings, or put them "
                    "on separate channels."
                )

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
    lines.append("local t_abs = 0")
    lines.append("")
    lines.append(f"for cycle = 1, {n_cycles} do")

    def emit_train_block(cfg, lp, label, indent="    "):
        rch = cfg['read_ch']
        wch = cfg['write_ch']
        n = cfg['n_pulses']
        period_s = lp['stim_period_s']

        lines.append(f"")
        lines.append(f"{indent}-- === {label}: {cfg['topology']} channel, {n} pulses ===")

        # Reconfigure NPLC if needed (fast register write, no relay)
        lines.append(f"{indent}{rch}.measure.nplc = {lp['nplc']}")

        if cfg['topology'] == 'single':
            # Set channel to read voltage for initial read
            lines.append(f"{indent}{wch}.source.levelv = {cfg['read_voltage']}")
            lines.append(f"{indent}delay({lp['settle_s']})")
        elif cfg['topology'] == 'dual':
            # Ensure read channel is at read voltage, write channel idle
            lines.append(f"{indent}{rch}.source.levelv = {cfg['read_voltage']}")
            if cfg.get('stim_drive_type', 'V') == 'V':
                lines.append(f"{indent}{wch}.source.levelv = 0")
            else:
                lines.append(f"{indent}{wch}.source.leveli = 0")
            lines.append(f"{indent}delay({lp['settle_s']})")

        # Initial read
        lines.append(f"{indent}{rch}.measure.i({rch}.nvbuffer1)")

        # Pulse loop
        lines.append(f"{indent}local t_train_start = t_abs")
        lines.append(f"{indent}for pulse = 1, {n} do")
        lines.append(f"{indent}    t_abs = t_train_start + pulse * {period_s}")

        if cfg['topology'] == 'single':
            # Write pulse then read pulse on same channel
            lines.append(f"{indent}    {wch}.source.levelv = {cfg['stim_level']}")
            lines.append(f"{indent}    delay({lp['stim_width_s']})")
            lines.append(f"{indent}    {wch}.source.levelv = {cfg['read_voltage']}")
            lines.append(f"{indent}    delay({lp['read_delay_s']})")
            lines.append(f"{indent}    {rch}.measure.i({rch}.nvbuffer1)")
        elif cfg['topology'] == 'dual':
            if cfg.get('stim_drive_type', 'V') == 'V':
                lines.append(f"{indent}    {wch}.source.levelv = {cfg['stim_level']}")
                lines.append(f"{indent}    delay({lp['stim_width_s']})")
                lines.append(f"{indent}    {wch}.source.levelv = 0")
            else:
                lines.append(f"{indent}    {wch}.source.leveli = {cfg['stim_level']}")
                lines.append(f"{indent}    delay({lp['stim_width_s']})")
                lines.append(f"{indent}    {wch}.source.leveli = 0")
            lines.append(f"{indent}    delay({lp['read_delay_s']})")
            lines.append(f"{indent}    {rch}.measure.i({rch}.nvbuffer1)")

        # Timer-based period enforcement
        lines.append(f"{indent}    local t_remain = t_abs - timer.measure.t()")
        lines.append(f"{indent}    if t_remain > 0 then delay(t_remain) end")
        lines.append(f"{indent}end")

        # Return channel to safe idle after train
        if cfg['topology'] == 'single':
            lines.append(f"{indent}{wch}.source.levelv = 0")
        elif cfg['topology'] == 'dual':
            if cfg.get('stim_drive_type', 'V') == 'V':
                lines.append(f"{indent}{wch}.source.levelv = 0")
            else:
                lines.append(f"{indent}{wch}.source.leveli = 0")

    # Train A
    emit_train_block(train_a_config, a_lp, "TRAIN A")

    # Inter-train delay + Train B
    if train_b_config:
        lines.append("")
        lines.append(f"    -- Inter-train delay")
        lines.append(f"    t_abs = t_abs + {inter_train_s}")
        lines.append(f"    local t_wait_train = t_abs - timer.measure.t()")
        lines.append(f"    if t_wait_train > 0 then delay(t_wait_train) end")

        emit_train_block(train_b_config, b_lp, "TRAIN B")

    # Inter-cycle delay
    lines.append("")
    lines.append(f"    -- Inter-cycle delay")
    lines.append(f"    if cycle < {n_cycles} then")
    lines.append(f"        t_abs = t_abs + {inter_cycle_s}")
    lines.append(f"        local t_wait_cycle = t_abs - timer.measure.t()")
    lines.append(f"        if t_wait_cycle > 0 then delay(t_wait_cycle) end")
    lines.append(f"    end")

    lines.append("end")

    # --- Cleanup ---
    lines.append("")
    lines.append("-- Cleanup")
    for ch in sorted(channels_used):
        lines.append(f"{ch}.source.output = {ch}.OUTPUT_OFF")
        lines.append(f"{ch}.measure.filter.enable = {ch}.FILTER_OFF")

    # Completion token (host reads this to detect script end)
    lines.append("")
    lines.append(f'print("{LUA_DONE_TOKEN}")')

    lua_script = '\n'.join(lines) + '\n'

    # --- Timing validation (warn but don't block) ---
    NPLC_PERIOD_MS = 1000.0 / line_freq_hz
    for cfg, lp, label in [(train_a_config, a_lp, "Train A")] + \
            ([(train_b_config, b_lp, "Train B")] if train_b_config else []):
        meas_time_ms = lp['nplc'] * NPLC_PERIOD_MS * (lp['measure_avg'] if lp['measure_avg'] > 1 else 1)
        min_period = cfg['stim_width_ms'] + cfg['read_delay_ms'] + meas_time_ms + cfg.get('settle_ms', 5)
        if cfg['stim_period_ms'] < min_period:
            print(f"⚠️  {label}: Requested period ({cfg['stim_period_ms']:.1f} ms) shorter than minimum "
                  f"({min_period:.1f} ms). Instrument will run as fast as possible.")

    return lua_script, buffer_map


# =============================================================================
# SAFE STATE — CALLED BEFORE EVERY SCRIPT UPLOAD
# =============================================================================

def _safe_state(instrument):
    """
    Drive both SMU channels to 0 V and clear the error queue.

    Called before every loadandrunscript to ensure the instrument is not
    stimulating the DUT during script upload.

    IMPORTANT: Does NOT toggle OUTPUT_ON/OUTPUT_OFF relays.
    - OUTPUT_OFF disconnects the output (high impedance), leaving external
      circuits floating — LEDs stay lit from stored charge, capacitors
      hold voltage.  The subsequent OUTPUT_ON in the script causes another
      relay toggle, compounding the problem.
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

def _upload_and_execute_lua(instrument, lua_script, expected_time_s=None):
    """
    Uploads and executes a LUA script via anonymous loadandrunscript.
    Waits for completion on LUA_DONE_TOKEN emitted by the script.
    Does NOT retrieve any buffer data — caller handles retrieval.
    """
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

    # Trigger execution
    instrument.write("endscript")

    # Wait for the TSP-native completion token emitted as the script's
    # final statement. See _wait_lua_complete() for rationale.
    try:
        _wait_lua_complete(instrument, expected_time_s)
    except Exception:
        # Fallback: sleep-based wait (only fires if the token never arrives,
        # e.g. VISA disruption — normal runs always receive the token).
        wait_time = expected_time_s if expected_time_s else 60
        print(f"  [LUA] completion token not received, waiting {wait_time*1.2:.1f}s...")
        time.sleep(wait_time * 1.2)


def _retrieve_buffer(instrument, channel, n_points):
    """
    Retrieves timestamps and currents from a channel's nvbuffer1.
    Uses chunked printbuffer (500 pts/read) to avoid VISA overflow.
    Queries actual buffer count first to avoid error 5038 (index exceeds maximum).

    Returns:
        tuple: (timestamps: list[float], currents: list[float])
    """
    CHUNK = 500
    buf = f"{channel}.nvbuffer1"

    # Query actual number of readings in buffer (prevents overread)
    actual_n = n_points
    try:
        actual_n = int(float(instrument.query(f"print({buf}.n)").strip()))
        if actual_n < n_points:
            print(f"WARNING: {channel} buffer has {actual_n}/{n_points} expected points")
    except Exception:
        pass

    all_currents = []
    all_timestamps = []

    for start_idx in range(1, actual_n + 1, CHUNK):
        end_idx = min(start_idx + CHUNK - 1, actual_n)
        curr_str = instrument.query(f"printbuffer({start_idx}, {end_idx}, {buf}.readings)").strip()
        time_str = instrument.query(f"printbuffer({start_idx}, {end_idx}, {buf}.timestamps)").strip()
        all_currents.extend(float(x) for x in curr_str.split(','))
        all_timestamps.extend(float(x) for x in time_str.split(','))

    return all_timestamps, all_currents


def execute_cycle_lua(instrument, lua_script, buffer_map, expected_time_s):
    """
    Executes a cycle LUA script and retrieves data from one or two buffers.

    Args:
        instrument: PyVISA resource
        lua_script (str): The cycle LUA script
        buffer_map (dict): Output from generate_cycle_lua_script
        expected_time_s (float): Expected execution time

    Returns:
        dict: {'train_a': {'timestamps': [...], 'currents': [...]},
               'train_b': {'timestamps': [...], 'currents': [...]} }
    """
    try:
        _upload_and_execute_lua(instrument, lua_script, expected_time_s)

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

        # Per-pulse timing drift analysis from buffer timestamps
        for train_key, train_label in [('train_a', 'Train A'), ('train_b', 'Train B')]:
            if train_key not in result:
                continue
            ts = result[train_key]['timestamps']
            if len(ts) < 3:
                continue
            # Determine expected period from buffer_map
            bm = buffer_map.get(train_key, buffer_map.get('train_a'))
            n_pts = bm['n_points_per_cycle']
            if n_pts < 2:
                continue
            # Expected period: total points per train per cycle = n_pulses + 1 (initial read)
            # Consecutive deltas within each cycle's train block
            try:
                drifts_ms = [
                    (ts[i+1] - ts[i]) for i in range(len(ts) - 1)
                ]
                # Median period as reference (robust against outliers at train boundaries)
                median_period = float(np.median(drifts_ms))
                deviations = [(i+1, (d - median_period) * 1000.0) for i, d in enumerate(drifts_ms)]
                drifted = [(pulse, dev) for pulse, dev in deviations if dev > 0.5]
                if drifted:
                    total_drift = sum(d for _, d in drifted)
                    worst_pulse, worst_ms = max(drifted, key=lambda x: x[1])
                    print(f"⚠️  {train_label} timing drift: +{total_drift:.1f} ms total")
                    if len(drifted) <= 5:
                        for pulse_num, d in drifted:
                            print(f"   Point {pulse_num}: +{d:.1f} ms")
                    else:
                        print(f"   {len(drifted)}/{len(drifts_ms)} intervals drifted >0.5 ms "
                              f"(worst: point {worst_pulse} at +{worst_ms:.1f} ms)")
            except Exception:
                pass

        # Clear buffers
        instrument.write(f"{a_info['channel']}.nvbuffer1.clear()")
        if 'train_b' in buffer_map and buffer_map['train_b']['channel'] != a_info['channel']:
            instrument.write(f"{buffer_map['train_b']['channel']}.nvbuffer1.clear()")

        # Check error queue
        try:
            err_cnt = int(float(instrument.query("print(errorqueue.count)").strip()))
            if err_cnt > 0:
                print(f"⚠️  {err_cnt} instrument error(s) during cycle execution:")
                for _ in range(min(5, err_cnt)):
                    print(f"   {instrument.query('print(errorqueue.next())').strip()}")
                instrument.write("errorqueue.clear()")
        except Exception:
            pass

        return result

    except Exception as e:
        try:
            instrument.write("smua.source.output = smua.OUTPUT_OFF")
            instrument.write("smub.source.output = smub.OUTPUT_OFF")
        except Exception:
            pass
        raise e


# =============================================================================
# FAST LUA EXECUTION FOR SYNAPSE CHARACTERIZATION (EXISTING)
# =============================================================================

def execute_lua_script_fast(instrument, lua_script, read_ch, n_expected_points,
                            expected_time_s=None, min_event_ms=None,
                            timing_out=None):
    """
    Uploads and executes LUA script with 'Safe Delete' to prevent Error -292.

    Uses concatenated upload (fewer VISA writes) and non-intrusive OPC waiting
    to avoid interfering with real-time script execution on the instrument.

    Args:
        instrument: PyVISA instrument resource
        lua_script (str): LUA script to upload and execute
        read_ch (str): Read channel name for buffer retrieval
        n_expected_points (int): Expected number of buffer data points
        expected_time_s (float, optional): Expected execution time in seconds.
            Used for timeout calculation and sleep-based fallback when OPC fails.
            If None, a conservative default is used.
    """
    try:
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

        # Wait for completion via LUA_DONE_TOKEN emitted as the script's
        # final statement. Deterministic and TSP-native — avoids the *OPC?
        # latching pathology that stalls repeated anonymous-script runs on
        # the 2636B (60-120 s VISA timeouts between measurements).
        # CRITICAL: Do NOT poll nvbuffer1.n during execution — that causes
        # TSP context switches which inject timing jitter into the running
        # script. The token-based wait does zero bus traffic during execution.
        lua_timing = {}
        try:
            lua_timing = _wait_lua_complete(instrument, expected_time_s) or {}
        except Exception:
            # Token never arrived (VISA disruption or LUA error). Sleep the
            # expected duration, then check buffer count ONCE as a final
            # integrity probe before giving up.
            #
            # The instrument's own timing report is lost on this path, so the
            # caller must not read its absence as "no overruns occurred".
            lua_timing = {'status': 'unavailable',
                          'reason': 'completion token not received'}
            wait_time = expected_time_s if expected_time_s else 60
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
        # from it. `n_expected_points` is an UPPER BOUND, not a count — the
        # visual continuous mode deliberately inflates it by 2*n_pulses "for
        # safety" — and requesting indices past `.n` makes the instrument raise
        # error 5038, which made continuous I(t) fail at retrieval every single
        # time. `_retrieve_buffer` already queries `.n` for exactly this reason;
        # this path did not.
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

        n_to_read = min(n_available, n_expected_points)
        if n_available < n_expected_points:
            # Not necessarily a fault: continuous mode over-allocates on
            # purpose. Reported so a genuinely short buffer is still visible.
            print(f"  [LUA] buffer holds {n_available} points, "
                  f"upper bound was {n_expected_points}")

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
        drift_report = {'status': 'not_analysed'}
        try:
            n_pulses = n_expected_points - 1
            if len(timestamps) >= 3 and expected_time_s and n_pulses > 0:
                expected_period_s = expected_time_s / n_pulses
                # Noise floor 0.1 ms matches typical 2636B TSP cycle jitter
                # (command dispatch, filter state updates). A threshold below
                # that reports instrument overhead, not real drift.
                if min_event_ms and min_event_ms > 0:
                    drift_threshold_ms = max(0.1, min(1.0, 0.25 * min_event_ms))
                else:
                    drift_threshold_ms = 0.25

                # Skip the first interval. timestamps[0] is the initial read,
                # which happens a few ms after timer.reset() and is NOT a full
                # stim_period away from timestamps[1]. Including it pulls the
                # reported mean toward a large negative value that does not
                # reflect actual per-cycle timing.
                drifts_ms = [
                    (timestamps[i+1] - timestamps[i] - expected_period_s) * 1000.0
                    for i in range(1, len(timestamps) - 1)
                ]
                if not drifts_ms:
                    raise ValueError("need at least two pulse timestamps for drift")

                total_drift_ms = sum(drifts_ms)
                max_drift_ms = max(drifts_ms)
                mean_drift_ms = total_drift_ms / len(drifts_ms)
                drifted = [(i+1, d) for i, d in enumerate(drifts_ms)
                           if d > drift_threshold_ms]

                # Delivered timing, measured from the instrument's own
                # timestamps — the authority on what the device actually saw.
                measured_periods_ms = [
                    (timestamps[i+1] - timestamps[i]) * 1000.0
                    for i in range(1, len(timestamps) - 1)
                ]
                measured_periods_ms.sort()
                _mid = len(measured_periods_ms) // 2
                median_period_ms = (
                    measured_periods_ms[_mid] if len(measured_periods_ms) % 2
                    else 0.5 * (measured_periods_ms[_mid - 1] + measured_periods_ms[_mid])
                )
                drift_report = {
                    'status': 'ok',
                    'requested_period_ms': expected_period_s * 1000.0,
                    'delivered_period_ms_median': median_period_ms,
                    'mean_drift_ms_per_cycle': mean_drift_ms,
                    'max_drift_ms': max_drift_ms,
                    'total_drift_ms': total_drift_ms,
                    'threshold_ms': drift_threshold_ms,
                    'n_pulses_exceeding_threshold': len(drifted),
                    'n_pulses': n_pulses,
                }

                # Tier 1: always-on summary if any meaningful slip occurred
                if abs(total_drift_ms) > 0.02 or max_drift_ms > 0.02:
                    print(f"ℹ️  Drift summary over {n_pulses} pulses: "
                          f"mean = {mean_drift_ms:+.3f} ms/cycle, "
                          f"max = {max_drift_ms:+.3f} ms, "
                          f"total = {total_drift_ms:+.2f} ms "
                          f"(alert threshold = {drift_threshold_ms:.3f} ms)")

                # Tier 2: per-pulse breakdown when any cycle crosses threshold
                if drifted:
                    worst_pulse, worst_ms = max(drifted, key=lambda x: x[1])
                    print(f"⚠️  {len(drifted)}/{n_pulses} pulses exceeded "
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
            drift_report = {'status': 'analysis_failed', 'error': repr(e)}
            print(f"⚠️  Drift analysis FAILED: {e!r}\n"
                  f"    Timing quality is UNKNOWN for this run — the absence of a "
                  f"drift warning below does NOT mean the timing was good.")

        # The instrument's own overrun count, if the script reported one.
        if lua_timing:
            drift_report['instrument_report'] = dict(lua_timing)
            n_over = lua_timing.get('overruns')
            if isinstance(n_over, float) and n_over > 0:
                print(f"⚠️  {int(n_over)}/{int(lua_timing.get('n_pulses', 0))} pulses could "
                      f"NOT meet the requested period (worst shortfall "
                      f"{lua_timing.get('worst_ms', float('nan')):.3f} ms).\n"
                      f"    The train ran at the hardware limit. The delivered period is "
                      f"longer than requested; use the buffer timestamps, not the nominal "
                      f"value, when reporting this measurement.")

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
        try:
            instrument.write("smua.source.output = smua.OUTPUT_OFF")
            instrument.write("smub.source.output = smub.OUTPUT_OFF")
        except: pass
        raise e

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
    """
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
    settle_ms = params.get('settle_ms', 5)
    measure_avg = params.get('measure_avg', 1)
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
        lua_data = None
        lua_timing_report = {}
        try:
            lua_data = execute_lua_script_fast(
                instrument, lua_script, read_ch, n_expected_points,
                expected_time_s=expected_time_s,
                min_event_ms=min_event_ms_local,
                timing_out=lua_timing_report,
            )
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
            return results


    # --- STANDARD EXECUTION PATH ---
    #
    # Reached when the instrument cannot run LUA, or when LUA execution failed
    # above. It is deliberately a faithful software mirror of
    # generate_pulse_read_lua_script: same baseline read, same fixed ranging,
    # same averaging derivation, same conductance convention. Only the timing
    # source differs (host clock instead of the instrument's timer), which is
    # unavoidable off-instrument and is the reason LUA is preferred.
    #
    # These two paths MUST stay equivalent. When they diverged, the same GUI
    # settings produced different physical quantities depending on which one
    # ran — pulse index 0 meant "baseline" on one and "after the first pulse"
    # on the other, so calculate_synapse_metrics reported PPF as G1/G0 in one
    # case and G2/G1 in the other, with nothing in the data recording which.

    # Stamped here rather than before the LUA attempt, so that a LUA run which
    # failed and fell through is recorded as what it actually was.
    results['execution_path'] = 'standard'

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
    nplc, measure_avg = derive_measurement_averaging(params)
    fixed_i_range = get_fixed_current_range(params['compliance_A'])
    read_voltage = params['read_voltage']

    # 1. Configure Read Channel (for low-level DC bias and measurement).
    #    Fixed ranging, matching the LUA path: autorange would make the
    #    per-pulse measurement time vary with the signal, which destroys the
    #    period reproducibility this measurement depends on.
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.rangev = {_get_fixed_voltage_range(read_voltage, params['max_voltage'])}")
    instrument.write(f"{read_ch}.source.limiti = {params['compliance_A']}")
    instrument.write(f"{read_ch}.measure.rangei = {fixed_i_range}")
    instrument.write(f"{read_ch}.measure.autozero = {read_ch}.AUTOZERO_OFF")
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF")
    instrument.write(f"{read_ch}.measure.nplc = {nplc}")
    instrument.write(f"{read_ch}.measure.count = 1")
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")

    # 2. Configure Stimulus Channel (fixed ranges, as in the LUA path)
    if params['stim_drive_type'] == 'V':
        instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS")
        instrument.write(f"{stim_ch}.source.levelv = 0")
        instrument.write(f"{stim_ch}.source.rangev = {_get_fixed_voltage_range(params['stim_level'], params['max_voltage'])}")
        instrument.write(f"{stim_ch}.source.limiti = {params['compliance_A']}")
    else:
        instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCAMPS")
        instrument.write(f"{stim_ch}.source.leveli = 0")
        instrument.write(f"{stim_ch}.source.rangei = {fixed_i_range}")
        # Voltage compliance from the instrument's actual ceiling, as the LUA
        # path does. A hardcoded 10 V here silently clamped current-driven
        # stimuli on a 200 V instrument.
        instrument.write(f"{stim_ch}.source.limitv = {params['max_voltage']}")

    instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")

    # 3. Set Read Channel to idle state (0V)
    instrument.write(f"{read_ch}.source.levelv = 0")
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")

    # === BUFFER CLEARING: Clear only what's necessary ===
    instrument.write("*CLS")  # Clear instrument error queue
    time.sleep(0.1)  # Give instrument time to process

    # --- MEASUREMENT LOOP ---

    n_samples = measure_avg

    # === CONFIGURE HARDWARE FILTER ONCE (outside loop) ===
    if n_samples > 1:
        # Ceiling is the hardware limit, shared with the LUA path. There used
        # to be a second, lower cap (50) here, so the same "Samples per Pulse"
        # value averaged differently depending on the execution path.
        n_samples_clamped = max(2, min(n_samples, FILTER_COUNT_MAX))

        instrument.write(f"{read_ch}.measure.filter.enable = {read_ch}.FILTER_ON")
        instrument.write(f"{read_ch}.measure.filter.type = {read_ch}.FILTER_REPEAT_AVG")
        instrument.write(f"{read_ch}.measure.filter.count = {n_samples_clamped}")

        if n_samples <= FILTER_COUNT_MAX:
            print(f"ℹ️  Hardware averaging: {n_samples_clamped} samples")
        else:
            print(f"ℹ️  Hardware averaging: {n_samples_clamped} samples (requested {n_samples}, clamped for efficiency)")
    
    def _read_conductance():
        """Apply the read bias, measure, return to 0 V. Returns (I_A, G_S).

        Mirrors the LUA script's read block: bias, settle, one averaged
        measurement, back to 0 V idle.
        """
        instrument.write(f"{read_ch}.source.levelv = {read_voltage}")
        time.sleep(params.get('read_settle_delay_ms', 5) / 1000.0)
        I = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        instrument.write(f"{read_ch}.source.levelv = 0")
        # Conductance is undefined at zero read bias; NaN says so rather than
        # fabricating a value that would average and plot as if measured.
        G = I / read_voltage if read_voltage != 0 else float('nan')
        return I, G

    t0 = time.time()

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

            # --- STIMULUS ON ---
            if params['stim_drive_type'] == 'V':
                instrument.write(f"{stim_ch}.source.levelv = {params['stim_level']}")
            else:
                instrument.write(f"{stim_ch}.source.leveli = {params['stim_level']}")
            
            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")
            
            # Wait for pulse duration
            time.sleep(params['stim_width_ms'] / 1000.0)
            
            # --- STIMULUS OFF ---
            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
            
            # --- WAIT READ DELAY ---
            time.sleep(params['read_delay_ms'] / 1000.0)
            
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
            elif remaining_time < -0.010:  # Warn if >10ms over budget
                print(f"⚠️  Pulse {i}: Period exceeded by {-remaining_time*1000:.1f} ms")
        
    finally:
        # Ensure all outputs are OFF
        instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        
        # === Disable hardware filter ===
        if n_samples > 1:
            instrument.write(f"{read_ch}.measure.filter.enable = {read_ch}.FILTER_OFF")
    
    # Check for instrument errors before returning
    try:
        error_check = instrument.query("print(errorqueue.count)")
        if int(error_check.strip()) > 0:
            print(f"⚠️ Warning: {error_check.strip()} errors in instrument queue")
            # Read and print errors
            for _ in range(min(5, int(error_check.strip()))):
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
        dict: Results in same format as pulse_read_sequence
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
    t0 = time.time()
    
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
        V_read = params['read_voltage']
        I_meas = G_current * V_read
        
        # Add measurement noise
        I_meas *= (1 + np.random.uniform(-0.01, 0.01))
        
        # Store results
        t_rel = time.time() - t0
        results["pulse_number"].append(i)
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
        # Calculate expected points for continuous mode
        pulse_period_s = params['pulse_period_ms'] / 1000.0
        sample_interval_s = params.get('sample_interval_ms', 1.0) / 1000.0
        total_time_s = pulse_period_s * params['n_pulses']
        n_expected_points = int(total_time_s / sample_interval_s) + params['n_pulses'] * 2
    else:
        lua_script = generate_visual_synapse_lua_script(stim_ch, read_ch, params)
        n_expected_points = params['n_pulses']

    expected_time_s = params['pulse_period_ms'] * params['n_pulses'] / 1000.0

    print(f"Using LUA execution for visual synapse measurement...")
    print(f"  Mode: {'Continuous I(t)' if continuous_mode else 'Jsc per pulse'}")
    print(f"  Pulses: {params['n_pulses']}")
    print(f"  Light voltage: {params['light_pulse_voltage']}V")

    try:
        # Execute LUA script
        timestamps, currents = execute_lua_script_fast(
            instrument, lua_script, read_ch, n_expected_points,
            expected_time_s=expected_time_s
        )

        # Process results
        t_start = timestamps[0] if timestamps else 0

        for i, (timestamp, current) in enumerate(zip(timestamps, currents)):
            t_elapsed = timestamp - t_start
            results['time_s'].append(t_elapsed)
            results['Jsc_A'].append(current)

            if not continuous_mode:
                results['pulse_number'].append(i)

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
        # Ensure outputs are off on error
        try:
            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
            instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        except:
            pass
        raise e


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
        freq_list_hz (list): List of frequencies to test in Hz
    
    Returns:
        dict: SRDP results containing:
            - frequencies_hz: List of tested frequencies
            - delta_g_S: List of conductance changes
            - delta_g_percent: List of relative conductance changes
            - g_initial_S: List of initial conductances
            - g_final_S: List of final conductances
            - base_params: Copy of base parameters
    """
    if stim_ch == read_ch:
        raise ValueError("Stim and Read channels must be different for SRDP measurements")

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
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }

    for freq_hz in freq_list_hz:
        # Calculate period from frequency
        period_ms = 1000.0 / freq_hz
        
        # Compute actual measurement time from read_width_ms and NPLC
        NPLC_PERIOD_MS = 1000.0 / _line_freq_hz(base_params)
        nplc = base_params.get('nplc', 1.0)
        read_width_ms = base_params.get('read_width_ms', None)
        if read_width_ms is not None and read_width_ms > 0:
            actual_nplc = min(nplc, read_width_ms / NPLC_PERIOD_MS)
            actual_nplc = max(0.001, actual_nplc)
            meas_time_ms = actual_nplc * NPLC_PERIOD_MS
        else:
            meas_time_ms = nplc * NPLC_PERIOD_MS

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
        
        if len(g_valid) >= 2:
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
            if timestamps.size >= 3:
                # Skip index 0: it is the pre-stimulus baseline read, not a
                # pulse, so the 0->1 interval is not a pulse period.
                intervals = np.diff(timestamps[1:])
                intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
                if intervals.size:
                    # Median rather than mean: robust to a single stalled
                    # interval from a VISA hiccup.
                    delivered_hz = 1.0 / float(np.median(intervals))

            shortfall = (freq_hz - delivered_hz) / freq_hz * 100 if freq_hz else 0.0
            if abs(shortfall) > 2.0:
                print(f"  {freq_hz:.2f} Hz requested, {delivered_hz:.2f} Hz "
                      f"delivered ({shortfall:+.1f}%). The DELIVERED rate is "
                      "what is recorded.")

            # Store results
            results["frequencies_hz"].append(delivered_hz)
            results["requested_frequencies_hz"].append(freq_hz)
            results["frequency_shortfall_percent"].append(shortfall)
            results["delta_g_S"].append(delta_g)
            results["delta_g_percent"].append(delta_g_percent)
            results["g_initial_S"].append(g_initial)
            results["g_final_S"].append(g_final)

        # Small delay between frequency measurements
        time.sleep(0.5)

    return results


def generate_stdp_lua_script(stim_ch, read_ch, params, delta_t_ms):
    """
    Generates LUA script for fast STDP spike-pair sequences.
    Executes entirely on instrument for deterministic millisecond-level timing.

    Args:
        stim_ch (str): Pre-synaptic channel ('smua' or 'smub')
        read_ch (str): Post-synaptic channel ('smua' or 'smub')
        params (dict): Measurement parameters
        delta_t_ms (float): Time difference between pre and post spikes (ms)

    Returns:
        str: LUA script for STDP measurement
    """
    # Extract parameters
    stim_level = params['stim_level']
    post_level = params.get('post_spike_level', stim_level)
    pulse_width_ms = params['stim_width_ms']
    n_pairs = params.get('n_pulses', 50)
    pair_period_ms = params.get('stim_period_ms', 200)
    read_voltage = params['read_voltage']
    compliance_A = params['compliance_A']
    settle_ms = params.get('settle_ms', 5)
    nplc = params.get('nplc', 0.1)
    line_freq_hz = _line_freq_hz(params)

    # Convert to seconds
    pulse_width_s = pulse_width_ms / 1000.0
    delta_t_s = delta_t_ms / 1000.0
    pair_period_s = pair_period_ms / 1000.0
    settle_s = settle_ms / 1000.0

    # Fixed range for speed (low-current regime)
    def get_fixed_current_range(compliance_A):
        ranges = [100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
        for r in ranges:
            if compliance_A <= r:
                return r
        return ranges[-1]

    fixed_i_range = get_fixed_current_range(compliance_A)
    max_voltage = params.get('max_voltage', 200.0)
    stim_v_range = _get_fixed_voltage_range(
        max(abs(stim_level), abs(post_level), abs(read_voltage)), max_voltage
    )
    read_v_range = _get_fixed_voltage_range(read_voltage, max_voltage)

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
    inter_pulse_gap_s = abs(delta_t_s) - pulse_width_s
    if inter_pulse_gap_s < 0:
        raise ValueError(
            f"STDP Delta_t = {delta_t_ms} ms is smaller than the pulse width "
            f"({pulse_width_ms} ms). Delta_t is defined onset-to-onset, so the "
            "two spikes would have to OVERLAP in time, which this sequential "
            "pulse generator cannot produce.\n"
            f"Either increase |Delta_t| above {pulse_width_ms} ms, or reduce "
            "the pulse width."
        )

    # Generate LUA script
    lua_script = f"""
-- Fast STDP Spike-Pair Sequence
-- Deterministic millisecond-level timing

-- Synchronise ADC to mains for normal-mode rejection of line-frequency pickup
localnode.linefreq = {int(line_freq_hz)}

-- Configuration
local stim_ch = {stim_ch}
local read_ch = {read_ch}
local n_pairs = {n_pairs}
local stim_level = {stim_level}
local post_level = {post_level}
local pulse_width = {pulse_width_s}
-- Gap between the two pulses. delta_t is ONSET-TO-ONSET per the spec, so the
-- silent interval between them is delta_t minus one pulse width.
local inter_pulse_gap = {inter_pulse_gap_s}
local pair_period = {pair_period_s}
local read_voltage = {read_voltage}
local settle_time = {settle_s}

-- Configure channels with fixed ranges for speed
{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS
{stim_ch}.source.rangev = {stim_v_range}
{stim_ch}.source.levelv = 0
{stim_ch}.source.limiti = {compliance_A}
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF

{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS
{read_ch}.source.rangev = {read_v_range}
{read_ch}.source.levelv = 0
{read_ch}.source.limiti = {compliance_A}
{read_ch}.measure.rangei = {fixed_i_range}
{read_ch}.measure.autozero = {read_ch}.AUTOZERO_OFF
{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_OFF
{read_ch}.measure.nplc = {nplc}
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Turn on outputs at 0V (avoid relay toggling during spike pairs)
{stim_ch}.source.output = {stim_ch}.OUTPUT_ON
{read_ch}.source.output = {read_ch}.OUTPUT_ON
delay(0.01)

-- Measure initial conductance
{read_ch}.source.levelv = read_voltage
delay(settle_time)
local I_initial = {read_ch}.measure.i()
{read_ch}.source.levelv = 0

-- STDP spike-pair loop (timer-based to prevent cumulative drift)
timer.reset()
local t_next = 0
for pair = 1, n_pairs do
    t_next = t_next + pair_period
"""

    if delta_t_ms >= 0:
        # Pre before Post (LTP)
        lua_script += f"""    -- Pre-spike (level change, no relay toggle)
    {stim_ch}.source.levelv = stim_level
    delay(pulse_width)
    {stim_ch}.source.levelv = 0

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
    {stim_ch}.source.levelv = stim_level
    delay(pulse_width)
    {stim_ch}.source.levelv = 0
"""

    lua_script += f"""
    -- Wait for next pair using absolute timer (prevents cumulative drift)
    local t_remain = t_next - timer.measure.t()
    if t_remain > 0 then delay(t_remain) end
end

-- Measure final conductance
{read_ch}.source.levelv = read_voltage
delay(settle_time)
local I_final = {read_ch}.measure.i()

-- Cleanup
{stim_ch}.source.levelv = 0
{read_ch}.source.levelv = 0
{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF
{read_ch}.source.output = {read_ch}.OUTPUT_OFF

-- Return results (stored in globals for retrieval)
_stdp_i_initial = I_initial
_stdp_i_final = I_final

-- Completion token (host reads this to detect script end)
print("{LUA_DONE_TOKEN}")
"""

    return lua_script


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
            - base_params: Copy of base parameters
    """
    if stim_ch == read_ch:
        raise ValueError("Stim and Read channels must be different for STDP measurements")

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
    # taken from it, so a failure at one Δt cannot affect the next.
    lua_supported = supports_lua_execution(instrument)
    results["execution_path"] = []

    for delta_t_ms in delta_t_list_ms:
        use_lua = lua_supported
        if use_lua:
            # === FAST LUA PATH: Deterministic millisecond-level timing ===
            print(f"STDP Δt={delta_t_ms}ms: Using LUA for deterministic timing...")

            try:
                # Generate and upload LUA script
                lua_script = generate_stdp_lua_script(stim_ch, read_ch, base_params, delta_t_ms)

                # Synchronize TSP, then load anonymous script
                _safe_state(instrument)
                instrument.write("loadandrunscript")
                time.sleep(0.02)

                # Upload script — concatenated for fewer VISA writes
                script_lines = []
                for line in lua_script.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('--'):
                        script_lines.append(line)

                chunk = []
                chunk_size = 0
                for line in script_lines:
                    line_size = len(line) + 1
                    if chunk_size + line_size > 512 and chunk:
                        instrument.write('\n'.join(chunk))
                        chunk = []
                        chunk_size = 0
                    chunk.append(line)
                    chunk_size += line_size
                if chunk:
                    instrument.write('\n'.join(chunk))

                # endscript triggers immediate execution for anonymous scripts
                print("Executing STDP LUA script on instrument...")
                instrument.write("endscript")

                # Wait for completion via LUA_DONE_TOKEN (TSP-native,
                # avoids *OPC? latching stalls on repeated runs).
                n_pairs = base_params.get('n_pulses', 50)
                pair_period_ms = base_params.get('stim_period_ms', 200)
                estimated_time_s = (n_pairs * pair_period_ms) / 1000.0 + 2.0

                try:
                    _wait_lua_complete(instrument, estimated_time_s)
                except Exception:
                    print(f"  [LUA] completion token not received, waiting {estimated_time_s:.1f}s...")
                    time.sleep(estimated_time_s)

                # Retrieve results from global variables
                I_initial = float(instrument.query("print(_stdp_i_initial)").strip())
                I_final = float(instrument.query("print(_stdp_i_final)").strip())

                # No cleanup needed — anonymous script (loadandrunscript) leaves no named artifact

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

            except Exception as e:
                # Scoped to THIS delta_t only — see the note on lua_supported.
                print(f"  WARNING: LUA failed at delta_t={delta_t_ms} ms ({e}).")
                print("  Falling back to the PC-timed path for this point only. "
                      "Its spike timing is not instrument-controlled and Delta_t "
                      "is therefore approximate; the point is stamped "
                      "'standard' in results['execution_path'].")
                use_lua = False  # Fall through to standard execution for this point

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

            # NPLC from the shared derivation, not a hardcoded 0.1 that
            # discarded whatever the GUI asked for.
            nplc_std, _ = derive_measurement_averaging(base_params)

            # Configure channels
            instrument.write(f"localnode.linefreq = {int(_line_freq_hz(base_params))}")
            instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS")
            instrument.write(f"{stim_ch}.source.limiti = {base_params['compliance_A']}")
            instrument.write(f"{stim_ch}.source.levelv = 0")
            instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
            instrument.write(f"{read_ch}.source.limiti = {base_params['compliance_A']}")
            instrument.write(f"{read_ch}.source.levelv = 0")
            instrument.write(f"{read_ch}.measure.nplc = {nplc_std}")

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
                    instrument.write(f"{stim_ch}.source.levelv = {base_params['stim_level']}")
                    time.sleep(pulse_width_s)
                    instrument.write(f"{stim_ch}.source.levelv = 0")

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

                    instrument.write(f"{stim_ch}.source.levelv = {base_params['stim_level']}")
                    time.sleep(pulse_width_s)
                    instrument.write(f"{stim_ch}.source.levelv = 0")

                time.sleep(pair_period_ms / 1000.0)

            # Measure final conductance
            g_final = _read_g()

            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
            instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")

        # Provenance for this point: LUA timing and host timing are not the
        # same measurement, and an array that mixes them must say so.
        results["execution_path"].append('lua' if use_lua else 'standard')

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
    
    # Configure sense mode if specified
    _apply_wire_modes(instrument, params, [read_ch])

    # Synchronise ADC to mains for normal-mode rejection of line-frequency
    # pickup, adopting the instrument's auto-detected frequency unless the
    # caller specified one.
    instrument.write(f"localnode.linefreq = {int(resolve_line_freq(instrument, params))}")

    # Configure read channel for retention measurement
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.limiti = {compliance_A}")
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_ON")
    instrument.write(f"{read_ch}.measure.nplc = {nplc}")
    
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
        
        # Measure conductance
        instrument.write(f"{read_ch}.source.levelv = {read_voltage}")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
        time.sleep(settle_ms / 1000.0)
        
        # Measure current
        I_measured = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        
        # Calculate conductance
        G_measured = I_measured / read_voltage if read_voltage != 0 else 0
        
        # Turn off output
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        
        # Calculate elapsed time from start
        t_elapsed = time.time() - t_start
        
        # Store results
        results['time_s'].append(t_elapsed)
        results['conductance_S'].append(G_measured)
        results['I_A'].append(I_measured)
        
        # Display progress
        print(f"G = {G_measured*1e6:.2f} µS (t = {t_elapsed:.1f}s)")
    
    print(f"\n✓ Retention measurement complete!")
    print(f"  Initial G: {results['conductance_S'][0]*1e6:.2f} µS")
    print(f"  Final G: {results['conductance_S'][-1]*1e6:.2f} µS")
    print(f"  Decay: {(1 - results['conductance_S'][-1]/results['conductance_S'][0])*100:.1f}%")
    
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