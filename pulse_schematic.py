"""
Pulse schematic renderer for SYNAPSYS synapse mode.

Draws live, dimensioned waveform diagrams of the stimulus and read channels
for the selected synapse submode. Used by the Keithley controller to give
the user an immediate visual readback of the pulse train they are designing
— period, widths, delays, voltages, relay state — without having to run a
measurement first.

Rendering rules (must stay faithful to the physics in synapse_engine.py):
- Source level is drawn as a piecewise-constant step.
- When the source output is OFF (relay open), the level is drawn as a thin
  dashed gray line at 0 V to indicate physical disconnect, not a driven 0 V.
- Dimension arrows annotate period, widths, delays and measurement windows.
- Voltages and timing values are shown numerically next to each element.
- Time axis always covers ~1.5 periods so periodicity is visible.
"""

import numpy as np

# Mains line frequency (Hz) — mirrors synapse_engine.DEFAULT_LINE_FREQ_HZ.
# Drives the ADC integration aperture (NPLC) when computing the read-pulse width
# drawn here, exactly as generate_pulse_read_lua_script does on the instrument.
# Per-call override via params['line_freq_hz']. Kept local so this renderer
# stays dependency-light and does not import the hardware module.
DEFAULT_LINE_FREQ_HZ = 50.0

# --- Visual style constants ---
COLOR_STIM_ON  = '#1976D2'   # blue — stim output ON
COLOR_READ_ON  = '#D32F2F'   # red — read output ON
COLOR_RELAY    = '#9E9E9E'   # gray — source driven but relay OPEN
COLOR_MEAS     = '#4CAF50'   # green — active measurement window
COLOR_ANNOT    = '#222222'
COLOR_TITLE    = '#111111'

TITLE_FS = 11
LABEL_FS = 10
ANNOT_FS = 9
TICK_FS  = 9

GRID_ALPHA = 0.22
RELAY_LW   = 1.1
SIGNAL_LW  = 2.1
ARROW_LW   = 0.9

# Annotation rows in axes fractions (0..1). Labels sit centered on each
# row (white bbox occludes the arrow line under the text), so rows just
# need ~one text-height of separation. With axes ~0.5 in tall, that maps
# to ~0.30 axes-frac per row.
ROW_PERIOD = 0.85   # top — outermost dimension bracket (pulse period)
ROW_WIDTH  = 0.50   # middle of the pulse area
ROW_INNER  = 0.32   # measurement-band label (green)
ROW_DELAY  = 0.15   # bottom — delays and inter-train gaps
ROW_LEVEL  = 0.55   # inline level label (passed to _level_label, data y)


# ============================================================================
# TOP-LEVEL DISPATCH
# ============================================================================

def render(ax_stim, ax_read, submode, params):
    """
    Clear both axes, then draw the schematic for `submode` with values from
    `params`. All parameters that cannot be parsed (blank entries, text) are
    silently skipped and a "waiting for parameters" placeholder is shown.

    The caller is responsible for calling canvas.draw_idle() afterwards.
    """
    for ax in (ax_stim, ax_read):
        ax.clear()
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, alpha=GRID_ALPHA, linestyle=':')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    renderers = {
        'Basic':                _render_basic,
        'SRDP':                 _render_srdp,
        'STDP':                 _render_stdp,
        'Visual (Self-Powered)': _render_visual,
        'Cycle':                _render_cycle,
        'Multi-Device':         _render_cycle,  # same schematic as Cycle
    }
    renderer = renderers.get(submode)
    if renderer is None:
        _render_placeholder(ax_stim, ax_read,
                            f"No schematic for mode: {submode}")
        return

    try:
        renderer(ax_stim, ax_read, params)
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        # The specific diagnostic is SHOWN, not discarded.
        #
        # This used to map every failure — including _parse_cycle_train's
        # deliberately loud ValueError, which names the offending field — to a
        # generic "Enter valid pulse parameters". The message that would have
        # told the user which parameter was wrong was constructed and thrown
        # away, leaving them to guess across a dozen entry boxes.
        detail = str(e).strip()
        message = "Cannot preview the pulse train"
        if detail:
            # Keep the placeholder readable if the exception text is long.
            if len(detail) > 160:
                detail = detail[:157] + "..."
            message += f":\n{detail}"
        else:
            message += f" ({type(e).__name__})"
        _render_placeholder(ax_stim, ax_read, message)


def _render_placeholder(ax_stim, ax_read, msg):
    for ax in (ax_stim, ax_read):
        ax.text(0.5, 0.5, msg, ha='center', va='center',
                fontsize=8, color='#888', style='italic',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


# ============================================================================
# LOW-LEVEL DRAWING HELPERS
# ============================================================================

def _float(v, default=None):
    """Best-effort float parse; returns default on failure."""
    try:
        f = float(str(v).strip())
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _int(v, default=None):
    f = _float(v, None)
    return int(f) if f is not None else default


def _configure_axis(ax, header, subheader, ylabel, ymin, ymax, t_end,
                    show_xlabel=True):
    """
    Configure an axis for a schematic panel.

    The header (channel + drive type) goes top-left as the axis title; the
    subheader (numeric parameter summary) goes just below it as a smaller
    sub-title using suptitle-style placement. Putting all numerics in text
    sidesteps in-plot collisions entirely — the waveform itself shows the
    relative timing, and the user reads exact values from the headers.
    """
    # Single-line bold title only — kept compact so multiple short axes
    # still leave room above for the title without clipping the figure.
    # All numeric values live in the in-plot dimension arrows.
    ax.set_title(header, fontsize=TITLE_FS, loc='left',
                 fontweight='bold', color=COLOR_TITLE, pad=4)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    dy = ymax - ymin
    if dy < 1e-9:
        dy = 1.0
    # Headroom above for period bracket; footroom below for relay caption.
    ax.set_ylim(ymin - dy * 0.30, ymax + dy * 0.55)
    ax.set_xlim(-0.02 * t_end, 1.02 * t_end)
    if show_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=LABEL_FS)
    else:
        ax.tick_params(labelbottom=False)


def _y_at(ax, frac):
    """Convert an axes-fraction y-value (0..1) to data coordinates."""
    ymin, ymax = ax.get_ylim()
    return ymin + frac * (ymax - ymin)


def _draw_segments(ax, segments, color_on, y_off=0.0):
    """
    Draws a list of waveform segments as a single solid step function.

    Each segment is a dict:
        {'t0': float, 't1': float, 'level': float, 'on': bool}

    The 'on' flag is no longer rendered differently — the source level
    itself encodes what the SMU is driving (0 V between pulses), and the
    user reads the train as a clean step function.
    """
    prev_y = None
    for seg in segments:
        t0, t1, level = seg['t0'], seg['t1'], seg['level']
        y = level
        if prev_y is not None and prev_y != y:
            ax.plot([t0, t0], [prev_y, y], color=color_on, lw=SIGNAL_LW)
        ax.plot([t0, t1], [y, y], color=color_on, lw=SIGNAL_LW)
        prev_y = y


def _alignment_lines(axes, xs, color='#888', alpha=0.55, lw=0.8):
    """
    Vertical dashed alignment lines drawn on every axis at each x-position
    so the user can visually align stim events with read events.
    """
    for ax in axes:
        for x in xs:
            ax.axvline(x, color=color, lw=lw, linestyle='--',
                       alpha=alpha, zorder=1)


def _dim_arrow(ax, x0, x1, row_frac, label, color=COLOR_ANNOT,
               label_above=True):  # label_above kept for API compat
    """
    Horizontal double-arrow at a fixed axes-fraction row.

    Label placement is adaptive: if the arrow spans at least 20 % of the
    visible x-range, the label sits CENTERED on the arrow (white bbox
    occludes the shaft, engineering-drawing style). For narrower arrows,
    the label is anchored to the right of the rightmost arrowhead so it
    can't spill over the y-axis tick labels at the left edge.
    """
    if x1 <= x0:
        return
    y = _y_at(ax, row_frac)
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=ARROW_LW, shrinkA=0, shrinkB=0),
                zorder=9)

    xlim_lo, xlim_hi = ax.get_xlim()
    axis_w = xlim_hi - xlim_lo
    arrow_w = x1 - x0
    if axis_w > 0 and arrow_w / axis_w >= 0.20:
        text_x, ha = 0.5 * (x0 + x1), 'center'
    else:
        # Narrow arrow: anchor label just past the right arrowhead.
        text_x, ha = x1 + 0.01 * axis_w, 'left'
    ax.text(text_x, y, label, ha=ha, va='center',
            fontsize=ANNOT_FS, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=2.5, alpha=0.96),
            zorder=10, clip_on=True)


def _level_label(ax, x, level, text, color):
    """Small text tag next to a horizontal voltage level."""
    ax.text(x, level, text, ha='center', va='center',
            fontsize=ANNOT_FS, color=color, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor=color, pad=2.0,
                      alpha=0.95, lw=0.6),
            zorder=10)


def _period_divider(ax, x, label=None):
    ax.axvline(x, color='#BBBBBB', lw=0.7, linestyle=':', zorder=1)
    if label:
        ax.text(x, _y_at(ax, 1.02), label, ha='center', va='bottom',
                fontsize=ANNOT_FS - 1, color='#777', style='italic')


def _meas_window(ax, x0, x1, label=None):
    """
    Shaded band marking the active ADC integration window. Spans the full
    axes height so it visually anchors the window without arguing with the
    waveform y-axis.
    """
    if x1 <= x0:
        return
    ax.axvspan(x0, x1, ymin=0, ymax=1, color=COLOR_MEAS, alpha=0.14, zorder=0)
    y_top = _y_at(ax, 1.0)
    y_bot = _y_at(ax, 0.0)
    ax.plot([x0, x0], [y_bot, y_top], color=COLOR_MEAS, lw=0.9,
            linestyle=':', zorder=1)
    ax.plot([x1, x1], [y_bot, y_top], color=COLOR_MEAS, lw=0.9,
            linestyle=':', zorder=1)
    if label:
        # Center vertically inside the band so it can't overlap with
        # the settle / read-delay arrows that live above and below.
        ax.text(0.5 * (x0 + x1), _y_at(ax, ROW_INNER), label,
                ha='center', va='center', fontsize=ANNOT_FS,
                color='#1B5E20', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor=COLOR_MEAS,
                          pad=2.0, alpha=0.95, lw=0.6),
                zorder=10)


def _unit_for_drive(drive):
    return 'V' if drive == 'V' else 'A'


# ============================================================================
# BASIC MODE
# ============================================================================

def _render_basic(ax_stim, ax_read, p):
    """
    Electrical pulse-read train.

    Per-period waveform mirrors generate_pulse_read_lua_script:
      - Stim relay OFF at 0 V, then turns ON, level rises to stim_level for
        stim_width, then output OFF again (relay open until next pulse).
      - Read stays at 0 V during stim; after stim_width + read_delay, goes
        up to read_voltage for read_settle + measurement_time; back to 0 V.
    """
    stim_ch   = p.get('stim_ch', 'smua')
    read_ch   = p.get('read_ch', 'smub')
    drive     = p.get('stim_drive_type', 'V')
    stim_lvl  = _float(p.get('stim_level'))
    sw_ms     = _float(p.get('stim_width_ms'))
    sp_ms     = _float(p.get('stim_period_ms'))
    rv        = _float(p.get('read_voltage'))
    rd_ms     = _float(p.get('read_delay_ms'))
    rs_ms     = _float(p.get('read_settle_delay_ms'))
    rw_ms     = _float(p.get('read_width_ms'), 50.0)
    nplc      = _float(p.get('nplc'), 1.0)

    # Name the offending fields. `render()` was deliberately built to SHOW the
    # diagnostic in the schematic placeholder, and then the diagnostic was
    # "missing basic params" — leaving a user mid-typing across eleven entry
    # boxes with no idea which one. The condition already knows exactly.
    _required = [
        ("Stim Level", stim_lvl), ("Stim Width", sw_ms), ("Stim Period", sp_ms),
        ("Read Voltage", rv), ("Read Delay", rd_ms), ("Read Settle Delay", rs_ms),
    ]
    _missing = [name for name, val in _required if val is None]
    if _missing:
        raise ValueError("cannot read: " + ", ".join(_missing))
    if sp_ms <= 0:
        raise ValueError(f"Stim Period = {sp_ms:g} ms must be greater than 0")
    if sw_ms <= 0:
        raise ValueError(f"Stim Width = {sw_ms:g} ms must be greater than 0")
    if sw_ms > sp_ms:
        raise ValueError(
            f"Stim Width = {sw_ms:g} ms exceeds Stim Period = {sp_ms:g} ms; "
            "the pulse cannot be longer than the period that contains it"
        )

    # Clamp non-negative durations: the hardware can't have negative
    # delays/settle, so the schematic should reflect the same reality.
    rd_ms = max(0.0, rd_ms)
    rs_ms = max(0.0, rs_ms)
    rw_ms = max(0.0, rw_ms)

    # Mirror generate_pulse_read_lua_script EXACTLY so the schematic
    # always reflects what the instrument will actually do:
    #   1. If the user's NPLC implies one integration longer than
    #      read_width_ms, the LUA reduces NPLC down to fit (clamped to
    #      MIN_NPLC = 0.001 — the Keithley 2600 hardware floor).
    #   2. Then measure_avg = floor(read_width_ms / one_integration),
    #      with FP-tolerant floor and a hard minimum of 1 integration.
    #   3. Actual measurement duration = measure_avg × one_integration.
    line_freq_hz = _float(p.get('line_freq_hz'), DEFAULT_LINE_FREQ_HZ)
    NPLC_PERIOD_MS = 1000.0 / line_freq_hz
    MIN_NPLC = 0.001
    nplc_actual = nplc if nplc and nplc > 0 else 1.0
    if rw_ms > 0:
        max_nplc_for_width = rw_ms / NPLC_PERIOD_MS
        if nplc_actual > max_nplc_for_width:
            nplc_actual = max(MIN_NPLC, max_nplc_for_width)
    single_integration_ms = max(nplc_actual * NPLC_PERIOD_MS, 1e-6)
    if rw_ms > 0:
        measure_avg = max(1, int(rw_ms / single_integration_ms + 1e-6))
    else:
        measure_avg = 1
    meas_ms = measure_avg * single_integration_ms

    # Total physical width of the read pulse on the wire = settle + ADC.
    read_pulse_ms = max(0.0, rs_ms) + meas_ms

    # t_end must accommodate the full read pulse, even if it extends
    # beyond the nominal stimulation period (configuration error: the
    # period_divider then visibly cuts through the pulse).
    period_one_end = max(sp_ms, sw_ms + max(0.0, rd_ms) + read_pulse_ms)
    t_end = period_one_end * 1.5

    # --- Stim axis: two pulses, relay OFF between and at edges ---
    stim_segs = []
    for k in range(2):
        tk = k * sp_ms
        # Relay OFF before pulse (level commanded to stim_level by LUA, but
        # that only matters to the DUT when OUTPUT_ON — we draw it gray
        # dashed at 0 since the external circuit is disconnected).
        if k == 0:
            stim_segs.append({'t0': 0, 't1': tk, 'level': 0, 'on': False})
        # Pulse ON
        stim_segs.append({'t0': tk, 't1': tk + sw_ms,
                          'level': stim_lvl, 'on': True})
        # Relay OFF after pulse until end of period
        stim_segs.append({'t0': tk + sw_ms, 't1': (k + 1) * sp_ms,
                          'level': 0, 'on': False})
    # Trim last segment to t_end
    while stim_segs and stim_segs[-1]['t0'] >= t_end:
        stim_segs.pop()
    if stim_segs and stim_segs[-1]['t1'] > t_end:
        stim_segs[-1]['t1'] = t_end

    _draw_segments(ax_stim, stim_segs, COLOR_STIM_ON)

    # Configure stim axis with all numerics in the header so they cannot
    # collide with in-plot text on a short-axis canvas.
    lvl_min = min(0, stim_lvl)
    lvl_max = max(0, stim_lvl)
    unit = _unit_for_drive(drive)
    _configure_axis(ax_stim,
                    f"STIM  ·  {stim_ch}  ({drive}-drive)",
                    (f"period = {sp_ms:g} ms     "
                     f"width = {sw_ms:g} ms     "
                     f"level = {stim_lvl:g} {unit}"),
                    f"Stim ({unit})",
                    lvl_min, lvl_max, t_end, show_xlabel=False)

    # In-plot dimensions on dedicated axes-fraction rows.
    _dim_arrow(ax_stim, 0, sp_ms, ROW_PERIOD, f"period = {sp_ms:g} ms")
    _dim_arrow(ax_stim, 0, sw_ms, ROW_WIDTH,  f"width = {sw_ms:g} ms")
    _period_divider(ax_stim, sp_ms)

    # Vertical dashed alignment lines spanning both axes so the user can
    # see exactly which read events line up with which stim events.
    align_xs = [
        0,                          # stim rising edge
        sw_ms,                      # stim falling edge
        sw_ms + rd_ms,              # read voltage rises (after read_delay)
        sw_ms + rd_ms + rs_ms,      # settle ends, ADC begins
        sw_ms + rd_ms + rs_ms + meas_ms,   # ADC done, read voltage falls
    ]
    _alignment_lines([ax_stim, ax_read], align_xs)

    # --- Read axis: initial baseline at 0, then read window each period ---
    # Read window anchor inside each period:
    #   start = stim_width + read_delay
    #   end   = start + read_settle + meas
    # Segments with t1 <= t0 are dropped so a read pulse longer than the
    # period (configuration error: rw_end > sp_ms) doesn't produce
    # negative-width segments — the user instead sees the pulse extend
    # past the period_divider, which is exactly what would happen on
    # hardware.
    read_segs = []
    for k in range(2):
        tk = k * sp_ms
        rw_start = tk + sw_ms + rd_ms
        rw_end_settle = rw_start + rs_ms
        rw_end = rw_end_settle + meas_ms
        period_end = (k + 1) * sp_ms
        for s in (
            {'t0': tk,       't1': rw_start,    'level': 0,  'on': True},
            {'t0': rw_start, 't1': rw_end,      'level': rv, 'on': True},
            {'t0': rw_end,   't1': period_end,  'level': 0,  'on': True},
        ):
            if s['t1'] > s['t0']:
                read_segs.append(s)
    while read_segs and read_segs[-1]['t0'] >= t_end:
        read_segs.pop()
    if read_segs and read_segs[-1]['t1'] > t_end:
        read_segs[-1]['t1'] = t_end

    _draw_segments(ax_read, read_segs, COLOR_READ_ON)

    rw_start = sw_ms + rd_ms
    rw_end_settle = rw_start + rs_ms
    rw_end = rw_end_settle + meas_ms

    lvl_min_r = min(0, rv)
    lvl_max_r = max(0, rv)
    nplc_label = (f"NPLC={nplc_actual:g}"
                  if abs(nplc_actual - (nplc or 0)) < 1e-6
                  else f"NPLC={nplc_actual:g} (auto-reduced from {nplc:g})")
    # ×N avg shows the number of consecutive ADC integrations the 2600
    # filter averages into each buffer entry. ×1 is omitted to avoid
    # clutter when no averaging is happening.
    avg_label = f" · ×{measure_avg} avg" if measure_avg > 1 else ""
    _configure_axis(ax_read,
                    f"READ  ·  {read_ch}  (V bias, measures I)",
                    (f"V_read = {rv:g} V     "
                     f"read delay = {rd_ms:g} ms     "
                     f"settle = {rs_ms:g} ms     "
                     f"measure = {meas_ms:.3g} ms · {nplc_label}{avg_label}"),
                    "Read (V)",
                    lvl_min_r, lvl_max_r, t_end, show_xlabel=True)

    # In-plot dimensions on dedicated axes-fraction rows. Voltage value
    # is in the subheader so the inline level chip is omitted.
    _dim_arrow(ax_read, sw_ms, rw_start, ROW_DELAY,
               f"read delay = {rd_ms:g} ms")
    _dim_arrow(ax_read, rw_start, rw_end_settle, ROW_WIDTH,
               f"settle = {rs_ms:g} ms")
    _meas_window(ax_read, rw_end_settle, rw_end,
                 label=f"meas {meas_ms:.3g} ms · {nplc_label}{avg_label}")

    # Zero read bias yields NO conductance. The engine returns NaN there
    # (conductance is undefined at V = 0; fabricating 0 S would average and
    # plot as though measured), so the whole run produces an all-NaN column.
    # The pre-run preview is exactly where that must be visible — it used to
    # render identically to a normal read.
    if rv == 0:
        ax_read.text(
            0.5, 0.5,
            "V_read = 0 V → conductance UNDEFINED (NaN)\n"
            "every conductance value in this run will be NaN",
            transform=ax_read.transAxes, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#b00020",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffe8ec",
                      edgecolor="#b00020", linewidth=1.2),
            zorder=20,
        )

    _period_divider(ax_read, sp_ms)


# ============================================================================
# SRDP MODE
# ============================================================================

def _render_srdp(ax_stim, ax_read, p):
    """
    Spike-Rate Dependent Plasticity schematic.

    SRDP sweeps frequency — the schematic shows the representative period
    at the midpoint of the start/end frequency range (geometric mean when
    log scale is used). Waveform shape is identical to Basic with period
    derived from 1/frequency.
    """
    f_start = _float(p.get('freq_start_hz'))
    f_end   = _float(p.get('freq_end_hz'))
    log_scale = bool(p.get('log_scale', True))

    if f_start is None or f_end is None or f_start <= 0 or f_end <= 0:
        raise ValueError("missing SRDP frequency range")

    if log_scale:
        f_mid = np.sqrt(f_start * f_end)
    else:
        f_mid = 0.5 * (f_start + f_end)

    period_ms = 1000.0 / f_mid

    # Delegate to Basic renderer with the derived period
    basic_p = dict(p)
    basic_p['stim_period_ms'] = period_ms
    _render_basic(ax_stim, ax_read, basic_p)

    # Override the stim title with SRDP-specific framing (frequency sweep
    # info), keeping single-line bold style.
    drive = p.get('stim_drive_type', 'V')
    stim_ch = p.get('stim_ch', 'smua')
    sweep_kind = "log" if log_scale else "linear"
    ax_stim.set_title(
        (f"STIM  ·  SRDP {f_start:g}–{f_end:g} Hz ({sweep_kind})  ·  "
         f"midpoint @ {f_mid:.2f} Hz  ·  {stim_ch}  ({drive}-drive)"),
        fontsize=TITLE_FS, loc='left', fontweight='bold',
        color=COLOR_TITLE, pad=4)


# ============================================================================
# STDP MODE
# ============================================================================

def _render_stdp(ax_stim, ax_read, p):
    """
    Spike-Timing Dependent Plasticity schematic.

    Shows one pre/post spike pair with the selected Δt, plus the next pair
    to illustrate pair_period. When Δt > 0: pre (stim_ch) fires first, then
    post (read_ch) after Δt — LTP regime. When Δt < 0: post fires first,
    then pre after |Δt| — LTD regime.
    """
    stim_ch  = p.get('stim_ch', 'smua')
    read_ch  = p.get('read_ch', 'smub')
    pre_lvl  = _float(p.get('pre_level'))
    post_lvl = _float(p.get('post_level'))
    pw_ms    = _float(p.get('pulse_width_ms'))
    pp_ms    = _float(p.get('pair_period_ms'))
    dt_ms    = _float(p.get('delta_t_ms'))

    if None in (pre_lvl, post_lvl, pw_ms, pp_ms, dt_ms):
        raise ValueError("missing STDP params")
    if pp_ms <= 0 or pw_ms <= 0:
        raise ValueError("non-positive timing")

    t_end = pp_ms * 1.5

    # Build pre and post waveforms for two pairs
    pre_segs = []
    post_segs = []
    for k in range(2):
        tk = k * pp_ms
        if dt_ms >= 0:
            # Pre first, then post at tk + pw + dt
            pre_t0 = tk
            pre_t1 = tk + pw_ms
            post_t0 = pre_t1 + dt_ms
            post_t1 = post_t0 + pw_ms
        else:
            # Post first, then pre after |dt|
            post_t0 = tk
            post_t1 = tk + pw_ms
            pre_t0 = post_t1 + abs(dt_ms)
            pre_t1 = pre_t0 + pw_ms

        # Pre-spike segments (between start of period and next period start)
        pre_segs += [
            {'t0': tk,      't1': pre_t0,  'level': 0,      'on': False} if tk < pre_t0 else None,
            {'t0': pre_t0,  't1': pre_t1,  'level': pre_lvl, 'on': True},
            {'t0': pre_t1,  't1': (k + 1) * pp_ms, 'level': 0, 'on': False},
        ]
        post_segs += [
            {'t0': tk,      't1': post_t0, 'level': 0,       'on': False} if tk < post_t0 else None,
            {'t0': post_t0, 't1': post_t1, 'level': post_lvl, 'on': True},
            {'t0': post_t1, 't1': (k + 1) * pp_ms, 'level': 0, 'on': False},
        ]
    pre_segs = [s for s in pre_segs if s is not None and s['t0'] < s['t1']]
    post_segs = [s for s in post_segs if s is not None and s['t0'] < s['t1']]

    def _trim(segs):
        out = []
        for s in segs:
            if s['t0'] >= t_end:
                break
            if s['t1'] > t_end:
                s = dict(s); s['t1'] = t_end
            out.append(s)
        return out

    pre_segs  = _trim(pre_segs)
    post_segs = _trim(post_segs)

    _draw_segments(ax_stim, pre_segs, COLOR_STIM_ON)
    _draw_segments(ax_read, post_segs, COLOR_READ_ON)

    # Annotations for the first pair (computed for both layout + level chips)
    if dt_ms >= 0:
        pre_t0_a = 0
        pre_t1_a = pw_ms
        post_t0_a = pre_t1_a + dt_ms
        post_t1_a = post_t0_a + pw_ms
        regime = "LTP (pre → post)"
    else:
        post_t0_a = 0
        post_t1_a = pw_ms
        pre_t0_a = post_t1_a + abs(dt_ms)
        pre_t1_a = pre_t0_a + pw_ms
        regime = "LTD (post → pre)"

    _configure_axis(ax_stim,
                    f"PRE  ·  {stim_ch}",
                    (f"pair period = {pp_ms:g} ms     "
                     f"width = {pw_ms:g} ms     "
                     f"level = {pre_lvl:g} V     "
                     f"Δt = {dt_ms:+g} ms · {regime}"),
                    "Pre (V)",
                    min(0, pre_lvl), max(0, pre_lvl),
                    t_end, show_xlabel=False)
    _configure_axis(ax_read,
                    f"POST  ·  {read_ch}",
                    (f"pair period = {pp_ms:g} ms     "
                     f"width = {pw_ms:g} ms     "
                     f"level = {post_lvl:g} V"),
                    "Post (V)",
                    min(0, post_lvl), max(0, post_lvl),
                    t_end, show_xlabel=True)

    # In-plot dimensions on dedicated axes-fraction rows.
    _dim_arrow(ax_stim, 0, pp_ms, ROW_PERIOD, f"pair period = {pp_ms:g} ms")
    if dt_ms >= 0:
        dt_x0, dt_x1 = pre_t1_a, post_t0_a
    else:
        dt_x0, dt_x1 = post_t1_a, pre_t0_a
    _dim_arrow(ax_stim, dt_x0, dt_x1, ROW_DELAY,
               f"Δt = {dt_ms:+g} ms")

    # Alignment lines on every spike edge so the user sees pre/post timing.
    _alignment_lines([ax_stim, ax_read],
                     [pre_t0_a, pre_t1_a, post_t0_a, post_t1_a])

    _period_divider(ax_stim, pp_ms)
    _period_divider(ax_read, pp_ms)


# ============================================================================
# VISUAL (SELF-POWERED) MODE
# ============================================================================

def _render_visual(ax_stim, ax_read, p):
    """
    Visual synapse: LED voltage pulses on stim channel, photocurrent measured
    at 0 V on read channel during each pulse.

    Read channel stays at 0 V throughout (self-powered), with the measurement
    window inside each light pulse: [measure_start_delay, pulse_width - measure_end_margin].
    """
    stim_ch     = p.get('stim_ch', 'smua')
    read_ch     = p.get('read_ch', 'smub')
    led_v       = _float(p.get('light_pulse_voltage'))
    pw_ms       = _float(p.get('pulse_width_ms'))
    pp_ms       = _float(p.get('pulse_period_ms'))
    ms_start_ms = _float(p.get('measure_start_delay_ms'))
    ms_end_ms   = _float(p.get('measure_end_margin_ms'))
    continuous  = bool(p.get('continuous_mode', False))
    nplc        = _float(p.get('nplc'), 1.0)

    if None in (led_v, pw_ms, pp_ms, ms_start_ms, ms_end_ms):
        raise ValueError("missing visual params")
    if pp_ms <= 0 or pw_ms <= 0:
        raise ValueError("non-positive timing")

    t_end = pp_ms * 1.5

    # Stim waveform: LED pulses, relay toggled between pulses
    stim_segs = []
    for k in range(2):
        tk = k * pp_ms
        if k == 0:
            stim_segs.append({'t0': 0, 't1': tk, 'level': 0, 'on': False})
        stim_segs.append({'t0': tk, 't1': tk + pw_ms, 'level': led_v, 'on': True})
        stim_segs.append({'t0': tk + pw_ms, 't1': (k + 1) * pp_ms,
                          'level': 0, 'on': False})
    while stim_segs and stim_segs[-1]['t0'] >= t_end:
        stim_segs.pop()
    if stim_segs and stim_segs[-1]['t1'] > t_end:
        stim_segs[-1]['t1'] = t_end

    _draw_segments(ax_stim, stim_segs, COLOR_STIM_ON)

    _configure_axis(ax_stim,
                    f"LIGHT  ·  {stim_ch}  (LED drive)",
                    (f"period = {pp_ms:g} ms     "
                     f"pulse width = {pw_ms:g} ms     "
                     f"V_LED = {led_v:g} V"),
                    "LED (V)",
                    min(0, led_v), max(0, led_v),
                    t_end, show_xlabel=False)

    _dim_arrow(ax_stim, 0, pp_ms, ROW_PERIOD, f"period = {pp_ms:g} ms")
    _dim_arrow(ax_stim, 0, pw_ms, ROW_WIDTH,  f"pulse = {pw_ms:g} ms")
    _period_divider(ax_stim, pp_ms)

    # Read waveform: flat 0 V throughout (self-powered measurement)
    read_segs = [{'t0': 0, 't1': t_end, 'level': 0, 'on': True}]
    _draw_segments(ax_read, read_segs, COLOR_READ_ON)

    if continuous:
        meas_dur = pw_ms
        meas_subhdr = "continuous I(t) sampling during entire pulse"
    else:
        meas_dur = max(0.0, pw_ms - ms_start_ms - ms_end_ms)
        meas_subhdr = (f"start delay = {ms_start_ms:g} ms     "
                       f"end margin = {ms_end_ms:g} ms     "
                       f"window = {meas_dur:.1f} ms · NPLC={nplc:g}")

    _configure_axis(ax_read,
                    f"READ  ·  {read_ch}  (0 V bias → measures Jsc)",
                    meas_subhdr,
                    "Read (V)", -0.5, 0.5, t_end, show_xlabel=True)

    # In-plot: green bands per pulse, plus dimension arrows for the start
    # delay (below baseline) and end margin (above baseline) on the first
    # pulse only — repeating them on the second pulse just creates clutter.
    for k in range(2):
        tk = k * pp_ms
        if continuous:
            mw0, mw1 = tk, tk + pw_ms
            label = "continuous I(t)" if k == 0 else None
        else:
            mw0 = tk + ms_start_ms
            mw1 = tk + pw_ms - ms_end_ms
            label = (f"meas {max(0, mw1 - mw0):.1f} ms · NPLC={nplc:g}"
                     if k == 0 else None)
        if mw1 > mw0 and mw0 < t_end:
            _meas_window(ax_read, max(0, mw0), min(t_end, mw1), label=label)

    if not continuous:
        _dim_arrow(ax_read, 0, ms_start_ms, ROW_DELAY,
                   f"start = {ms_start_ms:g} ms")
        _dim_arrow(ax_read, pw_ms - ms_end_ms, pw_ms, ROW_WIDTH,
                   f"end = {ms_end_ms:g} ms")

    _period_divider(ax_read, pp_ms)

    # Alignment lines for the LED-on / measurement-window edges.
    if continuous:
        align_xs = [0, pw_ms]
    else:
        align_xs = [0, ms_start_ms, pw_ms - ms_end_ms, pw_ms]
    _alignment_lines([ax_stim, ax_read], align_xs)


# ============================================================================
# CYCLE MODE
# ============================================================================

def _render_cycle(ax_stim, ax_read, p):
    """
    Cycling: train A followed by train B, repeated N times. The schematic
    shows one full cycle (A + inter-train delay + B) on each axis.

    Because a cycle can use either single-channel (write == read) or dual
    channel topology, and the two trains may target different channels, we
    display the stim axis as "all write activity" and the read axis as
    "all read activity" over one cycle. Channel labels on each train make
    topology explicit.
    """
    a = p.get('train_a') or {}
    b = p.get('train_b') or None
    inter_train_ms = _float(p.get('inter_train_delay_ms'), 500.0)
    n_cycles = _int(p.get('n_cycles'), 1)

    a_params = _parse_cycle_train(a, "Train A")
    b_params = _parse_cycle_train(b, "Train B") if b else None

    # Layout: lay out A first, then inter-train gap, then B
    a_duration = a_params['n_pulses'] * a_params['period_ms']
    b_duration = b_params['n_pulses'] * b_params['period_ms'] if b_params else 0
    cycle_duration = a_duration + (inter_train_ms + b_duration if b_params else 0)
    t_end = cycle_duration * 1.05 + 1.0

    # --- Draw STIM activity ---
    stim_segs = []
    stim_levels_seen = [0]

    def _add_train_stim(segs, start_t, tp):
        n = tp['n_pulses']
        for k in range(n):
            tk = start_t + k * tp['period_ms']
            if k == 0 and tk > (segs[-1]['t1'] if segs else 0):
                segs.append({'t0': segs[-1]['t1'] if segs else 0,
                             't1': tk, 'level': 0, 'on': False})
            segs.append({'t0': tk, 't1': tk + tp['stim_width_ms'],
                         'level': tp['stim_level'], 'on': True})
            segs.append({'t0': tk + tp['stim_width_ms'],
                         't1': tk + tp['period_ms'],
                         'level': 0, 'on': False})
            stim_levels_seen.append(tp['stim_level'])
        return segs

    # Train A stim
    stim_segs = _add_train_stim(stim_segs, 0, a_params)
    # Inter-train gap (relay OFF)
    if b_params:
        gap_start = a_duration
        gap_end = a_duration + inter_train_ms
        stim_segs.append({'t0': gap_start, 't1': gap_end, 'level': 0, 'on': False})
        stim_segs = _add_train_stim(stim_segs, gap_end, b_params)

    _draw_segments(ax_stim, stim_segs, COLOR_STIM_ON)

    lvl_min = min(stim_levels_seen)
    lvl_max = max(stim_levels_seen)
    a_ch = a_params['write_ch']
    b_ch = b_params['write_ch'] if b_params else None
    stim_title = f"STIM  ·  Train A: {a_ch}"
    if b_params:
        stim_title += f"     Train B: {b_ch}"
    stim_title += f"     ×{n_cycles} cycle(s)"
    sub_a = (f"A: {a_params['stim_level']:g} V × {a_params['n_pulses']}"
             f"  T={a_params['period_ms']:g}ms  w={a_params['stim_width_ms']:g}ms")
    sub_b = ""
    if b_params:
        sub_b = (f"   |   B: {b_params['stim_level']:g} V × {b_params['n_pulses']}"
                 f"  T={b_params['period_ms']:g}ms  w={b_params['stim_width_ms']:g}ms"
                 f"   |   Δ_train={inter_train_ms:g}ms")
    _configure_axis(ax_stim, stim_title, sub_a + sub_b, "Stim (V)",
                    lvl_min, lvl_max, t_end, show_xlabel=False)

    # Train boundaries marked with brackets at the top, plus inter-train
    # delay annotated with an arrow below baseline.
    _train_bracket(ax_stim, 0, a_duration, "A", _y_at(ax_stim, ROW_PERIOD))
    if b_params:
        _train_bracket(ax_stim, a_duration + inter_train_ms,
                       a_duration + inter_train_ms + b_duration, "B",
                       _y_at(ax_stim, ROW_PERIOD))
        _dim_arrow(ax_stim, a_duration, a_duration + inter_train_ms, ROW_DELAY,
                   f"inter-train = {inter_train_ms:g} ms")

    # --- Draw READ activity ---
    read_segs = []
    read_levels_seen = [0]

    def _add_train_read(segs, start_t, tp):
        """Draw the read channel as the hardware actually drives it.

        M25: this used to show the read channel sitting at 0 V for most of
        every period, spiking to read_voltage only for a brief window. The
        cycle LUA does the opposite — it holds the read channel AT read bias
        continuously and drops to 0 V only while the stimulus pulse is applied
        (`synapse_engine.py` cycle generator). At a 1000 ms period with a 1 ms
        stimulus the device is biased for 999 ms of every period, while the
        diagram showed roughly 50 ms: a ~20x understatement of read-disturb
        exposure, which is exactly the quantity this diagram exists to convey.

        The measurement APERTURE is drawn as a distinct emphasis within the
        held bias, since that is a different thing from the bias itself.
        """
        n = tp['n_pulses']
        v_read = tp['read_voltage']
        single = tp.get('topology') == 'single'

        for k in range(n):
            tk = start_t + k * tp['period_ms']
            if k == 0 and tk > (segs[-1]['t1'] if segs else 0):
                segs.append({'t0': segs[-1]['t1'] if segs else 0,
                             't1': tk, 'level': v_read, 'on': True})

            if single:
                # One SMU does both: it must leave read bias to apply the
                # stimulus, then return to it.
                segs.append({'t0': tk, 't1': tk + tp['stim_width_ms'],
                             'level': 0, 'on': True})
                segs.append({'t0': tk + tp['stim_width_ms'],
                             't1': tk + tp['period_ms'],
                             'level': v_read, 'on': True})
            else:
                # Dual channel: the read SMU is configured at read bias before
                # the run and never leaves it.
                segs.append({'t0': tk, 't1': tk + tp['period_ms'],
                             'level': v_read, 'on': True})

            read_levels_seen.append(v_read)
        return segs

    read_segs = _add_train_read(read_segs, 0, a_params)
    if b_params:
        gap_start = a_duration
        gap_end = a_duration + inter_train_ms
        read_segs.append({'t0': gap_start, 't1': gap_end, 'level': 0, 'on': True})
        read_segs = _add_train_read(read_segs, gap_end, b_params)

    _draw_segments(ax_read, read_segs, COLOR_READ_ON)

    lvl_min_r = min(read_levels_seen)
    lvl_max_r = max(read_levels_seen)
    a_rch = a_params['read_ch']
    b_rch = b_params['read_ch'] if b_params else None
    read_title = f"READ  ·  Train A: {a_rch}"
    if b_params:
        read_title += f"     Train B: {b_rch}"
    sub_ra = (f"A: V_read={a_params['read_voltage']:g}V"
              f"  read delay={a_params['read_delay_ms']:g}ms")
    sub_rb = ""
    if b_params:
        sub_rb = (f"   |   B: V_read={b_params['read_voltage']:g}V"
                  f"  read delay={b_params['read_delay_ms']:g}ms")
    _configure_axis(ax_read, read_title, sub_ra + sub_rb, "Read (V)",
                    lvl_min_r, lvl_max_r, t_end, show_xlabel=True)

    # Read-delay arrow on the first pulse of train A so the offset is visible.
    #
    # M27: the label now says what the interval IS. This arrow spans the gap
    # between the end of the stimulus and the measurement, and the device
    # spends that gap AT READ BIAS (see _add_train_read) — not at 0 V, which is
    # what the old "read delay" label implied when drawn over a trace that
    # showed 0 V there.
    _dim_arrow(ax_read, a_params['stim_width_ms'],
               a_params['stim_width_ms'] + a_params['read_delay_ms'],
               ROW_DELAY,
               f"settling at V_read = {a_params['read_voltage']:g} V "
               f"({a_params['read_delay_ms']:g} ms)")

    # The ADC aperture, drawn where the measurement actually happens.
    _ap_start = a_params['stim_width_ms'] + a_params['read_delay_ms']
    _ap_end = _ap_start + a_params['aperture_ms']
    _dim_arrow(ax_read, _ap_start, _ap_end, ROW_DELAY - 1,
               f"ADC aperture = {a_params['aperture_ms']:.3g} ms "
               f"(NPLC {a_params['nplc']:g}"
               + (f" x {a_params['measure_avg']}" if a_params['measure_avg'] > 1 else "")
               + ")")

    # Alignment lines for train transitions and the first read event.
    align_xs = [0, a_duration]
    if b_params:
        align_xs.append(a_duration + inter_train_ms)
    align_xs.append(a_params['stim_width_ms'])
    align_xs.append(a_params['stim_width_ms'] + a_params['read_delay_ms'])
    _alignment_lines([ax_stim, ax_read], align_xs)

    # Period markers on both axes (no labels — info is in the subheaders)
    _period_divider(ax_stim, a_params['period_ms'])
    _period_divider(ax_read, a_params['period_ms'])
    if b_params:
        _period_divider(ax_stim,
                        a_duration + inter_train_ms + b_params['period_ms'])
        _period_divider(ax_read,
                        a_duration + inter_train_ms + b_params['period_ms'])


def _parse_cycle_train(train, label):
    """Normalize a cycle train config dict (from keithley_analyser) into the
    shape expected by _render_cycle. Fails loudly if required fields missing."""
    topo = str(train.get('topology', 'Single SMU'))
    if topo.startswith('Single'):
        ch = train.get('channel', 'smua')
        write_ch = read_ch = ch
    else:
        write_ch = train.get('write_ch', 'smua')
        read_ch  = train.get('read_ch', 'smub')

    sl = _float(train.get('stim_level'))
    sw = _float(train.get('stim_width_ms'))
    rv = _float(train.get('read_voltage'))
    rd = _float(train.get('read_delay_ms'))
    pp = _float(train.get('period_ms'))
    np_ = _int(train.get('n_pulses'))

    if None in (sl, sw, rv, rd, pp) or np_ is None:
        raise ValueError(f"{label}: missing params")
    if pp <= 0 or sw <= 0 or np_ <= 0:
        raise ValueError(f"{label}: non-positive timing")

    # M26: the ADC aperture is DERIVED, exactly as the engine derives it, not
    # hardcoded. The renderer used `max(2.0, period * 0.05)` with a comment
    # claiming "the LUA uses a short hardcoded read" — it does not:
    # synapse_engine derives NPLC and averaging per train from the same
    # controls Basic mode uses. At NPLC = 0.01 on 50 Hz mains the true aperture
    # is 0.2 ms while the drawn window was 50 ms: a 250x overstatement.
    nplc = _float(train.get('nplc'), 1.0)
    line_freq_hz = _float(train.get('line_freq_hz'), 50.0)
    settle_ms = _float(train.get('settle_ms'), 0.0) or 0.0
    measure_avg = _int(train.get('measure_avg')) or 1

    NPLC_PERIOD_MS = 1000.0 / line_freq_hz
    MIN_NPLC = 0.001
    aperture_ms = max(nplc, MIN_NPLC) * NPLC_PERIOD_MS * max(1, measure_avg)

    return {
        'write_ch': write_ch,
        'read_ch':  read_ch,
        'stim_level': sl,
        'stim_width_ms': sw,
        'read_voltage': rv,
        'read_delay_ms': rd,
        'period_ms': pp,
        'n_pulses': np_,
        'nplc': nplc,
        'settle_ms': settle_ms,
        'measure_avg': measure_avg,
        'aperture_ms': aperture_ms,
        'topology': 'single' if topo.startswith('Single') else 'dual',
    }


def _train_bracket(ax, x0, x1, label, y):
    """Draws a horizontal bracket above a train labeling its extent."""
    if x1 <= x0:
        return
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='|-|', color=COLOR_ANNOT, lw=0.8))
    ax.text(0.5 * (x0 + x1), y, f" {label} ",
            ha='center', va='center', fontsize=ANNOT_FS,
            fontweight='bold', color=COLOR_ANNOT,
            bbox=dict(facecolor='white', edgecolor=COLOR_ANNOT,
                      pad=1.2, lw=0.5))
