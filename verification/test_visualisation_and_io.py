"""C9, M19-M21, M25-M27 and the units contract — the last batch.

Most of these are "silent" defects: they produce a plot or a pattern that looks
plausible and is wrong, which is precisely the class the project ground rules
exist to prevent.
"""

import inspect
import json
import os

import numpy as np
import pytest

import excel_patterns
import fitting
import pulse_schematic
import snn_visualization as snnv


# --- M11: the units contract ----------------------------------------------


def test_a_millisecond_time_axis_is_converted_to_seconds():
    """M11 — decay_tau is reported in seconds and compared against a 100 s
    default, so a retention curve recorded in ms produced a tau 1000x wrong
    with an excellent R², because the fit is perfectly good in the wrong unit."""
    times_ms = np.array([1000.0, 10000.0, 30000.0, 60000.0, 100000.0])
    values, note = fitting.convert_axis_to_si(
        times_ms, {'x_units': 'ms'}, axis='x', expected_quantity='time')

    assert np.allclose(values, times_ms / 1000.0)
    assert note and 'ms' in note


def test_an_si_axis_is_left_alone():
    times_s = np.array([1.0, 10.0, 30.0])
    values, note = fitting.convert_axis_to_si(
        times_s, {'x_units': 's'}, axis='x', expected_quantity='time')
    assert np.allclose(values, times_s)
    assert note is None


def test_an_unrecognised_unit_raises_rather_than_assuming_si():
    with pytest.raises(ValueError, match="Unrecognised"):
        fitting.convert_axis_to_si(
            [1, 2, 3], {'x_units': 'furlongs'}, axis='x',
            expected_quantity='time')


def test_a_dimensionless_axis_cannot_be_a_time_axis():
    """A pulse index is not a decay time."""
    with pytest.raises(ValueError, match="dimensionless"):
        fitting.convert_axis_to_si(
            [0, 1, 2], {'x_units': '#'}, axis='x', expected_quantity='time')


def test_ms_disambiguates_between_time_and_conductance():
    """'ms' is milliseconds on a time axis and millisiemens on a Y axis.

    Both tables contain the key, which is why expected_quantity is required
    rather than inferred.
    """
    t, _ = fitting.convert_axis_to_si([1000.0], {'x_units': 'ms'},
                                      axis='x', expected_quantity='time')
    g, _ = fitting.convert_axis_to_si([1.0], {'y_units': 'mS'},
                                      axis='y', expected_quantity='conductance')
    assert t[0] == pytest.approx(1.0)      # 1000 ms -> 1 s
    assert g[0] == pytest.approx(1e-3)     # 1 mS -> 1e-3 S


# --- C9: fitted-model save/load -------------------------------------------


def test_a_fitted_model_round_trips_through_json(tmp_path):
    """C9 — json.dump raised on the closures, so no fitted model could ever be
    saved, and the load branch could only be reached with a file that
    necessarily lacked the spectral curve."""
    model = {
        'G_min': 1e-6, 'G_max': 1e-4, 'alpha': 0.75, 'beta': 0.75,
        'decay_tau': 100.0,
        'lambda_peak': 365.0, 'lambda_width': 30.0,
        'wavelength_peaks': [
            {'amplitude': 1.0, 'wavelength_nm': 365.0, 'width_nm': 30.0,
             'type': 'potentiation'},
            {'amplitude': -1.0, 'wavelength_nm': 550.0, 'width_nm': 30.0,
             'type': 'depression'},
        ],
        'wavelength_curve': lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        'retention_curve': lambda t: np.zeros_like(np.asarray(t, dtype=float)),
        'extraction_report': {'extracted': ['dynamic_range'], 'failed': [],
                              'defaults_used': [], 'warnings': []},
    }

    path = str(tmp_path / "model.json")
    fitting.save_fitted_model(model, path)
    assert os.path.exists(path)

    # It is genuinely valid JSON, not a truncated partial write.
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    assert raw['fitted_model_format']

    restored = fitting.load_fitted_model(path)
    assert restored['G_min'] == pytest.approx(1e-6)
    assert callable(restored['wavelength_curve'])


def test_the_restored_spectral_curve_is_signed_and_multi_peak(tmp_path):
    """M2 — reconstruction rebuilt a positive SINGLE Gaussian at lambda_peak.

    Combined with M1 (lambda_peak being the depression band) that made a
    loaded model potentiate under green and do nothing under UV — the exact
    inverse of the device.
    """
    model = {
        'G_min': 1e-6, 'G_max': 1e-4, 'decay_tau': 100.0,
        'lambda_peak': 365.0, 'lambda_width': 30.0,
        'wavelength_peaks': [
            {'amplitude': 1.0, 'wavelength_nm': 365.0, 'width_nm': 30.0,
             'type': 'potentiation'},
            {'amplitude': -1.0, 'wavelength_nm': 550.0, 'width_nm': 30.0,
             'type': 'depression'},
        ],
    }
    path = str(tmp_path / "m.json")
    fitting.save_fitted_model(model, path)
    curve = fitting.load_fitted_model(path)['wavelength_curve']

    assert curve(np.array([365.0]))[0] > 0.9, "UV must potentiate"
    assert curve(np.array([550.0]))[0] < -0.9, "green must depress"


def test_loading_a_non_model_json_is_refused(tmp_path):
    path = str(tmp_path / "suite.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'datasets': [], 'n_datasets': 0}, f)

    with pytest.raises(ValueError, match="not a saved fitted model"):
        fitting.load_fitted_model(path)


def test_reconstruct_uses_the_stored_peaks_not_a_single_gaussian():
    model = {
        'lambda_peak': 550.0, 'lambda_width': 49.0,
        'wavelength_peaks': [
            {'amplitude': 0.6, 'wavelength_nm': 365.0, 'width_nm': 30.0,
             'type': 'potentiation'},
            {'amplitude': -1.0, 'wavelength_nm': 550.0, 'width_nm': 30.0,
             'type': 'depression'},
        ],
        'wavelength_curve': None,
    }
    rebuilt = fitting.reconstruct_fitted_curves(model)
    curve = rebuilt['wavelength_curve']
    assert curve(np.array([365.0]))[0] > 0, "the potentiation lobe was discarded"
    assert curve(np.array([550.0]))[0] < 0, "the sign was discarded"


# --- M19: firing-rate binning ---------------------------------------------


def test_every_spike_is_counted_exactly_once():
    """M19 — linspace edges are spaced duration/(n-1), wider than the counting
    window, so consecutive bins left a gap and ~11% of spikes fell into none."""
    duration, bin_ms = 1000.0, 100.0
    spikes = list(np.arange(5.0, duration, 7.0))   # 143 spikes, no clustering
    _, rates = snnv.compute_firing_rates([spikes], duration, bin_ms)

    counted = rates[0] * bin_ms / 1000.0
    assert counted.sum() == pytest.approx(len(spikes), abs=0.5), (
        f"counted {counted.sum():.1f} of {len(spikes)} spikes"
    )


def test_no_spurious_zero_bin_at_the_end():
    """The final linspace bin started at exactly duration_ms, so it covered no
    data and always read 0 Hz — a drop to zero at the right edge of the plot."""
    duration, bin_ms = 500.0, 100.0
    spikes = list(np.arange(1.0, duration, 5.0))
    bins, rates = snnv.compute_firing_rates([spikes], duration, bin_ms)

    assert bins[-1] < duration
    assert rates[0][-1] > 0, "the last bin reads zero despite containing spikes"


def test_bin_edges_are_exactly_one_bin_apart():
    bins, _ = snnv.compute_firing_rates([[]], 1000.0, 100.0)
    assert np.allclose(np.diff(bins), 100.0)


# --- M20: the double microsiemens conversion ------------------------------


def test_weight_evolution_does_not_convert_to_microsiemens_twice():
    """get_weight_matrix() already returns µS; multiplying again plotted
    values 1e6 too large."""
    source = inspect.getsource(snnv.plot_weight_evolution)
    assert "* 1e6" not in source, "the second µS conversion is back"


# --- SRDP overlay must not fabricate a curve ------------------------------


def test_the_srdp_overlay_refuses_to_invent_parameters():
    """`plot_srdp_curve` read 'amplitude'/'transition_freq_hz'/'slope' with
    defaults while fitting.py produces srdp_-prefixed names, so every lookup
    missed and it drew a fabricated f0=10/k=10 sigmoid labelled "Fitted SRDP"
    over the user's real data. Its sibling plot_stdp_curve raises instead."""
    import matplotlib
    matplotlib.use("Agg")

    srdp_data = {'frequencies_hz': [1.0, 10.0, 100.0],
                 'delta_g_percent': [1.0, 10.0, 19.0]}

    with pytest.raises(KeyError, match="SRDP parameters missing"):
        snnv.plot_srdp_curve(srdp_data, fitted_params={'unrelated': 1})


def test_the_srdp_overlay_accepts_the_prefixed_names_the_fitter_produces():
    import matplotlib
    matplotlib.use("Agg")

    srdp_data = {'frequencies_hz': [1.0, 10.0, 100.0],
                 'delta_g_percent': [1.0, 10.0, 19.0]}
    fitted = {'srdp_max_change': 20.0, 'srdp_transition_freq_hz': 10.0,
              'srdp_slope': 0.3}
    ax = snnv.plot_srdp_curve(srdp_data, fitted_params=fitted)
    assert ax is not None


# --- the spike-train RNG ---------------------------------------------------


def test_spike_generation_does_not_disturb_the_global_rng():
    """np.random.seed() reseeds the PROCESS-WIDE generator, so generating
    spike trains silently changed every other stochastic result."""
    np.random.seed(12345)
    expected = np.random.rand()

    np.random.seed(12345)
    snnv.generate_poisson_spike_train(50.0, 1000.0, seed=999)
    after = np.random.rand()

    assert after == pytest.approx(expected), (
        "generating a spike train perturbed the global RNG"
    )


def test_image_spike_trains_are_not_phase_locked():
    """M-series: every train started at exactly t = 0, so the whole image fired
    one synchronous volley and equal-intensity pixels stayed locked forever."""
    image = np.full((4, 4), 0.5)
    trains = snnv.image_to_spike_trains(image, duration_ms=1000.0, seed=7)

    first_spikes = [t[0] for t in trains if t]
    assert len(set(first_spikes)) > 1, (
        "all pixels of equal intensity still fire simultaneously"
    )


# --- M21: the Excel loader -------------------------------------------------


def _write_sheet(path, fills=None, values=None, size=(4, 4)):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    wb = Workbook()
    ws = wb.active
    for r in range(1, size[0] + 1):
        for c in range(1, size[1] + 1):
            cell = ws.cell(row=r, column=c)
            if values and (r, c) in values:
                cell.value = values[(r, c)]
            if fills and (r, c) in fills:
                cell.fill = PatternFill(start_color=fills[(r, c)],
                                        end_color=fills[(r, c)],
                                        fill_type='solid')
    wb.save(path)
    wb.close()


def test_an_all_zero_pattern_raises_instead_of_loading_silently(tmp_path):
    """M21 — a blank or unreadable sheet returned all zeros with no warning,
    indistinguishable from a sheet the user had actually filled in."""
    path = str(tmp_path / "blank.xlsx")
    _write_sheet(path)

    with pytest.raises(ValueError, match="ALL-ZERO"):
        excel_patterns.ExcelPatternLoader.load_pattern_from_excel(path)


def test_numeric_cell_values_are_read(tmp_path):
    """Cell VALUES were never read — only fills — so a sheet typed as numbers
    loaded as all zeros."""
    path = str(tmp_path / "numbers.xlsx")
    _write_sheet(path, values={(1, 1): 1.0, (2, 2): 0.5})

    pattern = excel_patterns.ExcelPatternLoader.load_pattern_from_excel(path)
    assert pattern.max() == pytest.approx(1.0)
    assert np.any(np.isclose(pattern, 0.5))


def test_solid_fills_are_still_read(tmp_path):
    path = str(tmp_path / "fills.xlsx")
    _write_sheet(path, fills={(1, 1): "FF000000", (2, 2): "FF000000"})
    pattern = excel_patterns.ExcelPatternLoader.load_pattern_from_excel(path)
    assert pattern.any()


def test_downsampling_uses_nearest_neighbour():
    """order=1 bilinear SAMPLES rather than area-averages, so thin features
    vanish when downsampling a binary pattern."""
    source = inspect.getsource(excel_patterns.ExcelPatternLoader.resize_pattern)
    assert "AREA AVERAGING" in source, "downsampling still samples rather than averages"

    # A single-cell line must survive a 2x downsample.
    pattern = np.zeros((8, 8))
    pattern[3, :] = 1.0
    resized = excel_patterns.ExcelPatternLoader.resize_pattern(pattern, (4, 4))
    assert resized.max() > 0.2, (
        f"the line vanished when downsampling (max {resized.max():.3f}); "
        "a 1-of-2 row line should survive as ~0.5 intensity"
    )


# --- M25-M27: the cycle schematic -----------------------------------------


def _cycle_train(**overrides):
    train = {
        'topology': 'Single SMU', 'channel': 'smua',
        'stim_level': 1.0, 'stim_width_ms': 1.0,
        'read_voltage': 0.1, 'read_delay_ms': 2.0,
        'period_ms': 1000.0, 'n_pulses': 3,
        'nplc': 0.01, 'measure_avg': 1, 'line_freq_hz': 50.0,
    }
    train.update(overrides)
    return train


def test_the_adc_aperture_is_derived_not_hardcoded():
    """M26 — the read window was max(2.0, period*0.05) with a comment claiming
    the LUA uses a short hardcoded read. It does not. At NPLC 0.01 the true
    aperture is 0.2 ms while the drawn window was 50 ms: 250x."""
    tp = pulse_schematic._parse_cycle_train(_cycle_train(), "Train A")

    # NPLC 0.01 on 50 Hz mains = 0.01 * 20 ms = 0.2 ms.
    assert tp['aperture_ms'] == pytest.approx(0.2, rel=1e-9)
    assert tp['aperture_ms'] != pytest.approx(max(2.0, 1000.0 * 0.05))


def test_the_aperture_scales_with_averaging_and_mains_frequency():
    tp = pulse_schematic._parse_cycle_train(
        _cycle_train(nplc=1.0, measure_avg=4, line_freq_hz=60.0), "A")
    assert tp['aperture_ms'] == pytest.approx(4 * 1000.0 / 60.0, rel=1e-9)


def test_the_read_channel_is_drawn_held_at_read_bias():
    """M25 — the diagram showed the read channel at 0 V for most of every
    period; the hardware holds it AT read bias. At a 1000 ms period with a 1 ms
    stimulus the device is biased for 999 ms and the diagram showed ~50 ms."""
    source = inspect.getsource(pulse_schematic._render_cycle)
    assert "_add_train_read" in source

    add_read = source.split("def _add_train_read")[1].split("return segs")[0]
    # The held-bias segment must span the period, not a short window.
    assert "'level': v_read" in add_read
    assert "period_ms'], " in add_read or "tp['period_ms']" in add_read
    assert "max(2.0, tp['period_ms'] * 0.05)" not in add_read, (
        "the hardcoded read window is back"
    )


def test_the_schematic_reports_the_specific_error():
    """A broad except mapped every failure — including _parse_cycle_train's
    deliberately loud ValueError — to a generic placeholder, discarding the
    diagnostic that names the offending field."""
    source = inspect.getsource(pulse_schematic.render)
    # Strip comments: the fix's own explanation quotes the old message.
    code = "\n".join(ln.split("#")[0] for ln in source.split("\n"))

    assert "Enter valid pulse parameters" not in code, (
        "the generic placeholder that discards the diagnostic is back"
    )
    assert "str(e)" in code, "the exception text is not surfaced"
    assert "_render_placeholder(ax_stim, ax_read, message)" in code


def test_a_bad_cycle_train_names_the_problem():
    with pytest.raises(ValueError, match="Train A"):
        pulse_schematic._parse_cycle_train(
            _cycle_train(period_ms=-5.0), "Train A")
