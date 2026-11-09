"""Missing readings (NaN/null) and spectral relaxation in the fitter.

The hardware path writes NaN conductance at zero read bias by design. Before
DataParser._sanitise_datasets, 5% missing readings turned G_max = 99 µS into
1.66 µS and reported it as successfully extracted. Separately, the spectral
initial rates carried the dark relaxation of each trace, which read as a
-0.1..-0.16 "depression response" at 650-750 nm and widened the depression
band by ~20%.
"""

import copy

import numpy as np
import pytest

import baseline as bl
import fitting
import pipeline


@pytest.fixture(scope="module")
def suite_and_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_suite_nan")
    return pipeline.run(str(out), seed=bl.PIPELINE_SEED, verbose=False)


def _with_gaps(suite, fraction=0.05, seed=7):
    s = copy.deepcopy(suite)
    rng = np.random.default_rng(seed)
    for d in s["datasets"]:
        y = np.array(d["y_data"], dtype=float)
        y[rng.random(y.size) < fraction] = np.nan
        d["y_data"] = y.tolist()
    return s


KEYS = ("G_min", "G_max", "alpha", "beta", "decay_tau", "lambda_pot_peak_nm",
        "lambda_dep_peak_nm")


def test_sparse_missing_readings_leave_the_fit_close_and_are_reported(suite_and_model):
    suite, clean = suite_and_model
    gappy = fitting.extract_synapse_model(_with_gaps(suite), verbose=False)
    for k in KEYS:
        assert gappy[k] == pytest.approx(clean[k], rel=0.08), k
    # Rate constants rest on fewer points but must stay physical, not 63x off.
    for k in ("A_peak", "B_peak"):
        assert gappy[k] == pytest.approx(clean[k], rel=0.3), k
    warnings = gappy["extraction_report"]["warnings"]
    assert any("non-finite" in w for w in warnings)


def test_null_readings_are_treated_like_nan(suite_and_model):
    suite, clean = suite_and_model
    s = _with_gaps(suite)
    for d in s["datasets"]:
        d["y_data"] = [None if v != v else v for v in d["y_data"]]
    model = fitting.extract_synapse_model(s, verbose=False)
    assert model["G_max"] == pytest.approx(clean["G_max"], rel=0.05)


def test_a_dataset_with_no_finite_reading_is_excluded_and_reported(suite_and_model):
    suite, _ = suite_and_model
    s = copy.deepcopy(suite)
    victim = next(d for d in s["datasets"]
                  if d["metadata"].get("experiment_type") == "depression")
    victim["y_data"] = [float("nan")] * len(victim["y_data"])
    model = fitting.extract_synapse_model(s, verbose=False)
    assert any(victim["filename"] in w and "excluded" in w
               for w in model["extraction_report"]["warnings"])
    assert np.isfinite(model["B_peak"])


def test_gaps_in_pulse_trains_keep_pulse_numbering():
    """Deleting a reading would merge two pulses into one increment."""
    ds = {"filename": "t.csv", "metadata": {"experiment_type": "potentiation"},
          "x_data": [0, 1, 2, 3], "y_data": [1.0, float("nan"), 3.0, 4.0]}
    cleaned, warnings = fitting.DataParser._sanitise_datasets([ds])
    assert len(cleaned[0]["y_data"]) == 4 and np.isnan(cleaned[0]["y_data"][1])
    assert ds["y_data"][1] != ds["y_data"][1]  # input untouched (still NaN, not replaced)
    assert len(warnings) == 1


def test_independent_point_experiments_drop_the_missing_pair():
    ds = {"filename": "r.csv", "metadata": {"experiment_type": "retention"},
          "x_data": [0, 10, 20], "y_data": [5.0, None, 3.0]}
    cleaned, _ = fitting.DataParser._sanitise_datasets([ds])
    assert cleaned[0]["x_data"] == [0.0, 20.0]
    assert cleaned[0]["y_data"] == [5.0, 3.0]


def test_relaxation_is_removed_from_the_spectral_response(suite_and_model):
    """Off-band wavelengths have no optical response once dark drift is removed."""
    suite, model = suite_and_model
    parsed = fitting.DataParser.parse_suite(suite)
    corrected = fitting._subtract_relaxation_from_wavelength_response(
        parsed["wavelength_response"], model["G_min"], model["decay_tau"])
    wl = np.array(corrected["wavelengths_nm"], dtype=float)
    resp = np.array(corrected["delta_g_normalized"])
    raw = np.array(parsed["wavelength_response"]["delta_g_normalized"])
    off_band = wl >= 650
    assert off_band.any()
    assert np.max(np.abs(resp[off_band])) < 0.05
    assert np.max(np.abs(raw[off_band])) > 0.08  # the artefact this removes
    assert corrected["relaxation_subtracted"]["decay_tau_s"] == model["decay_tau"]


def test_depression_band_width_is_recovered(suite_and_model):
    _, model = suite_and_model
    # Generator truth: 30 nm for both bands. Uncorrected relaxation gave ~36 nm.
    assert model["lambda_dep_width_nm"] == pytest.approx(30.0, rel=0.10)
    assert model["lambda_pot_width_nm"] == pytest.approx(30.0, rel=0.10)


def test_implausible_conductance_window_is_flagged(suite_and_model):
    suite, _ = suite_and_model
    s = copy.deepcopy(suite)
    for d in s["datasets"]:
        d["y_data"] = [v * 1e6 for v in d["y_data"]]  # µS values declared as S
    model = fitting.extract_synapse_model(s, verbose=False)
    assert any("implausible conductance window" in w
               for w in model["extraction_report"]["warnings"])
