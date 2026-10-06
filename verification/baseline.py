"""Recorded baseline for the synthetic pipeline, and the ground truth it should reach.

Table-driven so that landing a fix means flipping a `status`, not rewriting an
assertion. Each entry carries where the parameter is now and where it must end
up; `KNOWN_BAD` entries are audit findings not yet fixed, and their tests assert
the *current* wrong value so that an accidental change is still caught.

Numbers under `baseline` come from an actual seeded run (seed 20260731) of
`verification/pipeline.py`, not from the audit — the audit's run was unseeded,
so its exact figures are one realisation of the same noise process and are not
reproducible.

Ground truth is `generate_test_data.true_params`. As of step 7 the generator
declares what it actually generates (audit M22/M23/M24 fixed), so a comparison
against ground truth is now meaningful — it was not before.

## State against the audit's §1 table

Every parameter is now within 25% of ground truth, most within a few percent.
All six the audit marked FAIL are fixed, and so is the SRDP bias:

| Parameter    | Truth       | Audit fitted    | Now      | Error  |
|--------------|-------------|-----------------|----------|--------|
| alpha        | 0.75        | 0.7865          | 0.7572   | +1.0%  |
| beta         | 0.75        | 0.8183          | 0.7556   | +0.7%  |
| λ dep width  | 30 nm       | 48.87 (FAIL)    | 34.78 nm | +15.9% |
| B_peak       | 6e-4        | 9.99e-3 (16.7x) | 6.32e-4  | +5.3%  |
| A_peak       | 8e-4        | 1.105e-3        | 9.196e-4 | +15.0% |
| STDP A_plus  | 0.5         | 19.10 (38x)     | 0.4517   | -9.7%  |
| STDP A_minus | 0.3         | 6.50 (21.7x)    | 0.2306   | -23.1% |
| STDP tau_+   | 20 ms       | 14.00 (-30%)    | 23.26 ms | +16.3% |
| STDP tau_-   | 20 ms       | 41.96 (+110%)   | 24.03 ms | +20.1% |
| SRDP f0      | 10 Hz       | 13.71 (+37%)    | 9.162    | -8.4%  |

The last of the accuracy came from making the physics consistent rather than
from tuning the fitter. Relaxation is now applied during stimulation everywhere
— simulator, generator, and subtracted inside every rate and exponent fit —
which alone moved alpha from +4.3% to +1.0%, beta from -2.7% to +0.7%, A_peak
from +43% to +15% and B_peak from -22% to +5.3%.

ONE parameter carries a residual bias that is a property of the DATA rather than
of the fitter, and is documented as such below: G_min (+20%). A_peak's +15% is
alpha's residual +1% amplified through the power-law prefactor, which is
inherent to the model rather than a defect in the fit.
"""

PIPELINE_SEED = 20260731

# Status meanings:
#   PINNED    — currently correct; a test guards it against regression.
#   KNOWN_BAD — a live audit finding; the test asserts the current wrong value
#               and names the target, so the test fails loudly when it is fixed
#               (at which point flip the status and the assertion).
PINNED = "pinned"
KNOWN_BAD = "known_bad"

BASELINE = {
    # --- dynamic range ---
    "G_max": dict(status=PINNED, truth=1e-4, baseline=9.957e-5, rel_tol=0.10),

    # G_min is +20% and cannot be better from this suite. Since the generator
    # stopped applying an undeclared x10 to light depression (M24), no
    # depression trace bottoms out at G_min within 50 pulses, so the lowest
    # conductance any dataset reaches is the LTP trace's starting point at
    # G_min * 1.2. The suite does not contain the information; reporting 1.2e-6
    # is the honest answer, and inventing a lower one would not be.
    "G_min": dict(status=PINNED, truth=1e-6, baseline=1.2e-6, rel_tol=0.35,
                  note="limited by the data, not the fitter — no trace reaches G_min"),

    # --- exponents ---
    # Both within ~1% of truth once relaxation is subtracted from each sampled
    # increment rather than absorbed into the exponent. Tolerance is kept at 8%
    # rather than tightened to the measured accuracy: the underlying data still
    # carries 1.5% multiplicative noise, and a tolerance narrower than the
    # scatter would make this test flaky rather than strict.
    "alpha": dict(status=PINNED, truth=0.75, baseline=0.7572, rel_tol=0.08,
                  note="value is fine; provenance was not — see "
                       "test_alpha_is_fitted_from_the_dedicated_nonlinearity_trace"),
    "beta": dict(status=PINNED, truth=0.75, baseline=0.7556, rel_tol=0.08),

    # --- spectral response ---
    # Both branches now reported separately (M1) and measured from the initial
    # rate rather than a saturated endpoint (C11).
    "lambda_peak": dict(status=PINNED, truth=365.0, baseline=364.11, rel_tol=0.05,
                        note="top-level scalars are the POTENTIATION peak (M1)"),
    "lambda_width": dict(status=PINNED, truth=30.0, baseline=31.74, rel_tol=0.25),
    "lambda_pot_peak_nm": dict(status=PINNED, truth=365.0, baseline=364.11, rel_tol=0.05),
    "lambda_pot_width_nm": dict(status=PINNED, truth=30.0, baseline=31.74, rel_tol=0.25),
    "lambda_dep_peak_nm": dict(status=PINNED, truth=550.0, baseline=548.37, rel_tol=0.05),
    # The audit's headline failure: 48.87 nm against a true 30 nm, because the
    # response was read at a fixed pulse count by which the depression runs had
    # bottomed out. Now measured from dG/dn|0. The tolerance is 25% rather than
    # tighter because the depression traces carry the least signal in the suite
    # — see depression_nonlinearity_fit_R2 below.
    "lambda_dep_width_nm": dict(status=PINNED, truth=30.0, baseline=34.78, rel_tol=0.25),

    # --- retention ---
    "decay_tau": dict(status=PINNED, truth=100.0, baseline=103.00, rel_tol=0.20),

    # --- rate constants ---
    # B_peak was the audit's worst failure: 16.7x, railed at the optimiser's
    # upper bound and reported as a successful extraction. Now +5%.
    "B_peak": dict(status=PINNED, truth=6e-4, baseline=6.320e-4, rel_tol=0.20),

    # A_peak amplifies alpha's residual error through the power-law prefactor —
    # the base (G_max - G) is ~1e-4, so a percent of exponent error becomes
    # several percent here. At alpha = 0.757 that leaves +15%, down from +43%
    # when relaxation was still being absorbed into the exponent.
    "A_peak": dict(status=PINNED, truth=8e-4, baseline=9.196e-4, rel_tol=0.30,
                   note="amplifies alpha's residual error through the "
                        "power-law prefactor"),

    # --- fit quality ---
    "wavelength_fit_R2": dict(status=PINNED, truth=1.0, baseline=0.9717, min_value=0.90),
    "nonlinearity_fit_R2": dict(status=PINNED, truth=1.0, baseline=0.9365, min_value=0.85),
    "retention_fit_R2": dict(status=PINNED, truth=1.0, baseline=0.9988, min_value=0.95),
    "srdp_fit_R2": dict(status=PINNED, truth=1.0, baseline=0.9886, min_value=0.95),
    "stdp_fit_R2": dict(status=PINNED, truth=1.0, baseline=0.8668, min_value=0.75),
    # Lower than the others because the depression traces move little now that
    # the undeclared x10 is gone, so noise is a larger fraction of the signal.
    # beta is nonetheless accurate to 0.7% — a modest R² on a low-amplitude
    # trace is not the same thing as an inaccurate parameter.
    "depression_nonlinearity_fit_R2": dict(status=PINNED, truth=1.0,
                                           baseline=0.7515, min_value=0.60),

    # --- STDP, all four fixed ---
    "stdp_A_plus": dict(status=PINNED, truth=0.5, baseline=0.4517, rel_tol=0.25,
                        note="PERCENT of dynamic range — the spec rule carries "
                             "its own 0.01 factor"),
    "stdp_A_minus": dict(status=PINNED, truth=0.3, baseline=0.2306, rel_tol=0.30),
    "stdp_tau_plus_ms": dict(status=PINNED, truth=20.0, baseline=23.26, rel_tol=0.30),
    "stdp_tau_minus_ms": dict(status=PINNED, truth=20.0, baseline=24.03, rel_tol=0.30),

    # --- SRDP ---
    # Was 13.71 Hz in the audit (+37%). Three separate defects contributed and
    # all three are fixed: the sigmoid was fitted in LINEAR frequency over
    # log-spaced data (M7), its baseline was taken from the lowest-frequency
    # point which already carries ~27% of the response (M8), and the GENERATOR
    # emitted a linear-frequency sigmoid so it was not even describing the same
    # physics the fitter assumed. On hardware, H14 additionally recorded
    # nominal rather than delivered frequencies; that is now read back from the
    # instrument's own timestamps.
    "srdp_transition_freq_hz": dict(status=PINNED, truth=10.0, baseline=9.162,
                                    rel_tol=0.20),
    # Slope in DECADES of frequency, matching the fitted and default units.
    "srdp_slope": dict(status=PINNED, truth=0.3, baseline=0.2461, rel_tol=0.30),
    # PERCENT of dynamic range, matching STDP. Fitted and default paths agreed
    # on neither the value nor the unit before (0.2 fraction vs ~28.6 percent).
    "srdp_max_change": dict(status=PINNED, truth=20.0, baseline=20.03,
                            rel_tol=0.20),
}

# Ground-truth STDP values, for tests that assert the unit contract directly.
# A_plus/A_minus are PERCENTAGES of the dynamic range, because the spec rule
#     ΔG = A_plus · exp(−dt/τ) · (G_max − G_min) · 0.01
# already contains the 0.01. The fit target is 100·ΔG/(G_max−G_min) and neither
# consumer divides by 100 afterwards. Reading "fraction" intuitively lands you
# 100x off.
STDP_TRUTH = {
    "stdp_A_plus": 0.5,
    "stdp_A_minus": 0.3,
    "stdp_tau_plus_ms": 20.0,
    "stdp_tau_minus_ms": 20.0,
}
