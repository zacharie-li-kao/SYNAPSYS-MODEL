# SYNAPSYS verification harness

Built for the repair campaign in `AUDIT_2026-07-31.md` §11.2, then used to
verify all of §9 and every work package P1–P10. Nothing in the project could previously
prove a fix worked, and several of the fixes — the STDP normalisation above all
— are easy to get wrong in a way that still looks plausible.

## Running

```
python -m pytest verification/ -q
```

**197 tests, about 8 seconds.** No hardware, no GUI, no `PYTHONIOENCODING` (that
was a workaround for C5/H22, both now fixed at source).

This is the authority on what the suite currently does. `AUDIT_2026-07-31.md`
§1–§12 is the audit as originally written and is now largely historical; §13
records the repair campaign.

## What is here

**`fake_instrument.py` + `test_execution_path_parity.py`** — a fake 2600-series
SMU with `query()`/`write()`/`read()`. Records every command, so tests assert on
what was actually commanded. Locks in the §10 fixes: both execution paths return
`n_pulses + 1` points with index 0 = baseline, NaN at zero read bias, fixed
ranging, compliance from the instrument, one shared averaging derivation;
`MODEL_CAPABILITIES` is the only capability table; an unidentifiable instrument
raises; an unregistered mode raises; the LUA `try` guards only the instrument
call. The LUA path is exercised by monkeypatching `execute_lua_script_fast`
rather than emulating a TSP interpreter.

**`pipeline.py`** — headless generate → convert → fit. Reuses
`data_converter`'s real logic (`parse_metadata_file` called unbound,
`DataProcessor.process_dataset` as the GUI calls it); only the batch-load loop
is replicated, because it is interleaved with widget updates. Runnable directly:
`python verification/pipeline.py <outdir>`.

**`baseline.py` + `test_pipeline_regression.py`** — the recorded state of every
fitted parameter, table-driven. Each entry is `PINNED` (correct today, guarded
against regression) or `KNOWN_BAD` (a live audit finding, pinned at its current
wrong value with the target and the finding recorded). A `KNOWN_BAD` test
**failing is the fix landing** — confirm the new value, then flip `status` to
`PINNED`.

**`test_synapse_physics.py`** — the device ODE: branch selection with and
without a fitted spectral curve (C1), rest and decay, integration bookkeeping,
clipping.

**`test_export_contract.py`** — the Keithley → JSON path (§6, F1–F8). Asserts
against `fitting.py`'s real dispatch rules, not against the exporter's own idea
of what it produced, and ends with an end-to-end check that a realistic export
reaches all six extractors.

**`test_snn_learning.py`** — the spiking learning loop (C2/C3/C4): LTD
reachable, causal pairs counted, drive derived from the LIF chain.

**`test_network_gui_contract.py`** — mismatches between what `Main.py` calls and
what `network.py` provides (C12, M17, M18). No running GUI needed.

**`test_hardware_fixes.py`** — PV extraction, safety checks, STDP protocol
integrity, buffer retrieval, and the JV buffer/cleanup fixes (H3/H5/H6/H14–H18/H20).

**`test_cycle_and_variability.py`** — cycle de-interleaving and instrument state
leakage (H8/H9/H11/H12), and device variability that can no longer produce a
negative conductance (M14).

**`test_reentrancy_and_transistor.py`** — the measurement re-entrancy guard
(H21) and the transistor sweep's failure path (H19). Both do nothing on a clean
run and produce a second instrument session, or fabricated duplicate curves,
when something goes wrong.

**`test_visualisation_and_io.py`** — the units contract (M11), fitted-model
save/load (C9, M2), firing-rate binning (M19), the double µS conversion (M20),
the Excel loader's four silent failure modes (M21), and cycle-schematic fidelity
(M25–M27).

## Determinism

`generate_test_data.py` gained an optional `--seed N`. Unseeded behaviour is
unchanged for interactive use; the harness always passes seed `20260731`.
Verified bit-for-bit reproducible across runs. Without this, a shift in a fitted
parameter cannot be told apart from ordinary noise scatter.

## Current state against ground truth

**Every parameter is within 25% of ground truth, most within a few percent.**
Full table in `AUDIT_2026-07-31.md` §13.1; the headlines:

| Parameter | Truth | Audit §1 | Now |
|---|---|---|---|
| alpha | 0.75 | 0.7865 | 0.7572 (+1.0%) |
| beta | 0.75 | 0.8183 | 0.7556 (+0.7%) |
| λ dep width | 30 nm | 48.87 (FAIL +63%) | 34.78 (+15.9%) |
| B_peak | 6e-4 | 9.994e-3 (FAIL 16.7×) | 6.320e-4 (+5.3%) |
| STDP A_plus | 0.5 | 19.096 (FAIL 38×) | 0.4517 (−9.7%) |
| STDP A_minus | 0.3 | 6.503 (FAIL 21.7×) | 0.2306 (−23.1%) |
| STDP τ₊ / τ₋ | 20 ms | 14.00 / 41.96 (FAIL) | 23.26 / 24.03 ms |
| SRDP f₀ | 10 Hz | 13.71 (+37%) | 9.162 (−8.4%) |

The last of the accuracy came from making the physics consistent rather than
from tuning the fitter: relaxation is now applied during stimulation everywhere
(simulator, generator, and subtracted inside every rate and exponent fit), which
alone moved alpha from +4.3% to +1.0% and B_peak from −22% to +5.3%.

One residual is a property of the DATA and is documented as such in
`baseline.py`, with a tolerance that says so honestly: `G_min` at +20%, because
no trace in the suite reaches it any more. Do not tighten it by tuning the
fitter against this one seeded realisation — that is fitting the harness, not
the physics.

## The STDP trap

Recorded here because it is the single easiest thing to get wrong. The spec rule

```
ΔG = A_plus · exp(−dt/τ) · (G_max − G_min) · 0.01
```

already contains the `0.01`. So `A_plus = 0.5` means 0.5 **percent** of the
dynamic range — `A_plus` is numerically a percentage, not a fraction. The fit
target is `100 · ΔG / (G_max − G_min)` — normalised by the **dynamic range**,
never by the baseline conductance — and **neither consumer divides by 100
afterwards**. Reading "fraction" intuitively lands you 100× off.

Two things go with it, and both are guarded by tests:

- `stdp_model` builds its output with `dtype=float`, not `np.zeros_like(dt)`.
  A whole-millisecond hardware sweep gives an *integer* Δt array, which
  `zeros_like` inherits, and the float exponentials then truncate on assignment.
  Harmless at the old ~19 amplitudes; at the correct ~0.5 it silently zeroes the
  entire LTP branch while every fitted parameter still looks plausible.
- The generator averages 250 pairings per Δt point. At the spec's rule scale a
  single pairing changes G by 0.5% of range against 1.5% read noise — signal
  below noise. Real STDP protocols average for exactly this reason; the old
  generator instead used a 15× rule scale, which hid the problem and inflated
  every fitted amplitude.

## A note on bounded fits

Passing `bounds` to `curve_fit` switches SciPy from `lm` to `trf`, whose
stopping tests are far looser. On conductance data (residuals ~1e-6) `trf` reads
that as "already converged" and **returns the initial guess unchanged, reporting
success**. This was diagnosed twice the hard way here: `alpha` came back as
exactly 0.7999 from `p0 = 0.8`, and `decay_tau` as exactly 100.0 s — numerically
identical to the documented default, and therefore indistinguishable from a
parameter that was never fitted at all.

Every bounded fit in `fitting.py` passes `**_FIT_TOL` and `x_scale='jac'`, and
the critical ones call `_assert_moved_from_start`. Removing either silently
reintroduces fits that are not fits.
