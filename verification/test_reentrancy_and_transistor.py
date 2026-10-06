"""H19 and H21 — the transistor failure path and measurement re-entrancy.

Both are failure-path defects: they do nothing on a clean run and produce
fabricated data or a second instrument session when something goes wrong.
"""

import inspect

import pytest

import keithley_analyser as ka


# --- H21: re-entrancy ------------------------------------------------------

MEASUREMENT_ENTRY_POINTS = [
    "run_cycle_characterization",
    "run_multi_device_characterization",
    "run_srdp_characterization",
    "run_stdp_characterization",
]


@pytest.mark.parametrize("fn_name", MEASUREMENT_ENTRY_POINTS)
def test_every_measurement_entry_point_refuses_re_entry(fn_name):
    """H21 — `root.update()` dispatches input events during a live session.

    A second click on Run therefore re-entered the handler and opened a SECOND
    VISA session against the same instrument, mid-sweep. There was no
    in-progress guard of any kind.
    """
    fn = getattr(ka, fn_name)
    # The wrapped implementation, for the functions that have wrappers.
    source = inspect.getsource(fn)
    if "original_run_srdp" in source:
        source += inspect.getsource(ka.original_run_srdp)

    assert "_measurement_in_progress" in source, (
        f"{fn_name} has no re-entrancy guard"
    )
    assert "finally:" in source, (
        f"{fn_name} does not clear the guard in a finally — an exception would "
        "leave every subsequent measurement permanently refused"
    )


@pytest.mark.parametrize("fn_name", MEASUREMENT_ENTRY_POINTS)
def test_a_guarded_entry_point_returns_immediately_when_busy(fn_name, monkeypatch):
    """The guard, exercised rather than read.

    With the flag set, the function must return before touching any widget or
    opening any session.
    """
    warned = []
    monkeypatch.setattr(ka.messagebox, "showwarning",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(ka, "_measurement_in_progress", True)

    # Would raise on any widget access if the guard did not fire first.
    getattr(ka, fn_name)()

    assert warned, f"{fn_name} did not warn when already busy"
    assert "In Progress" in warned[0][0]


def test_the_guard_is_clear_at_rest():
    assert ka._measurement_in_progress is False


def test_realtime_callbacks_do_not_dispatch_input_events():
    """`update_idletasks()` redraws; `update()` also delivers button clicks."""
    for fn_name in ("run_cycle_characterization",
                    "run_multi_device_characterization"):
        source = inspect.getsource(getattr(ka, fn_name))
        assert "root.update_idletasks()" in source, (
            f"{fn_name}'s live-plot callback should use update_idletasks()"
        )
        assert "root.update()" not in source, (
            f"{fn_name} still calls root.update() during a live VISA session"
        )


# --- H19: the transistor sweep --------------------------------------------


def test_a_failed_gate_point_is_omitted_not_duplicated():
    """H19 — a mid-sweep failure relabelled the PREVIOUS curve and re-appended it.

    The old check was `if not jv_curves`, which only catches "no curve has ever
    been recorded". On a failure partway through a series, `jv_curves[-1]` was
    still the previous gate point's trace, so it was stamped with the new V_GS
    and appended again — two identical curves under different gate voltages,
    indistinguishable from real data.
    """
    source = inspect.getsource(ka.original_run_transistor)

    # Scope to the gate-sweep loop. A separate `if not jv_curves:` guards the
    # plotting step further down and is perfectly correct there.
    gate_loop = source.split("for vgs in gate_voltages:")[1].split(
        "if failed_gate_points:")[0]

    assert "n_before = len(jv_curves)" in gate_loop, (
        "the sweep does not detect whether a gate point produced a new curve"
    )
    assert "len(jv_curves) == n_before" in gate_loop
    assert "if not jv_curves:" not in gate_loop, (
        "the old check, which cannot detect a mid-series failure, is still "
        "there — it only fires when NO curve has ever been recorded"
    )
    assert "failed_gate_points" in source, (
        "omitted gate points are not reported to the user"
    )


# --- the safety checks must have survived the guard rewrite ----------------


@pytest.mark.parametrize("fn_name", MEASUREMENT_ENTRY_POINTS[:1] + MEASUREMENT_ENTRY_POINTS[2:])
def test_safety_checks_survived(fn_name):
    """Guard insertion re-indented these function bodies; H15 must still hold."""
    source = inspect.getsource(getattr(ka, fn_name))
    if "original_run_srdp" in source:
        source += inspect.getsource(ka.original_run_srdp)
    assert "check_synapse_safety" in source, (
        f"{fn_name} lost its safety check during the re-entrancy rewrite"
    )
