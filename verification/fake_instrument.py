"""Fake Keithley 2600-series instrument — exercises the hardware stack with no hardware.

Enough of the pyvisa resource surface for `synapse_engine` to run against:
`query()`, `write()`, `read()` and a `timeout` attribute. Every write is
recorded, so a test can assert on what was actually commanded — which is how
execution-path parity is checked (both paths must configure fixed ranging, take
compliance from the instrument, and derive averaging the same way).

The LUA path is exercised by monkeypatching `synapse_engine.execute_lua_script_fast`
rather than by emulating a TSP interpreter: what matters for parity is the shape
of the data each path returns and the commands each path issues, not that the
fake can run Lua.
"""

import re

import numpy as np


class FakeInstrumentError(RuntimeError):
    """Raised when the fake is asked something it was not configured to answer.

    Deliberately loud. A fake that returns a plausible default for an
    unrecognised query would let a test pass against behaviour nobody wrote.
    """


class FakeKeithley:
    """A 2600-series SMU that measures a fixed, configurable current.

    Args:
        model: model string reported by ``*IDN?``.
        current_A: current returned by every ``measure.i()`` query. May also be
            a callable taking the read channel name, for state-dependent
            behaviour.
        idn_raises: when set, ``*IDN?`` raises this exception — used to prove
            that an unidentifiable instrument raises
            ``InstrumentIdentificationError`` rather than guessing a ceiling.
    """

    def __init__(self, model="2636A", current_A=5.0e-6, idn_raises=None,
                 line_freq_hz=50.0):
        self.model = model
        self.current_A = current_A
        self.idn_raises = idn_raises
        self.line_freq_hz = line_freq_hz
        self.timeout = 5000

        self.writes = []          # every command issued, in order
        self.queries = []         # every query issued, in order
        self._read_queue = []     # lines handed back by read()

    # --- pyvisa surface ------------------------------------------------

    def write(self, command):
        self.writes.append(command)

    def read(self):
        if self._read_queue:
            return self._read_queue.pop(0)
        raise FakeInstrumentError("read() called with nothing queued")

    def query(self, command):
        self.queries.append(command)

        if "*IDN?" in command:
            if self.idn_raises is not None:
                raise self.idn_raises
            return f"Keithley Instruments Inc., Model {self.model}, 1398687, 1.4.2"

        if "errorqueue.count" in command:
            return "0"

        # Mains frequency the instrument auto-detected at power-up.
        if "localnode.linefreq" in command:
            return f"{float(self.line_freq_hz):.6e}"

        if "measure.i()" in command:
            ch = _channel_of(command)
            value = self.current_A(ch) if callable(self.current_A) else self.current_A
            return f"{value:.6e}"

        if "measure.v()" in command:
            return "0.000000e+00"

        # A buffer length probe: report exactly what was appended.
        if re.search(r"\bnvbuffer\d\.n\b", command):
            return "0"

        if "*OPC?" in command:
            return "1"

        raise FakeInstrumentError(
            f"FakeKeithley received an unhandled query: {command!r}. "
            "Add an explicit response rather than letting the fake guess."
        )

    def close(self):
        pass

    # --- test helpers ---------------------------------------------------

    def writes_matching(self, pattern):
        """Every recorded write matching `pattern` (regex, searched)."""
        rx = re.compile(pattern)
        return [w for w in self.writes if rx.search(w)]

    def wrote(self, pattern):
        return bool(self.writes_matching(pattern))


def _channel_of(command):
    m = re.search(r"\b(smu[ab])\b", command)
    return m.group(1) if m else "smua"


def fake_lua_result(n_pulses, period_ms=10.0, current_A=5.0e-6):
    """A stand-in return value for `execute_lua_script_fast`.

    Returns ``(timestamps, currents)`` with ``n_pulses + 1`` points — index 0 is
    the pre-stimulus baseline, matching what the real LUA script emits. A test
    asserting parity relies on this length being the same as the standard
    path's.
    """
    n_points = n_pulses + 1
    timestamps = np.arange(n_points) * (period_ms / 1000.0)
    currents = np.full(n_points, current_A, dtype=float)
    return timestamps.tolist(), currents.tolist()
