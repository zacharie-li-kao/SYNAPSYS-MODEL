"""Console output encoding for SYNAPSYS.

Why this module exists
----------------------
The suite prints physics in its natural notation: µS, Δt, λ, ✓, ⚠. On Windows
the default console codepage is cp1252, which cannot encode any of them, so
`print()` raises UnicodeEncodeError. That is not a cosmetic problem — the
failures observed were:

  * `generate_test_data.py` died partway through, having written 23 of its 26
    files. The STDP and SRDP datasets and METADATA_INSTRUCTIONS.txt were never
    written, so the characterization suite could not be batch-loaded or fitted
    at all.
  * `pulse_read_sequence` completed a LUA measurement successfully and then
    crashed on the "Completed" line. The exception was caught by the fallback
    handler, reported to the user as "LUA execution failed", and the entire
    measurement was re-run on the slower PC-timed path — blaming the instrument
    for a console limitation.

Both are the same root cause, and stripping the characters one by one would
only work until the next one is typed. Reconfiguring the stream fixes every
print in the process at once, and keeps the notation.

Call `enable_utf8_console()` once, as early as possible, in every entry point.
It is idempotent and safe when stdout is absent (pythonw, frozen GUI builds).
"""

import sys

_CONFIGURED = False


def enable_utf8_console():
    """Make stdout/stderr accept non-ASCII text. Idempotent; never raises.

    Uses errors='replace' rather than 'strict' so that a stream which genuinely
    cannot represent a character degrades to '?' in the console instead of
    aborting the measurement. Console rendering is cosmetic; the measurement is
    not.

    Returns True if both streams are usable for non-ASCII output afterwards.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return True

    ok = True
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            # pythonw.exe / frozen GUI build with no console attached. Nothing
            # to configure, and print() is a no-op sink, so nothing to fix.
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # Not a TextIOWrapper (redirected to a Tk widget, StringIO, etc.).
            # Such sinks are normally already Unicode-capable.
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            # Stream already detached or refuses reconfiguration. Fall back to
            # replacing unencodable characters rather than failing.
            try:
                reconfigure(errors="replace")
            except Exception:
                ok = False

    _CONFIGURED = True
    return ok
