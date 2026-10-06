"""Project-wide Tk callback error reporting.

Tk's default `report_callback_exception` prints a traceback to stderr and
returns. In a GUI that means an unhandled exception inside any button handler
is INVISIBLE: the window stays alive, the operation silently did nothing, and
any button the handler disabled on entry stays disabled forever — the user is
left looking at a greyed control reading "Stimulating..." with no explanation.

`Main.py` in particular had a 220-line `on_stimulate` with no `try` at all and
no `finally` around its button re-enable, so every exception raised inside
`apply_stimulus`, `apply_spatial_pattern`, `resize_pattern` or `update_plots`
failed exactly that way — including the carefully-worded refusals those
functions raise, whose text went only to a console the user was not reading.

Installing this hook converts that entire class of failure into a visible,
copyable dialog naming the operation and the exception, while still printing
the full traceback for diagnosis. It does not swallow anything: nothing is
suppressed, nothing is defaulted, the error is simply also SHOWN.
"""

import sys
import traceback


def install_tk_error_reporter(root, app_name="SYNAPSYS", on_error=None):
    """Route unhandled Tk callback exceptions to a dialog as well as stderr.

    Args:
        root: the Tk/CTk root window
        app_name: shown in the dialog title so the user knows which window
        on_error: optional zero-argument callable run before the dialog, for
            restoring UI state. Handlers here disable their button on entry and
            re-enable it only on the normal path, so an exception leaves a
            permanently greyed button reading "Running Demo...". Passing a
            restorer makes the window usable again after a failure. Its own
            exceptions are ignored — recovery must never mask the original
            fault, which has already been printed.

    Safe to call more than once; the last call wins.
    """
    from tkinter import messagebox

    def report_callback_exception(exc_type, exc_value, exc_tb):
        if on_error is not None:
            try:
                on_error()
            except Exception:
                pass
        # Always emit the full traceback first — it is the diagnostic record,
        # and it must survive even if the dialog itself cannot be shown.
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        try:
            detail = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
            # Keep the dialog readable; the console has the whole thing.
            tail = detail.strip().splitlines()
            if len(tail) > 12:
                tail = ["... (full traceback printed to the console) ..."] + tail[-12:]
            messagebox.showerror(
                f"{app_name} — Unhandled Error",
                f"{exc_type.__name__}: {exc_value}\n\n"
                "The operation did not complete. Nothing was saved or applied "
                "by the step that failed.\n\n"
                + "\n".join(tail)
            )
        except Exception:
            # A dialog failure must not mask the original exception, which has
            # already been printed above.
            pass

    root.report_callback_exception = report_callback_exception
    return root
