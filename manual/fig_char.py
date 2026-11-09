"""Figures for the characterisation chapter, from the real GUI on the simulated 2600.

Run:  python -m manual.fig_char            (all figures into manual/figures)
      FIG_OUT=somewhere python -m manual.fig_char visual retention   (a subset)
"""
import os, sys, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from manual.grab import grab, grab_dialog, client_bbox
from manual import sim_instrument
sim_instrument.install()
import matplotlib; matplotlib.use("TkAgg")
import keithley_analyser as ka
from tkinter import messagebox
from PIL import ImageDraw, ImageFont

FIG = os.path.join(ROOT, "manual", "figures")
OUT = os.environ.get("FIG_OUT", FIG)
LOG = []
root = ka.root
_real_info = messagebox.showinfo
_real_askyesno = messagebox.askyesno
GRAB_NEXT = {}   # dialog kind -> file name; the next dialog of that kind is photographed


def fake(kind):
    def f(title, msg, **k):
        LOG.append((kind, title, msg))
        name = GRAB_NEXT.pop(kind, None)
        if name:
            root.after(500, lambda: grab_dialog(title, os.path.join(OUT, name)))
            return (_real_info if kind == "showinfo" else getattr(messagebox, "_r_" + kind))(title, msg)
        return "ok"
    return f


for n in ("showwarning", "showerror"):
    setattr(messagebox, "_r_" + n, getattr(messagebox, n))
    setattr(messagebox, n, fake(n))
messagebox.showinfo = fake("showinfo")
ka.is_port_open = lambda *a, **k: False


def _no_prompt(title, msg=None, **k):
    raise AssertionError(f"unexpected confirmation prompt: {title}: {msg}")


# The suite warns above 2.5 V of stimulation; the figures never ask for more, and any
# confirmation prompt that does appear fails loudly instead of waiting for a click.
for n in ("askyesno", "askokcancel", "askretrycancel", "askquestion"):
    setattr(messagebox, n, _no_prompt)


def setv(entry, text):
    entry.delete(0, "end"); entry.insert(0, str(text))


def corner(widget):
    """Marker position: above the widget's right end, clear of its text."""
    x0, y0, _, _ = client_bbox(root)
    return (widget.winfo_rootx() - x0 + widget.winfo_width() - 40,
            widget.winfo_rooty() - y0 - 22)


def scroll_to(widget, pad=10):
    cv = ka.left_panel._parent_canvas
    cv.yview_moveto(0); root.update(); root.update_idletasks()
    total = cv.bbox("all")[3]
    off = widget.winfo_rooty() - ka.left_panel.winfo_rooty() - pad
    cv.yview_moveto(min(max(0.0, off / total), 1.0))
    root.update()


def centre(widget):
    """Centre of a widget in the grabbed client area."""
    x0, y0, _, _ = client_bbox(root)
    return (widget.winfo_rootx() - x0 + widget.winfo_width() // 2,
            widget.winfo_rooty() - y0 + widget.winfo_height() // 2)


def mark(img, items):
    """Numbered red markers; items = [(n, (x, y))]."""
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 26)
    for n, (x, y) in items:
        r = 20
        d.ellipse((x - r, y - r, x + r, y + r), fill=(200, 30, 30), outline="white", width=3)
        d.text((x, y - 1), str(n), fill="white", font=f, anchor="mm")
    return img


def shot(name):
    img = grab(root)
    img.save(os.path.join(OUT, name))
    print("saved", name, img.size)
    return img


def mode(sub, fields, button, name, g0=2e-6):
    sim_instrument.NEXT_G0 = g0
    sim_instrument.WALL_RELAX = sub == "Retention"
    ka.synapse_submode.set(sub); ka.update_synapse_submode(); root.update()
    for entry, val in fields:
        setv(getattr(ka, entry) if isinstance(entry, str) else entry, val)
    n0 = len(LOG)
    getattr(ka, button).invoke(); root.update()
    scroll_to(ka.synapse_frame)
    if name:
        shot(name)
    print(sub, "->", LOG[n0:])


def find(widget, cls, text=None):
    out = []
    for w in widget.winfo_children():
        if w.winfo_class() == cls and (text is None or w.cget("text") == text):
            out.append(w)
        out += find(w, cls, text)
    return out


def export_flow():
    ka.export_characterization_suite(); root.update()
    top = [w for w in root.winfo_children() if w.winfo_class() == "Toplevel"][-1]
    top.geometry("820x760+200+60"); root.update()
    grab(top, os.path.join(OUT, "char_export_empty.png"))
    lb = find(top, "Listbox")[0]
    # stored-run order: 0 LTP, 1 LTD, 2 Visual, 3 SRDP, 4 STDP, 5 Cycles, 6 Retention
    roles = {0: "Mark as LTP (α)", 1: "Mark as LTD (β)", 5: "Mark as Cycles (G_min/G_max)",
             6: "Mark as Retention (τ)", 4: "Mark as STDP", 3: "Mark as SRDP"}
    for idx, label in roles.items():
        lb.selection_clear(0, "end"); lb.selection_set(idx); lb.activate(idx)
        find(top, "Button", label)[0].invoke(); root.update()
    lb.selection_clear(0, "end"); root.update()
    grab(top, os.path.join(OUT, "char_export.png"))
    import tkinter.filedialog as fd
    import keithley_analyser
    keithley_analyser.asksaveasfilename = lambda **k: os.path.join(OUT, "_suite_from_instrument.json")
    fd.asksaveasfilename = keithley_analyser.asksaveasfilename
    n0 = len(LOG)
    find(top, "Button", "Export Suite")[0].invoke(); root.update()
    print("export ->", LOG[n0:])


def main(only=None):
    ka.connection_type.set("LAN")
    setv(ka.port_entry, "192.168.1.50")
    setv(ka.sample_name_entry, "ZnO-A3")
    ka.mode_selection.set("Synapse"); ka.update_mode_display(); root.update()
    for name in dir(ka):
        if name.endswith("sample_id_entry"):
            setv(getattr(ka, name), "ZnO-A3")
    ka.left_panel._parent_canvas.yview_moveto(0); root.update()
    img = grab(root)
    mark(img, [(1, corner(ka.connection_type)), (2, corner(ka.port_entry))])
    img.crop((0, 0, 560, 560)).save(os.path.join(OUT, "char_connection.png"))
    scroll_to(ka.mode_selection, pad=80)
    img = grab(root)
    mark(img, [(1, corner(ka.mode_selection)), (2, corner(ka.synapse_submode))])
    img.crop((0, 0, 560, 1050)).save(os.path.join(OUT, "char_modes.png"))
    steps = [
        ("Basic", "basic", [], "run_synapse_button", "char_basic.png"),
        ("Basic", "ltd", [("stim_level_entry", -1.0)], "run_synapse_button", None, 7e-5),
        ("Visual (Self-Powered)", "visual", [("visual_n_pulses_entry", 30), ("visual_light_voltage_entry", 2.0)],
         "run_visual_button", "char_visual.png"),
        ("SRDP", "srdp", [("srdp_freq_end_entry", 10), ("srdp_freq_points_entry", 5), ("srdp_n_pulses_entry", 20)],
         "run_srdp_button", "char_srdp.png"),
        ("STDP", "stdp", [("stdp_dt_points_entry", 8), ("stdp_n_pairs_entry", 20)], "run_stdp_button", "char_stdp.png"),
        ("Cycle", "cycle", [("cycle_n_cycles_entry", 3)], "run_cycle_button", "char_cycle.png"),
        ("Retention", "retention", [("retention_points_entry", 10), ("retention_interval_entry", 3)],
         "run_retention_button", "char_retention.png", 6e-5),
    ]
    for sub, key, fields, btn, name, *g0 in steps:
        if only and key not in only:
            continue
        if key == "basic":
            GRAB_NEXT["showinfo"] = "char_success.png"
        try:
            mode(sub, fields, btn, name, *g0)
        except Exception:
            traceback.print_exc()
    if not only or "warn" in only:
        # a deliberately too fast SRDP point: this instrument cannot deliver 100 Hz with these read settings
        GRAB_NEXT["showwarning"] = "char_warning.png"

        def photographed_yes(title, msg=None, **k):
            root.after(500, lambda: grab_dialog(title, os.path.join(OUT, "char_freqrange.png"), answer=6))
            return _real_askyesno(title, msg)
        messagebox.askyesno = photographed_yes
        mode("SRDP", [("srdp_freq_start_entry", 10), ("srdp_freq_end_entry", 100),
                      ("srdp_freq_points_entry", 3), ("srdp_n_pulses_entry", 20)],
             "run_srdp_button", None)
        messagebox.askyesno = _no_prompt
    if not only or "export" in only:
        export_flow()
    return LOG


if __name__ == "__main__":
    main(sys.argv[1:] or None)
    root.destroy()
