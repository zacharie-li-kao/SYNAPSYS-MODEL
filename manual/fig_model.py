"""Figures for the modelling chapter.

Run (scale 1.0 keeps the matplotlib canvases at their native size):
    SYN_SCALE=auto python -m manual.fig_model <suite.json> [step ...]
argv[1] is the suite written by the data converter.
"""
import os, sys, time, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from manual.grab import grab, client_bbox
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import customtkinter as ctk
import Main

SUITE = os.path.abspath(sys.argv[1])
ONLY = sys.argv[2:]
OUT = os.environ.get("FIG_OUT", os.path.join(ROOT, "manual", "figures"))
LOG = []
for n in ("showinfo", "showwarning", "showerror"):
    setattr(Main.tk_messagebox, n, lambda t, m, _n=n, **k: LOG.append((_n, t, m)))
Main.tk_messagebox.askyesno = lambda *a, **k: (_ for _ in ()).throw(AssertionError(a))
Main.askopenfilename = lambda *a, **k: SUITE
Main.asksaveasfilename = lambda *a, **k: os.path.join(OUT, "_fitted_model.json")
plt.show = lambda *a, **k: LOG.append(("plt.show", "", ""))

app = Main.SynapseSimulatorApp()
app.geometry(os.environ.get("GEOM", "1500x950"))
app.update()


def settle(seconds=1.5):
    """Let Tk and the canvases finish drawing before a window is grabbed."""
    end = time.time() + seconds
    while time.time() < end:
        app.update(); app.update_idletasks(); time.sleep(0.05)


def save(window, name, crop=None):
    settle()
    img = grab(window)
    if crop:
        img = img.crop((crop[0], crop[1], crop[2] or img.width, crop[3] or img.height))
    img.save(os.path.join(OUT, name))
    print("saved", name, img.size)


def button(text):
    def walk(w):
        for c in w.winfo_children():
            if type(c).__name__ == "CTkButton" and text in str(c.cget("text")):
                return c
            r = walk(c)
            if r:
                return r
    b = walk(app)
    if b is None:
        raise RuntimeError(f"no button {text!r}")
    return b


def click(text):
    n = len(LOG)
    button(text).invoke(); app.update()
    print("click", text, "->", LOG[n:])


def slide(slider, value):
    slider.set(value)
    if slider._command:
        slider._command(value)
    app.update()


def popups():
    return [w for w in app.winfo_children() if isinstance(w, ctk.CTkToplevel)]


def step(name, fn):
    if ONLY and name not in ONLY:
        return
    try:
        fn()
    except Exception:
        traceback.print_exc()


def s_load():
    app.notebook.set("Model Fitting"); app.update()
    click("Load Experimental Model")
    save(app, "model_fit_loaded.png", crop=(460, 0, None, None))


def s_sim():
    app.notebook.set("Simulation")
    slide(app.wavelength_slider, 365); slide(app.intensity_slider, 20); slide(app.n_pulses_slider, 30)
    click("Apply Potentiation")
    settle(4)
    save(app, "model_sim.png", crop=(0, 0, None, None))


def s_fit():
    app.notebook.set("Model Fitting"); app.update()
    seen = {}

    def photograph_dialog():
        for w in popups():
            if isinstance(w, Main.PeakOverrideDialog):
                seen["dialog"] = True
                save(w, "model_peaks.png")
                for c in w.winfo_children():
                    pass
                # press Apply with the automatic choice that the dialog proposes
                def walk(x):
                    for c in x.winfo_children():
                        if type(c).__name__ == "CTkButton" and c.cget("text") == "Apply":
                            return c
                        r = walk(c)
                        if r:
                            return r
                walk(w).invoke()
                return
        app.after(300, photograph_dialog)

    app.after(1200, photograph_dialog)
    click("Run Fitting")
    settle(4)
    wins = [w for w in popups() if not isinstance(w, Main.PeakOverrideDialog)]
    print("dialog photographed:", seen, "windows:", [w.title() for w in wins])
    for i, w in enumerate(wins):
        save(w, "model_fitplots.png" if i == 0 else f"_extra_{i}.png")
        w.destroy()
    app.update()


def s_net():
    app.notebook.set("Network"); app.update()
    app.target_var.set("network"); app.on_target_change(); app.update()
    for c in (app.network_rows_entry, app.network_cols_entry):
        c.delete(0, "end"); c.insert(0, "10")
    click("Initialize Network")
    app.pattern_var.set("Duck")
    app.mode_var.set("potentiation"); app.on_mode_change()
    slide(app.wavelength_slider, 365); slide(app.intensity_slider, 20)
    slide(app.n_pulses_slider, 20); slide(app.duration_slider, 100)
    print("apply button reads:", app.btn_stimulate.cget("text")); app.btn_stimulate.invoke(); app.update()
    settle(6)
    wins = popups()
    print("windows:", [w.title() for w in wins])
    app.notebook.set("Network"); app.update()
    save(app, "model_net.png", crop=(460, 0, None, None))
    for w in wins:
        save(w, "model_evolution.png")
        w.destroy()
    app.update()


def s_snn():
    app.notebook.set("SNN"); app.update()
    click("Run SNN Simulation")
    settle(6)
    save(app, "model_snn.png", crop=(460, 0, None, None))


step("load", s_load)
step("sim", s_sim)
step("fit", s_fit)
step("net", s_net)
step("snn", s_snn)
print(LOG)
app.destroy()
