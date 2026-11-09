"""Figures for the data converter chapter (needs the generated test data folder: argv[1])."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from manual.grab import grab, client_bbox
import matplotlib; matplotlib.use("TkAgg")
from PIL import ImageDraw, ImageFont
import data_converter as dc

DATA = os.path.abspath(sys.argv[1])
OUT = os.environ.get("FIG_OUT", os.path.join(ROOT, "manual", "figures"))
LOG = []
dc.messagebox.showinfo = lambda t, m, **k: LOG.append(("info", t, m))
dc.messagebox.showerror = lambda t, m, **k: LOG.append(("error", t, m))
dc.messagebox.showwarning = lambda t, m, **k: LOG.append(("warning", t, m))
dc.messagebox.askyesno = lambda *a, **k: (_ for _ in ()).throw(AssertionError(a))

app = dc.SynapseDataPrepApp()
app.geometry(os.environ.get("GEOM", "1900x1150")); app.update()


def find(w, cls, text):
    r = []
    for c in w.winfo_children():
        if type(c).__name__ == cls and text in str(c.cget("text")):
            r.append(c)
        r += find(c, cls, text)
    return r


def marker(img, n, widget):
    x0, y0, _, _ = client_bbox(app)
    x = widget.winfo_rootx() - x0 + widget.winfo_width() - 30
    y = widget.winfo_rooty() - y0 + widget.winfo_height() // 2
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 26)
    d.ellipse((x - 20, y - 20, x + 20, y + 20), fill=(200, 30, 30), outline="white", width=3)
    d.text((x, y - 1), str(n), fill="white", font=f, anchor="mm")


# 1. the metadata dialog for one file, filled in as for the 365 nm potentiation run
dlg = dc.MetadataDialog(app, os.path.join(DATA, "potentiation_365nm.csv"))
for combo, val in ((dlg.x_type, "pulse_number"), (dlg.y_type, "conductance"), (dlg.exp_type, "potentiation")):
    combo.set(val)
dlg.on_exp_type_change("potentiation")
for entry, val in ((dlg.x_units, "#"), (dlg.y_units, "S"), (dlg.wavelength_pot, "365"),
                   (dlg.pot_intensity, "50"), (dlg.pulse_width, "100"), (dlg.frequency, "10 Hz"),
                   (dlg.n_pulses, "80"), (dlg.read_timing, "10")):
    entry.delete(0, "end"); entry.insert(0, val)
dlg.notes.insert("1.0", "Single-wavelength potentiation at 365 nm (UV)")
dlg.update(); dlg.update()
grab(dlg, os.path.join(OUT, "conv_metadata.png"))
dlg.destroy(); app.update()

# 2. batch load from the metadata file
dc.filedialog.askopenfilename = lambda **k: os.path.join(DATA, "METADATA_INSTRUCTIONS.txt")
dc.filedialog.askdirectory = lambda **k: DATA
app.batch_load_with_metadata(); app.update()
print(LOG)
app.dataset_index_entry.delete(0, "end"); app.dataset_index_entry.insert(0, "3")
app.update_preview(); app.update()
img = grab(app)
marker(img, 1, find(app, "CTkButton", "Batch Load")[0])
marker(img, 2, find(app, "CTkButton", "Save JSON")[0])
img.crop((0, 0, 430, img.height)).save(os.path.join(OUT, "conv_loaded.png"))

# 3. save the suite
dc.filedialog.asksaveasfilename = lambda **k: os.path.join(os.environ.get("SUITE_OUT", OUT), "synapse_suite.json")
n = len(LOG)
app.save_json()
print(LOG[n:])
app.destroy()
