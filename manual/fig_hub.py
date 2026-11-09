import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from manual.grab import grab
import hub
app = hub.SynapseSuiteHub()
print(grab(app, os.path.join(ROOT, "manual", "figures", "hub.png")).size)
app.destroy()
