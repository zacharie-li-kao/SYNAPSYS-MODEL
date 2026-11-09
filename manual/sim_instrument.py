"""Run the characterisation GUI against the simulated 2600 of verification/tsp_sim.py.

pyvisa.ResourceManager is replaced so that connecting returns a SimulatedKeithley. The
stock simulator's DUT integrates |level| and never relaxes, which would draw cycling as
two rising ramps and retention as a flat line. For the manual's figures it is therefore
given the behaviour of a real synaptic device: signed drive with soft bounds (potentiation
toward G_max, depression toward G_min) and exponential relaxation toward G_min. A read
channel held at 0 V while the LED is lit returns a short-circuit photocurrent proportional
to the device state, which is what the self-powered Visual mode measures. The TSP
execution, the virtual clock and every instrument rule are the simulator's own.
"""
import math
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "verification"))
import pyvisa
import tsp_sim

INSTRUMENTS = []
G_MIN, G_MAX = 1e-7, 1e-4
RATE = 0.25      # fractional approach to the bound per second per volt of drive
TAU = 12.0       # s, relaxation toward G_min
ISC_MAX = 5e-6   # A, short-circuit photocurrent at G_max and 2 V LED drive
NEXT_G0 = 2e-6   # conductance of the next device that is connected
WALL_RELAX = False  # retention waits on the host clock; let the device relax in that time


class _Device(tsp_sim.SimulatedKeithley):
    _last_wall = None

    def _wall_relax(self):
        now = time.perf_counter()
        if WALL_RELAX and self._last_wall is not None:
            self.G = G_MIN + (self.G - G_MIN) * math.exp(-(now - self._last_wall) / TAU)
        self._last_wall = now

    def _api(self):
        t = super()._api()
        orig = t['measure']
        sim = self

        def measure(ch, what, buf):
            value = orig(ch, what, buf)
            src = sim.state[f'{ch}.source']
            drive = sum(abs(sim.state[f'{c}.source']['levelv']) for c in sim.stim_channels
                        if sim.state[f'{c}.source']['output'] == 'OUTPUT_ON')
            if (what == 'i' and ch not in sim.stim_channels and src['output'] == 'OUTPUT_ON'
                    and src['levelv'] == 0 and drive > 0):
                value = ISC_MAX * (sim.G / G_MAX) * drive / 2.0
                if buf is not None:
                    sim.buffers[buf['readings']['__buffer']]['readings'][-1] = value
            return value
        t['measure'] = measure
        return t

    def write(self, cmd):
        self._wall_relax()
        return super().write(cmd)

    def query(self, cmd):
        self._wall_relax()
        return super().query(cmd)

    def _advance(self, dt):
        if dt < 0:
            raise tsp_sim.TSPError(f"negative time step {dt}")
        drive = 0.0
        for ch in self.stim_channels:
            src = self.state[f'{ch}.source']
            if src['output'] == 'OUTPUT_ON':
                drive += src['levelv'] if src['func'] == 'OUTPUT_DCVOLTS' else src['leveli'] * 1e3
        left = dt
        while left > 0:
            h = min(left, 0.005)
            if drive > 0:
                self.G += RATE * drive * h * (G_MAX - self.G)
            elif drive < 0:
                self.G += RATE * drive * h * (self.G - G_MIN)
            self.G = G_MIN + (self.G - G_MIN) * math.exp(-h / TAU)
            left -= h
        self.clock += dt


class _RM:
    def open_resource(self, name, **kw):
        sim = _Device(model="2636B", line_freq=50.0, stim_channels=("smua",),
                      G0=NEXT_G0, G_bounds=(G_MIN, G_MAX))
        INSTRUMENTS.append(sim)
        return sim

    def list_resources(self, *a, **k):
        return ()


def install():
    pyvisa.ResourceManager = lambda *a, **k: _RM()
