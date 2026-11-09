"""A simulated Keithley 2600-series TSP instrument that EXECUTES what it is sent.

Defaults follow the 2600B Reference Manual (2600BS-901-01 Rev. B): measure.delay
DELAY_AUTO with the per-range settling of RM 2600A Table 4-4 (model 263x
default), delay(0) rejected (RM 7-47), ASCII precision 6 (format.asciiprecision
settable 1..16), offmode NORMAL (OUTPUT_OFF holds 0 V; it does not open a relay).


Every command the host writes — single TSP statements, `loadandrunscript ...
endscript` bodies, `print(...)` / `printbuffer(...)` queries — is run in a real
Lua 5.1 interpreter (TSP is Lua 5.0) against strict mock `smua`/`smub` objects:
reading or writing an attribute or constant the real instrument does not have
raises, exactly where the instrument would put a syntax/run-time error in its
queue and never print the completion token.

Time is virtual. `delay()` advances it; so does every TSP command (a fixed
per-command overhead), every ADC integration (NPLC/linefreq × filter count ×
autozero factor) and every output toggle. The DUT is a conductance G that
integrates the drive applied on the designated stimulus channel(s):

    dG/dt = k_drive · |level|   while that channel's output is ON

and the read current is I = G · V(read channel) whenever the read channel's
output is on. Because G is the time integral of the stimulus actually applied,
the measured currents prove — or disprove — that each pulse happened, with the
width, level and order the host intended.

Every command and every state change is logged with its virtual time.
"""

import math
import re

try:
    import lupa.lua51 as lua51
except ImportError as e:  # pragma: no cover
    raise ImportError("the TSP simulator needs lupa (pip install lupa)") from e

VALID_SMU_CONST = {
    'OUTPUT_DCVOLTS', 'OUTPUT_DCAMPS', 'OUTPUT_ON', 'OUTPUT_OFF', 'OUTPUT_HIGH_Z',
    'OUTPUT_NORMAL', 'OUTPUT_ZERO', 'AUTOZERO_OFF', 'AUTOZERO_ONCE', 'AUTOZERO_AUTO',
    'AUTORANGE_OFF', 'AUTORANGE_ON', 'AUTORANGE_FOLLOW_LIMIT', 'FILTER_ON', 'FILTER_OFF',
    'FILTER_REPEAT_AVG', 'FILTER_MOVING_AVG', 'FILTER_MEDIAN', 'FILL_ONCE', 'FILL_WINDOW',
    'SENSE_LOCAL', 'SENSE_REMOTE', 'SENSE_CALA', 'DELAY_OFF', 'DELAY_AUTO',
    'ENABLE', 'DISABLE',
}
SOURCE_ATTRS = {'func', 'levelv', 'leveli', 'rangev', 'rangei', 'limiti', 'limitv',
                'output', 'autorangev', 'autorangei', 'offmode', 'offfunc', 'offlimiti',
                'offlimitv', 'delay', 'highc', 'settling', 'lowrangev', 'lowrangei',
                'compliance', 'limitp'}
MEASURE_ATTRS = {'nplc', 'rangei', 'rangev', 'autorangei', 'autorangev', 'autozero',
                 'count', 'delay', 'delayfactor', 'interval', 'lowrangei', 'lowrangev',
                 'rel', 'analogfilter'}
FILTER_ATTRS = {'count', 'type', 'enable'}
BUFFER_ATTRS = {'appendmode', 'collecttimestamps', 'collectsourcevalues', 'fillmode',
                'timestampresolution', 'cachemode'}

LUA_PRELUDE = r"""
local py = ...
local function strict(name, tbl, allowed)
  return setmetatable(tbl, {
    __index = function(t, k)
      error("TSP: unknown attribute " .. name .. "." .. tostring(k), 2)
    end,
    __newindex = function(t, k, v)
      if not allowed[k] then
        error("TSP: cannot set unknown attribute " .. name .. "." .. tostring(k), 2)
      end
      py.set(name, k, v)
    end,
  })
end
local function set_of(list) local s = {} for _, k in ipairs(list) do s[k] = true end return s end

-- Attributes live on the Python side; Lua tables are views onto them.
local function view(name, attrs)
  return setmetatable({}, {
    __index = function(t, k)
      if attrs[k] then return py.get(name, k) end
      error("TSP: unknown attribute " .. name .. "." .. tostring(k), 2)
    end,
    __newindex = function(t, k, v)
      if not attrs[k] then
        error("TSP: cannot set unknown attribute " .. name .. "." .. tostring(k), 2)
      end
      py.set(name, k, v)
    end,
  })
end

local function make_buffer(ch, bname)
  local full = ch .. "." .. bname
  local b = {}
  local attrs = py.buffer_attrs()
  local readings = {__buffer = full, __field = "readings"}
  local timestamps = {__buffer = full, __field = "timestamps"}
  local sourcevalues = {__buffer = full, __field = "sourcevalues"}
  b.clear = function() py.buffer_clear(full) end
  return setmetatable(b, {
    __index = function(t, k)
      if k == "readings" then return readings end
      if k == "timestamps" then return timestamps end
      if k == "sourcevalues" then return sourcevalues end
      if k == "n" then return py.buffer_n(full) end
      if k == "capacity" then return py.buffer_capacity(full) end
      if attrs[k] then return py.get(full, k) end
      error("TSP: unknown attribute " .. full .. "." .. tostring(k), 2)
    end,
    __newindex = function(t, k, v)
      if not attrs[k] then error("TSP: cannot set unknown attribute " .. full .. "." .. tostring(k), 2) end
      py.set(full, k, v)
    end,
  })
end

local function make_smu(ch)
  local consts = py.smu_consts()
  local s = {}
  local source = view(ch .. ".source", py.source_attrs())
  local measure_attrs = py.measure_attrs()
  local filter = view(ch .. ".measure.filter", py.filter_attrs())
  local measure = setmetatable({filter = filter}, {
    __index = function(t, k)
      if k == "i" then return function(buf) return py.measure(ch, "i", buf and buf ~= nil and buf or nil) end end
      if k == "v" then return function(buf) return py.measure(ch, "v", buf) end end
      if k == "iv" then return function(bi, bv) return py.measure_iv(ch, bi, bv) end end
      if measure_attrs[k] then return py.get(ch .. ".measure", k) end
      error("TSP: unknown attribute " .. ch .. ".measure." .. tostring(k), 2)
    end,
    __newindex = function(t, k, v)
      if not measure_attrs[k] then error("TSP: cannot set unknown attribute " .. ch .. ".measure." .. tostring(k), 2) end
      py.set(ch .. ".measure", k, v)
    end,
  })
  local nb1, nb2 = make_buffer(ch, "nvbuffer1"), make_buffer(ch, "nvbuffer2")
  local sense_ok = {sense = true}
  return setmetatable(s, {
    __index = function(t, k)
      if k == "source" then return source end
      if k == "measure" then return measure end
      if k == "nvbuffer1" then return nb1 end
      if k == "nvbuffer2" then return nb2 end
      if k == "reset" then return function() py.smu_reset(ch) end end
      if k == "makebuffer" then return function(n) error("TSP sim: makebuffer not modelled") end end
      if k == "sense" then return py.get(ch, "sense") end
      if consts[k] then return k end
      error("TSP: unknown attribute " .. ch .. "." .. tostring(k), 2)
    end,
    __newindex = function(t, k, v)
      if k == "sense" then py.set(ch, k, v) return end
      error("TSP: cannot set " .. ch .. "." .. tostring(k), 2)
    end,
  })
end

smua = make_smu("smua")
smub = make_smu("smub")
delay = function(t) py.delay(t) end
timer = {reset = function() py.timer_reset() end,
         measure = {t = function() return py.timer_t() end}}
errorqueue = setmetatable({clear = function() py.eq_clear() end,
                           next = function() return py.eq_next() end},
  {__index = function(t, k) if k == "count" then return py.eq_count() end
                            error("TSP: unknown attribute errorqueue." .. tostring(k), 2) end})
localnode = setmetatable({}, {
  __index = function(t, k) if k == "linefreq" then return py.get("localnode", "linefreq") end
                           error("TSP: unknown attribute localnode." .. tostring(k), 2) end,
  __newindex = function(t, k, v) if k ~= "linefreq" then error("TSP: localnode." .. k, 2) end
                                 py.set("localnode", k, v) end})
print = function(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[i] = py.fmt(select(i, ...)) end
  py.out(table.concat(parts, "\t"))
end
printbuffer = function(a, b, ...)
  local fields = {...}
  py.printbuffer(a, b, fields)
end
waitcomplete = function() end
format = setmetatable({}, {
  __index = function(t, k) if k == "asciiprecision" then return py.get("format", k) end
                           error("TSP: unknown attribute format." .. tostring(k), 2) end,
  __newindex = function(t, k, v) if k ~= "asciiprecision" then error("TSP sim: format." .. k .. " not modelled", 2) end
                                 py.set("format", k, v) end})
"""


class TSPError(RuntimeError):
    pass


class SimulatedKeithley:
    """pyvisa-like resource backed by a Lua TSP emulator with a virtual clock."""

    def __init__(self, model="2636B", line_freq=50.0, stim_channels=("smua",),
                 k_drive=2e-5, G0=1e-6, G_bounds=(1e-7, 1e-3),
                 t_cmd=50e-6, t_output_toggle=0.0, autozero_auto_factor=3.0,
                 measure_delay_s=0.0, ascii_precision=6, buffer_capacity=100000,
                 stamp_at='start', delay_zero_is_error=True, auto_delay_table=True):
        self.model = model
        self.timeout = 5000
        self.t_cmd = t_cmd
        self.t_output_toggle = t_output_toggle
        self.autozero_auto_factor = autozero_auto_factor
        self.measure_delay_s = measure_delay_s
        self.ascii_precision = ascii_precision
        self.buffer_capacity_n = buffer_capacity
        self.stim_channels = set(stim_channels)
        self.k_drive = k_drive
        self.G = G0
        self.G_bounds = G_bounds
        self.stamp_at = stamp_at
        self.delay_zero_is_error = delay_zero_is_error
        self.auto_delay_table = auto_delay_table

        self.clock = 0.0
        self.t0 = 0.0
        self.writes, self.queries, self.log = [], [], []
        self.out_queue = []
        self.errors = []
        self.script_lines = None
        self.state = {}
        for ch in ('smua', 'smub'):
            self._reset_smu(ch)
        self.state['localnode'] = {'linefreq': float(line_freq)}
        self.state['format'] = {'asciiprecision': ascii_precision}
        self.buffers = {f"{ch}.{b}": self._new_buffer() for ch in ('smua', 'smub')
                        for b in ('nvbuffer1', 'nvbuffer2')}
        self.measurements = []   # (t, ch, value, G_true, V_read)
        self.scripts_run = []

        self.lua = lua51.LuaRuntime(unpack_returned_tuples=True)
        api = self._api()
        self.lua.execute(LUA_PRELUDE, api) if False else self.lua.eval(
            "function(src, api) return assert(loadstring(src))(api) end")(LUA_PRELUDE, api)

    # ------------------------------------------------------------- state
    def _reset_smu(self, ch):
        self.state[ch] = {'sense': 'SENSE_LOCAL'}
        self.state[f'{ch}.source'] = {
            'func': 'OUTPUT_DCVOLTS', 'levelv': 0.0, 'leveli': 0.0, 'rangev': 0.2,
            'rangei': 1e-7, 'limiti': 1e-4, 'limitv': 20.0, 'output': 'OUTPUT_OFF',
            'autorangev': 'AUTORANGE_ON', 'autorangei': 'AUTORANGE_ON',
            'offmode': 'OUTPUT_NORMAL', 'offfunc': 'OUTPUT_DCVOLTS', 'offlimiti': 1e-3,
            'offlimitv': 40.0, 'delay': 'DELAY_OFF', 'highc': 'DISABLE', 'settling': 0,
            'lowrangev': 0, 'lowrangei': 0, 'compliance': False, 'limitp': 1e9}
        self.state[f'{ch}.measure'] = {
            'nplc': 1.0, 'rangei': 1e-7, 'rangev': 0.2, 'autorangei': 'AUTORANGE_ON',
            'autorangev': 'AUTORANGE_ON', 'autozero': 'AUTOZERO_AUTO', 'count': 1,
            'delay': 'DELAY_AUTO', 'delayfactor': 1.0, 'interval': 0, 'lowrangei': 0,
            'lowrangev': 0, 'rel': None, 'analogfilter': 1}
        self.state[f'{ch}.measure.filter'] = {'count': 1, 'type': 'FILTER_REPEAT_AVG',
                                              'enable': 'FILTER_OFF'}

    def _new_buffer(self):
        return {'readings': [], 'timestamps': [], 'sourcevalues': [],
                'attrs': {'appendmode': 0, 'collecttimestamps': 0, 'collectsourcevalues': 0,
                          'fillmode': 'FILL_ONCE', 'timestampresolution': 1e-6,
                          'cachemode': 1}}

    # ----------------------------------------------------- virtual clock
    def _advance(self, dt):
        if dt < 0:
            raise TSPError(f"negative time step {dt}")
        drive = 0.0
        for ch in self.stim_channels:
            src = self.state[f'{ch}.source']
            if src['output'] == 'OUTPUT_ON':
                level = src['levelv'] if src['func'] == 'OUTPUT_DCVOLTS' else src['leveli'] * 1e3
                drive += abs(level)
        if drive:
            self.G = min(max(self.G + self.k_drive * drive * dt, self.G_bounds[0]), self.G_bounds[1])
        self.clock += dt

    # ---------------------------------------------------------- Lua API
    def _api(self):
        sim = self

        def get(name, key):
            sim._advance(sim.t_cmd)
            if name.endswith('nvbuffer1') or name.endswith('nvbuffer2'):
                return sim.buffers[name]['attrs'][key]
            return sim.state[name][key]

        def set_(name, key, value):
            sim._advance(sim.t_cmd)
            if name.endswith('nvbuffer1') or name.endswith('nvbuffer2'):
                sim.buffers[name]['attrs'][key] = value
                sim.log.append((sim.clock, name, key, value))
                return
            if name == 'format':
                p = int(value)
                if not 1 <= p <= 16:
                    raise TSPError(f"format.asciiprecision = {value}: range is 1..16")
                sim.ascii_precision = p
            if name == 'localnode':
                if float(value) not in (50.0, 60.0):
                    raise TSPError(f"localnode.linefreq = {value}")
            st = sim.state[name]
            if key == 'output' and value != st.get('output'):
                sim._advance(sim.t_output_toggle)
            if isinstance(value, str) and key not in ('sense',) and value not in VALID_SMU_CONST:
                raise TSPError(f"{name}.{key} = {value!r}: not a TSP constant")
            st[key] = value
            sim.log.append((sim.clock, name, key, value))

        def measure(ch, what, buf):
            sim._advance(sim.t_cmd)
            m = sim.state[f'{ch}.measure']
            f = sim.state[f'{ch}.measure.filter']
            lf = sim.state['localnode']['linefreq']
            n_avg = int(f['count']) if f['enable'] == 'FILTER_ON' else 1
            n_avg *= int(m['count'])
            az = sim.autozero_auto_factor if m['autozero'] == 'AUTOZERO_AUTO' else 1.0
            if m['delay'] == 'DELAY_AUTO':
                if sim.auto_delay_table:
                    table = [(1e-9, 0.36), (1e-8, 0.09), (1e-7, 0.015), (1e-6, 2.5e-3),
                             (1e-5, 5e-4), (1e-4, 1.5e-4), (1e-3, 1e-4), (1e-1, 8e-5), (10, 1.2e-4)]
                    rng = float(m['rangei'])
                    d = next(v for r, v in table if rng <= r * 1.0001)
                    sim._advance(d * float(m['delayfactor']))
                    sim.log.append((sim.clock, ch, 'auto_measure_delay', d))
                else:
                    sim._advance(sim.measure_delay_s)
            elif isinstance(m['delay'], (int, float)):
                sim._advance(float(m['delay']))
            t_start = sim.clock - sim.t0
            sim._advance(float(m['nplc']) / lf * n_avg * az)
            t_end = sim.clock - sim.t0
            src = sim.state[f'{ch}.source']
            v = src['levelv'] if src['output'] == 'OUTPUT_ON' and src['func'] == 'OUTPUT_DCVOLTS' else 0.0
            value = sim.G * v if what == 'i' else v
            sim.measurements.append((t_start, ch, value, sim.G, v))
            if buf is not None:
                name = buf['readings']['__buffer']
                b = sim.buffers[name]
                if len(b['readings']) >= sim.buffer_capacity_n:
                    if b['attrs']['fillmode'] == 'FILL_ONCE':
                        sim.errors.append(f"5042 buffer {name} full")
                        return value
                b['readings'].append(value)
                b['timestamps'].append(t_start if sim.stamp_at == 'start' else t_end)
                b['sourcevalues'].append(v)
            return value

        def fmt(v):
            if isinstance(v, bool):
                return 'true' if v else 'false'
            if isinstance(v, (int, float)):
                return sim._num(v)
            if v is None:
                return 'nil'
            return str(v)

        def out(s):
            sim.out_queue.append(s)

        def printbuffer(a, b, fields):
            a, b = int(a), int(b)
            cols = []
            for f in fields.values():
                buf = sim.buffers[f['__buffer']]
                n = len(buf['readings'])
                if a < 1 or b > n or a > b:
                    raise TSPError(f"5038 printbuffer index {a}..{b} outside 1..{n}")
                cols.append(buf[f['__field']][a - 1:b])
            vals = []
            for i in range(b - a + 1):
                for c in cols:
                    vals.append(sim._num(c[i]))
            sim.out_queue.append(', '.join(vals))

        api = {
            'get': get, 'set': set_, 'measure': measure,
            'measure_iv': lambda ch, bi, bv: (_ for _ in ()).throw(TSPError("measure.iv not modelled")),
            'delay': lambda t: sim._delay(t),
            'timer_reset': lambda: sim._timer_reset(),
            'timer_t': lambda: sim._timer_t(),
            'eq_clear': lambda: sim.errors.clear(),
            'eq_count': lambda: len(sim.errors),
            'eq_next': lambda: sim.errors.pop(0) if sim.errors else '0 no error',
            'fmt': fmt, 'out': out, 'printbuffer': printbuffer,
            'buffer_clear': lambda name: sim._buffer_clear(name),
            'buffer_n': lambda name: len(sim.buffers[name]['readings']),
            'buffer_capacity': lambda name: sim.buffer_capacity_n,
            'smu_reset': lambda ch: sim._reset_smu(ch),
            'smu_consts': lambda: self.lua.table_from({k: True for k in VALID_SMU_CONST}),
            'source_attrs': lambda: self.lua.table_from({k: True for k in SOURCE_ATTRS}),
            'measure_attrs': lambda: self.lua.table_from({k: True for k in MEASURE_ATTRS}),
            'filter_attrs': lambda: self.lua.table_from({k: True for k in FILTER_ATTRS}),
            'buffer_attrs': lambda: self.lua.table_from({k: True for k in BUFFER_ATTRS}),
        }
        return self.lua.table_from(api)

    def _resolve_buffer(self, buf):
        # The buffer object passed to measure.i(): identify it via its readings field.
        r = buf['readings']
        return r['__buffer']

    def _num(self, v):
        v = float(v)
        if v == int(v) and abs(v) < 1e15 and self.ascii_precision is None:
            return str(int(v))
        return f"{v:.{self.ascii_precision - 1}e}"

    def _delay(self, t):
        t = float(t)
        if self.delay_zero_is_error and t <= 0:
            raise TSPError(f"delay({t}): the 2600 cannot delay zero seconds (RM 7-47)")
        if not (t >= 0) or math.isinf(t):
            raise TSPError(f"delay({t}) invalid")
        self._advance(t)
        self.log.append((self.clock, 'delay', 't', t))

    def _timer_reset(self):
        self._advance(self.t_cmd)
        self.t0 = self.clock

    def _timer_t(self):
        self._advance(self.t_cmd)
        return self.clock - self.t0

    def _buffer_clear(self, name):
        self._advance(self.t_cmd)
        b = self.buffers[name]
        b['readings'].clear(); b['timestamps'].clear(); b['sourcevalues'].clear()

    # ------------------------------------------------- pyvisa surface
    def _run(self, src, where):
        try:
            self.lua.execute(src)
        except Exception as e:  # lupa.LuaError and TSPError
            msg = f"{where}: {e}"
            self.errors.append(msg)
            self.log.append((self.clock, 'ERROR', where, str(e)))
            return False
        return True

    def write(self, cmd):
        self.writes.append(cmd)
        stripped = cmd.strip()
        if stripped == 'loadandrunscript':
            self.script_lines = []
            return
        if stripped == 'endscript':
            body = '\n'.join(self.script_lines or [])
            self.script_lines = None
            self.scripts_run.append(body)
            self._run(body, 'script')
            return
        if self.script_lines is not None:
            self.script_lines.append(cmd)
            return
        self._run(cmd, 'command')

    def read(self):
        if self.out_queue:
            return self.out_queue.pop(0)
        raise TimeoutError("VISA timeout (simulated): nothing to read")

    def query(self, cmd):
        self.queries.append(cmd)
        c = cmd.strip()
        if c == '*IDN?':
            return f"Keithley Instruments Inc., Model {self.model}, 4000000, 3.3.5"
        if c == '*OPC?':
            return '1'
        n_before = len(self.out_queue)
        ok = self._run(c, 'query')
        if not ok:
            raise TimeoutError(f"query {c!r} raised on the instrument: {self.errors[-1]}")
        if len(self.out_queue) == n_before:
            raise TimeoutError(f"query {c!r} printed nothing")
        return self.out_queue.pop(n_before)

    def close(self):
        pass

    # ------------------------------------------------- analysis helpers
    def events(self, name=None, key=None):
        return [(t, n, k, v) for (t, n, k, v) in self.log
                if (name is None or n == name) and (key is None or k == key)]
