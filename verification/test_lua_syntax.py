"""Static syntax lint of every generated LUA script.

Motivated by a field failure (2026-08-18): the buffer-configuration line of the
pulse-read and visual generators was corrupted to `if X = if Y`, which is not
Lua. The 2636B rejected the whole anonymous script with a front-panel syntax
error and no synapse measurement could run, while the verification suite stayed
green because the fake instrument does not parse Lua. These checks close that
gap: they cannot prove the scripts are semantically right, but they reject the
whole class of mangled-line and unformatted-placeholder defects before release.
"""

import re

import synapse_engine as se


# ---------------------------------------------------------------------------
# Script corpus: one representative generation per generator.
# ---------------------------------------------------------------------------

def _pulse_read_params(**overrides):
    params = {
        'mode': 'electrical',
        'stim_drive_type': 'V',
        'stim_level': 0.3,
        'stim_width_ms': 100.0,
        'stim_period_ms': 200.0,
        'n_pulses': 10,
        'read_voltage': 1.5,
        'read_delay_ms': 10.0,
        'read_settle_delay_ms': 5.0,
        'read_width_ms': 50.0,
        'compliance_A': 1e-3,
        'nplc': 0.1,
        'measure_avg': 3,
        'settle_ms': 5,
        'max_voltage': 200.0,
    }
    params.update(overrides)
    return params


def _visual_params(**overrides):
    params = {
        'light_pulse_voltage': 3.0,
        'pulse_width_ms': 100.0,
        'pulse_period_ms': 200.0,
        'n_pulses': 10,
        'compliance_A': 1e-2,
        'nplc': 0.1,
        'measure_start_delay_ms': 5.0,
        'measure_end_margin_ms': 5.0,
        'readings_per_pulse': 3,
        'max_voltage': 200.0,
    }
    params.update(overrides)
    return params


def _cycle_train(**overrides):
    cfg = {
        'topology': 'single',
        'write_ch': 'smua',
        'read_ch': 'smua',
        'stim_drive_type': 'V',
        'stim_level': 1.0,
        'stim_width_ms': 1.0,
        'stim_period_ms': 10.0,
        'n_pulses': 5,
        'read_voltage': 0.1,
        'read_delay_ms': 2.0,
        'compliance_A': 1e-3,
        'nplc': 0.1,
        'settle_ms': 1.0,
        'measure_avg': 1,
    }
    cfg.update(overrides)
    return cfg


def _all_scripts():
    """Yield (name, lua_source) for every generator, in both drive types
    and both topologies where the generator distinguishes them."""
    yield ('pulse_read/V',
           se.generate_pulse_read_lua_script('smua', 'smub', _pulse_read_params()))
    yield ('pulse_read/I',
           se.generate_pulse_read_lua_script(
               'smua', 'smub',
               _pulse_read_params(stim_drive_type='I', stim_level=1e-3)))
    yield ('visual_standard',
           se.generate_visual_synapse_lua_script('smua', 'smub', _visual_params()))
    yield ('visual_continuous',
           se.generate_visual_continuous_lua_script('smua', 'smub', _visual_params()))
    yield ('stdp',
           se.generate_stdp_lua_script(
               'smua', 'smub', _pulse_read_params(stim_width_ms=5.0), 10.0))
    script, _ = se.generate_cycle_lua_script(
        _cycle_train(), _cycle_train(), n_cycles=2,
        inter_train_delay_ms=10.0, inter_cycle_delay_ms=10.0)
    yield ('cycle/single-single', script)
    script, _ = se.generate_cycle_lua_script(
        _cycle_train(),
        _cycle_train(topology='dual', write_ch='smua', read_ch='smub'),
        n_cycles=2, inter_train_delay_ms=10.0, inter_cycle_delay_ms=10.0)
    yield ('cycle/single-dual', script)


def _code_lines(lua_source):
    """Non-empty lines with comments stripped."""
    for raw in lua_source.splitlines():
        line = raw.split('--', 1)[0].rstrip()
        if line.strip():
            yield line


# ---------------------------------------------------------------------------
# Lint rules
# ---------------------------------------------------------------------------

def test_no_line_contains_the_mangled_assignment_pattern():
    """The exact field failure: `X = if Y` is never valid Lua."""
    for name, script in _all_scripts():
        for line in _code_lines(script):
            assert '= if ' not in line, (
                f"{name}: mangled assignment {line.strip()!r}"
            )


def test_every_if_has_a_then_and_every_loop_header_a_do():
    for name, script in _all_scripts():
        for line in _code_lines(script):
            stripped = line.strip()
            if re.match(r'(if|elseif)\b', stripped):
                assert ' then' in stripped, (
                    f"{name}: 'if' without 'then': {stripped!r}"
                )
            if re.match(r'(for|while)\b', stripped):
                assert re.search(r'\bdo\b', stripped), (
                    f"{name}: loop header without 'do': {stripped!r}"
                )


def test_no_unformatted_python_placeholders_survive_generation():
    """A `{read_ch}`-style token in the output means an f-string was edited
    into a plain string (or a placeholder typo'd) and the instrument would
    receive the literal braces."""
    placeholder = re.compile(r'\{[A-Za-z_][A-Za-z0-9_]*\}')
    for name, script in _all_scripts():
        for line in _code_lines(script):
            m = placeholder.search(line)
            assert m is None, (
                f"{name}: unformatted placeholder {m.group(0) if m else ''!r} "
                f"in {line.strip()!r}"
            )


def test_block_openers_and_ends_balance():
    """Every `then`, loop `do` and `function` opens a block that one `end`
    closes. The counts must match over a whole well-formed script."""
    for name, script in _all_scripts():
        openers = 0
        ends = 0
        for line in _code_lines(script):
            openers += len(re.findall(r'\bthen\b', line))
            openers += len(re.findall(r'\bfunction\b', line))
            # A `do` opens a block when it terminates a for/while header or
            # stands alone; `do` never appears mid-expression in these scripts.
            openers += len(re.findall(r'\bdo\b', line))
            ends += len(re.findall(r'\bend\b', line))
            # `else`/`elseif` reuse the `if` block's single `end`; `then` on an
            # `elseif` line would double-count, so compensate.
            openers -= len(re.findall(r'\belseif\b', line))
        assert openers == ends, (
            f"{name}: {openers} block openers vs {ends} 'end' keywords — "
            "the script cannot be well-formed Lua"
        )
