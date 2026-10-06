"""
Synapse Cycle Module for Keithley Sourcemeter
Unified LUA-based potentiation-depression cycling engine.

Supports:
- Single-channel write+read (same SMU does write pulse then read pulse)
- Dual-channel stim/read (one SMU writes, other reads at constant bias)
- Two independent sequential trains (A then B) per cycle
- N-cycle repetition with inter-train and inter-cycle delays
- All execution on-instrument via LUA for sub-ms timing precision

Enhanced Features:
- Parameter presets for common protocols
- Cycle-specific read voltages
- Multi-device sequential testing
"""

import numpy as np
import time
from tkinter import messagebox
import synapse_engine


# =============================================================================
# CORE LUA-BASED CYCLING ENGINE
# =============================================================================

def cycle_sequence(instrument, train_a_config, train_b_config, n_cycles,
                   inter_train_delay_ms=500, inter_cycle_delay_ms=1000):
    """
    Runs a complete cycle measurement sequence on-instrument via LUA.

    Args:
        instrument: PyVISA instrument resource
        train_a_config (dict): Train A configuration containing:
            - topology (str): 'single' or 'dual'
            - write_ch (str): 'smua' or 'smub'
            - read_ch (str): 'smua' or 'smub' (== write_ch for single)
            - stim_drive_type (str): 'V' (or 'I' for dual only)
            - stim_level (float): Write pulse amplitude
            - read_voltage (float): Read bias voltage
            - stim_width_ms (float): Write pulse duration
            - stim_period_ms (float): Period between pulse starts
            - n_pulses (int): Number of write-read pairs
            - read_delay_ms (float): Delay after write before read
            - compliance_A (float): Current compliance
            - nplc (float): Integration time in PLCs
            - read_width_ms (float, optional): Measurement window for averaging
            - settle_ms (float): Settling time
            - wire_mode (str): '2-Wire' or '4-Wire'
        train_b_config (dict or None): Same structure, or None if no Train B
        n_cycles (int): Number of A-B cycle repetitions
        inter_train_delay_ms (float): Delay between trains
        inter_cycle_delay_ms (float): Delay between cycles

    Returns:
        dict: Results compatible with calculate_cycle_summary_metrics()
    """
    # --- Validate instrument ---
    if not synapse_engine.supports_lua_execution(instrument):
        raise ValueError(
            "Cycle mode requires a LUA-capable instrument (2600 series).\n"
            "Connected instrument does not support LUA execution."
        )

    # --- Validate voltage/current limits ---
    max_voltage = synapse_engine.get_max_voltage(instrument)
    max_current = synapse_engine.get_max_current(instrument)

    for config, label in [(train_a_config, 'Train A'), (train_b_config, 'Train B')]:
        if config is None:
            continue
        drive = config.get('stim_drive_type', 'V')
        if drive == 'V' and abs(config['stim_level']) > max_voltage:
            model = synapse_engine.validate_instrument_model(instrument)
            raise ValueError(
                f"{label}: Voltage {config['stim_level']}V exceeds {model} limit ({max_voltage}V)")
        if drive == 'I' and abs(config['stim_level']) > max_current:
            model = synapse_engine.validate_instrument_model(instrument)
            raise ValueError(
                f"{label}: Current {config['stim_level']}A exceeds {model} limit ({max_current}A)")

    # --- Configure wire mode via VISA (before LUA) ---
    # Per-SMU wire modes live in train_a_config['wire_modes'] (preferred) or
    # in the legacy 'wire_mode' string (applied uniformly). Each SMU's sense
    # mode must match its physical wiring, so configs can differ per channel.
    channels_used = {train_a_config['write_ch'], train_a_config['read_ch']}
    if train_b_config:
        channels_used.update({train_b_config['write_ch'], train_b_config['read_ch']})

    synapse_engine._apply_wire_modes(instrument, train_a_config, list(channels_used))

    # Record instrument voltage ceiling in each train config so the cycle LUA
    # generator picks native source.rangev values (40 V vs 200 V family differ).
    max_voltage = synapse_engine.get_max_voltage(instrument)
    train_a_config['max_voltage'] = max_voltage
    if train_b_config:
        train_b_config['max_voltage'] = max_voltage

    # Adopt the instrument's auto-detected mains frequency unless overridden.
    # The cycle LUA generator reads Train A's value; both configs are stamped
    # so exported per-train metadata carries the frequency actually used.
    line_freq = synapse_engine.resolve_line_freq(instrument, train_a_config)
    if train_b_config:
        train_b_config['line_freq_hz'] = line_freq

    # --- Generate LUA script ---
    lua_script, buffer_map = synapse_engine.generate_cycle_lua_script(
        train_a_config, train_b_config, n_cycles,
        inter_train_delay_ms, inter_cycle_delay_ms
    )

    # --- Compute expected execution time ---
    time_a_ms = train_a_config['stim_period_ms'] * train_a_config['n_pulses']
    time_b_ms = 0
    if train_b_config:
        time_b_ms = train_b_config['stim_period_ms'] * train_b_config['n_pulses']
    time_per_cycle_ms = time_a_ms + inter_train_delay_ms + time_b_ms + inter_cycle_delay_ms
    expected_time_s = (time_per_cycle_ms * n_cycles) / 1000.0

    print(f"Executing cycle sequence via LUA...")
    print(f"  Cycles: {n_cycles}")
    print(f"  Train A: {train_a_config['topology']}, {train_a_config['n_pulses']} pulses, "
          f"write={train_a_config['stim_level']}V on {train_a_config['write_ch']}")
    if train_b_config:
        print(f"  Train B: {train_b_config['topology']}, {train_b_config['n_pulses']} pulses, "
              f"write={train_b_config['stim_level']}V on {train_b_config['write_ch']}")
    print(f"  Expected time: {expected_time_s:.1f}s")

    # --- Execute ---
    raw_data = synapse_engine.execute_cycle_lua(
        instrument, lua_script, buffer_map, expected_time_s
    )

    # --- Parse into cycle results ---
    results = _parse_cycle_raw_data(
        raw_data, buffer_map, train_a_config, train_b_config, n_cycles,
        inter_train_delay_ms, inter_cycle_delay_ms
    )

    # --- Summary metrics ---
    results["summary_metrics"] = calculate_cycle_summary_metrics(results)

    # --- Print summary ---
    for cycle in results["cycles"]:
        cn = cycle["cycle_number"]
        pot = cycle["potentiation"]
        if pot and "error" not in pot:
            g = np.array(pot["conductance_S"])
            g_valid = g[~np.isnan(g)]
            if len(g_valid) >= 2:
                dg = (g_valid[-1] - g_valid[0]) / g_valid[0] * 100
                print(f"  Cycle {cn} Train A: ΔG = {dg:.1f}%")
        dep = cycle["depression"]
        if dep and "error" not in dep:
            g = np.array(dep["conductance_S"])
            g_valid = g[~np.isnan(g)]
            if len(g_valid) >= 2:
                dg = (g_valid[-1] - g_valid[0]) / g_valid[0] * 100
                print(f"  Cycle {cn} Train B: ΔG = {dg:.1f}%")

    print(f"[OK] Cycle sequence complete")
    return results


def _parse_cycle_raw_data(raw_data, buffer_map, train_a_config, train_b_config,
                          n_cycles, inter_train_delay_ms, inter_cycle_delay_ms):
    """
    Parses raw buffer data into the cycle results dict compatible with
    calculate_cycle_summary_metrics() and plot_cycle_data().
    """
    a_read = train_a_config['read_ch']
    b_read = train_b_config['read_ch'] if train_b_config else None
    same_buffer = (b_read is None) or (b_read == a_read)

    pts_a = train_a_config['n_pulses'] + 1  # initial read + pulses
    pts_b = (train_b_config['n_pulses'] + 1) if train_b_config else 0

    results = {
        "cycles": [],
        "pot_params": train_a_config.copy(),
        "dep_params": train_b_config.copy() if train_b_config else {},
        "n_cycles": n_cycles,
        "inter_cycle_delay_ms": inter_cycle_delay_ms,
        "inter_train_delay_ms": inter_train_delay_ms,
    }

    # --- Buffer-length validation (audit H9) ---
    #
    # De-interleaving slices by a FIXED STRIDE. Python slicing silently yields
    # a short (or empty) list past the end of a list, so a buffer with fewer
    # points than expected did not fail — it produced cycles parsed at a
    # progressively wrong offset, each one built from data belonging partly to
    # its neighbour. `_retrieve_buffer` already DETECTS a short buffer and
    # prints a warning, then returns it anyway, and the summary went on to
    # report `n_successful_cycles = 4` with nothing indicating that three of
    # them were misaligned. That is silent data corruption, which is the
    # failure mode the project ground rules exist to prevent.
    #
    # Only whole cycles present in the buffer are parsed. Any shortfall is
    # recorded in the results and printed, so a truncated run is visibly
    # truncated rather than quietly wrong.
    def _resolve_complete_cycles(available, per_cycle, label):
        if per_cycle <= 0:
            return n_cycles
        complete = available // per_cycle
        if complete < n_cycles:
            print(f"WARNING: {label} buffer holds {available} points, enough "
                  f"for {complete} of {n_cycles} cycles ({per_cycle} points "
                  f"per cycle). The remaining cycles are NOT reported: "
                  f"parsing them would slice across cycle boundaries and "
                  f"produce misaligned data.")
        return min(complete, n_cycles)

    if same_buffer:
        all_ts = raw_data['train_a']['timestamps']
        all_curr = raw_data['train_a']['currents']
        stride = pts_a + pts_b

        n_complete = _resolve_complete_cycles(len(all_ts), stride, "interleaved")

        for c in range(n_complete):
            offset = c * stride

            a_ts = all_ts[offset: offset + pts_a]
            a_curr = all_curr[offset: offset + pts_a]
            pot_result = _build_train_result(a_ts, a_curr, train_a_config)

            dep_result = None
            if train_b_config and pts_b > 0:
                b_offset = offset + pts_a
                b_ts = all_ts[b_offset: b_offset + pts_b]
                b_curr = all_curr[b_offset: b_offset + pts_b]
                dep_result = _build_train_result(b_ts, b_curr, train_b_config)

            results["cycles"].append({
                "cycle_number": c + 1,
                "potentiation": pot_result,
                "depression": dep_result
            })
    else:
        a_ts_all = raw_data['train_a']['timestamps']
        a_curr_all = raw_data['train_a']['currents']
        b_ts_all = raw_data['train_b']['timestamps']
        b_curr_all = raw_data['train_b']['currents']

        n_complete = _resolve_complete_cycles(len(a_ts_all), pts_a, "Train A")
        if train_b_config and pts_b > 0:
            n_complete = min(
                n_complete,
                _resolve_complete_cycles(len(b_ts_all), pts_b, "Train B"),
            )

        for c in range(n_complete):
            a_off = c * pts_a
            pot_result = _build_train_result(
                a_ts_all[a_off: a_off + pts_a],
                a_curr_all[a_off: a_off + pts_a],
                train_a_config
            )

            dep_result = None
            if train_b_config and pts_b > 0:
                b_off = c * pts_b
                dep_result = _build_train_result(
                    b_ts_all[b_off: b_off + pts_b],
                    b_curr_all[b_off: b_off + pts_b],
                    train_b_config
                )

            results["cycles"].append({
                "cycle_number": c + 1,
                "potentiation": pot_result,
                "depression": dep_result
            })

    # Provenance: the caller must be able to tell a truncated run from a
    # complete one without counting the cycles itself.
    results["n_cycles_requested"] = n_cycles
    results["n_cycles_parsed"] = len(results["cycles"])
    results["truncated"] = len(results["cycles"]) < n_cycles

    return results


def _build_train_result(timestamps, currents, config):
    """
    Converts raw timestamp/current arrays into the standard pulse_read result format.
    """
    read_voltage = config['read_voltage']
    t_start = timestamps[0] if timestamps else 0

    result = {
        "pulse_number": [],
        "time_s": [],
        "I_A": [],
        "V_read_V": [],
        "conductance_S": [],
        "params": config.copy()
    }

    for i, (ts, curr) in enumerate(zip(timestamps, currents)):
        result["pulse_number"].append(i)
        result["time_s"].append(ts - t_start)
        result["I_A"].append(curr)
        result["V_read_V"].append(read_voltage)
        if read_voltage != 0:
            result["conductance_S"].append(curr / read_voltage)
        else:
            result["conductance_S"].append(float('nan'))

    return result


def simulate_cycle_sequence(train_a_config, train_b_config, n_cycles,
                            update_callback=None):
    """
    Simulates cycle sequence without hardware.
    Uses exponential convergence model matching simulate_pulse_read().
    """
    results = {
        "cycles": [],
        "pot_params": train_a_config.copy(),
        "dep_params": train_b_config.copy() if train_b_config else {},
        "n_cycles": n_cycles,
        "inter_cycle_delay_ms": 0,
        "inter_train_delay_ms": 0,
    }

    G_min = 0.1e-6
    G_max = 100e-6
    G = 1e-6  # Initial conductance — persists across cycles

    for cycle_idx in range(n_cycles):
        cycle_data = {"cycle_number": cycle_idx + 1, "potentiation": None, "depression": None}

        # --- Train A ---
        G, pot_result = _simulate_train(train_a_config, G, G_min, G_max)
        cycle_data["potentiation"] = pot_result

        # --- Train B ---
        if train_b_config:
            G, dep_result = _simulate_train(train_b_config, G, G_min, G_max)
            cycle_data["depression"] = dep_result

        results["cycles"].append(cycle_data)

        if update_callback is not None:
            try:
                update_callback(results)
            except Exception:
                pass

        time.sleep(0.05)

    results["summary_metrics"] = calculate_cycle_summary_metrics(results)
    return results


def _simulate_train(config, G_start, G_min, G_max):
    """Simulates a single train and returns (G_final, result_dict)."""
    n = config['n_pulses']
    read_v = config['read_voltage']
    stim = config['stim_level']
    period_s = config['stim_period_ms'] / 1000.0

    result = {
        "pulse_number": [], "time_s": [], "I_A": [],
        "V_read_V": [], "conductance_S": [], "params": config.copy()
    }

    G = G_start
    # Initial read
    noise = 1.0 + np.random.normal(0, 0.015)
    I_meas = G * read_v * noise
    result["pulse_number"].append(0)
    result["time_s"].append(0.0)
    result["I_A"].append(I_meas)
    result["V_read_V"].append(read_v)
    result["conductance_S"].append(G * noise)

    for p in range(1, n + 1):
        if stim > 0:
            G += (G_max - G) * 0.05
        else:
            G += (G_min - G) * 0.05
        G = np.clip(G, G_min, G_max)

        noise = 1.0 + np.random.normal(0, 0.015)
        I_meas = G * read_v * noise
        result["pulse_number"].append(p)
        result["time_s"].append(p * period_s)
        result["I_A"].append(I_meas)
        result["V_read_V"].append(read_v)
        result["conductance_S"].append(G * noise)

    return G, result


# =============================================================================
# MULTI-DEVICE SEQUENTIAL TESTING
# =============================================================================

def run_multi_device_cycles(instrument, device_configs, update_callback=None):
    """
    Runs cycle sequences across multiple devices sequentially.

    Args:
        instrument: PyVISA instrument resource
        device_configs (list): List of device configuration dicts, each containing:
            - device_name (str): Identifier for the device
            - train_a_config (dict): Train A configuration
            - train_b_config (dict or None): Train B configuration
            - n_cycles (int): Number of cycles for this device
            - inter_train_delay_ms (float): Delay between trains
            - inter_cycle_delay_ms (float): Delay between cycles
        update_callback (callable, optional): Called after each device

    Returns:
        dict: Multi-device results
    """
    multi_results = {
        "devices": [],
        "device_names": [],
        "n_devices": len(device_configs)
    }

    for device_idx, config in enumerate(device_configs):
        device_name = config.get("device_name", f"Device_{device_idx + 1}")

        print(f"\n{'#'*60}")
        print(f"# Testing Device: {device_name} ({device_idx + 1}/{len(device_configs)})")
        print(f"{'#'*60}")

        try:
            device_results = cycle_sequence(
                instrument,
                config["train_a_config"],
                config.get("train_b_config"),
                config["n_cycles"],
                config.get("inter_train_delay_ms", 500),
                config.get("inter_cycle_delay_ms", 1000),
            )

            device_results["device_name"] = device_name
            multi_results["devices"].append(device_results)
            multi_results["device_names"].append(device_name)

            print(f"\n[OK] Device {device_name} completed successfully")

            if update_callback is not None:
                try:
                    update_callback(multi_results)
                except Exception as e:
                    print(f"WARNING: Update callback failed: {str(e)}")

            if device_idx < len(device_configs) - 1:
                inter_device_delay = config.get("inter_device_delay_ms", 2000)
                print(f"\nWaiting {inter_device_delay} ms before next device...")
                time.sleep(inter_device_delay / 1000.0)

        except Exception as e:
            print(f"[FAIL] Device {device_name} failed: {str(e)}")
            error_result = {
                "device_name": device_name,
                "error": str(e),
                "cycles": [],
                "n_cycles": 0
            }
            multi_results["devices"].append(error_result)
            multi_results["device_names"].append(device_name)

    return multi_results


def simulate_multi_device_cycles(device_configs, update_callback=None):
    """Simulates multi-device cycling without hardware."""
    multi_results = {
        "devices": [],
        "device_names": [],
        "n_devices": len(device_configs)
    }

    for device_idx, config in enumerate(device_configs):
        device_name = config.get("device_name", f"Device_{device_idx + 1}")

        device_results = simulate_cycle_sequence(
            config["train_a_config"],
            config.get("train_b_config"),
            config["n_cycles"],
            update_callback=None
        )

        device_results["device_name"] = device_name
        multi_results["devices"].append(device_results)
        multi_results["device_names"].append(device_name)

        if update_callback is not None:
            try:
                update_callback(multi_results)
            except Exception as e:
                print(f"⚠️  Update callback failed: {str(e)}")

    return multi_results


# =============================================================================
# METRICS AND ANALYSIS
# =============================================================================

def calculate_cycle_summary_metrics(cycle_results):
    """
    Calculates aggregate metrics across all cycles.
    
    Args:
        cycle_results (dict): Results from run_potentiation_depression_cycle
    
    Returns:
        dict: Summary metrics including:
            - pot_delta_g_mean (%): Mean potentiation ΔG across cycles
            - pot_delta_g_std (%): Std dev of potentiation ΔG
            - dep_delta_g_mean (%): Mean depression ΔG across cycles
            - dep_delta_g_std (%): Std dev of depression ΔG
            - pot_reproducibility (%): Coefficient of variation for potentiation
            - dep_reproducibility (%): Coefficient of variation for depression
            - dynamic_range (S): Average difference between max pot and min dep conductance
    """
    metrics = {}
    
    pot_delta_g_list = []
    dep_delta_g_list = []
    pot_g_final_list = []
    dep_g_final_list = []
    
    for cycle in cycle_results["cycles"]:
        # Potentiation metrics
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            g_pot = np.array(pot_data["conductance_S"])
            g_pot_valid = g_pot[~np.isnan(g_pot)]
            
            if len(g_pot_valid) >= 2:
                delta_g_pot = (g_pot_valid[-1] - g_pot_valid[0]) / g_pot_valid[0] * 100
                pot_delta_g_list.append(delta_g_pot)
                pot_g_final_list.append(g_pot_valid[-1])
        
        # Depression metrics
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            g_dep = np.array(dep_data["conductance_S"])
            g_dep_valid = g_dep[~np.isnan(g_dep)]
            
            if len(g_dep_valid) >= 2:
                delta_g_dep = (g_dep_valid[-1] - g_dep_valid[0]) / g_dep_valid[0] * 100
                dep_delta_g_list.append(delta_g_dep)
                dep_g_final_list.append(g_dep_valid[-1])
    
    # Potentiation statistics
    if len(pot_delta_g_list) == 0:
        metrics["pot_delta_g_mean (%)"] = "N/A"
        metrics["pot_delta_g_std (%)"] = "N/A"
        metrics["pot_reproducibility_CV (%)"] = "N/A"
    else:
        metrics["pot_delta_g_mean (%)"] = round(np.mean(pot_delta_g_list), 2)
        metrics["pot_delta_g_std (%)"] = round(np.std(pot_delta_g_list), 2)
        
        # Reproducibility (coefficient of variation)
        if np.mean(pot_delta_g_list) != 0:
            cv_pot = (np.std(pot_delta_g_list) / abs(np.mean(pot_delta_g_list))) * 100
            metrics["pot_reproducibility_CV (%)"] = round(cv_pot, 2)
    
    # Depression statistics
    if len(dep_delta_g_list) == 0:
        metrics["dep_delta_g_mean (%)"] = "N/A"
        metrics["dep_delta_g_std (%)"] = "N/A"
        metrics["dep_reproducibility_CV (%)"] = "N/A"
    else:
        metrics["dep_delta_g_mean (%)"] = round(np.mean(dep_delta_g_list), 2)
        metrics["dep_delta_g_std (%)"] = round(np.std(dep_delta_g_list), 2)
        
        if np.mean(dep_delta_g_list) != 0:
            cv_dep = (np.std(dep_delta_g_list) / abs(np.mean(dep_delta_g_list))) * 100
            metrics["dep_reproducibility_CV (%)"] = round(cv_dep, 2)
    
    # Dynamic range
    if len(pot_g_final_list) > 0 and len(dep_g_final_list) > 0:
        avg_pot_g = np.mean(pot_g_final_list)
        avg_dep_g = np.mean(dep_g_final_list)
        dynamic_range = avg_pot_g - avg_dep_g
        metrics["dynamic_range (S)"] = f"{dynamic_range:.4e}"
        
        # On/Off ratio
        if avg_dep_g != 0:
            on_off_ratio = avg_pot_g / avg_dep_g
            metrics["on_off_ratio"] = round(on_off_ratio, 2)
    
    # Cycle-to-cycle variation
    if len(pot_delta_g_list) > 1:
        metrics["n_successful_cycles"] = len(pot_delta_g_list)
    
    return metrics


# =============================================================================
# DATA SAVING
# =============================================================================

def save_cycle_data(cycle_results, filename):
    """
    Saves potentiation-depression cycle data to CSV with metadata.
    
    Format:
        - Metadata header with parameters
        - Summary metrics
        - Per-cycle data with separate columns for pot/dep
    
    Args:
        cycle_results (dict): Results from run_potentiation_depression_cycle
        filename (str): Path to save CSV file
    
    Returns:
        None
    """
    csv_content = []
    
    # === METADATA SECTION ===
    csv_content.append("# Potentiation-Depression Cycle Characterization")
    csv_content.append(f"# n_cycles={cycle_results['n_cycles']}")
    csv_content.append(f"# inter_cycle_delay_ms={cycle_results.get('inter_cycle_delay_ms', 0)}")
    
    # Potentiation parameters
    pot_params = cycle_results["pot_params"]
    pot_param_str = " # ".join(f"pot_{k}={v}" for k, v in pot_params.items())
    csv_content.append(f"# {pot_param_str}")
    
    # Depression parameters
    dep_params = cycle_results["dep_params"]
    dep_param_str = " # ".join(f"dep_{k}={v}" for k, v in dep_params.items())
    csv_content.append(f"# {dep_param_str}")
    
    # Summary metrics
    summary = cycle_results.get("summary_metrics", {})
    if summary:
        summary_str = " # ".join(f"{k}={v}" for k, v in summary.items())
        csv_content.append(f"# SUMMARY: {summary_str}")
    
    csv_content.append("#")
    
    # === DATA SECTION ===
    # Column headers
    headers = [
        "cycle",
        "phase",  # 'potentiation' or 'depression'
        "pulse",
        "timestamp_s",
        "I_A",
        "V_read_V",
        "conductance_S"
    ]
    csv_content.append(",".join(headers))
    
    # Data rows
    for cycle in cycle_results["cycles"]:
        cycle_num = cycle["cycle_number"]
        
        # Potentiation data
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            for i in range(len(pot_data["pulse_number"])):
                row = [
                    str(cycle_num),
                    "potentiation",
                    str(pot_data["pulse_number"][i]),
                    f"{pot_data['time_s'][i]:.6f}",
                    f"{pot_data['I_A'][i]:.6e}",
                    f"{pot_data['V_read_V'][i]:.4f}",
                    f"{pot_data['conductance_S'][i]:.6e}"
                ]
                csv_content.append(",".join(row))
        
        # Depression data
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            for i in range(len(dep_data["pulse_number"])):
                row = [
                    str(cycle_num),
                    "depression",
                    str(dep_data["pulse_number"][i]),
                    f"{dep_data['time_s'][i]:.6f}",
                    f"{dep_data['I_A'][i]:.6e}",
                    f"{dep_data['V_read_V'][i]:.4f}",
                    f"{dep_data['conductance_S'][i]:.6e}"
                ]
                csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))

    print(f"Cycle data saved to {filename}")


def save_multi_device_data(multi_results, filename):
    """
    Saves multi-device cycle data to CSV.
    
    Args:
        multi_results (dict): Results from run_multi_device_cycles
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata
    csv_content.append("# Multi-Device Potentiation-Depression Cycle Characterization")
    csv_content.append(f"# n_devices={multi_results['n_devices']}")
    csv_content.append(f"# devices={','.join(multi_results['device_names'])}")
    csv_content.append("#")
    
    # Column headers
    headers = [
        "device",
        "cycle",
        "phase",
        "pulse",
        "timestamp_s",
        "I_A",
        "V_read_V",
        "conductance_S"
    ]
    csv_content.append(",".join(headers))
    
    # Data rows for each device
    for device_result in multi_results["devices"]:
        if "error" in device_result:
            continue
            
        device_name = device_result["device_name"]
        
        for cycle in device_result["cycles"]:
            cycle_num = cycle["cycle_number"]
            
            # Potentiation
            if cycle["potentiation"] and "error" not in cycle["potentiation"]:
                pot_data = cycle["potentiation"]
                for i in range(len(pot_data["pulse_number"])):
                    row = [
                        device_name,
                        str(cycle_num),
                        "potentiation",
                        str(pot_data["pulse_number"][i]),
                        f"{pot_data['time_s'][i]:.6f}",
                        f"{pot_data['I_A'][i]:.6e}",
                        f"{pot_data['V_read_V'][i]:.4f}",
                        f"{pot_data['conductance_S'][i]:.6e}"
                    ]
                    csv_content.append(",".join(row))
            
            # Depression
            if cycle["depression"] and "error" not in cycle["depression"]:
                dep_data = cycle["depression"]
                for i in range(len(dep_data["pulse_number"])):
                    row = [
                        device_name,
                        str(cycle_num),
                        "depression",
                        str(dep_data["pulse_number"][i]),
                        f"{dep_data['time_s'][i]:.6f}",
                        f"{dep_data['I_A'][i]:.6e}",
                        f"{dep_data['V_read_V'][i]:.4f}",
                        f"{dep_data['conductance_S'][i]:.6e}"
                    ]
                    csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))

    print(f"Multi-device data saved to {filename}")


# =============================================================================
# PARAMETER PRESETS
# =============================================================================

def get_preset_parameters(preset_name):
    """
    Returns preset parameter sets for common cycling protocols.
    
    Args:
        preset_name (str): One of:
            - "Standard Cycle": Balanced moderate protocol
            - "Fast Cycle": Quick cycling for screening
            - "High Endurance": Long-term stability testing
            - "Asymmetric": Different pot/dep parameters
    
    Returns:
        tuple: (pot_params, dep_params, n_cycles, inter_cycle_delay_ms)
    """
    presets = {
        "Standard Cycle": {
            "train_a": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": 1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "train_b": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": -1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "n_cycles": 5,
            "inter_train_delay_ms": 500,
            "inter_cycle_delay_ms": 1000,
        },

        "Fast Cycle": {
            "train_a": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": 1.2,
                "stim_width_ms": 50,
                "stim_period_ms": 100,
                "n_pulses": 20,
                "read_voltage": 0.1,
                "read_delay_ms": 5,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 3,
            },
            "train_b": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": -1.2,
                "stim_width_ms": 50,
                "stim_period_ms": 100,
                "n_pulses": 20,
                "read_voltage": 0.1,
                "read_delay_ms": 5,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 3,
            },
            "n_cycles": 10,
            "inter_train_delay_ms": 300,
            "inter_cycle_delay_ms": 500,
        },

        "High Endurance": {
            "train_a": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": 0.8,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 100,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "train_b": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": -0.8,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 100,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "n_cycles": 20,
            "inter_train_delay_ms": 1000,
            "inter_cycle_delay_ms": 2000,
        },

        "Asymmetric": {
            "train_a": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": 1.5,
                "stim_width_ms": 50,
                "stim_period_ms": 150,
                "n_pulses": 30,
                "read_voltage": 0.15,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "train_b": {
                "topology": "single",
                "write_ch": "smua",
                "read_ch": "smua",
                "stim_drive_type": "V",
                "stim_level": -1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.08,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "nplc": 1.0,
                "settle_ms": 5,
            },
            "n_cycles": 5,
            "inter_train_delay_ms": 500,
            "inter_cycle_delay_ms": 1500,
        },
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = presets[preset_name]
    return (
        preset["train_a"],
        preset["train_b"],
        preset["n_cycles"],
        preset["inter_train_delay_ms"],
        preset["inter_cycle_delay_ms"],
    )


def get_default_train_a_config():
    """Returns default Train A (potentiation) configuration."""
    return {
        "topology": "single",
        "write_ch": "smua",
        "read_ch": "smua",
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 50,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "nplc": 1.0,
        "settle_ms": 5,
    }


def get_default_train_b_config():
    """Returns default Train B (depression) configuration."""
    return {
        "topology": "single",
        "write_ch": "smua",
        "read_ch": "smua",
        "stim_drive_type": "V",
        "stim_level": -1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 50,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "nplc": 1.0,
        "settle_ms": 5,
    }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test the module with simulation"""
    print("Testing synapse_cycle module...")

    # Test preset loading
    print("\n=== Testing Presets ===")
    for preset_name in ["Standard Cycle", "Fast Cycle", "High Endurance", "Asymmetric"]:
        train_a, train_b, n_cyc, itd, icd = get_preset_parameters(preset_name)
        print(f"{preset_name}: {n_cyc} cycles, "
              f"A={train_a['stim_level']}V ({train_a['topology']}), "
              f"B={train_b['stim_level']}V ({train_b['topology']})")

    # Test single-device cycling (both trains)
    print("\n=== Testing Cycle Simulation (A+B) ===")
    train_a, train_b, n_cycles, itd, icd = get_preset_parameters("Fast Cycle")

    def test_callback(results):
        print(f"  Update: Completed {len(results['cycles'])}/{results['n_cycles']} cycles")

    results = simulate_cycle_sequence(train_a, train_b, n_cycles=3,
                                       update_callback=test_callback)

    print("\nSummary Metrics:")
    for key, value in results["summary_metrics"].items():
        print(f"  {key}: {value}")

    # Test single-train only (no Train B)
    print("\n=== Testing Single-Train Simulation ===")
    results_single = simulate_cycle_sequence(train_a, None, n_cycles=2,
                                              update_callback=test_callback)
    print(f"  Cycles: {len(results_single['cycles'])}, "
          f"Train B present: {results_single['cycles'][0]['depression'] is not None}")

    # Test multi-device
    print("\n=== Testing Multi-Device ===")
    device_configs = [
        {
            "device_name": "Device_A",
            "train_a_config": train_a,
            "train_b_config": train_b,
            "n_cycles": 2,
        },
        {
            "device_name": "Device_B",
            "train_a_config": train_a,
            "train_b_config": None,
            "n_cycles": 2,
        },
    ]

    def multi_callback(results):
        print(f"  Completed {len(results['devices'])}/{results['n_devices']} devices")

    multi_results = simulate_multi_device_cycles(device_configs, update_callback=multi_callback)
    print(f"\n[OK] Tested {multi_results['n_devices']} devices successfully")

    print("\n[OK] All tests complete!")