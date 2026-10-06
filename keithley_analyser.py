__version__ = "5.22"

# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import numpy as np
import pyvisa
import time
from tkinter.filedialog import asksaveasfilename
import synapse_engine
import pulse_schematic
import synapse_cycle
import sys
import os
from PIL import Image

from assembly_helpers import load_characterization_suite
import json
from fitting import extract_synapse_model, plot_fitting_results
import socket

def is_port_open(ip, port, timeout=2):
    """Check if a TCP port is open on the given IP address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((ip, port))
        return result == 0
    except:
        return False
    finally:
        sock.close()

def open_lan_instrument(rm, ip_address):
    """
    Attempts to connect to a Keithley instrument via LAN using multiple VISA resource formats.
    Tries different connection methods for compatibility with various Keithley models
    (2600B series: 2602B, 2604B, 2612B, 2614B, 2634B, 2636B, etc.).

    Args:
        rm: pyvisa ResourceManager instance
        ip_address: IP address of the instrument

    Returns:
        instrument: Connected VISA resource object

    Raises:
        ValueError: If no connection method succeeds
    """
    # Build connection methods list, checking port availability first to avoid timeouts
    connection_methods = [
        # VXI-11 formats (always try these - no port check needed)
        (f"TCPIP::{ip_address}::INSTR", {}),
        (f"TCPIP0::{ip_address}::inst0::INSTR", {}),
    ]

    # Only add raw socket methods if the port is actually open
    if is_port_open(ip_address, 5025):
        connection_methods.append(
            (f"TCPIP0::{ip_address}::5025::SOCKET", {'read_termination': '\n', 'write_termination': '\n'})
        )
    if is_port_open(ip_address, 1225):
        connection_methods.append(
            (f"TCPIP0::{ip_address}::1225::SOCKET", {'read_termination': '\n', 'write_termination': '\n'})
        )

    last_error = None
    for resource_string, settings in connection_methods:
        try:
            instrument = rm.open_resource(resource_string)
            instrument.timeout = 10000  # 10 second timeout for connection test
            for attr, value in settings.items():
                setattr(instrument, attr, value)
            # Test the connection with a simple query
            instrument.query("*IDN?")
            return instrument
        except Exception as e:
            last_error = e
            try:
                instrument.close()
            except:
                pass
            continue

    raise ValueError(f"Could not connect via LAN. Tried multiple methods. Last error: {last_error}")

# appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Arrays to store JV curves and PV parameters
jv_curves = []  # To store [(sample_name, voltages, current_density)]
pv_parameters = []  # To store [(sample_name, parameters_dict)]
synapse_data_storage = []  # To store synapse measurement results

def _interpolate_crossing(x_pairs, y_pairs):
    """Linear interpolation to y = 0 between two bracketing points."""
    (x0, x1), (y0, y1) = x_pairs, y_pairs
    if y1 == y0:
        return None
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def calculate_pv_parameters(voltages, currents, area, light_power):
    """Calculate photovoltaic parameters from JV curve data.

    H17: the previous implementation took the LAST index with J < 0 and the
    FIRST index with J > 0 and interpolated between them. Those two points
    bracket the crossing only on a monotonically ascending sweep; on a REVERSE
    sweep they are at opposite ends of the data, and the formula then
    extrapolates far outside the measured range. FF and PCE inherit the error,
    so the result is a set of publishable-looking numbers that are wrong. The
    fallback was an unflagged silent substitution, FF was unguarded against
    voc*jsc == 0, and Jsc was the current at the nearest measured point rather
    than interpolated to 0 V.

    The crossing is now found between genuinely ADJACENT samples, which makes
    the result independent of sweep direction, and every quantity that could
    not be determined is reported as None with a reason rather than as a
    plausible number.
    """
    voltages = np.asarray(voltages, dtype=float)
    current_density = np.asarray(currents, dtype=float) * 1e3 / area  # mA/cm²

    warnings = []

    # --- Voc: interpolate at the first adjacent J sign change ---------------
    voc = None
    sign_changes = np.flatnonzero(
        np.sign(current_density[:-1]) * np.sign(current_density[1:]) < 0
    )
    if len(sign_changes):
        i = int(sign_changes[0])
        voc = _interpolate_crossing(
            (voltages[i], voltages[i + 1]),
            (current_density[i], current_density[i + 1]),
        )
        if len(sign_changes) > 1:
            warnings.append(
                f"J crosses zero {len(sign_changes)} times; Voc taken at the "
                "first crossing"
            )
    else:
        warnings.append(
            "J never changes sign over the measured range, so Voc is outside "
            "it and cannot be determined"
        )

    # --- Jsc: interpolate J at V = 0 ----------------------------------------
    jsc = None
    v_crossings = np.flatnonzero(np.sign(voltages[:-1]) * np.sign(voltages[1:]) < 0)
    exact_zero = np.flatnonzero(voltages == 0.0)
    if len(exact_zero):
        jsc = -current_density[int(exact_zero[0])]
    elif len(v_crossings):
        i = int(v_crossings[0])
        j_at_zero = _interpolate_crossing(
            (current_density[i], current_density[i + 1]),
            (voltages[i], voltages[i + 1]),
        )
        jsc = None if j_at_zero is None else -j_at_zero
    else:
        warnings.append(
            "the sweep does not span V = 0, so Jsc cannot be determined"
        )

    # --- Maximum power point -------------------------------------------------
    power_density = voltages * current_density
    mpp_index = int(np.argmin(power_density))
    vmp = voltages[mpp_index]
    jmp = current_density[mpp_index]
    p_max = abs(power_density[mpp_index])

    # --- Fill factor ---------------------------------------------------------
    ff = None
    if voc is not None and jsc is not None:
        denominator = voc * jsc
        if denominator != 0:
            ff = abs(vmp * jmp / denominator) * 100
        else:
            warnings.append("Voc x Jsc is zero, so FF is undefined")
    else:
        warnings.append("FF requires both Voc and Jsc")

    # --- Efficiency ----------------------------------------------------------
    pce = None
    light_power_mw_cm2 = light_power / 10
    if light_power_mw_cm2 > 0:
        pce = p_max / light_power_mw_cm2 * 100
    else:
        warnings.append(
            "incident light power is zero, so PCE is undefined"
        )

    for message in warnings:
        print(f"WARNING: PV extraction — {message}")

    def _round(value, places):
        return None if value is None else round(float(value), places)

    return {
        "Voc": _round(voc, 3),
        "Jsc": _round(jsc, 3),
        "FF": _round(ff, 2),
        "PCE": _round(pce, 2),
        "Vmp": _round(vmp, 3),
        "Jmp": _round(jmp, 3),
        "warnings": warnings,
    }

def update_parameters_display(params):

    def _fmt(key, unit):
        # None means the quantity could not be determined from the sweep. It is
        # shown as such rather than as a plausible number — see
        # calculate_pv_parameters (H17).
        value = params.get(key)
        return f"{key}: not determinable\n" if value is None else f"{key}: {value} {unit}\n"

    params_text.insert("end", f"Cell # {params['Cell #']}:\n")
    params_text.insert("end", _fmt('Voc', 'V'))
    params_text.insert("end", _fmt('Jsc', 'mA/cm²'))
    params_text.insert("end", _fmt('FF', '%'))
    params_text.insert("end", _fmt('PCE', '%'))
    for message in params.get('warnings', []):
        params_text.insert("end", f"  ! {message}\n")
    params_text.insert("end", "-"*20 + "\n")

def validate_float(entry_value, field_name, minimum=None, maximum=None,
                   units="", hint=""):
    """Parse a numeric entry, optionally enforcing a physical range.

    The message echoes the literal text the user typed (`!r`, so '1O' is
    visibly not '10'), states the accepted range, and names the units — a bare
    "Please enter a valid number" told the user nothing they did not know.

    Range checking matters beyond tidiness: this function is the single funnel
    for ~40 numeric entries, and it used to validate SYNTAX ONLY. A negative
    SRDP start frequency therefore reached `np.logspace(np.log10(f), ...)`,
    which returns NaN — and every comparison against NaN is False, so the
    downstream `freq_end > max_achievable` ceiling guard silently did not fire
    and the sweep ran on NaN frequencies.
    """
    unit_str = f" {units}" if units else ""
    try:
        value = float(entry_value)
    except (ValueError, TypeError):
        raise ValueError(
            f"{field_name}: {entry_value!r} is not a number."
            + (f" Enter a value in{unit_str}." if units else " Enter a numeric value.")
            + (f" {hint}" if hint else "")
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name}: {entry_value!r} is not a finite number "
            f"(got {value}). NaN and infinity cannot be used as instrument "
            f"settings — every safety comparison against NaN silently "
            f"evaluates False."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{field_name}: {value:g}{unit_str} is below the minimum of "
            f"{minimum:g}{unit_str}." + (f" {hint}" if hint else "")
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{field_name}: {value:g}{unit_str} is above the maximum of "
            f"{maximum:g}{unit_str}." + (f" {hint}" if hint else "")
        )
    return value


def get_compliance_value():
    """
    Parses the general compliance dropdown value and returns it in Amperes.
    
    Returns:
        float: Compliance value in Amperes
    
    Raises:
        ValueError: If compliance value cannot be parsed
    """
    compliance_str = compliance.get()
    try:
        # Parse values like "100nA", "1µA", "10mA", "1A"
        if compliance_str.endswith("nA"):
            return float(compliance_str[:-2]) * 1e-9
        elif compliance_str.endswith("µA"):
            return float(compliance_str[:-2]) * 1e-6
        elif compliance_str.endswith("mA"):
            return float(compliance_str[:-2]) * 1e-3
        elif compliance_str.endswith("A") and len(compliance_str) > 1:
            # Simple "A" suffix (e.g., "1A")
            return float(compliance_str[:-1])
        else:
            raise ValueError("Invalid compliance format")

    except (ValueError, KeyError, IndexError):
        raise ValueError(f"Invalid compliance value: {compliance_str}")

def run_measurement_buffered(transistor_mode=False):
    global voltages, current_density, jv_curves
    try:
        # Validate and parse GUI inputs
        sample_name = sample_name_entry.get().strip()
        if not sample_name:
            raise ValueError("Cell # cannot be empty.")

        area = validate_float(surface_area.get(), "Sample Surface Area")
        if area <= 0:
            raise ValueError("Surface area must be greater than zero.")

        start = validate_float(start_voltage.get(), "Starting Voltage")
        end = validate_float(end_voltage.get(), "Ending Voltage")
        step = validate_float(voltage_step.get(), "Voltage Step")
        nplc = validate_float(nplc_entry.get(), "Measurement Speed (NPLC)", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz.")

        compliance_value = get_compliance_value()

        channel = channel_selection.get()
        connection = connection_type.get()
        port = port_entry.get().strip()
        if not port:
            raise ValueError("Port Address cannot be empty.")

        hysteresis_cycles_value = 1
        if hysteresis_var.get():
            try:
                hysteresis_cycles_value = int(hysteresis_cycles.get())
                if hysteresis_cycles_value <= 0:
                    raise ValueError("Hysteresis Cycles must be greater than zero.")
            except ValueError:
                raise ValueError("Invalid value for Hysteresis Cycles.")

        if not dark_measurement.get():
            light_power = validate_float(light_power_entry.get(), "Irradiance")
            if light_power <= 0:
                raise ValueError("Irradiance must be greater than zero.")

        # Initialize communication with Keithley 2636A
        instrument = None
        if connection == "GPIB":
            instrument = synapse_engine.open_instrument(port, "GPIB")
        elif connection == "RS232":
            instrument = synapse_engine.open_instrument(port, "RS232")
        elif connection == "LAN":
            rm = pyvisa.ResourceManager()
            instrument = open_lan_instrument(rm, port)

        if instrument is None:
            raise ValueError("Could not establish communication with the instrument.")

        # === VOLTAGE SAFETY CHECK FOR 2600B SERIES ===
        # Low voltage models (2601B/02B/04B): 40V limit
        # High voltage models (2611B/12B/14B/3xB): 200V limit
        try:
            max_voltage_instr = synapse_engine.get_max_voltage(instrument)
            requested_voltage = max(abs(start), abs(end))

            if requested_voltage > max_voltage_instr:
                model = instrument.query("*IDN?")
                instrument.close()
                messagebox.showerror(
                    "Voltage Limit Exceeded",
                    f"ERROR: Instrument voltage limit exceeded.\n\n"
                    f"Instrument: {model.strip()}\n"
                    f"Maximum voltage: {max_voltage_instr}V\n"
                    f"Requested voltage: {requested_voltage}V\n\n"
                    f"Please reduce voltage range to ≤{max_voltage_instr}V."
                )
                return
        except synapse_engine.InstrumentIdentificationError as e:
            # The model determines the 40 V / 200 V ceiling, and this is the
            # ONLY ceiling check on the JV path — pulse_read_sequence's
            # re-check does not run here. Swallowing this into a console print
            # left the sweep to proceed with no voltage limit at all, which is
            # worse than the "conservative default" the ground rules forbid.
            messagebox.showerror(
                "Cannot Verify Voltage Limits — Measurement Refused",
                f"The instrument could not be identified, so its voltage "
                f"ceiling is unknown.\n\n"
                f"Requested: {requested_voltage} V on {channel_cmd}.\n\n"
                f"{e}\n\n"
                "Nothing was applied. Check the connection and address, or add "
                "this model to synapse_engine.MODEL_CAPABILITIES with its "
                "specifications."
            )
            return
        except Exception as e:
            messagebox.showerror(
                "Voltage Limit Check Failed — Measurement Refused",
                f"An unexpected error occurred while verifying the instrument's "
                f"voltage limits, so the sweep was not started.\n\n"
                f"Requested: {requested_voltage} V on {channel_cmd}.\n"
                f"{type(e).__name__}: {e}\n\n"
                "Nothing was applied."
            )
            return

        # Get user-specified timeout
        try:
            user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
            if user_timeout <= 0:
                raise ValueError("Timeout duration must be greater than zero.")
            timeout_value = user_timeout * 1000  # Convert seconds to milliseconds
        except ValueError as timeout_error:
            # The user typed something; substituting a computed value
            # silently overrides their intent. Refuse instead, and
            # suggest the computed value in the message.
            raise ValueError(
                f"Timeout Duration: {timeout_error}\n"
                f"Enter the timeout in seconds (must be greater than zero). "
                f"Nothing was measured."
            )
        
        instrument.timeout = timeout_value

        # Instrument initialization
        if not transistor_mode:
            instrument.write("*RST")
        instrument.write("*CLS")
        channel_cmd = "smua" if channel == "Channel A" else "smub"

        # Configure source and measurement
        instrument.write(f"{channel_cmd}.source.func = {channel_cmd}.OUTPUT_DCVOLTS")
        instrument.write(f"{channel_cmd}.source.autorangev = {channel_cmd}.AUTORANGE_ON")
        if autorange_var.get():
            instrument.write(f"{channel_cmd}.measure.autorangei = {channel_cmd}.AUTORANGE_ON")
        else:
            instrument.write(f"{channel_cmd}.measure.autorangei = {channel_cmd}.AUTORANGE_OFF")

        # Update timeout if autorange is enabled (may need more time)
        if autorange_var.get():
            try:
                user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
                timeout_value = max(user_timeout * 1000, 10000)  # At least 10 seconds for autorange
            except ValueError:
                timeout_value = max(10000, nplc * (1/60) * 5 * 1000)
            instrument.timeout = timeout_value

        instrument.write(f"{channel_cmd}.source.limiti = {compliance_value}")
        instrument.write(f"{channel_cmd}.measure.nplc = {nplc}")
        instrument.write(f"{channel_cmd}.nvbuffer1.clear()")
        instrument.write(f"{channel_cmd}.nvbuffer1.appendmode = 1")
        instrument.write(f"{channel_cmd}.nvbuffer1.collectsourcevalues = 1")
        
        if get_wire_mode(channel_cmd) == "4-Wire":
            instrument.write(f"{channel_cmd}.sense = {channel_cmd}.SENSE_REMOTE")
        else:
            instrument.write(f"{channel_cmd}.sense = {channel_cmd}.SENSE_LOCAL")

        # Define voltage sweep (use linspace to avoid floating-point precision errors)
        num_points = int(round((end - start) / step)) + 1
        voltages = np.linspace(start, end, num_points)
        
        if hysteresis_var.get():
            reverse_voltages = voltages[::-1]
            full_cycle_voltages = np.concatenate([voltages, reverse_voltages])
        else:
            full_cycle_voltages = voltages

        # H6: size the buffer to the sweep BEFORE measuring.
        #
        # nvbuffer1 was cleared but its capacity never set, and the
        # 2600-series default is 100 readings. An ordinary -0.2 V -> 1.0 V
        # sweep at 0.01 V step is 121 points, so the run overflowed with
        # instrument error 5038 and the oldest 21 readings had already been
        # overwritten by the time anything was retrieved. With hysteresis it
        # is 242 points.
        total_readings = len(full_cycle_voltages) * hysteresis_cycles_value

        # Ask the INSTRUMENT what its buffer holds, rather than assuming.
        #
        # `nvbuffer1.capacity` is READ-ONLY on the 2600 series (Reference
        # Manual: "Attribute (R)" — "this read-only attribute reads the number
        # of readings that can be stored"), so the previous
        # `capacity = total_readings` write could never enlarge anything. And
        # the real capacity is not a fixed 50000: the manual states a dedicated
        # buffer holds "over 140,000 readings" for basic items but that
        # "turning on additional collection items, such as timestamps and
        # source values, decreases the capacity" — and collectsourcevalues is
        # enabled just above. A hardcoded ceiling is therefore wrong in both
        # directions.
        #
        # This matters because overflow is SILENT: with the default
        # fillmode = FILL_ONCE, "if the buffer fills up, new readings will be
        # discarded" and no error is raised.
        # Plausibility bound on the value the instrument reports. The manual
        # states a dedicated buffer stores "over 140,000 readings" when
        # collecting only basic items, which is the ceiling any 2600-series
        # channel can credibly return; a larger number means the query did not
        # return what we think it did, and trusting it would re-introduce the
        # silent-overflow bug from the other side.
        BUFFER_CAPACITY_MAX = 140000
        instrument.write(f"{channel_cmd}.nvbuffer1.fillmode = {channel_cmd}.FILL_ONCE")
        try:
            buffer_capacity = int(float(
                instrument.query(f"print({channel_cmd}.nvbuffer1.capacity)").strip()
            ))
        except Exception as e:
            raise RuntimeError(
                f"Could not read the instrument's buffer capacity: {e}\n"
                "The sweep was not started. Buffer overflow discards readings "
                "silently on this instrument, so the capacity must be known "
                "before committing to a sweep length."
            )
        if not (0 < buffer_capacity <= BUFFER_CAPACITY_MAX):
            raise RuntimeError(
                f"{channel_cmd}.nvbuffer1 reported an implausible capacity of "
                f"{buffer_capacity} (expected 1-{BUFFER_CAPACITY_MAX}).\n"
                "The sweep was not started rather than proceeding on a buffer "
                "size that cannot be trusted."
            )
        if total_readings > buffer_capacity:
            raise ValueError(
                f"This sweep requests {total_readings} readings, but "
                f"{channel_cmd}.nvbuffer1 holds only {buffer_capacity} with the "
                "collection options currently enabled (timestamps and source "
                "values each reduce it).\n"
                "The buffer would overflow and silently discard the excess. "
                "Increase the voltage step, narrow the range, or reduce the "
                "number of hysteresis cycles."
            )

        # Synchronise the ADC to mains, as every synapse path does. Without
        # this a JV-only session integrates against whatever line frequency the
        # node happens to hold, so NPLC does not reject mains pickup (H16).
        instrument.write(
            f"localnode.linefreq = {int(synapse_engine.DEFAULT_LINE_FREQ_HZ)}"
        )

        # Perform measurement
        all_currents = []
        measurement_delay = nplc * (1/60)
        for cycle in range(hysteresis_cycles_value):
            for voltage in full_cycle_voltages:
                instrument.write(f"{channel_cmd}.source.levelv = {voltage}")
                instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_ON")
                time.sleep(max(0.01, measurement_delay))
                instrument.write(f"{channel_cmd}.measure.i({channel_cmd}.nvbuffer1)")
                time.sleep(max(0.01, measurement_delay/2))
            instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_OFF")

        # Retrieve data
        currents = instrument.query_ascii_values(f"printbuffer(1, {len(full_cycle_voltages) * hysteresis_cycles_value}, {channel_cmd}.nvbuffer1.readings)")
        voltages = instrument.query_ascii_values(f"printbuffer(1, {len(full_cycle_voltages) * hysteresis_cycles_value}, {channel_cmd}.nvbuffer1.sourcevalues)")

        # Process and plot data
        current_density = np.array(currents) * 1e3 / area
        ax.clear()
        
        if hasattr(ax, 'ax2') and ax.ax2 is not None:
            ax.ax2.remove()
            ax.ax2 = None
        plt.cla()
        
        ax.plot(voltages, current_density, label="JV Curve", color="blue")
        ax.set_title("JV Curve")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm²)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        
        if not dark_measurement.get() and not hysteresis_var.get():
            power_density = np.array(voltages) * current_density
            pv_params = calculate_pv_parameters(
                voltages, 
                currents, 
                area,
                float(light_power_entry.get())
            )
            pv_params["Cell #"] = sample_name
            pv_parameters.append(pv_params)
            update_parameters_display(pv_params)
        
            ax2 = ax.twinx()
            ax2.plot(voltages, power_density, label="Power Density", color="red", linestyle="--")
            ax2.set_ylabel("Power Density (mW/cm²)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper right")
            ax.ax2 = ax2
        else:
            if hasattr(ax, 'ax2') and ax.ax2:
                ax.ax2.remove()
                ax.ax2 = None
        
        ax.legend(loc="upper left")
        canvas.draw()
        add_semilog_jv_curve(graph_frame, voltages, current_density)

        jv_entry = {"Cell #": sample_name, "Voltages": voltages, "Current Density": current_density, "Surface Area": area, "Measurement Type": "Dark" if dark_measurement.get() else "Illuminated"}
        jv_curves.append(jv_entry)

        if not transistor_mode:
            register_diode_run(
                jv_entry,
                pv_entry=pv_parameters[-1] if (not dark_measurement.get() and not hysteresis_var.get() and pv_parameters) else None
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        # H20: `channel_cmd` is first assigned partway through the try block,
        # so any exception raised before that point made this `finally` itself
        # raise NameError — replacing the real error with a misleading one and
        # leaving the output ON. Both names are checked, and the shutdown is
        # attempted on a best-effort basis: failing to turn the output off must
        # not mask the fault that got us here, but it must still be reported,
        # because a live output is a safety matter.
        if 'instrument' in locals() and instrument is not None:
            try:
                if 'channel_cmd' in locals() and channel_cmd:
                    instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_OFF")
            except Exception as shutdown_error:
                print(f"WARNING: could not turn the source output off: "
                      f"{shutdown_error}")
            try:
                instrument.close()
            except Exception as close_error:
                print(f"WARNING: could not close the instrument session: "
                      f"{close_error}")

def save_curve():
    try:
        jv_file = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save JV Curves")
        if jv_file:
            header_sample_names = []
            header_labels = []
            data_rows = []
            
            max_length = max(len(entry["Voltages"]) for entry in jv_curves)
            
            for entry in jv_curves:
                sample_name = entry["Cell #"]
                if entry["Measurement Type"] == "Dark":
                    sample_name += " -- Dark"
                    
                if entry["Measurement Type"] == "Dark":
                    header_sample_names.extend([f"{sample_name}", f"{entry['Surface Area']}"])
                else:
                    irradiance_value = light_power_entry.get().strip()
                    header_sample_names.extend([f"{sample_name} ({irradiance_value} W/m²)", f"{entry['Surface Area']}"])

                header_labels.extend(["Voltage (V)", "Current Density (mA/cm²)"])
                
                voltages = entry["Voltages"]
                current_density = entry["Current Density"]
                for i in range(max_length):
                    if len(data_rows) <= i:
                        data_rows.append([])
                    data_rows[i].extend([
                        voltages[i] if i < len(voltages) else "",
                        current_density[i] if i < len(current_density) else ""
                    ])
            
            with open(jv_file, "w", encoding="utf-8") as f:
                f.write(",".join(header_sample_names) + "\n")
                f.write(",".join(header_labels) + "\n")
                for row in data_rows:
                    f.write(",".join(map(str, row)) + "\n")

        pv_file = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save PV Parameters")
        if pv_file:
            with open(pv_file, "w", encoding="utf-8") as f:
                irradiance_value = light_power_entry.get().strip()
                f.write("Cell # (Irradiance W/m²),Voc (V),Jsc (mA/cm²),FF (%),PCE (%)\n")
                for entry in pv_parameters:
                    f.write(f"{entry['Cell #']} ({irradiance_value} W/m²),{entry['Voc']},{entry['Jsc']},{entry['FF']},{entry['PCE']}\n")

        messagebox.showinfo("Save Successful", "Data saved successfully.")
        params_text.delete("1.0", "end")

    except Exception as e:
        messagebox.showerror("Save Error", str(e))

def add_semilog_jv_curve(graph_frame, voltages, currents):

    if hasattr(graph_frame, "semilog_frame") and graph_frame.semilog_frame is not None:
        if hasattr(graph_frame, "_semilog_fig"):
            plt.close(graph_frame._semilog_fig)
        graph_frame.semilog_frame.destroy()

    graph_frame.semilog_frame = ctk.CTkFrame(graph_frame)
    graph_frame.semilog_frame.pack(fill="both", expand=True)

    fig_semi = plt.Figure(figsize=(5, 2))
    graph_frame._semilog_fig = fig_semi  # Store reference for cleanup
    ax_semi = fig_semi.add_subplot(111)

    abs_currents = np.abs(currents)
    ax_semi.semilogy(voltages, abs_currents, label="JV Curve (Semilog)", color="green")

    ax_semi.set_title("JV Curve (Semilog)")
    ax_semi.set_xlabel("Voltage (V)")
    ax_semi.set_ylabel("Current Density (mA/cm²)")
    ax_semi.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_semi.legend()

    canvas_semi = FigureCanvasTkAgg(fig_semi, master=graph_frame.semilog_frame)
    canvas_semi_widget = canvas_semi.get_tk_widget()
    canvas_semi_widget.pack(fill="both", expand=True)

    return graph_frame.semilog_frame

def plot_synapse_data(results):
    """
    Plots synapse measurement results (Conductance vs Pulse Number).
    
    Args:
        results (dict): Results from synapse_engine.pulse_read_sequence
    """
    global ax, canvas
    
    # Clear existing plots
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    conductance = np.array(results["conductance_S"])
    pulse_numbers = results["pulse_number"]
    currents = np.array(results["I_A"])
    
    # Main plot: Conductance vs Pulse Number
    color = 'tab:blue'
    ax.plot(pulse_numbers, conductance * 1e6, 'o-', color=color, linewidth=2, markersize=4)
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title(f"{synapse_mode_display(results['params']['mode'])} Synapse Response", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary y-axis: Current
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.plot(pulse_numbers, currents * 1e6, 's--', color=color, linewidth=1.5, markersize=3, alpha=0.7)
    ax2.set_ylabel("Current (µA)", color=color, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color)
    ax.ax2 = ax2
    
    # Add legend
    ax.legend(['Conductance'], loc='upper left')
    ax2.legend(['Current'], loc='upper right')
    
    canvas.draw()
    
    # Calculate and display metrics
    metrics = synapse_engine.calculate_synapse_metrics(results)
    
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"Synapse Measurement Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Mode: {results['params']['mode']}\n")
    params_text.insert("end", f"Pulses: {results['params']['n_pulses']}\n")
    params_text.insert("end", f"Stim Level: {results['params']['stim_level']} {results['params']['stim_drive_type']}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for key, value in metrics.items():
        params_text.insert("end", f"{key}: {value}\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    
    # Scroll to bottom
    params_text.see("end")



def plot_srdp_data(results):
    """
    Plots SRDP characterization results (ΔG vs Frequency).
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    frequencies = results["frequencies_hz"]
    delta_g_percent = results["delta_g_percent"]
    
    # Main plot: ΔG% vs Frequency
    color = 'tab:blue'
    ax.plot(frequencies, delta_g_percent, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Spike Frequency (Hz)", fontsize=11)
    ax.set_ylabel("ΔG (%)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title("Spike-Rate-Dependent Plasticity (SRDP)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    canvas.draw()
    
    # Display results in text box
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"SRDP Characterization Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Stim Level: {results['base_params']['stim_level']} V\n")
    params_text.insert("end", f"Pulses per freq: {results['base_params']['n_pulses']}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for i, freq in enumerate(frequencies):
        params_text.insert("end", 
            f"{freq:.1f} Hz → ΔG = {delta_g_percent[i]:.2f}%\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    params_text.see("end")


def plot_stdp_data(results):
    """
    Plots STDP characterization results (ΔG vs Δt).
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    delta_t = results["delta_t_ms"]
    delta_g_percent = results["delta_g_percent"]
    
    # Main plot: ΔG% vs Δt
    color = 'tab:green'
    ax.plot(delta_t, delta_g_percent, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Δt (ms) [Pre - Post]", fontsize=11)
    ax.set_ylabel("ΔG (%)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title("Spike-Timing-Dependent Plasticity (STDP)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Annotations
    ax.text(0.05, 0.95, 'LTP (Δt > 0)', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', color='blue')
    ax.text(0.05, 0.05, 'LTD (Δt < 0)', transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', color='red')
    
    canvas.draw()
    
    # Display results in text box
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"STDP Characterization Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Stim Level: {results['base_params']['stim_level']} V\n")
    params_text.insert("end", f"Spike pairs: {results['base_params'].get('n_pulses', 50)}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for i, dt in enumerate(delta_t):
        params_text.insert("end", 
            f"Δt = {dt:+.1f} ms → ΔG = {delta_g_percent[i]:+.2f}%\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    params_text.see("end")


def plot_cycle_data(results):
    """
    Plots potentiation-depression cycle results with real-time updates.
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    n_completed = len(results["cycles"])
    colors_pot = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_completed, 1)))
    colors_dep = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_completed, 1)))
    
    max_pulse_count = 0
    
    for i, cycle in enumerate(results["cycles"]):
        cycle_num = cycle["cycle_number"]
        pulse_offset = 0
        
        # Plot potentiation
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            g_pot = np.array(pot_data["conductance_S"]) * 1e6  # Convert to µS
            pulse_num = pot_data["pulse_number"]
            ax.plot(pulse_num, g_pot, 'o-', color=colors_pot[i], 
                   linewidth=2, markersize=4, label=f'Cycle {cycle_num} Pot', alpha=0.8)
            pulse_offset = len(pulse_num)
            max_pulse_count = max(max_pulse_count, max(pulse_num))
        
        # Plot depression
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            g_dep = np.array(dep_data["conductance_S"]) * 1e6  # Convert to µS
            pulse_num = dep_data["pulse_number"]
            pulse_num_shifted = [p + pulse_offset for p in pulse_num]
            ax.plot(pulse_num_shifted, g_dep, 's--', color=colors_dep[i], 
                   linewidth=2, markersize=4, label=f'Cycle {cycle_num} Dep', alpha=0.8)
            max_pulse_count = max(max_pulse_count, max(pulse_num_shifted))
    
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", fontsize=11)
    
    # Update title with progress
    if n_completed < results['n_cycles']:
        title = f"Pot-Dep Cycles (In Progress: {n_completed}/{results['n_cycles']})"
    else:
        title = f"Pot-Dep Cycles (Complete: {n_completed}/{results['n_cycles']})"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    canvas.draw()
    
    # Update text display (only show summary if complete)
    if n_completed == results['n_cycles'] and 'summary_metrics' in results:
        params_text.delete("1.0", "end")  # Clear previous text
        params_text.insert("end", f"\n{'='*30}\n")
        params_text.insert("end", f"Cycle Characterization Results\n")
        params_text.insert("end", f"{'='*30}\n")
        params_text.insert("end", f"Cycles completed: {results['n_cycles']}\n")
        params_text.insert("end", f"{'-'*30}\n")
        
        summary = results.get("summary_metrics", {})
        for key, value in summary.items():
            params_text.insert("end", f"{key}: {value}\n")
        
        params_text.insert("end", f"{'='*30}\n\n")
        params_text.see("end")


def plot_multi_device_data(multi_results):
    """
    Plots multi-device cycle results.
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    device_colors = plt.cm.tab10(np.linspace(0, 1, multi_results['n_devices']))
    
    for dev_idx, device_result in enumerate(multi_results["devices"]):
        if "error" in device_result:
            continue
        
        device_name = device_result["device_name"]
        
        # Calculate average conductance trajectory for this device
        all_pot_g = []
        all_dep_g = []
        
        for cycle in device_result["cycles"]:
            if cycle["potentiation"] and "error" not in cycle["potentiation"]:
                g_pot = np.array(cycle["potentiation"]["conductance_S"]) * 1e6
                all_pot_g.append(g_pot)
            
            if cycle["depression"] and "error" not in cycle["depression"]:
                g_dep = np.array(cycle["depression"]["conductance_S"]) * 1e6
                all_dep_g.append(g_dep)
        
        if all_pot_g:
            avg_pot = np.mean(all_pot_g, axis=0)
            pulse_num = range(len(avg_pot))
            ax.plot(pulse_num, avg_pot, 'o-', color=device_colors[dev_idx],
                   linewidth=2, markersize=4, label=f'{device_name} Pot')
        
        if all_dep_g:
            avg_dep = np.mean(all_dep_g, axis=0)
            pulse_offset = len(avg_pot) if all_pot_g else 0
            pulse_num = [p + pulse_offset for p in range(len(avg_dep))]
            ax.plot(pulse_num, avg_dep, 's--', color=device_colors[dev_idx],
                   linewidth=2, markersize=4, label=f'{device_name} Dep')
    
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", fontsize=11)
    ax.set_title(f"Multi-Device Comparison ({multi_results['n_devices']} devices)", 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    canvas.draw()
    
    # Display summary for each device
    params_text.delete("1.0", "end")
    params_text.insert("end", f"\n{'='*40}\n")
    params_text.insert("end", f"Multi-Device Results\n")
    params_text.insert("end", f"{'='*40}\n")
    
    for device_result in multi_results["devices"]:
        if "error" in device_result:
            params_text.insert("end", f"\n{device_result['device_name']}: ERROR\n")
            params_text.insert("end", f"  {device_result['error']}\n")
            continue
        
        params_text.insert("end", f"\n{device_result['device_name']}:\n")
        summary = device_result.get("summary_metrics", {})
        for key, value in summary.items():
            params_text.insert("end", f"  {key}: {value}\n")
    
    params_text.insert("end", f"\n{'='*40}\n")
    params_text.see("end")








def run_srdp_characterization():
    """
    Runs SRDP characterization with frequency sweep.
    """
    global _measurement_in_progress
    if _measurement_in_progress:
        messagebox.showwarning(
            "Measurement In Progress",
            "A measurement is already running, so SRDP characterization was not started. "
            "Starting a second one would open a second session to the "
            "same instrument mid-sweep."
        )
        return

    _measurement_in_progress = True
    try:
        global synapse_data_storage
    
        try:
            # Parse frequency range
            freq_start = validate_float(srdp_freq_start_entry.get(), "SRDP Start Frequency", minimum=1e-3, maximum=1e5, units="Hz", hint="Frequency must be positive; a log-spaced sweep is undefined at or below 0 Hz.")
            freq_end = validate_float(srdp_freq_end_entry.get(), "SRDP End Frequency", minimum=1e-3, maximum=1e5, units="Hz", hint="Frequency must be positive; a log-spaced sweep is undefined at or below 0 Hz.")
            freq_points = int(validate_float(srdp_freq_points_entry.get(), "SRDP Frequency Points"))
        
            if freq_points < 2:
                raise ValueError("Need at least 2 frequency points")
        
            # Generate frequency list (log scale for better coverage)
            if srdp_log_scale_var.get():
                freq_list = np.logspace(np.log10(freq_start), np.log10(freq_end), freq_points)
            else:
                freq_list = np.linspace(freq_start, freq_end, freq_points)
        
            # Check if frequency range is achievable with current timing parameters.
            #
            # H14: this ceiling used to omit `read_settle_delay_ms` and the ~2 ms
            # of per-pulse LUA overhead that pulse-read's own check includes, so it
            # reported a maximum the instrument could not actually deliver. A user
            # requesting 100 Hz against a true 62 Hz ceiling got a run at 62 Hz
            # while results["frequencies_hz"] recorded 100 — and the SRDP x-axis is
            # wrong exactly where f0 and the slope are determined.
            stim_width = validate_float(srdp_stim_width_entry.get(), "SRDP Stim Width", minimum=1e-6, units="ms", hint="Must be greater than zero.")
            read_delay = validate_float(read_delay_entry.get(), "Read Delay")
            read_settle = validate_float(read_settle_delay_entry.get(), "Read Settle Delay")
            read_width = validate_float(read_width_entry.get(), "Read Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero.")
            nplc_val = validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz.")
            # Compute actual measurement time (same logic as LUA script generator)
            NPLC_PERIOD_MS = 1000.0 / synapse_engine.DEFAULT_LINE_FREQ_HZ
            actual_nplc = min(nplc_val, read_width / NPLC_PERIOD_MS) if read_width > 0 else nplc_val
            actual_nplc = max(0.001, actual_nplc)
            meas_time = actual_nplc * NPLC_PERIOD_MS
            LUA_PER_PULSE_OVERHEAD_MS = 2.0
            min_period = (stim_width + read_delay + read_settle + meas_time
                          + LUA_PER_PULSE_OVERHEAD_MS)
            max_achievable_freq = 1000.0 / min_period

            if freq_end > max_achievable_freq:
                warning_msg = (
                    f"⚠️ Frequency range warning:\n\n"
                    f"Your timing parameters limit the maximum frequency:\n"
                    f"  • Stim Width: {stim_width} ms\n"
                    f"  • Read Delay: {read_delay} ms\n"
                    f"  • Measurement Time: {meas_time:.2f} ms (NPLC={actual_nplc:.4f})\n"
                    f"  • Min Period: {min_period:.2f} ms\n"
                    f"  • Max Frequency: {max_achievable_freq:.1f} Hz\n\n"
                    f"You requested up to {freq_end} Hz.\n"
                    f"Higher frequencies will run as fast as the instrument can.\n\n"
                    f"Continue?"
                )

                if not messagebox.askyesno("Frequency Range Warning", warning_msg):
                    return
        
            # Collect base parameters - USE SRDP-SPECIFIC FIELDS
            stim_ch_str = stim_channel_var.get().lower()
            read_ch_str = read_channel_var.get().lower()
        
            if stim_ch_str == read_ch_str:
                raise ValueError("Stim and Read channels must be different.")
        
            base_params = {
                "mode": "srdp",
                "stim_drive_type": stim_drive_var.get(),
                "stim_level": validate_float(srdp_stim_level_entry.get(), "SRDP Stim Level"),
                "stim_width_ms": validate_float(srdp_stim_width_entry.get(), "SRDP Stim Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
                "n_pulses": int(validate_float(srdp_n_pulses_entry.get(), "SRDP # Pulses")),
                "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
                "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
                "read_settle_delay_ms": validate_float(read_settle_delay_entry.get(), "Read Settle Delay"),
                "read_width_ms": validate_float(read_width_entry.get(), "Read Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
                "compliance_A": get_compliance_value(),
                "nplc": validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz."),
                # H16: the documented line_freq_hz key, now actually supplied
                # by the GUI instead of always falling back to 50 Hz.
                "line_freq_hz": get_line_freq_hz(),
                "settle_ms": 5,
                "measure_avg": 1,       
                "wire_modes": build_wire_modes(),
                "measure_avg_user": _read_samples_per_pulse(),
                "wavelength_nm": validate_float(srdp_wavelength_entry.get(), "Wavelength"),
                "intensity_mW_cm2": validate_float(srdp_light_intensity_entry.get(), "Light Intensity"),
                "sample_id": srdp_sample_id_entry.get().strip()
            }
        
            # Run measurement or simulation
            if simulate_var.get():
                results = synapse_engine.simulate_srdp(base_params, freq_list.tolist())
                messagebox.showinfo("Simulation", "SRDP simulation completed!")
            else:
                port = port_entry.get().strip()
                connection = connection_type.get()
            
                if not port:
                    raise ValueError("Port Address cannot be empty.")
            
                instrument = synapse_engine.open_instrument(port, connection)

                # H15: one shared check, against the right limit for the drive type.
                # This used to compare the stimulus VOLTAGE against 1.5 A on a
                # hardcoded "2612B" substring match, refusing a 2 V pre-spike as
                # "2.0A", while performing no voltage check at all.
                try:
                    if not check_synapse_safety(instrument, base_params, "SRDP"):
                        instrument.close()
                        return
                except Exception:
                    instrument.close()
                    raise

                try:
                    results = synapse_engine.measure_srdp(
                        instrument, stim_ch_str, read_ch_str, base_params, freq_list.tolist()
                    )
                    messagebox.showinfo("Success", "SRDP characterization completed!")
                finally:
                    instrument.close()
        
            # Plot and store results
            plot_srdp_data(results)
            register_synapse_run(results)
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
    finally:
        _measurement_in_progress = False

def run_stdp_characterization():
    """
    Runs STDP characterization with timing sweep.
    """
    global _measurement_in_progress
    if _measurement_in_progress:
        messagebox.showwarning(
            "Measurement In Progress",
            "A measurement is already running, so STDP characterization was not started. "
            "Starting a second one would open a second session to the "
            "same instrument mid-sweep."
        )
        return

    _measurement_in_progress = True
    try:
        global synapse_data_storage
    
        try:
            # Parse timing range
            dt_start = validate_float(stdp_dt_start_entry.get(), "STDP Start Δt")
            dt_end = validate_float(stdp_dt_end_entry.get(), "STDP End Δt")
            dt_points = int(validate_float(stdp_dt_points_entry.get(), "STDP Δt Points"))
        
            if dt_points < 2:
                raise ValueError("Need at least 2 timing points")
        
            # Generate Δt list
            delta_t_list = np.linspace(dt_start, dt_end, dt_points)
        
            # Collect base parameters - USE STDP-SPECIFIC FIELDS
            stim_ch_str = stim_channel_var.get().lower()
            read_ch_str = read_channel_var.get().lower()
        
            if stim_ch_str == read_ch_str:
                raise ValueError("Stim and Read channels must be different.")
        
            base_params = {
                "mode": "stdp",
                "stim_drive_type": "V",  # STDP typically uses voltage
                "stim_level": validate_float(stdp_pre_level_entry.get(), "Pre-spike Level"),
                "post_spike_level": validate_float(stdp_post_level_entry.get(), "Post-spike Level"),
                "stim_width_ms": validate_float(stdp_pulse_width_entry.get(), "Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
                "stim_period_ms": validate_float(stdp_pair_period_entry.get(), "Pair Period", minimum=1e-6, units="ms", hint="Must be greater than zero."),
                "n_pulses": int(validate_float(stdp_n_pairs_entry.get(), "# Spike Pairs")),
                "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
                "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
                "read_settle_delay_ms": validate_float(read_settle_delay_entry.get(), "Read Settle Delay"),
                "read_width_ms": validate_float(read_width_entry.get(), "Read Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
                "compliance_A": get_compliance_value(),
                "nplc": validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz."),
                # H16: the documented line_freq_hz key, now actually supplied
                # by the GUI instead of always falling back to 50 Hz.
                "line_freq_hz": get_line_freq_hz(),
                "settle_ms": 5,
                "measure_avg": 1,
                "wire_modes": build_wire_modes(),
                "measure_avg_user": _read_samples_per_pulse(),
                "wavelength_nm": validate_float(stdp_wavelength_entry.get(), "Wavelength"),
                "intensity_mW_cm2": validate_float(stdp_light_intensity_entry.get(), "Light Intensity"),
                "sample_id": stdp_sample_id_entry.get().strip()
            }
        
            # Run measurement or simulation
            if simulate_var.get():
                results = synapse_engine.simulate_stdp(base_params, delta_t_list.tolist())
                messagebox.showinfo("Simulation", "STDP simulation completed!")
            else:
                port = port_entry.get().strip()
                connection = connection_type.get()
            
                if not port:
                    raise ValueError("Port Address cannot be empty.")
            
                instrument = synapse_engine.open_instrument(port, connection)
            
                # === DETECT INSTRUMENT MODEL ===
                try:
                    model = instrument.query("*IDN?")
                    print(f"✓ Connected to: {model}")
                
                    pass
                except Exception as e:
                    print(f"Warning: Could not detect instrument model: {e}")

                # H15: both spikes checked against the right limit for the drive
                # type, plus the read voltage and the device-safety advisory. The
                # old check compared a VOLTAGE against 1.5 A on a hardcoded
                # "2612B" substring, and performed no voltage check whatsoever.
                try:
                    if not check_synapse_safety(instrument, base_params, "STDP pre-spike"):
                        instrument.close()
                        return
                    post_params = dict(base_params)
                    post_params['stim_level'] = base_params.get(
                        'post_spike_level', base_params['stim_level'])
                    if not check_synapse_safety(instrument, post_params, "STDP post-spike"):
                        instrument.close()
                        return
                except Exception:
                    instrument.close()
                    raise

                try:
                    results = synapse_engine.measure_stdp(
                        instrument, stim_ch_str, read_ch_str, base_params, delta_t_list.tolist()
                    )
                    messagebox.showinfo("Success", "STDP characterization completed!")
                finally:
                    instrument.close()
        
            # Plot and store results
            plot_stdp_data(results)
            register_synapse_run(results)
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
    finally:
        _measurement_in_progress = False

def _collect_train_config(topo_var, ch_var, write_ch_var, read_ch_var,
                          stim_level_entry, stim_width_entry, period_entry,
                          n_pulses_entry, read_voltage_entry, read_delay_entry,
                          label):
    """Builds a train config dict from GUI widget values."""
    topo_str = topo_var.get()
    if topo_str == "Single SMU":
        topology = "single"
        write_ch = ch_var.get().lower()
        read_ch = write_ch
    else:
        topology = "dual"
        write_ch = write_ch_var.get().lower()
        read_ch = read_ch_var.get().lower()
        if write_ch == read_ch:
            raise ValueError(f"{label}: Dual-channel mode requires different Write and Read channels.")

    # Device-safety confirmation. Basic mode checks its stimulus level and
    # Visual mode checks its LED voltage, but Cycle mode checked NOTHING — a
    # mistyped 50 in the Write Level box ran unchallenged all the way up to the
    # instrument ceiling, across every cycle of the train.
    _write_level = validate_float(stim_level_entry.get(), f"{label} Write Level")
    if not synapse_engine.safety_check(_write_level, drive_type="V"):
        raise ValueError(
            f"{label}: cancelled at the write-level safety prompt "
            f"({_write_level:g} V). Nothing was measured."
        )

    return {
        "topology": topology,
        "write_ch": write_ch,
        "read_ch": read_ch,
        "stim_drive_type": "V",
        "stim_level": _write_level,
        "stim_width_ms": validate_float(stim_width_entry.get(), f"{label} Write Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
        "stim_period_ms": validate_float(period_entry.get(), f"{label} Period", minimum=1e-6, units="ms", hint="Must be greater than zero."),
        "n_pulses": int(validate_float(n_pulses_entry.get(), f"{label} # Pulses")),
        "read_voltage": validate_float(read_voltage_entry.get(), f"{label} Read Voltage"),
        "read_delay_ms": validate_float(read_delay_entry.get(), f"{label} Read Delay"),
        "compliance_A": get_compliance_value(),
        "nplc": validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz."),
                # H16: the documented line_freq_hz key, now actually supplied
                # by the GUI instead of always falling back to 50 Hz.
                "line_freq_hz": get_line_freq_hz(),
        "settle_ms": 5,
        "wire_modes": build_wire_modes(),
        "measure_avg_user": _read_samples_per_pulse(),
        "sample_id": cycle_sample_id_entry.get().strip(),
    }


def run_cycle_characterization():
    """Runs unified LUA-based cycle characterization with real-time plotting."""
    global _measurement_in_progress
    if _measurement_in_progress:
        messagebox.showwarning(
            "Measurement In Progress",
            "A measurement is already running, so Cycle characterization was not started. "
            "Starting a second one would open a second session to the "
            "same instrument mid-sweep."
        )
        return

    _measurement_in_progress = True
    try:
        global synapse_data_storage

        try:
            # Check if multi-device mode is enabled
            if multi_device_var.get():
                run_multi_device_characterization()
                return

            # Build Train A config
            train_a_config = _collect_train_config(
                cycle_train_a_topo_var, cycle_train_a_ch_var,
                cycle_train_a_write_ch_var, cycle_train_a_read_ch_var,
                train_a_stim_level_entry, train_a_stim_width_entry,
                train_a_period_entry, train_a_n_pulses_entry,
                train_a_read_voltage_entry, train_a_read_delay_entry,
                "Train A")

            # Build Train B config (if enabled)
            train_b_config = None
            if cycle_enable_train_b_var.get():
                train_b_config = _collect_train_config(
                    cycle_train_b_topo_var, cycle_train_b_ch_var,
                    cycle_train_b_write_ch_var, cycle_train_b_read_ch_var,
                    train_b_stim_level_entry, train_b_stim_width_entry,
                    train_b_period_entry, train_b_n_pulses_entry,
                    train_b_read_voltage_entry, train_b_read_delay_entry,
                    "Train B")

            # Cycle control
            n_cycles = int(validate_float(cycle_n_cycles_entry.get(), "# Cycles"))
            inter_train_delay = validate_float(cycle_inter_train_delay_entry.get(), "Inter-train Delay")
            inter_cycle_delay = validate_float(cycle_delay_entry.get(), "Inter-cycle Delay")

            if n_cycles <= 0:
                raise ValueError("Number of cycles must be greater than zero")

            # Real-time update callback
            def update_plot(results):
                plot_cycle_data(results)
                # H21: update_idletasks(), NOT update().
                #
                # update() processes ALL pending events, including button clicks.
                # A second click on Run while a measurement was in flight therefore
                # re-entered this handler and opened a SECOND VISA session against
                # the same instrument, mid-sweep. update_idletasks() redraws the
                # plot without dispatching input events, which is all this callback
                # needs. The _measurement_in_progress guard backs it up.
                root.update_idletasks()

            # Run measurement or simulation
            if simulate_var.get():
                results = synapse_cycle.simulate_cycle_sequence(
                    train_a_config, train_b_config, n_cycles,
                    update_callback=update_plot
                )
                messagebox.showinfo("Simulation", "Cycle simulation completed!")
            else:
                port = port_entry.get().strip()
                connection = connection_type.get()

                if not port:
                    raise ValueError("Port Address cannot be empty.")

                instrument = synapse_engine.open_instrument(port, connection)

                # H15: cycle mode performed NO safety check of any kind — neither
                # an instrument limit nor the device-safety advisory that Basic and
                # Visual both apply. Each train is checked separately, because each
                # carries its own drive type, stimulus level and read voltage.
                try:
                    for label, cfg in (("Cycle Train A", train_a_config),
                                       ("Cycle Train B", train_b_config)):
                        if cfg is None:
                            continue
                        if not check_synapse_safety(instrument, cfg, label):
                            instrument.close()
                            return
                except Exception:
                    instrument.close()
                    raise

                try:
                    results = synapse_cycle.cycle_sequence(
                        instrument, train_a_config, train_b_config, n_cycles,
                        inter_train_delay, inter_cycle_delay
                    )
                    messagebox.showinfo("Success", "Cycle characterization completed!")
                finally:
                    instrument.close()

            # Final plot update
            plot_cycle_data(results)

            # Store results
            register_synapse_run(results)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    finally:
        _measurement_in_progress = False

def run_multi_device_characterization():
    """Runs cycling across multiple devices sequentially."""
    global _measurement_in_progress
    if _measurement_in_progress:
        messagebox.showwarning(
            "Measurement In Progress",
            "A measurement is already running, so Multi-device characterization was not started. "
            "Starting a second one would open a second session to the "
            "same instrument mid-sweep."
        )
        return

    _measurement_in_progress = True
    try:
        global synapse_data_storage

        try:
            # Get multi-device parameters
            n_devices = int(validate_float(n_devices_entry.get(), "# Devices"))
            inter_device_delay = validate_float(inter_device_delay_entry.get(), "Inter-device Delay")
            device_name_pattern = device_name_pattern_entry.get().strip()

            if n_devices <= 0:
                raise ValueError("Number of devices must be greater than zero")

            # Build train configs (same for all devices)
            train_a_config = _collect_train_config(
                cycle_train_a_topo_var, cycle_train_a_ch_var,
                cycle_train_a_write_ch_var, cycle_train_a_read_ch_var,
                train_a_stim_level_entry, train_a_stim_width_entry,
                train_a_period_entry, train_a_n_pulses_entry,
                train_a_read_voltage_entry, train_a_read_delay_entry,
                "Train A")

            train_b_config = None
            if cycle_enable_train_b_var.get():
                train_b_config = _collect_train_config(
                    cycle_train_b_topo_var, cycle_train_b_ch_var,
                    cycle_train_b_write_ch_var, cycle_train_b_read_ch_var,
                    train_b_stim_level_entry, train_b_stim_width_entry,
                    train_b_period_entry, train_b_n_pulses_entry,
                    train_b_read_voltage_entry, train_b_read_delay_entry,
                    "Train B")

            n_cycles = int(validate_float(cycle_n_cycles_entry.get(), "# Cycles"))
            inter_train_delay = validate_float(cycle_inter_train_delay_entry.get(), "Inter-train Delay")
            inter_cycle_delay = validate_float(cycle_delay_entry.get(), "Inter-cycle Delay")

            # The Multi-Device panel's own fields. These were displayed with
            # defaults and then read by NOTHING: a batch measured at 365 nm
            # recorded no wavelength and was named from the Cycle tab's
            # Sample ID rather than the batch prefix the user had typed.
            md_prefix = multidevice_sample_id_entry.get().strip() \
                or cycle_sample_id_entry.get().strip()
            md_wavelength = validate_float(
                multidevice_wavelength_entry.get(), "Multi-Device Wavelength",
                minimum=200, maximum=2000, units="nm")
            md_intensity = validate_float(
                multidevice_light_intensity_entry.get(),
                "Multi-Device Light Intensity", minimum=0.0, units="mW/cm2")

            # Create device configurations
            device_configs = []
            for i in range(n_devices):
                a_cfg = train_a_config.copy()
                a_cfg['sample_id'] = f"{md_prefix}_Device_{i+1}"
                a_cfg['wavelength_nm'] = md_wavelength
                a_cfg['intensity_mW_cm2'] = md_intensity
                b_cfg = None
                if train_b_config:
                    b_cfg = train_b_config.copy()
                    b_cfg['sample_id'] = a_cfg['sample_id']
                    b_cfg['wavelength_nm'] = md_wavelength
                    b_cfg['intensity_mW_cm2'] = md_intensity

                device_configs.append({
                    "device_name": f"{device_name_pattern}_{i+1}",
                    "train_a_config": a_cfg,
                    "train_b_config": b_cfg,
                    "n_cycles": n_cycles,
                    "inter_train_delay_ms": inter_train_delay,
                    "inter_cycle_delay_ms": inter_cycle_delay,
                    "inter_device_delay_ms": inter_device_delay,
                })

            # Real-time update callback
            def update_plot(multi_results):
                plot_multi_device_data(multi_results)
                # See the note in run_cycle_characterization (H21).
                root.update_idletasks()

            # Run measurement or simulation
            if simulate_var.get():
                results = synapse_cycle.simulate_multi_device_cycles(
                    device_configs, update_callback=update_plot
                )
                messagebox.showinfo("Simulation", f"Multi-device simulation completed for {n_devices} devices!")
            else:
                port = port_entry.get().strip()
                connection = connection_type.get()

                if not port:
                    raise ValueError("Port Address cannot be empty.")

                confirm = messagebox.askyesno(
                    "Confirm Multi-Device Measurement",
                    f"This will sequentially measure {n_devices} devices.\n"
                    f"Each device will undergo {n_cycles} cycles.\n\n"
                    f"This may take a long time. Continue?"
                )
                if not confirm:
                    return

                instrument = synapse_engine.open_instrument(port, connection)

                try:
                    results = synapse_cycle.run_multi_device_cycles(
                        instrument, device_configs, update_callback=update_plot
                    )
                    messagebox.showinfo("Success", f"Multi-device characterization completed for {n_devices} devices!")
                finally:
                    instrument.close()

            # Final plot update
            plot_multi_device_data(results)

            # Store results
            register_synapse_run(results)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    finally:
        _measurement_in_progress = False

def run_transistor_measurement():

    global jv_curves
    try:
        gate_start = validate_float(gate_start_voltage.get(), "Gate Start Voltage")
        gate_end = validate_float(gate_end_voltage.get(), "Gate End Voltage")
        gate_step = validate_float(gate_voltage_step.get(), "Gate Voltage Step")
        channel = channel_selection.get()
        port = port_entry.get().strip()
        connection = connection_type.get()

        if not port:
            raise ValueError("Port Address cannot be empty.")

        instrument = None
        if connection == "GPIB":
            instrument = synapse_engine.open_instrument(port, "GPIB")
        elif connection == "RS232":
            instrument = synapse_engine.open_instrument(port, "RS232")
        elif connection == "LAN":
            rm = pyvisa.ResourceManager()
            instrument = open_lan_instrument(rm, port)

        if instrument is None:
            raise ValueError("Could not establish communication with the instrument.")

        drain_channel = "smua" if channel == "Channel A" else "smub"
        gate_channel = "smub" if channel == "Channel A" else "smua"

        instrument.write("*CLS")

        instrument.write(f"{gate_channel}.source.func = {gate_channel}.OUTPUT_DCVOLTS")
        instrument.write(f"{gate_channel}.source.autorangev = {gate_channel}.AUTORANGE_ON")
        instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_ON")

        # Use linspace to avoid floating-point precision errors
        gate_num_points = int(round((gate_end - gate_start) / gate_step)) + 1
        gate_voltages = np.linspace(gate_start, gate_end, gate_num_points)
        all_transistor_data = []

        failed_gate_points = []

        for vgs in gate_voltages:
            vgs = round(vgs, 2)
            instrument.write(f"{gate_channel}.source.levelv = {vgs}")
            time.sleep(0.1)

            instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_ON")

            # H19: detect whether this gate point actually produced a curve.
            #
            # The old check was `if not jv_curves`, which only catches the case
            # where NO curve has ever been recorded. When a sweep failed
            # mid-series, `jv_curves[-1]` was still the PREVIOUS gate point's
            # curve — so it was relabelled with the new V_GS and appended a
            # second time, producing two identical traces under different gate
            # voltages. That is silently fabricated data: it looks like a
            # measurement, and nothing distinguishes it from one.
            n_before = len(jv_curves)
            run_measurement_buffered(transistor_mode=True)

            if len(jv_curves) == n_before:
                print(f"WARNING: no JV data collected for V_GS = {vgs} V; "
                      "this gate point is omitted rather than duplicated from "
                      "the previous one.")
                failed_gate_points.append(vgs)
                continue

            jv_curves[-1]["Gate Voltage (V)"] = vgs
            all_transistor_data.append(jv_curves[-1])

        if failed_gate_points:
            messagebox.showwarning(
                "Incomplete Transistor Sweep",
                f"{len(failed_gate_points)} of {len(gate_voltages)} gate points "
                "produced no data and have been omitted:\n"
                f"  V_GS = {', '.join(f'{v} V' for v in failed_gate_points)}\n\n"
                "The remaining curves are valid. Nothing has been substituted "
                "for the missing points."
            )

        instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_OFF")

        if not all_transistor_data:
            messagebox.showerror("Error", "No valid transistor JV data collected.")
            return

        jv_curves = all_transistor_data

        print(f"✅ Collected {len(jv_curves)} JV curves.")

        if not jv_curves:
            messagebox.showerror("Error", "No JV curves available for plotting.")
            return

        register_transistor_run(jv_curves)

        ax.clear()
        colors = plt.cm.viridis(np.linspace(0, 1, len(gate_voltages)))
        for i, entry in enumerate(jv_curves):
            if "Voltages" in entry and "Current Density" in entry:
                ax.plot(entry["Voltages"], entry["Current Density"], 
                        label=f"V_GS = {entry['Gate Voltage (V)']:.2f} V", color=colors[i])
            else:
                print(f"⚠️ Missing data for entry {i}")

        ax.set_title("Transistor JV Curves")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        ax.legend(loc="upper left")
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        if 'instrument' in locals() and instrument is not None:
            instrument.close()

def save_transistor_data():

    try:
        file_name = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Transistor JV Data")
        if not file_name:
            return

        with open(file_name, "w", encoding="utf-8") as f:
            if not jv_curves:
                messagebox.showerror("Save Error", "No JV curves available to save.")
                return

            try:
                area = validate_float(surface_area.get(), "Sample Surface Area")
            except ValueError:
                messagebox.showerror("Save Error", "Invalid surface area value.")
                return

            header_row = []
            for entry in jv_curves:
                if "Gate Voltage (V)" not in entry:
                    messagebox.showerror("Save Error", "'Gate Voltage (V)' key missing in data.")
                    return
                gate_voltage = entry["Gate Voltage (V)"]
                header_row.extend([f"V_GS = {gate_voltage} V", f"Surface Area = {area} cm²"])

            f.write(",".join(header_row) + "\n")

            column_labels = []
            for _ in jv_curves:
                column_labels.extend(["Voltage (V)", "Current Density (mA/cm²)"])
            f.write(",".join(column_labels) + "\n")

            max_length = max(len(entry["Voltages"]) for entry in jv_curves)

            for i in range(max_length):
                row = []
                for entry in jv_curves:
                    voltages = entry["Voltages"]
                    currents = entry["Current Density"]
                    row.append(str(voltages[i]) if i < len(voltages) else "")
                    row.append(str(currents[i]) if i < len(currents) else "")
                f.write(",".join(row) + "\n")

        messagebox.showinfo("Save Successful", "Transistor JV data saved successfully.")

    except Exception as e:
        messagebox.showerror("Save Error", str(e))

# Maps the Synapse Type combo labels to the canonical mode strings the engine
# dispatches on (see pulse_read_sequence). The combo label is display text and
# must never reach the engine directly: "Memristor (Pulse)" lowercased is
# 'memristor (pulse)', which matches no engine branch and silently drops the
# measurement onto the slow PC-timed path.
SYNAPSE_MODE_BY_LABEL = {
    "Electrical": "electrical",
    "Visual": "visual",
    "Memristor (Pulse)": "memristor_pulse",
}


SYNAPSE_LABEL_BY_MODE = {v: k for k, v in SYNAPSE_MODE_BY_LABEL.items()}
# Modes set directly by their own submodes rather than by the Synapse Type combo.
SYNAPSE_LABEL_BY_MODE.update({
    'srdp': 'SRDP',
    'stdp': 'STDP',
    'visual_standard': 'Visual (Self-Powered)',
    'visual_continuous': 'Visual Continuous I(t)',
})


_measurement_in_progress = False


def measurement_guard(label):
    """Refuse to start a measurement while another one is running.

    H21: the cycle and multi-device callbacks called `root.update()` during a
    live VISA session, which dispatches pending input events — so a second
    click on Run re-entered the handler and opened a SECOND session against the
    same instrument mid-sweep. Those callbacks now use `update_idletasks()`,
    but any future `update()`, modal dialog or long-running callback would
    reopen the hole, so re-entry is refused structurally as well.

    Use as a context manager:

        with measurement_guard("Cycle") as allowed:
            if not allowed:
                return
            ...
    """
    class _Guard:
        def __enter__(self):
            global _measurement_in_progress
            if _measurement_in_progress:
                messagebox.showwarning(
                    "Measurement In Progress",
                    f"A measurement is already running, so {label} was not "
                    "started.\n\nStarting a second measurement would open a "
                    "second session to the same instrument while the first is "
                    "mid-sweep."
                )
                return False
            _measurement_in_progress = True
            return True

        def __exit__(self, *exc_info):
            global _measurement_in_progress
            _measurement_in_progress = False
            return False

    return _Guard()


def check_synapse_safety(instrument, params, mode_label):
    """Validate stimulus and read levels against the instrument and the device.

    H15: SRDP and STDP compared a VOLTAGE against the 1.5 A current limit, so a
    2 V STDP pre-spike was refused on a 2612B as "2.0A" — while neither mode
    performed any voltage limit check at all, and neither called
    synapse_engine.safety_check(). Cycle mode called neither. Basic and Visual
    both did, so the same stimulus was accepted or refused depending only on
    which tab it was launched from.

    Checks, in order:
      1. the stimulus against the instrument's ceiling for its OWN drive type,
      2. the read voltage against the voltage ceiling (it is applied to the
         device for the whole inter-pulse period in some modes),
      3. the device-safety advisory, which asks the user to confirm.

    Returns True to proceed. Raises ValueError on an instrument limit — that is
    a hardware constraint, not a preference — and returns False if the user
    declines the device-safety prompt.
    """
    drive_type = params.get('stim_drive_type', 'V')
    stim_level = abs(float(params.get('stim_level', 0.0)))
    read_voltage = abs(float(params.get('read_voltage', 0.0)))

    max_voltage = synapse_engine.get_max_voltage(instrument)
    max_current = synapse_engine.get_max_current(instrument)
    model = synapse_engine.validate_instrument_model(instrument)

    if drive_type == 'I':
        if stim_level > max_current:
            raise ValueError(
                f"{mode_label}: stimulus current {stim_level} A exceeds the "
                f"{model} maximum of {max_current} A."
            )
    else:
        if stim_level > max_voltage:
            raise ValueError(
                f"{mode_label}: stimulus voltage {stim_level} V exceeds the "
                f"{model} maximum of {max_voltage} V."
            )

    if read_voltage > max_voltage:
        raise ValueError(
            f"{mode_label}: read voltage {read_voltage} V exceeds the {model} "
            f"maximum of {max_voltage} V."
        )

    # Device-safety advisory, in the units actually being driven. This used to
    # run ONLY for voltage drive, so a current stimulus — where the compliance
    # field does not limit the sourced quantity at all — was never checked.
    if not synapse_engine.safety_check(stim_level, drive_type=drive_type):
        return False

    return True


def synapse_mode_display(mode):
    """Human-readable label for a canonical mode string, for titles and reports.

    Display only — never feed the result back into the engine. Modes with no
    combo label (srdp, stdp, the visual sub-modes) are prettified rather than
    rejected, since this is cosmetic and must not break a plot.
    """
    if mode in SYNAPSE_LABEL_BY_MODE:
        return SYNAPSE_LABEL_BY_MODE[mode]
    return str(mode).replace('_', ' ').title()


def canonical_synapse_mode(label):
    """Translate a Synapse Type combo label into its canonical engine mode.

    Raises ValueError on an unknown label rather than passing display text
    through: an unrecognised mode must fail loudly, never degrade silently to a
    different execution path.
    """
    try:
        return SYNAPSE_MODE_BY_LABEL[label]
    except KeyError:
        raise ValueError(
            f"Unknown Synapse Type '{label}'.\n"
            f"Expected one of: {', '.join(SYNAPSE_MODE_BY_LABEL)}.\n"
            "Add the new type to SYNAPSE_MODE_BY_LABEL and to the LUA dispatch "
            "in synapse_engine.pulse_read_sequence before using it."
        )


def run_synapse_mode():
    """
    Runs synapse measurement using the synapse_engine module.
    Handles both hardware measurements and simulation.
    """
    global synapse_data_storage
    
    try:
        # Validate channel selection
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different (smua/smub).")
        
        # Parse and validate all parameters
        stim_level = validate_float(stim_level_entry.get(), "Stim Level")
        stim_width = validate_float(stim_width_entry.get(), "Stim Width", minimum=1e-6, units="ms", hint="Must be greater than zero.")
        stim_period = validate_float(stim_period_entry.get(), "Stim Period", minimum=1e-6, units="ms", hint="Must be greater than zero.")
        n_pulses = int(validate_float(n_pulses_entry.get(), "# Pulses"))
        read_voltage = validate_float(read_voltage_entry.get(), "Read Voltage")
        read_delay = validate_float(read_delay_entry.get(), "Read Delay")
        compliance_A = get_compliance_value()
        
        # Validation checks
        if stim_period < stim_width:
            raise ValueError("Stim Period must be >= Stim Width")
        
        if n_pulses <= 0:
            raise ValueError("Number of pulses must be greater than zero")
        
        if compliance_A <= 0:
            raise ValueError("Compliance must be greater than zero")
        
        # Safety check for high stimulus levels, in the driven units. Reading
        # the drive type matters: in current mode `stim_level` is amperes, and
        # comparing it against a voltage threshold passed 1 A unchallenged.
        if not synapse_engine.safety_check(stim_level,
                                           drive_type=stim_drive_var.get()):
            return
        
        # Collect parameters
        params = {
            "mode": canonical_synapse_mode(synapse_mode_selection.get()),
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": stim_level,
            "stim_width_ms": stim_width,
            "stim_period_ms": stim_period,
            "n_pulses": n_pulses,
            "read_voltage": read_voltage,
            "read_delay_ms": read_delay,
            "read_settle_delay_ms": validate_float(read_settle_delay_entry.get(), "Read Settle Delay"),
            "read_width_ms": validate_float(read_width_entry.get(), "Read Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero."),
            "compliance_A": compliance_A,
            "nplc": validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz."),
                # H16: the documented line_freq_hz key, now actually supplied
                # by the GUI instead of always falling back to 50 Hz.
                "line_freq_hz": get_line_freq_hz(),
            "measure_avg": 1,
            "settle_ms": 5,
            "wire_modes": build_wire_modes(),
            "measure_avg_user": _read_samples_per_pulse(),
            "wavelength_nm": validate_float(wavelength_entry.get(), "Wavelength"),
            "intensity_mW_cm2": validate_float(light_intensity_entry.get(), "Light Intensity"),
            "sample_id": sample_id_entry.get().strip()
        }
                
        # Run measurement or simulation
        if simulate_var.get():
            # Simulation mode
            results = synapse_engine.simulate_pulse_read(params)
            messagebox.showinfo("Simulation", "Simulation completed successfully!")
        else:
            # Hardware mode
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            # Open instrument
            instrument = synapse_engine.open_instrument(port, connection)
            
            
            # === DETECT INSTRUMENT MODEL ===
            try:
                model = instrument.query("*IDN?")
                print(f"✓ Connected to: {model}")

                # Check if current exceeds 2612B capability (only for current mode)
                if "2612B" in model and params['stim_drive_type'] == "I":
                    stim_current = abs(params['stim_level'])
                    if stim_current > 1.5:
                        instrument.close()
                        messagebox.showerror(
                            "Current Limit Exceeded",
                            f"ERROR: 2612B maximum current is 1.5A\n\n"
                            f"You requested: {stim_current}A\n\n"
                            f"Solutions:\n"
                            f"  • Reduce Stim Level to ≤1.5A, or\n"
                            f"  • Use Keithley 2636A (supports up to 10A)"
                        )
                        return

                # Check voltage limits for 2600B series (voltage mode only)
                # Low voltage models (2601B/02B/04B): 40V limit
                # High voltage models (2611B/12B/14B/3xB): 200V limit
                if params['stim_drive_type'] == "V":
                    max_voltage_instr = synapse_engine.get_max_voltage(instrument)
                    stim_voltage = abs(params['stim_level'])
                    read_voltage = abs(params['read_voltage'])
                    requested_voltage = max(stim_voltage, read_voltage)

                    if requested_voltage > max_voltage_instr:
                        instrument.close()
                        messagebox.showerror(
                            "Voltage Limit Exceeded",
                            f"ERROR: Instrument voltage limit exceeded.\n\n"
                            f"Instrument: {model.strip()}\n"
                            f"Maximum voltage: {max_voltage_instr}V\n"
                            f"Requested voltage: {requested_voltage}V\n\n"
                            f"Please reduce voltage levels to ≤{max_voltage_instr}V."
                        )
                        return
            except Exception as e:
                print(f"⚠️  Warning: Could not detect instrument model: {e}")
            
            
            # Configure timeout
            try:
                user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
                if user_timeout <= 0:
                    raise ValueError("Timeout duration must be greater than zero.")
                timeout_value = user_timeout * 1000
            except ValueError as timeout_error:
                # Refuse rather than silently substituting a computed value:
                # the user typed something, and overriding their intent with
                # no dialog and no console line destroyed both the value and
                # the message explaining why it was rejected.
                suggested_s = max(5.0, (params['stim_period_ms']
                                        * params['n_pulses']) / 1000.0 * 1.5)
                raise ValueError(
                    f"Timeout Duration: {timeout_error}\n"
                    f"Enter the timeout in seconds (must be greater than zero). "
                    f"Suggested for this sequence: {suggested_s:.0f} s "
                    f"(n_pulses x stim_period x 1.5). Nothing was measured."
                )
            
            instrument.timeout = timeout_value
            
            try:
                # Run the pulse-read sequence
                results = synapse_engine.pulse_read_sequence(
                    instrument, 
                    stim_ch_str, 
                    read_ch_str, 
                    params
                )
                messagebox.showinfo("Success", "Synapse measurement completed successfully!")
                
            finally:
                # Close instrument
                instrument.close()
        
        # Plot results
        plot_synapse_data(results)

        # Store results for saving
        register_synapse_run(results)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_visual_synapse_mode():
    """
    Runs visual (self-powered) synapse measurement.
    Measures photocurrent (Jsc) at 0V during light pulses.
    """
    global synapse_data_storage

    try:
        # Validate channel selection
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()

        if stim_ch_str == read_ch_str:
            raise ValueError("Stim (LED) and Read (Device) channels must be different.")

        # Parse visual synapse parameters
        light_voltage = validate_float(visual_light_voltage_entry.get(), "Light Pulse Voltage")
        pulse_width = validate_float(visual_pulse_width_entry.get(), "Pulse Width", minimum=1e-6, units="ms", hint="Must be greater than zero.")
        pulse_period = validate_float(visual_pulse_period_entry.get(), "Pulse Period", minimum=1e-6, units="ms", hint="Must be greater than zero.")
        n_pulses = int(validate_float(visual_n_pulses_entry.get(), "# Pulses"))
        measure_start_delay = validate_float(visual_measure_start_entry.get(), "Measure Start Delay")
        measure_end_margin = validate_float(visual_measure_end_entry.get(), "Measure End Margin")
        readings_per_pulse = int(validate_float(visual_readings_per_pulse_entry.get(), "Readings per Pulse"))
        sample_interval = validate_float(visual_sample_interval_entry.get(), "Sample Interval")

        continuous_mode = visual_continuous_var.get() == 1

        # Validation
        if pulse_period < pulse_width:
            raise ValueError("Pulse Period must be >= Pulse Width")

        if n_pulses <= 0:
            raise ValueError("Number of pulses must be greater than zero")

        measurement_window = pulse_width - measure_start_delay - measure_end_margin
        if measurement_window <= 0 and not continuous_mode:
            raise ValueError(
                f"Invalid measurement window.\n"
                f"Pulse width ({pulse_width}ms) must be > start delay ({measure_start_delay}ms) + end margin ({measure_end_margin}ms)"
            )

        compliance_A = get_compliance_value()

        # Safety check for high LED voltages
        if not synapse_engine.safety_check(light_voltage, max_safe_voltage=2.5):
            return

        # Collect parameters
        params = {
            "light_pulse_voltage": light_voltage,
            "pulse_width_ms": pulse_width,
            "pulse_period_ms": pulse_period,
            "n_pulses": n_pulses,
            "measure_start_delay_ms": measure_start_delay,
            "measure_end_margin_ms": measure_end_margin,
            "readings_per_pulse": readings_per_pulse,
            "sample_interval_ms": sample_interval,
            "compliance_A": compliance_A,
            "nplc": validate_float(nplc_entry.get(), "NPLC", minimum=0.001, maximum=25, units="power line cycles", hint="The 2600-series ADC accepts 0.001-25 NPLC; 1 NPLC = 20 ms at 50 Hz."),
                # H16: the documented line_freq_hz key, now actually supplied
                # by the GUI instead of always falling back to 50 Hz.
                "line_freq_hz": get_line_freq_hz(),
            "wire_modes": build_wire_modes(),
            "measure_avg_user": _read_samples_per_pulse(),
            "wavelength_nm": validate_float(visual_wavelength_entry.get(), "Wavelength"),
            "intensity_mW_cm2": validate_float(visual_intensity_entry.get(), "Light Intensity"),
            "sample_id": visual_sample_id_entry.get().strip()
        }

        # Run measurement or simulation
        if simulate_var.get():
            # Simulation mode
            results = synapse_engine.simulate_visual_synapse(params, continuous_mode=continuous_mode)
            messagebox.showinfo("Simulation", "Visual synapse simulation completed!")
        else:
            # Hardware mode
            port = port_entry.get().strip()
            connection = connection_type.get()

            if not port:
                raise ValueError("Port Address cannot be empty.")

            # Open instrument
            instrument = synapse_engine.open_instrument(port, connection)

            # Check instrument model
            try:
                model = instrument.query("*IDN?")
                print(f"✓ Connected to: {model}")

                # Check for LUA support (required for visual mode)
                if not synapse_engine.supports_lua_execution(instrument):
                    instrument.close()
                    messagebox.showerror(
                        "Instrument Not Supported",
                        "Visual synapse mode requires a LUA-capable instrument (2600 series).\n\n"
                        "Please use Keithley 2602B, 2612B, 2636A, or similar."
                    )
                    return

                # Check voltage limits for instrument
                max_voltage_instr = synapse_engine.get_max_voltage(instrument)
                if abs(light_voltage) > max_voltage_instr:
                    instrument.close()
                    messagebox.showerror(
                        "Voltage Limit Exceeded",
                        f"ERROR: Instrument voltage limit exceeded.\n\n"
                        f"Instrument: {model.strip()}\n"
                        f"Maximum voltage: {max_voltage_instr}V\n"
                        f"Requested LED voltage: {abs(light_voltage)}V\n\n"
                        f"Please reduce LED voltage to ≤{max_voltage_instr}V."
                    )
                    return

            except Exception as e:
                print(f"Warning: Could not detect instrument model: {e}")

            # Configure timeout
            try:
                user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
                timeout_value = max(user_timeout * 1000, 30000)
            except ValueError:
                total_time = (params['pulse_period_ms'] * params['n_pulses']) / 1000.0
                timeout_value = max(30000, total_time * 1000 * 1.5)

            instrument.timeout = timeout_value

            try:
                # Run visual synapse measurement
                results = synapse_engine.visual_synapse_sequence(
                    instrument,
                    stim_ch_str,
                    read_ch_str,
                    params,
                    continuous_mode=continuous_mode
                )
                messagebox.showinfo("Success", "Visual synapse measurement completed!")

            finally:
                instrument.close()

        # Plot results
        plot_visual_synapse_data(results)

        # Store results for saving
        register_synapse_run(results)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def plot_visual_synapse_data(results):
    """
    Plots visual synapse measurement results (Jsc vs Pulse Number or I vs Time).

    Args:
        results (dict): Results from synapse_engine.visual_synapse_sequence
    """
    global ax, canvas

    # Clear existing plots
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None

    jsc_data = np.array(results["Jsc_A"])
    time_data = np.array(results["time_s"])

    is_continuous = results.get("mode") == "visual_continuous"

    if is_continuous:
        # Continuous mode: I(t) plot
        ax.plot(time_data, jsc_data * 1e6, '-', color='tab:purple', linewidth=1.5)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Photocurrent (µA)", fontsize=11)
        ax.set_title("Visual Synapse - Continuous I(t)", fontsize=12, fontweight='bold')
    else:
        # Standard mode: Jsc vs Pulse#
        pulse_numbers = results["pulse_number"]
        ax.plot(pulse_numbers, jsc_data * 1e6, 'o-', color='tab:purple', linewidth=2, markersize=4)
        ax.set_xlabel("Pulse Number", fontsize=11)
        ax.set_ylabel("Photocurrent Jsc (µA)", fontsize=11)
        ax.set_title("Visual Synapse - Jsc vs Pulse", fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    canvas.draw()

    # Calculate and display metrics
    metrics = synapse_engine.calculate_visual_synapse_metrics(results)

    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"Visual Synapse Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Mode: {'Continuous I(t)' if is_continuous else 'Jsc per pulse'}\n")
    params_text.insert("end", f"Pulses: {results['params']['n_pulses']}\n")
    params_text.insert("end", f"Light Voltage: {results['params']['light_pulse_voltage']} V\n")
    params_text.insert("end", f"{'-'*30}\n")

    for key, value in metrics.items():
        params_text.insert("end", f"{key}: {value}\n")

    params_text.insert("end", f"{'='*30}\n\n")
    params_text.see("end")


def save_synapse_data():
    """
    Saves synapse measurement data to CSV file with metadata and metrics.
    Handles regular synapse, SRDP, and STDP data.
    """
    global synapse_data_storage
    
    try:
        if not synapse_data_storage:
            messagebox.showerror("Save Error", "No synapse data available to save.")
            return
        
        file_name = asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")], 
            title="Save Synapse/Memristor Data"
        )
        
        if not file_name:
            return
        
        # Get the most recent measurement
        last_data = synapse_data_storage[-1]
        
        # Determine data type and save accordingly
        if "frequencies_hz" in last_data:
            # SRDP data
            save_srdp_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"SRDP data saved successfully to:\n{file_name}"
            )
        elif "delta_t_ms" in last_data:
            # STDP data
            save_stdp_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"STDP data saved successfully to:\n{file_name}"
            )
        elif "cycles" in last_data and "devices" not in last_data:
            # Single-device cycle data
            synapse_cycle.save_cycle_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"Cycle data saved successfully to:\n{file_name}"
            )
        elif "devices" in last_data:
            # Multi-device cycle data
            synapse_cycle.save_multi_device_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"Multi-device data saved successfully to:\n{file_name}"
            )
        elif "Jsc_A" in last_data:
            # Visual (self-powered) synapse data
            metrics = synapse_engine.save_visual_synapse_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"Visual synapse data saved successfully to:\n{file_name}\n\n"
                f"Metrics calculated:\n" +
                "\n".join(f"  • {k}: {v}" for k, v in list(metrics.items())[:3])
            )
        else:
            # Regular synapse pulse-read data
            metrics = synapse_engine.process_and_save_synapse_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful",
                f"Synapse data saved successfully to:\n{file_name}\n\n"
                f"Metrics calculated:\n" +
                "\n".join(f"  • {k}: {v}" for k, v in list(metrics.items())[:3])
            )
        
        # Clear stored data
        synapse_data_storage.clear()
        
    except Exception as e:
        messagebox.showerror("Save Error", str(e))


def export_characterization_suite():
    """
    Export complete characterization suite for model fitting and network simulation.
    
    This assembles all measurements from synapse_data_storage into a standardized
    JSON file that can be used with fitting.py and network.py.
    
    Supports:
    - Wavelength sweep measurements (for wavelength_response)
    - LTP measurements (for nonlinearity extraction)
    - Potentiation-depression cycles (for dynamic_range)
    - Retention measurements (for decay_tau)
    """
    global synapse_data_storage
    
    if not synapse_data_storage:
        messagebox.showerror("No Data", "No characterization data to export!\n\nPlease run measurements first.")
        return
    
    # Import assembly helpers
    try:
        import assembly_helpers
    except ImportError:
        messagebox.showerror(
            "Module Not Found", 
            "assembly_helpers.py not found!\n\n"
            "Please ensure assembly_helpers.py is in the same directory as this script."
        )
        return
    
    try:
        # Create a dialog to help user organize their measurements
        from tkinter import Toplevel, Label, Button, Listbox, SINGLE, END
        
        # Create selection dialog
        dialog = Toplevel(root)
        dialog.title("Export Characterization Suite")
        dialog.geometry("600x500")
        
        Label(dialog, text="Select measurements to include:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Show available measurements
        Label(dialog, text=f"Available measurements: {len(synapse_data_storage)}", font=("Arial", 10)).pack()
        
        listbox = Listbox(dialog, height=15, width=80, selectmode=SINGLE)
        listbox.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Populate listbox with measurement descriptions
        for i, data in enumerate(synapse_data_storage):
            if "cycles" in data:
                desc = f"[{i}] Pot-Dep Cycles (n={len(data['cycles'])} cycles)"
            elif "frequencies_hz" in data:
                desc = f"[{i}] SRDP ({len(data['frequencies_hz'])} freq points)"
            elif "delta_t_ms" in data:
                desc = f"[{i}] STDP ({len(data['delta_t_ms'])} timing points)"
            elif "conductance_S" in data:
                n_pulses = len(data['conductance_S'])
                if n_pulses > 30:
                    desc = f"[{i}] LTP/LTD ({n_pulses} pulses)"
                else:
                    desc = f"[{i}] Pulse-Read ({n_pulses} pulses)"
            elif "time_s" in data:
                desc = f"[{i}] Retention ({len(data['time_s'])} time points)"
            else:
                desc = f"[{i}] Unknown measurement type"
            
            listbox.insert(END, desc)
        
        # Variables to store selections.
        #
        # LTD, STDP and SRDP had no entries here at all, so no dataset with
        # experiment_type 'depression', 'stdp' or 'srdp' could ever reach the
        # fitter from hardware: beta was unfittable and the STDP window and
        # SRDP sigmoid always came from defaults, however much data had been
        # measured.
        ltp_idx = None
        ltd_idx = None
        cycle_idx = None
        retention_idx = None
        stdp_idx = None
        srdp_idx = None

        def _mark(kind, setter, note):
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "Nothing Selected",
                    f"Select a measurement in the list first, then click "
                    f"'Mark as {kind}'."
                )
                return
            idx = selection[0]
            setter(idx)
            messagebox.showinfo(f"{kind} Selected", f"Measurement {idx} marked {note}")

        def mark_as_ltp():
            def _set(i):
                nonlocal ltp_idx
                ltp_idx = i
            _mark("LTP", _set, "as LTP for nonlinearity extraction (alpha)")

        def mark_as_ltd():
            def _set(i):
                nonlocal ltd_idx
                ltd_idx = i
            _mark("LTD", _set, "as LTD for depression nonlinearity (beta)")

        def mark_as_cycles():
            def _set(i):
                nonlocal cycle_idx
                cycle_idx = i
            _mark("Cycles", _set, "for dynamic range extraction")

        def mark_as_retention():
            def _set(i):
                nonlocal retention_idx
                retention_idx = i
            _mark("Retention", _set, "for decay tau extraction")

        def mark_as_stdp():
            def _set(i):
                nonlocal stdp_idx
                stdp_idx = i
            _mark("STDP", _set, "as the STDP timing window")

        def mark_as_srdp():
            def _set(i):
                nonlocal srdp_idx
                srdp_idx = i
            _mark("SRDP", _set, "as the SRDP frequency response")
        
        # === AUTO-DETECT WAVELENGTH SWEEP ===
        def auto_detect_wavelength_sweep():
            """Auto-detect wavelength sweep measurements from metadata."""
            wavelength_dict = {}
            for idx, data in enumerate(synapse_data_storage):
                if 'metadata' in data and 'wavelength_nm' in data['metadata']:
                    wl = data['metadata']['wavelength_nm']
                    # Only add if we don't already have this wavelength
                    if wl not in wavelength_dict:
                        wavelength_dict[wl] = data
            
            # Convert to lists if we found a wavelength sweep (3+ wavelengths).
            #
            # H4: the results and the wavelengths MUST be taken from the same
            # ordering. This used to return `list(wavelength_dict.values())` —
            # insertion, i.e. run, order — alongside `sorted(keys)`, and
            # assembly_helpers zips the two together. Measuring 550, then 450,
            # then 650 nm therefore tagged the 550 nm trace as 450 nm and vice
            # versa, and the multi-Gaussian fit was fitted to permuted data
            # with no error raised anywhere. Sorting the pairs jointly makes
            # the association independent of measurement order.
            if len(wavelength_dict) >= 3:
                ordered = sorted(wavelength_dict.items(), key=lambda kv: kv[0])
                wavelengths = [wl for wl, _ in ordered]
                wavelength_results = [res for _, res in ordered]
                print(f"Auto-detected wavelength sweep: {wavelengths} nm")
                return wavelength_results, wavelengths
            else:
                return None, None
        # === END AUTO-DETECT ===
        
        def proceed_export():
            dialog.destroy()
            
            # Collect metadata from the most recent cycle measurement if available
            metadata = {
                'sample_id': 'unknown',
                'operator': 'User',
                'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'notes': 'Exported from Keithley GUI'
            }
            
            # Try to get sample_id from any measurement with metadata
            for data in synapse_data_storage:
                if 'metadata' in data and 'sample_id' in data['metadata']:
                    metadata['sample_id'] = data['metadata']['sample_id']
                    break
                elif 'params' in data and 'sample_id' in data['params']:
                    metadata['sample_id'] = data['params']['sample_id']
                    break
            
            # F7: only fall back to a GUI entry when the measurements
            # themselves carried no sample ID. This block used to run
            # unconditionally and overwrite the detected ID with whatever the
            # Cycle tab's widget happened to hold, so a Basic-mode export was
            # always labelled with the Cycle tab's sample name.
            if metadata['sample_id'] == 'unknown':
                for widget_name, widget in (('cycle', globals().get('cycle_sample_id_entry')),
                                            ('basic', globals().get('sample_id_entry'))):
                    if widget is None:
                        continue
                    try:
                        candidate = widget.get().strip()
                    except Exception:
                        continue
                    if candidate:
                        metadata['sample_id'] = candidate
                        metadata['sample_id_source'] = f'{widget_name} tab entry'
                        break
            
            # Auto-detect wavelength sweep
            wavelength_results, wavelengths = auto_detect_wavelength_sweep()
            
            # Extract stimulus parameters from the first available measurement.
            #
            # F6: this block wrote wrong values under right-sounding keys.
            #   - 'intensity' received params['stim_level'], the stimulus
            #     VOLTAGE in volts, into the field the fitter reads as optical
            #     intensity in mW/cm² — while the correct intensity_mW_cm2 sat
            #     unused in the same dict.
            #   - 'pulse_width_ms' and 'frequency_Hz' looked up 'pulse_width_ms'
            #     and 'pulse_period_ms', which pulse-read params do not have
            #     (they are 'stim_width_ms' and 'stim_period_ms'), so both
            #     always fell to their hardcoded 100 and 10.
            #   - 'stimulus_type' received params['mode'] — 'electrical' or
            #     'visual' — rather than a stimulus kind.
            #   - 'wavelength_pot' defaulted to a hardcoded 550 nm.
            # Every key is now read from the name the engine actually uses, and
            # nothing is invented: a value that was not measured is omitted, so
            # the fitter can tell it is absent instead of fitting a fabricated
            # number.
            stimulus_params = {}
            for data in synapse_data_storage:
                p = data.get('params')
                if not p:
                    continue

                mode = p.get('mode', '')
                stimulus_params = {
                    'stimulus_type': 'light' if mode == 'visual' else 'electrical',
                    'measurement_mode': mode,
                }

                # Canonical key names. These used to be 'intensity' (unit-less)
                # and 'wavelength_pot' (branch pre-decided) — neither of which
                # any consumer read: assembly_helpers tests for
                # 'wavelength_nm', and fitting.py reads
                # 'light_intensity_mW_cm2'. A full spectral sweep therefore
                # exported with no wavelength and no intensity that the fitter
                # could see, and the wavelength-response extraction silently
                # had nothing to work with.
                if p.get('intensity_mW_cm2') is not None:
                    stimulus_params['light_intensity_mW_cm2'] = p['intensity_mW_cm2']
                if p.get('wavelength_nm') is not None:
                    stimulus_params['wavelength_nm'] = p['wavelength_nm']
                if p.get('stim_width_ms') is not None:
                    stimulus_params['pulse_width_ms'] = p['stim_width_ms']
                if p.get('stim_period_ms'):
                    stimulus_params['frequency_Hz'] = 1000.0 / p['stim_period_ms']
                if p.get('n_pulses') is not None:
                    stimulus_params['n_pulses'] = p['n_pulses']
                if p.get('stim_level') is not None:
                    # Recorded under its own name, in its own units — not as
                    # "intensity".
                    stimulus_params['stim_level'] = p['stim_level']
                    stimulus_params['stim_drive_type'] = p.get('stim_drive_type', 'V')
                break
            
            # Build characterization suite
            try:
                def _pick(idx):
                    return synapse_data_storage[idx] if idx is not None else None

                suite = assembly_helpers.build_characterization_suite(
                    wavelength_results=wavelength_results,
                    wavelengths=wavelengths,
                    ltp_result=_pick(ltp_idx),
                    ltd_result=_pick(ltd_idx),
                    cycle_results=_pick(cycle_idx),
                    retention_result=_pick(retention_idx),
                    stdp_result=_pick(stdp_idx),
                    srdp_result=_pick(srdp_idx),
                    stimulus_params=stimulus_params,
                    metadata=metadata
                )

                # Validate suite
                assembly_helpers.validate_suite(suite)

                # Determine what was included
                included = []
                if wavelength_results:
                    included.append("wavelength_response")
                if ltp_idx is not None:
                    included.append("nonlinearity (alpha)")
                if ltd_idx is not None:
                    included.append("depression nonlinearity (beta)")
                if cycle_idx is not None:
                    included.append("dynamic_range")
                if retention_idx is not None:
                    included.append("retention")
                if stdp_idx is not None:
                    included.append("STDP window")
                if srdp_idx is not None:
                    included.append("SRDP response")
                
                # Save dialog
                default_name = f"{metadata['sample_id']}_characterization.json"
                filename = asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json")],
                    initialfile=default_name,
                    title="Save Characterization Suite"
                )
                
                if filename:
                    assembly_helpers.save_characterization_suite(suite, filename)
                    messagebox.showinfo(
                        "Export Successful!", 
                        f"Characterization suite exported!\n\n"
                        f"File: {os.path.basename(filename)}\n"
                        f"Sample: {metadata['sample_id']}\n"
                        f"Datasets included: {', '.join(included)}\n\n"
                        f"This file can now be used with:\n"
                        f"  • fitting.py (extract_synapse_model)\n"
                        f"  • network.py (load into simulations)"
                    )
            
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export suite:\n\n{str(e)}")
        
        # Add buttons
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=10)
        
        Button(button_frame, text="Mark as LTP (α)", command=mark_as_ltp, bg="#2196F3", fg="white", width=18).pack(side="left", padx=3)
        Button(button_frame, text="Mark as LTD (β)", command=mark_as_ltd, bg="#3F51B5", fg="white", width=18).pack(side="left", padx=3)
        Button(button_frame, text="Mark as Cycles (G_min/G_max)", command=mark_as_cycles, bg="#4CAF50", fg="white", width=24).pack(side="left", padx=3)

        button_frame2 = ctk.CTkFrame(dialog)
        button_frame2.pack(pady=4)
        Button(button_frame2, text="Mark as Retention (τ)", command=mark_as_retention, bg="#FF9800", fg="white", width=20).pack(side="left", padx=3)
        Button(button_frame2, text="Mark as STDP", command=mark_as_stdp, bg="#9C27B0", fg="white", width=18).pack(side="left", padx=3)
        Button(button_frame2, text="Mark as SRDP", command=mark_as_srdp, bg="#00897B", fg="white", width=18).pack(side="left", padx=3)


        Button(dialog, text="Export Suite", command=proceed_export, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=20)
        Button(dialog, text="Cancel", command=dialog.destroy, bg="#F44336", fg="white", width=20).pack()
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Error during export:\n\n{str(e)}")
        


def save_srdp_data(results, filename):
    """
    Saves SRDP characterization data to CSV.
    
    Args:
        results (dict): SRDP results from measure_srdp or simulate_srdp
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata header
    params = results["base_params"]
    metadata_line = "# SRDP Characterization # " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)
    
    # Column headers
    csv_content.append("frequency_Hz,delta_G_S,delta_G_percent,G_initial_S,G_final_S")
    
    # Data rows
    for i in range(len(results["frequencies_hz"])):
        row = [
            f"{results['frequencies_hz'][i]:.4f}",
            f"{results['delta_g_S'][i]:.6e}",
            f"{results['delta_g_percent'][i]:.4f}",
            f"{results['g_initial_S'][i]:.6e}",
            f"{results['g_final_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))


def save_stdp_data(results, filename):
    """
    Saves STDP characterization data to CSV.
    
    Args:
        results (dict): STDP results from measure_stdp or simulate_stdp
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata header
    params = results["base_params"]
    metadata_line = "# STDP Characterization # " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)
    
    # Column headers
    csv_content.append("delta_t_ms,delta_G_S,delta_G_percent,G_initial_S,G_final_S")
    
    # Data rows
    for i in range(len(results["delta_t_ms"])):
        row = [
            f"{results['delta_t_ms'][i]:.4f}",
            f"{results['delta_g_S'][i]:.6e}",
            f"{results['delta_g_percent'][i]:.4f}",
            f"{results['g_initial_S'][i]:.6e}",
            f"{results['g_final_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))


def toggle_mode(event=None):
    if mode_selection.get() == "Transistor":
        gate_start_voltage.grid()
        gate_end_voltage.grid()
        gate_voltage_step.grid()
    else:
        gate_start_voltage.grid_remove()
        gate_end_voltage.grid_remove()
        gate_voltage_step.grid_remove()
        
def toggle_multi_device_mode():
    """Enables/disables multi-device controls."""
    if multi_device_var.get():
        n_devices_entry.configure(state="normal")
        inter_device_delay_entry.configure(state="normal")
        device_name_pattern_entry.configure(state="normal")
    else:
        n_devices_entry.configure(state="disabled")
        inter_device_delay_entry.configure(state="disabled")
        device_name_pattern_entry.configure(state="disabled")


def apply_preset(preset_name):
    """Applies predefined parameter presets for common synapse measurements."""
    if preset_name == "LTP Moderate":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "100")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "200")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "50")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "10")
        
    elif preset_name == "LTD Moderate":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "-1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "100")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "200")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "50")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "10")
        
    elif preset_name == "PPF Test":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.5")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "10")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "30")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "2")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "5")
        
    elif preset_name == "High Speed":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "10")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "20")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "100")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "2")


def _set_entry(entry, value):
    """Helper: clear and set an entry widget's value."""
    entry.delete(0, 'end')
    entry.insert(0, str(value))


def _apply_train_preset(config, topo_var, ch_var, write_ch_var, read_ch_var,
                         stim_level_entry, stim_width_entry, period_entry,
                         n_pulses_entry, read_voltage_entry, read_delay_entry,
                         update_topo_func):
    """Populates a train's GUI fields from a config dict."""
    topo = config.get('topology', 'single')
    if topo == 'single':
        topo_var.set("Single SMU")
        ch_var.set(config.get('write_ch', 'smua'))
    else:
        topo_var.set("Dual SMU (Write+Read)")
        write_ch_var.set(config.get('write_ch', 'smua'))
        read_ch_var.set(config.get('read_ch', 'smub'))
    update_topo_func()

    _set_entry(stim_level_entry, config['stim_level'])
    _set_entry(stim_width_entry, config['stim_width_ms'])
    _set_entry(period_entry, config['stim_period_ms'])
    _set_entry(n_pulses_entry, config['n_pulses'])
    _set_entry(read_voltage_entry, config['read_voltage'])
    _set_entry(read_delay_entry, config.get('read_delay_ms', 10))


def apply_cycle_preset(preset_name):
    """Applies cycle parameter presets to the unified cycle GUI."""
    if preset_name == "Custom":
        return

    try:
        train_a, train_b, n_cycles, itd, icd = synapse_cycle.get_preset_parameters(preset_name)

        # Train A
        _apply_train_preset(
            train_a, cycle_train_a_topo_var, cycle_train_a_ch_var,
            cycle_train_a_write_ch_var, cycle_train_a_read_ch_var,
            train_a_stim_level_entry, train_a_stim_width_entry,
            train_a_period_entry, train_a_n_pulses_entry,
            train_a_read_voltage_entry, train_a_read_delay_entry,
            _update_train_a_topology)

        # Train B — enable it and populate
        cycle_enable_train_b_var.set(True)
        _toggle_train_b()
        _apply_train_preset(
            train_b, cycle_train_b_topo_var, cycle_train_b_ch_var,
            cycle_train_b_write_ch_var, cycle_train_b_read_ch_var,
            train_b_stim_level_entry, train_b_stim_width_entry,
            train_b_period_entry, train_b_n_pulses_entry,
            train_b_read_voltage_entry, train_b_read_delay_entry,
            _update_train_b_topology)

        # Cycle control
        _set_entry(cycle_n_cycles_entry, n_cycles)
        _set_entry(cycle_inter_train_delay_entry, itd)
        _set_entry(cycle_delay_entry, icd)

        messagebox.showinfo("Preset Applied", f"'{preset_name}' preset loaded successfully!")

    except Exception as e:
        messagebox.showerror("Preset Error", f"Failed to apply preset: {str(e)}")


# ==============================================================================
# GUI SETUP
# ==============================================================================

# Create main application window
root = ctk.CTk()
# Unhandled Tk callback exceptions become a visible dialog rather than a
# stderr traceback, so a measurement handler that raises cannot fail silently.
from gui_errors import install_tk_error_reporter
install_tk_error_reporter(root, "Shockingly Accurate IV")
root.title(f"Shockingly Accurate IV  v{__version__}")
root.geometry("1400x700")

# Main container with two columns
main_container = ctk.CTkFrame(root)
main_container.pack(fill="both", expand=True, padx=10, pady=10)

# Left panel for inputs
left_panel = ctk.CTkScrollableFrame(main_container, width=350)
left_panel.pack(side="left", fill="both", expand=False, padx=(0, 10))

# Right panel for graphs and parameters
right_panel = ctk.CTkFrame(main_container)
right_panel.pack(side="left", fill="both", expand=True)

# Create horizontal split: graph on left, sidebar on right
graph_and_results_container = ctk.CTkFrame(right_panel)
graph_and_results_container.pack(fill="both", expand=True)

# Right sidebar for results and logo
right_sidebar = ctk.CTkFrame(graph_and_results_container, width=300)
right_sidebar.pack(side="right", fill="both", padx=(10, 0))
right_sidebar.pack_propagate(False)  # Maintain fixed width

# Graph frame (top of right panel)
graph_frame = ctk.CTkFrame(graph_and_results_container)
graph_frame.pack(side="left", fill="both", expand=True)

# Parameters display frame (top of right sidebar) — fixed height so the
# runs panel below can expand.
params_frame = ctk.CTkFrame(right_sidebar, height=220)
params_frame.pack(fill="x", expand=False, pady=(0, 8))
params_frame.pack_propagate(False)
params_text = ctk.CTkTextbox(params_frame, height=200)
params_text.pack(fill="both", expand=True, padx=5, pady=5)

# --- Stored-runs management panel ---
# Synapse measurement runs accumulate in `synapse_data_storage` across the
# session (for batch CSV export). This panel surfaces that list so the user
# can delete individual runs or clear everything without closing the app.
runs_frame = ctk.CTkFrame(right_sidebar)
runs_frame.pack(fill="both", expand=True, pady=(0, 8))

_runs_header = ctk.CTkFrame(runs_frame, fg_color="transparent")
_runs_header.pack(fill="x", padx=5, pady=(5, 2))
runs_count_label = ctk.CTkLabel(
    _runs_header, text="Stored runs: 0",
    font=("Arial", 11, "bold"))
runs_count_label.pack(side="left")
clear_all_runs_button = ctk.CTkButton(
    _runs_header, text="Clear All", width=80, height=24,
    fg_color="#B71C1C", hover_color="#7F0000",
    font=("Arial", 10, "bold"),
    command=lambda: _clear_all_runs())
clear_all_runs_button.pack(side="right")

runs_scroll = ctk.CTkScrollableFrame(runs_frame, fg_color="transparent")
runs_scroll.pack(fill="both", expand=True, padx=5, pady=(0, 5))

# Card widgets are tracked alongside their results dict in (card, results) pairs
# so delete can find and remove both. Using dict→list rather than indexing
# into synapse_data_storage keeps refs stable across deletions.
_run_cards = []  # list of {'card': CTkFrame, 'results': dict}


def _summarize_run(results):
    """Inspect a results dict and return (title, subtitle) for the card."""
    # SRDP — has 'frequencies_hz' array
    if isinstance(results, dict) and 'frequencies_hz' in results:
        n = len(results.get('frequencies_hz', []))
        bp = results.get('base_params', {})
        return ("SRDP",
                f"{n} frequencies  ·  "
                f"{bp.get('stim_level', '?')} V  ·  "
                f"{bp.get('n_pulses', '?')} pulses/freq")
    # STDP — has 'delta_t_ms'
    if isinstance(results, dict) and 'delta_t_ms' in results and 'base_params' in results:
        n = len(results.get('delta_t_ms', []))
        bp = results.get('base_params', {})
        return ("STDP",
                f"{n} Δt points  ·  "
                f"{bp.get('stim_level', '?')} V  ·  "
                f"{bp.get('n_pulses', '?')} pairs")
    # Cycle mode — has 'cycles' list of dicts
    if isinstance(results, dict) and 'cycles' in results:
        cycles = results.get('cycles', [])
        return ("Cycle",
                f"{len(cycles)} cycle(s) completed")
    # Multi-device — has a 'devices' key
    if isinstance(results, dict) and 'devices' in results:
        return ("Multi-Device",
                f"{len(results.get('devices', []))} devices")
    # Visual / Basic / Memristor pulse-read — has 'params' with 'mode'
    if isinstance(results, dict) and 'params' in results:
        p = results['params']
        mode = synapse_mode_display(p.get('mode', 'Basic'))
        sl = p.get('stim_level', '?')
        sd = p.get('stim_drive_type', 'V')
        n = p.get('n_pulses', '?')
        sid = p.get('sample_id', '') or results.get('metadata', {}).get('sample_id', '')
        subtitle = f"{n} pulses  ·  {sl} {sd}"
        if sid and sid != 'unknown':
            subtitle += f"  ·  {sid}"
        return (mode, subtitle)
    return ("Run", "—")


def _refresh_runs_count():
    runs_count_label.configure(text=f"Stored runs: {len(_run_cards)}")


def _delete_one_run(entry):
    """Delete a single run card and remove its backing data via the stored callback."""
    try:
        entry['remove_cb']()
    except Exception:
        pass
    try:
        entry['card'].destroy()
    except Exception:
        pass
    if entry in _run_cards:
        _run_cards.remove(entry)
    _refresh_runs_count()


def _clear_all_runs():
    """Drop all stored runs (synapse, diode, transistor) after confirmation."""
    if not _run_cards:
        return
    if not messagebox.askyesno(
            "Clear all runs",
            f"Delete all {len(_run_cards)} stored runs?\n\n"
            f"This only clears in-memory results. Any already-saved files "
            f"on disk are untouched."):
        return
    for entry in list(_run_cards):
        try:
            entry['card'].destroy()
        except Exception:
            pass
    _run_cards.clear()
    synapse_data_storage.clear()
    jv_curves.clear()
    pv_parameters.clear()
    _refresh_runs_count()


def _create_run_card(title, subtitle, remove_cb):
    """Shared card builder used by every run-type register_* function."""
    run_number = len(_run_cards) + 1

    card = ctk.CTkFrame(runs_scroll, fg_color=("#F5F5F5", "#2B2B2B"))
    card.pack(fill="x", padx=2, pady=2)

    entry = {'card': card, 'remove_cb': remove_cb}
    _run_cards.append(entry)

    text_col = ctk.CTkFrame(card, fg_color="transparent")
    text_col.pack(side="left", fill="x", expand=True, padx=(6, 2), pady=4)
    ctk.CTkLabel(text_col,
                 text=f"#{run_number}  ·  {title}",
                 font=("Arial", 10, "bold"),
                 anchor="w").pack(fill="x")
    ctk.CTkLabel(text_col,
                 text=subtitle,
                 font=("Arial", 9),
                 text_color="#888888",
                 anchor="w").pack(fill="x")

    ctk.CTkButton(card, text="✕", width=26, height=26,
                  fg_color="#B71C1C", hover_color="#7F0000",
                  font=("Arial", 11, "bold"),
                  command=lambda e=entry: _delete_one_run(e)
                  ).pack(side="right", padx=4, pady=4)

    _refresh_runs_count()
    return entry


def register_synapse_run(results):
    """Append a synapse-mode result to storage and add its run-panel card."""
    synapse_data_storage.append(results)
    title, subtitle = _summarize_run(results)

    def remove_cb():
        try:
            synapse_data_storage.remove(results)
        except ValueError:
            pass

    _create_run_card(title, subtitle, remove_cb)


def register_diode_run(jv_entry, pv_entry=None):
    """Add a run-panel card for a solar cell JV run."""
    meas = jv_entry.get("Measurement Type", "")
    cell = jv_entry.get("Cell #", "")
    npts = len(jv_entry.get("Voltages", []))
    subtitle = f"{cell}  ·  {meas}  ·  {npts} pts"
    if pv_entry is not None:
        try:
            subtitle += f"  ·  PCE {float(pv_entry.get('PCE', 0)):.2f}%"
        except (TypeError, ValueError):
            pass

    def remove_cb():
        try:
            jv_curves.remove(jv_entry)
        except ValueError:
            pass
        if pv_entry is not None:
            try:
                pv_parameters.remove(pv_entry)
            except ValueError:
                pass

    _create_run_card("Diode JV", subtitle, remove_cb)


def register_transistor_run(jv_entries):
    """Add one card representing a full transistor sweep (one entry per V_GS step)."""
    entries_ref = list(jv_entries)
    n = len(entries_ref)
    if n:
        first_vgs = entries_ref[0].get("Gate Voltage (V)", None)
        last_vgs = entries_ref[-1].get("Gate Voltage (V)", None)
        if first_vgs is not None and last_vgs is not None:
            subtitle = f"{n} V_GS steps  ·  {first_vgs:g} → {last_vgs:g} V"
        else:
            subtitle = f"{n} V_GS steps"
    else:
        subtitle = "empty sweep"

    def remove_cb():
        for e in entries_ref:
            try:
                jv_curves.remove(e)
            except ValueError:
                pass

    _create_run_card("Transistor", subtitle, remove_cb)


# Logo frame (bottom of right sidebar)
logo_frame = ctk.CTkFrame(right_sidebar, height=150)
logo_frame.pack(fill="x", expand=False)
logo_frame.pack_propagate(False)  # Maintain fixed height

def resource_path(relative_path):
    """Get absolute path to resource (Python file or PyInstaller EXE)."""
    if hasattr(sys, "_MEIPASS"):
        # EXE: PyInstaller extracts files here
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # Normal Python: folder where script sits
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

try:
    logo_path = resource_path("logo.png")
    logo_image = ctk.CTkImage(
        light_image=Image.open(logo_path),
        dark_image=Image.open(logo_path),
        size=(180, 130)
    )
    logo_label = ctk.CTkLabel(logo_frame, image=logo_image, text="")
    logo_label.pack(expand=True, pady=10)
except Exception as e:
    print("Logo not found:", e)
    logo_label = ctk.CTkLabel(logo_frame, text="Logo", font=("Arial", 20))
    logo_label.pack(expand=True)


# ==============================================================================
# COMMON PARAMETERS (Always Visible)
# ==============================================================================

ctk.CTkLabel(left_panel, text="GENERAL PARAMETERS", font=("Arial", 14, "bold")).pack(pady=(5, 10))

# Connection Type
ctk.CTkLabel(left_panel, text="Connection Type:").pack(anchor="w", padx=10)
connection_type = ctk.CTkComboBox(left_panel, values=["GPIB", "RS232", "LAN"])
connection_type.pack(fill="x", padx=10, pady=5)
connection_type.set("GPIB")

# Port Address
ctk.CTkLabel(left_panel, text="Port Address:").pack(anchor="w", padx=10)
port_entry = ctk.CTkEntry(left_panel)
port_entry.pack(fill="x", padx=10, pady=5)

# Channel Selection
ctk.CTkLabel(left_panel, text="Channel:").pack(anchor="w", padx=10)
channel_selection = ctk.CTkComboBox(left_panel, values=["Channel A", "Channel B"])
channel_selection.pack(fill="x", padx=10, pady=5)
channel_selection.set("Channel A")

# Add clarification label for transistor mode
channel_info_label = ctk.CTkLabel(
    left_panel, 
    text="ℹ️ Selected channel = Drain, Other channel = Gate",
    font=("Arial", 12, "italic"),
    text_color="gray"
)
channel_info_label.pack(anchor="w", padx=10, pady=(0, 5))

# NPLC
# NPLC ships with the instrument's own default (1.0) rather than empty. Every
# synapse mode validates this field, so a blank default made a first run fail
# at the first click with no indication of what the field wanted, what its
# units were, or what range the hardware accepts. The range is now in the
# label, per the 2600-series manual (0.001-25; 1 PLC = 20 ms at 50 Hz).
ctk.CTkLabel(left_panel,
             text="Measurement Speed (NPLC, 0.001-25):").pack(anchor="w", padx=10)
nplc_entry = ctk.CTkEntry(left_panel)
nplc_entry.insert(0, "1")
nplc_entry.pack(fill="x", padx=10, pady=5)

# Mains line frequency.
#
# H16: CLAUDE.md listed `line_freq_hz` in both the params and train_config
# contracts, but the GUI had ZERO occurrences of it, so every measurement fell
# back to DEFAULT_LINE_FREQ_HZ = 50. Correct for Barcelona, and silently wrong
# for a 60 Hz user with no way to change it.
#
# NPLC integrates each reading over a whole number of MAINS cycles, which is
# what rejects line-frequency pickup. If this value does not match the actual
# supply, the aperture no longer spans an integer number of cycles and mains
# noise leaks into every reading.
ctk.CTkLabel(left_panel, text="Mains Frequency (Hz):").pack(anchor="w", padx=10)
LINE_FREQ_AUTO_LABEL = "Auto (detect)"
line_freq_var = ctk.StringVar(value=LINE_FREQ_AUTO_LABEL)
line_freq_combo = ctk.CTkComboBox(left_panel,
                                  values=[LINE_FREQ_AUTO_LABEL, "50", "60"],
                                  variable=line_freq_var)
line_freq_combo.pack(fill="x", padx=10, pady=5)


def get_line_freq_hz():
    """The mains frequency for the current measurement, validated.

    Returns None for the Auto setting (the default): the engine then adopts
    the frequency the instrument itself auto-detected at power-up, which is
    correct on any installation, 50 or 60 Hz grid alike. A fixed value forced
    the wrong sync on other grids (a 60 Hz user measured against a 50 Hz
    default and no NPLC setting could reject their line pickup).

    Raises rather than defaulting on a bad explicit value: an ADC integrating
    against the wrong mains frequency degrades every reading silently.
    """
    raw = line_freq_var.get().strip()
    if raw == LINE_FREQ_AUTO_LABEL:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"Mains Frequency must be 50 or 60 Hz, got {raw!r}")
    if value not in (50.0, 60.0):
        raise ValueError(
            f"Mains Frequency must be 50 or 60 Hz (the 2600-series can only "
            f"synchronise to these), got {value}"
        )
    return value

# Timeout
ctk.CTkLabel(left_panel, text="Timeout Duration (s):").pack(anchor="w", padx=10)
timeout_entry = ctk.CTkEntry(left_panel)
timeout_entry.pack(fill="x", padx=10, pady=5)
timeout_entry.insert(0, "30")

# Compliance
ctk.CTkLabel(left_panel, text="Compliance (A):").pack(anchor="w", padx=10)
compliance = ctk.CTkComboBox(left_panel, values=["100nA", "1µA", "10µA", "100µA", "1mA", "10mA", "100mA", "1A", "1.5A"])
compliance.pack(fill="x", padx=10, pady=5)
compliance.set("1mA")

# Cell #
ctk.CTkLabel(left_panel, text="Cell #:").pack(anchor="w", padx=10)
sample_name_entry = ctk.CTkEntry(left_panel)
sample_name_entry.pack(fill="x", padx=10, pady=5)

# Sample Surface Area
ctk.CTkLabel(left_panel, text="Cell Surface Area (cm²):").pack(anchor="w", padx=10)
surface_area = ctk.CTkEntry(left_panel)
surface_area.pack(fill="x", padx=10, pady=5)

# Wire mode (per SMU channel — each SMU's sense mode must match its physical wiring)
ctk.CTkLabel(left_panel, text="SMU A wire mode:").pack(anchor="w", padx=10)
wire_mode_smua = ctk.CTkComboBox(left_panel, values=["2-Wire", "4-Wire"])
wire_mode_smua.pack(fill="x", padx=10, pady=5)
wire_mode_smua.set("2-Wire")

ctk.CTkLabel(left_panel, text="SMU B wire mode:").pack(anchor="w", padx=10)
wire_mode_smub = ctk.CTkComboBox(left_panel, values=["2-Wire", "4-Wire"])
wire_mode_smub.pack(fill="x", padx=10, pady=5)
wire_mode_smub.set("2-Wire")


def get_wire_mode(channel):
    """Return '2-Wire' or '4-Wire' for the given SMU channel ('smua'/'smub', any case)."""
    ch = (channel or "").lower()
    if ch.endswith("a"):
        return wire_mode_smua.get()
    return wire_mode_smub.get()


def build_wire_modes():
    """Return the wire-mode dict consumed by synapse_engine and synapse_cycle."""
    return {"smua": wire_mode_smua.get(), "smub": wire_mode_smub.get()}

# Autorange
autorange_var = ctk.IntVar(value=0)
autorange_check = ctk.CTkCheckBox(left_panel, text="Enable Autorange", variable=autorange_var)
autorange_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(left_panel, text="─" * 50).pack(pady=10)

# ==============================================================================
# MODE SELECTION
# ==============================================================================

ctk.CTkLabel(left_panel, text="MEASUREMENT MODE", font=("Arial", 14, "bold")).pack(pady=(5, 10))

mode_selection = ctk.CTkComboBox(left_panel, values=["Diode", "Transistor", "Synapse"], command=lambda x: update_mode_display())
mode_selection.pack(fill="x", padx=10, pady=5)
mode_selection.set("Diode")

# ==============================================================================
# DIODE MODE PARAMETERS
# ==============================================================================

diode_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(diode_frame, text="DIODE PARAMETERS", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(diode_frame, text="Starting Voltage (V):").pack(anchor="w", padx=10)
start_voltage = ctk.CTkEntry(diode_frame)
start_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Ending Voltage (V):").pack(anchor="w", padx=10)
end_voltage = ctk.CTkEntry(diode_frame)
end_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Voltage Step (V):").pack(anchor="w", padx=10)
voltage_step = ctk.CTkEntry(diode_frame)
voltage_step.pack(fill="x", padx=10, pady=5)

hysteresis_var = ctk.IntVar()
hysteresis_check = ctk.CTkCheckBox(diode_frame, text="Perform Hysteresis", variable=hysteresis_var)
hysteresis_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Hysteresis Cycles:").pack(anchor="w", padx=10)
hysteresis_cycles = ctk.CTkEntry(diode_frame)
hysteresis_cycles.pack(fill="x", padx=10, pady=5)

dark_measurement = ctk.IntVar()
dark_check = ctk.CTkCheckBox(diode_frame, text="Dark JV", variable=dark_measurement)
dark_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Irradiance (W/m²):").pack(anchor="w", padx=10)
light_power_entry = ctk.CTkEntry(diode_frame)
light_power_entry.pack(fill="x", padx=10, pady=5)
light_power_entry.insert(0, "1000")

# Diode buttons
diode_button_frame = ctk.CTkFrame(diode_frame)
diode_button_frame.pack(fill="x", padx=10, pady=10)

run_diode_button = ctk.CTkButton(diode_button_frame, text="Run Measurement", command=run_measurement_buffered)
run_diode_button.pack(fill="x", pady=5)

save_diode_button = ctk.CTkButton(diode_button_frame, text="Save Data", command=save_curve)
save_diode_button.pack(fill="x", pady=5)

# ==============================================================================
# TRANSISTOR MODE PARAMETERS
# ==============================================================================

transistor_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(transistor_frame, text="TRANSISTOR PARAMETERS", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(transistor_frame, text="Starting Voltage (V):").pack(anchor="w", padx=10)
start_voltage_trans = ctk.CTkEntry(transistor_frame)
start_voltage_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Ending Voltage (V):").pack(anchor="w", padx=10)
end_voltage_trans = ctk.CTkEntry(transistor_frame)
end_voltage_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Voltage Step (V):").pack(anchor="w", padx=10)
voltage_step_trans = ctk.CTkEntry(transistor_frame)
voltage_step_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate Start Voltage (V):").pack(anchor="w", padx=10)
gate_start_voltage = ctk.CTkEntry(transistor_frame)
gate_start_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate End Voltage (V):").pack(anchor="w", padx=10)
gate_end_voltage = ctk.CTkEntry(transistor_frame)
gate_end_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate Voltage Step (V):").pack(anchor="w", padx=10)
gate_voltage_step = ctk.CTkEntry(transistor_frame)
gate_voltage_step.pack(fill="x", padx=10, pady=5)

# Transistor buttons
transistor_button_frame = ctk.CTkFrame(transistor_frame)
transistor_button_frame.pack(fill="x", padx=10, pady=10)

run_transistor_button = ctk.CTkButton(transistor_button_frame, text="Run Measurement", command=run_transistor_measurement)
run_transistor_button.pack(fill="x", pady=5)

save_transistor_button = ctk.CTkButton(transistor_button_frame, text="Save Data", command=save_transistor_data)
save_transistor_button.pack(fill="x", pady=5)

# ==============================================================================
# SYNAPSE MODE PARAMETERS
# ==============================================================================

synapse_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(synapse_frame, text="SYNAPSE MODE", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(synapse_frame, text="Synapse Sub-mode:").pack(anchor="w", padx=10)
synapse_submode = ctk.CTkComboBox(synapse_frame, values=["Basic", "Visual (Self-Powered)", "SRDP", "STDP", "Cycle", "Multi-Device"],
                                   command=lambda x: update_synapse_submode())
synapse_submode.pack(fill="x", padx=10, pady=5)
synapse_submode.set("Basic")

# === CHANNEL SELECTION (Always visible for all synapse modes) ===
synapse_channel_frame = ctk.CTkFrame(synapse_frame)
synapse_channel_frame.pack(fill="x", padx=5, pady=5)

ctk.CTkLabel(synapse_channel_frame, text="─ Channel Selection ─", font=("Arial", 10, "bold")).pack(pady=(5, 5))

ctk.CTkLabel(synapse_channel_frame, text="Stim Channel (LED):").pack(anchor="w", padx=10)
stim_channel_var = ctk.CTkComboBox(synapse_channel_frame, values=["smua", "smub"])
stim_channel_var.pack(fill="x", padx=10, pady=5)
stim_channel_var.set("smua")

ctk.CTkLabel(synapse_channel_frame, text="Read Channel (Device):").pack(anchor="w", padx=10)
read_channel_var = ctk.CTkComboBox(synapse_channel_frame, values=["smua", "smub"])
read_channel_var.pack(fill="x", padx=10, pady=5)
read_channel_var.set("smub")

simulate_var = ctk.IntVar()
simulate_check = ctk.CTkCheckBox(synapse_channel_frame, text="Simulate (No Hardware)", variable=simulate_var)
simulate_check.pack(anchor="w", padx=10, pady=5)

# === COMMON PARAMETERS (conditionally rebuilt per submode) ===
synapse_common_params_frame = ctk.CTkFrame(synapse_frame)

# Save references to all labels so they can be conditionally shown/hidden
_common_title_label = ctk.CTkLabel(synapse_common_params_frame, text="─ Common Parameters ─", font=("Arial", 10, "bold"))

_synapse_type_label = ctk.CTkLabel(synapse_common_params_frame, text="Synapse Type:")
synapse_mode_selection = ctk.CTkComboBox(synapse_common_params_frame, values=["Electrical", "Visual", "Memristor (Pulse)"])
synapse_mode_selection.set("Electrical")

_stim_drive_label = ctk.CTkLabel(synapse_common_params_frame, text="Stim Drive Type:")
stim_drive_var = ctk.CTkComboBox(synapse_common_params_frame, values=["V", "I"])
stim_drive_var.set("V")

_read_voltage_label = ctk.CTkLabel(synapse_common_params_frame, text="Read Voltage (V):")
read_voltage_entry = ctk.CTkEntry(synapse_common_params_frame)
read_voltage_entry.insert(0, "0.1")

_read_delay_label = ctk.CTkLabel(synapse_common_params_frame, text="Read Delay (ms):")
read_delay_entry = ctk.CTkEntry(synapse_common_params_frame)
read_delay_entry.insert(0, "10")

_read_settle_delay_label = ctk.CTkLabel(synapse_common_params_frame, text="Read Settle Delay (ms):")
read_settle_delay_entry = ctk.CTkEntry(synapse_common_params_frame)
read_settle_delay_entry.insert(0, "5")

_read_width_label = ctk.CTkLabel(synapse_common_params_frame, text="Read Pulse Width (ms):")
read_width_entry = ctk.CTkEntry(synapse_common_params_frame)
read_width_entry.insert(0, "50")

_samples_per_pulse_label = ctk.CTkLabel(synapse_common_params_frame, text="Samples per Pulse (0 = auto):")
samples_per_pulse_entry = ctk.CTkEntry(synapse_common_params_frame)
samples_per_pulse_entry.insert(0, "0")

def _rebuild_common_params(submode):
    """Rebuild common params widget visibility based on synapse submode.

    Only called for Basic, SRDP, STDP (Cycle/Multi-Device/Visual are self-contained).
    Conditionally hides Stim Drive Type in STDP (hardcoded to V).
    """
    for w in synapse_common_params_frame.winfo_children():
        w.pack_forget()

    _common_title_label.pack(pady=(5, 5))

    _synapse_type_label.pack(anchor="w", padx=10)
    synapse_mode_selection.pack(fill="x", padx=10, pady=5)

    # Stim Drive Type: hidden in STDP (hardcoded to V)
    if submode != "STDP":
        _stim_drive_label.pack(anchor="w", padx=10)
        stim_drive_var.pack(fill="x", padx=10, pady=5)

    _read_voltage_label.pack(anchor="w", padx=10)
    read_voltage_entry.pack(fill="x", padx=10, pady=5)

    _read_delay_label.pack(anchor="w", padx=10)
    read_delay_entry.pack(fill="x", padx=10, pady=5)

    _read_settle_delay_label.pack(anchor="w", padx=10)
    read_settle_delay_entry.pack(fill="x", padx=10, pady=5)

    _read_width_label.pack(anchor="w", padx=10)
    read_width_entry.pack(fill="x", padx=10, pady=5)

    _samples_per_pulse_label.pack(anchor="w", padx=10)
    samples_per_pulse_entry.pack(fill="x", padx=10, pady=5)


def _read_samples_per_pulse():
    """Parse the Samples per Pulse entry. Returns an int, or 0 for auto.

    Raises ValueError on anything that is not a non-negative integer.

    0 is a LEGITIMATE value meaning "automatic averaging", so returning 0 on a
    parse failure — as this used to — made a typo indistinguishable from a
    deliberate choice: typing '1O' (letter O) for 10 silently selected auto,
    and neither the GUI nor the exported CSV recorded that anything had gone
    wrong. It also folded negatives into 0 via max(0, v) rather than objecting.
    """
    raw = samples_per_pulse_entry.get().strip()
    if not raw:
        return 0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"Samples per Pulse: {raw!r} is not a number. Enter 0 for "
            "automatic averaging, or a positive whole number of samples."
        )
    if value != int(value):
        raise ValueError(
            f"Samples per Pulse: {raw!r} is not a whole number. Enter 0 for "
            "automatic averaging, or a positive integer."
        )
    value = int(value)
    if value < 0:
        raise ValueError(
            f"Samples per Pulse: {value} is negative. Enter 0 for automatic "
            "averaging, or a positive integer."
        )
    return value


# Initial pack for default mode (Basic)
_rebuild_common_params("Basic")

# --- BASIC SYNAPSE SUBMODE ---
synapse_basic_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_basic_frame, text="─ Basic Synapse ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

preset_frame = ctk.CTkFrame(synapse_basic_frame)
preset_frame.pack(fill="x", padx=10, pady=5)
ctk.CTkLabel(preset_frame, text="Presets:").pack(anchor="w")
preset_var = ctk.CTkComboBox(preset_frame, values=["Custom", "LTP Moderate", "LTD Moderate", "PPF Test", "High Speed"],
                              command=lambda choice: apply_preset(choice))
preset_var.pack(fill="x")
preset_var.set("Custom")

ctk.CTkLabel(synapse_basic_frame, text="Stim Level (V or A):").pack(anchor="w", padx=10)
stim_level_entry = ctk.CTkEntry(synapse_basic_frame)
stim_level_entry.pack(fill="x", padx=10, pady=5)
stim_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_basic_frame, text="Stim Width (ms):").pack(anchor="w", padx=10)
stim_width_entry = ctk.CTkEntry(synapse_basic_frame)
stim_width_entry.pack(fill="x", padx=10, pady=5)
stim_width_entry.insert(0, "100")

ctk.CTkLabel(synapse_basic_frame, text="Stim Period (ms):").pack(anchor="w", padx=10)
stim_period_entry = ctk.CTkEntry(synapse_basic_frame)
stim_period_entry.pack(fill="x", padx=10, pady=5)
stim_period_entry.insert(0, "200")

ctk.CTkLabel(synapse_basic_frame, text="# Pulses:").pack(anchor="w", padx=10)
n_pulses_entry = ctk.CTkEntry(synapse_basic_frame)
n_pulses_entry.pack(fill="x", padx=10, pady=5)
n_pulses_entry.insert(0, "50")

# === MANUAL LIGHT SOURCE PARAMETERS ===
ctk.CTkLabel(synapse_basic_frame, text="─ Light Source (Manual Control) ─", 
             font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_basic_frame, text="Wavelength (nm):").pack(anchor="w", padx=10)
wavelength_entry = ctk.CTkEntry(synapse_basic_frame)
wavelength_entry.pack(fill="x", padx=10, pady=5)
wavelength_entry.insert(0, "550")  # Default green

ctk.CTkLabel(synapse_basic_frame, text="Light Intensity (mW/cm²):").pack(anchor="w", padx=10)
light_intensity_entry = ctk.CTkEntry(synapse_basic_frame)
light_intensity_entry.pack(fill="x", padx=10, pady=5)
light_intensity_entry.insert(0, "20")  # Default intensity

ctk.CTkLabel(synapse_basic_frame, text="Sample ID:").pack(anchor="w", padx=10)
sample_id_entry = ctk.CTkEntry(synapse_basic_frame)
sample_id_entry.pack(fill="x", padx=10, pady=5)
sample_id_entry.insert(0, "Sample_001")

synapse_basic_button_frame = ctk.CTkFrame(synapse_basic_frame)
synapse_basic_button_frame.pack(fill="x", padx=10, pady=10)

run_synapse_button = ctk.CTkButton(synapse_basic_button_frame, text="Run Measurement", command=run_synapse_mode, fg_color="#2E7D32", hover_color="#1B5E20")
run_synapse_button.pack(fill="x", pady=5)

save_synapse_button = ctk.CTkButton(synapse_basic_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_synapse_button.pack(fill="x", pady=5)

# --- SRDP SUBMODE ---
synapse_srdp_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_srdp_frame, text="─ SRDP Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_srdp_frame, text="Freq Start (Hz):").pack(anchor="w", padx=10)
srdp_freq_start_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_start_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_start_entry.insert(0, "1")

ctk.CTkLabel(synapse_srdp_frame, text="Freq End (Hz):").pack(anchor="w", padx=10)
srdp_freq_end_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_end_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_end_entry.insert(0, "100")

ctk.CTkLabel(synapse_srdp_frame, text="# Freq Points:").pack(anchor="w", padx=10)
srdp_freq_points_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_points_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_points_entry.insert(0, "10")

srdp_log_scale_var = ctk.IntVar(value=1)
srdp_log_scale_check = ctk.CTkCheckBox(synapse_srdp_frame, text="Log Scale Frequency", variable=srdp_log_scale_var)
srdp_log_scale_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(synapse_srdp_frame, text="Stim Level (V):").pack(anchor="w", padx=10)
srdp_stim_level_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_stim_level_entry.pack(fill="x", padx=10, pady=5)
srdp_stim_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_srdp_frame, text="Stim Width (ms):").pack(anchor="w", padx=10)
srdp_stim_width_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_stim_width_entry.pack(fill="x", padx=10, pady=5)
srdp_stim_width_entry.insert(0, "10")  # *** CHANGED FROM 100 TO 10 ***

ctk.CTkLabel(synapse_srdp_frame, text="# Pulses per freq:").pack(anchor="w", padx=10)
srdp_n_pulses_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_n_pulses_entry.pack(fill="x", padx=10, pady=5)
srdp_n_pulses_entry.insert(0, "50")

# === MANUAL LIGHT SOURCE PARAMETERS ===
ctk.CTkLabel(synapse_srdp_frame, text="─ Light Source (Manual Control) ─", 
             font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_srdp_frame, text="Wavelength (nm):").pack(anchor="w", padx=10)
srdp_wavelength_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_wavelength_entry.pack(fill="x", padx=10, pady=5)
srdp_wavelength_entry.insert(0, "550")

ctk.CTkLabel(synapse_srdp_frame, text="Light Intensity (mW/cm²):").pack(anchor="w", padx=10)
srdp_light_intensity_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_light_intensity_entry.pack(fill="x", padx=10, pady=5)
srdp_light_intensity_entry.insert(0, "20")

ctk.CTkLabel(synapse_srdp_frame, text="Sample ID:").pack(anchor="w", padx=10)
srdp_sample_id_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_sample_id_entry.pack(fill="x", padx=10, pady=5)
srdp_sample_id_entry.insert(0, "Sample_001")

synapse_srdp_button_frame = ctk.CTkFrame(synapse_srdp_frame)
synapse_srdp_button_frame.pack(fill="x", padx=10, pady=10)

run_srdp_button = ctk.CTkButton(synapse_srdp_button_frame, text="Run SRDP", command=run_srdp_characterization, fg_color="#FF6F00", hover_color="#E65100")
run_srdp_button.pack(fill="x", pady=5)

save_srdp_button = ctk.CTkButton(synapse_srdp_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_srdp_button.pack(fill="x", pady=5)

# --- STDP SUBMODE ---
synapse_stdp_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_stdp_frame, text="─ STDP Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_stdp_frame, text="Δt Start (ms):").pack(anchor="w", padx=10)
stdp_dt_start_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_start_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_start_entry.insert(0, "-50")

ctk.CTkLabel(synapse_stdp_frame, text="Δt End (ms):").pack(anchor="w", padx=10)
stdp_dt_end_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_end_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_end_entry.insert(0, "50")

ctk.CTkLabel(synapse_stdp_frame, text="# Δt Points:").pack(anchor="w", padx=10)
stdp_dt_points_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_points_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_points_entry.insert(0, "15")

ctk.CTkLabel(synapse_stdp_frame, text="# Spike Pairs:").pack(anchor="w", padx=10)
stdp_n_pairs_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_n_pairs_entry.pack(fill="x", padx=10, pady=5)
stdp_n_pairs_entry.insert(0, "50")

ctk.CTkLabel(synapse_stdp_frame, text="Pre-spike Level (V):").pack(anchor="w", padx=10)
stdp_pre_level_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pre_level_entry.pack(fill="x", padx=10, pady=5)
stdp_pre_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_stdp_frame, text="Post-spike Level (V):").pack(anchor="w", padx=10)
stdp_post_level_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_post_level_entry.pack(fill="x", padx=10, pady=5)
stdp_post_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_stdp_frame, text="Pulse Width (ms):").pack(anchor="w", padx=10)
stdp_pulse_width_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pulse_width_entry.pack(fill="x", padx=10, pady=5)
stdp_pulse_width_entry.insert(0, "10")

ctk.CTkLabel(synapse_stdp_frame, text="Pair Period (ms):").pack(anchor="w", padx=10)
stdp_pair_period_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pair_period_entry.pack(fill="x", padx=10, pady=5)
stdp_pair_period_entry.insert(0, "100")

# === MANUAL LIGHT SOURCE PARAMETERS ===
ctk.CTkLabel(synapse_stdp_frame, text="─ Light Source (Manual Control) ─", 
             font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_stdp_frame, text="Wavelength (nm):").pack(anchor="w", padx=10)
stdp_wavelength_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_wavelength_entry.pack(fill="x", padx=10, pady=5)
stdp_wavelength_entry.insert(0, "550")

ctk.CTkLabel(synapse_stdp_frame, text="Light Intensity (mW/cm²):").pack(anchor="w", padx=10)
stdp_light_intensity_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_light_intensity_entry.pack(fill="x", padx=10, pady=5)
stdp_light_intensity_entry.insert(0, "20")

ctk.CTkLabel(synapse_stdp_frame, text="Sample ID:").pack(anchor="w", padx=10)
stdp_sample_id_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_sample_id_entry.pack(fill="x", padx=10, pady=5)
stdp_sample_id_entry.insert(0, "Sample_001")

synapse_stdp_button_frame = ctk.CTkFrame(synapse_stdp_frame)
synapse_stdp_button_frame.pack(fill="x", padx=10, pady=10)

run_stdp_button = ctk.CTkButton(synapse_stdp_button_frame, text="Run STDP", command=run_stdp_characterization, fg_color="#7B1FA2", hover_color="#4A148C")
run_stdp_button.pack(fill="x", pady=5)

save_stdp_button = ctk.CTkButton(synapse_stdp_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_stdp_button.pack(fill="x", pady=5)

# --- CYCLE SUBMODE (Unified LUA Engine) ---
synapse_cycle_frame = ctk.CTkFrame(synapse_frame)

# === PRESETS ===
cycle_preset_frame = ctk.CTkFrame(synapse_cycle_frame)
cycle_preset_frame.pack(fill="x", padx=10, pady=5)
ctk.CTkLabel(cycle_preset_frame, text="Cycle Presets:").pack(anchor="w")
cycle_preset_var = ctk.CTkComboBox(
    cycle_preset_frame,
    values=["Custom", "Standard Cycle", "Fast Cycle", "High Endurance", "Asymmetric"],
    command=lambda choice: apply_cycle_preset(choice))
cycle_preset_var.pack(fill="x")
cycle_preset_var.set("Custom")


# --- Helper: build a train parameter sub-frame ---
def _build_train_frame(parent, label, default_stim, default_ch="smua"):
    """Creates a train parameter frame and returns a dict of its widgets."""
    frame = ctk.CTkFrame(parent)
    w = {}

    ctk.CTkLabel(frame, text=f"-- {label} --", font=("Arial", 10, "bold")).pack(pady=(5, 2))

    # Channel topology
    ctk.CTkLabel(frame, text="Channel Mode:").pack(anchor="w", padx=10)
    w['topo_var'] = ctk.CTkComboBox(frame, values=["Single SMU", "Dual SMU (Write+Read)"])
    w['topo_var'].pack(fill="x", padx=10, pady=2)
    w['topo_var'].set("Single SMU")

    # Single-channel selector
    w['single_frame'] = ctk.CTkFrame(frame)
    w['single_frame'].pack(fill="x", padx=10, pady=2)
    ctk.CTkLabel(w['single_frame'], text="Channel:").pack(anchor="w")
    w['ch_var'] = ctk.CTkComboBox(w['single_frame'], values=["smua", "smub"])
    w['ch_var'].pack(fill="x")
    w['ch_var'].set(default_ch)

    # Dual-channel selectors (hidden by default)
    w['dual_frame'] = ctk.CTkFrame(frame)
    ctk.CTkLabel(w['dual_frame'], text="Write Channel:").pack(anchor="w")
    w['write_ch_var'] = ctk.CTkComboBox(w['dual_frame'], values=["smua", "smub"])
    w['write_ch_var'].pack(fill="x")
    w['write_ch_var'].set("smua")
    ctk.CTkLabel(w['dual_frame'], text="Read Channel:").pack(anchor="w")
    w['read_ch_var'] = ctk.CTkComboBox(w['dual_frame'], values=["smua", "smub"])
    w['read_ch_var'].pack(fill="x")
    w['read_ch_var'].set("smub")

    # Pulse parameters
    ctk.CTkLabel(frame, text="Write Level (V):").pack(anchor="w", padx=10)
    w['stim_level'] = ctk.CTkEntry(frame)
    w['stim_level'].pack(fill="x", padx=10, pady=2)
    w['stim_level'].insert(0, str(default_stim))

    ctk.CTkLabel(frame, text="Write Width (ms):").pack(anchor="w", padx=10)
    w['stim_width'] = ctk.CTkEntry(frame)
    w['stim_width'].pack(fill="x", padx=10, pady=2)
    w['stim_width'].insert(0, "100")

    ctk.CTkLabel(frame, text="Read Voltage (V):").pack(anchor="w", padx=10)
    w['read_voltage'] = ctk.CTkEntry(frame)
    w['read_voltage'].pack(fill="x", padx=10, pady=2)
    w['read_voltage'].insert(0, "0.1")

    ctk.CTkLabel(frame, text="Read Delay (ms):").pack(anchor="w", padx=10)
    w['read_delay'] = ctk.CTkEntry(frame)
    w['read_delay'].pack(fill="x", padx=10, pady=2)
    w['read_delay'].insert(0, "10")

    ctk.CTkLabel(frame, text="Period (ms):").pack(anchor="w", padx=10)
    w['period'] = ctk.CTkEntry(frame)
    w['period'].pack(fill="x", padx=10, pady=2)
    w['period'].insert(0, "200")

    ctk.CTkLabel(frame, text="# Pulses:").pack(anchor="w", padx=10)
    w['n_pulses'] = ctk.CTkEntry(frame)
    w['n_pulses'].pack(fill="x", padx=10, pady=2)
    w['n_pulses'].insert(0, "50")

    return frame, w


# === TRAIN A (Potentiation) ===
cycle_train_a_frame, _ta_w = _build_train_frame(synapse_cycle_frame, "Train A (Potentiation)", 1.0, "smua")
cycle_train_a_frame.pack(fill="x", padx=5, pady=5)

# Expose Train A widgets as module-level names for function access
cycle_train_a_topo_var = _ta_w['topo_var']
cycle_train_a_ch_var = _ta_w['ch_var']
cycle_train_a_write_ch_var = _ta_w['write_ch_var']
cycle_train_a_read_ch_var = _ta_w['read_ch_var']
train_a_stim_level_entry = _ta_w['stim_level']
train_a_stim_width_entry = _ta_w['stim_width']
train_a_read_voltage_entry = _ta_w['read_voltage']
train_a_read_delay_entry = _ta_w['read_delay']
train_a_period_entry = _ta_w['period']
train_a_n_pulses_entry = _ta_w['n_pulses']
_ta_single_frame = _ta_w['single_frame']
_ta_dual_frame = _ta_w['dual_frame']


def _update_train_a_topology(*args):
    """Update Train A channel widgets and enforce mutual exclusion with Train B."""
    if cycle_train_a_topo_var.get() == "Single SMU":
        _ta_dual_frame.pack_forget()
        _ta_single_frame.pack(fill="x", padx=10, pady=2,
                              after=cycle_train_a_topo_var)
        # Single SMU: Train B becomes available again
        cycle_enable_train_b_check.configure(state="normal")
    else:
        # Dual SMU: both SMUs occupied → Train B impossible
        _ta_single_frame.pack_forget()
        _ta_dual_frame.pack(fill="x", padx=10, pady=2,
                            after=cycle_train_a_topo_var)
        cycle_enable_train_b_var.set(False)
        _toggle_train_b()
        cycle_enable_train_b_check.configure(state="disabled")

cycle_train_a_topo_var.configure(command=lambda *a: _update_train_a_topology())


# === ENABLE TRAIN B CHECKBOX ===
cycle_enable_train_b_var = ctk.BooleanVar(value=True)
cycle_enable_train_b_check = ctk.CTkCheckBox(
    synapse_cycle_frame, text="Enable Train B (Depression)",
    variable=cycle_enable_train_b_var,
    command=lambda: _toggle_train_b())
cycle_enable_train_b_check.pack(anchor="w", padx=10, pady=(5, 0))


# === TRAIN B (Depression) ===
cycle_train_b_frame, _tb_w = _build_train_frame(synapse_cycle_frame, "Train B (Depression)", -1.0, "smua")
cycle_train_b_frame.pack(fill="x", padx=5, pady=5)

cycle_train_b_topo_var = _tb_w['topo_var']
cycle_train_b_ch_var = _tb_w['ch_var']
cycle_train_b_write_ch_var = _tb_w['write_ch_var']
cycle_train_b_read_ch_var = _tb_w['read_ch_var']
train_b_stim_level_entry = _tb_w['stim_level']
train_b_stim_width_entry = _tb_w['stim_width']
train_b_read_voltage_entry = _tb_w['read_voltage']
train_b_read_delay_entry = _tb_w['read_delay']
train_b_period_entry = _tb_w['period']
train_b_n_pulses_entry = _tb_w['n_pulses']
_tb_single_frame = _tb_w['single_frame']
_tb_dual_frame = _tb_w['dual_frame']

# Train B is always Single SMU (enforced by interlocking)
cycle_train_b_topo_var.set("Single SMU")
cycle_train_b_topo_var.configure(state="disabled")

# Train B starts enabled → lock Train A topology to Single SMU too
cycle_train_a_topo_var.configure(state="disabled")


def _toggle_train_b():
    """Show/hide Train B and enforce topology interlocking."""
    if cycle_enable_train_b_var.get():
        cycle_train_b_frame.pack(fill="x", padx=5, pady=5,
                                 after=cycle_enable_train_b_check)
        # Two trains → both must be Single SMU
        cycle_train_a_topo_var.set("Single SMU")
        _update_train_a_topology()
        cycle_train_a_topo_var.configure(state="disabled")
    else:
        cycle_train_b_frame.pack_forget()
        # Single train → Train A free to choose topology
        cycle_train_a_topo_var.configure(state="normal")


# === CYCLE CONTROL ===
cycle_ctrl_frame = ctk.CTkFrame(synapse_cycle_frame)
cycle_ctrl_frame.pack(fill="x", padx=5, pady=5)
ctk.CTkLabel(cycle_ctrl_frame, text="-- Cycle Control --", font=("Arial", 10, "bold")).pack(pady=(5, 2))

ctk.CTkLabel(cycle_ctrl_frame, text="# Cycles:").pack(anchor="w", padx=10)
cycle_n_cycles_entry = ctk.CTkEntry(cycle_ctrl_frame)
cycle_n_cycles_entry.pack(fill="x", padx=10, pady=2)
cycle_n_cycles_entry.insert(0, "5")

ctk.CTkLabel(cycle_ctrl_frame, text="Inter-train Delay (ms):").pack(anchor="w", padx=10)
cycle_inter_train_delay_entry = ctk.CTkEntry(cycle_ctrl_frame)
cycle_inter_train_delay_entry.pack(fill="x", padx=10, pady=2)
cycle_inter_train_delay_entry.insert(0, "500")

ctk.CTkLabel(cycle_ctrl_frame, text="Inter-cycle Delay (ms):").pack(anchor="w", padx=10)
cycle_delay_entry = ctk.CTkEntry(cycle_ctrl_frame)
cycle_delay_entry.pack(fill="x", padx=10, pady=2)
cycle_delay_entry.insert(0, "1000")

# === SAMPLE ID ===
ctk.CTkLabel(cycle_ctrl_frame, text="Sample ID:").pack(anchor="w", padx=10)
cycle_sample_id_entry = ctk.CTkEntry(cycle_ctrl_frame)
cycle_sample_id_entry.pack(fill="x", padx=10, pady=2)
cycle_sample_id_entry.insert(0, "Sample_001")

# === BUTTONS ===
synapse_cycle_button_frame = ctk.CTkFrame(synapse_cycle_frame)
synapse_cycle_button_frame.pack(fill="x", padx=10, pady=10)

run_cycle_button = ctk.CTkButton(
    synapse_cycle_button_frame, text="Run Cycles",
    command=run_cycle_characterization, fg_color="#C62828", hover_color="#8E0000")
run_cycle_button.pack(fill="x", pady=5)

save_cycle_button = ctk.CTkButton(
    synapse_cycle_button_frame, text="Save Data",
    command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_cycle_button.pack(fill="x", pady=5)

export_suite_button = ctk.CTkButton(
    synapse_cycle_button_frame,
    text="Export Characterization Suite",
    command=export_characterization_suite,
    fg_color="#2E7D32", hover_color="#1B5E20")
export_suite_button.pack(fill="x", pady=5)


# --- MULTI-DEVICE SUBMODE ---
synapse_multidevice_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_multidevice_frame, text="─ Multi-Device Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

multi_device_var = ctk.IntVar()
multi_device_check = ctk.CTkCheckBox(synapse_multidevice_frame, text="Enable Multi-Device Mode", variable=multi_device_var,
                                      command=toggle_multi_device_mode)
multi_device_check.pack(anchor="w", padx=10, pady=5)

# === MANUAL LIGHT SOURCE PARAMETERS (Multi-Device) ===
ctk.CTkLabel(synapse_multidevice_frame, text="─ Light Source (Manual Control) ─", 
             font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_multidevice_frame, text="Wavelength (nm):").pack(anchor="w", padx=10)
multidevice_wavelength_entry = ctk.CTkEntry(synapse_multidevice_frame)
multidevice_wavelength_entry.pack(fill="x", padx=10, pady=5)
multidevice_wavelength_entry.insert(0, "550")

ctk.CTkLabel(synapse_multidevice_frame, text="Light Intensity (mW/cm²):").pack(anchor="w", padx=10)
multidevice_light_intensity_entry = ctk.CTkEntry(synapse_multidevice_frame)
multidevice_light_intensity_entry.pack(fill="x", padx=10, pady=5)
multidevice_light_intensity_entry.insert(0, "20")

ctk.CTkLabel(synapse_multidevice_frame, text="Sample ID Prefix:").pack(anchor="w", padx=10)
multidevice_sample_id_entry = ctk.CTkEntry(synapse_multidevice_frame)
multidevice_sample_id_entry.pack(fill="x", padx=10, pady=5)
multidevice_sample_id_entry.insert(0, "Batch_001")

ctk.CTkLabel(synapse_multidevice_frame, text="# Devices:").pack(anchor="w", padx=10)
n_devices_entry = ctk.CTkEntry(synapse_multidevice_frame)
n_devices_entry.pack(fill="x", padx=10, pady=5)
n_devices_entry.insert(0, "2")
n_devices_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="Inter-device Delay (ms):").pack(anchor="w", padx=10)
inter_device_delay_entry = ctk.CTkEntry(synapse_multidevice_frame)
inter_device_delay_entry.pack(fill="x", padx=10, pady=5)
inter_device_delay_entry.insert(0, "2000")
inter_device_delay_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="Device Name Pattern:").pack(anchor="w", padx=10)
device_name_pattern_entry = ctk.CTkEntry(synapse_multidevice_frame)
device_name_pattern_entry.pack(fill="x", padx=10, pady=5)
device_name_pattern_entry.insert(0, "Device")
device_name_pattern_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="(Uses Cycle parameters above)", font=("Arial", 9, "italic")).pack(pady=5)

synapse_multidevice_button_frame = ctk.CTkFrame(synapse_multidevice_frame)
synapse_multidevice_button_frame.pack(fill="x", padx=10, pady=10)

run_multidevice_button = ctk.CTkButton(synapse_multidevice_button_frame, text="Run Multi-Device", command=run_cycle_characterization, fg_color="#C62828", hover_color="#8E0000")
run_multidevice_button.pack(fill="x", pady=5)

save_multidevice_button = ctk.CTkButton(synapse_multidevice_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_multidevice_button.pack(fill="x", pady=5)

# --- VISUAL (SELF-POWERED) SUBMODE ---
synapse_visual_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_visual_frame, text="─ Visual (Self-Powered) Synapse ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_visual_frame,
             text="Measures photocurrent (Jsc) at short-circuit (0V)\nduring light pulses. The synaptic weight is\nencoded in the Jsc amplitude.",
             font=("Arial", 9, "italic"), justify="left").pack(pady=5, padx=10, anchor="w")

# Info box about channels
visual_info_frame = ctk.CTkFrame(synapse_visual_frame, fg_color="#2D2D44")
visual_info_frame.pack(fill="x", padx=10, pady=5)
ctk.CTkLabel(visual_info_frame, text="Stim Ch → LED driver | Read Ch → Device at 0V",
             font=("Arial", 9), text_color="#AAAAFF").pack(pady=5)

ctk.CTkLabel(synapse_visual_frame, text="─ Light Pulse Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_visual_frame, text="Light Pulse Voltage (V):").pack(anchor="w", padx=10)
visual_light_voltage_entry = ctk.CTkEntry(synapse_visual_frame)
visual_light_voltage_entry.pack(fill="x", padx=10, pady=5)
visual_light_voltage_entry.insert(0, "3.0")

ctk.CTkLabel(synapse_visual_frame, text="Light Pulse Width (ms):").pack(anchor="w", padx=10)
visual_pulse_width_entry = ctk.CTkEntry(synapse_visual_frame)
visual_pulse_width_entry.pack(fill="x", padx=10, pady=5)
visual_pulse_width_entry.insert(0, "10")

ctk.CTkLabel(synapse_visual_frame, text="Light Pulse Period (ms):").pack(anchor="w", padx=10)
visual_pulse_period_entry = ctk.CTkEntry(synapse_visual_frame)
visual_pulse_period_entry.pack(fill="x", padx=10, pady=5)
visual_pulse_period_entry.insert(0, "100")

ctk.CTkLabel(synapse_visual_frame, text="# Light Pulses:").pack(anchor="w", padx=10)
visual_n_pulses_entry = ctk.CTkEntry(synapse_visual_frame)
visual_n_pulses_entry.pack(fill="x", padx=10, pady=5)
visual_n_pulses_entry.insert(0, "50")

ctk.CTkLabel(synapse_visual_frame, text="─ Measurement Window ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_visual_frame, text="Jsc is measured DURING the light pulse.\nAdjust delays to avoid transients.",
             font=("Arial", 8, "italic"), text_color="gray").pack(anchor="w", padx=10)

ctk.CTkLabel(synapse_visual_frame, text="Start Delay (ms):").pack(anchor="w", padx=10)
visual_measure_start_entry = ctk.CTkEntry(synapse_visual_frame)
visual_measure_start_entry.pack(fill="x", padx=10, pady=5)
visual_measure_start_entry.insert(0, "1")

ctk.CTkLabel(synapse_visual_frame, text="End Margin (ms):").pack(anchor="w", padx=10)
visual_measure_end_entry = ctk.CTkEntry(synapse_visual_frame)
visual_measure_end_entry.pack(fill="x", padx=10, pady=5)
visual_measure_end_entry.insert(0, "1")

ctk.CTkLabel(synapse_visual_frame, text="Readings per Pulse (hw avg):").pack(anchor="w", padx=10)
visual_readings_per_pulse_entry = ctk.CTkEntry(synapse_visual_frame)
visual_readings_per_pulse_entry.pack(fill="x", padx=10, pady=5)
visual_readings_per_pulse_entry.insert(0, "5")

ctk.CTkLabel(synapse_visual_frame, text="─ Continuous Mode (Optional) ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

visual_continuous_var = ctk.IntVar()
visual_continuous_check = ctk.CTkCheckBox(synapse_visual_frame, text="Continuous I(t) Mode", variable=visual_continuous_var)
visual_continuous_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(synapse_visual_frame, text="Sample Interval (ms):").pack(anchor="w", padx=10)
visual_sample_interval_entry = ctk.CTkEntry(synapse_visual_frame)
visual_sample_interval_entry.pack(fill="x", padx=10, pady=5)
visual_sample_interval_entry.insert(0, "1")

ctk.CTkLabel(synapse_visual_frame, text="─ Metadata ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_visual_frame, text="Wavelength (nm):").pack(anchor="w", padx=10)
visual_wavelength_entry = ctk.CTkEntry(synapse_visual_frame)
visual_wavelength_entry.pack(fill="x", padx=10, pady=5)
visual_wavelength_entry.insert(0, "550")

ctk.CTkLabel(synapse_visual_frame, text="Light Intensity (mW/cm²):").pack(anchor="w", padx=10)
visual_intensity_entry = ctk.CTkEntry(synapse_visual_frame)
visual_intensity_entry.pack(fill="x", padx=10, pady=5)
visual_intensity_entry.insert(0, "20")

ctk.CTkLabel(synapse_visual_frame, text="Sample ID:").pack(anchor="w", padx=10)
visual_sample_id_entry = ctk.CTkEntry(synapse_visual_frame)
visual_sample_id_entry.pack(fill="x", padx=10, pady=5)
visual_sample_id_entry.insert(0, "Visual_Sample_001")

synapse_visual_button_frame = ctk.CTkFrame(synapse_visual_frame)
synapse_visual_button_frame.pack(fill="x", padx=10, pady=10)

run_visual_button = ctk.CTkButton(synapse_visual_button_frame, text="Run Visual Synapse", command=lambda: run_visual_synapse_mode(), fg_color="#7B1FA2", hover_color="#4A148C")
run_visual_button.pack(fill="x", pady=5)

save_visual_button = ctk.CTkButton(synapse_visual_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_visual_button.pack(fill="x", pady=5)

# ==============================================================================
# GRAPH SETUP
# ==============================================================================

plt.ioff()  # Turn off interactive mode to prevent empty window
# Three-axis layout. Positions are explicit so the schematic block stays
# pinned at exactly 30 % of figure height with tight internal spacing,
# regardless of figure size. ax_stim/ax_read are hidden when the active
# top-level mode is not Synapse, and the results axis is repositioned
# to fill the figure.
fig = plt.figure(figsize=(7.5, 5.4))

# Position presets ([left, bottom, width, height], figure fractions).
# Schematic block spans figure_y 0.64 → 0.94 (= 30 %), with a 0.10
# gap between the two schematic axes for title + subheader clearance.
# Results axis sits below from y=0.10 to y=0.55 in synapse mode (≈ 45 %),
# or fills the figure when the schematic is hidden.
_POS_AX_STIM           = [0.11, 0.78, 0.85, 0.13]   # top spine 0.91
_POS_AX_READ           = [0.11, 0.61, 0.85, 0.13]   # top spine 0.74
_POS_AX_RESULTS_TOP    = [0.11, 0.10, 0.85, 0.45]   # schematic block = 30 %
_POS_AX_RESULTS_FULL   = [0.10, 0.12, 0.86, 0.80]   # schematic hidden

ax_stim = fig.add_axes(_POS_AX_STIM)
ax_read = fig.add_axes(_POS_AX_READ, sharex=ax_stim)
ax      = fig.add_axes(_POS_AX_RESULTS_FULL)
ax.set_title("JV Curve")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current Density (mA/cm²)")
# Schematic is hidden until the user switches to Synapse mode.
ax_stim.set_visible(False)
ax_read.set_visible(False)
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)


# ==============================================================================
# PULSE SCHEMATIC — live preview of the synapse pulse train
# ==============================================================================

_schematic_after_id = None
_schematic_visible = False


def _show_schematic(show):
    """Toggle visibility of the pulse-schematic axes and reposition `ax`
    to either share the figure with the schematic (top 30 %) or fill it."""
    global _schematic_visible
    _schematic_visible = bool(show)
    ax_stim.set_visible(_schematic_visible)
    ax_read.set_visible(_schematic_visible)
    if _schematic_visible:
        ax_stim.set_position(_POS_AX_STIM)
        ax_read.set_position(_POS_AX_READ)
        ax.set_position(_POS_AX_RESULTS_TOP)
    else:
        ax.set_position(_POS_AX_RESULTS_FULL)
    canvas.draw_idle()


def _schedule_schematic_update(*args):
    """Debounced schematic redraw — coalesces key-by-key edits."""
    global _schematic_after_id
    if not _schematic_visible:
        return
    if _schematic_after_id is not None:
        try:
            root.after_cancel(_schematic_after_id)
        except Exception:
            pass
    _schematic_after_id = root.after(180, _do_schematic_update)


def _gather_schematic_params(submode):
    """
    Collect the current GUI values for the given submode into a dict shaped
    for pulse_schematic.render(). Missing / unparseable values are left out
    and the renderer falls back to its "enter valid parameters" placeholder.
    """
    def _e(entry):
        try:
            return entry.get()
        except Exception:
            return None

    if submode == 'Basic':
        return {
            'stim_ch': stim_channel_var.get(),
            'read_ch': read_channel_var.get(),
            'stim_drive_type': stim_drive_var.get(),
            'stim_level':  _e(stim_level_entry),
            'stim_width_ms':  _e(stim_width_entry),
            'stim_period_ms': _e(stim_period_entry),
            'read_voltage': _e(read_voltage_entry),
            'read_delay_ms': _e(read_delay_entry),
            'read_settle_delay_ms': _e(read_settle_delay_entry),
            'read_width_ms': _e(read_width_entry),
            'nplc': _e(nplc_entry),
        }

    if submode == 'SRDP':
        return {
            'stim_ch': stim_channel_var.get(),
            'read_ch': read_channel_var.get(),
            'stim_drive_type': stim_drive_var.get(),
            'stim_level':  _e(srdp_stim_level_entry),
            'stim_width_ms':  _e(srdp_stim_width_entry),
            'freq_start_hz': _e(srdp_freq_start_entry),
            'freq_end_hz':   _e(srdp_freq_end_entry),
            'log_scale': bool(srdp_log_scale_var.get()),
            'read_voltage': _e(read_voltage_entry),
            'read_delay_ms': _e(read_delay_entry),
            'read_settle_delay_ms': _e(read_settle_delay_entry),
            'read_width_ms': _e(read_width_entry),
            'nplc': _e(nplc_entry),
        }

    if submode == 'STDP':
        # Δt varies across the sweep — show the midpoint as a representative
        # pair so the user sees which regime (LTP/LTD) dominates their range.
        try:
            dt_start = float(_e(stdp_dt_start_entry))
            dt_end   = float(_e(stdp_dt_end_entry))
            dt_mid   = 0.5 * (dt_start + dt_end)
        except (TypeError, ValueError):
            dt_mid = None
        return {
            'stim_ch': stim_channel_var.get(),
            'read_ch': read_channel_var.get(),
            'pre_level':  _e(stdp_pre_level_entry),
            'post_level': _e(stdp_post_level_entry),
            'pulse_width_ms':  _e(stdp_pulse_width_entry),
            'pair_period_ms':  _e(stdp_pair_period_entry),
            'delta_t_ms': dt_mid,
        }

    if submode == 'Visual (Self-Powered)':
        return {
            'stim_ch': stim_channel_var.get(),
            'read_ch': read_channel_var.get(),
            'light_pulse_voltage': _e(visual_light_voltage_entry),
            'pulse_width_ms':  _e(visual_pulse_width_entry),
            'pulse_period_ms': _e(visual_pulse_period_entry),
            'measure_start_delay_ms': _e(visual_measure_start_entry),
            'measure_end_margin_ms':  _e(visual_measure_end_entry),
            'continuous_mode': bool(visual_continuous_var.get()),
            'nplc': _e(nplc_entry),
        }

    if submode in ('Cycle', 'Multi-Device'):
        def _train(topo_var, ch_var, write_var, read_var,
                   lvl_e, w_e, rv_e, rd_e, p_e, n_e):
            return {
                'topology':  topo_var.get(),
                'channel':   ch_var.get(),
                'write_ch':  write_var.get(),
                'read_ch':   read_var.get(),
                'stim_level':    _e(lvl_e),
                'stim_width_ms': _e(w_e),
                'read_voltage':  _e(rv_e),
                'read_delay_ms': _e(rd_e),
                'period_ms':     _e(p_e),
                'n_pulses':      _e(n_e),
                # The measurement-aperture inputs. _parse_cycle_train derives
                # the ADC window from exactly these, and the real run passes
                # them via _collect_train_config — but this preview omitted
                # all four, so the schematic always drew a 20 ms aperture
                # regardless of the NPLC typed. A user setting NPLC = 0.01
                # for fast cycling saw 20 ms and designed a period around it
                # while the instrument integrated 0.2 ms.
                'nplc':          _e(nplc_entry),
                'line_freq_hz':  get_line_freq_hz(),
                'settle_ms':     _e(read_settle_delay_entry),
                'measure_avg':   _e(samples_per_pulse_entry),
            }
        train_a = _train(cycle_train_a_topo_var, cycle_train_a_ch_var,
                         cycle_train_a_write_ch_var, cycle_train_a_read_ch_var,
                         train_a_stim_level_entry, train_a_stim_width_entry,
                         train_a_read_voltage_entry, train_a_read_delay_entry,
                         train_a_period_entry, train_a_n_pulses_entry)
        train_b = None
        if cycle_enable_train_b_var.get():
            train_b = _train(cycle_train_b_topo_var, cycle_train_b_ch_var,
                             cycle_train_b_write_ch_var, cycle_train_b_read_ch_var,
                             train_b_stim_level_entry, train_b_stim_width_entry,
                             train_b_read_voltage_entry, train_b_read_delay_entry,
                             train_b_period_entry, train_b_n_pulses_entry)
        return {
            'train_a': train_a,
            'train_b': train_b,
            'inter_train_delay_ms': _e(cycle_inter_train_delay_entry),
            'n_cycles': _e(cycle_n_cycles_entry),
        }

    return {}


def _do_schematic_update():
    """Run a scheduled schematic redraw."""
    global _schematic_after_id
    _schematic_after_id = None
    submode = synapse_submode.get()
    params = _gather_schematic_params(submode)
    pulse_schematic.render(ax_stim, ax_read, submode, params)
    canvas.draw_idle()


def _bind_entry_for_schematic(entry):
    """Bind <KeyRelease> + <FocusOut> to trigger a debounced schematic redraw."""
    if entry is None:
        return
    entry.bind("<KeyRelease>", _schedule_schematic_update, add="+")
    entry.bind("<FocusOut>",   _schedule_schematic_update, add="+")


def _wrap_combo_command(combo, extra_callback):
    """
    CTkComboBox allows a single `command` callback. Wrap any pre-existing
    command so we can attach our schematic trigger without clobbering it.
    """
    if combo is None:
        return
    try:
        existing = combo.cget('command')
    except Exception:
        existing = None

    def _wrapped(value, _ex=existing, _cb=extra_callback):
        if callable(_ex):
            try:
                _ex(value)
            except Exception:
                pass
        try:
            _cb()
        except Exception:
            pass

    combo.configure(command=_wrapped)

# ==============================================================================
# MODE DISPLAY LOGIC
# ==============================================================================

def update_mode_display():
    """Show/hide parameter frames based on selected mode."""
    mode = mode_selection.get()

    # Destroy lingering semilog plot from Diode IV curve
    if hasattr(graph_frame, "semilog_frame") and graph_frame.semilog_frame is not None:
        if hasattr(graph_frame, "_semilog_fig"):
            plt.close(graph_frame._semilog_fig)
            graph_frame._semilog_fig = None
        graph_frame.semilog_frame.destroy()
        graph_frame.semilog_frame = None

    # Hide all mode frames
    diode_frame.pack_forget()
    transistor_frame.pack_forget()
    synapse_frame.pack_forget()
    
    # Show selected mode frame
    if mode == "Diode":
        diode_frame.pack(fill="both", expand=True, pady=10)
        _show_schematic(False)
    elif mode == "Transistor":
        transistor_frame.pack(fill="both", expand=True, pady=10)
        # Link transistor entries to diode entries for shared parameters
        start_voltage_trans.delete(0, 'end')
        start_voltage_trans.insert(0, start_voltage.get() if start_voltage.get() else "")
        end_voltage_trans.delete(0, 'end')
        end_voltage_trans.insert(0, end_voltage.get() if end_voltage.get() else "")
        voltage_step_trans.delete(0, 'end')
        voltage_step_trans.insert(0, voltage_step.get() if voltage_step.get() else "")
        _show_schematic(False)
    elif mode == "Synapse":
        synapse_frame.pack(fill="both", expand=True, pady=10)
        update_synapse_submode()
        _show_schematic(True)
        _schedule_schematic_update()

def update_synapse_submode():
    """Show/hide synapse parameter frames based on selected submode."""
    submode = synapse_submode.get()

    # Hide all synapse submode frames
    synapse_common_params_frame.pack_forget()
    synapse_basic_frame.pack_forget()
    synapse_srdp_frame.pack_forget()
    synapse_stdp_frame.pack_forget()
    synapse_cycle_frame.pack_forget()
    synapse_multidevice_frame.pack_forget()
    synapse_visual_frame.pack_forget()

    # Show selected submode frame
    if submode == "Visual (Self-Powered)":
        # Visual mode has its own self-contained parameters
        synapse_visual_frame.pack(fill="both", expand=True, pady=5)
    elif submode in ("Cycle", "Multi-Device"):
        # Cycle mode is fully self-contained (per-train channel/params)
        synapse_cycle_frame.pack(fill="both", expand=True, pady=5)
        if submode == "Multi-Device":
            synapse_multidevice_frame.pack(fill="both", expand=True, pady=5)
    else:
        # Basic, SRDP, STDP share common params
        _rebuild_common_params(submode)
        synapse_common_params_frame.pack(fill="x", padx=5, pady=5)

        if submode == "Basic":
            synapse_basic_frame.pack(fill="both", expand=True, pady=5)
        elif submode == "SRDP":
            synapse_srdp_frame.pack(fill="both", expand=True, pady=5)
        elif submode == "STDP":
            synapse_stdp_frame.pack(fill="both", expand=True, pady=5)

    # Refresh the pulse schematic whenever the submode changes.
    _schedule_schematic_update()

# ==============================================================================
# HELPER FUNCTIONS FOR TRANSISTOR MODE
# ==============================================================================

def sync_transistor_to_diode():
    """Point the shared voltage-entry names at the TRANSISTOR widgets.

    `run_measurement_buffered` reads the module-level `start_voltage`,
    `end_voltage` and `voltage_step`, which normally refer to the Diode panel's
    entries. A transistor sweep must use the Transistor panel's own fields, so
    those names are redirected for the duration of the run.

    Despite the historical name, this does not copy anything — the copying is
    done separately by `update_mode_display`, which SEEDS the transistor
    entries from the diode ones when the user switches to Transistor mode. The
    two are complementary: the seed gives the transistor fields a sensible
    starting value, and this redirect makes the measurement read whatever the
    user then typed into them.

    Returns the previous bindings so the caller can restore them.
    """
    global start_voltage, end_voltage, voltage_step
    previous = (start_voltage, end_voltage, voltage_step)
    start_voltage = start_voltage_trans
    end_voltage = end_voltage_trans
    voltage_step = voltage_step_trans
    return previous


def restore_voltage_entry_bindings(previous):
    """Undo sync_transistor_to_diode."""
    global start_voltage, end_voltage, voltage_step
    start_voltage, end_voltage, voltage_step = previous


# ==============================================================================
# MODIFIED RUN FUNCTIONS TO HANDLE PARAMETER SOURCES
# ==============================================================================

# Wrap run_transistor_measurement to sync parameters first.
#
# The redirect is RESTORED afterwards. Left permanent — which is what the
# original did — the module-level `start_voltage` would go on pointing at the
# transistor widget after the first transistor run, so a subsequent DIODE sweep
# would silently read the transistor panel's voltages. That never surfaced only
# because this wrapper was never called at all (H18); enabling it without the
# restore would have introduced the bug the moment the binding was fixed.
# `update_mode_display` also seeds the transistor entries from the diode ones
# by reading `start_voltage.get()`, so a permanent redirect would additionally
# have made that seed copy the transistor entry into itself.
original_run_transistor = run_transistor_measurement
def run_transistor_measurement():
    previous = sync_transistor_to_diode()
    try:
        original_run_transistor()
    finally:
        restore_voltage_entry_bindings(previous)

# Wrap run_srdp_characterization to use correct stim parameters
original_run_srdp = run_srdp_characterization
def run_srdp_characterization():
    # Temporarily update stim_level_entry reference for SRDP
    global stim_level_entry, stim_width_entry, n_pulses_entry
    old_stim_level = stim_level_entry
    old_stim_width = stim_width_entry
    old_n_pulses = n_pulses_entry
    
    stim_level_entry = srdp_stim_level_entry
    stim_width_entry = srdp_stim_width_entry
    n_pulses_entry = srdp_n_pulses_entry
    
    try:
        original_run_srdp()
    finally:
        stim_level_entry = old_stim_level
        stim_width_entry = old_stim_width
        n_pulses_entry = old_n_pulses


# H18: rebind the buttons to the wrappers.
#
# `command=` captures the FUNCTION OBJECT at button-creation time, and both
# buttons are created hundreds of lines above this point — so they held the
# originals and these wrappers were never called at all. The visible
# consequence was that `sync_transistor_to_diode()` never ran, so the
# Transistor panel's own voltage fields were ignored in favour of the Diode
# panel's; it produced correct results only by accident, via an unrelated
# pre-copy elsewhere.
#
# Rebinding here, immediately after the wrappers are defined, keeps the
# definition and the binding together so they cannot drift apart again.
run_transistor_button.configure(command=run_transistor_measurement)
run_srdp_button.configure(command=run_srdp_characterization)


# ==============================================================================
# BIND SYNAPSE PARAMETER WIDGETS TO THE PULSE-SCHEMATIC LIVE PREVIEW
# ==============================================================================
# Every entry/combo that affects the pulse train triggers a debounced redraw
# of the schematic. Chaining preserves any pre-existing combo command (e.g.
# the cycle train topology switchers).
_schematic_entries = [
    # Common (Basic/SRDP/STDP share these)
    read_voltage_entry, read_delay_entry, read_settle_delay_entry,
    read_width_entry, nplc_entry,
    # Basic
    stim_level_entry, stim_width_entry, stim_period_entry, n_pulses_entry,
    # SRDP
    srdp_freq_start_entry, srdp_freq_end_entry,
    srdp_stim_level_entry, srdp_stim_width_entry, srdp_n_pulses_entry,
    # STDP
    stdp_dt_start_entry, stdp_dt_end_entry,
    stdp_pre_level_entry, stdp_post_level_entry,
    stdp_pulse_width_entry, stdp_pair_period_entry, stdp_n_pairs_entry,
    # Visual
    visual_light_voltage_entry, visual_pulse_width_entry, visual_pulse_period_entry,
    visual_n_pulses_entry, visual_measure_start_entry, visual_measure_end_entry,
    # Cycle (both trains + controls)
    train_a_stim_level_entry, train_a_stim_width_entry,
    train_a_read_voltage_entry, train_a_read_delay_entry,
    train_a_period_entry, train_a_n_pulses_entry,
    train_b_stim_level_entry, train_b_stim_width_entry,
    train_b_read_voltage_entry, train_b_read_delay_entry,
    train_b_period_entry, train_b_n_pulses_entry,
    cycle_n_cycles_entry, cycle_inter_train_delay_entry,
]
for _e in _schematic_entries:
    _bind_entry_for_schematic(_e)

_schematic_combos = [
    stim_drive_var, stim_channel_var, read_channel_var,
    cycle_train_a_topo_var, cycle_train_a_ch_var,
    cycle_train_a_write_ch_var, cycle_train_a_read_ch_var,
    cycle_train_b_topo_var, cycle_train_b_ch_var,
    cycle_train_b_write_ch_var, cycle_train_b_read_ch_var,
]
for _c in _schematic_combos:
    _wrap_combo_command(_c, _schedule_schematic_update)

# IntVar-backed checkboxes — tk traces fire on set().
srdp_log_scale_var.trace_add('write', _schedule_schematic_update)
visual_continuous_var.trace_add('write', _schedule_schematic_update)
cycle_enable_train_b_var.trace_add('write', _schedule_schematic_update)

# Initialize display to show Diode mode
update_mode_display()

# ==============================================================================
# WATERMARK
# ==============================================================================

watermark_label = ctk.CTkLabel(root, text="Zacharie Jehl Li-Kao --- zacharie.jehl@upc.edu", text_color="gray")
watermark_label.pack(side="bottom", anchor="se", padx=10, pady=10)

# ==============================================================================
# START APPLICATION
# ==============================================================================

# Guarded so the module can be imported without entering the event loop.
# hub.py launches this file as a script, so __name__ == "__main__" there and the
# GUI starts exactly as before. Without the guard, `import keithley_analyser`
# blocks forever, which makes SYNAPSE_MODE_BY_LABEL and canonical_synapse_mode
# — the mode-dispatch contract the engine depends on — impossible to verify.
if __name__ == "__main__":
    root.mainloop()