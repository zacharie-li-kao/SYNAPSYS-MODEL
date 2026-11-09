# The characterisation tool

The characterisation tool drives a Keithley 2600-series sourcemeter and measures the conductance of a synaptic device under electrical or optical stimulation. The window is divided in three parts (Figure {char_basic} shows it after a first run). The left panel holds the parameters and scrolls, the centre shows the stimulus and read waveforms above the measured curve, and the right panel gives a numerical summary of the last run, the list of runs kept in memory and the logos. All the figures of this chapter were recorded against a simulated 2636B, so the values only illustrate what the screens look like.

![The window after a first Basic run. The left panel holds the parameters, the centre the pulse schematic and the measured curve, the right panel the summary and the stored runs.](figures/char_basic.png)

The connection is set in the General Parameters at the top of the left panel (Figure {char_connection}). Connection Type is GPIB, RS232 or LAN, and Port Address is the VISA address for GPIB and RS232, or the IP address of the instrument for LAN. There is no Connect button: the instrument is opened when a measurement is started, identified, and released at the end of the run. A model that cannot be identified stops the run, because the voltage limit of the safety checks depends on the model. Measurement Speed is the NPLC of the read; a larger value gives a quieter reading and a longer read window. Mains Frequency is detected from the instrument by default and should only be set by hand to 50 or 60 Hz if the detection is not wanted, since the NPLC only rejects mains pickup when it matches the grid. The sample identifier of each sub-mode is stored with every run and ends up in the exported files.

![Connection type (1) and port address (2).](figures/char_connection.png)

Measurement Mode, below the general parameters, selects Diode, Transistor or Synapse. The two first modes record current-voltage curves and are not covered here. In Synapse mode the Synapse Sub-mode lists Basic, Visual (Self-Powered), SRDP, STDP, Cycle, Multi-Device and Retention, and the parameters below change with the choice (Figure {char_modes}). Stim Channel is the source that applies the stimulus (an LED or an electrical pulse) and Read Channel is the one that measures the device. The checkbox Simulate (No Hardware) allows to try the interface without an instrument.

![Measurement mode (1) and synapse sub-mode (2).](figures/char_modes.png)

Basic, SRDP and STDP share a block of common parameters. Synapse Type is Electrical, Visual or Memristor (Pulse), and Stim Drive Type is voltage or current. The read is made at Read Voltage, starting Read Delay after the end of the stimulus pulse, after a Read Settle Delay that lets the current range settle, and during Read Pulse Width. The read settle delay matters when the device is measured on a low current range: the range needs about 2.5 ms to settle at 1 µA, 15 ms at 100 nA and 90 ms at 10 nA, and a shorter value is reported as a settling warning at the end of the run, because the first readings are then still drifting. Samples per Pulse set to 0 lets the tool choose.

The Basic sub-mode applies a train of identical pulses and reads the conductance after each of them. The fields are Stim Level, Stim Width, Stim Period, the number of pulses, and the wavelength and intensity of the light source, which is set by hand and only recorded with the run. The presets fill typical sets of values. Run Measurement starts the train. The plot shows the stimulus and the read waveforms of the first period, with the width, period, settle time and read delay dimensioned, and below them the conductance and the current at each pulse. Point 0 is the baseline measured before the first pulse, so that a train of n pulses gives n + 1 points. When the run is finished a message reports where the train ran and whether the period was kept (Figure {char_success}). Execution path lua means that the train was run by the instrument itself, which is the accurate case, and standard means that it was timed by the computer, which is less precise for short periods.

![The message at the end of a run.](figures/char_success.png)

Visual (Self-Powered) is for devices that produce a photocurrent without bias. The LED is driven with Light Pulse Voltage during Light Pulse Width every Light Pulse Period, and the current is read at 0 V during the pulse, between Start Delay and End Margin (Figure {char_visual}). A Light Pulse Voltage above 2.5 V has to be confirmed before the run starts. The checkbox Continuous I(t) Mode records the current continuously at every Sample Interval, whereas the default gives one value per pulse. The plot gives the photocurrent at each pulse. A Visual run cannot be selected for a suite, since a photocurrent is not a conductance.

![A Visual run. The photocurrent increases pulse after pulse.](figures/char_visual.png)

SRDP measures the change of conductance as a function of the pulse frequency. The frequency goes from Freq Start to Freq End in the given number of points, on a linear or a logarithmic scale, with a given number of pulses at each frequency (Figure {char_srdp}). The highest frequency that the instrument can deliver is set by the stimulus width, the read delay and the measurement time. When the requested range goes beyond it, a message gives the limit and asks whether to continue (Figure {char_freqrange}). Continuing runs the points above the limit as fast as the instrument can, and the frequency in the file is the one that was delivered. After such a run the end message reports that the period was not delivered (Figure {char_warning}). The data are valid, however, and a shorter stimulus width or a smaller NPLC raises the limit if the requested frequencies are needed.

![An SRDP run.](figures/char_srdp.png)

![The frequency range message.](figures/char_freqrange.png)

![End of a run in which the period was not delivered.](figures/char_warning.png)

STDP applies a pre-spike on the stimulus channel and a post-spike on the read channel, separated by Δt, and gives the change of conductance as a function of Δt (Figure {char_stdp}). The fields are the range of Δt and its number of points, the number of spike pairs, the pre and post levels, the pulse width and the pair period. A range with a point at Δt = 0 is refused, since there is no order between the two spikes, and an even number of points on a symmetric range avoids it.

![An STDP run.](figures/char_stdp.png)

Cycle alternates two trains, A and B, and repeats them for the chosen number of cycles. Train A potentiates and train B depresses with the default values. Each train has its own channel mode (a single SMU that writes and reads, or two SMUs, one for each), write level and width, read voltage, period and number of pulses, and train B can be disabled. # Cycles, Inter-train Delay and Inter-cycle Delay set the rest of the sequence. The plot shows the conductance against the pulse number for all the cycles (Figure {char_cycle}), and the summary gives the mean change of conductance in each phase, its reproducibility, the dynamic range and the ON/OFF ratio. Multi-Device uses the same parameters and repeats the cycle sequence on a number of devices one after the other, with a delay between devices and a name pattern that numbers the samples.

![A Cycle run of three cycles.](figures/char_cycle.png)

Retention follows the conductance of a device after it has been set to a state. The device is first potentiated with another sub-mode and the retention run is started afterwards. The conductance is read at Read Voltage, either at the sample times given as a comma-separated list in seconds, or at every Interval between readings for the given number of points (Figure {char_retention}). The run takes as long as its schedule, since it is timed by the computer.

![A Retention run. The conductance decays after the potentiation.](figures/char_retention.png)

Every run is kept in memory and appears in the Stored runs list of the right panel, where its cross removes it and Clear All empties the list. Save Data writes the displayed run to a CSV file, and the header of the file records the execution path of the run. A run that is neither saved nor exported is lost when the tool is closed. Export Characterization Suite, at the bottom of the Cycle and Retention panels, assembles the stored runs in the JSON file that the modelling module reads. The window lists the stored runs, and each run to be used is selected and given its role with the Mark as buttons: LTP for the exponent α, LTD for β, Cycles for the dynamic range, Retention for the decay time constant, STDP and SRDP (Figure {char_export}). Optical runs at three wavelengths or more are recognised as a wavelength sweep without being marked, and every run keeps its own wavelength and intensity. Export Suite then asks for the file name. The suite is checked before it is written, and a missing or invalid element is listed in a message. A suite that lacks a measurement is accepted, and the corresponding parameters are then shown as defaults by the modelling module.

![The export window with the roles marked. The list shows the stored runs in the order in which they were made.](figures/char_export.png)
