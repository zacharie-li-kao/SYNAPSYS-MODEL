# The data converter

The data converter turns measurements made on another setup into the same JSON suite as the one that the characterisation tool exports. A file has to be a CSV, Excel or text table with at least two columns, the first for the x axis and the second for the y axis, and every other column is ignored. Each file is loaded with a description of its content, called its metadata, because the modelling module uses it to know which model parameter the file informs.

There are two ways to load files. Load Data File takes one file and opens the metadata window. Batch Load (with metadata file) first asks for a text file with the metadata of every file and then for the folder in which the data files are (Figure {conv_loaded}). The text file has the format of the `METADATA_INSTRUCTIONS.txt` written by Generate Test Data, which is the easiest way to see how it is built. Each dataset appears in the list on the left with its type, axes, number of points and change of conductance, which are computed when the file is loaded.

![The converter after a batch load of the test data. The numbers mark Batch Load (1) and Save JSON (2).](figures/conv_loaded.png)

The metadata window asks what the axes represent and their units, and the type of experiment, for instance potentiation, depression, retention, SRDP or STDP (Figure {conv_metadata}). The fields for the stimulus change with the type: the wavelength and intensity of the light, or the voltage of an electrical pulse, then the pulse width, the frequency or period, the number of pulses and the read timing, and the cycle or train parameters when the type needs them. Missing readings in a file are kept as gaps and are reported by the modelling module. The OK and Cancel buttons are above the stimulus parameters in the window, and the stimulus fields have to be filled before OK is pressed. Each wavelength of a sweep is its own file with its own wavelength and intensity.

![The metadata window for a potentiation measurement at 365 nm.](figures/conv_metadata.png)

A dataset is corrected with Edit Metadata after its index has been typed in the field above the buttons, and removed with Remove Dataset. Clear All empties the list. Save JSON writes the suite, with a file name that carries the date and time by default.
