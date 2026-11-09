# The launcher

SYNAPSYS is started from `hub.py`, or from the hub executable in the installed version. The hub is a single window with three tiles, one per module, and has nothing to configure (Figure {hub}). Each button opens its module in a window of its own, so the hub can stay open and the same module can be launched more than once.

![The hub. Red opens the characterisation tool, blue the data converter, green the modelling module.](figures/hub.png)

The red tile opens the characterisation tool, which drives the Keithley sourcemeter. The blue tile opens the data converter, for measurements taken on another setup. The green tile opens the modelling module, which reads a characterisation suite and fits it. The modules are linked by files only: the characterisation tool and the data converter both write a JSON suite, and the modelling module reads it.

The grey button under the data converter, Generate Test Data, asks for an output folder and writes a set of synthetic measurements as CSV files, together with a `METADATA_INSTRUCTIONS.txt` that lists what to type in the data converter for each file. If no folder is chosen, the files go to `test_data_output` in the working directory. This is the quickest way to try the data converter and the modelling module without a device. The message box that appears after the click only confirms that the generator was started; the files are there once the folder has been chosen.
