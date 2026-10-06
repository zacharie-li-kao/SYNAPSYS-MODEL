"""Headless generate → convert → fit pipeline.

Runs the full synthetic path with no GUI: `generate_test_data.py` writes the
CSV suite, `data_converter`'s own parsing and derivation logic turns it into a
v2.0 JSON suite, and `fitting.extract_synapse_model` fits it.

The conversion step deliberately reuses `data_converter`'s real code rather than
reimplementing it. `SynapseDataPrepApp.parse_metadata_file` never touches
`self`, so it can be called unbound; `DataProcessor.process_dataset` is exactly
the call the GUI makes. Only the batch-load loop — which is interleaved with
widget updates and modal dialogs — is replicated here, and it is replicated
faithfully: first two columns of each CSV as x and y, in file order.
"""

import os
import subprocess
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

METADATA_FILENAME = "METADATA_INSTRUCTIONS.txt"

# Every file the generator is expected to write. The three at the end are the
# ones that went missing when the generator died on a console-encoding fault
# (audit C5): without METADATA_INSTRUCTIONS.txt the suite cannot be batch
# loaded at all, and without the STDP/SRDP CSVs those extractors silently fall
# back to defaults.
REQUIRED_OUTPUTS = (
    METADATA_FILENAME,
    "stdp_timing_window.csv",
    "srdp_frequency_response.csv",
)


def generate(output_dir, seed=20260731):
    """Run generate_test_data.py into `output_dir`. Returns the file list.

    Always seeded: unseeded, the noise and per-dataset variability differ every
    run, and a shift in a fitted parameter cannot be told apart from scatter.
    """
    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(PROJECT_ROOT, "generate_test_data.py")

    proc = subprocess.run(
        [sys.executable, script, output_dir, "--seed", str(seed)],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"generate_test_data.py exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )

    written = sorted(os.listdir(output_dir))
    missing = [f for f in REQUIRED_OUTPUTS if f not in written]
    if missing:
        raise RuntimeError(
            f"generator completed but did not write: {missing}\n"
            f"wrote {len(written)} files: {written}"
        )
    return written


def convert(data_dir):
    """Build a v2.0 characterization suite from a generated CSV directory.

    Mirrors `SynapseDataPrepApp.batch_load_with_metadata` without the GUI.
    """
    import data_converter as dc

    metadata_path = os.path.join(data_dir, METADATA_FILENAME)
    # Called unbound: parse_metadata_file never references self.
    file_metadata = dc.SynapseDataPrepApp.parse_metadata_file(None, metadata_path)
    if not file_metadata:
        raise RuntimeError(f"no metadata parsed from {metadata_path}")

    datasets = []
    skipped = []
    for filename, metadata in file_metadata.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            skipped.append(f"{filename} (not found)")
            continue

        df = pd.read_csv(filepath)
        if df.shape[1] < 2:
            skipped.append(f"{filename} (< 2 columns)")
            continue

        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        derived = dc.DataProcessor.process_dataset(x_data, y_data, metadata)

        datasets.append({
            "filename": filename,
            "filepath": filepath,
            "x_data": x_data.tolist(),
            "y_data": y_data.tolist(),
            "metadata": metadata,
            "derived_metrics": derived,
        })

    if skipped:
        raise RuntimeError(
            "conversion skipped files, so the fit would be run on a partial "
            f"suite: {skipped}"
        )

    return {
        "datasets": datasets,
        "n_datasets": len(datasets),
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool_version": "2.0",
    }


def fit(suite, verbose=False):
    import fitting
    return fitting.extract_synapse_model(suite, verbose=verbose)


def run(output_dir, seed=20260731, verbose=False):
    """generate → convert → fit. Returns (suite, fitted_model)."""
    generate(output_dir, seed=seed)
    suite = convert(output_dir)
    return suite, fit(suite, verbose=verbose)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "_verification_run")
    suite, model = run(target, verbose=True)
    print(f"\n{suite['n_datasets']} datasets → fitted model with "
          f"{len(model)} keys")
    for key in sorted(k for k in model if not callable(model.get(k))):
        if key != "extraction_report":
            print(f"  {key}: {model[key]}")
