import json
import sys

# Load your JSON file
json_file = sys.argv[1] if len(sys.argv) > 1 else "your_file.json"

with open(json_file, 'r') as f:
    suite = json.load(f)

print("="*70)
print("DIAGNOSTIC: What does fitting.py receive?")
print("="*70)

from fitting import DataParser

parsed = DataParser.parse_suite(suite)

# Check wavelength data
if 'wavelength_response' in parsed and parsed['wavelength_response']:
    wr = parsed['wavelength_response']
    print("\nWavelength Response Data:")
    print(f"  Wavelengths: {wr.get('wavelengths_nm', [])}")
    print(f"  Delta_G (S): {wr.get('delta_g_S', [])}")
    print(f"  Delta_G (%): {wr.get('delta_g_percent', [])}")
else:
    print("\n✗ NO wavelength response data found!")
    print("\nChecking raw datasets for wavelength info:")
    for i, ds in enumerate(suite.get('datasets', [])):
        wl = ds['metadata'].get('wavelength_pot')
        exp_type = ds['metadata'].get('experiment_type')
        print(f"  Dataset {i+1}: type={exp_type}, wavelength_pot={wl}")

# Check nonlinearity data
if 'nonlinearity' in parsed and parsed['nonlinearity']:
    nl = parsed['nonlinearity']
    print(f"\nNonlinearity Data:")
    print(f"  Number of points: {len(nl.get('G_initial_S', []))}")
    print(f"  G_initial range: {min(nl['G_initial_S'])*1e6:.2f} to {max(nl['G_initial_S'])*1e6:.2f} µS")
    print(f"  delta_g range: {min(nl['delta_g_S'])*1e6:.2f} to {max(nl['delta_g_S'])*1e6:.2f} µS")
    import numpy as np
    delta_g = np.array(nl['delta_g_S'])
    print(f"  Negative deltas: {np.sum(delta_g < 0)} / {len(delta_g)}")
else:
    print("\n✗ NO nonlinearity data found!")

# Check dynamic range
if 'dynamic_range' in parsed and parsed['dynamic_range']:
    dr = parsed['dynamic_range']
    print(f"\nDynamic Range:")
    print(f"  G_min: {dr['G_min_S']*1e6:.2f} µS")
    print(f"  G_max: {dr['G_max_S']*1e6:.2f} µS")
else:
    print("\n✗ NO dynamic range data found!")