# Dirichlet sampling seed design

This change makes Mixed-dataset Dirichlet cell-fraction draws reproducible
without changing unrelated random sampling in SimuTME.

## Configuration

Add optional `dirichlet_sampling_seed` under global `parameters`, with an
optional dataset-level override for Mixed datasets using
`sampling_method_prop: "dirichlet"`. The dataset-level value takes precedence.
The example pan-cancer config sets the global seed to `123`.

Remove the manually specified all-ones `dirichlet_alpha` dictionary from that
example config. Its `pred_cell_prop_file_path` remains unchanged and continues
to estimate alpha from the CSV, taking precedence over any manual dictionary.
If no alpha dictionary or proportion file is supplied, the existing all-ones
default remains available.

## Random-number handling

When a seed is set, create one local `numpy.random.Generator` for the Mixed
generation run and pass it through each Dirichlet fraction batch. Do not
reinitialize it per batch, which would repeat the same fractions. When no seed
is set, retain the current global NumPy random behavior for backward
compatibility. The seed controls Dirichlet cell fractions only, not other
stochastic steps such as SCT-profile sampling.

Completed output files continue to be reused by the existing workflow. A new
seed does not force regeneration. Exact continuation from partial intermediate
output is outside scope because RNG state is not checkpointed.

## Implementation and verification

- Add the seed to `SimuConfig` and `GEPGenerationConfig`.
- Resolve the global value with an optional Mixed-dataset override.
- Pass the persistent generator through `generate_cell_fraction` to
  `dirichlet_sampling`; leave non-Dirichlet methods unchanged.
- Update the SimuTME configuration example, parameter documentation, and the
  user's pan-cancer example config.
- Test that identical seeds reproduce the same sequence across batches,
  successive batches do not repeat, different seeds change the sequence, and
  an unset seed preserves the existing fallback.
- Confirm the CSV-based alpha estimate still runs after removing the manual
  all-ones dictionary, and run the focused tests and YAML parsing checks.
