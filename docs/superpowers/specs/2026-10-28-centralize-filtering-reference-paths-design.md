# Centralize Mixed filtering reference paths

## Goal

Keep shared file locations together in the example YAML's `paths` section
without changing reference fallback or dataset-override behavior.

## Design

Move the global `gep_filtering_reference_file` and
`gene_filtering_reference_file` keys from `parameters` to `paths` in
`src/config_example.yaml`. Also move the example's
`pred_cell_prop_file_path` from its Dirichlet dataset entry to `paths`.
Keep the existing reference values and preserve `tcga_merged_tpm_file_path`
as the fallback when either reference override is unset or `null`.

Update `docs/configuration_io.md` to list all three paths under Paths, remove
the moved keys from their old configuration lists, and document their fallback
and use conditions. Update `docs/usage.md` and `docs/workflow_mixed.md` to
describe the shared paths and their input requirements.

Add `pred_cell_prop_file_path` to `SimuConfig` so the global `paths` value is
available to Mixed workflows. Resolve a dataset-level value before the global
value when one is explicitly supplied. Use the shared file to estimate
Dirichlet parameters for Dirichlet sampling, or to cap segment fractions when
GEP filtering is enabled. Ignore the shared path for datasets using neither
supported mode so an unused global file does not make unrelated datasets
fail. Preserve dataset-level `gep_filtering_reference_file` and
`gene_filtering_reference_file` overrides. Existing configurations with the
reference keys under `parameters` and prediction paths under dataset entries
remain readable.

## Verification

- Parse `src/config_example.yaml` and confirm all three keys are under `paths`
  rather than `parameters` or `datasets`.
- Load the configuration through `SimuConfig.from_yaml` and confirm all three
  shared values are available on the resulting config.
- Verify Dirichlet estimation, segment caps with GEP filtering, and ignoring
  an unused global prediction path for other dataset modes.
- Check the updated documentation references and run `git diff --check`.
