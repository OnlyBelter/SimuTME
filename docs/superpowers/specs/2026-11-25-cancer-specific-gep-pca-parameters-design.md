# Cancer-specific GEP PCA parameters

Allow a Mixed dataset to configure `n_subgroup` and
`gep_pca_n_components` separately for each cancer type selected as a GEP
filtering reference. Existing scalar-only configurations retain their
current behavior.

## Configuration

Add an optional dataset-level `gep_filtering_by_cancer_type` mapping. Each
key is an exact cancer-type label, and each value is a mapping containing
either or both of `n_subgroup` and `gep_pca_n_components`:

```yaml
datasets:
  Mixed_Cancer_Overrides:
    dataset_type: Mixed
    gep_filtering: true
    gep_filtering_method: median_gep
    gep_filtering_in_pca_space: true
    filtering_ref_types: [ACC, BLCA, LIHC]
    n_subgroup: 200
    gep_pca_n_components: 0.9
    gep_filtering_by_cancer_type:
      ACC:
        n_subgroup: 100
        gep_pca_n_components: 0.95
      LIHC:
        n_subgroup: 150
        gep_pca_n_components: 0.99
```

The existing dataset-level scalar values take precedence over their existing
global defaults. For an eligible type or an individual field without an
override, use the corresponding resolved scalar value. In this example,
BLCA uses `200` and `0.9`. An empty or omitted override mapping preserves
the existing scalar-only workflow and filtering identities.

An override mapping requires a positive resolved scalar `n_subgroup`,
enabled GEP filtering, PCA-space `median_gep`, and the existing
cancer-type mapping file. It cannot be used to activate subgroup behavior
when the default `n_subgroup` is `null`. The existing requirement for a
positive subgroup in `simulation_for_stat` remains unchanged.

`filtering_ref_types` continues to determine the eligible reference cohort.
When it is omitted, all cancer types represented in the resolved reference
are eligible. Overrides do not add or remove eligible types.

## Per-batch behavior

Apply the resolved parameters in both `full_simulation` and
`simulation_for_stat`, for serial and supported parallel candidate
generation:

1. Select one eligible cancer type using the existing uniform selection and
   random stream.
2. Resolve that type's subgroup cap and PCA component or explained-variance
   target.
3. Draw up to that cap from the selected type without replacement. If fewer
   reference samples are available, use all of them.
4. Fit the batch PCA using that type's resolved target, calculate its
   reference-distance quantiles, and apply the resulting filter to the
   candidate batch.

Different batches may reuse reference samples or select the same type. The
existing per-batch seed derivation, worker-count independence, ordered batch
consumption, checkpoint resume, stopping rules, and attribution of counts to
the selected reference type remain unchanged.

For one eligible cancer type whose resolved cap includes its entire reference
cohort, the existing full-reference PCA reuse optimization may continue, but
it must use that type's resolved PCA target and record that target in the
model identity. With multiple eligible types, keep fitting a separate PCA
per batch even if every type's cap includes its full cohort.

## Validation and isolation

Validate the mapping and every resolved eligible-type setting before
candidate generation. Reject unknown or excluded cancer-type keys, empty
type names, non-mapping type entries, unknown fields, explicit null override
values, Boolean or nonpositive subgroup sizes, and invalid PCA component or
variance targets. Omitted fields inherit the corresponding scalar default.
PCA targets are positive integer component counts or floating-point
explained-variance fractions strictly between zero and one.
Reject a type whose effective subgroup and available gene and reference
counts cannot support its effective PCA target. The error identifies the
dataset and cancer type.

Only the PCA reference used for GEP-level subgroup filtering changes. Keep
fixed-gene selection, gene-range bounds, post-GEP gene filtering,
cell-proportion sampling, SCT-profile sampling, and Dirichlet or segment
prior behavior unchanged. The cancer type in these settings is the selected
reference type, not an inferred biological label for generated profiles.

## Metadata and reuse

When overrides are supplied, record the resolved subgroup cap and PCA target
for every eligible cancer type in the filtering identity and
`mixed_dataset_summary.json`, including the isolated statistics-only summary.
Keep the existing scalar summary fields as the scalar defaults. Continue to
record per-type candidate counts and acceptance rates under their existing
field names, and include the effective parameters in per-batch logs.

Changing any effective cancer-specific setting must make old PCA models,
filtering artifacts, and resumable candidate checkpoints incompatible.
Canonicalize the mapping so changing YAML key order alone does not change
the identity. Preserve existing identities and resume behavior for runs
without overrides. Never overwrite existing outputs merely because the
settings changed; require an appropriate fresh output or the existing
explicit compatibility workflow.

## Verification and documentation

Add tests and documentation covering the following:

- Configuration parsing, scalar and per-field fallback, dataset-level
  precedence, and rejection of unsupported or invalid overrides.
- Known but excluded types, unknown types, undersized cohorts, integer PCA
  components exceeding the available samples or genes, and invalid variance
  targets fail before candidate generation.
- Each selected batch uses the correct type-specific sample cap and PCA
  target. Small cohorts use all available samples.
- Single-type full-reference reuse uses the resolved override; multiple
  types continue to use batch-specific PCA.
- Serial and parallel generation, both simulation modes, per-type summary
  counts, deterministic worker-count behavior, and checkpoint resume use
  the resolved settings.
- Different mappings invalidate cached or resumed outputs; equivalent
  mappings in a different order remain compatible; scalar-only runs retain
  their existing identity.

Document the mapping, fallback rules, reporting, and compatibility behavior
in the package configuration and Mixed workflow guides. Add an illustrative
commented example to the package configuration; leave existing user
simulation configurations and outputs unchanged.
