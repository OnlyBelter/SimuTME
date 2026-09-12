# GEP-level and gene-level filtering migration from DeSide to SimuTME

Last updated: 2026-09-11

## Status

Draft for review.

## Goal

Migrate and integrate the following workflows from DeSide into SimuTME:

- GEP-level filtering during mixed-bulk simulation
- gene-level filtering after mixed-bulk generation

The integration should follow **SimuTME's existing workflow and YAML style**
rather than copying DeSide's standalone config layout directly.

## Problem summary

DeSide already provides a usable, config-driven filtering workflow for bulk
simulation:

- **GEP-level filtering** is applied inside mixed-bulk generation
- **gene-level filtering** is applied later as a post-generation stage
- outputs include filtered `.h5ad`, filtered gene list, PCA artifacts, and a
  summary JSON

SimuTME currently has only a partial version of this behavior:

- mixed-bulk simulation can already call TCGA-guided filtering during
  generation
- a post-processing hook exists for gene filtering and PCA
- several low-level utilities already exist in `src/simutme/workflow.py`
- but the exposed parameter surface is incomplete and somewhat inconsistent
  with the richer DeSide workflow

As a result, SimuTME does not yet offer a complete, explicit, and well-scoped
filtering workflow that users can configure at both the global and dataset
levels.

## Comparison summary

### DeSide

Relevant pieces:

- `deside/workflow/bulk_simulation.py`
- `deside/configs/__init__.py`
- `deside/simulation/generate_data.py`

Strengths:

- filtering is broken into two clear stages
- config surface is explicit:
  - `gep_filtering`
  - `gene_filtering`
  - output and summary paths
- generated artifacts are recorded consistently
- runtime edge cases are already handled:
  - missing PCA model vs existing PCA data
  - high-correlation gene list reuse
  - optional TCGA subtype restriction

Weaknesses for direct reuse in SimuTME:

- config shape is DeSide-specific
- naming and output conventions differ from SimuTME
- workflow assumes DeSide's standalone bulk-simulation entry point

### SimuTME

Relevant pieces:

- `src/simutme/workflow.py`
- `src/simutme/utility/read_file.py`
- `src/config_example.yaml`
- `docs/workflow_mixed.md`

Strengths:

- mixed and SCT workflows are already unified in one YAML-driven pipeline
- dataset-level overrides are already common in `datasets:`
- post-processing already exists in `_run_post_processing(...)`
- required utilities already exist in SimuTME:
  - `get_gene_list_for_filtering(...)`
  - `filtering_by_gene_list_and_pca_plot(...)`
  - `ReadH5AD`
  - `ReadExp`
  - PCA utilities

Current gaps:

- filtering configuration is too global and under-specified
- GEP-level filtering settings are not explicitly grouped or validated
- gene-level filtering behavior is tied to global `gene_list_type` rather than
  a clearer filtering contract
- artifact naming and summary reporting are weaker than in DeSide

## Design constraints

The migration should:

- preserve SimuTME's current YAML style
- keep Mixed-dataset filtering behavior explicit
- allow global defaults with dataset-level overrides
- avoid breaking current SCT workflows
- avoid forcing a DeSide-style top-level `gep_filtering` / `gene_filtering`
  block structure
- reuse SimuTME's current post-processing hook instead of introducing a second
  disconnected execution path
- remain compatible with the existing Mixed workflow concepts:
  - `sct_reference`
  - `sampling_method_prop`
  - `filtering`
  - `filtering_ref_types`

## Options considered

### Option A: copy DeSide's config and workflow layout directly

Pros:

- highest parity with DeSide
- easiest mental mapping from old code to new code

Cons:

- does not fit SimuTME's current YAML structure
- creates two competing workflow styles inside one package
- higher long-term maintenance burden

### Option B: hybrid layout with nested filtering blocks inside Mixed datasets

Pros:

- keeps filtering parameters grouped
- closer to DeSide than flat overrides

Cons:

- introduces a new nested style not widely used elsewhere in SimuTME
- larger schema change than necessary

### Option C: SimuTME-native integration with global defaults and dataset overrides

Pros:

- matches current SimuTME design
- smallest conceptual jump for existing users
- keeps one workflow entry point
- still allows near-DeSide feature parity

Cons:

- requires careful parameter naming and mapping
- some DeSide names need translation rather than direct reuse

## Recommended approach

Use **Option C**.

That means:

- keep filtering inside SimuTME's existing workflow
- keep global defaults in `SimuConfig`
- allow Mixed dataset entries to override filtering-related fields
- treat filtering as two explicit stages:
  - **GEP-level filtering** during mixed-bulk generation
  - **gene-level filtering** during post-processing

This preserves SimuTME's identity while bringing over the missing DeSide
capabilities.

## Scope

### In scope

- migrate DeSide's full GEP-level filtering parameter surface that is relevant
  to SimuTME mixed-bulk generation
- migrate DeSide's gene-level filtering workflow and artifact management
- improve output naming and summary reporting for SimuTME filtered outputs
- add validation rules for filtering-related configuration
- update SimuTME docs and example config
- add tests for the new filtering behavior

### Out of scope

- SCT-only filtering workflows
- training-time use of the filtered outputs
- migrating DeSide's entire standalone workflow summary JSON format as-is
- introducing a second standalone filtering CLI independent of SimuTME's main
  workflow

## Proposed configuration design

## Global defaults in `parameters`

SimuTME should keep global filtering defaults under `parameters`, because that
is already how the package exposes other workflow-wide settings.

### Global GEP-level filtering defaults

Add or formalize these fields in `parameters`:

- `gep_filtering_enabled_default: bool`
- `gep_filtering_reference_file: path`
- `gep_filtering_ref_exp_type: str`
- `gep_filtering_method: str`
- `gep_filtering_quantile: [lower, upper]`
- `gep_filtering_n_top: int`
- `gep_show_filtering_info: bool`
- `gep_filtering_by_gene_range: bool`
- `gep_min_percentage_within_gene_range: float`
- `gep_gene_quantile_range: [q_low, q_mid, q_high]`
- `gep_filtering_in_pca_space: bool`
- `gep_pca_n_components: float | int`
- `gep_norm_ord: int`
- `gep_high_corr_gene_list_file: path | null`

### Global gene-level filtering defaults

Add or formalize these fields in `parameters`:

- `gene_filtering_enabled_default: bool`
- `gene_filtering_reference_file: path`
- `gene_filtering_type: str`
- `gene_filtering_quantile_range: [q_low, q_mid, q_high]`
- `gene_filtering_q_col_name: [name_low, name_mid, name_high]`
- `gene_filtering_corr_threshold: float`
- `gene_filtering_n_gene_max: int`
- `gene_filtering_high_corr_gene_file: path | null`
- `gene_filtering_save_filtered_h5ad: bool`
- `gene_filtering_filtered_dataset_postfix: str`
- `gene_filtering_plot_pca: bool`
- `gene_filtering_pca_n_components: int`
- `gene_filtering_pca_figsize: [w, h]`

### Output controls

Add global defaults for optional explicit artifact paths only if needed later.
For the first migration, deterministic defaults are preferred.

## Dataset-level Mixed overrides

Each Mixed dataset may override the global defaults with flat dataset-level
fields, consistent with the current SimuTME style.

### Mixed dataset fields to support

- `filtering`
  Existing flag. This becomes the primary Mixed-dataset switch for
  **GEP-level filtering**.
- `filtering_ref_types`
  Existing field. Keep its current meaning.
- `cell_prop_prior`
  Existing field. Keep unchanged.
- `filtering_in_pca_space`
  Existing field. Keep as a dataset-level override.

### New or clarified dataset-level overrides

- `gep_filtering_method`
- `gep_filtering_quantile`
- `gep_filtering_reference_file`
- `gep_high_corr_gene_list_file`
- `gep_filtering_by_gene_range`
- `gep_min_percentage_within_gene_range`
- `gep_gene_quantile_range`
- `gep_pca_n_components`
- `gep_norm_ord`
- `gep_n_top`
- `gep_show_filtering_info`
- `gene_filtering`
  New explicit dataset-level switch for post-generation gene filtering
- `gene_filtering_type`
- `gene_filtering_reference_file`
- `gene_filtering_high_corr_gene_file`
- `gene_filtering_corr_threshold`
- `gene_filtering_n_gene_max`
- `gene_filtering_quantile_range`
- `gene_filtering_q_col_name`
- `gene_filtering_plot_pca`
- `gene_filtering_pca_n_components`
- `gene_filtering_pca_figsize`

## Why flat overrides instead of nested blocks

Flat overrides are recommended because:

- they match current SimuTME dataset entries
- they are easier to merge with existing fields like `filtering` and
  `filtering_ref_types`
- they minimize schema churn

## Proposed workflow design

## Stage 1: Mixed generation with GEP-level filtering

This stage remains inside `run_bulk_simulation(...)` for Mixed datasets.

Behavior:

1. resolve the SCT reference as SimuTME already does
2. build the mixed `BulkGEPGenerator`
3. construct a generation config that includes resolved GEP-level filtering
   settings
4. run mixed-bulk generation
5. save bulk GEPs and cell fractions as today

Integration point:

- keep GEP-level filtering embedded inside the generator path
- do **not** move it into `_run_post_processing(...)`

Reason:

- this matches DeSide's behavior
- GEP-level filtering removes candidate simulated samples before final output
  materialization

## Stage 2: Gene-level filtering and PCA post-processing

This stage stays inside `_run_post_processing(...)`, but is expanded to become a
fully explicit workflow.

Behavior:

1. resolve the effective gene-filtering parameters
2. resolve or build the filtered gene list
3. apply gene-level filtering to the mixed bulk `.h5ad`
4. optionally compute PCA and generate PCA plots against the reference bulk
   dataset
5. save the filtered `.h5ad`
6. write summary metadata for filtered sample/gene counts and artifact paths

## Effective parameter resolution order

For both filtering stages:

1. use dataset-level override if present
2. otherwise use global `parameters` default
3. otherwise use code default only when the behavior is intentionally optional

This should be implemented explicitly, not through scattered `dict.get(...)`
calls across the workflow.

## Proposed code organization

## `src/simutme/workflow.py`

Keep this file as the orchestration layer, but split filtering-related logic
into smaller helpers.

### New helper groups

- `_resolve_gep_filtering_params(dataset_name, params, config)`
- `_resolve_gene_filtering_params(dataset_name, params, config)`
- `_run_gene_filtering_postprocess(dataset_name, generated_bulk_gep_fp, params, config)`
- `_build_mixed_dataset_summary(...)`

### Existing functions to revise

- `run_bulk_simulation(...)`
- `_run_post_processing(...)`
- `get_gene_list_for_filtering(...)`
- `filtering_by_gene_list_and_pca_plot(...)`

## `src/simutme/config.py`

Keep biological constants here only.

Do **not** move workflow-specific filtering logic into this file.

## `src/config_example.yaml`

Update the example config to:

- show new global filtering defaults in `parameters`
- show one Mixed dataset with default filtering
- show one Mixed dataset with dataset-level overrides

## Artifact design

## Mixed bulk outputs

Keep SimuTME's current mixed-bulk naming and directory conventions.

Do not force DeSide's `simu_bulk_exp_<name>_log2cpm1p.h5ad` naming into
SimuTME.

## Gene-level filtering outputs

For each Mixed dataset, write dataset-local filtering artifacts under:

`paths.simu_bulk_dir/<dataset_name>/`

Recommended files:

- `gene_list_filtered_by_<gene_filtering_type>.csv`
- `gene_corr_with_cell_fraction.csv`
- `both_TCGA_and_simu_<dataset_name>_PCA_<gene_filtering_type>.joblib`
- `<dataset_name>_PCA_with_TCGA_<gene_filtering_type>.csv`
- `<dataset_name>_PCA_with_TCGA_<gene_filtering_type>_PC0_PC1.png`
- filtered `.h5ad` file beside the original mixed file or under the
  dataset-local directory, but using one consistent rule

## Summary outputs

Add a dataset-level summary JSON for Mixed runs with filtering enabled.

Recommended fields:

- `dataset_name`
- `dataset_type`
- `generated_bulk_gep_file`
- `generated_cell_fraction_file`
- `generated_n_samples`
- `generated_n_genes`
- `gep_filtering_enabled`
- `gep_filtering_method`
- `gene_filtering_enabled`
- `gene_filtering_type`
- `filtered_bulk_gep_file`
- `filtered_n_samples`
- `filtered_n_genes`
- `filtered_gene_count`
- `gene_list_file`
- `pca_model_file`
- `pca_data_file`

Reason:

- this brings SimuTME closer to DeSide's artifact traceability
- it makes downstream inspection and debugging much easier

## Parameter mapping from DeSide to SimuTME

### GEP-level filtering

- DeSide `reference_file` -> SimuTME `gep_filtering_reference_file`
- DeSide `ref_exp_type` -> SimuTME `gep_filtering_ref_exp_type`
- DeSide `filtering_method` -> SimuTME `gep_filtering_method`
- DeSide `filtering_ref_types` -> SimuTME existing `filtering_ref_types`
- DeSide `gep_filtering_quantile` -> SimuTME `gep_filtering_quantile`
- DeSide `n_top` -> SimuTME `gep_n_top`
- DeSide `show_filtering_info` -> SimuTME `gep_show_filtering_info`
- DeSide `high_corr_gene_list_file` -> SimuTME `gep_high_corr_gene_list_file`
- DeSide `filtering_by_gene_range` -> SimuTME `gep_filtering_by_gene_range`
- DeSide `min_percentage_within_gene_range` -> SimuTME `gep_min_percentage_within_gene_range`
- DeSide `gene_quantile_range` -> SimuTME `gep_gene_quantile_range`
- DeSide `filtering_in_pca_space` -> SimuTME `gep_filtering_in_pca_space`
- DeSide `pca_n_components` -> SimuTME `gep_pca_n_components`
- DeSide `norm_ord` -> SimuTME `gep_norm_ord`

### Gene-level filtering

- DeSide `filtering_type` -> SimuTME `gene_filtering_type`
- DeSide `reference_file` -> SimuTME `gene_filtering_reference_file`
- DeSide `quantile_range` -> SimuTME `gene_filtering_quantile_range`
- DeSide `q_col_name` -> SimuTME `gene_filtering_q_col_name`
- DeSide `corr_threshold` -> SimuTME `gene_filtering_corr_threshold`
- DeSide `n_gene_max` -> SimuTME `gene_filtering_n_gene_max`
- DeSide `high_corr_gene_file` -> SimuTME `gene_filtering_high_corr_gene_file`
- DeSide `save_filtered_h5ad` -> SimuTME `gene_filtering_save_filtered_h5ad`
- DeSide `filtered_dataset_postfix` -> SimuTME `gene_filtering_filtered_dataset_postfix`
- DeSide `plot_pca` -> SimuTME `gene_filtering_plot_pca`
- DeSide `pca_n_components` -> SimuTME `gene_filtering_pca_n_components`
- DeSide `pca_figsize` -> SimuTME `gene_filtering_pca_figsize`

## Validation rules

Add validation for:

- Mixed datasets only:
  - GEP-level filtering overrides
  - gene-level filtering overrides
- `filtering_ref_types` requires a cancer-type mapping file when subtype
  restriction is requested
- gene-level filtering requires a reference expression file when enabled
- quantile-based filtering requires a valid quantile triplet
- `gene_filtering_q_col_name` must match the quantile triplet length
- PCA plotting requires `gene_filtering_pca_n_components >= 2`
- missing PCA model with existing PCA data should trigger recomputation, not
  crash

## Backward compatibility

This migration should preserve current SimuTME behavior for configs that only
use the existing fields:

- `filtering`
- `filtering_ref_types`
- `gene_quantile_range`
- `gep_filtering_quantile`
- `pca_n_components`
- `gene_list_type`

Compatibility strategy:

1. keep old global fields working in the first migration
2. map them internally to the new resolved filtering parameter model
3. update docs to prefer the new explicit names
4. deprecate ambiguous older names later only if necessary

## Testing plan

Add tests covering:

- Mixed dataset with no filtering
- Mixed dataset with GEP-level filtering only
- Mixed dataset with gene-level filtering only
- Mixed dataset with both filtering stages enabled
- dataset-level override precedence over global defaults
- `filtering_ref_types: null` vs non-empty values
- high-correlation gene file reuse
- quantile-range gene list generation
- missing PCA model with existing PCA data
- summary JSON correctness
- filtered gene count matches filtered `.h5ad` gene dimension

## Documentation plan

Update:

- `src/config_example.yaml`
- `docs/workflow_mixed.md`
- `README.md` if workflow examples are shown there

Documentation should explain:

- the difference between GEP-level and gene-level filtering
- where filtering happens in the workflow
- which parameters are global defaults vs dataset-level overrides
- what artifacts are generated

## Implementation plan

Recommended order:

1. add resolved filtering parameter helpers
2. wire GEP-level filtering resolution into Mixed generation
3. expand gene-level post-processing into a full workflow
4. add summary JSON output
5. update config example and docs
6. add tests

## Open decisions already resolved in this spec

- **Config style**: keep SimuTME-native style
- **Workflow shape**: keep one main workflow entry point
- **GEP-level filtering location**: during mixed generation
- **gene-level filtering location**: post-processing
- **Artifact naming**: keep SimuTME naming, not DeSide naming

## Success criteria

This migration is complete when:

- SimuTME can express both filtering stages clearly in its own YAML style
- Mixed datasets can use global defaults or dataset-level overrides
- gene-level filtering produces filtered `.h5ad`, gene list, PCA outputs, and
  summary metadata
- GEP-level filtering behavior is aligned with DeSide at the algorithm level
- existing SimuTME configs continue to run without unnecessary breakage
