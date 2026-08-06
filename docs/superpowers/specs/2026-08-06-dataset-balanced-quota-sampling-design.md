# Stratified dataset_id balanced-quota sampling for SCT candidate pools

Last updated: 2026-08-06

## Status
Draft for implementation.

## Goal / Motivation

A primary upstream cause of the "recon GEP collapse" diagnosed during VAEDecon training
(i.e., `recon_vs_recon` off-diagonal CCC ~ 0.9–0.99 even when `true_vs_true` CCC is substantially lower) is that the training SCT candidate-cell pool used by `BulkGEPGenerator` / `SingleCellTypeGEPGenerator` can be dominated by a handful of scRNA-seq datasets whose cell count for a given cell type is much larger than the other datasets. For example, one dataset might have 30,000 CD4 T cells while ten other datasets have 2,000 each. Unrestricted pooling then produces a pool where ~94% of all CD4 T SCT samples originate from the same study, which (a) concentrates patient-specific batch effects into the training targets and (b) causes the SCT distribution to under-cover the true cell-type phenotypic modes present in smaller datasets. The downstream VAE trained on such skewed targets easily learns a collapsed centroid GEP + study-level confound instead of sample-level structural differences. This effect occurs even when `n_neighbors` sampling is used because, while it anchors to marker-ratio reference space, the neighbor pool itself is already skewed by dataset cell count.

This design adds one new knob that reduces each (cell_type, dataset_id) group to the same balanced quota. Downstream Mixed datasets inherit the balanced pool automatically because they read the already-generated SCT `.h5ad`.

## Background / Existing behavior

Currently:

```
merged_sc_dataset_obs = obs.loc[cancer_mask | non_cancer_mask]
```

built in `BulkGEPGenerator.get_info_in_merged_single_cell_dataset` after applying `cancer_cell_dataset_ids` / `non_cancer_cell_dataset_ids`. Rows are never rebalanced across dataset_id; every dataset contributes rows in proportion to its raw cell count.

## Proposed change (minimal, backward compatible)

### 1. Config interface

Add new key (global `parameters`):
```yaml
parameters:
  stratify_by_dataset: "balanced_quota"   # enum: "none" (default), "balanced_quota"
```

Allowed values:
- `"none"`: no stratification, existing behavior unchanged (backward compat.
- `"balanced_quota"`: for each cell type c independently, each dataset's rows are down-sampled (without replacement) to the size of the smallest dataset that still has cell type c. Smallest-dataset size is the quota per (cell_type). If a dataset does not contain cell type c, it simply contributes zero rows.

Additionally, override per-SCT-dataset override the same semantics matching the existing `run_sample_diversity_analysis` pattern):
```yaml
datasets:
  SCT_Name:
    dataset_type: "SCT"
    stratify_by_dataset: "balanced_quota"   # optional, overrides parameters.stratify_by_dataset
```

The override is only supported for SCT datasets. For Mixed datasets it is forbidden via explicit guard (same pattern as `cancer_cell_dataset_ids`).

### 2. Implementation placement

Add private method `BulkGEPGenerator._stratify_obs_by_dataset_balanced_quota(obs_df) -> df`:

1. Iterate `obs_df already filtered by cancer/non cancer masks.
2. For each distinct cell_type c in obs[`self.cell_type_col_name`]:
   a. Subset obs_df to cell_type == c → per_ds_counts = groupby dataset_id → counts n_d(c)
   b. K(c) = min(per_ds_counts.values) if per_ds_counts not empty else skip.
   c. If K(c) < 10 logger warning: logger.warning(f"Balanced quota for {c}: K(c) = K(c), bottleneck dataset = {bottleneck}. Consider adding more datasets.")
   d. per dataset d in per_ds_counts.index:
      n_keep = min(K(c), per_ds_counts[d])
      rng = np.random.default_rng(seed = derived_seed(self.sorted_bulk_imputation_seed ^ hash(c) % 2**32)
      take first rng.choice(rows, size=n_keep, replace=False) → rows_kept.append(idx)
3. Concat all kept indices, return obs_df.loc[rows_kept].

Apply right after building `merged_sc_dataset_obs = ...copy()` in `get_info_in_merged_single_cell_dataset`, i.e. after applying cancer/non-cancer dataset filtering.

Reproducibility: seed derived from existing user-configurable `sorted_bulk_imputation_seed` XORed a hash of the cell_type name, so same config + same seed = same stratified rows across runs, and different cell types get independent draws.

Per-cell-type INFO log print quota info including before/after per cell type:
"Stratified CD4 T: K=2370 (bottleneck dataset: lu total: 11 ds x 2370 = 26070 kept (was 45000)

### 3. Interaction with filters

Zero-ratio filtering (threshold 97) is applied after sampling, so it operates on the already-stratified pool; this correct because zero-ratio filtering can still filter out already-balanced rows independently.

Cancer/non-cancer dataset filtering applies BEFORE stratification. Example: in current user config Cancer Cells come from pdac_peng_02), so Cancer Cells balanced quota trivially single dataset → rows unchanged (correct: intentional), and 11-dataset non-cancer cells get balanced per non-cancer cell type.

### 4. Mixed datasets: inherit, no explicit config

Mixed datasets get the already-stratified SCT h5ad rows via `sct_reference`; no additional work or key needed.

### 5. Example updated

Set `stratify_by_dataset: "balanced_quota"` in `config_example.yaml` and user's active example config with a commented example showing `# balanced_quota: smallest-dataset size sets per (cell_type) pool.

### 6. Testing plan

1. Unit: build obs 3 cell types × 4 datasets with imbalanced counts → after stratification, per (dataset_id, cell_type) counts == == min_count for that cell type; no rows dropped from smallest dataset.
2. Reproducibility: two separate calls with identical seed → identical row indices returned; different seed → different.
3. Config guard: enum validates "none" / "balanced_quota"; unknown enum raises ValueError; SCT override not allowed on Mixed datasets (else raise.
4. Integration direct runner / pytest equivalent).

## File changes list

- `SimuTME/src/simutme/workflow.py`: add `stratify_by_dataset` field to SimuConfig (defaults to "none"); pass from_yaml` validation.
- `SimuTME/src/simutme/simulation/simulator.py`: add _stratify_obs_by_dataset_balanced_quota() to BulkGEPGenerator and wire into get_info_in_merged_single_cell_dataset; also accept stratify_by_dataset as __init__ parameter.
- `SimuTME/src/simutme/workflow.py run_dataset`: pass stratify_by_dataset parameter from (config or SCT override) into SingleCellTypeGEPGenerator and BulkGEPGenerator constructors; add SCT-only ValueError guard for override on Mixed datasets.
- `SimuTME/src/config_example.yaml`: add example key and comment.
- `SimuTME_example/config_n_neighborGEP_nbase_30_eps1e-3_pdac_imputation_1%.yaml`: add key (commented or active as user prefers.
- `SimuTME/tests/`: add tests.
