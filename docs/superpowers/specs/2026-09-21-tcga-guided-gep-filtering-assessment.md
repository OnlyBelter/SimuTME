# Assessment of TCGA-guided GEP filtering

## Summary

The current TCGA-guided workflow is highly selective, but it does not perform
full distribution matching. The `median_gep` and `mean_gep` modes retain
simulated samples within a scalar distance threshold from one TCGA center. This
does not reproduce TCGA multimodality, covariance, tails, or cancer-type
structure.

The observed PCA separation is therefore consistent with the current design.
The largest likely contributors are the synthetic cell-proportion model, the
pan-cancer reference mismatch, cancer-cell reference differences, equal
total-RNA coefficients, and gene selection based partly on simulated
cell-fraction correlations.

## Evidence from the current output

The reviewed output is:

`SimuTME_example/datasets/simulated_bulk_cell_dataset/segment_12ds_n_base30_with_filtering_0.975_median_gep/Mixed_N10K_pan_cancer/Mixed_N10K_pan_cancer_PCA_with_TCGA_high_corr_gene_and_quantile_range_PC0_PC1.png`

The saved PCA table contains 10,000 simulated samples and 7,699 TCGA samples.
Their approximate PC1 summaries are:

- Simulated mean: `+47.4`
- TCGA mean: `-61.6`
- Simulated median: `+51.2`
- TCGA median: `-63.4`

The associated workflow summary reports 135,000 candidate samples, of which
10,345 passed GEP-level filtering before the requested 10,000 samples were
saved. The filter is therefore already rejecting most candidates, but the
retained candidates still occupy a different region of expression space.

## How the current filters work

### GEP-level filtering

In `src/simutme/simulation/simulator.py`, the `median_gep` and `mean_gep`
methods:

1. Transform TCGA and simulated expression into a PCA space.
2. Compute one coordinate-wise TCGA median or mean.
3. Compute each TCGA sample's L1 distance from that center.
4. Keep simulated samples whose distance falls within the corresponding
   quantile range.

This is a radial acceptance rule around one point. It does not match the joint
TCGA distribution. It can accept samples in low-density regions or between
TCGA modes, and it can reject valid samples from secondary modes or tails.

### Gene-level filtering

The post-generation `high_corr_gene_and_quantile_range` workflow selects genes
using both TCGA quantiles and correlations between simulated expression and
simulated cell fractions. This can be useful for deconvolution training, but
it is not an unbiased feature set for evaluating similarity to TCGA.

The PCA diagnostic then fits a new PCA model on the concatenated simulated and
TCGA matrices. This is useful for detecting domain separation, but it is not
the same space used for GEP-level rejection filtering.

## Implementation risks

The following implementation details can affect reproducibility or
interpretation:

1. **PCA artifacts are not fully keyed.** Cached PCA files are keyed mainly by
   reference parent directory and component setting. The cache does not encode
   the TCGA subset, gene list, or reference-file content.
2. **Empty TCGA subsets fall back to the full cohort.** An unmatched
   `filtering_ref_types` value produces a warning and broadens filtering to all
   TCGA samples. An explicitly requested empty subset must instead fail fast or
   require an explicit fallback option.
3. **Filtering methods can collide in output paths.** The Mixed output path
   contains `median_gep` even when another filtering method is selected. A
   later run can therefore reuse artifacts generated with a different method.
4. **PCA coordinates are not standardized or whitened.** The default L1
   distance is applied directly to PCA scores, so high-variance components
   dominate the acceptance rule.
5. **The validation PCA is jointly fitted.** The simulated samples influence
   the axes used to display separation, so the plot is a diagnostic rather
   than a reference-fixed similarity metric.

## Likely biological and statistical causes

### Cell-proportion mismatch

The `segment` sampler uses an artificial iterative remainder distribution. It
does not estimate proportions from TCGA. A pan-cancer TCGA cohort has
cancer-type-specific composition, purity, and stromal structure that this
generic distribution does not reproduce.

### Pan-cancer reference mismatch

The current configuration uses a pan-cancer SCT reference and
`filtering_ref_types: null`, which compares one simulated distribution with the
combined distribution of multiple TCGA cancer types. This can produce a large
shift even when individual cancer-specific simulations are reasonable.

### Cancer-cell and imputation mismatch

Malignant single-cell profiles may not represent the bulk tumor programs in
TCGA. Sorted-bulk imputation can improve sparsity while also moving profiles
toward a different biological cohort if the mapping is imperfect.

### RNA-content mismatch

Configurations with `total_rna_coefficient: 1.0` for every cell type treat
cell fractions as RNA fractions. This is usually inaccurate for cancer,
plasma, stromal, and myeloid populations.

### Feature-set mismatch

High-correlation gene selection emphasizes genes that track the simulated
fractions. It can improve deconvolution signal while making the PCA comparison
more sensitive to synthetic composition and marker structure.

## Recommended diagnostic sequence

Run the following ablations before changing the filtering algorithm:

1. Compare original SCT and sorted-bulk-imputed SCT references.
2. Compare `n_base: 30` and `n_base: 50`.
3. Compare `segment`, `dirichlet`, and a TCGA-calibrated composition model.
4. Compare `all_genes` with
   `high_corr_gene_and_quantile_range`.
5. Fit PCA on TCGA only, then transform both TCGA and simulated samples.
6. Repeat the comparison after removing mitochondrial, ribosomal, and
   housekeeping genes.
7. Inspect PC loadings to identify the genes driving PC1.
8. Compare total TPM, detected genes, zero fraction, and per-gene medians.
9. Compare cell-fraction means, variances, zero frequencies, and pairwise
   correlations.

Use a held-out TCGA split for evaluation so that filtering choices are not
evaluated on the same samples used to define the target.

## Recommended workflow improvements

### Match cancer types before matching expression

Generate separate datasets for LUAD, PDAC, HNSCC, GBM, and other relevant
cohorts. For each dataset:

- Use the matching TCGA cancer-type subset.
- Use an SCT reference appropriate for that cancer type.
- Match the number of generated samples to the TCGA cohort composition.
- Combine cancer-specific simulations only after validating each one.

### Calibrate cell proportions

Use TCGA-compatible cell-fraction estimates to fit a Dirichlet or
logistic-normal model. Then use `sampling_method_prop: "dirichlet"` with
estimated parameters instead of relying on the generic `segment` distribution.

### Calibrate RNA abundance

Estimate `total_rna_coefficient` values from matched sorted-bulk or bulk
references. Validate both cell fractions and RNA-weighted fractions after
calibration.

### Replace radial rejection with distribution matching

Generate a large candidate pool, fit PCA on TCGA only, and select candidates
using a joint-distribution criterion such as:

- k-nearest-neighbor distance to TCGA samples;
- robust Mahalanobis distance;
- local density or kernel-density acceptance;
- cluster-specific acceptance;
- optimal transport between TCGA and simulated samples.

The selection procedure must preserve candidate diversity and report how it
changes the cell-fraction distribution.

### Make validation reference-fixed

For the primary PCA diagnostic:

1. Select a gene set independently of simulated cell fractions.
2. Fit preprocessing and PCA on the TCGA training split only.
3. Transform the TCGA validation split and simulated data with that model.
4. Report PC mean and covariance differences.
5. Add MMD or energy distance and a TCGA-versus-simulation classifier AUC.

## Priority plan

The highest-value next experiment is:

1. Use TCGA-fitted PCA.
2. Use all shared genes as a neutral baseline.
3. Match cancer-type-specific TCGA subsets.
4. Use TCGA-calibrated Dirichlet cell proportions.
5. Compare original and imputed SCT references.
6. Only then tune GEP filtering quantiles or replace the distance rule.

Filtering alone cannot recover biological variation that is absent from the
SCT reference or the cell-composition model. The workflow must first generate
samples with the correct biological support, then use filtering for refinement.
