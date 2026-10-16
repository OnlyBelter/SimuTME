# Cancer-stratified TCGA calibration and local GEP filtering

This specification extends the correlation-first workflow with a
cancer-stratified TCGA calibration split and a `knn_residual` acceptance
method. It describes the implementation proposed for review.

## Scope and decisions

The goal is to retain simulated profiles close to actual TCGA expression
profiles, including directions outside the retained reference PCA space.
This is not a guarantee of overlap in a separately fitted joint PCA plot.

- Stratify by **cancer type**.
- Use the existing `paths.tcga2cancer_type_file_path` annotation CSV.
- Reserve 20% of each cancer type's eligible samples for calibration.
- Split individual samples, as requested; do not group by patient.
- Fit PCA and the neighbor database on the remaining reference samples.
- Calibrate neighbor-distance and residual thresholds on held-out samples.
- Keep existing centroid methods available without changing their behavior.
- Preserve fixed gene list G, normalization, SCT generation, cell-fraction
  sampling, sorted-bulk imputation, and the unfiltered correlation source.

The inspected annotation contains 7,699 sample IDs across 19 cancer types,
with no duplicate sample IDs or missing cancer labels. It includes 29
patients with multiple sample rows. These counts describe the annotation,
not necessarily the final expression/annotation intersection.

## Sample alignment and stratified split

Apply the split before PCA fitting, reference-range estimation, neighbor
fitting, or target generation.

1. Load the configured expression reference and sample annotation. Match
   sample IDs exactly; do not truncate identifiers or infer missing labels.
2. Require unique reference sample IDs and unique annotation sample IDs.
   Fail on missing or blank cancer labels for reference samples. Ignore
   annotation rows absent from expression, and log their count.
3. Apply `filtering_ref_types`, when configured, to annotated reference
   samples. Fail if the resulting reference is empty.
4. Sort cancer labels and sample IDs. For each cancer type with n eligible
   samples, assign `ceil(holdout_fraction * n)` samples to calibration.
   Require at least one calibration and one fitting sample in every type;
   fail rather than silently drop an infeasible cancer type.
5. Sample without replacement using a dedicated reproducible seed. Derive
   a per-cancer seed from the configured seed and a stable hash of the
   cancer label, not Python's process-dependent `hash()`. Input row order
   and the presence of unrelated cancer strata must not change membership.
6. Persist disjoint fitting and calibration sample lists. Their union must
   equal the complete eligible reference cohort.

Rounding means the held-out fraction can be slightly above 20%. For
example, 79 ACC samples yield 16 held-out and 63 fitting samples. Report
actual counts and fractions for every cancer type.

All samples belonging to the same patient are allowed to cross the split,
consistent with the selected sample-level policy. Record the number of
patients represented in both sets. This split calibrates acceptance
thresholds; it is not a patient-independent final evaluation set.

## Configuration

Reuse existing reference paths and the dataset-over-global override rules.
Activate the new method explicitly; omitted method keys retain the existing
`median_gep` default.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `gep_filtering_method` | `median_gep` | Set to `knn_residual` for the new method |
| `gep_reference_holdout_fraction` | `0.2` | Held-out fraction within each cancer type |
| `gep_reference_holdout_seed` | `42` | Dedicated reference-split seed |
| `gep_knn_n_neighbors` | `10` | Neighbors used for each local-distance score |
| `gep_knn_distance_quantile` | `0.95` | Quantile of held-out neighbor-distance scores |
| `gep_pca_residual_quantile` | `0.95` | Quantile of held-out reconstruction residuals |
| `gep_max_generation_batches` | `100` | Maximum cumulative candidate batches in the new method |

The new method requires both `gep_filtering: true` and
`gep_filtering_in_pca_space: true`. With GEP filtering disabled, do not
split, fit a filtering PCA, or reject profiles, even if the configured
method name is `knn_residual`.

Continue using `gep_pca_n_components` for reference PCA. The example will
use `0.9` to retain 90% of fitting-reference variance. Integers remain
explicit component counts. `gep_filtering_n_pcs` remains read-only
metadata.

Use `gep_norm_ord` for neighbor distance, accepting 1 for Manhattan or 2
for Euclidean distance. The example will use 2. Reconstruction residuals
always use the Euclidean norm in normalized log-expression space.

`gep_filtering_quantile` remains specific to the existing centroid methods.
It does not define either new threshold and does not impose a lower
distance cutoff in `knn_residual`. Log this distinction when a shared
centroid quantile setting is present.

Validate a finite holdout fraction in (0, 1), finite calibration quantiles
in [0, 1], a nonnegative integer split seed, positive integer neighbor and
batch counts, a feasible split, at least two fitting-reference samples for
PCA, and enough fitting samples for the requested number of neighbors.
Do not silently reduce k.

## Fixed gene space and reference preparation

The fixed ordered list G is selected as in the approved correlation-first
design. Neither fitting nor held-out expression is used for additional
expression-range gene selection.

Normalize every eligible reference and simulated row independently over G
to 1,000,000, then compute `log2(TPM + 1)`. Preserve float64 precision for
fitting, calibration, and scoring. Keep the existing validation for
duplicates, missing selected genes, invalid expression, and nonpositive
selected-gene sums.

Only fitting-reference samples contribute to:

- PCA means, loadings, and retained component selection.
- The TCGA neighbor database.
- Optional per-gene reference quantile boundaries.

Held-out samples determine the two acceptance thresholds. Do not refit the
model on the full TCGA cohort after calibration; that would change the
space and invalidate the calibrated scores.

## Scores and acceptance

For normalized log-expression x, let mu be the fitting-reference PCA
mean, U the retained orthonormal loadings, and z its reference-PC scores.
Use raw PC scores without whitening.

```text
z = U^T (x - mu)

d_knn(x) = mean(||z - z_r||_p for r in the k nearest fitting TCGA samples)
x_reconstructed = mu + U z
d_residual(x) = ||x - x_reconstructed||_2
```

Fit the neighbor database on fitting-reference scores only. Held-out
queries cannot match themselves because they are absent from that
database. Simulated queries are never added to it.

Compute thresholds once, before the first target batch:

```text
t_knn = quantile(held_out_d_knn, gep_knn_distance_quantile)
t_residual = quantile(held_out_d_residual, gep_pca_residual_quantile)
```

Score all held-out samples for calibration, including those failing the
optional gene-range check. Do not censor calibration scores based on
either acceptance threshold.

Use pooled, sample-weighted held-out quantiles across the eligible cancer
types. Do not assign simulated samples an inferred cancer label or use
cancer-specific thresholds in this version. Report held-out performance
per cancer type so sparse or poorly represented strata remain visible.

A candidate survives only if:

1. It passes the optional per-gene range fraction check on G, using
   fitting-reference boundaries.
2. Its neighbor-distance score is at most `t_knn`.
3. Its reconstruction residual is at most `t_residual`.

Compute both scores for every range-surviving candidate. Report individual
pass counts and the cumulative conjunction separately.

No minimum distance is required. Closer candidates must not be rejected
because their distance is below a reference quantile.

Two marginal 95th-percentile thresholds do not imply 95% joint acceptance,
especially with the optional gene-range check. Record the observed
held-out pass counts for each check and their conjunction.

Batch neighbor queries and residual calculations to avoid materializing
all candidate/reference pairwise distances or a full-run reconstruction
matrix. Cache the fitted PCA, neighbor database, and thresholds across
generation batches.

## Generation and bounded failure

Keep accepted sample IDs, selected-gene normalized exports, and
cell-fraction rows synchronized. Preserve empty-batch and partial-survivor
handling, generation parameters, and existing seed handling.

Stop with a descriptive error if the new method reaches
`gep_max_generation_batches` before producing the requested sample count.
Include generated, range-surviving, neighbor-surviving, residual-surviving,
and exported counts in the error. Do not loosen thresholds automatically.
Persist the cumulative batch count so resuming does not reset this bound.

The independent unfiltered correlation source and existing centroid modes
do not acquire this new stopping rule.

## Artifacts, compatibility, and summaries

Use a new workflow/cache identity for `knn_residual`. Include G, the
normalization definition, reference expression identity, annotation labels,
reference subset, ordered fitting/calibration IDs, split algorithm version,
fraction, seed, PCA target, neighbor count, norm, both calibration
quantiles, range settings, generation settings, and cumulative batch bound.

Validate metadata before reusing models, thresholds, generated targets, or
intermediate batches. Existing v1 centroid artifacts remain supported for
their existing mode; they are not reused or relabeled as local-filter runs.
Fail on missing or incompatible new-mode metadata with an instruction to
use a new dataset name/output location. Do not delete historical artifacts.

Persist these dataset-local artifacts:

- `tcga_reference_split.csv`: sample ID, cancer type, and fitting/held-out role.
- `tcga_reference_split_summary.csv`: per-type counts and actual fractions.
- `tcga_heldout_calibration_scores.csv`: both scores and stage pass flags.
- The reference-only PCA and its compatible metadata.
- The fitting-reference neighbor database, thresholds, and metadata.
- Candidate scores and pass/export flags for range-surviving batches,
  appended with sample IDs without accumulating a full-run table in memory.

The summary must retain existing fields and add the split unit, fraction,
seed, fitting/held-out counts, patient overlap count, neighbor count, both
thresholds, marginal and joint calibration pass counts, per-stage
simulation counts, batch count/bound, and all new artifact paths.
Distinguish profiles passing all filters from profiles exported after
truncation to the requested sample count.

## Diagnostics

Keep `plot_pca_for_comparison` independent of acceptance.

- Preserve the joint comparison PCA using all eligible TCGA samples and
  accepted simulated profiles on the same normalized G.
- Add a reference-only PCA plot using the actual filtering PCA, with
  fitting TCGA, held-out TCGA, and simulated samples labeled separately.
  If the filtering PCA has fewer than two components, skip only this 2D
  plot and log the reason.
- Plot held-out and simulated score distributions with threshold markers.
- Keep comparison plotting dimensions separate from filtering dimensions.
- Never substitute a joint comparison PCA for the reference filtering
  model or recalibrate thresholds from simulated samples.

Diagnostics do not impose convex-hull membership or require identical
TCGA and simulated densities. They reveal whether the local and residual
constraints improve proximity in the reference expression space.

## Example migration and verification

Update the active `_0p9_0p005_0p995.yaml` configuration after spec approval.
Use `knn_residual`, cancer-stratified fraction 0.2, split seed 42, k = 10,
Euclidean neighbor distance, two 0.95 calibration quantiles, and PCA target
0.9. Give the target a fresh dataset name instead of reusing the existing
`Mixed_N5K_pan_cancer_Oct16_6_0p9_2PCA` output. Keep the earlier unfiltered
correlation-source entry reusable.

Focused tests must verify:

- Exact sample-ID alignment and descriptive errors for invalid annotations.
- Per-cancer ceiling counts, disjoint roles, and complete eligible coverage.
- Reproducibility under reordered input rows and unrelated cancer strata.
- Individual-sample policy and reporting of repeated-patient overlap.
- Split and calibration completed before the first generated target batch.
- Only fitting samples reach PCA fitting, range estimation, and neighbors.
- Held-out queries use the same normalization and model as simulated queries.
- Known toy neighbor scores, residuals, quantiles, and conjunction decisions.
- No self-neighbor leakage, simulated-data fitting, or post-generation
  gene selection.
- Empty and partially accepted batches retain correct sample IDs.
- Compatible cached/resumed runs preserve split membership and thresholds.
- Changed labels, split settings, G, PCA, or filtering settings reject
  incompatible artifacts without deleting them.
- Batch-bound failures are descriptive and resuming preserves the bound.
- Existing centroid methods, disabled filtering, supplied lists, and the
  unfiltered source preserve their documented behavior.
- Exports, summaries, calibration tables, and diagnostic model choices agree.
