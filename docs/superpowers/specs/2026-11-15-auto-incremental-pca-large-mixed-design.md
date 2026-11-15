# Automatic IncrementalPCA for Large Mixed Datasets

## Problem

The comparison PCA workflow currently uses full-matrix PCA unless
`n_sample_for_pca` is explicitly configured. Large Mixed datasets can exceed
the integer-indexing limit of the LAPACK SVD used by scikit-learn PCA, even
after successful sample generation. Some PCA paths also materialize the
complete H5AD matrix before fitting.

## Goals

- Automatically use `IncrementalPCA` when the actual comparison input contains
  more than 50,000 simulated samples.
- Process all simulated samples in batches in automatic mode, keeping the full
  dataset comparison semantics.
- Preserve explicit `n_sample_for_pca` counts and the existing full-PCA
  behavior at 50,000 samples or fewer.
- Apply the same bounded-memory behavior to the post-GEP gene-quantile PCA.
- Retain the complete selected TCGA reference cohort in each comparison.

## Behavior

1. Determine the sample count from the actual H5AD used for each PCA, not only
   from the configured target count.
2. If `n_sample_for_pca` is set, continue to run the requested sample counts
   through the existing IncrementalPCA workflow.
3. If it is unset and the actual H5AD has more than 50,000 samples, run one
   IncrementalPCA comparison over all its samples.
4. If it is unset and the input has 50,000 samples or fewer, retain the
   existing full-PCA path and artifact naming.
5. Use backed H5AD reads and bounded row batches for large-data PCA. Do not
   construct a full simulated expression DataFrame or dense copy before
   selecting the IncrementalPCA path.
6. Apply the same selection and batching rules to fixed-filter, gene-filtered,
   unfiltered, and post-GEP gene-quantile comparison PCA paths. Post-filter
   artifacts must remain distinct from pre-filter artifacts.
7. Preserve existing normalization, ordered gene selection, sample IDs,
   figure content, PCA metadata, and summary reporting. Keep the existing
   batch-size and matrix-size validation for IncrementalPCA inputs.

## Compatibility

Explicit `n_sample_for_pca` configuration remains authoritative. With the
parameter unset, datasets of at most 50,000 samples retain current behavior.
For larger datasets, the output uses the existing sample-count artifact
structure and records the full simulated sample count in `pca_runs`.

## Tests     

- Verify the automatic decision is off at 50,000 and on above 50,000.
- Verify automatic mode processes all simulated samples in batches.
- Verify explicit sample counts remain authoritative.
- Verify a large post-GEP quantile PCA takes the bounded-memory path and
  writes separate artifacts.
- Verify small-data PCA retains its existing full-PCA path and artifact names.
