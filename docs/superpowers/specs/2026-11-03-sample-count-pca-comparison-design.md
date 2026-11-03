# Sample-count PCA comparisons for Mixed datasets

## Goal

Let a Mixed dataset produce PCA comparison figures at selected simulated
sample counts without loading and fitting PCA on the entire large simulated
matrix. Each comparison includes all TCGA reference samples.

## Configuration

Add `n_sample_for_pca` as an optional global parameter with a dataset-level
override. Accept `null`, one positive integer, or a nonempty list of unique
positive integers. `null` preserves the existing one-figure PCA behavior.
When configured, each integer is the number of simulated samples in that
comparison; TCGA reference samples are additional.

Resolve and validate the setting before simulation against the dataset's
configured `n_samples`. Before PCA, also validate against the actual generated
H5AD sample count. Reject invalid values and any requested count greater than
the corresponding simulated dataset size with a clear error.

Add `pca_sample_seed` for reproducible selection and
`incremental_pca_batch_size` (default 5,000) to control fitting and transform
batch sizes. The seed defaults to 42, and the batch size must be at least
twice the number of PCA components. Generate one seeded permutation of
simulated row indices; use prefixes of that permutation so smaller requested
subsets are nested within larger ones.

## PCA behavior and outputs

For every requested N, select N simulated profiles, fit IncrementalPCA on
those profiles together with the complete TCGA reference, transform those
same samples, and plot the selected simulated profiles alongside all TCGA
samples. Process H5AD rows and PCA transforms in batches; do not materialize
the full simulated matrix or all 400K PCA coordinates.

Save each N-specific model, coordinates, metadata, loadings, and figure in a
new `pca_by_sample_count/` directory beneath the dataset's existing result
directory. Include N in each artifact filename. Return the per-N artifact
paths in a new summary field while preserving the existing singular PCA
summary fields and file layout when `n_sample_for_pca` is `null`.

## Example configuration

The example Mixed dataset contains 100 generated samples, so demonstrate
`n_sample_for_pca: [10, 50, 100]` there. Larger lists such as
`[10000, 50000, 100000, 200000]` belong on datasets whose configured and
generated sample counts are at least 200,000.

## Tests

- Validate null, scalar, and list configuration values, including invalid,
  duplicate, and over-size requests.
- Verify deterministic, nested selections for a fixed seed.
- Verify each PCA run includes exactly N simulated samples and all reference
  samples, and produces isolated artifacts under the new folder.
- Verify the IncrementalPCA implementation reads and transforms data in
  configured-size batches.
- Preserve existing single-PCA behavior and output paths when the setting is
  null.
