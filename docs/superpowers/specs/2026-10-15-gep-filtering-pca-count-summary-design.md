# GEP Filtering PCA Count Summary Design

## Goal

Record both the configured GEP-filtering PCA variance target and the actual
number of principal components used to form the filtering space.

## Design

- Capture `pca_model.n_components_` after the GEP-filtering PCA model is loaded
  or fitted in `BulkGEPGenerator`.
- Add `gep_pca_n_components` and `gep_filtering_n_pcs` to Mixed dataset
  summaries. The first records the configured target; the second records the
  fitted component count.
- Set the component count to null when GEP filtering is disabled or is not
  performed in PCA space.
- For the existing Oct15_3 summary, use the cached `pca_model_0.9` model's
  actual metadata: target `0.9`, count `576`. Do not label this prior run as
  `0.85`, which is the current config setting.

## Verification

- Test that a fitted or loaded GEP PCA model records its actual component
  count.
- Test that summary JSON includes the target and count, and leaves the count
  null when PCA is not used.
- Confirm the backfilled summary agrees with the cached model metadata.
