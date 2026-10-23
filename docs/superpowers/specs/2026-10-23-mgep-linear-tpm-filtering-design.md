# Linear-TPM mGEP filtering

## Goal

Add `gep_filtering_method: "mgep"` for filtering simulated Mixed profiles by
their L1 distance from the coordinate-wise median reference GEP in linear TPM
space. Keep the existing `median_gep` behavior unchanged.

## Existing behavior

The correlation-first `FixedGeneFilter` normalizes reference and simulated
profiles over the selected gene set, then transforms them to
`log2(TPM + 1)`. With `filtering_in_pca_space: true`, `median_gep` calculates
the median and distances in PCA coordinates. With PCA disabled, it calculates
them in log-expression gene space. Neither mode computes distances in linear
TPM space.

## Configuration

Add `mgep` as a supported value of `gep_filtering_method`. It always operates
in linear expression space, so require `filtering_in_pca_space: false` and
fail validation with a clear message if PCA-space filtering is enabled.

Update the Mixed example to select `mgep` and explicitly disable PCA-space
filtering. Use the upper value of `gep_filtering_quantile` as the reference
distance cutoff. Set its lower value to `null` or `0`; `mgep` applies no
lower-distance cutoff.

## Calculation

Use the resolved reference cohort and final selected gene set `G`. Normalize
each reference and simulated profile over `G` to a total of 1,000,000, as the
existing workflow does. Keep these values linear; do not apply a log
transformation or PCA projection.

For reference profiles `r_i` and genes `g` in `G`, calculate the mGEP
coordinate-wise:

`m_g = median_i(r_ig)`

Calculate reference distances `d_i = sum_g |r_ig - m_g|` and set the
acceptance cutoff to the configured upper quantile of these distances.
Calculate each simulated profile's L1 distance to the same mGEP and retain it
when its distance is less than or equal to the cutoff.

## Compatibility and outputs

Do not change `median_gep`, `mean_gep`, or the local `knn` methods. Include
`mgep` in filtering identity and output-directory naming so its cached
artifacts cannot collide with existing centroid methods. Report `mgep` in the
dataset summary.

## Tests

Add tests that verify the linear-TPM coordinate median, reference L1
distribution and upper-quantile cutoff, inclusive threshold behavior, and
rejection above the cutoff. Confirm that PCA-space use fails clearly and that
existing `median_gep` behavior remains unchanged.
