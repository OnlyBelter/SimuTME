# mGEP lower distance quantile

## Goal

Allow GEP-filtered Mixed datasets using `mgep` to apply both configured
reference-distance quantiles. Preserve the existing upper-only behavior when
the lower quantile is `null` or `0`.

## Filtering behavior

Calculate the reference mGEP and L1 distances on selected-gene linear TPM as
before. For `gep_filtering_quantile: [0.05, 0.95]`, accept a simulated
profile when its distance is greater than or equal to the reference 5th
percentile and less than or equal to the reference 95th percentile. Both
boundaries are inclusive.

Treat a lower value of `null` or `0` as no lower cutoff, matching existing
`mgep` behavior. Require a finite upper quantile in `[0, 1]` and, when
provided, a finite lower quantile in `[0, upper]`. Keep the requirement for
PCA-space filtering to be disabled.

## Outputs and compatibility

Log the effective lower and upper cutoffs when both are applied. The GEP
distance-distribution figure shows both cutoff lines for a positive lower
quantile, and shows only the upper line for `null` or `0`.

Retain the current filtering identity: it already records both configured
quantiles. Existing compatible `null` and `0` caches retain their original
semantics. A different quantile setting must not reuse an incompatible cache.
Do not change the user's example YAML.

## Verification

Test lower and upper boundary inclusion, rejection on either side, the
`null` and `0` upper-only behavior, invalid quantiles, and the diagnostic
figure's cutoff lines. Cover compatible cached outputs and changed-quantile
cache rejection. Update the Mixed workflow documentation and changelog.
