# Post-GEP quantile regeneration

Regenerate a post-simulation gene-quantile-filtered Mixed dataset without
repeating bulk simulation. The user changes the quantile configuration,
removes the existing filtered H5AD, and reruns the same configuration.

## Activation and safeguards

Apply this behavior only to `high_corr_gene_and_quantile_range` when the
primary accepted Mixed H5AD and generation artifacts pass existing validation.
Treat an absent post-filter H5AD as the explicit authorization to rebuild its
derived outputs. If the filtered H5AD exists, keep the existing identity
validation and refuse to overwrite an incompatible artifact.

Preserve the accepted H5AD, cell fractions, SCT provenance, fixed G gene list,
generation metadata, GEP filtering caches, and pre-filter PCA artifacts. Reuse
the accepted samples rather than simulating new candidates. If generation
configuration or source identities do not match, fail under the existing
validation rules instead of rebuilding from mismatched samples.

## Derived outputs and recovery

Before rebuilding, move existing post-filter metadata, G2 gene list, and
post-filter PCA artifacts into a uniquely named backup under the dataset result
directory. Preserve both standard PCA and sample-count PCA outputs, including
their metadata, figures, and loadings. Move only known post-filter artifacts;
never follow paths outside the expected output directories. Retain existing
backups and log their locations. An initial run with no previous artifacts
does not need an archive.

Track an in-progress rebuild with a persistent marker and an explicit artifact
move plan. The completed-run cache must not treat a marked rebuild as complete.
If interrupted, resume or safely restart the same rebuild without losing the
archive, mistaking partial output for a valid cache, or touching unrelated
artifacts. Refuse conflicting configuration changes during an incomplete
rebuild with an actionable error.

Write the new filtered H5AD through a temporary file and publish it only after
the write completes. Regenerate its identity metadata, G2 gene list, and
post-filter PCA for the new quantiles. Keep the Mixed summary visibly
incomplete during regeneration and publish a complete summary only after all
requested outputs succeed. When the quantiles select no genes or normalization
fails, do not publish a new completed result.

## Filtering and validation

Retain the existing comparison of accepted Mixed sample quantiles against the
resolved reference quantile interval. Normalize each selected-gene profile to
1,000,000 and export `log2(TPM + 1)`. Do not change profile-level filtering or
the original accepted samples.

Add regression tests for changed and unchanged quantiles after H5AD removal,
existing incompatible H5AD protection, archived artifact preservation,
standard and incremental PCA regeneration, interrupted rebuild recovery,
generation-identity mismatch, and unchanged cache reuse. Verify summary
identities, retained sample IDs, gene lists, and normalization. Document the
workflow and run the relevant tests plus a broader regression suite.
