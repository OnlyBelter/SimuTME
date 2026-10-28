# Mixed workflow figure update

## Goal

Update the Mixed bulk RNA-seq workflow figure to reflect current gene
selection, filtering, and output stages while preserving its simple vertical
layout.

## Design

Update both `scripts/generate_workflow_figures.py` and the generated
`docs/figures/workflow_bulk.svg` so regenerating the figure retains the
updated workflow. Preserve the current color palette, box style, and general
level of detail.

Show eight broad steps:

1. Load the YAML configuration and resolve the SCT and filtering references.
2. When filtering is enabled, select and persist the shared gene set `G`
   before candidate generation. Apply the configured reference-variance
   screen before supplied-list or high-correlation selection when either
   filtering mode is enabled; this includes GEP-level filtering without
   gene-level filtering. The high-correlation method selects genes whose
   expression correlates with cell proportions across samples in at least one
   cell type under the configured thresholds. Keep the figure concise: label
   the selection “reference variance + high-correlation genes” and omit
   correlation details and thresholds.
3. Generate cell-type proportions and sample SCT profiles.
4. Mix candidate bulk profiles in linear expression space.
5. Apply the same gene set `G` to the reference and simulated profiles and
   normalize them for comparison.
6. Optionally filter candidate samples using the configured reference method
   and expression space.
7. Save output profiles as `log2(TPM + 1)` with matching cell fractions and
   summaries.
8. Optionally create comparison PCA plots and diagnostics.

Update the bulk figure height to fit the added step without clipping boxes.
Leave the SCT workflow figure and unrelated pending changes untouched.

## Verification

- Regenerate figures with `python scripts/generate_workflow_figures.py`.
- Confirm the Mixed SVG has the new high-correlation selection step before
  candidate generation and has no clipped content.
- Confirm `workflow_sct.svg` is unchanged.
- Parse the SVG as XML and check the diff for accidental changes.
