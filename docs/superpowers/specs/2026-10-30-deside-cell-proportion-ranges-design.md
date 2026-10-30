# DeSide cell-proportion range constant

Record the observed raw minimum and maximum for each cell type in the supplied
DeSide prediction table as a static reference constant in `simutme.config`.

## Data and semantics

The source is `SimuTME_example/datasets/predicted_cell_proportion_by_DeSide/
Table_S3.csv`, containing 7,699 samples and 16 cell-type columns. Its SHA-256 is
`803f25d93982ff8d3272954e33f900ead44e4af4a7bbbac0cd948a7db6fc5378`.
Compute column-wise minima and maxima from the values as stored, without
normalizing, smoothing, or filtering. Ignore metadata columns.

Expose a dictionary mapping each cell type to its `(minimum, maximum)` pair.
Document the source, sample count, and raw-value semantics beside the constant.
It is a reference snapshot, not an automatic sampling prior or an alternative
to the existing row-normalized, input-specific segment cap calculation.

## Verification

Check all 16 ranges against the supplied CSV, confirm all values are finite and
within `[0, 1]`, and add a self-contained test that does not require the example
dataset to be installed. Existing simulation behavior remains unchanged.
