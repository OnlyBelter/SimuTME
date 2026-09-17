# Mixed SCT Output-Type Alignment

## Problem

A Mixed dataset uses the global `cell_type2subtypes` mapping to generate cell
fractions. When its referenced SCT dataset was generated with
`selected_subtypes_by_cell_type`, the SCT profiles retain the parent label
(for example, `DC`) and do not expose the selected subtype (for example,
`pDC`) as a Mixed sampling label.

The Mixed workflow did not pass global `output_types` to `BulkGEPGenerator`.
Consequently, it expanded all configured subtypes and could request a
nonexistent subtype label from the referenced SCT dataset.

## Design

Pass `config.output_types` to `BulkGEPGenerator` for Mixed datasets. This
constrains Mixed fraction generation and sampling to the same output labels
used by the SCT reference.

No data files or sampling algorithms change. A regression test will assert
that the Mixed workflow forwards `output_types` to the generator.

## Verification

Run the focused workflow test module. The test will confirm that a
subtype-mapped cell type constrained to its parent output label is forwarded
as that parent label, preventing attempts to sample the unavailable subtype.
