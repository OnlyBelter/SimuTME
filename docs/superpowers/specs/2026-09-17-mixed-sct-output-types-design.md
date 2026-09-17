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

Pass `config.output_types` to `BulkGEPGenerator` for Mixed datasets. Derive a
separate fraction-label mapping from the original `cell_type2subtypes` mapping
and those output types. When a parent type is requested, such as `DC`, the
fraction mapping retains the parent label rather than expanding to its
subtypes. The original mapping remains available for SCT source-cell
restriction and subtype validation.

No data files or sampling algorithms change. Regression tests assert that the
Mixed workflow forwards `output_types` to the generator and that a
`DC: [pDC, mDC]` mapping produces a `DC` fraction column when `DC` is the
requested output label.

## Verification

Run the focused workflow test module. The test will confirm that a
subtype-mapped cell type constrained to its parent output label is forwarded
as that parent label, preventing attempts to sample the unavailable subtype.
