# Optional legacy gene-list type design

Allow SimuTME configurations to omit the legacy `gene_list_type` when they use
the newer `gene_filtering_type` setting.

## Behavior

Give `SimuConfig.gene_list_type` the existing default value
`"high_corr_gene_and_quantile_range"`. Keep the current precedence unchanged:
an explicit `gene_filtering_type` remains authoritative, while older configs
that omit it continue to fall back to `gene_list_type`.

Do not add `gene_list_type` back to the pan-cancer example config. Its explicit
`gene_filtering_type` and `gene_filtering_quantile_range` already define the
active gene-filtering behavior.

## Verification

Add a config-loader regression test that omits `gene_list_type`, supplies
`gene_filtering_type`, and confirms `SimuConfig.from_yaml` loads successfully
with the explicit type intact. Also confirm that a config omitting both fields
receives the legacy default.
