import math

from simutme.config import DESIDE_TCGA_CELL_PROPORTION_RANGES


def test_deside_tcga_cell_proportion_ranges_record_raw_extrema():
    expected_maxima = {
        "Plasma B cells": 0.239,
        "Non-plasma B cells": 0.704,
        "CD4 T": 0.106,
        "CD8 T effector": 0.161,
        "CD8 T (GZMK high)": 0.164,
        "Double-neg-like T": 0.137,
        "Cancer Cells": 0.996,
        "DC": 0.165,
        "Endothelial Cells": 0.506,
        "CAFs": 0.716,
        "Myofibroblasts": 0.479,
        "Macrophages": 0.612,
        "Mast Cells": 0.050,
        "NK": 0.029,
        "Neutrophils": 0.081,
        "Monocytes": 0.069,
    }
    assert set(DESIDE_TCGA_CELL_PROPORTION_RANGES) == set(expected_maxima)
    for cell_type, expected_maximum in expected_maxima.items():
        minimum, maximum = DESIDE_TCGA_CELL_PROPORTION_RANGES[cell_type]
        assert math.isfinite(minimum) and math.isfinite(maximum)
        assert 0 <= minimum <= maximum <= 1
        assert minimum == (0.028 if cell_type == "Cancer Cells" else 0.0)
        assert maximum == expected_maximum
