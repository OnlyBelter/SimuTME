# src/simutme/config.py

# The default cell type list in the pooled single-cell RNA-seq dataset, including major cell types and subtypes.
DEFAULT_CELL_TYPES = [
    'B Cells', 'Plasma B cells', 'Non-plasma B cells', 'CD4 T', 'CD4 T conv', 'CD4 Treg',
    'CD8 T', 'CD8 T effector', 'CD8 T (GZMK high)', 'Double-neg-like T',
    'Cancer Cells', 'Epithelial Cells', 'Glioma Cells',
    'DC', 'mDC', 'pDC', 'Endothelial Cells', 'Fibroblasts', 'CAFs', 'Myofibroblasts',
    'Macrophages', 'Mast Cells', 'NK', 'Neutrophils', 'Monocytes', 'T Cells'
]

# Raw column-wise extrema from DeSide Table_S3.csv (7,699 TCGA samples).
DESIDE_TCGA_CELL_PROPORTION_RANGES = {
    'Plasma B cells': (0.0, 0.239),
    'Non-plasma B cells': (0.0, 0.704),
    'CD4 T': (0.0, 0.106),
    'CD8 T effector': (0.0, 0.161),
    'CD8 T (GZMK high)': (0.0, 0.164),
    'Double-neg-like T': (0.0, 0.137),
    'Cancer Cells': (0.028, 0.996),
    'DC': (0.0, 0.165),
    'Endothelial Cells': (0.0, 0.506),
    'CAFs': (0.0, 0.716),
    'Myofibroblasts': (0.0, 0.479),
    'Macrophages': (0.0, 0.612),
    'Mast Cells': (0.0, 0.050),
    'NK': (0.0, 0.029),
    'Neutrophils': (0.0, 0.081),
    'Monocytes': (0.0, 0.069),
}

DEFAULT_CORE_MARKER_GENES = {'Cancer Cells': ['KRT19', 'KRT18', 'KRT8', 'EPCAM'],
                             'CD4 T': ['BATF', 'ICOS', 'CD4', 'IL7R', 'FOXP3', 'TIGIT'],
                             'CD8 T': ['CD8A', 'CD8B'],
                             'T Cells': ['CD2', 'CD3D', 'CD3E'],
                             'B Cells': ['BANK1', 'CD79A', 'FCRL5', 'MS4A1'],
                             'DC': ['GZMB', 'CCR7', 'LAMP3', 'IRF7', 'IRF8'],
                             'Endothelial Cells': ['CLDN5', 'ENG', 'PLVAP', 'VWF'],
                             'Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'MYL9'],
                             'Macrophages': ['AIF1', 'CD14', 'CD68', 'MS4A7'],
                             'Mast Cells': ['CPA3', 'HPGDS', 'GATA2'],
                             'NK': ['GNLY', 'NKG7', 'KLRD1'],
                             'Neutrophils': ['CSF3R', 'CXCR2', 'FPR1', 'SLC25A37']}

# Cell type to subtypes in the pooled single-cell RNA-seq dataset. And will be used to simulate the single-cell-type GEPs (sctGEPs).
# If no subtypes, the same cell type name will be used as the subtype name.
CELL_TYPE2SUBTYPE_SCT = {'B Cells': ['Non-plasma B cells', 'Plasma B cells'],
                         'CD4 T': ['CD4 T conv', 'CD4 Treg'], 'CD8 T': ['CD8 T (GZMK high)', 'CD8 T effector'],
                         'DC': ['mDC', 'pDC'], 'Endothelial Cells': ['Endothelial Cells'],
                         'Cancer Cells': ['Epithelial Cells', 'Glioma Cells'],
                         'Fibroblasts': ['CAFs', 'Myofibroblasts'], 'Macrophages': ['Macrophages'],
                         'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
                         'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}

DEFAULT_LOG_BASE = 2
DEFAULT_DPI = 300
