# src/simutme/config.py

# The default cell type list
DEFAULT_CELL_TYPES = [
    'B Cells', 'Plasma B cells', 'Non-plasma B cells', 'CD4 T', 'CD4 T conv', 'CD4 Treg',
    'CD8 T', 'CD8 T effector', 'CD8 T (GZMK high)', 'Double-neg-like T',
    'Cancer Cells', 'Epithelial Cells', 'Glioma Cells',
    'DC', 'mDC', 'pDC', 'Endothelial Cells', 'Fibroblasts', 'CAFs', 'Myofibroblasts',
    'Macrophages', 'Mast Cells', 'NK', 'Neutrophils', 'Monocytes', 'T Cells'
]

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

DEFAULT_LOG_BASE = 2
DEFAULT_DPI = 300
