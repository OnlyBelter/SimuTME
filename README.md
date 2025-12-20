# SimuTME: Bulk RNA-seq Simulator for Tumor Microenvironment

**SimuTME** is a Python package designed to simulate bulk RNA-seq data specifically for the **Tumor Microenvironment (TME)**. It facilitates the benchmarking, validation, and training of cellular deconvolution methods by generating high-quality synthetic datasets with known ground-truth cell proportions.

SimuTME addresses the complexity of TME data—such as high tumor purity, sparse immune cell infiltration, and compositional biases—by providing diverse and mathematically rigorous sampling strategies.

---

## 🚀 Streamlined Simulation Workflow

SimuTME offers a unified, configuration-driven workflow that handles the entire simulation pipeline in three automated steps:
1.  **Generation:** Synthesizing bulk GEPs from single-cell data using specific sampling strategies.
2.  **Filtering:** Removing low-quality genes based on correlation with TCGA data and quantile filtering.
3.  **Validation:** Automatically performing PCA to compare the simulated manifold against real TCGA bulk data.

### 1. Configuration (`config_example.yaml`)
Manage all paths, biological constants, and dataset parameters in a single YAML file.

```yaml
# config.yaml
paths:
  simu_bulk_dir: "./datasets/simulated_bulk"
  tcga_merged_tpm_file_path: "./data/tcga_tpm.csv"
  # ... other paths

parameters:
  gene_quantile_range: [0.005, 0.5, 0.995]
  pca_n_components: 0.9

datasets:
  # Define multiple datasets to generate in one batch
  Mixed_N10K_segment:
    n_samples: 8000
    sampling_method: "segment"  # Uses the 'segment' strategy
    filtering: true
```

### 2. Execution
Run the entire pipeline with a few lines of Python code.

```python
from simutme import SimuConfig, run_bulk_simulation
import yaml

# 1. Load Configuration
config = SimuConfig.from_yaml('config.yaml')

# 2. Load Dataset Definitions
with open('config.yaml', 'r') as f:
    datasets_to_run = yaml.safe_load(f).get('datasets', {})

# 3. Run Workflow
for dataset_name, params in datasets_to_run.items():
    # Automatically handles Generation -> Filtering -> PCA Plotting
    run_bulk_simulation(dataset_name, params, config)
```

## 🧬 Data Simulation & Sampling Strategies

To capture the heterogeneity of real-world tumor samples, SimuTME implements multiple strategies for generating cell type fractions. These methods ensure that downstream models are robust to various biological scenarios, from immune-cold tumors (sparse) to high-purity samples.

### Sampling Methods for Cellular Proportions
The following sampling functions (for cellular proportions) are available in `simutme.simulation`:

#### 1. Sparse Random Sampling (`sparse_random_sampling`)
**Source:** Adapted from *Menden et al. (2020)* (Scaden).

In the TME, specific immune subsets (e.g., B cells, NK cells) are often completely absent from a given tissue sample. To simulate this "sparsity," this method performs a leave-out procedure:
1.  For each sample, a random number of cell types are selected to be **absent** (fraction = 0).
2.  Random fractions are generated for the remaining present cell types.
3.  This is crucial for training Deep Learning models to predict true zeros and avoid over-predicting rare cell types.

#### 2. Dirichlet Sampling (`dirichlet_sampling`)
The **Dirichlet distribution** is the standard probability distribution for compositional data (vectors that sum to 1). It allows for precise control over the "concentration" of cell types via the $\alpha$ parameter:
*   **$\alpha = 1.0$:** Uniform distribution (unbiased).
*   **$\alpha < 1.0$:** Sparse distribution (samples tend toward the "corners" of the simplex).
*   **$\alpha > 1.0$:** Dense distribution (samples tend toward equal proportions).

#### 3. Segment-Random Sampling (`seg_random_sampling`)
**Use Case:** High Tumor Purity / Dominant Cell Types.

Tumor biopsies often consist largely of malignant cells with a smaller fraction of infiltrating immune and stromal cells. This method simulates such scenarios by:
1.  Assigning a **dominant fraction** (e.g., 80% Tumor Cells) to one randomly selected cell type.
2.  Distributing the remaining fraction (e.g., 20%) randomly among the background cell types.

#### 4. Segment Sampling (`segment_sampling`)
**Source:** Adapted from *Xiong et al. (2024)* (DeSide).

Uses an iterative "stick-breaking" approach to generate distributions with high variance. This creates skewed datasets where cell proportions decay rapidly, testing the model's ability to detect cell types across a wide dynamic range.

#### 5. Random (Dense) Sampling (`random_sampling`)
A baseline method generating "dense" compositions where all cell types are likely to be present, derived from a uniform distribution normalized to sum to 1.


### Sampling Methods for sctGEPs
SimuTME also provides strategies for sampling single cells to construct bulk single-cell-type GEPs (sctGEPs).

#### 1. Random Sampling (`random`)
Randomly samples `n_base` single cells from each cell type to create the sctGEPs.

#### 2. Nearest Neighbors Sampling (`n_neighbors`)
Sample 1 cell randomly, then select its `n_base - 1` nearest neighbors based on Euclidean distance across all cells within the same type. This captures local transcriptional similarity.

---

## 📚 References

The sampling methodologies implemented in SimuTME are based on the following foundational works in cellular deconvolution:

> **Xiong, X., Liu, Y., et al.** DeSide: A unified deep learning approach for cellular deconvolution of tumor microenvironment. *Proc. Natl. Acad. Sci. U.S.A.* **121**, e2407096121 (2024). DOI: [10.1073/pnas.2407096121](https://www.pnas.org/doi/10.1073/pnas.2407096121)

> **Menden, K., Marouf, A., Oller, S. et al.** Deep learning-based cell composition analysis from tissue expression profiles. *Sci Adv* **6**, eaba2619 (2020). DOI: [10.1126/sciadv.aba2619](https://doi.org/10.1126/sciadv.aba2619)
