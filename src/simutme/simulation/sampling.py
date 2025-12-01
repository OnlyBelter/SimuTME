import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union


def segment_sampling(n_samples: int,
                     cell_types: List[str],
                     max_value: int = 10000,
                     sample_prefix: str = 'seg',
                     cell_prop_prior: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """
    Generate cell fractions using an iterative 'remainder' approach.
    This creates a distribution where some cell types have large proportions
    while others have very small proportions (sparsity).

    Args:
        n_samples: Total number of samples.
        cell_types: List of cell type names.
        max_value: Precision for integer generation (e.g., 10000).
        sample_prefix: Prefix for sample index.
        cell_prop_prior: Dictionary defining valid ranges, e.g., {'B Cells': (0.0, 0.1)}.
                         Samples violating this are rejected.
    """
    n_cell_types = len(cell_types)
    valid_samples = []

    # We use a loop because of the rejection sampling (cell_prop_prior)
    while len(valid_samples) < n_samples:
        # 1. Generate fragments using iterative subtraction
        # This logic mimics the original: pick a random chunk of the *current remainder*
        frag_for_one_sample = []
        current_max = max_value

        for _ in range(n_cell_types - 1):
            if current_max > 1:
                # Random integer between 0 and current_max
                _frag = np.random.randint(0, current_max)
                current_max -= _frag
                frag_for_one_sample.append(_frag)
            else:
                frag_for_one_sample.append(0)

        # The last one gets the rest
        frag_for_one_sample.append(current_max)
        assert np.sum(frag_for_one_sample) == max_value

        # Normalize to 0-1
        fractions = np.array(frag_for_one_sample) / max_value

        # Shuffle to ensure the "first" cell type isn't always the largest
        np.random.shuffle(fractions)

        # Map to cell types
        sample_dict = dict(zip(cell_types, fractions))

        # 2. Check Priors (Rejection Sampling)
        is_valid = True
        if cell_prop_prior:
            for ct, (min_p, max_p) in cell_prop_prior.items():
                val = sample_dict.get(ct, 0)
                if not (min_p <= val <= max_p):
                    is_valid = False
                    break

        if is_valid:
            valid_samples.append(sample_dict)

    index = [f"{sample_prefix}_{i + 1}" for i in range(n_samples)]
    df = pd.DataFrame(valid_samples, index=index, columns=cell_types)
    return df.round(4)


def seg_random_sampling(n_samples: int,
                        cell_types: List[str],
                        sample_prefix: str = 'seg_random') -> pd.DataFrame:
    """
    Generate fractions by combining a dominant segment with random sampling.
    Uses np.random.rand() as requested.
    """
    n_cell_types = len(cell_types)
    all_samples = []

    for _ in range(n_samples):
        # 1. Generate the first (dominant) segment
        first_segment = np.random.rand()

        # 2. Generate others using random uniform
        others_fraction = np.random.rand(n_cell_types - 1)

        # 3. Scale others to sum to (1 - first_segment)
        if others_fraction.sum() == 0:
            others_fraction = np.ones(n_cell_types - 1)  # Avoid div by zero

        others_fraction = others_fraction / others_fraction.sum() * (1 - first_segment)

        # 4. Combine
        frag_for_one_sample = np.concatenate(([first_segment], others_fraction))

        # 5. Final check to ensure sum is 1 (handling float precision)
        frag_for_one_sample = frag_for_one_sample / frag_for_one_sample.sum()

        # 6. Shuffle
        np.random.shuffle(frag_for_one_sample)

        all_samples.append(dict(zip(cell_types, frag_for_one_sample)))

    index = [f"{sample_prefix}_{i + 1}" for i in range(n_samples)]
    df = pd.DataFrame(all_samples, index=index, columns=cell_types)
    return df.round(4)


def uni_random_sampling(n_samples: int,
                    cell_types: List[str],
                    sample_prefix: str = 's_random',
                    fixed_range: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """
    Generate pure random cell fractions using np.random.rand().
    """
    n_cell_types = len(cell_types)
    all_samples = []

    for _ in range(n_samples):
        if fixed_range is None:
            # Standard random: Uniform(0,1) then normalize
            fracs = np.random.rand(n_cell_types)
        else:
            # Constrained random
            fracs = []
            for ct in cell_types:
                low, high = fixed_range.get(ct, (0, 100))
                # Assuming range is 0-100 integer based on original code context
                val = np.random.randint(low, high) / 100.0
                fracs.append(val)
            fracs = np.array(fracs)

        # Normalize to sum to 1
        if fracs.sum() == 0:
            fracs = np.ones(n_cell_types)
        fracs = fracs / fracs.sum()

        all_samples.append(dict(zip(cell_types, fracs)))

    index = [f"{sample_prefix}_{i + 1}" for i in range(n_samples)]
    df = pd.DataFrame(all_samples, index=index, columns=cell_types)
    return df.round(4)


def dirichlet_sampling(n_samples: int,
                       cell_types: List[str],
                       alpha: Union[float, Dict[str, float]] = 1.0,
                       sample_prefix: str = 'dirichlet') -> pd.DataFrame:
    """
    Generate cell fractions using the Dirichlet distribution.
    This is the mathematically standard way to generate compositional data.

    Args:
        n_samples: Total number of samples.
        cell_types: List of cell type names.
        alpha: Concentration parameter.
               - If float: alpha=1.0 -> Uniform over the simplex.
                           alpha<1.0 -> Sparse (corners).
                           alpha>1.0 -> Dense (center).
               - If Dict: {'CellTypeA': 0.5, 'CellTypeB': 2.0}.
                 Missing keys default to 1.0.
        sample_prefix: Prefix for sample index.

    Returns:
        pd.DataFrame: Samples x Cell Types
    """
    n_cell_types = len(cell_types)

    # Determine the alpha vector based on input type
    if isinstance(alpha, (float, int)):
        # Create an array where all values are the same alpha
        alpha_values = np.full(n_cell_types, float(alpha))

    elif isinstance(alpha, dict):
        # STRICT ORDERING:
        # We must ensure the alpha values correspond exactly to the order of `cell_types`.
        # We iterate through `cell_types` and look up the value in the `alpha` dict.
        alpha_values = np.array([alpha.get(ct, 1.0) for ct in cell_types])

    else:
        raise TypeError(f"Alpha must be a float or a dictionary, got {type(alpha)}")

    # Generate data
    # frac shape: (n_samples, n_cell_types)
    fracs = np.random.dirichlet(alpha_values, size=n_samples)

    index = [f"{sample_prefix}_{i + 1}" for i in range(n_samples)]
    df = pd.DataFrame(fracs, index=index, columns=cell_types)

    return df.round(4)


def sparse_random_sampling(n_samples: int,
                           cell_types: List[str],
                           sample_prefix: str = 'sparse') -> pd.DataFrame:
    """
    Generate sparse cell fractions as described in the Scaden method (K. Menden, et al., Sci Adv 6, eaba2619, 2020).

    Methodology:
    1. For each sample, randomly determine a number of cell types to be ABSENT.
       (Constraint: At least one cell type must remain present).
    2. Randomly select which specific cell types are absent.
    3. Generate random fractions (np.random.rand) for the remaining (present) cell types.
    4. Normalize to sum to 1.

    Args:
        n_samples: Total number of samples.
        cell_types: List of cell type names.
        sample_prefix: Prefix for sample index.

    Returns:
        pd.DataFrame: Samples x Cell Types (with many zeros).
    """
    n_cell_types = len(cell_types)
    all_samples = []

    for _ in range(n_samples):
        # 1. Determine how many cell types to KEEP (Present)
        # The text says "random number... selected to be absent", with "at least one... constituting the sample".
        # So, n_keep can range from 1 to n_cell_types.
        n_keep = np.random.randint(1, n_cell_types + 1)

        # 2. Select which specific cell types to KEEP
        # replace=False ensures we pick unique indices
        keep_indices = np.random.choice(n_cell_types, size=n_keep, replace=False)

        # 3. Generate random fractions for the kept types
        # Using np.random.rand() as requested
        present_fracs = np.random.rand(n_keep)

        # 4. Normalize to sum to 1
        present_fracs = present_fracs / present_fracs.sum()

        # 5. Construct the full vector (zeros for absent types)
        full_vector = np.zeros(n_cell_types)
        full_vector[keep_indices] = present_fracs

        all_samples.append(dict(zip(cell_types, full_vector)))

    index = [f"{sample_prefix}_{i + 1}" for i in range(n_samples)]
    df = pd.DataFrame(all_samples, index=index, columns=cell_types)
    return df.round(4)
