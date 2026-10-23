import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union


def flatten_cell_type2subtype(cell_type2subtype: Dict[str, List[str]]) -> List[str]:
    """Flatten the output cell-type list while preserving subtype names."""
    cell_type_with_subtype = []
    for k, v in cell_type2subtype.items():
        if len(v) == 1 and v[0] == k:
            cell_type_with_subtype.append(k)
        else:
            cell_type_with_subtype.extend(v)
    return cell_type_with_subtype


def restrict_cell_type2subtype_to_output_types(
    cell_type2subtype: Dict[str, List[str]],
    output_types: Optional[List[str]],
) -> Dict[str, List[str]]:
    """Build a fraction-label mapping constrained to the requested outputs."""
    if output_types is None:
        return {cell_type: list(subtypes) for cell_type, subtypes in cell_type2subtype.items()}

    output_type_set = set(output_types)
    resolved_mapping: Dict[str, List[str]] = {}
    for cell_type, subtypes in cell_type2subtype.items():
        if cell_type in output_type_set:
            # A parent output label represents the parent, not all of its subtypes.
            resolved_mapping[cell_type] = [cell_type]
            continue

        selected_subtypes = [subtype for subtype in subtypes if subtype in output_type_set]
        if selected_subtypes:
            resolved_mapping[cell_type] = selected_subtypes

    return resolved_mapping


def get_default_dirichlet_alpha(cell_type2subtype: Dict[str, List[str]]) -> Dict[str, float]:
    """Build the default symmetric Dirichlet prior (all ones)."""
    return {cell_type: 1.0 for cell_type in flatten_cell_type2subtype(cell_type2subtype)}


def estimate_dirichlet_alpha_from_pred_cell_props(
    pred_cell_prop_file_path: str | Path,
    cell_types: List[str],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Estimate Dirichlet concentration parameters from predicted cell proportions.

    The input CSV can contain extra metadata columns, but it must contain one
    numeric column for every requested cell type.
    """
    try:
        import dirichlet
    except ImportError as exc:
        raise ImportError(
            'Estimating "dirichlet_alpha" from "pred_cell_prop_file_path" requires the '
            '`dirichlet` package. Please install it before using Mixed datasets with '
            '"sampling_method_prop: dirichlet" and "pred_cell_prop_file_path".'
        ) from exc

    pred_cell_prop_file_path = Path(pred_cell_prop_file_path)
    if not pred_cell_prop_file_path.exists():
        raise FileNotFoundError(
            f'"pred_cell_prop_file_path" does not exist: {pred_cell_prop_file_path}'
        )

    pred_cell_prop = pd.read_csv(pred_cell_prop_file_path)
    missing_cell_types = [cell_type for cell_type in cell_types if cell_type not in pred_cell_prop.columns]
    if missing_cell_types:
        raise ValueError(
            f'"pred_cell_prop_file_path" is missing required cell-type columns: {missing_cell_types}. '
            f'Expected columns for all output cell types: {cell_types}.'
        )

    pred_cell_prop = pred_cell_prop.loc[:, cell_types].apply(pd.to_numeric, errors='coerce')
    if pred_cell_prop.isnull().any().any():
        raise ValueError(
            f'"pred_cell_prop_file_path" contains non-numeric values in the requested cell-type columns: '
            f'{pred_cell_prop_file_path}'
        )
    if (pred_cell_prop < 0).any().any():
        raise ValueError(
            f'"pred_cell_prop_file_path" contains negative cell proportions: {pred_cell_prop_file_path}'
        )

    pred_cell_prop = pred_cell_prop + eps
    row_sums = pred_cell_prop.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError(
            f'"pred_cell_prop_file_path" contains rows with non-positive total proportion after smoothing: '
            f'{pred_cell_prop_file_path}'
        )
    pred_cell_prop = pred_cell_prop.div(row_sums, axis=0)

    alpha_hat = dirichlet.mle(pred_cell_prop.to_numpy(), method="meanversion")
    return {cell_type: float(alpha_value) for cell_type, alpha_value in zip(cell_types, alpha_hat)}


def estimate_cell_prop_maxima_from_pred_cell_props(
    pred_cell_prop_file_path: str | Path,
    cell_types: List[str],
) -> tuple[Dict[str, float], dict]:
    """Read, row-normalize, and summarize global per-type empirical maxima."""
    path = Path(pred_cell_prop_file_path)
    if not path.exists():
        raise FileNotFoundError(f'pred_cell_prop_file_path does not exist: {path}')
    if not cell_types or len(cell_types) != len(set(cell_types)):
        raise ValueError("Segment fraction caps require unique output cell-type labels.")
    frame = pd.read_csv(path)
    missing = [cell_type for cell_type in cell_types if cell_type not in frame.columns]
    if missing:
        raise ValueError(
            f'"pred_cell_prop_file_path" is missing required cell-type columns: {missing}. '
            f"Expected columns for all segment output types: {cell_types}."
        )
    values = frame.loc[:, cell_types].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=np.float64)
    if array.shape[0] == 0:
        raise ValueError(f'"pred_cell_prop_file_path" has no sample rows: {path}')
    if not np.isfinite(array).all():
        raise ValueError(f'"pred_cell_prop_file_path" contains nonnumeric or non-finite cell proportions: {path}')
    if (array < 0).any():
        raise ValueError(f'"pred_cell_prop_file_path" contains negative cell proportions: {path}')
    row_sums = array.sum(axis=1)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise ValueError(f'"pred_cell_prop_file_path" contains rows with non-positive totals: {path}')
    normalized = array / row_sums[:, None]
    maxima = normalized.max(axis=0)
    caps = {cell_type: float(value) for cell_type, value in zip(cell_types, maxima)}
    metadata = {
        "segment_fraction_cap_source": str(path.resolve()),
        "segment_fraction_cap_source_n_rows": len(normalized),
        "segment_fraction_cap_scope": "global-normalized-row-maximum",
        "segment_fraction_cap_row_normalization": "divide-by-row-sum",
        "segment_fraction_max_caps": caps,
    }
    return caps, metadata


def format_segment_fraction_caps(caps: Dict[str, float]) -> str:
    """Format empirical caps for logs without reducing stored precision."""
    formatted_caps = ", ".join(
        f"{cell_type!r}: {value:.3g}" for cell_type, value in caps.items()
    )
    return f"{{{formatted_caps}}}"


def merge_cell_prop_prior_with_maxima(cell_types, cell_prop_prior, maxima):
    """Intersect optional segment prior ranges with empirical upper caps."""
    cell_types = list(cell_types)
    if not cell_types or len(cell_types) != len(set(cell_types)):
        raise ValueError("Segment fraction caps require unique output cell-type labels.")
    if set(maxima) != set(cell_types):
        raise ValueError("Empirical segment caps must cover every output cell type exactly.")
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in maxima.values()):
        raise ValueError("Empirical segment fraction maxima must be finite and in [0, 1].")
    if sum(maxima.values()) < 1.0:
        raise ValueError("Empirical segment fraction maxima sum to less than one; no valid composition exists.")

    manual = {} if cell_prop_prior is None else cell_prop_prior
    if not isinstance(manual, dict):
        raise ValueError("cell_prop_prior must be a mapping when empirical caps are enabled.")
    unknown = set(manual).difference(cell_types)
    if unknown:
        raise ValueError(
            f"cell_prop_prior contains cell types absent from segment outputs: {sorted(unknown)}."
        )
    combined = {}
    minimum_total = 0.0
    maximum_total = 0.0
    for cell_type in cell_types:
        bounds = manual.get(cell_type, (0.0, 1.0))
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) for value in bounds)):
            raise ValueError(f'cell_prop_prior for "{cell_type}" must be a finite [minimum, maximum] pair.')
        lower, upper = (float(value) for value in bounds)
        if not 0 <= lower <= upper <= 1:
            raise ValueError(f'cell_prop_prior for "{cell_type}" must satisfy 0 <= minimum <= maximum <= 1.')
        upper = min(upper, maxima[cell_type])
        if lower > upper:
            raise ValueError(
                f'cell_prop_prior minimum for "{cell_type}" exceeds its empirical maximum cap.'
            )
        combined[cell_type] = (lower, upper)
        minimum_total += lower
        maximum_total += upper
    if minimum_total > 1.0:
        raise ValueError("Combined segment fraction prior minima sum to more than one.")
    if maximum_total < 1.0:
        raise ValueError("Combined segment fraction prior maxima sum to less than one.")
    return combined


def segment_sampling(n_samples: int,
                     cell_types: List[str],
                     max_value: int = 10000,
                     sample_prefix: str = 'seg',
                     cell_prop_prior: Optional[Dict[str, tuple]] = None,
                     max_attempts: Optional[int] = None) -> pd.DataFrame:
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
    attempts = 0

    # We use a loop because of the rejection sampling (cell_prop_prior)
    while len(valid_samples) < n_samples:
        attempts += 1
        if max_attempts is not None and attempts > max_attempts:
            raise RuntimeError(
                f"Segment fraction cap sampling accepted {len(valid_samples)}/{n_samples} "
                f"proposals after {max_attempts} attempts. Check the predicted maxima "
                "and any cell_prop_prior constraints."
            )
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
    df = df.round(4)
    df.attrs["segment_sampling_stats"] = {
        "requested": n_samples,
        "attempted": attempts,
        "accepted": len(valid_samples),
        "rejected": attempts - len(valid_samples),
    }
    return df


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
                       sample_prefix: str = 'dirichlet',
                       rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
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
        rng: Optional generator reused across batches for seeded sampling.

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
    if rng is None:
        fracs = np.random.dirichlet(alpha_values, size=n_samples)
    else:
        fracs = rng.dirichlet(alpha_values, size=n_samples)

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


def generate_cell_fraction(sampling_method: Union[str, List[str]],
                           n_cell_frac: int,
                           cell_type2subtype: Dict[str, List[str]],
                           sample_prefix: str = 'sample',
                           cell_prop_prior: Optional[Dict] = None,
                           dirichlet_alpha: Optional[Union[float, Dict[str, float]]] = None,
                           rng: Optional[np.random.Generator] = None,
                           max_segment_sampling_attempts: Optional[int] = None) -> pd.DataFrame:
    """
    Generate cell proportions for simulated bulk GEPs using various sampling strategies.

    This function supports generating cell fractions using a single method, a specific list of methods,
    or a 'mixed' mode that utilizes all available strategies. When multiple methods are used,
    the total number of samples (`n_cell_frac`) is distributed evenly among them.

    :param sampling_method: The strategy to generate cell fractions.
        - str: One of 'segment', 'seg_random', 'sparse', 'random', 'dirichlet'.
        - str: 'mixed' to use all available methods.
        - list: A list of specific methods, e.g., ['segment', 'dirichlet'].
    :param n_cell_frac: The total number of samples (cell fractions) to generate.
    :param cell_type2subtype: A dictionary mapping major cell types to their subtypes.
        Used to flatten the cell type list. e.g., {'T cells': ['CD4', 'CD8'], ...}
    :param sample_prefix: The prefix for the sample indices in the returned DataFrame.
    :param cell_prop_prior: (Optional) A dictionary defining prior proportions for specific cell types.
        Only used by the 'segment' method.
    :param rng: (Optional) Generator used by the 'dirichlet' method.

    :return: pd.DataFrame
        A DataFrame where rows are samples and columns are cell types.
        Values represent the proportion of each cell type (summing to 1.0 per row).
    """

    # 1. Flatten cell types (handle subtypes)
    cell_type_with_subtype = flatten_cell_type2subtype(cell_type2subtype)

    # 2. Determine which methods to use
    valid_methods = ['segment', 'seg_random', 'sparse', 'random', 'dirichlet']
    methods_to_use = []

    if isinstance(sampling_method, str):
        if sampling_method == 'mixed':
            methods_to_use = valid_methods
        elif sampling_method in valid_methods:
            methods_to_use = [sampling_method]
        else:
            raise ValueError(f"Invalid sampling_method: '{sampling_method}'. "
                             f"Supported: {valid_methods} or 'mixed'.")
    elif isinstance(sampling_method, list):
        # Validate list inputs
        for m in sampling_method:
            if m not in valid_methods:
                raise ValueError(f"Invalid method in list: '{m}'. Supported: {valid_methods}")
        methods_to_use = sampling_method
    else:
        raise TypeError("sampling_method must be a string or a list of strings.")

    # 3. Distribute n_cell_frac evenly among selected methods
    n_methods = len(methods_to_use)
    base_n = n_cell_frac // n_methods
    remainder = n_cell_frac % n_methods

    # Create a list of sample counts: [base+1, base+1, base, base...] to handle remainder
    n_samples_per_method = [base_n + 1 if i < remainder else base_n for i in range(n_methods)]

    generated_dfs = []

    # 4. Loop through methods and generate data
    for method, n_count in zip(methods_to_use, n_samples_per_method):
        if n_count == 0:
            continue

        # Update prefix to distinguish samples from different methods (optional but recommended)
        # e.g., sample_segment_0, sample_dirichlet_10
        current_prefix = f"{sample_prefix}_{method}"

        df = None
        if method == 'segment':
            df = segment_sampling(n_samples=n_count,
                                  max_value=10000,
                                  sample_prefix=current_prefix,
                                  cell_types=cell_type_with_subtype,
                                  cell_prop_prior=cell_prop_prior,
                                  max_attempts=max_segment_sampling_attempts)

        elif method == 'seg_random':
            df = seg_random_sampling(n_samples=n_count,
                                     sample_prefix=current_prefix,
                                     cell_types=cell_type_with_subtype)

        elif method == 'sparse':
            df = sparse_random_sampling(n_samples=n_count,
                                        cell_types=cell_type_with_subtype,
                                        sample_prefix=current_prefix)

        elif method == 'random':
            df = uni_random_sampling(n_samples=n_count,
                                     cell_types=cell_type_with_subtype,
                                     sample_prefix=current_prefix)

        elif method == 'dirichlet':
            df = dirichlet_sampling(n_samples=n_count,
                                    cell_types=cell_type_with_subtype,
                                    alpha=1.0 if dirichlet_alpha is None else dirichlet_alpha,
                                    sample_prefix=current_prefix,
                                    rng=rng)

        if df is not None:
            generated_dfs.append(df)

    # 5. Concatenate results
    if not generated_dfs:
        return pd.DataFrame(columns=cell_type_with_subtype)

    final_df = pd.concat(generated_dfs)
    if len(methods_to_use) == 1 and methods_to_use[0] == "segment":
        final_df.attrs["segment_sampling_stats"] = generated_dfs[0].attrs.get(
            "segment_sampling_stats", {}
        )

    # Ensure the order matches the requested n_cell_frac (though concat usually preserves order)
    return final_df
