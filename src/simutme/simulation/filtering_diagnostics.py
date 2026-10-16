"""Diagnostic plots for fitting, held-out, and simulated TCGA filter scores."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utility import safe_pca_transform

logger = logging.getLogger(__name__)


def plot_local_filter_diagnostics(space, simulated_log, dataset_name):
    outputs = {}
    colors = {"Fitting TCGA": "#e69f00", "Held-out TCGA": "#009e73", "Simulated": "#0072b2"}
    markers = {"Fitting TCGA": ".", "Held-out TCGA": "^", "Simulated": "o"}
    with plt.rc_context({
        "font.family": "DejaVu Sans", "text.usetex": False, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    }):
        if space.pca.n_components_ >= 2:
            parts = []
            for label, expression in (
                ("Fitting TCGA", space.reference_log),
                ("Held-out TCGA", space.heldout_log),
                ("Simulated", simulated_log),
            ):
                for start in range(0, len(expression), space.score_chunk_size):
                    chunk = expression.iloc[start:start + space.score_chunk_size]
                    coords = safe_pca_transform(space.pca, chunk, label="Reference-only diagnostic PCA")
                    part = pd.DataFrame(coords[:, :2], index=chunk.index, columns=["PC1", "PC2"])
                    part["source"] = label
                    parts.append(part)
            coordinates = pd.concat(parts)
            coordinate_path = space.result_dir / "reference_only_PCA_coordinates.csv"
            coordinates.to_csv(coordinate_path, index_label="sample_id", float_format="%.17g")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for label, group in coordinates.groupby("source", sort=False):
                ax.scatter(group.PC1, group.PC2, s=9, alpha=0.35,
                           color=colors[label], marker=markers[label], label=f"{label} (n={len(group)})")
            ax.set(
                xlabel=f"PC1 ({space.pca.explained_variance_ratio_[0]:.1%} fitting TCGA variance)",
                ylabel=f"PC2 ({space.pca.explained_variance_ratio_[1]:.1%} fitting TCGA variance)",
                title="TCGA-only filtering PCA",
            )
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = space.result_dir / f"reference_only_PCA_PC1_PC2.{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                outputs[f"reference_pca_{'plot' if suffix == 'png' else 'pdf'}_file"] = str(path)
            plt.close(fig)
            outputs["reference_pca_data_file"] = str(coordinate_path)
        else:
            logger.info("Skipping reference-only 2D PCA plot: filtering PCA has fewer than two PCs.")

        accepted_parts = []
        if space.candidate_scores_path.exists():
            for chunk in pd.read_csv(space.candidate_scores_path, chunksize=10000):
                accepted_parts.append(chunk.loc[chunk.exported, ["knn_distance", "pca_residual"]])
        accepted = pd.concat(accepted_parts) if accepted_parts else pd.DataFrame(
            columns=["knn_distance", "pca_residual"], dtype=float
        )
        diagnostic_residual = not space.residual_filter_enabled
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.3 if diagnostic_residual else 3))
        for ax, column, threshold, title, xlabel in zip(
            axes, ("knn_distance", "pca_residual"),
            (space.knn_threshold, space.residual_threshold),
            ("Neighbor proximity", "PCA residual (diagnostic only)" if diagnostic_residual
             else "PCA reconstruction residual"),
            (f"Mean {space.gep.knn_n_neighbors}-neighbor distance (L{space.gep.norm_ord})",
             "Residual L2 norm in log-expression space"),
        ):
            heldout = space.calibration_scores[column].to_numpy()
            simulated = accepted[column].to_numpy()
            bins = np.histogram_bin_edges(np.concatenate([heldout, simulated]), bins=40)
            ax.hist(heldout, bins=bins, density=True, histtype="step",
                    color=colors["Held-out TCGA"], linewidth=1.5, label=f"Held-out TCGA (n={len(heldout)})")
            if len(simulated):
                ax.hist(simulated, bins=bins, density=True, histtype="step",
                        color=colors["Simulated"], linestyle="--", linewidth=1.5,
                        label=f"Exported simulated (n={len(simulated)})")
            if column == "pca_residual" and diagnostic_residual:
                ax.axvline(threshold, color="0.4", linestyle="-.",
                           label="Residual percentile (diagnostic only)")
            else:
                ax.axvline(threshold, color="black", linestyle=":", label="Acceptance threshold")
            ax.set(xlabel=xlabel, ylabel="Density", title=title)
            ax.set_xlim(left=0)
        handles, labels = axes[0].get_legend_handles_labels()
        if diagnostic_residual:
            diagnostic_handles, diagnostic_labels = axes[1].get_legend_handles_labels()
            handles.append(diagnostic_handles[-1])
            labels.append(diagnostic_labels[-1])
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1),
                   ncol=2 if diagnostic_residual else 3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.85 if diagnostic_residual else 0.9))
        for suffix in ("png", "pdf"):
            path = space.result_dir / f"local_filter_score_distributions.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            outputs[f"score_distribution_{'plot' if suffix == 'png' else 'pdf'}_file"] = str(path)
        plt.close(fig)
    logger.info("Saved local-filter diagnostics for %s.", dataset_name)
    return outputs
