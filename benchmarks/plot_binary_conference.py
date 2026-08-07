#!/usr/bin/env python3
"""Create compact conference figures from a validated large binary benchmark run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCHMARKS_ROOT = Path(__file__).resolve().parent
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from plot_binary_results import (  # noqa: E402
    DEFAULT_HEATMAP_METRICS,
    add_optimizer_labels,
    build_relative_performance_table,
    default_manifest_path,
    default_result_paths,
    load_manifest,
    load_summaries,
    prepare_metric_columns,
    validate_results,
)


DEFAULT_SIZES = (1, 5, 10, 20, 30)
SELECTED_CONFIGS = {
    "heuristic": "BH_MP_MI_AMR-0_ACP-1",
    "genetic": "BG_ACP-1_AMR-1",
    "milp": "BM_ACP-1_AMR-1",
}
EQUIVALENT_HEURISTIC = "BH_MA_MP_MI_AMR-0_ACP-0"
F1_TOLERANCE = 1e-12
DEFAULT_LABELS = {
    "MiMiC_v1": "MiMiC",
    "MiMiC_actual": "MiMiC_actual",
    "BH_MA_MP_MI_AMR-0_ACP-0": "MiMiC-equivalent",
    "BH_MP_MI_AMR-0_ACP-1": "Heuristic_v2",
    "BG_ACP-1_AMR-1": "Genetic",
    "BM_ACP-1_AMR-1": "MILP",
}
CONFERENCE_ORDER = ["MiMiC", "MiMiC_actual", "MiMiC-equivalent", "Heuristic_v2", "Genetic", "MILP"]
QUALITY_METRICS = ["TPR/recall", "PPV/precision", "F1_score"]
QUALITY_LABELS = {
    "TPR/recall": r"Recall: $\frac{TP}{TP + FN}$",
    "PPV/precision": r"Precision: $\frac{TP}{TP + FP}$",
    "F1_score": r"F1 score: $\frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$",
}
HEATMAP_ORDER = [
    "ACC", "BA", "F1_score", "Jaccard", "MCC", "TNR/specificity",
    "TPR/recall", "PPV/precision", "FNR", "FPR", "logical_run_runtime_seconds",
]
LEGACY_PALETTE = {
    "MiMiC": "#291f66",
    "MiMiC_actual": "#6f6f6f",
    "MiMiC-equivalent": "#9624ed",
    "Heuristic_v2": "#99cfff",
    "Genetic": "#00ffa1",
    "MILP": "#ff211c",
}


def conference_labels(frame: pd.DataFrame, labels: Mapping[str, str] = DEFAULT_LABELS) -> pd.DataFrame:
    frame = add_optimizer_labels(prepare_metric_columns(frame))
    frame["conference_label"] = frame["optimizer_label"].map(labels)
    return frame


def rank_optimizer_configs(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank every available optimizer configuration by F1, recall, precision."""
    frame = conference_labels(frame)
    candidates = frame[frame["strategy"].isin(["heuristic", "genetic", "milp"])].copy()
    if candidates.empty:
        raise ValueError("No heuristic, genetic, or MILP rows are available for ranking.")

    grouped = (
        candidates.groupby(["strategy", "optimizer_label"], dropna=False)
        .agg(
            mean_F1_score=("F1_score", "mean"),
            mean_TPR_recall=("TPR/recall", "mean"),
            mean_PPV_precision=("PPV/precision", "mean"),
            mean_runtime_seconds=("logical_run_runtime_seconds", "mean"),
            scenario_rows=("scenario_index", "size"),
            consortium_sizes=("consortia_size", lambda values: ",".join(
                str(value) for value in sorted(pd.unique(values), key=float)
            )),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["strategy", "mean_F1_score", "mean_TPR_recall", "mean_PPV_precision", "mean_runtime_seconds"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    grouped["rank_within_strategy"] = grouped.groupby("strategy", sort=False).cumcount() + 1
    grouped["selected_historical"] = grouped["optimizer_label"].isin(
        [key for key in DEFAULT_LABELS if key.startswith(("BH_", "BG_", "BM_"))]
    )
    return grouped


def verify_selected_configs(frame: pd.DataFrame, sizes: Sequence[int] = DEFAULT_SIZES) -> pd.DataFrame:
    """Verify the explicitly selected configuration is F1-optimal per class."""
    ranking = rank_optimizer_configs(frame[frame["consortia_size"].isin(list(sizes))])
    failures = []
    for strategy, selected in SELECTED_CONFIGS.items():
        match = ranking[(ranking["strategy"] == strategy) & (ranking["optimizer_label"] == selected)]
        if match.empty:
            failures.append(f"{strategy}: selected configuration {selected!r} is unavailable")
            continue
        row = match.iloc[0]
        if int(row["rank_within_strategy"]) != 1:
            best = ranking[(ranking["strategy"] == strategy) & (ranking["rank_within_strategy"] == 1)].iloc[0]
            failures.append(
                f"{strategy}: selected {selected} mean F1={row['mean_F1_score']:.12g}; "
                f"best is {best['optimizer_label']} mean F1={best['mean_F1_score']:.12g}"
            )
    if failures:
        raise ValueError("Conference optimizer selection verification failed: " + "; ".join(failures))
    ranking["selected_explicitly"] = ranking.apply(
        lambda row: SELECTED_CONFIGS.get(row["strategy"]) == row["optimizer_label"], axis=1
    )
    return ranking


def equivalent_heuristic_matches(frame: pd.DataFrame, sizes: Sequence[int] = DEFAULT_SIZES) -> bool:
    """Return whether the configured equivalent heuristic matches MiMiC v1 F1 values."""
    prepared = conference_labels(frame)
    subset = prepared[prepared["consortia_size"].isin(list(sizes))]
    mimic = subset[subset["optimizer_label"] == "MiMiC_v1"]
    equivalent = subset[subset["optimizer_label"] == EQUIVALENT_HEURISTIC]
    keys = ["scenario_index", "consortia_size"]
    if mimic.empty or equivalent.empty:
        return False
    joined = mimic[keys + ["F1_score"]].merge(
        equivalent[keys + ["F1_score"]], on=keys, suffixes=("_mimic", "_equivalent")
    )
    return len(joined) == len(mimic) == len(equivalent) and np.allclose(
        joined["F1_score_mimic"], joined["F1_score_equivalent"], atol=F1_TOLERANCE, rtol=0
    )


def select_conference_rows(
    frame: pd.DataFrame,
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    labels: Mapping[str, str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    frame = conference_labels(frame, labels)
    frame = frame[frame["consortia_size"].isin(list(sizes))].copy()
    include_equivalent = equivalent_heuristic_matches(frame, sizes)
    effective_labels = dict(labels)
    if not include_equivalent:
        effective_labels.pop(EQUIVALENT_HEURISTIC, None)
        effective_labels.pop("BH_MA_MP_MI_AMR-0-ACP-0", None)
    available = set(frame["optimizer_label"].dropna())
    missing = [label for label in effective_labels if label not in available]
    if missing:
        raise ValueError(f"Selected configurations are missing from the run: {', '.join(missing)}")
    selected = frame[frame["optimizer_label"].isin(effective_labels)].copy()
    selected["conference_label"] = selected["optimizer_label"].map(effective_labels)
    selected["conference_label"] = pd.Categorical(
        selected["conference_label"], categories=CONFERENCE_ORDER, ordered=True
    )
    return selected


def _runtime_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("run_id", "job_key", "consortia_size") if column in frame.columns]
    return frame.drop_duplicates(keys) if keys else frame


def plot_conference_metrics(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    palette: Mapping[str, Any] | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Conference plotting requires the `benchmark` extra.") from exc

    frame = frame.copy()
    palette = palette or LEGACY_PALETTE
    if isinstance(frame["conference_label"].dtype, pd.CategoricalDtype):
        frame["conference_label"] = frame["conference_label"].astype(object)
    present_labels = set(frame["conference_label"].dropna().astype(str))
    hue_order = [label for label in CONFERENCE_ORDER if label in present_labels]
    runtime = _runtime_rows(frame)
    figure = plt.figure(figsize=(12, 12), dpi=300)
    outer = figure.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.30)
    top = outer[0].subgridspec(1, 1)
    bottom = outer[1].subgridspec(3, 1, hspace=0.1)
    axes = [figure.add_subplot(top[0, 0])]
    axes.extend(figure.add_subplot(bottom[index, 0]) for index in range(3))
    sizes = sorted(frame["consortia_size"].dropna().unique(), key=float)
    size_labels = [str(int(size)) if float(size).is_integer() else str(size) for size in sizes]
    size_position = {size: index for index, size in enumerate(sizes)}
    runtime = runtime.copy()
    runtime["size_position"] = runtime["consortia_size"].map(size_position)

    sns.lineplot(
        data=runtime,
        x="size_position",
        y="logical_run_runtime_seconds",
        hue="conference_label",
        hue_order=hue_order,
        marker="o",
        linestyle="--",
        linewidth=3,
        palette=palette,
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    positive_runtime = pd.to_numeric(runtime["logical_run_runtime_seconds"], errors="coerce")
    positive_runtime = positive_runtime[positive_runtime > 0]
    if not positive_runtime.empty:
        axes[0].set_ylim(
            max(positive_runtime.min() * 0.75, 1e-3),
            positive_runtime.max() * 10.0,
        )
    axes[0].set_ylabel("Execution Time (s)", fontsize=16)
    axes[0].set_xticks(range(len(sizes)), size_labels)
    axes[0].set_xlabel("Consortia Size", fontsize=16)

    for axis, metric in zip(axes[1:], QUALITY_METRICS):
        sns.boxplot(
        data=frame,
            x="consortia_size",
            y=metric,
            hue="conference_label",
            hue_order=hue_order,
            palette=palette,
            fliersize=0,
            linewidth=0.7,
            legend=False,
            ax=axis,
        )
        sns.swarmplot(
            data=frame,
            x="consortia_size",
            y=metric,
            hue="conference_label",
            hue_order=hue_order,
            palette=palette,
            dodge=True,
            size=3,
            edgecolor="black",
            linewidth=0.4,
            legend=False,
            ax=axis,
        )
        axis.set_ylabel(QUALITY_LABELS[metric], fontsize=16)
        axis.set_xlabel("")
        if axis is not axes[-1]:
            axis.tick_params(axis="x", labelbottom=False)

    handles = [
        plt.Line2D([0], [0], marker="o", color=palette[label], linestyle="", label=label)
        for label in CONFERENCE_ORDER
        if label in present_labels
    ]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        bbox_to_anchor=(0.5, 0.71),
        frameon=False,
        ncol=5,
        fontsize=14,
        markerscale=1.5,
    )
    for axis in axes:
        if axis.legend_ is not None:
            axis.legend_.remove()
    axes[-1].set_xlabel("Consortia Size", fontsize=16)
    axes[-1].set_xticks(range(len(sizes)), size_labels)
    # Explicit margins avoid tight_layout's incompatibility warning with the
    # nested GridSpec and leave room for the long metric labels.
    figure.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.10)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    plt.close(figure)
    return output_path


def plot_conference_heatmap(table: pd.DataFrame, output_path: Path) -> Path:
    """Plot raw relative scores using the legacy two-facet heatmap layout."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise RuntimeError("Conference plotting requires the `benchmark` extra.") from exc

    quality = table.reindex(index=[metric for metric in HEATMAP_ORDER if metric in table.index])
    quality = quality.reindex(columns=[label for label in CONFERENCE_ORDER if label in table.columns])
    relative_table = quality.astype(float).replace(0, np.nan)
    display_names = {
        "ACC": "ACC",
        "BA": "BA",
        "F1_score": "F1_score",
        "Jaccard": "Jaccard",
        "MCC": "MCC",
        "TNR/specificity": "TNR/specificity",
        "TPR/recall": "TPR/recall",
        "PPV/precision": "precision",
        "FNR": "FNR",
        "FPR": "FPR",
        "logical_run_runtime_seconds": "runtime",
    }
    higher_keys = [metric for metric in HEATMAP_ORDER if metric in relative_table.index
                   and metric not in {"FNR", "FPR", "logical_run_runtime_seconds"}]
    lower_keys = [metric for metric in ("FNR", "FPR", "logical_run_runtime_seconds")
                  if metric in relative_table.index]
    relative_table = relative_table.rename(index=display_names)
    labels = relative_table.round(2).astype(object).where(relative_table.notna(), "")
    cmap = LinearSegmentedColormap.from_list(
        "conference_relative",
        [(0, "#99cfff"), (0.5, "#f7f7f7"), (0.7, "#00ffa1"), (1, "#00ad7d")],
        N=256,
    )
    higher = [display_names[metric] for metric in higher_keys]
    lower = [display_names[metric] for metric in lower_keys]
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    gridspec = figure.add_gridspec(2, 1, height_ratios=[len(higher), len(lower)])
    axes = [figure.add_subplot(gridspec[0, 0]), figure.add_subplot(gridspec[1, 0])]
    for axis, metrics, title, show_x in zip(
        axes, (higher, lower), ("Higher = better", "Lower = better"), (False, True)
    ):
        subset = relative_table.loc[metrics]
        subset_labels = labels.loc[metrics]
        sns.heatmap(
            subset,
            annot=subset_labels,
            fmt="",
            cmap=cmap,
            center=1,
            vmin=0.0,
            vmax=1.8,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 18},
            ax=axis,
        )
        axis.set_title(title, fontsize=18, pad=8)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="y", rotation=0, labelsize=18)
        if show_x:
            axis.tick_params(axis="x", rotation=25, labelsize=18)
        else:
            axis.set_xticklabels([])
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.8), cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axes, location="right", label="Relative Score")
    colorbar.ax.tick_params(labelsize=18)
    colorbar.ax.yaxis.label.set_size(18)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    plt.close(figure)
    return output_path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results-dir", type=Path, required=True)
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--external-results", type=Path,
                     help="CSV produced by external_mimic/run_mimic_pibc.sh.")
    cli.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    cli.add_argument("--summary-only", action="store_true")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = default_result_paths(args.results_dir)
    if not paths:
        raise SystemExit(f"No binary CSV summaries found under {args.results_dir}.")
    manifest_path = args.manifest or default_manifest_path(args.results_dir, paths)
    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    frame = validate_results(load_summaries(paths), manifest)
    if args.external_results is not None:
        external = pd.read_csv(args.external_results)
        required = {"optimizer_label", "scenario_index", "consortia_size", "selected_count", "F1_score"}
        missing = sorted(required - set(external.columns))
        if missing:
            raise SystemExit(f"External MiMiC results are missing columns: {', '.join(missing)}")
        external["profile"] = frame["profile"].iloc[0]
        external["run_id"] = frame["run_id"].iloc[0]
        external["scenario_name"] = external.get("scenario_name", external["scenario_index"].map(
            dict(frame[["scenario_index", "scenario_name"]].drop_duplicates().values)
        ))
        frame = pd.concat([frame, external], ignore_index=True, sort=False)
    selected = select_conference_rows(frame, sizes=args.sizes)
    output_dir = args.output_dir or args.results_dir.parent / "conference_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(output_dir / "selected_conference_rows.csv", index=False)
    ranking = verify_selected_configs(frame, args.sizes)
    ranking.to_csv(output_dir / "optimizer_selection_audit.csv", index=False)
    print(f"Wrote optimizer audit to {output_dir / 'optimizer_selection_audit.csv'}")
    print(f"Wrote selected rows to {output_dir / 'selected_conference_rows.csv'}")

    if not args.summary_only:
        plot_conference_metrics(selected, output_dir / "selected_optimizers_metrics.png")
        relative = selected.copy()
        relative["optimizer_label"] = relative["conference_label"].astype(str)
        heatmap_metrics = [metric for metric in DEFAULT_HEATMAP_METRICS if metric in relative.columns]
        table = build_relative_performance_table(
            relative,
            comparator="MiMiC",
            metrics=heatmap_metrics,
        )
        table = table.reindex(columns=[label for label in CONFERENCE_ORDER if label in table.columns])
        table.to_csv(output_dir / "selected_optimizers_relative_performance.csv")
        plot_conference_heatmap(table, output_dir / "selected_optimizers_relative_heatmap.png")
        print(f"Wrote conference plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
