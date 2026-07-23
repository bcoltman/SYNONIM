#!/usr/bin/env python3
"""Summarize and plot binary benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_ROOT = REPO_ROOT / "benchmarks" / "outputs"
RESULTS_ROOT = OUTPUTS_ROOT
OPTIMIZER_LABEL_COLUMN = "optimizer_label"
DEFAULT_HEATMAP_COMPARATOR = "MiMiC_v1"
DEFAULT_PALETTE = ["#291f66", "#99cfff", "#00ffa1", "#ff211c", "#9624ed", "#00ad7d"]
METRICS = [
    "objective",
    "runtime_seconds",
    "TPR/recall",
    "PPV/precision",
    "F1_score",
    "Jaccard",
    "MCC",
    "selection_jaccard",
    "exact_selection_recovery",
]
DEFAULT_HEATMAP_METRICS = [
    "ACC",
    "BA",
    "F1_score",
    "Jaccard",
    "MCC",
    "TNR/specificity",
    "TPR/recall",
    "PPV/precision",
    "FNR",
    "FPR",
    "runtime_seconds",
    "selection_jaccard",
]
LOWER_IS_BETTER = {"FDR", "FNR", "FOR", "FPR", "objective_gap", "runtime_seconds"}
HEURISTIC_PARAMETER_COLUMNS = [
    "absence_cover_penalty",
    "absence_match_reward",
    "mask_covered_absent_features",
    "mask_covered_present_features",
    "mask_covered_isolate_features",
]
GROUP_COLUMNS = [
    "profile",
    "strategy",
    "method",
    OPTIMIZER_LABEL_COLUMN,
    "consortia_size",
    "absence_cover_penalty",
    "absence_match_reward",
    "heuristic_mask_key",
]


def safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def format_config_value(value: Any) -> str | None:
    if not has_value(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return None
    return str(int(number)) if number.is_integer() else f"{number:g}"


def optimizer_label(row: Mapping[str, Any]) -> str:
    method = str(row.get("method")) if has_value(row.get("method")) else ""
    strategy = str(row.get("strategy")).lower() if has_value(row.get("strategy")) else ""

    if strategy == "mimic_v1" or method == "MiMiC_v1":
        return "MiMiC_v1"

    if strategy == "heuristic" or method.startswith("BinaryHeuristic"):
        parts = ["BH"]
        mask_key = row.get("heuristic_mask_key")
        if has_value(mask_key):
            mask_flags = set(str(mask_key).split("-"))
            for key, label in (("ma1", "MA"), ("mp1", "MP"), ("mi1", "MI")):
                if key in mask_flags:
                    parts.append(label)

        amr = format_config_value(row.get("absence_match_reward"))
        acp = format_config_value(row.get("absence_cover_penalty"))
        if amr is not None:
            parts.append(f"AMR-{amr}")
        if acp is not None:
            parts.append(f"ACP-{acp}")
        return "_".join(parts)

    if strategy == "genetic" or method.startswith("BinaryGenetic"):
        parts = ["BG"]
    elif strategy == "milp" or method.startswith("MILPOptimizer"):
        parts = ["BM"]
    else:
        return method or strategy or "unknown"

    acp = format_config_value(row.get("absence_cover_penalty"))
    amr = format_config_value(row.get("absence_match_reward"))
    if acp is not None:
        parts.append(f"ACP-{acp}")
    if amr is not None:
        parts.append(f"AMR-{amr}")
    return "_".join(parts)


def add_optimizer_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    labels = frame.apply(optimizer_label, axis=1)
    if OPTIMIZER_LABEL_COLUMN in frame.columns:
        frame[OPTIMIZER_LABEL_COLUMN] = frame[OPTIMIZER_LABEL_COLUMN].where(
            frame[OPTIMIZER_LABEL_COLUMN].notna(),
            labels,
        )
    else:
        frame[OPTIMIZER_LABEL_COLUMN] = labels
    return frame


def optimizer_label_order(frame: pd.DataFrame) -> list[str]:
    strategy_rank = {"mimic_v1": 0, "heuristic": 1, "genetic": 2, "milp": 3}
    labels = frame.loc[:, [column for column in ("strategy", OPTIMIZER_LABEL_COLUMN) if column in frame.columns]]
    labels = labels.drop_duplicates()
    if "strategy" not in labels.columns:
        return sorted(labels[OPTIMIZER_LABEL_COLUMN].dropna().astype(str).unique())
    labels["_rank"] = labels["strategy"].map(lambda value: strategy_rank.get(str(value).lower(), 99))
    labels = labels.sort_values(["_rank", OPTIMIZER_LABEL_COLUMN], kind="stable")
    return labels[OPTIMIZER_LABEL_COLUMN].dropna().astype(str).tolist()


def optimizer_palette(labels: Sequence[str]) -> dict[str, Any]:
    import seaborn as sns

    colors = sns.color_palette(DEFAULT_PALETTE)
    return {label: colors[index % len(colors)] for index, label in enumerate(labels)}


def default_plot_output_dir(paths: Sequence[Path], results_dir: Path) -> Path:
    if len(paths) == 1 and paths[0].parent.name == "results":
        return paths[0].parent.parent / "plots"
    return results_dir / "plots"


def prepare_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    def fill_metric(name: str, values: Sequence[float]) -> None:
        derived = pd.Series(values, index=frame.index)
        if name in frame.columns:
            frame[name] = frame[name].where(frame[name].notna(), derived)
        else:
            frame[name] = derived

    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in ("TP", "FN", "FP", "TN", "P", "N")
        if column in frame.columns
    }
    if {"TP", "FN"}.issubset(frame.columns) and "P" not in frame.columns:
        frame["P"] = numeric["TP"] + numeric["FN"]
        numeric["P"] = frame["P"]
    if {"FP", "TN"}.issubset(frame.columns) and "N" not in frame.columns:
        frame["N"] = numeric["FP"] + numeric["TN"]
        numeric["N"] = frame["N"]
    if {"TP", "P"}.issubset(numeric):
        fill_metric("TPR/recall", [safe_divide(tp, p) for tp, p in zip(numeric["TP"], numeric["P"])])
    if {"FP", "N"}.issubset(numeric):
        fill_metric("FPR", [safe_divide(fp, n) for fp, n in zip(numeric["FP"], numeric["N"])])
    if {"FN", "P"}.issubset(numeric):
        fill_metric("FNR", [safe_divide(fn, p) for fn, p in zip(numeric["FN"], numeric["P"])])
    if {"TN", "N"}.issubset(numeric):
        fill_metric("TNR/specificity", [safe_divide(tn, n) for tn, n in zip(numeric["TN"], numeric["N"])])
    if {"TP", "FP"}.issubset(numeric):
        fill_metric("PPV/precision", [safe_divide(tp, tp + fp) for tp, fp in zip(numeric["TP"], numeric["FP"])])
    if {"TN", "FN"}.issubset(numeric):
        fill_metric("NPV", [safe_divide(tn, tn + fn) for tn, fn in zip(numeric["TN"], numeric["FN"])])
    if {"FP", "TP"}.issubset(numeric):
        fill_metric("FDR", [safe_divide(fp, tp + fp) for fp, tp in zip(numeric["FP"], numeric["TP"])])
    if {"FN", "TN"}.issubset(numeric):
        fill_metric("FOR", [safe_divide(fn, tn + fn) for fn, tn in zip(numeric["FN"], numeric["TN"])])
    if {"TP", "TN", "P", "N"}.issubset(numeric):
        fill_metric("ACC", [
            safe_divide(tp + tn, p + n)
            for tp, tn, p, n in zip(numeric["TP"], numeric["TN"], numeric["P"], numeric["N"])
        ])
    if {"TPR/recall", "PPV/precision"}.issubset(frame.columns):
        fill_metric("F1_score", [
            safe_divide(2 * precision * recall, precision + recall)
            for precision, recall in zip(frame["PPV/precision"], frame["TPR/recall"])
        ])
    if {"TPR/recall", "TNR/specificity"}.issubset(frame.columns):
        fill_metric("BA", [
            safe_divide(recall + specificity, 2)
            for recall, specificity in zip(frame["TPR/recall"], frame["TNR/specificity"])
        ])
    if {"TP", "FP", "FN"}.issubset(numeric):
        fill_metric("Jaccard", [safe_divide(tp, tp + fp + fn) for tp, fp, fn in zip(numeric["TP"], numeric["FP"], numeric["FN"])])
    if {"TP", "TN", "FP", "FN"}.issubset(numeric):
        fill_metric("MCC", [
            safe_divide((tp * tn) - (fp * fn), np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            for tp, tn, fp, fn in zip(numeric["TP"], numeric["TN"], numeric["FP"], numeric["FN"])
        ])
    return frame


def load_summary(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            frame = pd.DataFrame(json.load(handle))
    else:
        raise ValueError(f"Unsupported result format: {path}")
    return prepare_metric_columns(frame)


def load_summaries(paths: Sequence[Path]) -> pd.DataFrame:
    return pd.concat([load_summary(path) for path in paths], ignore_index=True, sort=False)


def default_result_paths(results_dir: Path) -> list[Path]:
    paths = sorted(results_dir.glob("**/*binary_results.csv"))
    paths.extend(sorted(results_dir.glob("**/*binary_results.json")))
    return paths


def build_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_metric_columns(frame)
    frame = add_optimizer_labels(frame)
    groups = [column for column in GROUP_COLUMNS if column in frame.columns]
    metrics = [column for column in METRICS if column in frame.columns]
    if not metrics:
        return frame.loc[:, groups].drop_duplicates().reset_index(drop=True)

    numeric = frame.copy()
    for column in metrics:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if not groups:
        return numeric.loc[:, metrics].mean().to_frame().T
    return numeric.groupby(groups, dropna=False)[metrics].mean().reset_index()


def write_summary_table(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def plot_metric(frame: pd.DataFrame, metric: str, output_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Plotting requires the `benchmark` extra.") from exc

    if metric not in frame.columns:
        raise ValueError(f"Metric {metric!r} is not present in the result summary.")
    if "consortia_size" not in frame.columns:
        raise ValueError("Result summary must include 'consortia_size' for plotting.")

    plot_frame = add_optimizer_labels(prepare_metric_columns(frame))
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["consortia_size", metric])
    if plot_frame.empty:
        raise ValueError(f"No finite values found for metric {metric!r}.")

    sns.set_theme(style="whitegrid")
    label_order = optimizer_label_order(plot_frame)
    palette = optimizer_palette(label_order)
    size_order = sorted(plot_frame["consortia_size"].dropna().unique(), key=lambda value: float(value))
    size_labels = [format_config_value(value) or str(value) for value in size_order]
    size_map = dict(zip(size_order, size_labels))
    plot_frame["consortia_size_label"] = pd.Categorical(
        plot_frame["consortia_size"].map(size_map),
        categories=size_labels,
        ordered=True,
    )

    legend_columns = 1 if len(label_order) <= 18 else 2
    _, ax = plt.subplots(figsize=(18 if legend_columns > 1 else 12, 7))
    sns.boxplot(
        data=plot_frame,
        x="consortia_size_label",
        y=metric,
        hue=OPTIMIZER_LABEL_COLUMN,
        hue_order=label_order,
        palette=palette,
        fliersize=0,
        linewidth=0.8,
        ax=ax,
    )
    for patch in ax.patches:
        red, green, blue, alpha = patch.get_facecolor()
        patch.set_facecolor((red, green, blue, min(alpha, 0.3)))
    sns.swarmplot(
        data=plot_frame,
        x="consortia_size_label",
        y=metric,
        hue=OPTIMIZER_LABEL_COLUMN,
        hue_order=label_order,
        palette=palette,
        dodge=True,
        marker="o",
        edgecolor="k",
        linewidth=0.6,
        size=3,
        ax=ax,
    )
    if metric == "runtime_seconds" and (plot_frame[metric] > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel("Consortia size")
    ax.set_ylabel(metric)
    ax.set_title(f"SYNONIM binary benchmark {metric} across scenarios")

    handles, labels = ax.get_legend_handles_labels()
    deduped = dict(zip(labels, handles))
    ax.legend(
        deduped.values(),
        deduped.keys(),
        title="Optimizer configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        ncol=legend_columns,
        fontsize="small",
        title_fontsize="small",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    return output_path


def parse_metric_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return [item.strip() for item in values if str(item).strip()]


def relative_ratio(value: Any, reference: Any) -> float:
    try:
        value = float(value)
        reference = float(reference)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    if reference == 0:
        return 1.0 if value == 0 else np.nan
    ratio = value / reference
    return float(ratio) if np.isfinite(ratio) else np.nan


def relative_index_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["profile", "run_id", "scenario_index", "scenario_name", "consortia_size"]
    return [column for column in candidates if column in frame.columns]


def available_metric_columns(frame: pd.DataFrame, metrics: Sequence[str]) -> list[str]:
    return [metric for metric in metrics if metric in frame.columns]


def build_relative_performance_table(
    frame: pd.DataFrame,
    *,
    comparator: str = DEFAULT_HEATMAP_COMPARATOR,
    metrics: Sequence[str] = DEFAULT_HEATMAP_METRICS,
) -> pd.DataFrame:
    frame = add_optimizer_labels(prepare_metric_columns(frame))
    metrics = available_metric_columns(frame, metrics)
    if not metrics:
        raise ValueError("No requested heatmap metrics are present in the result summary.")
    if comparator not in set(frame[OPTIMIZER_LABEL_COLUMN].dropna()):
        available = ", ".join(sorted(frame[OPTIMIZER_LABEL_COLUMN].dropna().unique()))
        raise ValueError(f"Comparator {comparator!r} is not present. Available labels: {available}")

    numeric = frame.copy()
    for metric in metrics:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")

    rows: list[dict[str, Any]] = []
    group_columns = relative_index_columns(numeric)
    grouped = numeric.groupby(group_columns, dropna=False) if group_columns else [((), numeric)]
    for _, group in grouped:
        reference = group[group[OPTIMIZER_LABEL_COLUMN] == comparator]
        if reference.empty:
            continue
        reference_values = reference.loc[:, metrics].mean(numeric_only=True)
        for _, row in group.iterrows():
            relative = {OPTIMIZER_LABEL_COLUMN: row[OPTIMIZER_LABEL_COLUMN]}
            for metric in metrics:
                relative[metric] = relative_ratio(row[metric], reference_values[metric])
            rows.append(relative)

    if not rows:
        raise ValueError(f"No comparable groups contained comparator {comparator!r}.")

    relative_frame = pd.DataFrame(rows)
    table = relative_frame.groupby(OPTIMIZER_LABEL_COLUMN, dropna=False)[metrics].mean().T
    order = [label for label in optimizer_label_order(frame) if label in table.columns]
    return table.loc[:, order]


def plot_relative_performance_heatmap(
    table: pd.DataFrame,
    output_path: Path,
    *,
    split_directions: bool = True,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise RuntimeError("Plotting requires the `benchmark` extra.") from exc

    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if table.empty:
        raise ValueError("No finite relative-performance values are available for heatmap plotting.")

    labels = table.round(2).astype(object).where(table.notna(), "")
    width = max(12, 6 + 0.5 * len(table.columns))
    height = max(4, 1.4 + 0.45 * len(table.index))
    cmap = LinearSegmentedColormap.from_list(
        "purple_blue_green",
        [
            (0, "#99cfff"),
            (0.5, "#f7f7f7"),
            (0.7, "#00ffa1"),
            (1, "#00ad7d"),
        ],
        N=256,
    )

    def draw(ax: Any, data: pd.DataFrame, title: str, cbar: bool) -> None:
        sns.heatmap(
            data,
            annot=labels.loc[data.index, data.columns],
            fmt="",
            cmap=cmap,
            center=1,
            vmin=0,
            vmax=1.8,
            cbar=cbar,
            cbar_kws={"label": "Relative score"},
            annot_kws={"fontsize": 10},
            linewidths=0.3,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

    high_metrics = [metric for metric in table.index if metric not in LOWER_IS_BETTER]
    low_metrics = [metric for metric in table.index if metric in LOWER_IS_BETTER]
    if split_directions and high_metrics and low_metrics:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(width, height + 1.5),
            gridspec_kw={"height_ratios": [len(high_metrics), len(low_metrics)]},
            constrained_layout=True,
        )
        draw(axes[0], table.loc[high_metrics], "Higher is better", cbar=False)
        axes[0].set_xticklabels([])
        draw(axes[1], table.loc[low_metrics], "Lower is better", cbar=True)
    else:
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        draw(ax, table, "Relative performance", cbar=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def mask_flag_from_row(row: Mapping[str, Any], token: str) -> bool:
    strategy = str(row.get("strategy")).lower() if has_value(row.get("strategy")) else ""
    method = str(row.get("method")) if has_value(row.get("method")) else ""
    if strategy != "heuristic" and not method.startswith("BinaryHeuristic"):
        return np.nan
    mask_key = row.get("heuristic_mask_key")
    return has_value(mask_key) and token in set(str(mask_key).split("-"))


def add_heuristic_parameter_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["mask_covered_absent_features"] = frame.apply(lambda row: mask_flag_from_row(row, "ma1"), axis=1)
    frame["mask_covered_present_features"] = frame.apply(lambda row: mask_flag_from_row(row, "mp1"), axis=1)
    frame["mask_covered_isolate_features"] = frame.apply(lambda row: mask_flag_from_row(row, "mi1"), axis=1)
    return frame


def heuristic_parameter_order(frame: pd.DataFrame, comparator: str) -> list[str]:
    order_frame = frame.drop_duplicates(OPTIMIZER_LABEL_COLUMN).copy()
    for column in HEURISTIC_PARAMETER_COLUMNS:
        order_frame[column] = pd.to_numeric(order_frame[column], errors="coerce")
    order_frame = order_frame.sort_values(HEURISTIC_PARAMETER_COLUMNS, kind="stable")
    labels = order_frame[OPTIMIZER_LABEL_COLUMN].tolist()
    if comparator in labels:
        labels.remove(comparator)
        labels = [comparator] + labels
    return labels


def plot_heuristic_parameter_overview(
    frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    *,
    comparator: str = DEFAULT_HEATMAP_COMPARATOR,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("Plotting requires the `benchmark` extra.") from exc

    frame = add_heuristic_parameter_columns(add_optimizer_labels(prepare_metric_columns(frame)))
    if "strategy" not in frame.columns:
        raise ValueError("Current benchmark summaries must include a 'strategy' column.")
    heuristic = frame[frame["strategy"].isin(["mimic_v1", "heuristic"])].copy()
    if heuristic.empty:
        raise ValueError("No heuristic or mimic_v1 rows are present in the result summary.")
    if metric not in heuristic.columns:
        raise ValueError(f"Metric {metric!r} is not present in the result summary.")
    if comparator not in set(heuristic[OPTIMIZER_LABEL_COLUMN].dropna()):
        available = ", ".join(sorted(heuristic[OPTIMIZER_LABEL_COLUMN].dropna().unique()))
        raise ValueError(
            f"Comparator {comparator!r} is not present in heuristic or mimic_v1 rows. "
            f"Available labels: {available}"
        )

    heuristic[metric] = pd.to_numeric(heuristic[metric], errors="coerce")
    heuristic = heuristic.dropna(subset=[metric, "consortia_size"])
    if heuristic.empty:
        raise ValueError(f"No finite heuristic values found for metric {metric!r}.")

    rows: list[dict[str, Any]] = []
    group_columns = relative_index_columns(heuristic)
    grouped = heuristic.groupby(group_columns, dropna=False) if group_columns else [((), heuristic)]
    for _, group in grouped:
        reference = group[group[OPTIMIZER_LABEL_COLUMN] == comparator]
        if reference.empty:
            continue
        reference_value = reference[metric].mean()
        for _, row in group.iterrows():
            ratio = relative_ratio(row[metric], reference_value)
            rows.append({
                **row.to_dict(),
                "relative_ratio": ratio,
                "log_relative_ratio": np.log(ratio) if ratio and ratio > 0 else np.nan,
            })
    if not rows:
        raise ValueError(f"No comparable heuristic groups contained comparator {comparator!r}.")

    relative = pd.DataFrame(rows)
    order = heuristic_parameter_order(relative, comparator)
    relative["parameter_index"] = relative[OPTIMIZER_LABEL_COLUMN].map({label: index for index, label in enumerate(order)})
    relative["parameter_position"] = relative["parameter_index"] + 0.5
    grouped_metric = (
        relative
        .groupby([OPTIMIZER_LABEL_COLUMN, "parameter_index", "parameter_position", "consortia_size"], as_index=False)[[metric, "log_relative_ratio"]]
        .median()
    )

    parameter_matrix = (
        relative
        .drop_duplicates(OPTIMIZER_LABEL_COLUMN)
        .set_index(OPTIMIZER_LABEL_COLUMN)
        .loc[order, HEURISTIC_PARAMETER_COLUMNS]
        .T
    )
    parameter_matrix = parameter_matrix.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    parameter_annotations = parameter_matrix.apply(
        lambda column: column.map(lambda value: "" if pd.isna(value) else str(int(value)))
    )

    sns.set_theme(style="whitegrid")
    width = max(12, 0.45 * len(order))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width, 11),
        gridspec_kw={"height_ratios": [3, 2, 1.2]},
        sharex=True,
        constrained_layout=True,
    )

    sns.swarmplot(
        data=grouped_metric,
        x="parameter_position",
        y=metric,
        hue="consortia_size",
        dodge=False,
        palette="deep",
        marker="o",
        edgecolor="k",
        linewidth=0.6,
        native_scale=True,
        size=6,
        ax=axes[0],
    )
    axes[0].set_title(f"Heuristic {metric} by parameter combination")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels([])
    axes[0].legend(title="Consortia size", loc="center left", bbox_to_anchor=(1.01, 0.5))

    sns.boxplot(
        data=grouped_metric,
        x="parameter_position",
        y="log_relative_ratio",
        color="lightblue",
        fliersize=0,
        native_scale=True,
        ax=axes[1],
    )
    sns.swarmplot(
        data=grouped_metric,
        x="parameter_position",
        y="log_relative_ratio",
        color="black",
        native_scale=True,
        size=5,
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Log-ratios versus {comparator}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("log(relative score)")
    axes[1].set_xticklabels([])

    sns.heatmap(
        parameter_matrix,
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=axes[2],
        annot=parameter_annotations,
        fmt="",
        mask=parameter_matrix.isna(),
    )
    axes[2].set_title("Parameter values")
    axes[2].set_xlabel("Parameter combinations")
    axes[2].set_ylabel("")
    axes[2].set_xticks([index + 0.5 for index in range(len(order))])
    axes[2].set_xticklabels(order, rotation=90, ha="center", fontsize=7)
    axes[2].tick_params(axis="y", rotation=0)

    comparator_index = order.index(comparator)
    for ax in axes[:2]:
        ax.axvspan(comparator_index, comparator_index + 1, color="blue", alpha=0.15, zorder=0)
    axes[2].axvspan(comparator_index, comparator_index + 1, color="blue", alpha=0.15, zorder=10)
    for label in order:
        if label == comparator:
            continue
        subset = grouped_metric[grouped_metric[OPTIMIZER_LABEL_COLUMN] == label]["log_relative_ratio"].dropna()
        if subset.empty or not (subset > 0).any():
            continue
        color = "green" if (subset > 0).all() else "yellow"
        index = order.index(label)
        for ax in axes[:2]:
            ax.axvspan(index, index + 1, color=color, alpha=0.3, zorder=0)
        axes[2].axvspan(index, index + 1, color=color, alpha=0.3, zorder=10)

    for ax in axes:
        ax.set_xlim(0, len(order))

    axes[1].legend(
        handles=[
            Patch(facecolor="blue", alpha=0.15, label="Comparator"),
            Patch(facecolor="yellow", alpha=0.3, label="Some sizes higher"),
            Patch(facecolor="green", alpha=0.3, label="All sizes higher"),
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Summarize or plot binary benchmark results.")
    cli.add_argument("paths", nargs="*", type=Path, help="CSV or JSON result summaries.")
    cli.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--plot-kind", choices=["metric", "relative-heatmap", "heuristic-parameters"], default="metric")
    cli.add_argument("--metric", default="F1_score")
    cli.add_argument("--heatmap-metrics", default=",".join(DEFAULT_HEATMAP_METRICS))
    cli.add_argument("--comparator", default=DEFAULT_HEATMAP_COMPARATOR)
    cli.add_argument("--summary-only", action="store_true")
    cli.add_argument("--summary-name", default="summary_table.csv")
    cli.add_argument("--plot-name")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = args.paths or default_result_paths(args.results_dir)
    if not paths:
        raise SystemExit(f"No result summaries found under {args.results_dir}.")

    frame = load_summaries(paths)
    output_dir = args.output_dir or default_plot_output_dir(paths, args.results_dir)
    table = build_summary_table(frame)
    summary_path = write_summary_table(table, output_dir / args.summary_name)
    print(f"Wrote summary table to {summary_path}")
    if not args.summary_only:
        if args.plot_kind == "metric":
            plot_name = args.plot_name or f"{args.metric.replace('/', '_')}.png"
            plot_path = plot_metric(frame, args.metric, output_dir / plot_name)
        elif args.plot_kind == "relative-heatmap":
            plot_name = args.plot_name or "relative_performance_heatmap.png"
            heatmap_table = build_relative_performance_table(
                frame,
                comparator=args.comparator,
                metrics=parse_metric_list(args.heatmap_metrics),
            )
            heatmap_table_path = output_dir / "relative_performance_table.csv"
            heatmap_table_path.parent.mkdir(parents=True, exist_ok=True)
            heatmap_table.to_csv(heatmap_table_path)
            print(f"Wrote relative performance table to {heatmap_table_path}")
            plot_path = plot_relative_performance_heatmap(heatmap_table, output_dir / plot_name)
        elif args.plot_kind == "heuristic-parameters":
            plot_name = args.plot_name or f"heuristic_{args.metric.replace('/', '_')}_parameters.png"
            plot_path = plot_heuristic_parameter_overview(
                frame,
                args.metric,
                output_dir / plot_name,
                comparator=args.comparator,
            )
        else:
            raise ValueError(f"Unknown plot kind: {args.plot_kind}")
        print(f"Wrote plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
