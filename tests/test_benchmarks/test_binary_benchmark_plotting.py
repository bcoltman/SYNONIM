import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_plotter():
    path = REPO_ROOT / "benchmarks" / "plot_binary_results.py"
    spec = importlib.util.spec_from_file_location("plot_binary_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_binary_results = load_plotter()


def test_metric_normalization_uses_safe_division():
    frame = pd.DataFrame(
        [
            {"TP": 0, "FP": 0, "FN": 0, "P": 0},
            {"TP": 1, "FP": 3, "FN": 1, "P": 2},
        ]
    )

    normalized = plot_binary_results.prepare_metric_columns(frame)

    assert normalized.loc[0, "TPR/recall"] == 0.0
    assert normalized.loc[0, "PPV/precision"] == 0.0
    assert normalized.loc[1, "TPR/recall"] == 0.5
    assert normalized.loc[1, "PPV/precision"] == 0.25


def test_relative_heatmap_table_uses_current_summary_labels():
    frame = pd.DataFrame(
        [
            {
                "profile": "medium",
                "run_id": "local",
                "strategy": "mimic_v1",
                "method": "MiMiC_v1",
                "scenario_index": 0,
                "scenario_name": "s0",
                "consortia_size": 3,
                "F1_score": 0.5,
                "runtime_seconds": 2.0,
            },
            {
                "profile": "medium",
                "run_id": "local",
                "strategy": "genetic",
                "method": "BinaryGenetic_ACP-1.0_AMR-1.0",
                "scenario_index": 0,
                "scenario_name": "s0",
                "consortia_size": 3,
                "absence_cover_penalty": 1,
                "absence_match_reward": 1,
                "F1_score": 1.0,
                "runtime_seconds": 1.0,
            },
        ]
    )

    table = plot_binary_results.build_relative_performance_table(
        frame,
        metrics=["F1_score", "runtime_seconds"],
    )

    assert table.loc["F1_score", "BG_ACP-1_AMR-1"] == 2.0
    assert table.loc["runtime_seconds", "BG_ACP-1_AMR-1"] == 0.5

