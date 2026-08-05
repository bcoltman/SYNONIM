import importlib.util
from pathlib import Path

import pandas as pd
import pytest


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


def complete_frame():
    return pd.DataFrame(
        [
            {
                "profile": "tiny",
                "run_id": "run-1",
                "job_key": "mimic-k2",
                "strategy": "mimic_v1",
                "method": "MiMiC_v1",
                "scenario_index": scenario,
                "scenario_name": f"s{scenario}",
                "consortia_size": 2,
                "selected_count": 2,
                "scenario_has_solution": True,
                "F1_score": 0.5,
                "logical_run_runtime_seconds": 4.0,
            }
            for scenario in range(2)
        ]
        + [
            {
                "profile": "tiny",
                "run_id": "run-1",
                "job_key": "heuristic-k2",
                "strategy": "heuristic",
                "method": "BinaryHeuristic_ACP-1_AMR-1",
                "absence_cover_penalty": 1,
                "absence_match_reward": 1,
                "scenario_index": scenario,
                "scenario_name": f"s{scenario}",
                "consortia_size": 2,
                "selected_count": 2,
                "scenario_has_solution": True,
                "F1_score": 1.0,
                "logical_run_runtime_seconds": 2.0,
            }
            for scenario in range(2)
        ]
    )


def complete_manifest():
    return {
        "schema_version": 1,
        "profile": "tiny",
        "run_id": "run-1",
        "expected_scenarios": 2,
        "jobs": [{"job_key": "mimic-k2"}, {"job_key": "heuristic-k2"}],
    }


def test_validation_accepts_complete_run_and_rejects_bad_rows():
    frame = complete_frame()
    validated = plot_binary_results.validate_results(frame, complete_manifest())
    assert len(validated) == 4

    with_duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        plot_binary_results.validate_results(with_duplicate, complete_manifest())

    wrong_size = frame.copy()
    wrong_size.loc[0, "selected_count"] = 1
    with pytest.raises(ValueError, match="incorrect consortium size"):
        plot_binary_results.validate_results(wrong_size, complete_manifest())

    mixed = frame.copy()
    mixed.loc[0, "run_id"] = "another-run"
    with pytest.raises(ValueError, match="mixed"):
        plot_binary_results.validate_results(mixed, complete_manifest())


def test_validation_fails_incomplete_by_default_and_allows_explicit_partial_run():
    incomplete = complete_frame().iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete"):
        plot_binary_results.validate_results(incomplete, complete_manifest())
    assert len(
        plot_binary_results.validate_results(
            incomplete,
            complete_manifest(),
            allow_incomplete=True,
        )
    ) == 3

    infeasible = complete_frame()
    infeasible.loc[0, "scenario_has_solution"] = False
    with pytest.raises(ValueError, match="without a feasible solution"):
        plot_binary_results.validate_results(infeasible, complete_manifest())
    partial = plot_binary_results.validate_results(
        infeasible,
        complete_manifest(),
        allow_incomplete=True,
    )
    assert len(partial) == 3


def test_logical_runtime_is_compared_once_per_job_not_once_per_scenario():
    table = plot_binary_results.build_relative_performance_table(
        complete_frame(),
        metrics=["F1_score", "logical_run_runtime_seconds"],
    )
    assert table.loc["F1_score", "BH_AMR-1_ACP-1"] == 2.0
    assert table.loc["logical_run_runtime_seconds", "BH_AMR-1_ACP-1"] == 0.5


def test_default_discovery_reads_csv_only(tmp_path):
    (tmp_path / "binary_mimic_v1.csv").write_text("run_id\\nrun-1\\n", encoding="utf-8")
    (tmp_path / "binary_mimic_v1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "summary_table.csv").write_text("metric\\n1\\n", encoding="utf-8")
    assert plot_binary_results.default_result_paths(tmp_path) == [
        tmp_path / "binary_mimic_v1.csv"
    ]
