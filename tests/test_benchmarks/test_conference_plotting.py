import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "plot_binary_conference", ROOT / "benchmarks" / "plot_binary_conference.py"
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def row(strategy, method, f1, *, mask=None, acp=None, amr=None, optimizer="job"):
    return {
        "profile": "tiny",
        "run_id": "run-1",
        "job_key": optimizer,
        "strategy": strategy,
        "method": method,
        "scenario_index": 0,
        "scenario_name": "s0",
        "consortia_size": 1,
        "selected_count": 1,
        "scenario_has_solution": True,
        "F1_score": f1,
        "TPR/recall": f1,
        "PPV/precision": f1,
        "logical_run_runtime_seconds": 1.0,
        "heuristic_mask_key": mask,
        "absence_cover_penalty": acp,
        "absence_match_reward": amr,
    }


def complete_candidates(selected_f1=0.8):
    return pd.DataFrame(
        [
            row("mimic_v1", "MiMiC_v1", 0.5, optimizer="mimic"),
            row("heuristic", "BinaryHeuristic", selected_f1, mask="ma0-mp1-mi1", acp=1, amr=0, optimizer="heuristic"),
            row("genetic", "BinaryGenetic", 0.7, acp=1, amr=1, optimizer="genetic"),
            row("milp", "MILPOptimizer", 0.75, acp=1, amr=1, optimizer="milp"),
        ]
    )


def test_selected_config_verification_accepts_best_candidates():
    audit = plotter.verify_selected_configs(complete_candidates(), sizes=[1])
    selected = audit[audit["selected_explicitly"]]
    assert set(selected["optimizer_label"]) == {
        "BH_MP_MI_AMR-0_ACP-1", "BG_ACP-1_AMR-1", "BM_ACP-1_AMR-1"
    }


def test_selected_config_verification_fails_when_candidate_is_not_best():
    frame = pd.concat(
        [
            complete_candidates(selected_f1=0.7),
            pd.DataFrame([row("heuristic", "BinaryHeuristic", 0.9, mask="ma0-mp0-mi0", acp=0, amr=0, optimizer="better")]),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="heuristic.*best"):
        plotter.verify_selected_configs(frame, sizes=[1])


def test_equivalent_heuristic_requires_matching_mimic_f1():
    frame = pd.concat(
        [
            complete_candidates(),
            pd.DataFrame([row("heuristic", "BinaryHeuristic", 0.5, mask="ma1-mp1-mi1", acp=0, amr=0, optimizer="equivalent")]),
        ],
        ignore_index=True,
    )
    assert plotter.equivalent_heuristic_matches(frame, sizes=[1])
    frame.loc[frame["job_key"] == "equivalent", "F1_score"] = 0.51
    assert not plotter.equivalent_heuristic_matches(frame, sizes=[1])


def test_metrics_plot_writes_png_and_pdf(tmp_path):
    frame = plotter.conference_labels(complete_candidates())
    frame["conference_label"] = frame["optimizer_label"].map({
        "MiMiC_v1": "MiMiC",
        "BH_MP_MI_AMR-0_ACP-1": "Heuristic_v2",
        "BG_ACP-1_AMR-1": "Genetic",
        "BM_ACP-1_AMR-1": "MILP",
    })
    output = plotter.plot_conference_metrics(frame, tmp_path / "metrics.png")
    assert output.exists()
    assert output.with_suffix(".pdf").exists()


def test_relative_heatmap_writes_png_and_pdf(tmp_path):
    table = pd.DataFrame(
        {
            "MiMiC": [1.0, 1.0, 1.0],
            "MiMiC_actual": [1.01, 0.98, 1.2],
            "Heuristic_v2": [1.05, 0.95, 0.8],
            "Genetic": [1.1, 0.9, 0.6],
            "MILP": [1.2, 0.85, 0.4],
        },
        index=["F1_score", "FPR", "logical_run_runtime_seconds"],
    )
    output = plotter.plot_conference_heatmap(table, tmp_path / "heatmap.png")
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
