import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = REPO_ROOT / "benchmarks" / "run_binary_optimizers.py"
    spec = importlib.util.spec_from_file_location("run_binary_optimizers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_binary_optimizers = load_runner()


def profile_shape(profile):
    genomes, metagenomes = run_binary_optimizers.load_benchmark_frames(profile)
    return genomes.shape[0], genomes.shape[1], metagenomes.shape[1], int(genomes.to_numpy().sum())


def assert_restricted_gurobi_size(profile):
    features, genomes, _, nonzeros = profile_shape(profile)
    variables = features + genomes
    constraints = features + nonzeros + 1
    assert variables < 2000
    assert constraints < 2000


def test_run_binary_optimizers_requires_explicit_profile():
    with pytest.raises(SystemExit):
        run_binary_optimizers.build_parser().parse_args([])


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_time_limit_override_must_be_positive_and_finite(value):
    with pytest.raises(SystemExit):
        run_binary_optimizers.build_parser().parse_args(
            ["--profile", "large", "--time-limit", value]
        )


def test_four_canonical_profiles_are_available():
    assert set(run_binary_optimizers.load_profiles()) == {"tiny", "medium", "recovery", "large"}


def test_large_profile_preserves_original_validation_matrix():
    profile = run_binary_optimizers.load_profile("large")

    heuristic_jobs = run_binary_optimizers.expand_profile(profile, strategies=["heuristic"])
    genetic_jobs = run_binary_optimizers.expand_profile(profile, strategies=["genetic"])
    milp_jobs = run_binary_optimizers.expand_profile(profile, strategies=["milp"])
    mimic_v1_jobs = run_binary_optimizers.expand_profile(profile, strategies=["mimic_v1"])

    assert len(heuristic_jobs) == 13 * 32
    assert len(genetic_jobs) == 10 * 4
    assert len(milp_jobs) == 3 * 4
    assert len(mimic_v1_jobs) == 13
    assert {job["warm_start"] for job in milp_jobs} == {"heuristic"}
    assert profile["data"]["expected_scenarios"] == 8
    assert profile["data"]["expected_genomes"] == 111
    assert profile["data"]["expected_features"] == 17929


def test_medium_and_recovery_profiles_expand_full_local_grids():
    for name in ["medium", "recovery"]:
        profile = run_binary_optimizers.load_profile(name)
        heuristic_jobs = run_binary_optimizers.expand_profile(profile, strategies=["heuristic"])
        genetic_jobs = run_binary_optimizers.expand_profile(profile, strategies=["genetic"])
        milp_jobs = run_binary_optimizers.expand_profile(profile, strategies=["milp"])
        mimic_v1_jobs = run_binary_optimizers.expand_profile(profile, strategies=["mimic_v1"])

        assert len(heuristic_jobs) == 3 * 32
        assert len(genetic_jobs) == 3 * 4
        assert len(milp_jobs) == 3 * 4
        assert len(mimic_v1_jobs) == 3
        assert milp_jobs[0]["time_limit"] == 300
        assert_restricted_gurobi_size(profile)


def test_job_overrides_apply_time_limit_only_to_milp():
    jobs = [
        {"strategy": "heuristic"},
        {"strategy": "genetic", "processes": 1},
        {"strategy": "milp", "processes": 1, "time_limit": 300},
    ]

    run_binary_optimizers.apply_job_overrides(jobs, processes=8, time_limit=69120)

    assert "time_limit" not in jobs[0]
    assert jobs[1] == {"strategy": "genetic", "processes": 8}
    assert jobs[2] == {"strategy": "milp", "processes": 8, "time_limit": 69120}


def test_milp_uses_one_multi_scenario_solve_with_all_heuristic_starts(monkeypatch):
    import synonim.optimizers.binary as binary_optimizers

    heuristic_calls = []

    class FakeHeuristic:
        def __init__(self, **kwargs):
            heuristic_calls.append(kwargs)

        def optimize(self):
            return [SimpleNamespace(X_opt=[1, 0]), SimpleNamespace(X_opt=[0, 1])]

    class FakeMILP:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.starts = []
            self.optimize_calls = 0
            self.__class__.instances.append(self)

        def warmup(self, solution):
            self.starts.append(solution)

        def optimize(self):
            self.optimize_calls += 1
            return "multi-scenario-result"

    monkeypatch.setattr(run_binary_optimizers, "BinaryHeuristic", FakeHeuristic)
    monkeypatch.setitem(binary_optimizers.__dict__, "BinaryMILP", FakeMILP)
    model = object()
    job = {
        "consortia_size": 3,
        "absence_cover_penalty": 1,
        "absence_match_reward": 0,
        "processes": 8,
        "time_limit": 69120,
        "warm_start": "heuristic",
    }

    result = run_binary_optimizers.run_milp(model, job)

    assert result == "multi-scenario-result"
    assert len(FakeMILP.instances) == 1
    optimizer = FakeMILP.instances[0]
    assert optimizer.kwargs["model"] is model
    assert optimizer.kwargs["time_limit"] == 69120
    assert optimizer.optimize_calls == 1
    assert [start.tolist() for start in optimizer.starts] == [[1, 0], [0, 1]]
    assert len(heuristic_calls) == 1
    assert heuristic_calls[0]["model"] is model
    assert heuristic_calls[0]["mask_covered_absent_features"] is True
    assert heuristic_calls[0]["mask_covered_present_features"] is True
    assert heuristic_calls[0]["mask_covered_isolate_features"] is True


def test_tiny_benchmark_writes_stable_summaries(tmp_path):
    exit_code = run_binary_optimizers.main(
        [
            "--profile",
            "tiny",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "pytest",
        ]
    )

    assert exit_code == 0
    csv_path = tmp_path / "binary_results.csv"
    json_path = tmp_path / "binary_results.json"
    assert csv_path.exists()
    assert json_path.exists()

    frame = pd.read_csv(csv_path)
    assert set(frame["strategy"]) == {"mimic_v1", "heuristic", "genetic"}
    assert frame["profile"].unique().tolist() == ["tiny"]
    assert frame["selected_count"].tolist() == [2, 2, 2]
    assert "PPV/precision" in frame.columns
    assert frame["job_key"].nunique() == 3
    assert (frame["logical_run_runtime_seconds"] >= 0).all()
    manifest = __import__("json").loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "pytest"
    assert len(manifest["jobs"]) == 3
    assert set(frame["job_key"]) == {job["job_key"] for job in manifest["jobs"]}


def test_recovery_benchmark_emits_ground_truth_columns(tmp_path):
    exit_code = run_binary_optimizers.main(
        [
            "--profile",
            "recovery",
            "--strategy",
            "heuristic",
            "--consortia-size",
            "3",
            "--absence-cover-penalty",
            "1",
            "--absence-match-reward",
            "0",
            "--heuristic-mask-key",
            "ma0-mp1-mi0",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "pytest",
        ]
    )

    assert exit_code == 0
    frame = pd.read_csv(tmp_path / "binary_results.csv")
    assert "expected_selected_names" in frame.columns
    assert "selection_jaccard" in frame.columns
    assert "exact_selection_recovery" in frame.columns
    assert frame.loc[frame["scenario_name"] == "recovery_k3_exact", "expected_selected_names"].iloc[0] == (
        "source_01;source_05;source_09"
    )


@pytest.mark.milp
def test_medium_milp_runs_with_restricted_license(tmp_path):
    pytest.importorskip("gurobipy")

    exit_code = run_binary_optimizers.main(
        [
            "--profile",
            "medium",
            "--strategy",
            "milp",
            "--consortia-size",
            "3",
            "--absence-cover-penalty",
            "1",
            "--absence-match-reward",
            "0",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "pytest",
        ]
    )

    assert exit_code == 0
    frame = pd.read_csv(tmp_path / "binary_results.csv")
    assert set(frame["strategy"]) == {"milp"}
    assert {"solver_status", "solver_status_code", "mip_gap"} <= set(frame.columns)
    assert frame["solver_status"].notna().all()
    assert set(frame["scenario_name"]) == {
        "medium_k3_trap",
        "medium_k5_overlap",
        "medium_k7_sparse",
        "medium_noise_control",
        "medium_greedy_trap",
    }


@pytest.mark.milp
def test_recovery_includes_mimic_v1_failure_that_milp_solves(tmp_path):
    pytest.importorskip("gurobipy")

    exit_code = run_binary_optimizers.main(
        [
            "--profile",
            "recovery",
            "--strategy",
            "mimic_v1",
            "--strategy",
            "milp",
            "--consortia-size",
            "3",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "pytest",
        ]
    )

    assert exit_code == 0
    frame = pd.read_csv(tmp_path / "binary_results.csv")
    trap = frame[frame["scenario_name"] == "recovery_greedy_trap"]

    mimic = trap[trap["strategy"] == "mimic_v1"].iloc[0]
    assert mimic["selected_names"] == "trap_decoy_high;trap_exact_a;trap_exact_b"
    assert mimic["F1_score"] < 1.0
    assert mimic["selection_jaccard"] < 1.0
    assert not bool(mimic["exact_selection_recovery"])

    milp = trap[
        (trap["strategy"] == "milp")
        & (trap["absence_cover_penalty"] == 1)
        & (trap["absence_match_reward"] == 0)
    ].iloc[0]
    assert milp["selected_names"] == "trap_exact_a;trap_exact_b;trap_exact_c"
    assert milp["F1_score"] == pytest.approx(1.0)
    assert milp["selection_jaccard"] == pytest.approx(1.0)
    assert bool(milp["exact_selection_recovery"])
