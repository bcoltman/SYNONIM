#!/usr/bin/env python3
"""Run explicit binary optimizer benchmark profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synonim.io import model_from_frames
from synonim.optimizers import Solution
from synonim.optimizers.binary import BinaryGenetic, BinaryHeuristic


PROFILE_PATH = Path(__file__).with_name("profiles.json")
OUTPUTS_ROOT = REPO_ROOT / "benchmarks" / "outputs"
SUMMARY_PREFIX = "binary_results"
STRATEGIES = ("mimic_v1", "heuristic", "genetic", "milp")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Run binary optimizer benchmarks. --profile is required."
    )
    cli.add_argument("--profile", required=True, help="Profile name from profiles.json.")
    cli.add_argument("--profiles-path", type=Path, default=PROFILE_PATH)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--summary-prefix", default=SUMMARY_PREFIX)
    cli.add_argument("--run-id")
    cli.add_argument("--strategy", action="append", choices=STRATEGIES, dest="strategies")
    cli.add_argument("--consortia-size", action="append", type=int, dest="sizes")
    cli.add_argument("--absence-cover-penalty", action="append", type=float, dest="acps")
    cli.add_argument("--absence-match-reward", action="append", type=float, dest="amrs")
    cli.add_argument("--heuristic-mask-key", action="append", dest="mask_keys")
    cli.add_argument("--processes", type=int, help="Override genetic/MILP process count.")
    cli.add_argument("--dry-run", action="store_true", help="Print expanded jobs without optimizing.")
    return cli


def build_parser() -> argparse.ArgumentParser:
    return parser()


def load_profiles(path: Path = PROFILE_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)["profiles"]
    return profiles


def load_profile(name: str, path: Path = PROFILE_PATH) -> dict[str, Any]:
    profiles = load_profiles(path)
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown benchmark profile {name!r}. Available profiles: {available}")
    profile = dict(profiles[name])
    profile["name"] = name
    profile["profile_path"] = path
    return profile


def _numeric(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def heuristic_mask_key(job: Mapping[str, Any]) -> str:
    return (
        f"ma{int(bool(job.get('mask_covered_absent_features', False)))}-"
        f"mp{int(bool(job.get('mask_covered_present_features', False)))}-"
        f"mi{int(bool(job.get('mask_covered_isolate_features', False)))}"
    )


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def expand_profile(
    profile: Mapping[str, Any],
    *,
    strategies: Sequence[str] | None = None,
    consortia_sizes: Sequence[int] | None = None,
    absence_cover_penalties: Sequence[float] | None = None,
    absence_match_rewards: Sequence[float] | None = None,
    heuristic_mask_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    strategy_filter = set(strategies or [])
    size_filter = set(consortia_sizes or [])
    acp_filter = {_numeric(value) for value in absence_cover_penalties or []}
    amr_filter = {_numeric(value) for value in absence_match_rewards or []}
    mask_filter = set(heuristic_mask_keys or [])

    jobs = []
    for strategy, settings in profile["runs"].items():
        if strategy_filter and strategy not in strategy_filter:
            continue
        fixed_settings = {
            key: value
            for key, value in settings.items()
            if key not in {"consortia_sizes", "config_grid"}
        }
        for size in settings["consortia_sizes"]:
            if size_filter and size not in size_filter:
                continue
            for config in expand_grid(settings["config_grid"]):
                config = {
                    key: _numeric(value) if key.startswith("absence_") else value
                    for key, value in config.items()
                }
                if (
                    acp_filter
                    and "absence_cover_penalty" in config
                    and config.get("absence_cover_penalty") not in acp_filter
                ):
                    continue
                if (
                    amr_filter
                    and "absence_match_reward" in config
                    and config.get("absence_match_reward") not in amr_filter
                ):
                    continue

                job = {
                    "profile": profile["name"],
                    "strategy": strategy,
                    "consortia_size": size,
                    **fixed_settings,
                    **config,
                }
                if strategy == "heuristic":
                    job["heuristic_mask_key"] = heuristic_mask_key(job)
                    if mask_filter and job["heuristic_mask_key"] not in mask_filter:
                        continue
                jobs.append(job)
    return jobs


def _data_path(profile: Mapping[str, Any], key: str) -> Path:
    profile_path = Path(profile["profile_path"]).resolve()
    return (profile_path.parent / profile["data"][key]).resolve()


def load_benchmark_frames(profile: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = profile["data"]
    index_col = data.get("index_col", "PfamID")
    genomes = pd.read_csv(_data_path(profile, "genomes"), sep="\t", index_col=index_col)
    metagenomes = pd.read_csv(_data_path(profile, "metagenomes"), sep="\t", index_col=index_col)

    if data.get("feature_limit") is not None:
        genomes = genomes.iloc[: data["feature_limit"], :]
        metagenomes = metagenomes.iloc[: data["feature_limit"], :]
    if data.get("genome_limit") is not None:
        genomes = genomes.iloc[:, : data["genome_limit"]]
    if data.get("scenario_limit") is not None:
        metagenomes = metagenomes.iloc[:, : data["scenario_limit"]]

    common_features = genomes.index.intersection(metagenomes.index)
    genomes = genomes.loc[common_features]
    metagenomes = metagenomes.loc[common_features]
    loaded = {
        "expected_features": len(common_features),
        "expected_genomes": genomes.shape[1],
        "expected_scenarios": metagenomes.shape[1],
    }
    for key, value in loaded.items():
        if data.get(key) is not None and data[key] != value:
            raise ValueError(f"{profile['name']} expected {data[key]} for {key}, loaded {value}.")
    return genomes, metagenomes


def build_model(genomes: pd.DataFrame, metagenomes: pd.DataFrame):
    return model_from_frames(genomes_binary=genomes, metagenomes_binary=metagenomes)


def common_optimizer_args(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "consortia_size": int(job["consortia_size"]),
        "absence_cover_penalty": float(job["absence_cover_penalty"]),
        "absence_match_reward": float(job["absence_match_reward"]),
    }


def run_mimic_v1(model: Any, job: Mapping[str, Any]):
    G = model.genome_binary_matrix.astype(bool)
    M = model.metagenome_binary_matrix.astype(bool)
    genome_names = list(model.genome_names)
    metagenome_names = list(model.metagenome_names)
    analyzer = BinaryHeuristic(
        model=model,
        consortia_size=int(job["consortia_size"]),
        absence_cover_penalty=0,
        absence_match_reward=0,
    )
    solutions = []

    for scenario_index, scenario_name in enumerate(metagenome_names):
        start_time = time.time()
        remaining_target = M[:, scenario_index].copy()
        remaining_genomes = G.copy()
        selected_indices: list[int] = []
        match_counts: list[int] = []

        for _ in range(int(job["consortia_size"])):
            totals = remaining_genomes.sum(axis=0)
            matches = (remaining_genomes & remaining_target[:, None]).sum(axis=0)
            ratios = matches / totals.clip(min=1)
            ratios[totals == 0] = -1.0

            best_ratio = ratios.max()
            if best_ratio < 0:
                break

            ratio_candidates = np.flatnonzero(ratios == best_ratio)
            best_match = int(matches[ratio_candidates].max())
            if best_match == 0:
                break

            best_candidates = ratio_candidates[matches[ratio_candidates] == best_match]
            selected_index = int(best_candidates[0])
            selected_indices.append(selected_index)
            match_counts.append(best_match)

            matched_features = remaining_genomes[:, selected_index] & remaining_target
            remaining_target[matched_features] = False
            remaining_genomes[matched_features, :] = False
            remaining_genomes[:, selected_index] = False

        x_opt = np.zeros(G.shape[1], dtype=int)
        x_opt[selected_indices] = 1
        analysis_metrics = analyzer.analyze_solution(M[:, scenario_index].astype(int), x_opt)
        solutions.append(
            Solution(
                name=scenario_name,
                method="MiMiC_v1",
                X_opt=x_opt,
                objective=float(sum(match_counts)),
                genome_names=genome_names,
                selection_order=[genome_names[index] for index in selected_indices],
                details={
                    "scenario": scenario_index,
                    "runtime": time.time() - start_time,
                    "match_counts": match_counts,
                    "remaining_target_pfams": int(remaining_target.sum()),
                    "analysis": analysis_metrics,
                },
            )
        )

    return solutions[0] if len(solutions) == 1 else solutions


def run_heuristic(model: Any, job: Mapping[str, Any]):
    return BinaryHeuristic(
        model=model,
        **common_optimizer_args(job),
        mask_covered_absent_features=bool(job.get("mask_covered_absent_features", False)),
        mask_covered_present_features=bool(job.get("mask_covered_present_features", False)),
        mask_covered_isolate_features=bool(job.get("mask_covered_isolate_features", False)),
    ).optimize()


def run_genetic(model: Any, job: Mapping[str, Any]):
    return BinaryGenetic(
        model=model,
        **common_optimizer_args(job),
        population_size=int(job["population_size"]),
        generations=int(job["generations"]),
        max_unchanged_generations=int(job["max_unchanged_generations"]),
        mutation_rate=float(job.get("mutation_rate", 0.05)),
        tournament_size=int(job.get("tournament_size", 10)),
        exploration_rate=float(job.get("exploration_rate", 0.2)),
        processes=int(job.get("processes", 1)),
    ).optimize()


def run_milp(
    model: Any,
    genomes: pd.DataFrame,
    metagenomes: pd.DataFrame,
    job: Mapping[str, Any],
):
    try:
        from synonim.optimizers.binary import BinaryMILP
    except ImportError as exc:
        raise RuntimeError("MILP benchmarks require the `milp` extra and Gurobi.") from exc

    if job.get("warm_start") != "heuristic":
        return BinaryMILP(
            model=model,
            **common_optimizer_args(job),
            processes=int(job["processes"]),
            time_limit=float(job["time_limit"]),
        ).optimize()

    solutions = []
    for scenario_index, scenario_name in enumerate(metagenomes.columns):
        scenario_model = build_model(genomes, metagenomes.loc[:, [scenario_name]])
        warm_solution = run_heuristic(scenario_model, job)
        optimizer = BinaryMILP(
            model=scenario_model,
            **common_optimizer_args(job),
            processes=int(job["processes"]),
            time_limit=float(job["time_limit"]),
        )
        optimizer.warmup(np.asarray(warm_solution.X_opt, dtype=int))
        solution = optimizer.optimize()
        solution.details["scenario"] = scenario_index
        solutions.append(solution)
    return solutions[0] if len(solutions) == 1 else solutions


def run_job(
    model: Any,
    genomes: pd.DataFrame,
    metagenomes: pd.DataFrame,
    job: Mapping[str, Any],
):
    if job["strategy"] == "mimic_v1":
        return run_mimic_v1(model, job)
    if job["strategy"] == "heuristic":
        return run_heuristic(model, job)
    if job["strategy"] == "genetic":
        return run_genetic(model, job)
    if job["strategy"] == "milp":
        return run_milp(model, genomes, metagenomes, job)
    raise ValueError(f"Unknown strategy {job['strategy']!r}.")


def solution_records(
    result: Any,
    job: Mapping[str, Any],
    run_id: str,
    truth: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    solutions = result if isinstance(result, list) else [result]
    return [solution_record(solution, job, run_id, truth) for solution in solutions]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def selection_recovery_metrics(selected: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    selected_set = set(selected)
    expected_set = set(expected)
    tp = len(selected_set & expected_set)
    fp = len(selected_set - expected_set)
    fn = len(expected_set - selected_set)
    union = selected_set | expected_set
    return {
        "expected_selected_names": ";".join(expected),
        "selection_TP": tp,
        "selection_FP": fp,
        "selection_FN": fn,
        "selection_jaccard": float(tp / len(union)) if union else 1.0,
        "exact_selection_recovery": selected_set == expected_set,
    }


def solution_record(
    solution: Any,
    job: Mapping[str, Any],
    run_id: str,
    truth: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    details = solution.details or {}
    analysis = details.get("analysis", {})
    x_opt = np.asarray(solution.X_opt)
    record = {
        "profile": job["profile"],
        "run_id": run_id,
        "strategy": job["strategy"],
        "method": solution.method,
        "scenario_index": details.get("scenario"),
        "scenario_name": solution.name,
        "consortia_size": int(job["consortia_size"]),
        "selected_count": int(x_opt.sum()),
        "selected_names": ";".join(solution.selected_names),
        "objective": _json_value(solution.objective),
        "runtime_seconds": _json_value(details.get("runtime")),
        "absence_cover_penalty": job.get("absence_cover_penalty"),
        "absence_match_reward": job.get("absence_match_reward"),
        "heuristic_mask_key": job.get("heuristic_mask_key"),
        "population_size": job.get("population_size"),
        "generations": job.get("generations"),
        "max_unchanged_generations": job.get("max_unchanged_generations"),
        "generations_run": details.get("generations_run"),
        "processes": job.get("processes"),
        "time_limit": job.get("time_limit"),
        "warm_start": job.get("warm_start"),
        "solver_status": details.get("solver_status"),
        "solver_status_code": details.get("solver_status_code"),
        "mip_gap": details.get("mip_gap"),
    }
    for key, value in analysis.items():
        value = _json_value(value)
        record[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    if truth and solution.name in truth:
        record.update(selection_recovery_metrics(solution.selected_names, truth[solution.name]))
    return record


def write_summary(records: list[dict[str, Any]], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"

    frame = pd.DataFrame(records)
    first_columns = [
        "profile",
        "run_id",
        "strategy",
        "method",
        "scenario_index",
        "scenario_name",
        "consortia_size",
        "selected_count",
        "objective",
        "runtime_seconds",
        "absence_cover_penalty",
        "absence_match_reward",
        "heuristic_mask_key",
        "time_limit",
        "solver_status",
        "solver_status_code",
        "mip_gap",
    ]
    ordered = [column for column in first_columns if column in frame.columns]
    ordered += [column for column in frame.columns if column not in ordered]
    frame.loc[:, ordered].to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def job_line(job: Mapping[str, Any]) -> str:
    fields = [
        job["strategy"],
        f"k={job['consortia_size']}",
    ]
    if "absence_cover_penalty" in job:
        fields.append(f"acp={job['absence_cover_penalty']}")
    if "absence_match_reward" in job:
        fields.append(f"amr={job['absence_match_reward']}")
    if job["strategy"] == "heuristic":
        fields.append(job["heuristic_mask_key"])
    return " ".join(fields)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile = load_profile(args.profile, args.profiles_path)
    jobs = expand_profile(
        profile,
        strategies=args.strategies,
        consortia_sizes=args.sizes,
        absence_cover_penalties=args.acps,
        absence_match_rewards=args.amrs,
        heuristic_mask_keys=args.mask_keys,
    )
    if args.processes is not None:
        for job in jobs:
            if job["strategy"] in {"genetic", "milp"}:
                job["processes"] = args.processes
    if not jobs:
        raise SystemExit("No benchmark jobs matched the requested filters.")

    if args.dry_run:
        for job in jobs:
            print(job_line(job))
        print(f"{len(jobs)} job(s)")
        return 0

    genomes, metagenomes = load_benchmark_frames(profile)
    model = build_model(genomes, metagenomes)
    run_id = args.run_id or utc_run_id()
    records: list[dict[str, Any]] = []
    truth = profile.get("truth")
    for job in jobs:
        result = run_job(model, genomes, metagenomes, job)
        records.extend(solution_records(result, job, run_id, truth))

    output_dir = args.output_dir or OUTPUTS_ROOT / args.profile / "results"
    csv_path, json_path = write_summary(records, output_dir, args.summary_prefix)
    print(f"Wrote {len(records)} row(s) to {csv_path}")
    print(f"Wrote JSON summary to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
