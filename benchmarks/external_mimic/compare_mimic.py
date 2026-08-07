#!/usr/bin/env python3
"""Validate PiBC inputs and convert upstream MiMiC selections to benchmark rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GENOME_TOKEN = re.compile(r"(?:PIG|BHZ|BMZ|DB|EYZ|SYZ|ZXZ)[A-Za-z0-9_.-]+")
DEFAULT_SIZES = (1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 25, 30)


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if "PfamID" in frame.columns:
        frame = frame.set_index("PfamID")
    elif frame.columns[0].lower() in {"pfamid", "pfam", "feature", "x"}:
        frame = frame.set_index(frame.columns[0])
    elif any(str(value).upper().startswith("PF") for value in frame.iloc[:, 0].head(20)):
        frame = frame.set_index(frame.columns[0])
    elif any(str(column).upper().startswith("PF") for column in frame.columns):
        frame = frame.set_index(frame.columns[0]).T
    frame.index = frame.index.astype(str)
    frame = frame.apply(pd.to_numeric, errors="raise").astype(int)
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_mimic(source: Path, output: Path, columns: int | None = None) -> None:
    frame = _read_matrix(source)
    if columns is not None:
        if columns < 1:
            raise ValueError("--columns must be positive")
        frame = frame.iloc[:, :columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index_label="PfamID")


def verify_pibc(local_genomes: Path, local_metagenomes: Path, mimic_example: Path, output: Path) -> dict:
    upstream_genomes = mimic_example / "PiBC_Genome_Binary_Vector_111_update_3oct_2019.txt"
    upstream_metagenomes = mimic_example / "PiBC_Vector_284_updated_3rdOct.txt"
    if not upstream_genomes.exists() or not upstream_metagenomes.exists():
        raise FileNotFoundError("MiMiC example_data does not contain the documented PiBC vectors.")

    local_g = _read_matrix(local_genomes)
    local_m = _read_matrix(local_metagenomes)
    upstream_g = _read_matrix(upstream_genomes)
    upstream_m = _read_matrix(upstream_metagenomes)
    local_m = local_m.iloc[:, :8]
    upstream_m = upstream_m.iloc[:, :8]

    checks = {
        "genome_shape": list(local_g.shape),
        "metagenome_shape": list(local_m.shape),
        "genome_features_match": local_g.index.tolist() == upstream_g.index.tolist(),
        "genome_names_match": local_g.columns.tolist() == upstream_g.columns.tolist(),
        "metagenome_features_match": local_m.index.tolist() == upstream_m.index.tolist(),
        "metagenome_names_match": local_m.columns.tolist() == upstream_m.columns.tolist(),
        "genome_values_match": bool(local_g.equals(upstream_g)),
        "metagenome_values_match": bool(local_m.equals(upstream_m)),
        "local_genomes_sha256": _sha256(local_genomes),
        "local_metagenomes_sha256": _sha256(local_metagenomes),
        "upstream_genomes_sha256": _sha256(upstream_genomes),
        "upstream_metagenomes_sha256": _sha256(upstream_metagenomes),
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "genome_features_match",
            "genome_names_match",
            "metagenome_features_match",
            "metagenome_names_match",
            "genome_values_match",
            "metagenome_values_match",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise ValueError(f"PiBC identity check failed; see {output}")
    return checks


def _parse_selection_table(path: Path, genome_names: Iterable[str], scenario_names: Iterable[str]) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    known_genomes = set(genome_names)
    known_scenarios = set(scenario_names)
    selections: dict[str, list[str]] = {name: [] for name in known_scenarios}

    try:
        dialect = csv.Sniffer().sniff(text[:4096])
        rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    except csv.Error:
        rows = []
    if rows:
        fields = list(rows[0])
        scenario_field = next(
            (f for f in fields if f.lower() in {"metagenome", "sample", "scenario"}),
            next((f for f in fields if any(token in f.lower() for token in ("metagenome", "sample", "scenario"))), None),
        )
        genome_field = next(
            (f for f in fields if f.lower() in {"bacterialgenome", "genome", "species", "strain"}),
            next((f for f in fields if f != scenario_field and any(token in f.lower() for token in ("genome", "species", "strain"))), None),
        )
        if genome_field and scenario_field:
            for row in rows:
                scenario = row.get(scenario_field, "")
                genome = row.get(genome_field, "")
                if scenario in selections and genome in known_genomes and genome not in selections[scenario]:
                    selections[scenario].append(genome)
            if any(selections.values()):
                return selections

    for line in text.splitlines():
        found_genomes = [token for token in GENOME_TOKEN.findall(line) if token in known_genomes]
        if not found_genomes:
            continue
        scenario = next((name for name in known_scenarios if name in line), None)
        if scenario is None:
            scenario = next(iter(known_scenarios)) if len(known_scenarios) == 1 else None
        if scenario is not None:
            for genome in found_genomes:
                if genome not in selections[scenario]:
                    selections[scenario].append(genome)
    if not any(selections.values()):
        raise ValueError("Could not parse MiMiC selections; inspect the upstream output format.")
    return selections


def _metrics(target: np.ndarray, coverage: np.ndarray) -> dict[str, float | int]:
    target = target.astype(bool)
    coverage = coverage.astype(bool)
    tp = int(np.sum(target & coverage)); fn = int(np.sum(target & ~coverage))
    fp = int(np.sum(~target & coverage)); tn = int(np.sum(~target & ~coverage))
    p = tp + fn; n = fp + tn
    div = lambda a, b: float(a / b) if b else 0.0
    recall = div(tp, p); precision = div(tp, tp + fp)
    out: dict[str, float | int] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "P": p, "N": n}
    out.update({"TPR/recall": recall, "PPV/precision": precision, "F1_score": div(2 * precision * recall, precision + recall),
                "FPR": div(fp, n), "FNR": div(fn, p), "TNR/specificity": div(tn, n),
                "Jaccard": div(tp, tp + fn + fp), "ACC": div(tp + tn, p + n),
                "BA": (recall + div(tn, n)) / 2})
    return out


def convert(
    selections_path: Path,
    genomes_path: Path,
    metagenomes_path: Path,
    output: Path,
    sizes: tuple[int, ...],
    logical_runtime: float | None = None,
) -> None:
    genomes = _read_matrix(genomes_path)
    metagenomes = _read_matrix(metagenomes_path).iloc[:, :8]
    selections = _parse_selection_table(selections_path, genomes.columns, metagenomes.columns)
    rows = []
    for scenario_index, scenario in enumerate(metagenomes.columns):
        ordered = selections.get(scenario, [])
        if not ordered:
            raise ValueError(f"No MiMiC selection found for scenario {scenario!r}.")
        for size in sizes:
            if len(ordered) < size:
                continue
            selected = ordered[:size]
            coverage = genomes.loc[:, selected].astype(bool).any(axis=1).to_numpy()
            record = {"profile": "large", "run_id": "external-mimic", "job_key": f"external-mimic:{scenario}:{size}",
                      "strategy": "external_mimic", "method": "MiMiC_actual", "optimizer_label": "MiMiC_actual",
                      "scenario_index": scenario_index, "scenario_name": scenario, "consortia_size": size,
                      "selected_count": size, "selected_names": ";".join(selected), "scenario_has_solution": True,
                      "logical_run_runtime_seconds": logical_runtime}
            record.update(_metrics(metagenomes[scenario].to_numpy(), coverage))
            rows.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--local-genomes", type=Path, required=True); verify.add_argument("--local-metagenomes", type=Path, required=True)
    verify.add_argument("--mimic-example-data", type=Path, required=True); verify.add_argument("--output", type=Path, required=True)
    convert_parser = sub.add_parser("convert")
    convert_parser.add_argument("--selections", type=Path, required=True); convert_parser.add_argument("--genomes", type=Path, required=True)
    convert_parser.add_argument("--metagenomes", type=Path, required=True); convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    convert_parser.add_argument("--logical-runtime", type=float)
    normalize_parser = sub.add_parser("normalize")
    normalize_parser.add_argument("--input", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, required=True)
    normalize_parser.add_argument("--columns", type=int)
    args = parser.parse_args()
    if args.command == "verify":
        verify_pibc(args.local_genomes, args.local_metagenomes, args.mimic_example_data, args.output)
    elif args.command == "convert":
        convert(args.selections, args.genomes, args.metagenomes, args.output, tuple(args.sizes), args.logical_runtime)
    else:
        normalize_for_mimic(args.input, args.output, args.columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
