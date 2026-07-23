#!/usr/bin/env python3
"""Generate deterministic synthetic binary fixtures for local benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "synthetic"
FEATURES = [f"f{index:03d}" for index in range(1, 81)]

MEDIUM_GENOME_FILE = "medium_binary_genomes.txt"
MEDIUM_METAGENOME_FILE = "medium_binary_metagenomes.txt"
RECOVERY_GENOME_FILE = "recovery_binary_genomes.txt"
RECOVERY_METAGENOME_FILE = "recovery_binary_metagenomes.txt"

RECOVERY_TRUTH = {
    "recovery_k3_exact": ["source_01", "source_05", "source_09"],
    "recovery_k5_exact": ["source_02", "source_04", "source_06", "source_08", "source_10"],
    "recovery_k7_exact": [
        "source_01",
        "source_03",
        "source_05",
        "source_07",
        "source_09",
        "source_11",
        "source_12",
    ],
    "recovery_greedy_trap": ["trap_exact_a", "trap_exact_b", "trap_exact_c"],
}


def _feature_ids(indices: Sequence[int]) -> list[str]:
    return [f"f{index:03d}" for index in indices]


def _medium_clean_block(index: int) -> list[str]:
    start = (index - 1) * 5 + 1
    return _feature_ids(range(start, start + 5))


def _recovery_source_block(index: int) -> list[str]:
    start = (index - 1) * 6 + 1
    return _feature_ids(range(start, start + 6))


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(0, index=FEATURES, columns=list(columns), dtype=int)
    frame.index.name = "PfamID"
    return frame


def _set_features(frame: pd.DataFrame, column: str, features: Sequence[str]) -> None:
    frame.loc[list(dict.fromkeys(features)), column] = 1


def _or_profile(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(columns)].any(axis=1).astype(int)


def _cycled_features(features: Sequence[str], start: int, count: int) -> list[str]:
    return [features[(start + offset) % len(features)] for offset in range(count)]


def precision_first_greedy_selection(genomes: pd.DataFrame, target: pd.Series, consortia_size: int) -> list[str]:
    """Select profiles using the precision-first greedy rule."""

    remaining_target = target.astype(bool).to_numpy(copy=True)
    remaining_genomes = genomes.astype(bool).to_numpy(copy=True)
    selected: list[str] = []

    for _ in range(consortia_size):
        totals = remaining_genomes.sum(axis=0)
        matches = (remaining_genomes & remaining_target[:, None]).sum(axis=0)
        ratios = matches / totals.clip(min=1)
        ratios[totals == 0] = -1.0

        best_ratio = ratios.max()
        if best_ratio < 0:
            break
        ratio_candidates = [index for index, ratio in enumerate(ratios) if ratio == best_ratio]
        best_match = max(matches[index] for index in ratio_candidates)
        if best_match == 0:
            break

        selected_index = next(index for index in ratio_candidates if matches[index] == best_match)
        selected.append(genomes.columns[selected_index])

        matched_features = remaining_genomes[:, selected_index] & remaining_target
        remaining_target[matched_features] = False
        remaining_genomes[matched_features, :] = False
        remaining_genomes[:, selected_index] = False

    return selected


def build_medium_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the medium benchmark fixture.

    The medium fixture is deliberately small enough for a restricted Gurobi install
    while still containing clean modules, overlapping bridge genomes, partial decoys,
    rare profiles, and broad noisy profiles.
    """

    clean_names = [f"clean_{index:02d}" for index in range(1, 13)]
    bridge_names = [f"bridge_{index:02d}" for index in range(1, 9)]
    decoy_names = [f"decoy_{index:02d}" for index in range(1, 11)]
    rare_names = [f"rare_{index:02d}" for index in range(1, 7)]
    noisy_names = [f"broad_noisy_{index:02d}" for index in range(1, 7)]
    trap_names = ["trap_exact_a", "trap_exact_b", "trap_exact_c", "trap_decoy_high"]
    genomes = _empty_frame(clean_names + bridge_names + decoy_names + rare_names + noisy_names + trap_names)

    for index, name in enumerate(clean_names, start=1):
        _set_features(genomes, name, _medium_clean_block(index))

    for index, name in enumerate(bridge_names, start=1):
        left = _medium_clean_block(index)
        right = _medium_clean_block(index + 1)
        _set_features(genomes, name, left[-2:] + right[:3])

    noise_features = _feature_ids(range(61, 69))
    for index, name in enumerate(decoy_names, start=1):
        clean = _medium_clean_block(index)
        _set_features(genomes, name, clean[:4] + _cycled_features(noise_features, 2 * (index - 1), 2))

    rare_blocks = [
        _feature_ids([61, 62, 63, 64]),
        _feature_ids([65, 66, 67, 68]),
        _feature_ids([61, 63, 65, 67]),
        _feature_ids([62, 64, 66, 68]),
        _feature_ids([61, 62, 67, 68]),
        _feature_ids([63, 64, 65, 66]),
    ]
    for name, features in zip(rare_names, rare_blocks):
        _set_features(genomes, name, features)

    for index, name in enumerate(noisy_names, start=1):
        module_features = [
            _medium_clean_block(module)[(index + module) % 5]
            for module in range(index, min(index + 6, 13))
        ]
        _set_features(genomes, name, module_features + _cycled_features(noise_features, index - 1, 5))

    trap_a = _feature_ids(range(69, 73))
    trap_b = _feature_ids(range(73, 77))
    trap_c = _feature_ids(range(77, 81))
    _set_features(genomes, "trap_exact_a", trap_a)
    _set_features(genomes, "trap_exact_b", trap_b)
    _set_features(genomes, "trap_exact_c", trap_c)
    _set_features(genomes, "trap_decoy_high", trap_a[:2] + trap_b[:2] + trap_c[:2])

    metagenomes = _empty_frame(
        [
            "medium_k3_trap",
            "medium_k5_overlap",
            "medium_k7_sparse",
            "medium_noise_control",
            "medium_greedy_trap",
        ]
    )
    metagenomes["medium_k3_trap"] = _or_profile(genomes, ["clean_01", "clean_02", "clean_03"])
    metagenomes["medium_k5_overlap"] = _or_profile(
        genomes, ["clean_04", "clean_05", "clean_06", "clean_07", "clean_08"]
    )
    metagenomes["medium_k7_sparse"] = _or_profile(
        genomes,
        ["clean_02", "clean_04", "clean_06", "clean_08", "clean_10", "clean_12", "rare_01"],
    )
    metagenomes["medium_noise_control"] = (
        _or_profile(genomes, ["clean_09", "clean_10", "clean_11"]) | genomes["rare_05"]
    ).astype(int)
    metagenomes["medium_greedy_trap"] = _or_profile(
        genomes, ["trap_exact_a", "trap_exact_b", "trap_exact_c"]
    )

    return genomes, metagenomes


def build_recovery_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the OR-recovery fixture with known source genomes."""

    source_names = [f"source_{index:02d}" for index in range(1, 13)]
    alt_names = [f"alt_{index:02d}" for index in range(1, 13)]
    decoy_names = [f"decoy_{index:02d}" for index in range(1, 11)]
    trap_names = ["trap_exact_a", "trap_exact_b", "trap_exact_c", "trap_decoy_high"]
    noise_names = [f"noise_{index:02d}" for index in range(1, 7)]
    genomes = _empty_frame(source_names + alt_names + decoy_names + trap_names + noise_names)

    for index, name in enumerate(source_names, start=1):
        _set_features(genomes, name, _recovery_source_block(index))

    for index, name in enumerate(alt_names, start=1):
        own = _recovery_source_block(index)
        next_block = _recovery_source_block((index % 12) + 1)
        _set_features(genomes, name, own[:4] + next_block[:2])

    for index, name in enumerate(decoy_names, start=1):
        first = _recovery_source_block(index)
        second = _recovery_source_block(((index + 3) % 12) + 1)
        noise = _feature_ids([73 + ((index - 1) % 8)])
        _set_features(genomes, name, first[:3] + second[3:] + noise)

    trap_a = _feature_ids(range(69, 73))
    trap_b = _feature_ids(range(73, 77))
    trap_c = _feature_ids(range(77, 81))
    _set_features(genomes, "trap_exact_a", trap_a)
    _set_features(genomes, "trap_exact_b", trap_b)
    _set_features(genomes, "trap_exact_c", trap_c)
    _set_features(genomes, "trap_decoy_high", trap_a[:2] + trap_b[:2] + trap_c[:2])

    for index, name in enumerate(noise_names, start=1):
        noise_features = _feature_ids(range(73, 81))
        start = (index - 1) % len(noise_features)
        _set_features(genomes, name, noise_features[start : start + 3])

    metagenomes = _empty_frame(list(RECOVERY_TRUTH))
    for scenario, sources in RECOVERY_TRUTH.items():
        metagenomes[scenario] = _or_profile(genomes, sources)

    return genomes, metagenomes


def assert_binary_frame(frame: pd.DataFrame, name: str) -> None:
    values = set(frame.to_numpy().ravel())
    if not values <= {0, 1}:
        raise ValueError(f"{name} contains non-binary values: {sorted(values)}")


def assert_medium_trap(genomes: pd.DataFrame, metagenomes: pd.DataFrame) -> None:
    target = metagenomes["medium_greedy_trap"]
    exact_names = ["trap_exact_a", "trap_exact_b", "trap_exact_c"]
    exact_coverage = _or_profile(genomes, exact_names)
    if not exact_coverage.equals(target):
        raise ValueError("medium_greedy_trap is not exactly covered by the declared trap profiles.")

    greedy_names = precision_first_greedy_selection(genomes, target, consortia_size=3)
    greedy_coverage = _or_profile(genomes, greedy_names)
    if greedy_coverage.equals(target):
        raise ValueError("medium_greedy_trap does not defeat the precision-first greedy selector.")
    if greedy_names[0] != "trap_decoy_high":
        raise ValueError(f"Expected trap_decoy_high to be selected first, found {greedy_names[0]!r}.")


def assert_recovery_trap(genomes: pd.DataFrame, metagenomes: pd.DataFrame) -> None:
    target = metagenomes["recovery_greedy_trap"]
    exact_names = RECOVERY_TRUTH["recovery_greedy_trap"]
    exact_coverage = _or_profile(genomes, exact_names)
    if not exact_coverage.equals(target):
        raise ValueError("recovery_greedy_trap is not exactly covered by the declared source profiles.")

    greedy_names = precision_first_greedy_selection(genomes, target, consortia_size=3)
    greedy_coverage = _or_profile(genomes, greedy_names)
    if greedy_coverage.equals(target):
        raise ValueError("recovery_greedy_trap does not defeat the precision-first greedy selector.")
    if greedy_names[0] != "trap_decoy_high":
        raise ValueError(f"Expected trap_decoy_high to be selected first, found {greedy_names[0]!r}.")


def write_fixtures(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    medium_genomes, medium_metagenomes = build_medium_fixture()
    recovery_genomes, recovery_metagenomes = build_recovery_fixture()
    assert_medium_trap(medium_genomes, medium_metagenomes)
    assert_recovery_trap(recovery_genomes, recovery_metagenomes)
    fixtures = {
        MEDIUM_GENOME_FILE: medium_genomes,
        MEDIUM_METAGENOME_FILE: medium_metagenomes,
        RECOVERY_GENOME_FILE: recovery_genomes,
        RECOVERY_METAGENOME_FILE: recovery_metagenomes,
    }
    written = []
    for filename, frame in fixtures.items():
        assert_binary_frame(frame, filename)
        path = output_dir / filename
        frame.to_csv(path, sep="\t")
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic binary matrices for local benchmarks."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in write_fixtures(args.output_dir):
        frame = pd.read_csv(path, sep="\t", index_col="PfamID")
        print(f"Wrote {path} ({frame.shape[0]} features x {frame.shape[1]} profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
