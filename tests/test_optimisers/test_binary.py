import pandas as pd
import pytest

from synonim.io import model_from_frames
from synonim.optimizers.binary import BinaryHeuristic


def build_toy_model(toy_frames):
    return model_from_frames(
        genomes_binary=toy_frames["genomes_binary"],
        metagenomes_binary=toy_frames["metagenomes_binary"].iloc[:, :1],
        genomes_info=toy_frames["genomes_info"],
        taxonomy_cols=toy_frames["taxonomy_cols"],
    )


def test_binary_heuristic_selects_requested_consortium_size(toy_frames):
    model = build_toy_model(toy_frames)
    optimizer = BinaryHeuristic(
        model=model,
        consortia_size=2,
        taxonomy_constraints={"genus": {"default": {"max": 1}}},
        taxonomic_levels=["domain", "genus"],
        absence_cover_penalty=1,
        absence_match_reward=0,
    )

    solution = optimizer.optimize()

    assert int(solution.X_opt.sum()) == 2
    assert len(solution.selected_names) == 2
    assert set(solution.selected_names).issubset(model.genome_names)
    assert solution.details["analysis"]["TP"] >= 1
    assert solution.details["analysis"]["Taxonomic_counts"]["genus"]


def test_binary_analysis_handles_undefined_metric_denominators():
    genomes = pd.DataFrame({"g1": [1, 0, 1], "g2": [0, 1, 0]}, index=["f1", "f2", "f3"])
    metagenomes = pd.DataFrame({"m1": [1, 1, 0]}, index=["f1", "f2", "f3"])
    model = model_from_frames(genomes_binary=genomes, metagenomes_binary=metagenomes)

    solution = BinaryHeuristic(model=model, consortia_size=2).optimize()
    analysis = solution.details["analysis"]

    assert analysis["NPV"] == 0.0
    assert analysis["F1_score"] == pytest.approx(0.8)


def test_binary_heuristic_uses_explicit_feature_weights():
    genomes = pd.DataFrame({"g1": [1, 0], "g2": [0, 1]}, index=["f1", "f2"])
    metagenomes = pd.DataFrame({"m1": [1, 1], "m2": [1, 0]}, index=["f1", "f2"])
    model = model_from_frames(genomes_binary=genomes, metagenomes_binary=metagenomes)

    optimizer = BinaryHeuristic(
        model=model,
        consortia_size=1,
        weights={"f2": 10.0},
        absence_match_reward=0,
    )
    solutions = optimizer.optimize()

    assert optimizer.weights.tolist() == [[1.0, 1.0], [10.0, 10.0]]
    assert solutions[0].selected_names == ["g2"]
    assert solutions[0].objective == pytest.approx(10.0)
    assert solutions[0].method.startswith("BinaryHeuristic_Weighted")
    assert solutions[1].selected_names == ["g1"]
