import pandas as pd
import pytest

from synonim.io import model_from_frames


@pytest.mark.milp
def test_binary_milp_solves_tiny_model_with_restricted_license():
    pytest.importorskip("gurobipy")
    from synonim.optimizers.binary import BinaryMILP

    genomes = pd.DataFrame(
        {"g1": [1, 0, 0], "g2": [0, 1, 0], "g3": [0, 0, 1]},
        index=["f1", "f2", "f3"],
    )
    metagenomes = pd.DataFrame({"m1": [1, 1, 0]}, index=["f1", "f2", "f3"])
    model = model_from_frames(genomes_binary=genomes, metagenomes_binary=metagenomes)

    solution = BinaryMILP(
        model=model,
        consortia_size=2,
        absence_cover_penalty=1,
        absence_match_reward=0,
        time_limit=30,
    ).optimize()

    assert solution.selected_names == ["g1", "g2"]
    assert solution.objective == pytest.approx(4.0)
    assert solution.details["solver_status"] == "OPTIMAL"
    assert solution.details["solver_status_code"] > 0
    assert solution.details["mip_gap"] == pytest.approx(0.0)
    assert solution.details["analysis"]["TP"] == 2
    assert solution.details["analysis"]["FP"] == 0
