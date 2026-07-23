import numpy as np
import pandas as pd

from synonim.io import model_from_frames
from synonim.optimizers.binary import BinaryHeuristic


def test_binary_heuristic_runs_on_pibc_subset(pibc_data_directory):
    genomes = pd.read_csv(
        pibc_data_directory / "pibc_binary_genomes.txt",
        sep="\t",
        index_col="PfamID",
    ).iloc[:, :25]
    metagenomes = pd.read_csv(
        pibc_data_directory / "pibc_binary_metagenomes.txt",
        sep="\t",
        index_col="PfamID",
    ).iloc[:, :1]
    common_features = genomes.index.intersection(metagenomes.index)
    model = model_from_frames(
        genomes_binary=genomes.loc[common_features],
        metagenomes_binary=metagenomes.loc[common_features],
    )

    solution = BinaryHeuristic(
        model=model,
        consortia_size=3,
        absence_cover_penalty=1,
        absence_match_reward=0,
    ).optimize()

    assert int(solution.X_opt.sum()) == 3
    assert len(solution.selected_names) == 3
    assert np.isfinite(solution.objective)
    assert solution.details["analysis"]["P"] > 0
