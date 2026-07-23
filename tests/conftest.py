from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data_directory() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def pibc_data_directory() -> Path:
    return Path(__file__).parents[1] / "benchmarks" / "data" / "pibc"


@pytest.fixture
def toy_frames():
    features = ["f1", "f2", "f3", "f4"]
    genomes = ["g1", "g2", "g3"]
    metagenomes = ["m1", "m2"]

    genomes_binary = pd.DataFrame(
        {
            "g1": [1, 0, 1, 0],
            "g2": [0, 1, 0, 0],
            "g3": [1, 1, 0, 1],
        },
        index=features,
    )
    metagenomes_binary = pd.DataFrame(
        {
            "m1": [1, 1, 0, 0],
            "m2": [1, 0, 1, 0],
        },
        index=features,
    )
    genomes_info = pd.DataFrame(
        {
            "domain": ["bacteria", "bacteria", "fungi"],
            "genus": ["A", "B", "C"],
            "source": ["isolate", "isolate", "culture"],
        },
        index=genomes,
    )

    return {
        "genomes_binary": genomes_binary,
        "metagenomes_binary": metagenomes_binary,
        "genomes_info": genomes_info,
        "taxonomy_cols": ["domain", "genus"],
        "genomes": genomes,
        "metagenomes": metagenomes,
    }
