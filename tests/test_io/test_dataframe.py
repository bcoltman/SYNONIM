import logging

import pandas as pd
import pytest

from synonim.io import model_from_frames


def test_model_from_frames_builds_expected_profiles_and_matrices(toy_frames):
    model = model_from_frames(
        genomes_binary=toy_frames["genomes_binary"],
        metagenomes_binary=toy_frames["metagenomes_binary"],
        genomes_info=toy_frames["genomes_info"],
        taxonomy_cols=toy_frames["taxonomy_cols"],
        model_id="toy",
        model_name="Toy model",
    )

    assert model.id == "toy"
    assert model.name == "Toy model"
    assert model.genome_names == toy_frames["genomes"]
    assert model.metagenome_names == toy_frames["metagenomes"]
    assert [feature.id for feature in model.features] == ["f1", "f2", "f3", "f4"]
    assert model.genome_binary_matrix.shape == (4, 3)
    assert model.metagenome_binary_matrix.shape == (4, 2)
    assert model.genome_profiles["g1"].taxonomy == {"domain": "bacteria", "genus": "A"}
    assert model.genome_profiles["g3"].metadata["source"] == "culture"
    assert model.genome_profiles["g1"].features[model.features["f1"]] == {"presence": 1}


def test_model_from_frames_warns_when_genome_info_is_not_indexed_by_sample(caplog):
    genomes_binary = pd.DataFrame({"g1": [1]}, index=["f1"])
    metagenomes_binary = pd.DataFrame({"m1": [1]}, index=["f1"])
    genomes_info = pd.DataFrame({"genome_id": ["g1"], "genus": ["A"]})

    with caplog.at_level(logging.WARNING):
        model = model_from_frames(
            genomes_binary=genomes_binary,
            metagenomes_binary=metagenomes_binary,
            genomes_info=genomes_info,
            taxonomy_cols=["genus"],
        )

    assert model.genome_profiles["g1"].taxonomy == {}
    assert "genomes_info must be indexed by sample ID" in caplog.text


def test_model_from_frames_requires_binary_data():
    with pytest.raises(ValueError, match="At least one binary data frame"):
        model_from_frames()
