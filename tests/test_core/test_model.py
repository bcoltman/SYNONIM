from synonim import Feature, Model, Profile


def test_model_registers_profile_features_and_classifies_profiles():
    feature_a = Feature("f1", "Feature 1")
    feature_b = Feature("f2", "Feature 2")
    genome = Profile("g1", "Genome 1", profile_type="genome", taxonomy={"genus": "A"})
    metagenome = Profile("m1", "Metagenome 1", profile_type="metagenome")

    genome.add_features({feature_a: {"presence": 1}, feature_b: {"presence": 0}})
    metagenome.add_features({feature_a: {"presence": 1}})

    model = Model("model")
    model.add_profiles([genome, metagenome])

    assert model.genome_profiles["g1"] is genome
    assert model.metagenome_profiles["m1"] is metagenome
    assert [feature.id for feature in model.features] == ["f1", "f2"]
    assert model.genome_binary_matrix.tolist() == [[1], [0]]
    assert model.metagenome_binary_matrix.tolist() == [[1], [0]]


def test_model_context_reverts_temporary_profile_addition():
    model = Model("model")
    feature = Feature("f1")
    profile = Profile("g1", "Genome 1", profile_type="genome")
    profile.add_features({feature: {"presence": 1}})

    with model:
        model.add_profile(profile)
        assert "g1" in model.genome_profiles
        assert model.genome_binary_matrix.tolist() == [[1]]

    assert "g1" not in model.profiles
    assert model.genome_binary_matrix.shape == (1, 0)
