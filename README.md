**SYNONIM**
*Synthetic Community Design via Approximation*

[![PyPI Version](https://img.shields.io/pypi/v/synonim.svg)](https://pypi.org/project/synonim/) [![Build Status](https://github.com/bcoltman/SYNONIM/actions/workflows/ci.yml/badge.svg)](https://github.com/bcoltman/SYNONIM/actions) [![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Usage](#usage)
6. [Configuration & Constraints](#configuration--constraints)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [Citation](#citation)
10. [License](#license)

---

## Overview

**SYNONIM** (SYNthetic cOmmunity desigN via approxIMation) is a flexible, modular framework for rationally designing synthetic microbial communities (SynComs). It translates high-dimensional genomic, functional, or ecological data into community designs that best represent input functional profiles.

Use SYNONIM to:

* Select isolate combinations that best represent a functional profile of a target environment e.g. metagenome-derived
* Incorporate metadata (e.g. taxonomy, phylogeny, environmental origin) as constraints into SynCom design

*SYNONIM is under active development; expect new features, detailed tutorials, and expanded test coverage soon.*

---

## Key Features

* **Objective-driven optimization**: Pick mathematical or biologically meaningful objectives (e.g., weight certain functions, ensure certain taxa included)
* **Constraint integration**: Enforce taxonomic, phylogenetic, environmental, or metadata-based constraints
* **Modular components**: Swap optimization algorithms (genetic algorithms, MILP)

---

## Installation

For development and tests:

```bash
git clone https://github.com/bcoltman/SYNONIM.git
cd SYNONIM
pip install -e ".[dev]"
```

To include the optional Gurobi-backed MILP solver:

```bash
pip install -e ".[dev,milp]"
```

Install releases from PyPI:

```bash
pip install synonim
```

Bioconda packaging is planned after the first PyPI release.

---

## Quickstart

### Binary-based Design

Load binary presence/absence tables as pandas DataFrames indexed by feature ID, with columns as sample IDs.

1. **Load binary profiles** (e.g., isolate or metagenomic function presence/absence data):

   ```python
   import pandas as pd
   from synonim.io import model_from_frames
   
   genomes_info = pd.read_csv("candidates_info.csv", index_col=0)
   genomes = pd.read_csv("candidates_binary.csv", index_col=0)
   metagenomes = pd.read_csv("metagenomes_binary.csv", index_col=0)
   ```

2. **Build a Model**:

   ```python
   model = model_from_frames(
       genomes_binary=genomes,
       metagenomes_binary=metagenomes, 
       genomes_info=genomes_info,
       taxonomy_cols=["domain", "genus"]
   )
   ```

3. **Instantiate an optimizer**:
  Specify an upper limit of one member from each genus and a total consortia size of 10. Optional
  feature weights can be supplied as a dictionary, pandas Series, pandas DataFrame, or NumPy array.
   ```python
   from synonim.optimizers.binary import BinaryHeuristic

   feature_weights = {"PF00001": 5.0, "PF00002": 2.0}

   optimizer = BinaryHeuristic(
       model=model,
       consortia_size=10,
       weights=feature_weights,
       taxonomy_constraints={"genus": {"default": {"max": 1}}},
       taxonomic_levels=["domain", "genus"],
       absence_cover_penalty=1,
       absence_match_reward=0
   )
   ```

4. **Run optimization and view results**:

   ```python
   solutions = optimizer.optimize()
   print(solutions)
   ```

---

## Usage

1. **Load data**: import isolate or metagenomic profiles (CSV, BIOM, or Pandas).
2. **Configure objectives**: customise the binary objective with feature weights and mismatch penalties.
3. **Apply constraints**: narrow search space via taxonomy, metadata ranges, or co-occurrence networks.
4. **Run optimization**: select from algorithms.
5. **Inspect results**: obtain community composition, predicted functions, and diagnostic plots.

Detailed API docs and examples are planned.

---

## Configuration & Constraints

Customize your design by:

* Setting taxonomy-level inclusion/exclusion
* and more ...


---

## Models & Feature Profiles

SYNONIM defines three core classes for programmatic model construction:

* **Feature**: Represents a functional trait (e.g., gene function, pathway). Each `Feature` has:

  * `id`: unique identifier (string)
  * `name`: human-readable label

* **Profile**: Encapsulates the feature composition of a sample, with attributes:

  * `id`, `name`: identifiers for the sample
  * `profile_type`: either "genome" or "metagenome"
  * `metadata`: arbitrary key/value pairs describing the sample
  * `taxonomy`: taxonomic annotations (for genome profiles)
  * **Features**: added via `profile.add_features({Feature: {"presence": int}})`

* **Model**: Container for a set of `Feature` and `Profile` objects, ready for optimization. A `Model` provides methods:

  * `add_features(List[Feature])` to register all features
  * `add_profiles(List[Profile])` to attach sample profiles

**Building a Model**


1. Instantiate an empty model:

   ```python
   model = Model(id_or_model="my_model", name="Example Model")
   ```
3. Create `Feature` objects (e.g., one per feature ID) and add them:

   ```python
   features = [Feature(id=fid, name=fid) for fid in ["Feat1", "Feat2", "Feat3"]]
   model.add_features(features)
   ```
4. For each sample, create a `Profile`, attach metadata and taxonomy, then add feature values:

   ```python
   
   profile = Profile(id=sample, name=sample, profile_type="genome", metadata=meta_dict, taxonomy=tax_dict)
   profile.add_features({feature: {"presence": pres} for feature, pres in ...})
   model.add_profiles([profile])
   ```
5. The assembled `Model` can now be passed to an optimizer:

   ```python
   optimizer = BinaryHeuristic(model=model, consortia_size=10)
   solution = optimizer.optimize()
   ```

---

## Optimizers & Solvers

### Binary Optimizers

* **BinaryGenetic**: Uses presence/absence matrices to select consortia members.
  **Parameters**: `model`, `consortia_size`, `weights`, `taxonomy_constraints`, `taxonomic_levels`, `population_size`, `generations`, `processes`, `absence_cover_penalty`, `absence_match_reward`.
  **Usage**:

  ```python
  optimizer = BinaryGenetic(
      model=model,
      consortia_size=5,
      taxonomy_constraints=tax_constraints,
      taxonomic_levels=["domain", "genus"],
      population_size=1000,
      generations=200,
      processes=4
  )
  solutions = optimizer.optimize()
  ```

* **BinaryHeuristic**: Fast heuristic solver
  **Parameters**: `model`, `consortia_size`, `weights`, `taxonomy_constraints`, `taxonomic_levels`, `absence_cover_penalty`, `absence_match_reward`, plus masking options (`mask_covered_absent_features`, `mask_covered_present_features`, `mask_covered_isolate_features`).
  **Usage**:

  ```python
  heur = BinaryHeuristic(...)
  heur_sols = heur.optimize()
  
  ```

* **BinaryMILP**: Mixed‑Integer Linear Programming solver for binary design.
  Requires `gurobipy`, available through the `milp` extra.
  **Parameters**: `model`, `consortia_size`, `weights`, `taxonomy_constraints`, `taxonomic_levels`, `absence_cover_penalty`, `absence_match_reward`, plus `time_limit`.
  **Usage**:

  ```python
  milp = BinaryMILP(
      model=model,
      consortia_size=5,
      taxonomy_constraints=tax_constraints,
      taxonomic_levels=["domain", "genus"],
      time_limit=86400
  )
  solutions = milp.optimize()
  ```

## Testing

Run unit and integration tests:

```bash
pytest -q -m "not milp"
```

Run the optional Gurobi-backed MILP smoke test:

```bash
pytest -q -m milp
```

## Benchmarks

Binary benchmark inputs and runners live under `benchmarks/`. They are not executed
during install, import, build, or normal tests. See `benchmarks/README.md` for the
tiny, medium, recovery, and large profiles, plotting workflow, and SLURM launcher.

---

## Contributing

Contributions are welcome! Please follow the steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes with clear messages.
4. Submit a pull request and fill out the template.
5. We review and iterate!

---


## Citation

If you use SYNONIM in your work, please cite our publication (details coming soon).  
For now, feel free to reference this repository or contact the author.

---

## License

Distributed under the [GPL-3.0 License](LICENSE).

---

## Contact

Developed by [Benjamin Coltman](mailto:benjamin.coltman@univie.ac.at)  
University of Vienna
