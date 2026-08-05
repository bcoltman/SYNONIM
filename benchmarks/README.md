# Binary Benchmarks

This directory contains public benchmark inputs and scripts for SYNONIM binary
optimizers. Benchmarks are never run during package install, import, build, or
normal tests; run them explicitly from this directory or the repository root.

## Data

Input tables are split by source:

- `benchmarks/data/pibc/pibc_binary_genomes.txt`: Pfam-by-genome PiBC matrix.
- `benchmarks/data/pibc/pibc_binary_metagenomes.txt`: Pfam-by-metagenome PiBC matrix.
- `benchmarks/data/synthetic/medium_binary_*.txt`: deterministic synthetic medium benchmark matrices.
- `benchmarks/data/synthetic/recovery_binary_*.txt`: deterministic synthetic OR-recovery benchmark matrices.

The PiBC matrices were released with MiMiC and are used here as benchmark fixtures.
All tables use `PfamID` as the feature index column.

## Profiles

`profiles.json` is the benchmark matrix. It is deliberately small and flat:

- `data` selects a reproducible slice of the input tables and records expected counts.
- `runs` contains one section per optimizer strategy.
- `consortia_sizes` lists the community sizes to evaluate.
- `config_grid` is expanded as a Cartesian product.

Available profiles:

- `tiny`: very fast PiBC slice using 80 features, 12 genomes, and 1 scenario.
  This is for tests and CLI sanity checks, not an informative benchmark.
- `medium`: synthetic optimizer-comparison benchmark using 80 features, 46 genomes,
  and 5 scenarios. It includes clean modules, overlapping bridges, noisy decoys,
  and a precision-first greedy trap, and is sized for restricted `gurobipy`.
- `recovery`: synthetic OR-recovery benchmark using 80 features, 44 genomes, and
  4 generated metagenomes with known source-genome truth. The
  `recovery_greedy_trap` scenario demonstrates a case where the original
  precision-first greedy baseline misses the generating profiles while MILP
  recovers them. It is sized for restricted `gurobipy`.
- `large`: historical PiBC validation using the first 8 scenarios, 111 genomes,
  and 17,929 features from the MiMiC-released dataset. Full MILP runs require a
  licensed Gurobi setup.

The `medium` and `recovery` profiles evaluate the full local objective/config grid:

- MiMiC v1: original precision-first greedy baseline across the same local
  consortia sizes.
- Heuristic: ACP/AMR grid plus all 8 combinations of the three masking flags.
- Genetic: ACP/AMR grid with fixed local-safe GA settings.
- MILP: ACP/AMR grid with heuristic warm start and 300-second local time limit.

The `large` profile records the historical validation grid:

- MiMiC v1: original precision-first greedy baseline across sizes
  `1,2,3,4,5,7,10,12,15,17,20,25,30`.
- Heuristic: 32 mask/penalty/reward configurations across sizes
  `1,2,3,4,5,7,10,12,15,17,20,25,30`.
- Genetic: ACP/AMR grid across sizes `3,5,7,10,12,15,17,20,25,30`,
  population `1000`, generations `200`, and unchanged-generation stop `10`.
- MILP: ACP/AMR grid across sizes `10,20,30`, heuristic warm start, and a
  one-day solver time limit.

## Run Locally

List the jobs that a profile would run:

```bash
python benchmarks/run_binary_optimizers.py --profile tiny --dry-run
```

Run the tiny benchmark:

```bash
python benchmarks/run_binary_optimizers.py --profile tiny
```

Run one filtered medium job:

```bash
python benchmarks/run_binary_optimizers.py \
  --profile medium \
  --strategy milp \
  --consortia-size 3 \
  --absence-cover-penalty 1 \
  --absence-match-reward 0
```

Override the profile's MILP solver limit for a local run with
`--time-limit <seconds>`. This accepts a finite positive number of seconds.

Run one filtered large historical job:

```bash
python benchmarks/run_binary_optimizers.py \
  --profile large \
  --strategy heuristic \
  --consortia-size 10 \
  --absence-cover-penalty 1 \
  --absence-match-reward 0 \
  --heuristic-mask-key ma0-mp0-mi0
```

The runner creates a unique run ID and writes CSV and JSON summaries to
`benchmarks/outputs/<profile>/<run-id>/results/`. Its manifest records the exact
logical-job matrix expected in that run. Plot outputs live in the sibling
`plots/` directory.
It does not write raw solution pickles; keep large raw result bundles outside source
control and convert them into small summaries for plotting.

## Plot Results

Install plotting dependencies when needed:

```bash
pip install -e ".[benchmark]"
```

Build a summary table and plot a metric:

```bash
python benchmarks/plot_binary_results.py \
  --results-dir benchmarks/outputs/tiny/<run-id>/results \
  --metric F1_score
```

Plot relative performance against the explicit `MiMiC_v1` baseline:

```bash
python benchmarks/plot_binary_results.py \
  --results-dir benchmarks/outputs/medium/<run-id>/results \
  --plot-kind relative-heatmap
```

Plot the heuristic parameter overview for a metric. The `MiMiC_v1` baseline is
shown as the comparator column; its parameter cells are blank because it is not
one of the newer heuristic parameter combinations:

```bash
python benchmarks/plot_binary_results.py \
  --results-dir benchmarks/outputs/medium/<run-id>/results \
  --plot-kind heuristic-parameters \
  --metric F1_score
```

The plotting script discovers CSV summaries only, validates them against the run
manifest, and fails if results are missing, duplicated, mixed across runs, have
the wrong selected consortium size, or contain an infeasible solver scenario.
Use `--allow-incomplete` only to inspect a partial or legacy run.

Plots use consortia size on the x-axis and box scenario-level rows for each
optimizer configuration rather than averaging everything within the broad
heuristic/genetic/MILP classes. Labels use `MiMiC_v1` for the original baseline
and the compact historical style: `BH` for binary heuristic, `BG` for binary
genetic, `BM` for binary MILP, with mask flags
and ACP/AMR settings appended where relevant.

## SLURM

Submit one SLURM job per strategy/configuration. MiMiC v1 and heuristic jobs
evaluate all matching consortia sizes in one runner invocation; genetic and MILP
jobs remain separate per size so each receives the historical wall-time budget.
The full `large` profile therefore submits 85 jobs:

```bash
benchmarks/slurm/submit_binary_profile.sh large
```

The launcher prints the generated run ID and plotting command. All jobs in the
submission share that ID and write beneath
`benchmarks/outputs/large/<run-id>/`; the manifest expects 481 logical runs and
3,848 scenario rows. Genetic and MILP filenames include their consortium size,
so independently scheduled jobs cannot overwrite one another.

Use `--consortia-size` when you want to submit only a selected size. Otherwise,
the job evaluates every size defined for that strategy in the profile.

The launcher does not hard-code cluster paths or conda environments. Use environment
variables for local setup:

```bash
export SYNONIM_BENCHMARK_SETUP_COMMAND='module load python; source /path/to/env/bin/activate'
export SYNONIM_BENCHMARK_OUTPUT_DIR=/path/to/benchmark-output-root
benchmarks/slurm/submit_binary_profile.sh large --strategy genetic
```

`SYNONIM_BENCHMARK_OUTPUT_DIR` is the base directory; the launcher always adds
`<run-id>/results`, `<run-id>/logs`, and `<run-id>/manifest.json`. Set
`SYNONIM_BENCHMARK_RUN_ID` to a new filesystem-safe value only when a predictable
ID is required. Existing run directories are never reused.

Default resources preserve the historical settings:

- MiMiC v1: 1 CPU, 50G memory, 12 hours.
- Heuristic: 1 CPU, 50G memory, 12 hours.
- Genetic: 8 CPU, 50G memory, 12 hours.
- MILP: 8 CPU, 250G memory, 1 day for SLURM and 19 hours 12 minutes for
  the multi-scenario Gurobi solve.

Override them with `SYNONIM_BENCHMARK_HEURISTIC_CPUS`,
`SYNONIM_BENCHMARK_HEURISTIC_MEM`, `SYNONIM_BENCHMARK_HEURISTIC_TIME`, and the
corresponding `SYNONIM_BENCHMARK_V1_*`, `SYNONIM_BENCHMARK_GENETIC_*`,
or `SYNONIM_BENCHMARK_MILP_*` variables.

For submitted MILP jobs, the launcher converts `SYNONIM_BENCHMARK_MILP_TIME`
to seconds and overrides the profile's solver limit with exactly 80% of that
allocation. The remaining 20% is reserved for setup, heuristic warm starts,
result processing, and shutdown. Direct local runs continue to use the profile
limit unless `--time-limit` is supplied explicitly.

Cross-method timing uses `logical_run_runtime_seconds`: wall time for one
strategy/configuration/consortium-size invocation across all scenarios. Scenario
rows share that value, and plotting counts it once per logical job. The older
optimizer-reported `runtime_seconds` remains in summaries for diagnostics but is
not used by default comparisons.

## Local Artifacts

Generated results, plots, logs, pickle payloads, and MILP solver files are ignored by
git. Keep large historical result bundles outside committed source unless they are
converted into small stable summaries.
