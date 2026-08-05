#!/usr/bin/env bash
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SYNONIM_BENCHMARK_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

: "${SYNONIM_BENCHMARK_PROFILE:?Set SYNONIM_BENCHMARK_PROFILE to tiny, medium, recovery, or large.}"
: "${SYNONIM_BENCHMARK_STRATEGY:?Set SYNONIM_BENCHMARK_STRATEGY to mimic_v1, heuristic, genetic, or milp.}"
: "${SYNONIM_BENCHMARK_CONSORTIA_SIZES:?Set SYNONIM_BENCHMARK_CONSORTIA_SIZES to the benchmark consortia sizes.}"

if [[ -n "${SYNONIM_BENCHMARK_SETUP_COMMAND:-}" ]]; then
  eval "${SYNONIM_BENCHMARK_SETUP_COMMAND}"
fi

summary_prefix="binary_${SYNONIM_BENCHMARK_STRATEGY}"

ARGS=(
  "${REPO_ROOT}/benchmarks/run_binary_optimizers.py"
  --profile "${SYNONIM_BENCHMARK_PROFILE}"
  --strategy "${SYNONIM_BENCHMARK_STRATEGY}"
  --output-dir "${SYNONIM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmarks/outputs/${SYNONIM_BENCHMARK_PROFILE}/results}"
  --run-id "${SYNONIM_BENCHMARK_RUN_ID:?Set SYNONIM_BENCHMARK_RUN_ID for an isolated benchmark submission.}"
)

if [[ -n "${SYNONIM_BENCHMARK_PROFILES_PATH:-}" ]]; then
  ARGS+=(--profiles-path "${SYNONIM_BENCHMARK_PROFILES_PATH}")
fi

IFS=',' read -r -a CONSORTIA_SIZES <<< "${SYNONIM_BENCHMARK_CONSORTIA_SIZES}"
for size in "${CONSORTIA_SIZES[@]}"; do
  ARGS+=(--consortia-size "${size}")
done

if [[ "${SYNONIM_BENCHMARK_STRATEGY}" == "genetic" || "${SYNONIM_BENCHMARK_STRATEGY}" == "milp" ]]; then
  if (( ${#CONSORTIA_SIZES[@]} != 1 )); then
    echo "${SYNONIM_BENCHMARK_STRATEGY} jobs require exactly one consortium size." >&2
    exit 2
  fi
  summary_prefix="${summary_prefix}_k${CONSORTIA_SIZES[0]}"
fi

if [[ -n "${SYNONIM_BENCHMARK_ACP:-}" ]]; then
  ARGS+=(--absence-cover-penalty "${SYNONIM_BENCHMARK_ACP}")
  summary_prefix="${summary_prefix}_acp${SYNONIM_BENCHMARK_ACP}"
fi

if [[ -n "${SYNONIM_BENCHMARK_AMR:-}" ]]; then
  ARGS+=(--absence-match-reward "${SYNONIM_BENCHMARK_AMR}")
  summary_prefix="${summary_prefix}_amr${SYNONIM_BENCHMARK_AMR}"
fi

summary_prefix="${summary_prefix}${SYNONIM_BENCHMARK_NAME_SUFFIX:-}"
ARGS+=(--summary-prefix "${summary_prefix}")

if [[ -n "${SYNONIM_BENCHMARK_HEURISTIC_MASK_KEY:-}" ]]; then
  ARGS+=(--heuristic-mask-key "${SYNONIM_BENCHMARK_HEURISTIC_MASK_KEY}")
fi

if [[ -n "${SYNONIM_BENCHMARK_PROCESSES:-}" ]]; then
  ARGS+=(--processes "${SYNONIM_BENCHMARK_PROCESSES}")
fi

if [[ -n "${SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT:-}" ]]; then
  ARGS+=(--time-limit "${SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT}")
fi

python "${ARGS[@]}"
