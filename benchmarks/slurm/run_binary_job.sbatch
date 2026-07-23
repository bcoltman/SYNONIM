#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SYNONIM_BENCHMARK_PROFILE:?Set SYNONIM_BENCHMARK_PROFILE to tiny, medium, recovery, or large.}"
: "${SYNONIM_BENCHMARK_STRATEGY:?Set SYNONIM_BENCHMARK_STRATEGY to mimic_v1, heuristic, genetic, or milp.}"
: "${SYNONIM_BENCHMARK_CONSORTIA_SIZE:?Set SYNONIM_BENCHMARK_CONSORTIA_SIZE to the benchmark consortia size.}"

if [[ -n "${SYNONIM_BENCHMARK_SETUP_COMMAND:-}" ]]; then
  eval "${SYNONIM_BENCHMARK_SETUP_COMMAND}"
fi

summary_prefix="binary_${SYNONIM_BENCHMARK_STRATEGY}_k${SYNONIM_BENCHMARK_CONSORTIA_SIZE}"

ARGS=(
  "${REPO_ROOT}/benchmarks/run_binary_optimizers.py"
  --profile "${SYNONIM_BENCHMARK_PROFILE}"
  --strategy "${SYNONIM_BENCHMARK_STRATEGY}"
  --consortia-size "${SYNONIM_BENCHMARK_CONSORTIA_SIZE}"
  --output-dir "${SYNONIM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmarks/outputs/${SYNONIM_BENCHMARK_PROFILE}/results}"
)

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

python "${ARGS[@]}"
