#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROFILE="${1:-}"

if [[ -z "${PROFILE}" ]]; then
  echo "Usage: $0 <profile> [runner filters...]" >&2
  echo "Example: $0 large --strategy heuristic --consortia-size 10" >&2
  exit 2
fi
shift || true

RUNNER="${REPO_ROOT}/benchmarks/run_binary_optimizers.py"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_binary_job.sbatch"
OUTPUT_DIR="${SYNONIM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmarks/outputs/${PROFILE}/results}"
LOG_DIR="$(pwd)/logs"
mkdir -p "${LOG_DIR}"

declare -A GROUP_SIZES
declare -A GROUP_SEEN
GROUP_ORDER=()

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" =~ ^[0-9]+[[:space:]]job\(s\)$ ]] && continue

  strategy=""
  size=""
  acp=""
  amr=""
  mask=""
  for token in ${line}; do
    case "${token}" in
      mimic_v1|heuristic|genetic|milp) strategy="${token}" ;;
      k=*) size="${token#k=}" ;;
      acp=*) acp="${token#acp=}" ;;
      amr=*) amr="${token#amr=}" ;;
      ma*-mp*-mi*) mask="${token}" ;;
    esac
  done

  group_size=""
  if [[ "${strategy}" == "genetic" || "${strategy}" == "milp" ]]; then
    group_size="${size}"
  fi
  key="${strategy}|${acp}|${amr}|${mask}|${group_size}"
  if [[ -z "${GROUP_SEEN[${key}]+x}" ]]; then
    GROUP_SEEN["${key}"]=1
    GROUP_ORDER+=("${key}")
  fi
  if [[ -n "${GROUP_SIZES[${key}]:-}" ]]; then
    GROUP_SIZES["${key}"]+=",${size}"
  else
    GROUP_SIZES["${key}"]="${size}"
  fi
done < <(python "${RUNNER}" --profile "${PROFILE}" --dry-run "$@")

for key in "${GROUP_ORDER[@]}"; do
  IFS='|' read -r strategy acp amr mask group_size <<< "${key}"

  case "${strategy}" in
    mimic_v1)
      cpus="${SYNONIM_BENCHMARK_V1_CPUS:-1}"
      mem="${SYNONIM_BENCHMARK_V1_MEM:-50G}"
      time_limit="${SYNONIM_BENCHMARK_V1_TIME:-12:00:00}"
      ;;
    heuristic)
      cpus="${SYNONIM_BENCHMARK_HEURISTIC_CPUS:-1}"
      mem="${SYNONIM_BENCHMARK_HEURISTIC_MEM:-50G}"
      time_limit="${SYNONIM_BENCHMARK_HEURISTIC_TIME:-12:00:00}"
      ;;
    genetic)
      cpus="${SYNONIM_BENCHMARK_GENETIC_CPUS:-8}"
      mem="${SYNONIM_BENCHMARK_GENETIC_MEM:-50G}"
      time_limit="${SYNONIM_BENCHMARK_GENETIC_TIME:-12:00:00}"
      ;;
    milp)
      cpus="${SYNONIM_BENCHMARK_MILP_CPUS:-8}"
      mem="${SYNONIM_BENCHMARK_MILP_MEM:-250G}"
      time_limit="${SYNONIM_BENCHMARK_MILP_TIME:-1-00:00:00}"
      ;;
    *)
      echo "Could not parse strategy from dry-run line: ${line}" >&2
      exit 1
      ;;
  esac

  name="synonim-${strategy}"
  if [[ -n "${group_size}" ]]; then
    name="${name}-k${group_size}"
  fi
  suffix=""
  extra_exports=""
  if [[ -n "${acp}" ]]; then
    name="${name}-acp${acp}"
    extra_exports="${extra_exports},SYNONIM_BENCHMARK_ACP=${acp}"
  fi
  if [[ -n "${amr}" ]]; then
    name="${name}-amr${amr}"
    extra_exports="${extra_exports},SYNONIM_BENCHMARK_AMR=${amr}"
  fi
  if [[ -n "${mask}" ]]; then
    name="${name}-${mask}"
    suffix="_${mask}"
    extra_exports="${extra_exports},SYNONIM_BENCHMARK_HEURISTIC_MASK_KEY=${mask},SYNONIM_BENCHMARK_NAME_SUFFIX=${suffix}"
  fi

  sbatch \
    --job-name="${name}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="ALL,SYNONIM_BENCHMARK_REPO_ROOT=${REPO_ROOT},SYNONIM_BENCHMARK_PROFILE=${PROFILE},SYNONIM_BENCHMARK_STRATEGY=${strategy},SYNONIM_BENCHMARK_CONSORTIA_SIZES=${GROUP_SIZES[${key}]},SYNONIM_BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},SYNONIM_BENCHMARK_PROCESSES=${cpus}${extra_exports}" \
    "${SBATCH_SCRIPT}"
done
