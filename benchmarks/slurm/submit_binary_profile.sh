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

FILTER_ARGS=("$@")
PROFILES_PATH=""
index=0
while (( index < ${#FILTER_ARGS[@]} )); do
  option="${FILTER_ARGS[${index}]}"
  case "${option}" in
    --strategy|--consortia-size|--absence-cover-penalty|--absence-match-reward|--heuristic-mask-key|--profiles-path)
      if (( index + 1 >= ${#FILTER_ARGS[@]} )); then
        echo "Missing value for ${option}." >&2
        exit 2
      fi
      if [[ "${option}" == "--profiles-path" ]]; then
        PROFILES_PATH="${FILTER_ARGS[$((index + 1))]}"
      fi
      index=$((index + 2))
      ;;
    *)
      echo "Unsupported launcher option '${option}'. Use profile filtering options only." >&2
      exit 2
      ;;
  esac
done

RUNNER="${REPO_ROOT}/benchmarks/run_binary_optimizers.py"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_binary_job.sbatch"
RUN_ID="${SYNONIM_BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%S%NZ)-$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid benchmark run ID '${RUN_ID}'." >&2
  exit 2
fi
OUTPUT_ROOT="${SYNONIM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmarks/outputs/${PROFILE}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Benchmark run directory already exists: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

slurm_time_to_seconds() {
  local value="$1"
  local rest="${value}"
  local days=0
  local hours=0
  local minutes=0
  local seconds=0
  local has_days=0
  local -a parts

  if [[ "${rest}" =~ ^([0-9]+)-(.+)$ ]]; then
    days=$((10#${BASH_REMATCH[1]}))
    rest="${BASH_REMATCH[2]}"
    has_days=1
  elif [[ "${rest}" == *-* ]]; then
    echo "Invalid finite SLURM time limit '${value}'." >&2
    return 1
  fi

  IFS=':' read -r -a parts <<< "${rest}"
  if (( ${#parts[@]} < 1 || ${#parts[@]} > 3 )); then
    echo "Invalid finite SLURM time limit '${value}'." >&2
    return 1
  fi
  for part in "${parts[@]}"; do
    if [[ ! "${part}" =~ ^[0-9]+$ ]]; then
      echo "Invalid finite SLURM time limit '${value}'." >&2
      return 1
    fi
  done

  if (( has_days )); then
    hours=$((10#${parts[0]}))
    (( ${#parts[@]} >= 2 )) && minutes=$((10#${parts[1]}))
    (( ${#parts[@]} == 3 )) && seconds=$((10#${parts[2]}))
    if (( hours >= 24 || minutes >= 60 || seconds >= 60 )); then
      echo "Invalid finite SLURM time limit '${value}'." >&2
      return 1
    fi
  else
    case "${#parts[@]}" in
      1) minutes=$((10#${parts[0]})) ;;
      2)
        minutes=$((10#${parts[0]}))
        seconds=$((10#${parts[1]}))
        ;;
      3)
        hours=$((10#${parts[0]}))
        minutes=$((10#${parts[1]}))
        seconds=$((10#${parts[2]}))
        ;;
    esac
    if (( (${#parts[@]} == 3 && minutes >= 60) || seconds >= 60 )); then
      echo "Invalid finite SLURM time limit '${value}'." >&2
      return 1
    fi
  fi

  local total_seconds=$((days * 86400 + hours * 3600 + minutes * 60 + seconds))
  if (( total_seconds <= 0 )); then
    echo "SLURM time limit must be greater than zero, got '${value}'." >&2
    return 1
  fi
  printf '%d\n' "${total_seconds}"
}

gurobi_time_limit_from_slurm() {
  local total_seconds
  if ! total_seconds="$(slurm_time_to_seconds "$1")"; then
    return 1
  fi
  local scaled=$((total_seconds * 4))
  local whole_seconds=$((scaled / 5))
  local remainder=$((scaled % 5))
  if (( remainder == 0 )); then
    printf '%d\n' "${whole_seconds}"
  else
    printf '%d.%d\n' "${whole_seconds}" "$((remainder * 2))"
  fi
}

declare -A GROUP_SIZES
declare -A GROUP_SEEN
GROUP_ORDER=()

DRY_RUN_OUTPUT="$(python "${RUNNER}" --profile "${PROFILE}" --run-id "${RUN_ID}" --plan-output "${MANIFEST_PATH}" --dry-run "${FILTER_ARGS[@]}")"
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
done <<< "${DRY_RUN_OUTPUT}"

for key in "${GROUP_ORDER[@]}"; do
  IFS='|' read -r strategy acp amr mask group_size <<< "${key}"
  gurobi_time_limit=""

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
      if ! gurobi_time_limit="$(gurobi_time_limit_from_slurm "${time_limit}")"; then
        exit 2
      fi
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
  EXPORTS=(
    "SYNONIM_BENCHMARK_REPO_ROOT=${REPO_ROOT}"
    "SYNONIM_BENCHMARK_PROFILE=${PROFILE}"
    "SYNONIM_BENCHMARK_STRATEGY=${strategy}"
    "SYNONIM_BENCHMARK_CONSORTIA_SIZES=${GROUP_SIZES[${key}]}"
    "SYNONIM_BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR}"
    "SYNONIM_BENCHMARK_PROCESSES=${cpus}"
    "SYNONIM_BENCHMARK_RUN_ID=${RUN_ID}"
  )
  if [[ -n "${PROFILES_PATH}" ]]; then
    EXPORTS+=("SYNONIM_BENCHMARK_PROFILES_PATH=${PROFILES_PATH}")
  fi
  if [[ -n "${gurobi_time_limit}" ]]; then
    EXPORTS+=("SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT=${gurobi_time_limit}")
  fi
  if [[ -n "${acp}" ]]; then
    name="${name}-acp${acp}"
    EXPORTS+=("SYNONIM_BENCHMARK_ACP=${acp}")
  fi
  if [[ -n "${amr}" ]]; then
    name="${name}-amr${amr}"
    EXPORTS+=("SYNONIM_BENCHMARK_AMR=${amr}")
  fi
  if [[ -n "${mask}" ]]; then
    name="${name}-${mask}"
    suffix="_${mask}"
    EXPORTS+=(
      "SYNONIM_BENCHMARK_HEURISTIC_MASK_KEY=${mask}"
      "SYNONIM_BENCHMARK_NAME_SUFFIX=${suffix}"
    )
  fi

  env "${EXPORTS[@]}" sbatch \
    --job-name="${name}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export=ALL \
    "${SBATCH_SCRIPT}"
done

echo "Submitted benchmark run ${RUN_ID}"
echo "Results: ${OUTPUT_DIR}"
echo "Plot: python ${REPO_ROOT}/benchmarks/plot_binary_results.py --results-dir ${OUTPUT_DIR}"
