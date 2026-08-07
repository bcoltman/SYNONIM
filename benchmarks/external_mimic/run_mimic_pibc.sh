#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --mimic-repo PATH --output-dir PATH [--iterations N] [--python PYTHON]" >&2
}

MIMIC_REPO=""
OUTPUT_DIR=""
ITERATIONS=30
PYTHON_BIN="python3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mimic-repo) MIMIC_REPO="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MIMIC_REPO" || -z "$OUTPUT_DIR" ]]; then
  usage
  exit 2
fi
if [[ ! -d "$MIMIC_REPO" ]]; then
  echo "MiMiC repository does not exist: $MIMIC_REPO" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXAMPLE_DATA="$MIMIC_REPO/example_data"
mkdir -p "$OUTPUT_DIR"

GENOMES="$EXAMPLE_DATA/PiBC_Genome_Binary_Vector_111_update_3oct_2019.txt"
METAGENOMES="$EXAMPLE_DATA/PiBC_Vector_284_updated_3rdOct.txt"
if [[ ! -f "$GENOMES" || ! -f "$METAGENOMES" ]]; then
  echo "Could not find the documented PiBC vectors under $EXAMPLE_DATA" >&2
  exit 1
fi

git -C "$MIMIC_REPO" rev-parse HEAD > "$OUTPUT_DIR/mimic_commit.txt"
"$PYTHON_BIN" "$SCRIPT_DIR/compare_mimic.py" verify \
  --local-genomes "$REPO_ROOT/benchmarks/data/pibc/pibc_binary_genomes.txt" \
  --local-metagenomes "$REPO_ROOT/benchmarks/data/pibc/pibc_binary_metagenomes.txt" \
  --mimic-example-data "$EXAMPLE_DATA" \
  --output "$OUTPUT_DIR/pibc_identity.json"

"$PYTHON_BIN" "$SCRIPT_DIR/compare_mimic.py" normalize \
  --input "$GENOMES" --output "$OUTPUT_DIR/PiBC_Genome_Binary_Vector.txt"
"$PYTHON_BIN" "$SCRIPT_DIR/compare_mimic.py" normalize \
  --input "$METAGENOMES" --output "$OUTPUT_DIR/PiBC_Vector.txt" --columns 8

MIMIC_STEP4=""
for candidate in \
  "$MIMIC_REPO/script/step_4_mimic.R" \
  "$MIMIC_REPO/script/step_4_mimic.txt" \
  "$MIMIC_REPO/script/script_4_mimic.R"; do
  if [[ -f "$candidate" ]]; then MIMIC_STEP4="$candidate"; break; fi
done
if [[ -z "$MIMIC_STEP4" ]]; then
  echo "Could not locate MiMiC Step 4 under $MIMIC_REPO/script" >&2
  exit 1
fi
command -v Rscript >/dev/null || { echo "Rscript is required" >&2; exit 1; }

START_NS="$(date +%s%N)"
R_LOG="$OUTPUT_DIR/mimic_step4.log"
if ! Rscript --vanilla "$MIMIC_STEP4" \
    -m "$OUTPUT_DIR/PiBC_Vector.txt" \
    -g "$OUTPUT_DIR/PiBC_Genome_Binary_Vector.txt" \
    -i "$ITERATIONS" \
    -o "$OUTPUT_DIR/MiMiC.txt" \
    -k "$OUTPUT_DIR/MiMiC_KneePoint.txt" >"$R_LOG" 2>&1; then
  echo "MiMiC Step 4 failed; last log lines:" >&2
  tail -n 40 "$R_LOG" >&2
  exit 1
fi
END_NS="$(date +%s%N)"
LOGICAL_RUNTIME="$(awk "BEGIN { printf \"%.6f\", ($END_NS - $START_NS) / 1000000000 }")"

"$PYTHON_BIN" "$SCRIPT_DIR/compare_mimic.py" convert \
  --selections "$OUTPUT_DIR/MiMiC.txt" \
  --genomes "$REPO_ROOT/benchmarks/data/pibc/pibc_binary_genomes.txt" \
  --metagenomes "$REPO_ROOT/benchmarks/data/pibc/pibc_binary_metagenomes.txt" \
  --output "$OUTPUT_DIR/mimic_actual_results.csv" \
  --logical-runtime "$LOGICAL_RUNTIME"

echo "Wrote upstream MiMiC results to $OUTPUT_DIR/mimic_actual_results.csv"
