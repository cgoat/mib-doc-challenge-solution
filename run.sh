#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: run.sh <input_pdf_dir> <output_predictions_path>}"
output_path="${2:?usage: run.sh <input_pdf_dir> <output_predictions_path>}"

# The container root filesystem is read-only; keep all scratch under /tmp.
export MPLCONFIGDIR=/tmp
export XDG_CACHE_HOME=/tmp
export TMPDIR=/tmp

exec python -m mib.main "$input_dir" "$output_path"
