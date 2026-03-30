#!/usr/bin/env bash
set -euo pipefail

# Runs each *.onnx model in an isolated process and records peak RAM (RSS).
# Requires: /usr/bin/time, onnxruntime installed in the selected Python env,
# and run_onnx_once.py present in the same directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QUERY="${1:-a photo of a cat on a chair}"
OUT_JSON="${OUT_JSON:-$SCRIPT_DIR/result.json}"

if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ ! -f "$SCRIPT_DIR/run_onnx_once.py" ]]; then
  echo "ERROR: run_onnx_once.py not found in $SCRIPT_DIR"
  echo "Create it first, then rerun this script."
  exit 1
fi

shopt -s nullglob
models=("$SCRIPT_DIR/models"/*.onnx)

if [[ ${#models[@]} -eq 0 ]]; then
  echo "No .onnx files found in $SCRIPT_DIR/models"
  exit 1
fi

tmp_ndjson="$(mktemp)"
trap 'rm -f "$tmp_ndjson"' EXIT

echo "Using Python: $PYTHON_BIN"
echo "Query: $QUERY"
echo "Found ${#models[@]} model(s). Running one-by-one..."

for model_path in "${models[@]}"; do
  model_file="$(basename "$model_path")"
  tmp_json="$(mktemp)"
  tmp_time="$(mktemp)"

  echo "----"
  echo "Running: $model_file"

  set +e
  /usr/bin/time -v "$PYTHON_BIN" "$SCRIPT_DIR/run_onnx_once.py" "$model_path" --query "$QUERY" > "$tmp_json" 2> "$tmp_time"
  exit_code=$?
  set -e

  peak_rss_kb="NA"
  if grep -q "Maximum resident set size" "$tmp_time"; then
    peak_rss_kb="$(grep -m1 "Maximum resident set size" "$tmp_time" | awk -F: '{print $2}' | xargs)"
  fi

  if [[ $exit_code -eq 0 ]]; then
    status="ok"
  else
    status="error($exit_code)"
  fi

  echo "Done: $model_file | status=$status | peak_rss_kb=$peak_rss_kb"

  "$PYTHON_BIN" - "$model_file" "$model_path" "$status" "$peak_rss_kb" "$tmp_json" "$tmp_time" >> "$tmp_ndjson" <<'PY'
import json
import os
import sys

model_name, model_path, status, peak_rss_kb, run_json_path, time_log_path = sys.argv[1:7]

record = {
    "model": model_name,
    "model_path": model_path,
    "status": status,
    "peak_rss_kb": None if peak_rss_kb == "NA" else int(peak_rss_kb),
    "run": None,
    "run_parse_error": None,
    "error_excerpt": None,
}

if os.path.exists(run_json_path) and os.path.getsize(run_json_path) > 0:
    try:
        with open(run_json_path, "r", encoding="utf-8") as f:
            record["run"] = json.load(f)
    except json.JSONDecodeError as exc:
        record["run_parse_error"] = f"invalid json: {exc}"
else:
    record["run_parse_error"] = "empty output"

if status != "ok" and os.path.exists(time_log_path):
  with open(time_log_path, "r", encoding="utf-8", errors="replace") as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]
  if lines:
    # Keep the tail so traceback and runtime error are visible in result.json.
    record["error_excerpt"] = "\n".join(lines[-25:])

print(json.dumps(record))
PY

  rm -f "$tmp_json" "$tmp_time"
done

"$PYTHON_BIN" - "$tmp_ndjson" "$OUT_JSON" "$QUERY" <<'PY'
import json
import sys
from datetime import datetime, timezone

ndjson_path, out_path, query = sys.argv[1:4]

results = []
with open(ndjson_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

payload = {
    "query": query,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "results": results,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

echo "----"
echo "All runs complete."
echo "Result JSON: $OUT_JSON"
