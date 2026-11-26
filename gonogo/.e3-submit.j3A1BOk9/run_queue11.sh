#!/usr/bin/env bash
# E3: compare label, quotation and math-answer responses on the same candidates.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
out=out/e3_readout
mkdir -p "$out" logs
exec 9>"$out/queue.lock"
flock -n 9 || { echo "E3 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

for model in qwen3-4b qwen3-8b; do
    manifest="$out/manifest_$model.jsonl"
    "$PY" e3_design.py --sets "out/e1_sets_$model.jsonl" \
        --e2-manifest "out/e2_position_label/manifest_$model.jsonl" \
        --graded "out/graded_$model.jsonl" --problems data/problems.json \
        --model "$model" --out "$manifest.tmp"
    if [[ -e "$manifest" ]]; then
        cmp "$manifest.tmp" "$manifest"
        rm "$manifest.tmp"
    else
        mv "$manifest.tmp" "$manifest"
    fi
done

for mode in direct think; do
    for model in qwen3-4b qwen3-8b; do
        stem="e3_${mode}_${model}"
        manifest="$out/manifest_$model.jsonl"
        data="$out/$stem.jsonl"
        while true; do
            pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
            [[ -z "$pids" ]] && break
            echo "[$(date -Is)] waiting for GPU 0 before $stem"
            sleep 60
        done
        echo "[$(date -Is)] START $stem"
        CUDA_VISIBLE_DEVICES=0 "$PY" e3_run.py --manifest "$manifest" --model "$model" \
            --mode "$mode" --out "$data" --batch-size 48 >>"logs/$stem.log" 2>&1
        "$PY" - "$manifest" "$data" <<'PY'
import json
import sys
from collections import Counter

def ids(path):
    with open(path) as stream:
        return Counter(json.loads(line)["request_id"] for line in stream if line.strip())

expected, observed = map(ids, sys.argv[1:])
if not expected or expected != observed or max(expected.values()) != 1:
    raise SystemExit("E3 result IDs differ from the assigned manifest or include duplicates")
print(f"Verified {len(expected)} requests")
PY
        paired=()
        if [[ "$mode" == think ]]; then
            paired=(--paired-input "$out/e3_direct_$model.jsonl")
        fi
        "$PY" e3_analyze.py --input "$data" "${paired[@]}" --output "$out/report_$stem.md.tmp"
        mv "$out/report_$stem.md.tmp" "$out/report_$stem.md"
        echo "[$(date -Is)] OK $stem"
    done
done
echo "[$(date -Is)] QUEUE11_DONE"
