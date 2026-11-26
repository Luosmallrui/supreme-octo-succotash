#!/usr/bin/env bash
# E2: first displayed candidate versus printed Solution 1, on one GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
dir=out/e2_position_label
mkdir -p "$dir" logs
exec 9>"$dir/queue.lock"
flock -n 9 || { echo "E2 queue is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL line $LINENO"' ERR

for model in qwen3-4b qwen3-8b; do
    manifest="$dir/manifest_$model.jsonl"
    "$PY" e2_design.py --model "$model" --sets "out/e1_sets_$model.jsonl" --out "$manifest.tmp"
    if [[ -e "$manifest" ]]; then
        cmp "$manifest.tmp" "$manifest"
        rm "$manifest.tmp"
    else
        mv "$manifest.tmp" "$manifest"
    fi
done
for mode in direct think; do
    for model in qwen3-4b qwen3-8b; do
        while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
            echo "[$(date -Is)] waiting for GPU 0"
            sleep 60
        done
        stem="e2_${mode}_${model}"
        echo "[$(date -Is)] START $stem"
        CUDA_VISIBLE_DEVICES=0 "$PY" e2_run.py --manifest "$dir/manifest_$model.jsonl" --model "$model" \
            --mode "$mode" --out "$dir/$stem.jsonl" --batch-size 48 >>"logs/$stem.log" 2>&1
        "$PY" e2_analyze.py --input "$dir/$stem.jsonl" --output "$dir/report_$stem.md"
        echo "[$(date -Is)] OK $stem"
    done
done
echo "[$(date -Is)] QUEUE10_DONE"
