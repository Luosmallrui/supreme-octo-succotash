#!/usr/bin/env bash
# E4: 2x2 intervention (attention isolation x shared position ids) on the E2 layouts, direct readout, HF forward.
# Each model runs a selftest first (explicit causal mask == stock forward; isolated+shared is order-invariant)
# and the queue stops if it fails. Waits for E3 v2 (queue 12) and for GPU 0.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
out=out/e4_mask_position
mkdir -p "$out" logs
exec 9>"$out/queue.lock"
flock -n 9 || { echo "E4 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

while tmux has-session -t '=attention_e3v2' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 12 (tmux attention_e3v2)"
    sleep 300
done
"$PY" -m py_compile e4_run.py e4_analyze.py

for model in qwen3-8b qwen3-4b; do
    manifest="out/e2_position_label/manifest_$model.jsonl"
    test -s "$manifest"
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before E4 $model"
        sleep 60
    done
    echo "[$(date -Is)] SELFTEST $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$manifest" --model "$model" --selftest \
        >>"logs/e4_selftest_$model.log" 2>&1
    echo "[$(date -Is)] START E4 $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$manifest" --model "$model" \
        --out "$out/e4_$model.jsonl" >>"logs/e4_$model.log" 2>&1
    "$PY" - "$manifest" "$out/e4_$model.jsonl" "$model" <<'PY'
import json
import sys
from collections import Counter

manifest, data, model = sys.argv[1:]
expected = {json.loads(l)["request_id"] for l in open(manifest)
            if l.strip() and json.loads(l)["model"] == model and json.loads(l)["task"] == "select"}
seen = Counter((json.loads(l)["request_id"], json.loads(l)["mask"], json.loads(l)["pos"]) for l in open(data) if l.strip())
if max(seen.values()) != 1 or {k[0] for k in seen} != expected or len(seen) != 4 * len(expected):
    raise SystemExit("E4 output does not cover every layout x condition exactly once")
print(f"Verified {len(expected)} layouts x 4 conditions")
PY
    "$PY" e4_analyze.py --input "$out/e4_$model.jsonl" --output "$out/report_e4_$model.md.tmp"
    mv "$out/report_e4_$model.md.tmp" "$out/report_e4_$model.md"
    echo "[$(date -Is)] OK E4 $model"
done
echo "[$(date -Is)] QUEUE13_DONE"
