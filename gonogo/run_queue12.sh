#!/usr/bin/env bash
# E3 protocol v2: rerun direct (pre-filled "FINAL:"), and finish think by reusing v1 generations plus forced
# endings. Waits for queue 11 (E3 v1) to finish, because the 8B think reuse needs its complete output.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
v1=out/e3_readout
out=out/e3v2_readout
mkdir -p "$out" logs
exec 9>"$out/queue.lock"
flock -n 9 || { echo "E3 v2 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

while tmux has-session -t '=attention_e3' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 11 (tmux attention_e3) to finish"
    sleep 300
done
grep -q QUEUE11_DONE logs/queue11.log || echo "[$(date -Is)] WARNING: queue 11 did not log QUEUE11_DONE; think reuse covers only saved rows"

"$PY" -m py_compile e3_run.py e3_analyze.py
for model in qwen3-4b qwen3-8b; do
    test -s "$v1/manifest_$model.jsonl"
done

for mode in direct think; do
    for model in qwen3-4b qwen3-8b; do
        stem="e3_${mode}_${model}"
        manifest="$v1/manifest_$model.jsonl"
        data="$out/$stem.jsonl"
        while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
            echo "[$(date -Is)] waiting for GPU 0 before $stem"
            sleep 60
        done
        reuse=()
        if [[ "$mode" == think && -s "$v1/$stem.jsonl" ]]; then
            reuse=(--reuse "$v1/$stem.jsonl")
        fi
        echo "[$(date -Is)] START $stem ${reuse[*]:-}"
        CUDA_VISIBLE_DEVICES=0 "$PY" e3_run.py --manifest "$manifest" --model "$model" --mode "$mode" \
            --out "$data" --batch-size 48 "${reuse[@]}" >>"logs/e3v2_$stem.log" 2>&1
        "$PY" - "$manifest" "$data" <<'PY'
import json
import sys
from collections import Counter

def ids(path):
    with open(path) as stream:
        return Counter(json.loads(line)["request_id"] for line in stream if line.strip())

expected, observed = map(ids, sys.argv[1:])
if not expected or expected != observed or max(expected.values()) != 1:
    raise SystemExit("E3 v2 result IDs differ from the assigned manifest or include duplicates")
print(f"Verified {len(expected)} requests")
PY
        paired=()
        if [[ "$mode" == think ]]; then
            paired=(--paired-input "$out/e3_direct_$model.jsonl")
        fi
        "$PY" e3_analyze.py --input "$data" "${paired[@]}" --output "$out/report_$stem.md.tmp"
        mv "$out/report_$stem.md.tmp" "$out/report_$stem.md"
        "$PY" e3_analyze.py --input "$data" "${paired[@]}" --strict-quote --output "$out/report_${stem}_strict.md.tmp"
        mv "$out/report_${stem}_strict.md.tmp" "$out/report_${stem}_strict.md"
        echo "[$(date -Is)] OK $stem"
    done
done
echo "[$(date -Is)] QUEUE12_DONE"
