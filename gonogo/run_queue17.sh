#!/usr/bin/env bash
# Route-2 check (count-free duels): full order-balanced duel matrices over individual candidates, no grouping by
# answer, on the 130 E1 sets with N = 8 (100 MATH500 + 30 comp), 56 ordered pairs per set = 7,280 two-candidate
# prompts per model. Sparse duel schedules are then simulated offline from these matrices.
# Waits for queue 16 (tmux attention_q16) and for GPU 0. Expected time: ~15-25 min per model.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
out=out/e6_duels
mkdir -p "$out" logs
exec 9>"$out/queue17.lock"
flock -n 9 || { echo "queue 17 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

while tmux has-session -t '=attention_q16' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 16 (tmux attention_q16)"
    sleep 300
done
"$PY" -m py_compile e6_run.py

for model in qwen3-8b qwen3-4b; do
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $model"
        sleep 60
    done
    data="$out/e6_allcand_N8_$model.jsonl"
    echo "[$(date -Is)] START all-candidate duels $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e6_run.py --model "$model" --all-candidates --n 8 --out "$data.tmp" \
        >>"logs/e6_allcand_$model.log" 2>&1
    "$PY" - "$data.tmp" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = Counter(r["request_id"] for r in rows)
sets = Counter(r["set_id"] for r in rows)
if max(ids.values()) != 1 or any(v != 56 for v in sets.values()) or len(sets) != 130:
    raise SystemExit(f"incomplete: {len(rows)} rows, {len(sets)} sets")
low = sum(r["p_mass"] < 0.5 for r in rows)
print(f"Verified {len(rows)} duels over {len(sets)} sets ({low} with digit mass < 0.5)")
PY
    mv "$data.tmp" "$data"
    echo "[$(date -Is)] OK all-candidate duels $model"
done
echo "[$(date -Is)] QUEUE17_DONE"
