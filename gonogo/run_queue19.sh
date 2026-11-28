#!/usr/bin/env bash
# Confirmation data, listwise half: Llama-3.1-8B-Instruct selects on Qwen3-8B's natural sets (E1 design, direct,
# pre-filled label distribution): three random orderings, the majority-first ordering and the cyclic shifts.
# Together with queue 18 (duels) this is the pre-registered confirmation on a second family. Do NOT analyse the
# output before out/e6_duels/prereg_llama.md is committed.
#
#   LLAMA=/path/to/Llama-3.1-8B-Instruct tmux new-session -d -s attention_q19 'bash run_queue19.sh >>logs/queue19.log 2>&1'
# Waits for queue 18 (tmux attention_q18) and GPU 0. A 16-call smoke test runs first and must read the labels from
# single label tokens with at least 0.9 mean probability mass. Expected time: ~1.5-3 h on one A800.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
: "${LLAMA:?set LLAMA to the local Llama-3.1-8B-Instruct directory}"
mkdir -p out logs
exec 9>"out/queue19.lock"
flock -n 9 || { echo "queue 19 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR
"$PY" -m py_compile e1_run.py

while tmux has-session -t '=attention_q18' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 18 (tmux attention_q18)"
    sleep 300
done
wait_gpu() {
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $1"
        sleep 60
    done
}

data=out/e1_direct_llama-3.1-8b_on_qwen3-8b.jsonl
smoke=out/e1_smoke_llama-3.1-8b.jsonl
if [[ ! -s "$data" ]]; then
    wait_gpu smoke
    echo "[$(date -Is)] SMOKE E1 llama"
    CUDA_VISIBLE_DEVICES=0 "$PY" e1_run.py --model qwen3-8b --judge "$LLAMA" --sets out/e1_sets_qwen3-8b.jsonl \
        --mode direct --filter 's["N"] >= 16' --limit 16 --out "$smoke" >>logs/e1_llama-3.1-8b.log 2>&1
    "$PY" - "$smoke" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
mass = sum(r["p_mass"] for r in rows) / len(rows)
if len(rows) != 16 or any(r.get("readout") != "label tokens" for r in rows) or mass < 0.9:
    raise SystemExit(f"smoke test failed: {len(rows)} rows, readout {set(r.get('readout') for r in rows)}, mass {mass:.3f}")
print(f"Smoke test passed: 16 calls with N >= 16, mean label mass {mass:.3f}")
PY
    wait_gpu "E1 llama"
    echo "[$(date -Is)] START E1 llama"
    CUDA_VISIBLE_DEVICES=0 "$PY" e1_run.py --model qwen3-8b --judge "$LLAMA" --sets out/e1_sets_qwen3-8b.jsonl \
        --mode direct --out "$data.tmp" >>logs/e1_llama-3.1-8b.log 2>&1
    "$PY" - "$data.tmp" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
sets = [json.loads(l) for l in open("out/e1_sets_qwen3-8b.jsonl") if l.strip()]
keys = Counter((r["set_id"], r["order_name"]) for r in rows)
planned = sum(len(s["orders"]) for s in sets)
full_r = sum(all(keys[s["set_id"], f"r{i}"] == 1 for i in range(3)) for s in sets)
if max(keys.values()) != 1 or full_r < 0.98 * len(sets) or len(rows) < 0.95 * planned:
    raise SystemExit(f"incomplete: {len(rows)}/{planned} calls, {full_r}/{len(sets)} sets with all three random orders")
low = sum(r["p_mass"] < 0.5 for r in rows)
print(f"Verified {len(rows)}/{planned} calls (the rest exceed the prompt limit); {full_r}/{len(sets)} sets with all "
      f"three random orders; {low} calls with label mass < 0.5")
PY
    mv "$data.tmp" "$data"
fi
echo "[$(date -Is)] OK E1 llama"
echo "[$(date -Is)] QUEUE19_DONE"
