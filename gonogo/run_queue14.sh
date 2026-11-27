#!/usr/bin/env bash
# E6 (order-balanced answer duels) and E5 (decoy in slot 1) on the E1 natural sets, direct readout, vLLM.
# Both models; each run is checked for completeness. Analyses (e5_analyze.py, e6_analyze.py) run locally after sync.
# Waits for GPU 0. Expected time: E6 ~10-15 min and E5 ~20-30 min per model.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
mkdir -p out/e5_decoy out/e6_duels logs
exec 9>"out/e6_duels/queue.lock"
flock -n 9 || { echo "queue 14 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

"$PY" -m py_compile label_readout.py e5_run.py e6_run.py

wait_gpu() {
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $1"
        sleep 60
    done
}

for model in qwen3-8b qwen3-4b; do
    test -s "out/e1_sets_$model.jsonl"
    test -s "out/graded_$model.jsonl"

    wait_gpu "E6 $model"
    echo "[$(date -Is)] START E6 $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e6_run.py --model "$model" --out "out/e6_duels/e6_$model.jsonl.tmp" \
        >>"logs/e6_$model.log" 2>&1
    "$PY" - "$model" <<'PY'
import json, sys
sys.path.insert(0, ".")
from e6_run import duel_calls
from common import load_jsonl
model = sys.argv[1]
sets = load_jsonl(f"out/e1_sets_{model}.jsonl")
problems = {p["pid"]: p["problem"] for p in json.load(open("data/problems.json", encoding="utf-8"))}
cands = {(r["pid"], r["k"]): "" for r in load_jsonl(f"out/graded_{model}.jsonl")}
want = {c["request_id"] for c in duel_calls(sets, problems, cands)}
got = [json.loads(l)["request_id"] for l in open(f"out/e6_duels/e6_{model}.jsonl.tmp") if l.strip()]
if len(got) != len(set(got)) or set(got) != want:
    raise SystemExit(f"E6 output does not cover every duel exactly once ({len(set(got))} of {len(want)})")
print(f"Verified {len(want)} duels")
PY
    mv "out/e6_duels/e6_$model.jsonl.tmp" "out/e6_duels/e6_$model.jsonl"
    echo "[$(date -Is)] OK E6 $model"

    wait_gpu "E5 $model"
    echo "[$(date -Is)] START E5 $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e5_run.py --model "$model" --out "out/e5_decoy/e5_$model.jsonl.tmp" \
        >>"logs/e5_$model.log" 2>&1
    "$PY" - "$model" <<'PY'
import json, sys
model = sys.argv[1]
rows = [json.loads(l) for l in open(f"out/e5_decoy/e5_{model}.jsonl.tmp") if l.strip()]
ids = [r["request_id"] for r in rows]
if len(ids) != len(set(ids)) or len(ids) < 2400:
    raise SystemExit(f"E5 output looks incomplete or duplicated: {len(ids)} rows, {len(set(ids))} unique")
bad = sum(r["p_mass"] < 0.5 for r in rows)
print(f"Verified {len(ids)} decoy calls ({bad} with digit mass < 0.5)")
PY
    mv "out/e5_decoy/e5_$model.jsonl.tmp" "out/e5_decoy/e5_$model.jsonl"
    echo "[$(date -Is)] OK E5 $model"
done
echo "[$(date -Is)] QUEUE14_DONE"
