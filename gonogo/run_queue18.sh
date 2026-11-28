#!/usr/bin/env bash
# Confirmation data for the selection rules on a second model family: Llama-3.1-8B-Instruct judges Qwen3-8B's
# natural sets through order-balanced duels.
#   (1) answer duels: one representative per distinct answer, every ordered pair (as queue 13/E6 for Qwen3)
#   (2) all-candidate duels on the N = 8 sets: every ordered pair of the 8 candidates (as queue 17)
# The analysis is pre-registered: do NOT analyse these files before the analysis plan
# (out/e6_duels/prereg_llama.md) is committed. Collecting the data first is fine; it does not depend on the rules.
#
#   LLAMA=/path/to/Llama-3.1-8B-Instruct tmux new-session -d -s attention_q18 'bash run_queue18.sh >>logs/queue18.log 2>&1'
# Waits for GPU 0. Expected time: ~30-60 min in total on one A800.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
: "${LLAMA:?set LLAMA to the local Llama-3.1-8B-Instruct directory}"
out=out/e6_duels
mkdir -p "$out" logs
exec 9>"$out/queue18.lock"
flock -n 9 || { echo "queue 18 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR
"$PY" -m py_compile e6_run.py

wait_gpu() {
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $1"
        sleep 60
    done
}

run() {   # name, output, extra e6_run arguments...
    local name=$1 data=$2
    shift 2
    if [[ -s "$data" ]]; then
        echo "[$(date -Is)] REUSE $name"
        cp "$data" "$data.tmp"
    elif [[ ! -s "$data.tmp" ]]; then
        wait_gpu "$name"
        echo "[$(date -Is)] START $name"
        CUDA_VISIBLE_DEVICES=0 "$PY" e6_run.py --model qwen3-8b --judge "$LLAMA" "$@" --out "$data.tmp" \
            >>"logs/e6_$name.log" 2>&1
    fi
}

verify() {   # data, mode (answers | allcand)
    "$PY" - "$1" "$2" <<'PY'
import json, sys
from collections import Counter
from e6_run import representatives
data, mode = sys.argv[1:]
rows = [json.loads(l) for l in open(data) if l.strip()]
sets = [json.loads(l) for l in open("out/e1_sets_qwen3-8b.jsonl") if l.strip()]
if mode == "allcand":
    want = {s["set_id"]: 56 for s in sets if s["N"] == 8}
else:
    want = {s["set_id"]: len(representatives(s)) * (len(representatives(s)) - 1) for s in sets}
    want = {k: v for k, v in want.items() if v}
got = Counter(r["set_id"] for r in rows)
ids = Counter(r["request_id"] for r in rows)
if not rows or max(ids.values()) != 1 or dict(got) != want:
    raise SystemExit(f"incomplete: {len(rows)} rows over {len(got)} sets, want {sum(want.values())} over {len(want)}")
if any("llama" not in str(r.get("judge", "")).lower() for r in rows):
    raise SystemExit("rows without the Llama judge tag")
low = sum(r["p_mass"] < 0.5 for r in rows)
print(f"Verified {len(rows)} duels over {len(got)} sets ({low} with label mass < 0.5)")
PY
}

a="$out/e6_llama-3.1-8b_on_qwen3-8b.jsonl"
run llama-answers "$a"
verify "$a.tmp" answers
mv "$a.tmp" "$a"
echo "[$(date -Is)] OK answer duels"

c="$out/e6_allcand_N8_llama-3.1-8b_on_qwen3-8b.jsonl"
run llama-allcand "$c" --all-candidates --n 8
verify "$c.tmp" allcand
mv "$c.tmp" "$c"
echo "[$(date -Is)] OK all-candidate duels"
echo "[$(date -Is)] QUEUE18_DONE"
