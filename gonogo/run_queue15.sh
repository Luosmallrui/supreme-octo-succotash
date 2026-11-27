#!/usr/bin/env bash
# Reviewer follow-ups, direct readout:
#   (1) cross-judging inside Qwen3 on the E2 layouts: Qwen3-4B judges Qwen3-8B's candidates and vice versa,
#       separating self- from other-judging at a fixed family;
#   (2) E4 follow-up: PINE-style bidirectional attention between candidate blocks, with sequential and with shared
#       position ids (the latter is order-invariant by construction; the selftest checks this);
#   (3) optional: a larger judge on Qwen3-8B's candidates, if LARGE_JUDGE is set to a model path
#       (e.g. LARGE_JUDGE=/path/to/Qwen3-14B bash run_queue15.sh; Qwen3-32B in bf16 needs an 80 GB GPU).
# Waits for queue 14 (tmux attention_e56) and for GPU 0. Expected time: ~10-15 min per cross-judging run,
# ~1 h (8B) and ~50 min (4B) for the two E4 conditions.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
e2=out/e2_position_label
e4=out/e4_mask_position
mkdir -p "$e2/cross" "$e4" logs
exec 9>"$e2/cross/queue.lock"
flock -n 9 || { echo "queue 15 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

while tmux has-session -t '=attention_e56' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 14 (tmux attention_e56)"
    sleep 300
done
"$PY" -m py_compile e2_run.py e2_analyze.py e4_run.py e4_analyze.py

wait_gpu() {
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $1"
        sleep 60
    done
}

cross() {  # cross <candidate model> <judge key or path> <judge name for files>
    local cand=$1 judge=$2 name=$3
    local out="$e2/cross/e2_direct_${name}_on_${cand}.jsonl"
    wait_gpu "cross-judging $name on $cand"
    echo "[$(date -Is)] START cross $name on $cand"
    CUDA_VISIBLE_DEVICES=0 "$PY" e2_run.py --manifest "$e2/manifest_$cand.jsonl" --model "$cand" --judge "$judge" \
        --mode direct --out "$out" --batch-size 48 >>"logs/e2_cross_${name}_on_${cand}.log" 2>&1
    "$PY" - "$e2/manifest_$cand.jsonl" "$out" "$cand" <<'PY'
import json, sys
manifest, data, cand = sys.argv[1:]
want = {json.loads(l)["request_id"] for l in open(manifest) if l.strip() and json.loads(l)["model"] == cand}
got = [json.loads(l)["request_id"] for l in open(data) if l.strip()]
if len(got) != len(set(got)) or set(got) != want:
    raise SystemExit(f"cross-judging output incomplete: {len(set(got))} of {len(want)}")
print(f"Verified {len(want)} requests")
PY
    "$PY" e2_analyze.py --input "$out" --output "$e2/cross/report_e2_direct_${name}_on_${cand}.md"
    echo "[$(date -Is)] OK cross $name on $cand"
}

cross qwen3-8b qwen3-4b qwen3-4b
cross qwen3-4b qwen3-8b qwen3-8b

for model in qwen3-8b qwen3-4b; do
    manifest="$e2/manifest_$model.jsonl"
    wait_gpu "E4 bidir $model"
    echo "[$(date -Is)] SELFTEST E4 bidir $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$manifest" --model "$model" --selftest \
        --conditions bidir:seq,bidir:shared >>"logs/e4_bidir_selftest_$model.log" 2>&1
    echo "[$(date -Is)] START E4 bidir $model"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$manifest" --model "$model" \
        --conditions bidir:seq,bidir:shared --out "$e4/e4_bidir_$model.jsonl" >>"logs/e4_bidir_$model.log" 2>&1
    "$PY" - "$manifest" "$e4/e4_bidir_$model.jsonl" "$model" <<'PY'
import json, sys
from collections import Counter
manifest, data, model = sys.argv[1:]
want = {json.loads(l)["request_id"] for l in open(manifest)
        if l.strip() and json.loads(l)["model"] == model and json.loads(l)["task"] == "select"}
seen = Counter((json.loads(l)["request_id"], json.loads(l)["mask"], json.loads(l)["pos"]) for l in open(data) if l.strip())
if max(seen.values()) != 1 or {k[0] for k in seen} != want or len(seen) != 2 * len(want):
    raise SystemExit("E4 bidir output does not cover every layout x condition exactly once")
print(f"Verified {len(want)} layouts x 2 bidir conditions")
PY
    "$PY" e4_analyze.py --input "$e4/e4_$model.jsonl" "$e4/e4_bidir_$model.jsonl" \
        --output "$e4/report_e4_with_bidir_$model.md"
    echo "[$(date -Is)] OK E4 bidir $model"
done

if [[ -n "${LARGE_JUDGE:-}" ]]; then
    cross qwen3-8b "$LARGE_JUDGE" "$(basename "$LARGE_JUDGE" | tr 'A-Z' 'a-z')"
fi
echo "[$(date -Is)] QUEUE15_DONE"
