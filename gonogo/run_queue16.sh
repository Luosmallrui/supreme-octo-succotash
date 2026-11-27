#!/usr/bin/env bash
# Reviewer follow-ups, all on the existing E2 layouts unless noted:
#   (1) E7 label -> final-answer lookup (direct): does the selector bind printed labels to candidates?
#       Qwen3-8B and Qwen3-4B on their own layouts; every judge in JUDGES on Qwen3-8B's layouts.
#   (2) E2 crossed position x label design (direct) with judges from other families and a larger Qwen3, all on
#       Qwen3-8B's candidates: JUDGES="/path/Llama-3.1-8B-Instruct /path/gemma-3-12b-it /path/Qwen3-32B".
#   (3) E4 forward-pass intervention (2x2 + PINE-style bidirectional) on one non-Qwen judge, E4_JUDGE=/path/to/a
#       Llama-architecture model (Gemma-3's sliding-window layers would be overridden by the custom mask).
#       The selftest must pass first (explicit causal mask = stock forward; invariant conditions order-free).
#   (4) E3 thinking-mode label/answer/solo readouts rerun with Qwen3's recommended sampling instead of greedy
#       decoding (E3_SAMPLE=0 skips it; it is the longest step, ~1-2 h per model).
# Waits for queue 15 (tmux attention_q15) and for GPU 0. Every judge is optional: unset variables skip steps.
# Large judges: Qwen3-32B in bf16 needs an 80 GB GPU; GPU_MEM_LARGE (default 0.95) is used for paths containing 32B.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$HOME/venvs/vllm/bin/python3"
e2=out/e2_position_label
e4=out/e4_mask_position
e7=out/e7_label_answer
e3=out/e3v2_readout
mkdir -p "$e2/cross" "$e4/judges" "$e7" "$e3" logs
exec 9>"$e7/queue.lock"
flock -n 9 || { echo "queue 16 is already running"; exit 1; }
trap 'echo "[$(date -Is)] FAIL at line $LINENO"' ERR

while tmux has-session -t '=attention_q15' 2>/dev/null; do
    echo "[$(date -Is)] waiting for queue 15 (tmux attention_q15)"
    sleep 300
done
"$PY" -m py_compile common.py e2_run.py e2_analyze.py e3_run.py e3_analyze.py e4_run.py e4_analyze.py e7_label_answer.py

wait_gpu() {
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; do
        echo "[$(date -Is)] waiting for GPU 0 before $1"
        sleep 60
    done
}
name_of() { basename "$1" | tr 'A-Z' 'a-z'; }
mem_of() { if [[ "$1" == *32[Bb]* ]]; then echo "${GPU_MEM_LARGE:-0.95}"; else echo 0.92; fi; }

# (1) E7 label -> answer lookup
e7_run() {  # e7_run <candidate model> <judge key or path>
    local cand=$1 judge=$2 name out
    name=$(name_of "$judge")
    out="$e7/e7_${name}_on_${cand}.jsonl"
    wait_gpu "E7 $name on $cand"
    echo "[$(date -Is)] START E7 $name on $cand"
    CUDA_VISIBLE_DEVICES=0 "$PY" e7_label_answer.py --manifest "$e2/manifest_$cand.jsonl" --model "$cand" \
        --judge "$judge" --judge-name "$name" --gpu-mem "$(mem_of "$judge")" --max-model-len 26624 \
        --out "$out" >>"logs/e7_${name}_on_${cand}.log" 2>&1
    "$PY" - "$out" <<'PY'
import json, sys
ids = [json.loads(l)["request_id"] for l in open(sys.argv[1]) if l.strip()]
if not ids or len(ids) != len(set(ids)):
    raise SystemExit("E7 output empty or has duplicate request IDs")
print(f"Verified {len(ids)} E7 queries")
PY
    echo "[$(date -Is)] OK E7 $name on $cand"
}
if [[ "${E7_RUN:-1}" == 1 ]]; then
    e7_run qwen3-8b qwen3-8b
    e7_run qwen3-4b qwen3-4b
    for judge in ${JUDGES:-}; do e7_run qwen3-8b "$judge"; done
    "$PY" e7_label_answer.py --analyze "$e7"/e7_*.jsonl --output "$e7/report_e7.md"
fi

# (2) E2 crossed design with other judges on Qwen3-8B's layouts
for judge in ${JUDGES:-}; do
    name=$(name_of "$judge")
    out="$e2/cross/e2_direct_${name}_on_qwen3-8b.jsonl"
    wait_gpu "E2 $name"
    echo "[$(date -Is)] START E2 $name on qwen3-8b"
    CUDA_VISIBLE_DEVICES=0 "$PY" e2_run.py --manifest "$e2/manifest_qwen3-8b.jsonl" --model qwen3-8b --judge "$judge" \
        --mode direct --out "$out" --batch-size 48 --max-model-len 26624 --gpu-mem "$(mem_of "$judge")" \
        >>"logs/e2_cross_${name}_on_qwen3-8b.log" 2>&1
    "$PY" - "$e2/manifest_qwen3-8b.jsonl" "$out" <<'PY'
import json, sys
manifest, data = sys.argv[1:]
want = {json.loads(l)["request_id"] for l in open(manifest) if l.strip() and json.loads(l)["model"] == "qwen3-8b"}
got = [json.loads(l)["request_id"] for l in open(data) if l.strip()]
if len(got) != len(set(got)) or set(got) != want:
    raise SystemExit(f"E2 output incomplete: {len(set(got))} of {len(want)}")
print(f"Verified {len(want)} requests")
PY
    "$PY" e2_analyze.py --input "$out" --output "$e2/cross/report_e2_direct_${name}_on_qwen3-8b.md"
    echo "[$(date -Is)] OK E2 $name on qwen3-8b"
done

# (3) E4 on a non-Qwen judge
if [[ -n "${E4_JUDGE:-}" ]]; then
    name=$(name_of "$E4_JUDGE")
    out="$e4/judges/e4_${name}_on_qwen3-8b.jsonl"
    conds="causal:seq,isolated:seq,causal:shared,isolated:shared,bidir:seq,bidir:shared"
    wait_gpu "E4 $name"
    echo "[$(date -Is)] SELFTEST E4 $name"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$e2/manifest_qwen3-8b.jsonl" --model qwen3-8b \
        --judge "$E4_JUDGE" --judge-name "$name" --selftest --conditions "$conds" >>"logs/e4_selftest_$name.log" 2>&1
    echo "[$(date -Is)] START E4 $name"
    CUDA_VISIBLE_DEVICES=0 "$PY" e4_run.py --manifest "$e2/manifest_qwen3-8b.jsonl" --model qwen3-8b \
        --judge "$E4_JUDGE" --judge-name "$name" --conditions "$conds" --out "$out" >>"logs/e4_$name.log" 2>&1
    "$PY" - "$e2/manifest_qwen3-8b.jsonl" "$out" <<'PY'
import json, sys
from collections import Counter
manifest, data = sys.argv[1:]
want = {json.loads(l)["request_id"] for l in open(manifest)
        if l.strip() and json.loads(l)["model"] == "qwen3-8b" and json.loads(l)["task"] == "select"}
seen = Counter((json.loads(l)["request_id"], json.loads(l)["mask"], json.loads(l)["pos"]) for l in open(data) if l.strip())
if max(seen.values()) != 1 or {k[0] for k in seen} != want or len(seen) != 6 * len(want):
    raise SystemExit("E4 output does not cover every layout x condition exactly once")
print(f"Verified {len(want)} layouts x 6 conditions")
PY
    "$PY" e4_analyze.py --input "$out" --output "$e4/judges/report_e4_${name}_on_qwen3-8b.md"
    echo "[$(date -Is)] OK E4 $name"
fi

# (4) E3 thinking-mode readouts with sampling
if [[ "${E3_SAMPLE:-1}" == 1 ]]; then
    for model in qwen3-8b qwen3-4b; do
        stem="e3_think_sampled_$model"
        manifest=out/e3_readout/manifest_$model.jsonl
        wait_gpu "$stem"
        echo "[$(date -Is)] START $stem"
        CUDA_VISIBLE_DEVICES=0 "$PY" e3_run.py --manifest "$manifest" --model "$model" --mode think --sample \
            --readouts label,answer,solo_answer --out "$e3/$stem.jsonl" --batch-size 64 >>"logs/$stem.log" 2>&1
        "$PY" - "$manifest" "$e3/$stem.jsonl" "$model" <<'PY'
import json, sys
from collections import Counter
manifest, data, model = sys.argv[1:]
keep = {"label", "answer", "solo_answer"}
want = Counter(json.loads(l)["request_id"] for l in open(manifest)
               if l.strip() and json.loads(l)["model"] == model and json.loads(l)["readout"] in keep)
got = Counter(json.loads(l)["request_id"] for l in open(data) if l.strip())
if got != want or max(want.values()) != 1:
    raise SystemExit("sampled E3 output does not match the selected manifest rows")
print(f"Verified {len(want)} sampled think requests")
PY
        "$PY" e3_analyze.py --input "$e3/$stem.jsonl" --paired-input "$e3/e3_direct_$model.jsonl" \
            --output "$e3/report_$stem.md"
        echo "[$(date -Is)] OK $stem"
    done
fi
"$PY" e34_checks.py
echo "[$(date -Is)] QUEUE16_DONE"
