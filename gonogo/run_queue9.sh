#!/bin/bash
# queue9: Qwen3-4B replication, including direct/thinking E1; stop at the first failed step.
cd ~/attnbasin_gonogo || exit 1
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() {
    name=$1
    shift
    echo "[$(date +%T)] START $name"
    if "$@" > "logs/$name.log" 2>&1; then
        echo "[$(date +%T)] OK $name"
    else
        echo "[$(date +%T)] FAIL $name"
        exit 1
    fi
}

M=qwen3-4b
ACAPS=4:40,8:40,16:30,32:20
mkdir -p out/smoke

step smoke_gen_$M $PY gen_candidates.py --model $M --sources math500 --limit 1 --n 8 --out out/smoke/cands_$M.jsonl
step gen_$M $PY gen_candidates.py --model $M --sources math500,aime24,aime25,hmmt25 --out out/cands_$M.jsonl
step grade_$M $PY grade_candidates.py --inp out/cands_$M.jsonl --out out/graded_$M.jsonl
step build_$M $PY build_select.py --model $M --R 1 --cap_math 4:150,8:120,16:60,32:30 --out out/specs_$M.jsonl
step e1_sets_$M $PY e1_sets.py --model $M --out out/e1_sets_$M.jsonl
step select_$M $PY run_select.py --model $M --specs out/specs_$M.jsonl --out out/sel_$M.jsonl
step attnA_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes A --caps $ACAPS --out out/attnA_$M.pkl
step attnC_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes C --caps $ACAPS --out out/attnC_$M.pkl
step e1_direct_$M $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode direct --out out/e1_direct_$M.jsonl
step e1_think_$M $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode think --orders r,maj --max_think_tokens 12288 --out out/e1_think_$M.jsonl

echo "[$(date +%T)] QUEUE9_DONE"
