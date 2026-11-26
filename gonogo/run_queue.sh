#!/bin/bash
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
for M in qwen3-8b qwen3-4b; do
  if [ ! -s out/cands_$M.jsonl ]; then step gen_$M $PY gen_candidates.py --model $M --out out/cands_$M.jsonl || continue; fi
  step grade_$M $PY grade_candidates.py --inp out/cands_$M.jsonl --out out/graded_$M.jsonl || continue
  step build_$M $PY build_select.py --model $M --out out/specs_$M.jsonl || continue
  step select_$M $PY run_select.py --model $M --specs out/specs_$M.jsonl --out out/sel_$M.jsonl
  step attnA_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes A --caps 4:60,8:60,16:40,32:24 --out out/attnA_$M.pkl
  step attnC_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes C --caps 4:60,8:60,16:40,32:24 --out out/attnC_$M.pkl
done
echo "[$(date +%T)] QUEUE_DONE"
