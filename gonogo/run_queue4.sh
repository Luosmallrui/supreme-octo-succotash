#!/bin/bash
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
ACAPS=4:40,8:40,16:30,32:20
M=qwen3-8b
step select_$M $PY run_select.py --model $M --specs out/specs_$M.jsonl --out out/sel_$M.jsonl
step attnA_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes A --caps $ACAPS --out out/attnA_$M.pkl
step build2a_$M $PY build_select.py --model $M --R 2 --Ns 16 --seed 1 --draw_offset 1 --cap_math 16:60 --out out/specs2a_$M.jsonl
step build2b_$M $PY build_select.py --model $M --R 4 --Ns 32 --seed 2 --draw_offset 1 --cap_math 32:30 --out out/specs2b_$M.jsonl
cat out/specs2a_$M.jsonl out/specs2b_$M.jsonl > out/specs2_$M.jsonl
step select2_$M $PY run_select.py --model $M --specs out/specs2_$M.jsonl --out out/sel2_$M.jsonl
step attnC_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes C --caps $ACAPS --out out/attnC_$M.pkl
M=qwen3-4b
step gen_$M $PY gen_candidates.py --model $M --sources math500 --out out/cands_$M.jsonl
step grade_$M $PY grade_candidates.py --inp out/cands_$M.jsonl --out out/graded_$M.jsonl
step build_$M $PY build_select.py --model $M --R 1 --cap_math 4:150,8:120,16:60,32:30 --out out/specs_$M.jsonl
step select_$M $PY run_select.py --model $M --specs out/specs_$M.jsonl --out out/sel_$M.jsonl
step attnA_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes A --caps $ACAPS --out out/attnA_$M.pkl
echo "[$(date +%T)] QUEUE_DONE"
