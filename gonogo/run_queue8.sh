#!/bin/bash
# queue8: rerun of queue7's capture step (queue7 failed at import with a transient oneMKL load error)
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
M=qwen3-8b
step attnT_$M $PY attn_think.py --model $M --sel out/selT2_$M.jsonl --caps 4:30,8:30,16:20 --out out/attnT_$M.pkl
echo "[$(date +%T)] QUEUE8_DONE"
