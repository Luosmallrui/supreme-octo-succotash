#!/bin/bash
# queue6: thinking-mode position sweep (user switched the queue from direct-mode draws/4B to this)
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
M=qwen3-8b
step selectT2_$M $PY run_select.py --model $M --specs out/specsT2_$M.jsonl --modes A --think --max_think_tokens 12288 --out out/selT2_$M.jsonl
echo "[$(date +%T)] QUEUE6_DONE"
