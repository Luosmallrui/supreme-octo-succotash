#!/bin/bash
# E1: positional prior + single-call calibration. Direct mode first (fast), then thinking mode.
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
M=qwen3-8b
step e1_direct_$M $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode direct --out out/e1_direct_$M.jsonl
step e1_think_$M $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode think --orders r,maj --max_think_tokens 12288 --out out/e1_think_$M.jsonl
echo "[$(date +%T)] E1_DONE"
