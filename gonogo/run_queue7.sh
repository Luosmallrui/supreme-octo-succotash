#!/bin/bash
# queue7: reasoning-token attention capture, after the thinking sweep (queue6) finishes
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
M=qwen3-8b
QPID=$1
while kill -0 $QPID 2>/dev/null; do sleep 60; done
grep -q "OK selectT2_$M" logs/queue.log || { echo "[$(date +%T)] SKIP attnT_$M (selectT2 did not succeed)"; exit 1; }
step attnT_smoke_$M $PY attn_think.py --model $M --sel out/selT2_$M.jsonl --caps 4:1 --out out/attnT_smoke_$M.pkl || exit 1
step attnT_$M $PY attn_think.py --model $M --sel out/selT2_$M.jsonl --caps 4:30,8:30,16:20 --out out/attnT_$M.pkl
echo "[$(date +%T)] QUEUE7_DONE"
