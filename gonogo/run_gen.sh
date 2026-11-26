#!/bin/bash
set -e
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
for M in qwen3-8b qwen3-4b; do
  $PY gen_candidates.py --model $M --out out/cands_$M.jsonl
  $PY grade_candidates.py --inp out/cands_$M.jsonl --out out/graded_$M.jsonl
done
echo ALL_GEN_DONE
