#!/bin/bash
# E1 part 2: thinking mode, with the checkpointing / chunked-scoring version of e1_run.py gated by a smoke test
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
M=qwen3-8b
DPID=$1
while kill -0 $DPID 2>/dev/null; do sleep 30; done
if grep -q "^math500 N=4 order=r" logs/e1_direct_$M.log; then echo "[$(date +%T)] OK e1_direct_$M"; else echo "[$(date +%T)] FAIL e1_direct_$M"; fi
F='s["N"]==32'
step e1_smoke_new $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode think --orders r,maj --filter "$F" --limit 3 --max_think_tokens 300 --out out/smoke/e1_smoke_new.jsonl || exit 1
step e1_smoke_resume $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode think --orders r,maj --filter "$F" --limit 3 --gen_from out/smoke/e1_smoke_new.jsonl.gen.jsonl --out out/smoke/e1_smoke_resume.jsonl || exit 1
$PY - <<'PYEOF' || { echo "[$(date +%T)] FAIL smoke compare"; exit 1; }
import json, sys
a = [json.loads(l) for l in open("out/smoke/e1_smoke_new.jsonl")]
b = [json.loads(l) for l in open("out/smoke/e1_smoke_resume.jsonl")]
assert len(a) == len(b) == 3, (len(a), len(b))
for x, y in zip(a, b):
    assert x["N"] == 32 and abs(sum(x["p"]) - 1) < 1e-3 and max(abs(u - v) for u, v in zip(x["p"], y["p"])) < 1e-3, (x["p"][:5], y["p"][:5])
    assert any(v > 0 for v in x["p"][9:]), "no mass on two-digit indices"
print("smoke compare ok")
PYEOF
echo "[$(date +%T)] OK smoke compare"
step e1_think_$M $PY e1_run.py --model $M --sets out/e1_sets_$M.jsonl --mode think --orders r,maj --max_think_tokens 12288 --out out/e1_think_$M.jsonl
echo "[$(date +%T)] E1_DONE"
