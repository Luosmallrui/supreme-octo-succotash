#!/bin/bash
# queue5: replaces the tail of queue4 (reordered: thinking-mode check first, 4B replication last)
cd ~/attnbasin_gonogo
export PYTHONPATH=~/attnbasin_gonogo/pylib
PY=~/venvs/vllm/bin/python3
step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > logs/$name.log 2>&1 && echo "[$(date +%T)] OK $name" || { echo "[$(date +%T)] FAIL $name"; return 1; }; }
ACAPS=4:40,8:40,16:30,32:20
M=qwen3-8b
# 1. wait for the orphaned attnA capture (started by queue4) to finish
APID=$1
while kill -0 $APID 2>/dev/null; do sleep 30; done
if tail -1 logs/attnA_$M.log | grep -q "^saved"; then echo "[$(date +%T)] OK attnA_$M"; else echo "[$(date +%T)] FAIL attnA_$M"; fi
# 2. thinking-mode selection check (mode A, N=4: 40 problems, N=8: 30 problems, draw 0)
$PY - <<'PYEOF'
import random, sys
sys.path.insert(0, ".")
from common import load_jsonl, dump_jsonl
S = [s for s in load_jsonl("out/specs_qwen3-8b.jsonl") if s["draw"] == 0 and s["N"] in (4, 8)]
rng = random.Random(0); keep = {}
for N, cap in [(4, 40), (8, 30)]:
    pids = sorted({s["pid"] for s in S if s["N"] == N}); rng.shuffle(pids); keep[N] = set(pids[:cap])
dump_jsonl([s for s in S if s["pid"] in keep[s["N"]]], "out/specsT_qwen3-8b.jsonl")
PYEOF
step selectT_$M $PY run_select.py --model $M --specs out/specsT_$M.jsonl --modes A --think --max_think_tokens 6144 --out out/selT_$M.jsonl
# 3. attention for the no-header condition
step attnC_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes C --caps $ACAPS --out out/attnC_$M.pkl
# 4. extra draws for N=16 / N=32
step build2a_$M $PY build_select.py --model $M --R 2 --Ns 16 --seed 1 --draw_offset 1 --cap_math 16:60 --out out/specs2a_$M.jsonl
step build2b_$M $PY build_select.py --model $M --R 4 --Ns 32 --seed 2 --draw_offset 1 --cap_math 32:30 --out out/specs2b_$M.jsonl
cat out/specs2a_$M.jsonl out/specs2b_$M.jsonl > out/specs2_$M.jsonl
step select2_$M $PY run_select.py --model $M --specs out/specs2_$M.jsonl --out out/sel2_$M.jsonl
# 5. 4B replication
M=qwen3-4b
step gen_$M $PY gen_candidates.py --model $M --sources math500 --out out/cands_$M.jsonl
step grade_$M $PY grade_candidates.py --inp out/cands_$M.jsonl --out out/graded_$M.jsonl
step build_$M $PY build_select.py --model $M --R 1 --cap_math 4:150,8:120,16:60,32:30 --out out/specs_$M.jsonl
step select_$M $PY run_select.py --model $M --specs out/specs_$M.jsonl --out out/sel_$M.jsonl
step attnA_$M $PY attn_capture.py --model $M --specs out/specs_$M.jsonl --modes A --caps $ACAPS --out out/attnA_$M.pkl
echo "[$(date +%T)] QUEUE_DONE"
