"""Submit E3 once, using the existing authenticated OpenSSH master for gpu."""
import fcntl
import os
from pathlib import Path
import re
import subprocess as sp
import time

root = Path(__file__).resolve().parents[2]
os.chdir(root)
lock = open(root / 'out/e3_dispatch/submit.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
files = ['e3_design.py', 'e3_run.py', 'e3_analyze.py', 'run_queue11.sh']
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6']
print('Waiting for the existing gpu SSH connection', flush=True)
while True:
    check = sp.run(['ssh', '-O', 'check', 'gpu'], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    if check.returncode:
        time.sleep(8)
        continue
    try:
        result = sp.check_output(['ssh', *opts, 'gpu',
            'cd ~/attnbasin_gonogo && test -s out/e2_position_label/manifest_qwen3-8b.jsonl && mktemp -d .e3-submit.XXXXXXXX'], text=True)
        stage = result.strip()
        if not re.fullmatch(r'\.e3-submit\.[a-zA-Z0-9]+', stage):
            raise RuntimeError(f'Unexpected staging directory {stage!r}')
        sp.run(['scp', '-pq', *opts, *files, f'gpu:attnbasin_gonogo/{stage}/'], check=True)
        command = r"""
set -eu
cd ~/attnbasin_gonogo
stage=$1
for f in e3_design.py e3_run.py e3_analyze.py run_queue11.sh; do
    if test -e "$f"; then cmp "$stage/$f" "$f"; else ln "$stage/$f" "$f"; fi
done
~/venvs/vllm/bin/python3 -m py_compile e3_design.py e3_run.py e3_analyze.py
bash -n run_queue11.sh
if tmux has-session -t '=attention_e3' 2>/dev/null; then echo 'Already submitted'
elif test -f logs/queue11.log && grep -q 'QUEUE11_DONE' logs/queue11.log; then echo 'Already finished'
else tmux new-session -d -s attention_e3 'bash run_queue11.sh >>logs/queue11.log 2>&1'; fi
echo "E3 submitted $(date -Is)"
"""
        sp.run(['ssh', *opts, 'gpu', f'bash -s -- {stage}'], input=command, text=True, check=True)
        break
    except sp.CalledProcessError as e:
        if e.returncode != 255:
            raise
        time.sleep(8)
