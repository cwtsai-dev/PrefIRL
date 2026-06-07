#!/bin/bash
# Wait for the 4 new sp500 reward-method runs, then auto-eval best-val and
# regenerate the full report (now covering IRL + closed_form + GAIL).
cd /home/twchuang/SmartFolio/.claude/worktrees/stupefied-lederberg-233f52
PIDS="$@"  # passed: HGAT_CF_PID HGAT_GAIL_PID MLP_CF_PID MLP_GAIL_PID
echo "[finalize] waiting for sp500 runs: $PIDS  $(date)"
while true; do
    alive=0
    for p in $PIDS; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    [ "$alive" -eq 0 ] && break
    sleep 120
done
echo "[finalize] all 4 new sp500 runs done  $(date)"

# pick a GPU with free memory for the best-val eval
GPU=$(.venv/bin/python - <<'PY'
import subprocess
out = subprocess.check_output(
    ['nvidia-smi','--query-gpu=index,memory.used,memory.total',
     '--format=csv,noheader,nounits']).decode()
best, bf = 0, -1
for ln in out.strip().split('\n'):
    i, u, t = [int(x.strip()) for x in ln.split(',')]
    if t - u > bf: bf = t - u; best = i
print(best)
PY
)
echo "[finalize] best-val eval on GPU $GPU"
for pol in HGAT MLP; do
    for rew in closed_form gail; do
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u eval_best.py \
            --market sp500 --policy "$pol" --reward "$rew" --device "cuda:$GPU"
    done
done
.venv/bin/python -u baselines.py --market sp500
.venv/bin/python -u make_report.py --market sp500
echo "[finalize] DONE  RESULTS_sp500.md regenerated  $(date)"
