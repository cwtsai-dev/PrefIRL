#!/bin/bash
# Full paper-exact experiment sweep.
# Paper config (IJCAI-25 #1054, sec. 4.1): learning rate 1e-4, batch size 128,
# 128-d hidden layers, 8 attention heads, 200 training epochs, averaged over
# 3 repeated runs. lr / batch / hidden / heads are already set in the code;
# this runner supplies 200 epochs and seeds {0,1,2}.
cd /home/twchuang/SmartFolio/.claude/worktrees/stupefied-lederberg-233f52
PY=.venv/bin/python
RUN="$PY -u run_experiments.py --seeds 0 1 2 --gpus 0 1 2 3 --jobs-per-gpu 1 --epochs 200 --logdir run_logs"

echo "[orch] === MLP runs (paper 'w/o HGAT' ablation) === $(date)"
$RUN --markets hs300 zz500 nd100 tw50 sp500 --policies MLP --extra --eval_every 20
echo "[orch] MLP runs finished $(date)"

echo "[orch] === HGAT runs (paper full model) === $(date)"
$RUN --markets hs300 zz500 nd100 tw50 --policies HGAT --extra --eval_every 20
echo "[orch] HGAT (hs300/zz500/nd100/tw50) finished $(date)"

# sp500 HGAT: 472 stocks -> a 128 minibatch OOMs the shared GPUs; use 32.
# (All other config remains paper-exact.)
$RUN --markets sp500 --policies HGAT --extra --eval_every 20 --batch_size 32
echo "[orch] ALL EXPERIMENTS DONE $(date)"

# auto-aggregate results into results_summary.csv
$PY aggregate_results.py --logdir logs || true
echo "[orch] results aggregated $(date)"
