#!/bin/bash
# Wait for the two sp500 runs to finish, then auto-generate the report.
cd /home/twchuang/SmartFolio/.claude/worktrees/stupefied-lederberg-233f52
HGAT_PID="$1"
MLP_PID="$2"
echo "[finalize] waiting for sp500 runs (HGAT=$HGAT_PID MLP=$MLP_PID) $(date)"
until ! kill -0 "$HGAT_PID" 2>/dev/null && ! kill -0 "$MLP_PID" 2>/dev/null; do
    sleep 60
done
echo "[finalize] both sp500 runs exited $(date)"
.venv/bin/python -u baselines.py --market sp500
# best-validation checkpoint test metrics (the training metric is noisy,
# so the final-epoch model is not necessarily the best one)
.venv/bin/python -u eval_best.py --market sp500 --policy MLP  --device cuda:1
.venv/bin/python -u eval_best.py --market sp500 --policy HGAT --device cuda:0
.venv/bin/python -u make_report.py --market sp500
echo "[finalize] DONE — RESULTS_sp500.md + results/pnl_curve_sp500.png generated $(date)"
