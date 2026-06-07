#!/usr/bin/env bash
# =============================================================================
# run_reproduce_hs300_resume.sh — resume hs300 sweep from the point it was
# killed, i.e. starting at hs300_HGAT_pbirl_mean-reward_irlprior_s0.
#
# Completed before kill:
#   seed0 MLP  : irl, gail, norm×2, norm-nokl×2, mean-reward×2  (all done)
#   seed0 HGAT : irl, gail, norm×2, norm-nokl×2                 (all done)
#
# Picks up at:
#   seed0 HGAT : mean-reward irl-prior (+ gail-prior)
#   seed1      : MLP full, HGAT full
#   seed2      : MLP full, HGAT full
#
# Stage-1 priors for seed0 HGAT already exist — script uses them directly.
# =============================================================================
set -uo pipefail

PY="${PY:-/data1/jeffreytsai/SmartFolio/.venv/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

EPOCHS="${EPOCHS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DEVICE="${DEVICE:-cuda:0}"
LOGDIR="${LOGDIR:-logs}"
PB_HORIZON="${PB_HORIZON:-60}"
PB_MARGIN="${PB_MARGIN:-0.5}"
PB_RECENCY="${PB_RECENCY:-1.0}"
PB_KL_COEF="${PB_KL_COEF:-1.0}"
PB_PAIRS="${PB_PAIRS:-256}"

ORCH="run_orch_logs"
mkdir -p "$LOGDIR" "$ORCH" results
FAILLOG="$ORCH/failures_hs300_resume.txt"; : > "$FAILLOG"
ts() { date '+%F %T'; }

batch_for() { [ "$1" = "HGAT" ] && echo 32 || echo 0; }

run_job() {
  local name="$1"; shift
  local log="$ORCH/${name}.log"
  echo "[orch $(ts)] START $name"
  if "$@" > "$log" 2>&1; then
    echo "[orch $(ts)] OK    $name"
    return 0
  fi
  echo "[orch $(ts)] FAIL  $name  (see $log)"
  echo "$name" >> "$FAILLOG"
  return 1
}

bestval() {
  local m="$1" p="$2" rd="$3"
  if [ ! -f "$rd/best_model.zip" ]; then
    echo "[orch $(ts)] SKIP  bestval ${rd} (no best_model.zip)"
    echo "bestval_missing:${rd}" >> "$FAILLOG"
    return 0
  fi
  run_job "bestval_$(basename "$rd")" \
    "$PY" eval_best.py --market "$m" --policy "$p" --run_dir "$rd" --device "$DEVICE"
}

echo "[orch $(ts)] ===== hs300 resume sweep ====="
echo "[orch $(ts)] starting at seed0 HGAT mean-reward irl-prior"

# ---- seed 0 HGAT: only mean-reward (irl + gail prior) ----------------------
# Stage-1 priors already exist from the original sweep.
m=hs300; s=0; p=HGAT
bs="$(batch_for "$p")"
common=(--market "$m" --policy "$p" --seed "$s" --max_epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" --device "$DEVICE" --batch_size "$bs")
pbargs=(--reward pb_irl --pb_variant mean-reward \
        --pb_horizon "$PB_HORIZON" --pb_margin "$PB_MARGIN" \
        --pb_recency "$PB_RECENCY" --pb_kl_coef "$PB_KL_COEF" \
        --pb_pairs "$PB_PAIRS")

irl_dir="$LOGDIR/${m}_${p}_seed${s}"
gail_dir="$LOGDIR/${m}_${p}_gail_seed${s}"

run_job "${m}_${p}_pbirl_mean-reward_irlprior_s${s}" \
  "$PY" main.py "${common[@]}" "${pbargs[@]}" \
  --pb_prior_source irl --prior_run_dir "$irl_dir"
bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_mean-reward_seed${s}"

run_job "${m}_${p}_pbirl_mean-reward_gailprior_s${s}" \
  "$PY" main.py "${common[@]}" "${pbargs[@]}" \
  --pb_prior_source gail --prior_run_dir "$gail_dir"
bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_mean-reward_gailprior_seed${s}"

# ---- seeds 1 and 2: full sweep (MLP + HGAT) ---------------------------------
for s in 1 2; do
  for p in MLP HGAT; do
    bs="$(batch_for "$p")"
    common=(--market "$m" --policy "$p" --seed "$s" --max_epochs "$EPOCHS" \
            --eval_every "$EVAL_EVERY" --device "$DEVICE" --batch_size "$bs")

    irl_dir="$LOGDIR/${m}_${p}_seed${s}"
    gail_dir="$LOGDIR/${m}_${p}_gail_seed${s}"

    run_job "${m}_${p}_irl_s${s}" "$PY" main.py "${common[@]}" --reward irl
    bestval "$m" "$p" "$irl_dir"

    run_job "${m}_${p}_gail_s${s}" "$PY" main.py "${common[@]}" --reward gail
    bestval "$m" "$p" "$gail_dir"

    for v in norm norm-nokl mean-reward; do
      pbargs=(--reward pb_irl --pb_variant "$v" \
              --pb_horizon "$PB_HORIZON" --pb_margin "$PB_MARGIN" \
              --pb_recency "$PB_RECENCY" --pb_kl_coef "$PB_KL_COEF" \
              --pb_pairs "$PB_PAIRS")

      run_job "${m}_${p}_pbirl_${v}_irlprior_s${s}" \
        "$PY" main.py "${common[@]}" "${pbargs[@]}" \
        --pb_prior_source irl --prior_run_dir "$irl_dir"
      bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_${v}_seed${s}"

      run_job "${m}_${p}_pbirl_${v}_gailprior_s${s}" \
        "$PY" main.py "${common[@]}" "${pbargs[@]}" \
        --pb_prior_source gail --prior_run_dir "$gail_dir"
      bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_${v}_gailprior_seed${s}"
    done
  done
done

# ---- aggregate (collects all markets present in logs/) ----------------------
run_job "aggregate" "$PY" aggregate_results.py --logdir "$LOGDIR"

echo "[orch $(ts)] ===== DONE ====="
if [ -s "$FAILLOG" ]; then
  echo "[orch $(ts)] failures/skips ($(wc -l < "$FAILLOG")):"
  cat "$FAILLOG"
else
  echo "[orch $(ts)] no failures"
fi
