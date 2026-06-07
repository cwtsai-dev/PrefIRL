#!/usr/bin/env bash
# sp500 seed 2 full sweep
set -uo pipefail

PY="${PY:-.venv/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

MARKETS="sp500"
SEEDS="2"
POLICIES="${POLICIES:-MLP HGAT}"
EPOCHS="${EPOCHS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DEVICE="${DEVICE:-cuda:0}"
LOGDIR="${LOGDIR:-logs}"
PB_VARIANTS="${PB_VARIANTS:-norm norm-nokl mean-reward}"
PB_HORIZON="${PB_HORIZON:-60}"
PB_MARGIN="${PB_MARGIN:-0.5}"
PB_RECENCY="${PB_RECENCY:-1.0}"
PB_KL_COEF="${PB_KL_COEF:-1.0}"
PB_PAIRS="${PB_PAIRS:-256}"

ORCH="run_orch_logs"
mkdir -p "$LOGDIR" "$ORCH" results
FAILLOG="$ORCH/failures_sp500_seed2.txt"; : > "$FAILLOG"
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

has_data() { [ -n "$(ls "dataset/data_train_predict_$1/1_hy/"*.pkl 2>/dev/null)" ]; }

echo "[orch $(ts)] ===== sp500 seed2 sweep ====="

for m in $MARKETS; do
  if has_data "$m"; then
    echo "[orch $(ts)] data present: $m"
  else
    run_job "build_${m}" "$PY" gen_data/build_dataset.py --market "$m"
  fi
done

has_data sp500 && run_job "baselines_sp500" "$PY" baselines.py --market sp500

for s in $SEEDS; do
  for p in $POLICIES; do
    bs="$(batch_for "$p")"
    m=sp500
    common=(--market "$m" --policy "$p" --seed "$s" --max_epochs "$EPOCHS" \
            --eval_every "$EVAL_EVERY" --device "$DEVICE" --batch_size "$bs")

    irl_dir="$LOGDIR/${m}_${p}_seed${s}"
    run_job "${m}_${p}_irl_s${s}" "$PY" main.py "${common[@]}" --reward irl
    bestval "$m" "$p" "$irl_dir"

    gail_dir="$LOGDIR/${m}_${p}_gail_seed${s}"
    run_job "${m}_${p}_gail_s${s}" "$PY" main.py "${common[@]}" --reward gail
    bestval "$m" "$p" "$gail_dir"

    for v in $PB_VARIANTS; do
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

run_job "aggregate" "$PY" aggregate_results.py --logdir "$LOGDIR"

echo "[orch $(ts)] ===== DONE ====="
if [ -s "$FAILLOG" ]; then
  echo "[orch $(ts)] failures/skips ($(wc -l < "$FAILLOG")):"
  cat "$FAILLOG"
else
  echo "[orch $(ts)] no failures"
fi
