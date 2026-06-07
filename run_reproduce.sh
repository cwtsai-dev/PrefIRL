#!/usr/bin/env bash
# =============================================================================
# run_reproduce.sh — single sequential reproduction sweep
#
# Reproduces RESULTS_nd100_example.md-style result tables for the markets
# hs300, zz500, nd100, sp500. For each market it:
#   1. generates the preprocessed dataset if missing (gen_data/build_dataset.py)
#   2. computes the non-learning baselines (baselines.py)
#   3. trains the learned methods, SEQUENTIALLY, for seeds {0,1,2}:
#        - IRL                 ({MLP,HGAT})
#        - GAIL                ({MLP,HGAT})
#        - PB-IRL irl-prior    ({MLP,HGAT} x {norm,norm-nokl,mean-reward})
#        - PB-IRL gail-prior   ({MLP,HGAT} x {norm,norm-nokl,mean-reward})
#      and, after each run, evaluates the best-validation checkpoint
#      (eval_best.py --run_dir ...) so both *final* and *best-val* metrics exist.
#   4. aggregates everything (aggregate_results.py).
#
# Per market & seed: IRL 2 + GAIL 2 + PB-IRL irl 6 + PB-IRL gail 6 = 16 runs.
#   x 3 seeds = 48 runs/market ; x 4 markets = 192 training runs (+ best-val
#   evals, baselines, data-gen).
#
# WARNING — RUNTIME. At EPOCHS=200 (paper config) this is on the order of
# days-to-weeks of wall time on one GPU (HGAT and sp500 N=472 are slow). For a
# quick functional check, subset and shorten, e.g.:
#     MARKETS=hs300 SEEDS=0 EPOCHS=3 EVAL_EVERY=1 ./run_reproduce.sh
#
# Everything below is overridable via environment variables.
# =============================================================================
set -uo pipefail

PY="${PY:-/data1/jeffreytsai/SmartFolio/.venv/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

MARKETS="${MARKETS:-hs300 zz500 nd100 sp500}"
SEEDS="${SEEDS:-0 1 2}"
POLICIES="${POLICIES:-MLP HGAT}"
EPOCHS="${EPOCHS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DEVICE="${DEVICE:-cuda:0}"
LOGDIR="${LOGDIR:-logs}"
PB_VARIANTS="${PB_VARIANTS:-norm norm-nokl mean-reward}"
# PB-IRL preference-learning hyperparameters (main.py defaults)
PB_HORIZON="${PB_HORIZON:-60}"
PB_MARGIN="${PB_MARGIN:-0.5}"
PB_RECENCY="${PB_RECENCY:-1.0}"
PB_KL_COEF="${PB_KL_COEF:-1.0}"
PB_PAIRS="${PB_PAIRS:-256}"

ORCH="run_orch_logs"
mkdir -p "$LOGDIR" "$ORCH" results
FAILLOG="$ORCH/failures.txt"; : > "$FAILLOG"
ts() { date '+%F %T'; }

# per-policy PPO minibatch: HGAT uses 32 (memory; sp500 N=472 OOMs at 128),
# MLP uses the code default (128) by passing 0.
batch_for() { [ "$1" = "HGAT" ] && echo 32 || echo 0; }

# run_job <name> <cmd...> : run, tee to its own log, record failures, continue.
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

# best-val evaluation for a finished run dir (skips cleanly if the run failed).
bestval() {  # <market> <policy> <run_dir>
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

echo "[orch $(ts)] ===== reproduction sweep ====="
echo "[orch $(ts)] markets=[$MARKETS] policies=[$POLICIES] seeds=[$SEEDS] epochs=$EPOCHS device=$DEVICE"
echo "[orch $(ts)] pb_variants=[$PB_VARIANTS] (irl- and gail-prior)"

# ---- 1. data generation -----------------------------------------------------
for m in $MARKETS; do
  if has_data "$m"; then
    echo "[orch $(ts)] data present: $m"
  else
    run_job "build_${m}" "$PY" gen_data/build_dataset.py --market "$m"
  fi
done

# ---- 2. baselines (non-learning table rows) ---------------------------------
for m in $MARKETS; do
  has_data "$m" && run_job "baselines_${m}" "$PY" baselines.py --market "$m"
done

# ---- 3. learned methods (sequential) ----------------------------------------
for m in $MARKETS; do
  if ! has_data "$m"; then
    echo "[orch $(ts)] SKIP market $m (no preprocessed data)"; continue
  fi
  for s in $SEEDS; do
    for p in $POLICIES; do
      bs="$(batch_for "$p")"
      common=(--market "$m" --policy "$p" --seed "$s" --max_epochs "$EPOCHS" \
              --eval_every "$EVAL_EVERY" --device "$DEVICE" --batch_size "$bs")

      # --- stage 1: IRL ---
      irl_dir="$LOGDIR/${m}_${p}_seed${s}"
      run_job "${m}_${p}_irl_s${s}" "$PY" main.py "${common[@]}" --reward irl
      bestval "$m" "$p" "$irl_dir"

      # --- stage 1: GAIL ---
      gail_dir="$LOGDIR/${m}_${p}_gail_seed${s}"
      run_job "${m}_${p}_gail_s${s}" "$PY" main.py "${common[@]}" --reward gail
      bestval "$m" "$p" "$gail_dir"

      # --- stage 2: PB-IRL from each prior, each variant ---
      for v in $PB_VARIANTS; do
        pbargs=(--reward pb_irl --pb_variant "$v" \
                --pb_horizon "$PB_HORIZON" --pb_margin "$PB_MARGIN" \
                --pb_recency "$PB_RECENCY" --pb_kl_coef "$PB_KL_COEF" \
                --pb_pairs "$PB_PAIRS")

        # irl-prior  -> run dir: <m>_<p>_pb_irl_<v>_seed<s>
        run_job "${m}_${p}_pbirl_${v}_irlprior_s${s}" \
          "$PY" main.py "${common[@]}" "${pbargs[@]}" \
          --pb_prior_source irl --prior_run_dir "$irl_dir"
        bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_${v}_seed${s}"

        # gail-prior -> run dir: <m>_<p>_pb_irl_<v>_gailprior_seed<s>
        run_job "${m}_${p}_pbirl_${v}_gailprior_s${s}" \
          "$PY" main.py "${common[@]}" "${pbargs[@]}" \
          --pb_prior_source gail --prior_run_dir "$gail_dir"
        bestval "$m" "$p" "$LOGDIR/${m}_${p}_pb_irl_${v}_gailprior_seed${s}"
      done
    done
  done
done

# ---- 4. aggregate -----------------------------------------------------------
run_job "aggregate" "$PY" aggregate_results.py --logdir "$LOGDIR"

echo "[orch $(ts)] ===== DONE ====="
if [ -s "$FAILLOG" ]; then
  echo "[orch $(ts)] failures/skips ($(wc -l < "$FAILLOG")):"
  cat "$FAILLOG"
else
  echo "[orch $(ts)] no failures"
fi
