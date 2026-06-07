# SmartFolio — IJCAI-25 reproduction & reward-method study

A paper-faithful reproduction of **"Enhancing Portfolio Optimization via
Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective Reward
and Graph-based Policy Learning"** ([IJCAI-25 #1054][paper], Zhang et al.,
ECNU / Tongji), with bug fixes, an extended baseline suite, an alternative
reward-method study, and a Taiwan-market extension.

[paper]: https://www.ijcai.org/proceedings/2025/1054.pdf

The original method is **MaxEnt IRL** on a heuristic-greedy expert (paper
Algorithm 1) feeding a **PPO + HGAT** policy. This fork keeps that pipeline
paper-faithful and adds two alternatives that distill the same expert
signal differently (paper §3.2 closed-form rewards, and GAIL).

## What's in here vs. the original

| | Original | This fork |
|---|---|---|
| Runnable end-to-end? | ❌ (11 latent bugs, see `ANALYSIS.md`) | ✅ |
| HGAT graph policy | dimension mismatch, copy-paste bug | fixed; **2.5× faster** sparse→dense softmax |
| Reward signal | MaxEnt IRL (paper §3.3) | + paper §3.2 analytical reward, + GAIL discriminator |
| Markets | CSI 300 only (preprocessed shipped) | + CSI 500, NASDAQ 100, S&P 500, **TW 50** (Taiwan extension) |
| Baselines | none | 1/N, Buy & Hold, Random top-k, Momentum top-k, **greedy expert (oracle / causal)** |
| Reports | none | `RESULTS_sp500.md`, PnL plot, per-run tensorboard |
| Environment | unspecified | `uv` (`pyproject.toml` + `uv.lock`) |
| Code/data integrity | several leak / off-by-one bugs | causal env timing, leak-free split boundaries |

Full bug-by-bug audit lives in **[`ANALYSIS.md`](ANALYSIS.md)**.

## Quick start

```bash
# 0) clone + restore environment (Python 3.10, torch cu121)
uv sync

# 1) (optional) build a market's preprocessed pkl files from raw OHLCV
python gen_data/build_dataset.py --market sp500

# 2) train + eval one (market, policy, reward) configuration
python main.py --market sp500 --policy HGAT --reward irl --max_epochs 200

# 3) compute baselines on the same test set
python baselines.py --market sp500

# 4) benchmark the greedy expert directly
python benchmark_expert.py --market sp500

# 5) compile the PnL plot + metrics table + RESULTS_sp500.md
python make_report.py --market sp500
```

## Entry points

| Script | Purpose |
|---|---|
| **`main.py`** | The training+eval driver — one (market, policy, reward, seed) → tensorboard logs + final/best-val test metrics. |
| `baselines.py` | Compute classic non-learning baselines (1/N, Buy&Hold, Random-topk, Momentum-topk) on the same test set. |
| `benchmark_expert.py` | Run the paper's greedy expert (Algorithm 1) directly — oracle (uses next-day labels, what training imitates) and causal (uses yesterday's realised return) variants. |
| `eval_best.py` | Re-evaluate a run's saved `best_model.zip` (best validation Sharpe) on test. |
| `make_report.py` | Cumulative-PnL plot + combined metrics table + `RESULTS_<market>.md`. |
| `aggregate_results.py` | Cross-run summary (mean ± std over seeds). |
| `run_experiments.py` | Multi-GPU launcher for sweeps (markets × policies × seeds). |
| `gen_data/build_dataset.py` | Build a market's preprocessed pkl files (corr + per-day samples) from raw OHLCV. |
| `gen_data/fetch_sectors.py` | Source GICS sectors for NASDAQ-100 / S&P 500 (Wikipedia + yfinance fallback). |
| `gen_data/fetch_taiwan.py` | Source 7 yrs of OHLCV + sectors for TW 50 via yfinance. |
| `gen_data/generate_expert.py` | The paper's greedy Algorithm 1, used both at train time and by `benchmark_expert.py`. |
| `finalize_sweep.sh` | Auto-pipeline: wait for runs → eval best-val → regenerate report. |

## CLI knobs (main.py)

The full list is in `main.py:get_args`; key ones:

```bash
--market {hs300, zz500, nd100, sp500, tw50}        # which market
--policy {MLP, HGAT}                                # policy net
--reward {irl, closed_form, gail}                   # reward signal source ← study variable
--seed 0
--max_epochs 200                                    # paper §4.1
--eval_every 20
--batch_size 0                                      # 0 = paper default 128; sp500-HGAT needs 32
--n_steps 0                                         # 0 = default 2048
--lambda_return 1.0  --lambda_div 0.1               # paper §3.2 weights (closed_form only)
--lambda_pos 0.1     --lambda_neg 0.1
--m_threshold 0.0
```

### Reward modes

- **`irl`** — the paper's MaxEnt IRL (§3.3). Learns a reward network from
  the greedy expert via the entropy loss `−E_E[R] + log E_A[exp R]`, with
  gradient clipping. Trains a `MultiRewardNetwork` with 4 modality encoders
  `{base, ind, pos, neg}` (paper §3.3 equation).
- **`closed_form`** — the paper's §3.2 analytical reward, computed in the
  env every step: `λ₁·log(1+r) + λ₂·H(sector_weights) + λ₃·R_pos + λ₄·R_neg`.
  No learned reward network, no IRL alternation — PPO trains directly.
- **`gail`** — Adversarial imitation. A discriminator (same architecture
  as the IRL reward net) is trained with BCE on expert vs. agent
  trajectories; the env reward is `softplus(logits)`, which is bounded
  below by 0 (no entropy-loss magnitude drift).

All three are evaluated identically (env uses **realised return** during
test, regardless of the training reward) — so test metrics are directly
comparable across methods.

## Where artifacts land

```
logs/<market>_<policy>[_<reward>]_seed<seed>/
    events.out.tfevents.*       # TensorBoard
    best_model.zip               # checkpoint with the best val/SR
    final_model.zip              # end-of-training checkpoint
    test_metrics.csv             # final-epoch test metrics
    test_net_value.csv           # cumulative-wealth curve, final
    test_metrics_bestval.csv     # best-val-checkpoint test metrics (after eval_best.py)
    test_net_value_bestval.csv

results/
    pnl_curve_<market>.png       # PnL plot (cumulative wealth vs. day)
    all_metrics_<market>.csv     # combined table: agent + baselines + expert
    baselines_metrics_<market>.csv
    baselines_curves_<market>.csv
    expert_metrics_<market>.csv
    expert_curves_<market>.csv

RESULTS_<market>.md             # final analytical write-up
```

## Reading the current results

- **`ANALYSIS.md`** — paper-vs-code conformance audit, lists every bug
  fixed, what was added, and known caveats.
- **`RESULTS_sp500.md`** — auto-generated comparison of SmartFolio
  (HGAT / MLP × IRL / closed_form / GAIL) against the baselines and the
  greedy expert on the S&P 500 2024 test.

Headline so far (IRL only — closed_form and GAIL still training):
**SmartFolio-HGAT-IRL (final) reaches SR 1.51** on S&P 500 2024 vs. 1/N at
SR 1.16. The MLP ablation only matches 1/N — the HGAT graph policy is the
active ingredient. The greedy expert with no look-ahead loses 7.9 % (and
the oracle expert with look-ahead achieves a physically-impossible SR 40),
which shows the agent is not just imitating the expert.

## Original codebase

Original implementation by Wenyi Zhang et al. (paper authors):
<https://github.com/ChloeWenyiZhang/SmartFolio>

This fork keeps the original algorithmic intent intact; the diff is
predominantly bug fixes, infrastructure, and the additional reward methods
listed above (see `ANALYSIS.md` for the line-by-line audit).
