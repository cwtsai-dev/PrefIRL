# PB-IRL Integration Design

Date: 2026-06-06
Status: Approved (design), pending implementation
Source method doc: `pb_irl_impl.md`

## Goal

Add Preference-Based IRL (PB-IRL) as a new reward-learning mode in SmartFolio,
alongside the existing `irl` (MaxEnt), `closed_form`, and `gail` modes. The HGAT
policy and the PPO training loop are **not** changed — only the reward signal and
how the reward network is trained.

PB-IRL replaces "directly regress a reward value" (MaxEnt IRL) with "learn which
of two trajectories is better" (Bradley-Terry preference learning), anchored to an
IRL prior via a parameter-space penalty.

## Scope

- Implement `--reward pb_irl` plus the three documented variants and a `base`
  variant, selectable through a single CLI enum.
- Support a **decoupled two-stage** workflow: stage 1 (IRL or GAIL) is an
  ordinary run that now also saves a reward checkpoint; stage 2 (`pb_irl`) loads
  the cached stage-1 checkpoints and trains further.
- Support both **IRL-prior** and **GAIL-prior** for stage 2.
- Prove the pipeline trains end-to-end with short smoke runs via the project
  venv (`/data1/jeffreytsai/SmartFolio/.venv/bin/python`).
- **Out of scope:** full reproduction of the result tables (long GPU runs); the
  user runs those separately.

## Key decisions (resolved during brainstorming)

1. **Decoupled two-stage; stage 2 loads cached checkpoints.** The two stages are
   separate `main.py` invocations:
   - **Stage 1** = an ordinary `--reward irl` or `--reward gail` run, extended to
     save its learned reward net at the val-SR best epoch.
   - **Stage 2** = `--reward pb_irl --prior_run_dir <stage1 run dir>`. It loads
     **both** the stage-1 reward checkpoint (→ φ_prior, used as the trainable
     reward net's initialization *and* the frozen anchor) **and** the stage-1 PPO
     policy (warm-start), then continues training both under the BT loss. No
     internal warmup phase.
2. **Prior source — separate flag.** `--pb_prior_source {irl,gail}` (default
   `irl`), orthogonal to `--pb_variant`. It selects how the stage-1 reward
   checkpoint is parsed (GAIL = extract the discriminator's `body.` params, which
   *are* a `MultiRewardNetwork`) and tags the run name. This keeps the 4-variant
   enum from doubling.
3. **Variant selection — single enum.** `--pb_variant {base,mean-reward,norm,norm-nokl}`,
   mapping 1:1 to the doc tables / run names. Internally decomposed into three
   orthogonal switches (aggregation, normalization, KL).
4. **Preference dataset — offline precompute, fixed-hold equal-weight.** Built
   once at the start of stage 2 and cached; BT loss samples pairs from this fixed
   set.

## Architecture

### A. CLI (`main.py`)

Add `pb_irl` to `--reward` choices. Stage 1 (`irl`/`gail`) needs no new args — it
just gains reward-checkpoint saving. Stage 2 (`pb_irl`) new args:

| arg | default | meaning |
|---|---|---|
| `--prior_run_dir` | (required) | stage-1 run dir holding the reward + policy checkpoints to load |
| `--pb_prior_source` | `irl` | `irl` / `gail` — how to parse the stage-1 reward checkpoint |
| `--prior_reward_net` | (auto) | optional explicit path override for the reward checkpoint |
| `--init_policy` | (auto) | optional explicit path override for the PPO policy checkpoint |
| `--pb_variant` | `norm` | `base` / `mean-reward` / `norm` / `norm-nokl` |
| `--pb_horizon` | `60` | forward window (trading days) for Sharpe |
| `--pb_margin` | `0.5` | minimum Sharpe gap to keep a pair |
| `--pb_recency` | `1.0` | recency weighting strength (0 = uniform) |
| `--pb_kl_coef` | `1.0` | λ_KL, the prior-anchor coefficient |
| `--pb_pairs` | `256` | BT minibatch size (pairs per reward update) |

When `--prior_reward_net` / `--init_policy` are omitted they default to the
standard artifact names inside `--prior_run_dir` (see §D). `run_name` becomes e.g.
`sp500_HGAT_pb_irl_norm_gailprior_seed0`.

Variant → internal switches:

| variant | agg | norm | kl |
|---|---|---|---|
| base | sum | off | on |
| mean-reward | mean | off | on |
| norm | sum | on | on |
| norm-nokl | sum | on | off |

The existing `rew` prefix logic in `main.py` is extended so `pb_irl` runs carry
both the variant and (when not `irl`) the prior-source tag, so logs and result
CSVs line up with the doc's tables.

### B. New module `trainer/pb_irl.py`

Keeps `irl_trainer.py` focused. Three independent units:

1. **`build_preference_dataset(args, train_dataset)`** — runs once, offline.
   - For each train day `t`: expert picks stocks via the existing
     `generate_expert_strategy` (heuristic greedy, Algorithm 1); equal-weight the
     basket.
   - Hold the basket fixed for `pb_horizon` days. The realized daily return on
     day `t+j` is `mean(train_dataset[t+j]['labels'][picked])`. Sharpe =
     `mean(daily) / std(daily)` over the window.
   - Days without a full forward window (near the end of the train period) are
     skipped — no Sharpe label.
   - Form all pairs `(A, B)` where `Sharpe_A > Sharpe_B + pb_margin`. Pair weight
     = recency factor (later day → higher weight), scaled by `pb_recency`.
   - Returns: a trajectory cache `{t: (state, multi_hot_action)}` (state built the
     same way as `generate_expert_trajectories`, honoring `ind_yn/pos_yn/neg_yn`)
     plus the pair list `[(t_better, t_worse, weight)]`.
   - Result cached to disk keyed by market + horizon + margin to avoid recompute.

2. **`RewardNormalizer(nn.Module)`** — wraps the reward net for the env-facing
   reward only.
   - Registered buffers `mu`, `sigma`. `forward(s, a) = (net(s, a) - mu) / sigma`.
   - `update(frozen_net, ref_states, ref_actions)`: called at each epoch end for
     `norm` variants. Computes `mu, sigma` from a **frozen snapshot** of the
     reward net evaluated over a reference batch (preference-trajectory states),
     DQN-target-style, to stop the reward magnitude self-inflating under BT loss.
   - Active from epoch 0 for `norm` variants (init `mu=0, sigma=1` until the first
     `update`); passthrough for non-norm variants.
   - BT loss uses the **raw** net output (z-scoring is monotone, does not change
     pairwise ranking); only the PPO/env reward is normalized.

3. **`PBIRLTrainer`** — `train_step()` performs one reward update:
   - Sample `pb_pairs` pairs from the precomputed set.
   - `R_A = net(τ_A)`, `R_B = net(τ_B)` (scalar trajectory rewards).
   - `bt_loss = weighted BCE( sigmoid(R_A − R_B), 1 )`, weights = recency.
   - `kl = pb_kl_coef * sum ||φ − φ_prior||²` over reward-net parameters. This is
     the parameter-space L2 penalty the doc writes literally as the "KL" anchor
     (not a distributional KL). Dropped entirely when variant = `norm-nokl`.
   - `loss = bt_loss + kl`; Adam step with gradient clipping (reuse the existing
     `grad_clip` convention).
   - Returns stats `{loss, bt_loss, kl, pref_acc}` for TensorBoard.

### C. `MultiRewardNetwork` tweak (`trainer/irl_trainer.py`)

Add an `agg='sum'|'mean'` constructor param controlling stock-axis aggregation of
per-modality encoded rewards. Default `'sum'` keeps existing `irl` / `gail`
behavior byte-for-byte. The `mean-reward` variant passes `'mean'` to prevent long
forward windows from inflating reward magnitude differences.

### D. Decoupled two-stage flow

**Stage 1 (existing `irl` / `gail` modes, one small addition).**
The only change is checkpoint saving. At each val-SR best epoch, in addition to
the existing `best_model` (PPO policy), save the learned reward:
- `irl` → `best_reward_net.pt` = `reward_net.state_dict()` (a `MultiRewardNetwork`).
- `gail` → `best_discriminator.pt` = `discriminator.state_dict()` (a
  `GAILDiscriminator`, whose `body.*` keys are a `MultiRewardNetwork`).

No behavioral change to stage-1 training otherwise.

**Stage 2 (`pb_irl` branch in `train_model_and_predict`).** No warmup phase; one
alternating loop from epoch 0:

1. Resolve checkpoint paths from `--prior_run_dir` (overridable):
   - reward: `best_reward_net.pt` (source `irl`) or `best_discriminator.pt`
     (source `gail`).
   - policy: `best_model.zip`.
2. Build `reward_net = MultiRewardNetwork(agg=<per variant>)`. Initialize its
   weights from the stage-1 reward checkpoint:
   - source `irl`: load directly.
   - source `gail`: remap, loading only the `body.*` sub-state-dict.
3. `φ_prior = deepcopy(reward_net.state_dict())`, frozen — serves as both the
   init (already loaded into `reward_net`) and the anchor target. Skipped as an
   anchor when variant = `norm-nokl` (still used as init).
4. Wrap `reward_net` in `RewardNormalizer` for norm variants; pass the (wrapped)
   net to the env so PPO reads its reward.
5. **Warm-start the PPO policy** from the stage-1 `best_model` (load weights into
   the already-constructed PPO model via `PPO.set_parameters`, keeping the env /
   optimizer config from `main.py`).
6. Build the preference dataset; construct `PBIRLTrainer(reward_net, φ_prior,
   pairs, traj_cache, kl_on=<variant>)`.
7. **Alternating loop (all epochs):** `PBIRLTrainer.train_step` + PPO `learn`. At
   each epoch end, norm variants call `normalizer.update` from a frozen snapshot
   of the current reward net.
8. Save `best_reward_net.pt` (val-SR best) next to `best_model`, so a stage-2 run
   can itself seed a further stage.

Periodic eval, val-SR checkpointing, and logging are unchanged. Reward-net beta
logging keeps working since the backbone is still a `MultiRewardNetwork`.

## Data flow

```
STAGE 1 (irl | gail):  PPO + reward learning ─> save best_model.zip
                                              └> save best_reward_net.pt / best_discriminator.pt
                                                                 │ (φ_prior + warm-start policy)
                                                                 v
STAGE 2 (pb_irl):  load checkpoints ── init reward_net = φ_prior (frozen copy = anchor)
                                    └─ PPO.set_parameters(best_model)

train_dataset ─┬─> generate_expert_strategy ─> per-day basket ─> fixed-hold 60d Sharpe
               │                                                        │
               │                                              margin + recency
               │                                                        v
               └─> trajectory cache {t:(state,action)} <──── preference pairs [(A,B,w)]
                                   │                                    │
                                   v                                    v
                          PBIRLTrainer.train_step:  BT loss(σ(R_A−R_B)) + λ‖φ−φ_prior‖²
                                   │
              reward_net ─(optional RewardNormalizer z-score)─> env reward ─> PPO.learn
```

## Error handling / edge cases

- **Too few pairs** (tiny train period or large margin): if the pair set is empty
  after filtering, raise a clear error suggesting a smaller `--pb_margin` or
  `--pb_horizon`.
- **σ collapse** in `RewardNormalizer`: clamp `sigma` to a small floor (e.g.
  `1e-6`) to avoid divide-by-zero.
- **Forward window overrun:** days within `pb_horizon` of the train-period end are
  skipped, not zero-padded.
- **Variant ↔ policy:** HGAT forces `ind_yn=pos_yn=neg_yn=True` (existing
  behavior); the reward net input dims follow the same rule as `irl`.
- **Checkpoint mismatch:** the stage-1 reward checkpoint must match the stage-2
  reward-net architecture (same market `num_stocks` and `ind/pos/neg` modality
  flags). On a `load_state_dict` shape error, raise a clear message telling the
  user the stage-1 and stage-2 configs disagree. Missing `--prior_run_dir` or
  absent checkpoint files → explicit error naming the expected path.

## Testing

Smoke runs via the project venv on the smallest market that has preprocessed data
(tw50 if present, else nd100). Two stages, exercising both prior sources:

```
PY=/data1/jeffreytsai/SmartFolio/.venv/bin/python
M=<smallest>

# Stage 1a: IRL prior (must save best_reward_net.pt)
$PY main.py --reward irl  --market $M --policy MLP --max_epochs 3 --ppo_steps 256 --num_expert 200 --tag smoke
# Stage 1b: GAIL prior (must save best_discriminator.pt)
$PY main.py --reward gail --market $M --policy MLP --max_epochs 3 --ppo_steps 256 --num_expert 200 --tag smoke

# Stage 2: PB-IRL from IRL prior
$PY main.py --reward pb_irl --pb_variant norm --pb_prior_source irl \
    --prior_run_dir logs/${M}_MLP_smoke_seed123 \
    --market $M --policy MLP --max_epochs 3 --ppo_steps 256 --pb_pairs 64

# Stage 2: PB-IRL from GAIL prior + no-KL path
$PY main.py --reward pb_irl --pb_variant norm-nokl --pb_prior_source gail \
    --prior_run_dir logs/${M}_MLP_gail_smoke_seed123 \
    --market $M --policy MLP --max_epochs 3 --ppo_steps 256 --pb_pairs 64
```

Pass criteria: every process exits 0; stage-1 reward checkpoints written; stage-2
loads them + warm-starts the policy (logged); BT loss / reward scalars appear in
TensorBoard; stage-2 `best_reward_net.pt`, `best_model`, `test_metrics.csv`
written. (Exact run-dir names confirmed during implementation against `main.py`'s
`run_name` logic.) No full reproduction of result tables.

## Files touched

- `main.py` — new stage-2 args (`--prior_run_dir`, `--pb_prior_source`,
  `--prior_reward_net`, `--init_policy`, `--pb_*`), `pb_irl` choice, run_name
  variant + prior-source tag.
- `trainer/irl_trainer.py` — stage-1 reward-checkpoint saving for `irl`
  (`best_reward_net.pt`) and `gail` (`best_discriminator.pt`); `pb_irl` branch in
  `train_model_and_predict` (load checkpoints, warm-start policy, alternating
  PB-IRL loop); `agg` param on `MultiRewardNetwork`.
- `trainer/pb_irl.py` — **new**: `build_preference_dataset`, `RewardNormalizer`,
  `PBIRLTrainer`, and the stage-1 checkpoint loaders (irl / gail-body remap).
