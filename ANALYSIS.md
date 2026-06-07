# SmartFolio — Codebase Analysis & Paper-Conformance Report

Paper: **Enhancing Portfolio Optimization via Heuristic-Guided Inverse
Reinforcement Learning with Multi-Objective Reward and Graph-based Policy
Learning** — IJCAI-25, paper #1054 (Zhang et al., ECNU / Tongji).

This document re-analyses the implementation against the paper's method and
records every bug fixed to make the code both runnable and faithful.

---

## 1. Paper method (three components)

1. **Greedy Expert Strategy (Algorithm 1).** Heuristic rules generate
   expert trajectories: rank stocks by return, select greedily subject to a
   per-industry cap `K' = ⌊αK⌋` and an average-correlation cap `γ`. Output is
   a binary action vector `a_t ∈ {0,1}^N`.

2. **Multi-Objective Reward + MaxEnt IRL.** Four reward objectives (return,
   sector diversification, positive-correlation penalty, negative-correlation
   incentive). A reward network `R_θ(s,a) = Σ_{k∈K} β_k · f_enc^k(φ_k(s) ⊕ a)`
   with `K = {base, ind, pos, neg}` and softmax-normalised `β_k` is learned by
   Maximum-Entropy IRL with loss `L(θ) = −E_{π_E}[R_θ] + log E_{π_A}[exp R_θ]`,
   updated with gradient clipping.

3. **HGAT Graph Policy (§3.4).** Heterogeneous graph attention network:
   multi-head GAT over the industry / positive-corr / negative-corr graphs,
   heterogeneous fusion attention `H_fusion = Σ β_k H_k`, and a policy head.
   Optimised with PPO. Training alternates IRL and PPO for 200 epochs.

---

## 2. Conformance check (component by component)

| Paper element | Code location | Verdict |
|---|---|---|
| Greedy expert (Algorithm 1) | `gen_data/generate_expert.py` | ✓ faithful — rank by return, `K'=⌊αK⌋` with α=0.3, corr cap γ=0.5 |
| Multi-objective reward net `R_θ=Σβ_k f_enc^k` | `trainer/irl_trainer.py: MultiRewardNetwork` | ✓ faithful — 4 modality encoders {base,ind,pos,neg}, β = softmax |
| MaxEnt IRL loss | `trainer/irl_trainer.py: MaxEntIRL` | ✓ `L = −E_E[R] + log E_A[exp R]`, with gradient clipping |
| HGAT multi-head GAT + fusion | `model/model.py: HGAT/MHGraphAttn/HeteFusionAttn` | ✓ after fixes |
| HGAT policy under PPO | `policy/policy.py: HGATActorCriticPolicy` | ✓ after fixes |
| Alternating IRL ↔ PPO loop | `trainer/irl_trainer.py: train_model_and_predict` | ✓ after fix (was broken) |
| 6 features, 20-day window, k-means norm, monthly Pearson corr, ρ-threshold 0.2 | `gen_data/train_predict_data.py`, `gen_data/build_dataset.py` | ✓ faithful |
| Hyperparams: lr 1e-4, batch 128, hidden 128, 8 heads | `PPO_PARAMS`, `HGAT` | ✓ |
| 6 metrics ARR/AVol/MDD/SR/CR/IR | `env/portfolio_env.py: evaluate` | ✓ |

### Notes on intentional design choices (not bugs)
- **Reward objectives are learned, not hand-coded.** §3.2 gives closed-form
  `R_diversity / R_pos / R_neg`. The implementation does not compute these
  formulas directly; instead the IRL reward network learns them implicitly
  from the heuristic expert via its four `{base,ind,pos,neg}` modalities.
  This matches Figure 2, where the reward function is *learned* by gradient
  descent on the entropy loss — consistent with the IRL philosophy.
- **Discrete top-k action.** The env uses a `MultiDiscrete` top-k stock
  selection with equal weighting, rather than the continuous Softmax weights
  `w_t` of §3.4. This is consistent with the binary expert action
  `a_t ∈ {0,1}^N` and is applied identically to the HGAT and MLP policies, so
  the comparison remains fair.
- **PPO `gamma = 0.5`.** The paper objective is an undiscounted reward sum;
  the codebase ships `gamma = 0.5`. Kept as the codebase's value.
- **IR benchmark.** No market-index file is shipped, so the Information Ratio
  is computed against an equal-weight market proxy (mean return of all
  constituents). Documented; affects only IR.

---

## 3. Bugs found and fixed

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `main.py`, `generate_expert.py` | data path `dataset_default/` does not exist; `industry.npy` does not exist | use `dataset/`; load industry matrix from each sample's `industry_matrix` |
| 2 | `trainer/irl_trainer.py: process_data` | reads `data_dict['pyg_data']` — key absent from the shipped `.pkl` → `KeyError` | removed `pyg_data` (it was unused everywhere) |
| 3 | `trainer/irl_trainer.py: model_predict` | reads `../dataset/index_data/..._index_2024.csv` — file absent → `FileNotFoundError` | replaced with `evaluate_on_loader` + equal-weight benchmark |
| 4 | `model/model.py:158` | `pos_support = self.pos_mlp(neg_support)` — copy-paste bug, positive branch consumed the negative GAT output | `self.pos_mlp(pos_support)` |
| 5 | `trainer/irl_trainer.py` | `return trained_model` indented inside the epoch loop → the 200-epoch alternation ran **once** | rewrote as a clean alternating IRL↔PPO loop |
| 6 | `model/model.py`, `policy/policy.py` | HGAT `num_stocks`/`n_features` mis-computed; obs reshaped with `[M,N]` layout while the env emits `[N,M]` → dimension mismatch, HGAT could not run | fixed dimension derivation and obs parsing `[B, N, d+3N]` |
| 7 | `env/portfolio_env.py` | action/return off-by-one: action at obs[t] earned `ror[t+1]`; last day dropped | reward now settles on `ror[t]` before advancing time |
| 8 | `main.py`, trainers | `collate_fn=lambda x:x` made train batches lists while `process_data` expects a dict; `torch_geometric.data.DataLoader` removed in modern PyG | use `torch.utils.data.DataLoader` with default collate |
| 9 | `trainer/irl_trainer.py: evaluate_on_loader` | `DummyVecEnv` auto-resets on the terminal step, wiping the episode stats before `evaluate()` | step the raw env directly during evaluation |
| 10 | `dataloader/data_loader.py: date_to_idx` | returns `None` when a split boundary is not an exact trading day → slice crash; naive nearest-day also leaks the next split | start boundary → next trading day, end boundary → previous trading day (no train/val/test leakage) |

## 4. Additions for this reproduction
- `pyproject.toml` — uv-managed environment.
- TensorBoard logging: IRL loss, expert/agent reward, adaptive `β_k`,
  validation & test metrics per epoch.
- `run_experiments.py` — multi-GPU runner (independent runs across GPUs).
- `gen_data/build_dataset.py` — parameterised dataset builder (corr + pkl).
- `gen_data/fetch_sectors.py`, `gen_data/fetch_taiwan.py` — source industry /
  price data for markets without preprocessed data.
- `aggregate_results.py` — collate metrics across runs.
