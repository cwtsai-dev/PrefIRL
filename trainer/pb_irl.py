"""Preference-Based IRL (PB-IRL): decoupled stage-2 reward learning.

Stage 1 (irl/gail, run separately) saves a reward checkpoint. Stage 2 loads it
as phi_prior, builds a preference dataset from forward-Sharpe rankings of the
heuristic expert, and trains the reward net with a Bradley-Terry loss anchored
to phi_prior. See docs/superpowers/specs/2026-06-06-pb-irl-design.md.
"""
import os
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_data.generate_expert import generate_expert_strategy
from trainer.irl_trainer import MultiRewardNetwork


# --------------------------------------------------------------------------
# Variant decode
# --------------------------------------------------------------------------
_VARIANTS = {
    # variant      : (agg,   norm,  kl)
    "base":         ("sum",  False, True),
    "mean-reward":  ("mean", False, True),
    "norm":         ("sum",  True,  True),
    "norm-nokl":    ("sum",  True,  False),
}


def decode_variant(variant):
    """Map a --pb_variant enum to (agg, use_norm, kl_on) switches."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown pb_variant {variant!r}; choose from {list(_VARIANTS)}")
    return _VARIANTS[variant]


# --------------------------------------------------------------------------
# Reward normalizer (env-facing z-score, DQN-target style)
# --------------------------------------------------------------------------
class RewardNormalizer(nn.Module):
    """Wrap a reward net; z-score its output with stats from a frozen snapshot.

    Drop-in for the env: forward(state, action) returns (R - mu) / sigma. mu/sigma
    are refreshed once per epoch via ``update`` from a frozen copy of the net so
    BT-loss reward magnitudes cannot self-inflate.
    """

    def __init__(self, reward_net):
        super().__init__()
        self.net = reward_net
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def forward(self, state, action):
        return (self.net(state, action) - self.mu) / self.sigma

    @torch.no_grad()
    def update(self, ref_trajectories, device):
        """Recompute mu/sigma from a frozen snapshot over reference trajectories."""
        frozen = copy.deepcopy(self.net).eval()
        rs = []
        for state, action in ref_trajectories:
            s = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
            a = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=device)
            rs.append(frozen(s, a).reshape(-1))
        r = torch.cat(rs)
        std = r.std()  # unbiased; NaN if <2 samples -> guard before clamp (no-op on NaN)
        if not torch.isfinite(std):
            std = torch.ones_like(std)
        self.mu.copy_(r.mean())
        self.sigma.copy_(std.clamp_min(1e-6))


# --------------------------------------------------------------------------
# Preference dataset (Step 1 of the method doc)
# --------------------------------------------------------------------------
def portfolio_forward_sharpe(selected, forward_labels):
    """Sharpe of an equal-weight, fixed basket held over a forward window.

    selected: 1D array of stock indices.
    forward_labels: list of per-day label vectors for the next H days.
    Returns 0.0 when the realized daily series has ~zero volatility.
    """
    daily = np.array([labels[selected].mean() for labels in forward_labels],
                     dtype=np.float64)
    std = daily.std()
    if std < 1e-12:
        return 0.0
    return float(daily.mean() / std)


def build_preference_pairs(sharpes, days, margin, recency):
    """All (better_day, worse_day, weight) pairs with a Sharpe gap > margin.

    weight = 1 + recency * (mean(day pair) / max_day); recency=0 -> uniform 1.0.
    """
    pairs = []
    n = len(days)
    max_day = max(days) if days else 1
    for i in range(n):
        for j in range(n):
            if sharpes[i] > sharpes[j] + margin:
                rec = ((days[i] + days[j]) / 2.0) / max(1, max_day)
                pairs.append((days[i], days[j], float(1.0 + recency * rec)))
    return pairs


def build_preference_dataset(args, train_dataset, cache_dir="dataset/pref_cache"):
    """Build (or load cached) the preference dataset for stage 2.

    Returns {"traj_cache": {day: (state, multi_hot_action)},
             "pairs": [(better_day, worse_day, weight)]}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = (f"{args.market}_h{args.pb_horizon}_m{args.pb_margin}"
           f"_r{args.pb_recency}_{args.train_start}_{args.train_end}"
           f"_i{int(args.ind_yn)}{int(args.pos_yn)}{int(args.neg_yn)}")
    cache_path = os.path.join(cache_dir, key + ".pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n_days = len(train_dataset)
    traj_cache, sharpes, days = {}, [], []
    for t in range(n_days - args.pb_horizon):
        data = train_dataset[t]
        returns = data["labels"].numpy()
        ind_matrix = data["industry_matrix"].numpy()
        corr = data["corr"].numpy()
        action = generate_expert_strategy(
            returns=returns, industry_relation_matrix=ind_matrix,
            correlation_matrix=corr)
        selected = np.where(action == 1)[0]
        if selected.size == 0:
            continue
        forward_labels = [train_dataset[t + j]["labels"].numpy()
                          for j in range(1, args.pb_horizon + 1)]
        sr = portfolio_forward_sharpe(selected, forward_labels)

        # state built exactly like generate_expert_trajectories
        state = data["features"].numpy().squeeze()
        if args.ind_yn:
            state = np.concatenate([state, ind_matrix], axis=1)
        if args.pos_yn:
            state = np.concatenate([state, data["pos_matrix"].numpy()], axis=1)
        if args.neg_yn:
            state = np.concatenate([state, data["neg_matrix"].numpy()], axis=1)

        traj_cache[t] = (state.astype(np.float32), action.astype(np.float32))
        sharpes.append(sr)
        days.append(t)

    pairs = build_preference_pairs(sharpes, days, args.pb_margin, args.pb_recency)
    if not pairs:
        raise RuntimeError(
            f"no preference pairs after margin filter (pb_margin={args.pb_margin}); "
            f"try a smaller --pb_margin or --pb_horizon")
    print(f"[pb_irl] preference dataset: {len(traj_cache)} days, {len(pairs)} pairs")
    dataset = {"traj_cache": traj_cache, "pairs": pairs}
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


# --------------------------------------------------------------------------
# Stage-1 checkpoint loading
# --------------------------------------------------------------------------
def strip_body_prefix(state_dict):
    """Keep only a GAILDiscriminator's '.body' params (themselves a MultiRewardNetwork)."""
    return {k[len("body."):]: v for k, v in state_dict.items()
            if k.startswith("body.")}


def _default_reward_path(args):
    fname = "best_discriminator.pt" if args.pb_prior_source == "gail" else "best_reward_net.pt"
    return os.path.join(args.prior_run_dir, fname)


def default_policy_path(args):
    """Stage-1 PPO checkpoint to warm-start from (sb3 appends '.zip')."""
    return args.init_policy or os.path.join(args.prior_run_dir, "best_model")


def load_prior_reward_net(args):
    """Build a MultiRewardNetwork and load stage-1 weights into it (= phi_prior).

    --pb_prior_source selects the checkpoint format:
      irl  -> best_reward_net.pt   (a MultiRewardNetwork state_dict)
      gail -> best_discriminator.pt (a GAILDiscriminator; keep body.* only)
    """
    agg, _, _ = decode_variant(args.pb_variant)
    net = MultiRewardNetwork(
        input_dim=args.input_dim, num_stocks=args.num_stocks,
        ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn, agg=agg,
    ).to(args.device)

    path = args.prior_reward_net or _default_reward_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prior reward checkpoint not found: {path} "
            f"(set --prior_run_dir or --prior_reward_net)")
    sd = torch.load(path, map_location=args.device)
    if args.pb_prior_source == "gail":
        sd = strip_body_prefix(sd)
    try:
        net.load_state_dict(sd)
    except RuntimeError as e:
        raise RuntimeError(
            f"prior checkpoint {path} does not match the stage-2 reward net "
            f"(market/num_stocks/modality flags must agree): {e}")
    return net


# --------------------------------------------------------------------------
# Preference-based reward learner
# --------------------------------------------------------------------------
class PBIRLTrainer:
    """Bradley-Terry reward learner anchored to a frozen prior (phi_prior).

    P(A > B) = sigmoid(R(tau_A) - R(tau_B)); every stored pair has A as the
    higher-Sharpe day, so the BT target is always 1. The anchor is the doc's
    parameter-space L2 penalty ||phi - phi_prior||^2 (called "KL" there).

    The signature of ``train_step`` mirrors MaxEntIRL/GAILTrainer so the existing
    alternating loop in train_model_and_predict can call it unchanged; the
    agent_envs / model / batch_size arguments are accepted and ignored.
    """

    def __init__(self, reward_net, pref_dataset, kl_coef=1.0, kl_on=True,
                 lr=1e-4, grad_clip=1.0, n_pairs=256):
        self.reward_net = reward_net
        self.traj_cache = pref_dataset["traj_cache"]
        self.pairs = pref_dataset["pairs"]
        self.kl_coef = kl_coef
        self.kl_on = kl_on
        self.n_pairs = n_pairs
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)
        # frozen prior snapshot for the L2 anchor
        self.prior_params = [p.detach().clone() for p in reward_net.parameters()]

    def _reward(self, day, device):
        state, action = self.traj_cache[day]
        s = torch.as_tensor(state, dtype=torch.float32, device=device)
        a = torch.as_tensor(action, dtype=torch.float32, device=device)
        return self.reward_net(s, a).reshape(())

    def train_step(self, agent_envs=None, model=None, batch_size=None,
                   device="cuda:0"):
        n = min(self.n_pairs, len(self.pairs))
        idx = np.random.choice(len(self.pairs), size=n, replace=False)
        batch = [self.pairs[i] for i in idx]

        # one forward per unique day in the minibatch
        unique_days = {d for a, b, _ in batch for d in (a, b)}
        rcache = {d: self._reward(d, device) for d in unique_days}

        diffs = torch.stack([rcache[a] - rcache[b] for a, b, _ in batch])
        weights = torch.tensor([w for _, _, w in batch],
                               dtype=torch.float32, device=device)
        bt = F.binary_cross_entropy_with_logits(
            diffs, torch.ones_like(diffs), weight=weights)

        kl = torch.zeros((), device=device)
        if self.kl_on:
            for p, p0 in zip(self.reward_net.parameters(), self.prior_params):
                kl = kl + ((p - p0.to(device)) ** 2).sum()
            kl = self.kl_coef * kl
        loss = bt + kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), self.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            acc = (diffs > 0).float().mean().item()
        return {"loss": loss.item(), "bt_loss": bt.item(),
                "kl": float(kl.item()), "pref_acc": acc}
