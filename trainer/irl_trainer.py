"""Heuristic-guided Maximum-Entropy IRL trainer.

Implements the alternating optimisation of the IJCAI-25 paper:
  - a multi-objective reward network R_theta is learned by MaxEnt IRL so that
    the heuristic greedy expert (Algorithm 1) is preferred over the agent;
  - the PPO actor-critic policy is optimised under the current learned reward.
The two steps alternate for ``max_epochs`` epochs.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env.portfolio_env import StockPortfolioEnv


# --------------------------------------------------------------------------
# Reward networks
# --------------------------------------------------------------------------
class RewardNetwork(nn.Module):
    """Plain single-stream reward network (used when multi_reward is off)."""

    def __init__(self, input_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state = state.squeeze()
        action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


class MultiRewardNetwork(nn.Module):
    """Multi-objective reward network (paper section 3.3).

    R_theta(s, a) = sum_k beta_k * f_enc^k( phi_k(s) (+) a ),  k in {base, ind, pos, neg}
    beta_k are learnable weights normalised by softmax.
    """

    def __init__(self, input_dim, num_stocks, hidden_dim=64,
                 ind_yn=False, pos_yn=False, neg_yn=False, agg="sum"):
        super().__init__()
        assert agg in ("sum", "mean"), f"agg must be sum|mean, got {agg!r}"
        self.agg = agg
        self.feature_dims = {
            'base': input_dim,
            'ind': num_stocks if ind_yn else 0,
            'pos': num_stocks if pos_yn else 0,
            'neg': num_stocks if neg_yn else 0,
        }
        # one encoder per active modality
        self.encoders = nn.ModuleDict()
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                self.encoders[feat] = nn.Sequential(
                    nn.Linear(dim + 1, hidden_dim),  # +1 for the action channel
                    nn.ReLU(),
                )
        active_feats = [k for k, v in self.feature_dims.items() if v > 0]
        self.num_rewards = len(active_feats)
        # adaptive reward weights beta_k
        self.weights = nn.Parameter(torch.ones(self.num_rewards))

    def beta(self):
        return F.softmax(self.weights, dim=0)

    def forward(self, state, action):
        # split the observation into the per-modality feature blocks
        ptr = 0
        features = {}
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                features[feat] = state[..., ptr:ptr + dim]
                ptr += dim

        rewards = []
        for feat, data in features.items():
            action_exp = action.unsqueeze(-1)                       # [N, 1]
            fused = torch.cat([data.squeeze(), action_exp], dim=-1)  # [N, dim+1]
            encoded = self.encoders[feat](fused).mean(dim=1)         # [N]
            if self.agg == "mean":
                rewards.append(encoded.mean(dim=-1, keepdim=True))   # [1]
            else:
                rewards.append(encoded.sum(dim=-1, keepdim=True))    # [1]

        beta = self.beta()
        weighted = sum(w * r for w, r in zip(beta, rewards))
        return weighted


# --------------------------------------------------------------------------
# GAIL discriminator (bounded-reward alternative to MaxEnt IRL)
# --------------------------------------------------------------------------
class GAILDiscriminator(nn.Module):
    """Per-stock discriminator with permutation-invariant aggregation.

    Re-uses the MultiRewardNetwork backbone (4 modality encoders {base, ind,
    pos, neg}) but interprets its scalar output as a logit. The env reads
    ``forward(s, a)`` and gets a *bounded* reward via softplus(logits), so
    the entropy-loss magnitude drift of MaxEnt IRL cannot happen.
    """

    def __init__(self, input_dim, num_stocks, hidden_dim=64,
                 ind_yn=False, pos_yn=False, neg_yn=False):
        super().__init__()
        self.body = MultiRewardNetwork(input_dim, num_stocks, hidden_dim,
                                       ind_yn=ind_yn, pos_yn=pos_yn,
                                       neg_yn=neg_yn)

    def logits(self, state, action):
        return self.body(state, action)

    def forward(self, state, action):
        # env-facing reward: r = softplus(logits) = -log(1 - sigmoid(logits))
        # bounded below by 0; bounded above in practice when D stays calibrated.
        return F.softplus(self.logits(state, action))


class GAILTrainer:
    """Adversarial imitation: discriminator distinguishes expert vs agent.

    Same alternation skeleton as MaxEntIRL — sample agent rollouts, sample
    an expert minibatch, do one BCE update on the discriminator. The agent
    reward is the discriminator's softplus output (env-side, see
    ``GAILDiscriminator.forward``).
    """

    def __init__(self, discriminator, expert_data, lr=1e-4, grad_clip=1.0):
        self.D = discriminator
        self.expert_data = expert_data
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        self.grad_clip = grad_clip

    def _calculate_logits(self, trajectories, device):
        logits = []
        for state, action in trajectories:
            s = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
            a = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=device)
            logits.append(self.D.logits(s, a).reshape(-1))
        return torch.cat(logits)

    def _generate_agent_trajectories(self, env, model, n_steps):
        trajectories = []
        obs = env.reset()
        num_stocks = obs.shape[-2]
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            action_multi_hot = np.zeros(num_stocks, dtype=np.float32)
            act = np.atleast_2d(action)
            for k in range(act.shape[-1]):
                action_multi_hot[int(act[0, k])] = 1.0
            trajectories.append((obs[0].copy(), action_multi_hot))
            obs = next_obs
            if np.any(done):
                obs = env.reset()
        return trajectories

    def train_step(self, agent_envs, model, batch_size=64, device="cuda:0"):
        agent_traj = []
        per_env = max(1, batch_size // len(agent_envs))
        for env in agent_envs:
            agent_traj += self._generate_agent_trajectories(env, model, per_env)
        n_expert = min(batch_size, len(self.expert_data))
        idx = np.random.choice(len(self.expert_data), size=n_expert, replace=False)
        expert_batch = [self.expert_data[i] for i in idx]

        expert_logits = self._calculate_logits(expert_batch, device)
        agent_logits = self._calculate_logits(agent_traj, device)
        loss = (F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
                + F.binary_cross_entropy_with_logits(agent_logits, torch.zeros_like(agent_logits)))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            acc_e = (torch.sigmoid(expert_logits) > 0.5).float().mean().item()
            acc_a = (torch.sigmoid(agent_logits) < 0.5).float().mean().item()
        return {"loss": loss.item(), "acc_expert": acc_e, "acc_agent": acc_a,
                "expert_logit_mean": expert_logits.mean().item(),
                "agent_logit_mean": agent_logits.mean().item()}


# --------------------------------------------------------------------------
# Maximum-Entropy IRL
# --------------------------------------------------------------------------
class MaxEntIRL:
    """MaxEnt IRL reward learner.

    Loss (paper section 3.3):
        L(theta) = - E_{pi_E}[R_theta] + log E_{pi_A}[exp(R_theta)]
    The reward-network parameters are updated with gradient clipping.
    """

    def __init__(self, reward_net, expert_data, lr=1e-4, grad_clip=1.0):
        self.reward_net = reward_net
        self.expert_data = expert_data
        self.optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)
        self.grad_clip = grad_clip

    def _calculate_rewards(self, trajectories, device):
        rewards = []
        for state, action in trajectories:
            state_tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=device)
            rewards.append(self.reward_net(state_tensor, action_tensor).reshape(-1))
        return torch.cat(rewards)

    def _generate_agent_trajectories(self, env, model, n_steps):
        """Roll out the current policy to collect (state, multi-hot action) pairs."""
        trajectories = []
        obs = env.reset()
        num_stocks = obs.shape[-2]
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            action_multi_hot = np.zeros(num_stocks, dtype=np.float32)
            act = np.atleast_2d(action)
            for k in range(act.shape[-1]):
                action_multi_hot[int(act[0, k])] = 1.0
            trajectories.append((obs[0].copy(), action_multi_hot))
            obs = next_obs
            if np.any(done):
                obs = env.reset()
        return trajectories

    def train_step(self, agent_envs, model, batch_size=64, device='cuda:0'):
        """One IRL update: refresh agent rollouts and descend the MaxEnt loss."""
        agent_traj = []
        per_env = max(1, batch_size // len(agent_envs))
        for env in agent_envs:
            agent_traj += self._generate_agent_trajectories(env, model, per_env)

        # sample an expert minibatch
        n_expert = min(batch_size, len(self.expert_data))
        idx = np.random.choice(len(self.expert_data), size=n_expert, replace=False)
        expert_batch = [self.expert_data[i] for i in idx]

        expert_rewards = self._calculate_rewards(expert_batch, device)
        agent_rewards = self._calculate_rewards(agent_traj, device)
        n = agent_rewards.shape[0]

        # L = - E_E[R] + log E_A[exp(R)]   (logsumexp - log n = log mean exp)
        loss = -(expert_rewards.mean()
                 - (torch.logsumexp(agent_rewards, dim=0) - float(np.log(n))))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            'loss': loss.item(),
            'expert_reward': expert_rewards.mean().item(),
            'agent_reward': agent_rewards.mean().item(),
        }


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def process_data(data_dict, device="cuda:0"):
    """Prepare a (default-collated) batch dict for the environment.

    Env tensors are kept on CPU: the environment converts observations to
    numpy anyway, and for large-N markets (e.g. sp500, N=472) the
    [T, N, N] relation matrices would otherwise exhaust GPU memory. Only
    the policy / reward networks live on the GPU. (The pre-processed .pkl
    samples carry no pyg_data graph object, so it is not referenced here.)
    """
    corr = data_dict['corr'].squeeze()
    ts_features = data_dict['ts_features'].squeeze()
    features = data_dict['features'].squeeze()
    industry_matrix = data_dict['industry_matrix'].squeeze()
    pos_matrix = data_dict['pos_matrix'].squeeze()
    neg_matrix = data_dict['neg_matrix'].squeeze()
    labels = data_dict['labels'].squeeze()
    mask = data_dict['mask']
    return corr, ts_features, features, industry_matrix, pos_matrix, neg_matrix, labels, mask


def _build_env(args, data, reward_net=None, mode="train",
               benchmark_return=None, closed_form_reward=False):
    corr, ts, feat, ind, pos, neg, labels, mask = process_data(data, device=args.device)
    env = StockPortfolioEnv(
        args=args, corr=corr, ts_features=ts, features=feat,
        ind=ind, pos=pos, neg=neg, returns=labels,
        benchmark_return=benchmark_return, mode=mode,
        reward_net=reward_net, device=args.device,
        ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
        closed_form_reward=closed_form_reward,
        lambda_return=getattr(args, "lambda_return", 1.0),
        lambda_div=getattr(args, "lambda_div", 0.1),
        lambda_pos=getattr(args, "lambda_pos", 0.1),
        lambda_neg=getattr(args, "lambda_neg", 0.1),
        momentum_threshold=getattr(args, "m_threshold", 0.0),
    )
    env.seed(seed=args.seed)
    return env


def create_env_init(args, dataset=None, data_loader=None):
    """Build a placeholder vec-env so PPO can be constructed."""
    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)
    for data in data_loader:
        env = _build_env(args, data, reward_net=None, mode="train")
        venv, _ = env.get_sb_env()
        print("placeholder env created")
        return venv


PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "gamma": 0.5,
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_on_loader(args, model, loader, verbose=False):
    """Run the (deterministic) policy over a full-period loader and return
    portfolio metrics plus the net-value curve.

    The raw env is stepped directly (not via DummyVecEnv) so the episode
    statistics are not wiped by an automatic reset on the terminal step.
    """
    metrics, net_values = None, None
    for data in loader:
        corr, ts, feat, ind, pos, neg, labels, mask = process_data(data, device=args.device)
        # equal-weight market return as the IR benchmark (no index file shipped)
        benchmark = labels.mean(dim=-1).detach().cpu().numpy()
        env = StockPortfolioEnv(args=args, corr=corr, ts_features=ts, features=feat,
                                ind=ind, pos=pos, neg=neg, returns=labels,
                                benchmark_return=benchmark,
                                mode="test" if verbose else "eval",
                                device=args.device,
                                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn)
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        metrics = env.evaluate()
        net_values = list(env.net_value_s)
        break
    return metrics, net_values


# --------------------------------------------------------------------------
# Main training routine
# --------------------------------------------------------------------------
def train_model_and_predict(model, args, train_loader, val_loader, test_loader,
                            writer=None, run_dir=None):
    """PPO + one of three reward setups, picked by ``args.reward``:

      'irl'         -- MaxEnt IRL on the heuristic expert (paper §3.3)
      'closed_form' -- paper §3.2 analytical reward, no learned net
      'gail'        -- GAIL discriminator, bounded reward via softplus
    """
    method = getattr(args, "reward", "irl")
    print(f"[reward mode] {method}")

    # --- setup the reward signal --------------------------------------------
    reward_net, rew_trainer, expert_trajectories = None, None, None
    save_net, ckpt_name = None, None
    pb_normalizer, pb_ref_trajs = None, None
    if method == "closed_form":
        # no learned reward; the env computes the §3.2 formula every step
        train_data = next(iter(train_loader))
        train_env_obj = _build_env(args, train_data, reward_net=None,
                                   mode="train", closed_form_reward=True)
    elif method == "pb_irl":
        from trainer.pb_irl import (decode_variant, RewardNormalizer, PBIRLTrainer,
                                    build_preference_dataset, load_prior_reward_net,
                                    default_policy_path)
        agg, use_norm, kl_on = decode_variant(args.pb_variant)
        # phi_prior: build the trainable reward net initialised from stage 1
        bare_net = load_prior_reward_net(args)
        reward_net = RewardNormalizer(bare_net).to(args.device) if use_norm else bare_net
        # warm-start the PPO policy from stage 1 (sb3 appends '.zip')
        policy_path = default_policy_path(args)
        if not (os.path.exists(policy_path) or os.path.exists(policy_path + ".zip")):
            raise FileNotFoundError(
                f"stage-1 PPO policy not found: {policy_path}(.zip) "
                f"(set --prior_run_dir or --init_policy)")
        model.set_parameters(policy_path, device=args.device)
        # preference dataset + BT-loss trainer
        pref = build_preference_dataset(args, train_loader.dataset)
        rew_trainer = PBIRLTrainer(bare_net, pref, kl_coef=args.pb_kl_coef,
                                   kl_on=kl_on, lr=args.irl_lr, n_pairs=args.pb_pairs)
        save_net, ckpt_name = bare_net, "best_reward_net.pt"
        if use_norm:
            pb_normalizer = reward_net
            pb_ref_trajs = list(pref["traj_cache"].values())
        train_data = next(iter(train_loader))
        train_env_obj = _build_env(args, train_data, reward_net=reward_net, mode="train")
    else:
        # IRL or GAIL: build the heuristic expert dataset
        from gen_data.generate_expert import generate_expert_trajectories
        expert_trajectories = generate_expert_trajectories(
            args, train_loader.dataset, num_trajectories=args.num_expert)
        print(f"generated {len(expert_trajectories)} expert trajectories")

        if method == "gail":
            reward_net = GAILDiscriminator(
                input_dim=args.input_dim, num_stocks=args.num_stocks,
                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
            ).to(args.device)
            rew_trainer = GAILTrainer(reward_net, expert_trajectories, lr=args.irl_lr)
            save_net, ckpt_name = reward_net, "best_discriminator.pt"
        else:  # 'irl'
            obs_len = args.input_dim
            for flag in (args.ind_yn, args.pos_yn, args.neg_yn):
                if flag:
                    obs_len += args.num_stocks
            if not args.multi_reward:
                reward_net = RewardNetwork(input_dim=obs_len + 1).to(args.device)
            else:
                reward_net = MultiRewardNetwork(
                    input_dim=args.input_dim, num_stocks=args.num_stocks,
                    ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
                ).to(args.device)
            rew_trainer = MaxEntIRL(reward_net, expert_trajectories, lr=args.irl_lr)
            save_net, ckpt_name = reward_net, "best_reward_net.pt"

        train_data = next(iter(train_loader))
        train_env_obj = _build_env(args, train_data, reward_net=reward_net, mode="train")

    train_env, _ = train_env_obj.get_sb_env()

    # --- alternating loop ----------------------------------------------------
    ppo_steps = max(args.ppo_steps, PPO_PARAMS["n_steps"])
    best_val_sr = -1e18
    for epoch in range(args.max_epochs):
        # (a) reward update (skipped in closed-form mode)
        stats = {}
        if rew_trainer is not None:
            stats = rew_trainer.train_step(
                [train_env], model, batch_size=args.irl_batch, device=args.device)

        # (b) PPO under the current reward signal
        model.set_env(train_env)
        model.learn(total_timesteps=ppo_steps, reset_num_timesteps=True)

        # (b2) refresh reward normalization stats from a frozen snapshot
        if pb_normalizer is not None:
            pb_normalizer.update(pb_ref_trajs, device=args.device)

        # (c) logging
        if writer is not None:
            writer.add_scalar("epoch", epoch, epoch)
            for k, v in stats.items():
                writer.add_scalar(f"{method}/{k}", float(v), epoch)
            beta_net = getattr(reward_net, "net", reward_net)  # unwrap RewardNormalizer
            if isinstance(beta_net, MultiRewardNetwork):
                for name, b in zip(beta_net.feature_dims.keys(),
                                   beta_net.beta().detach().cpu().numpy()):
                    writer.add_scalar(f"reward_net/beta_{name}", float(b), epoch)

        # (d) periodic evaluation + best-val checkpoint
        if epoch % args.eval_every == 0 or epoch == args.max_epochs - 1:
            val_m, _ = evaluate_on_loader(args, model, val_loader, verbose=False)
            test_m, _ = evaluate_on_loader(args, model, test_loader, verbose=False)
            if writer is not None:
                for k, v in val_m.items():
                    writer.add_scalar(f"val/{k}", v, epoch)
                for k, v in test_m.items():
                    writer.add_scalar(f"test/{k}", v, epoch)
            extra = (f"loss={stats['loss']:.3f}" if "loss" in stats else "no_reward_train")
            print(f"[epoch {epoch:3d}] {extra} | "
                  f"val ARR={val_m['ARR']:+.3f} SR={val_m['SR']:+.3f} | "
                  f"test ARR={test_m['ARR']:+.3f} SR={test_m['SR']:+.3f} "
                  f"MDD={test_m['MDD']:+.3f}")
            if run_dir is not None and val_m["SR"] > best_val_sr:
                best_val_sr = val_m["SR"]
                model.save(os.path.join(run_dir, "best_model"))
                if save_net is not None:
                    torch.save(save_net.state_dict(),
                               os.path.join(run_dir, ckpt_name))

    # --- final test eval -----------------------------------------------------
    test_m, test_nv = evaluate_on_loader(args, model, test_loader, verbose=True)
    if run_dir is not None:
        pd.DataFrame({"net_value": test_nv}).to_csv(
            os.path.join(run_dir, "test_net_value.csv"), index=False)
        pd.DataFrame([test_m]).to_csv(
            os.path.join(run_dir, "test_metrics.csv"), index=False)
    return model, test_m
