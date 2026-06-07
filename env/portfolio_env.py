import gym
import pandas as pd
import torch
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    """Portfolio selection environment.

    One episode walks through the trading days contained in ``returns``.
    At each day t the agent picks ``top_k`` stocks (MultiDiscrete action);
    the selected stocks are equally weighted. The *realized* portfolio
    return for day t is ``w . returns[t]`` and is used for all evaluation
    metrics. During IRL training the reward fed to PPO is the learned
    reward network output instead of the realized return.
    """

    def __init__(self, args, corr=None, ts_features=None, features=None,
                 ind=None, pos=None, neg=None, returns=None,
                 benchmark_return=None, mode="train", reward_net=None, device='cuda:0',
                 ind_yn=False, pos_yn=False, neg_yn=False,
                 closed_form_reward=False,
                 lambda_return=1.0, lambda_div=0.1,
                 lambda_pos=0.1, lambda_neg=0.1,
                 momentum_threshold=0.0):
        super(StockPortfolioEnv, self).__init__()
        self.current_step = 0
        self.max_step = returns.shape[0] - 1
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = []
        self.num_stocks = returns.shape[-1]
        self.benchmark_return = benchmark_return

        self.corr_tensor = corr
        self.ts_features_tensor = ts_features
        self.features_tensor = features
        self.ind_tensor = ind
        self.pos_tensor = pos
        self.neg_tensor = neg
        self.ror_batch = returns
        self.ind_yn = ind_yn
        self.pos_yn = pos_yn
        self.neg_yn = neg_yn

        # closed-form reward (paper section 3.2): replaces the IRL reward net
        self.closed_form_reward = closed_form_reward
        self.lambdas = (lambda_return, lambda_div, lambda_pos, lambda_neg)
        self.m_threshold = momentum_threshold
        self.sector_ids = None
        self.n_sectors = 0
        if closed_form_reward and self.ind_tensor is not None:
            from scipy.sparse.csgraph import connected_components
            ind_np = self.ind_tensor[0].cpu().numpy() if self.ind_tensor.ndim == 3 \
                else self.ind_tensor.cpu().numpy()
            n, ids = connected_components((ind_np > 0).astype(int), directed=False)
            self.sector_ids = ids
            self.n_sectors = n

        # 允许选择固定数量的股票（论文 top-k = 10%）
        self.top_k = max(1, int(0.1 * self.num_stocks))
        # 动作空间：离散，表示选择的股票索引（multi-hot 选股，对应专家的二元 action）
        self.action_space = spaces.MultiDiscrete([self.num_stocks] * self.top_k)

        # 观测空间：股票特征及各个关系图（部分可观测）
        obs_len = args.input_dim
        if self.ind_yn:
            obs_len += self.num_stocks
        if self.pos_yn:
            obs_len += self.num_stocks
        if self.neg_yn:
            obs_len += self.num_stocks
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_stocks, obs_len),
                                            dtype=np.float32)
        self.mode = mode
        self.reward_net = reward_net  # 注入 IRL 奖励网络
        self.device = device

    def load_observation(self, ind_yn=False, pos_yn=False, neg_yn=False):
        # Stable-Baselines3 的 DummyVecEnv 需要将观测值存为 NumPy 数组
        if torch.isnan(self.features_tensor).any():
            print("warning: NaN in features tensor")
        features = self.features_tensor[self.current_step].cpu().numpy()
        obs = features
        if ind_yn:
            obs = np.concatenate([obs, self.ind_tensor[self.current_step].cpu().numpy()], axis=1)
        if pos_yn:
            obs = np.concatenate([obs, self.pos_tensor[self.current_step].cpu().numpy()], axis=1)
        if neg_yn:
            obs = np.concatenate([obs, self.neg_tensor[self.current_step].cpu().numpy()], axis=1)
        self.observation = obs.astype(np.float32)
        # ror at the *current* step — the return realized by an action taken now
        self.ror = self.ror_batch[self.current_step].cpu().numpy()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = []
        self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
        return self.observation

    def seed(self, seed):
        return np.random.seed(seed)

    def step(self, actions):
        # 动作对应当前观测 (obs at current_step)；先结算再前进时间
        selected_indices = list(set(int(a) for a in np.atleast_1d(actions)))
        weights = np.zeros(self.num_stocks, dtype=np.float32)
        weights[selected_indices] = 1.0 / len(selected_indices)

        # 当期实际投资组合收益（等权），用于净值与所有评估指标
        realized_return = float(np.dot(weights, np.asarray(self.ror, dtype=np.float32)))

        # 奖励：closed-form (论文 §3.2) > IRL/GAIL 学习奖励 > 实际收益
        if self.closed_form_reward:
            self.reward = self._closed_form_reward(weights, realized_return)
        elif self.reward_net is not None:
            state_tensor = torch.FloatTensor(np.expand_dims(self.observation, 1)).to(self.device)
            action_multi_hot = np.zeros(self.num_stocks, dtype=np.float32)
            action_multi_hot[selected_indices] = 1.0
            action_tensor = torch.FloatTensor(action_multi_hot).to(self.device)
            with torch.no_grad():
                self.reward = float(self.reward_net(state_tensor, action_tensor).mean().cpu().item())
        else:
            self.reward = realized_return

        # 净值始终跟踪实际收益
        self.net_value *= (1.0 + realized_return)
        self.daily_return_s.append(realized_return)
        self.net_value_s.append(self.net_value)

        # 时间前进
        self.current_step += 1
        self.done = self.current_step > self.max_step
        if not self.done:
            self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)

        info = {}
        if self.done:
            metrics = self.evaluate()
            info = {'metrics': metrics}
            if self.mode == "test":
                print("================ test episode finished ================")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
                print("=======================================================")
        return self.observation, self.reward, self.done, info

    def _closed_form_reward(self, weights, realized_return):
        """Paper section 3.2 analytical reward, computed directly from
        (action, ρ, momentum) -- no learned reward network.

        R_total = λ1 * log(1 + r) + λ2 * H(sector_weights)
                  + λ3 * R_pos + λ4 * R_neg
            R_pos = -Σ_ij w_i w_j max(0, ρ_ij) 𝕀(m_i < m_thr)
            R_neg =  Σ_ij w_i w_j |min(0, ρ_ij)| 𝕀(m_i ≥ m_thr)
        """
        lr, ld, lp, ln = self.lambdas

        # R_return: log-return of the portfolio this step
        r_return = float(np.log(max(1e-8, 1.0 + realized_return)))

        # R_diversity: entropy of the selected sector-weight distribution
        r_div = 0.0
        wsum = float(weights.sum())
        if self.sector_ids is not None and wsum > 0:
            sw = np.zeros(self.n_sectors, dtype=np.float64)
            np.add.at(sw, self.sector_ids, weights)
            p = sw / wsum
            p_nz = p[p > 0]
            r_div = float(-(p_nz * np.log(p_nz)).sum())

        # R_pos / R_neg: correlation management gated by per-stock momentum
        r_pos = 0.0
        r_neg = 0.0
        if self.corr_tensor is not None:
            corr = self.corr_tensor[self.current_step].cpu().numpy()
            mom = (self.ror_batch[self.current_step - 1].cpu().numpy()
                   if self.current_step > 0
                   else np.zeros(self.num_stocks, dtype=np.float32))
            ind_low = (mom < self.m_threshold).astype(np.float32)
            ind_high = (mom >= self.m_threshold).astype(np.float32)
            ww = weights[:, None] * weights[None, :]
            pos_part = np.maximum(corr, 0.0)
            neg_part = np.abs(np.minimum(corr, 0.0))
            r_pos = -float((ww * pos_part * ind_low[:, None]).sum())
            r_neg = float((ww * neg_part * ind_high[:, None]).sum())

        return lr * r_return + ld * r_div + lp * r_pos + ln * r_neg

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_df_net_value(self):
        df_net_value = pd.DataFrame(self.net_value_s)
        df_net_value.columns = ["net_value"]
        return df_net_value

    def get_df_daily_return(self):
        df_daily_return = pd.DataFrame(self.daily_return_s)
        df_daily_return.columns = ["daily_return"]
        return df_daily_return

    def evaluate(self):
        """Compute ARR, AVol, Sharpe, MDD, Calmar, Information Ratio."""
        metrics = {'ARR': 0.0, 'AVol': 0.0, 'SR': 0.0, 'MDD': 0.0, 'CR': 0.0, 'IR': 0.0}
        if len(self.daily_return_s) == 0:
            return metrics
        df_daily_return = self.get_df_daily_return()
        if df_daily_return["daily_return"].std() != 0:
            # 年化收益 ARR（假设一年 252 个交易日）
            arr = (1 + df_daily_return['daily_return'].mean()) ** 252 - 1
            # 年化波动率 AVol
            avol = df_daily_return["daily_return"].std() * (252 ** 0.5)
            # 夏普比率 SR
            sp = ((252 ** 0.5) * df_daily_return["daily_return"].mean()
                  / df_daily_return["daily_return"].std())
            # 累积收益与最大回撤 MDD
            df_daily_return['cumulative_return'] = (1 + df_daily_return['daily_return']).cumprod()
            running_max = df_daily_return['cumulative_return'].cummax()
            drawdown = df_daily_return['cumulative_return'] / running_max - 1
            mdd = drawdown.min()
            # 卡玛比率 CR
            cr = arr / abs(mdd) if mdd != 0 else 0.0
            # 信息比率 IR（相对基准的超额收益）
            ir = 0.0
            if self.benchmark_return is not None:
                bench = np.asarray(self.benchmark_return, dtype=np.float64)
                if len(bench) == len(df_daily_return):
                    ex_return = df_daily_return["daily_return"].values - bench
                    if ex_return.std() != 0:
                        ir = ex_return.mean() / ex_return.std() * (252 ** 0.5)
            metrics = {'ARR': float(arr), 'AVol': float(avol), 'SR': float(sp),
                       'MDD': float(mdd), 'CR': float(cr), 'IR': float(ir)}
        return metrics
