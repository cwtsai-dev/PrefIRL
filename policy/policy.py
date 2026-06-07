import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from model.model import HGAT


class HGATNetwork(nn.Module):
    """Custom mlp_extractor that uses the heterogeneous graph attention
    network (paper section 3.4) as the shared trunk of the actor-critic.

    SB3 flattens the Box observation; we recover the [N, d + 3N] layout
    inside ``HGAT.forward``. The trunk emits an N-dim latent for both the
    policy and value heads (SB3 adds the final Linear action/value heads).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 n_head: int = 8, hidden_dim: int = 128,
                 no_ind: bool = False, no_neg: bool = False):
        super(HGATNetwork, self).__init__()
        num_stocks, obs_len = observation_space.shape
        input_dim = obs_len - 3 * num_stocks
        assert input_dim > 0, (
            f"HGAT expects obs layout [N, d + 3N]; got N={num_stocks}, obs_len={obs_len}. "
            f"All three relation graphs (ind/pos/neg) must be enabled for the HGAT policy."
        )
        self.num_stocks = num_stocks
        # SB3 reads these to size the action/value heads
        self.latent_dim_pi = num_stocks
        self.latent_dim_vf = num_stocks

        self.policy_net = HGAT(num_stocks=num_stocks, n_features=input_dim,
                               num_heads=n_head, hidden_dim=hidden_dim,
                               no_ind=no_ind, no_neg=no_neg)
        self.value_net = HGAT(num_stocks=num_stocks, n_features=input_dim,
                              num_heads=n_head, hidden_dim=hidden_dim,
                              no_ind=no_ind, no_neg=no_neg)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class HGATActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs,
                 ):
        # HGAT does its own Xavier init; skip SB3 orthogonal init of the trunk
        kwargs['ortho_init'] = False
        super(HGATActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HGATNetwork(self.observation_space,
                                         n_head=8, hidden_dim=128)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return super().forward(obs, deterministic)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic)
        return actions
