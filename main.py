import os
import time
import glob
import pickle
import random
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from policy.policy import HGATActorCriticPolicy
from trainer.irl_trainer import PPO_PARAMS, create_env_init, train_model_and_predict


def str2bool(v):
    return str(v).lower() in ("y", "yes", "true", "1")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    p = argparse.ArgumentParser(description="Heuristic-guided IRL for portfolio optimization")
    p.add_argument("--market", default="hs300", help="hs300 / zz500 / nd100 / sp500 / tw50")
    p.add_argument("--policy", default="MLP", choices=["MLP", "HGAT"])
    p.add_argument("--relation_type", default="hy")
    p.add_argument("--horizon", default="1")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--ppo_steps", type=int, default=2048)
    p.add_argument("--irl_lr", type=float, default=1e-4)
    p.add_argument("--irl_batch", type=int, default=64)
    p.add_argument("--num_expert", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--input_dim", type=int, default=6)
    p.add_argument("--ind_yn", default="y")
    p.add_argument("--pos_yn", default="y")
    p.add_argument("--neg_yn", default="y")
    p.add_argument("--multi_reward", default="y")
    p.add_argument("--train_start", default="2018-01-01")
    p.add_argument("--train_end", default="2022-12-31")
    p.add_argument("--val_start", default="2023-01-01")
    p.add_argument("--val_end", default="2023-12-31")
    p.add_argument("--test_start", default="2024-01-01")
    p.add_argument("--test_end", default="2024-12-31")
    p.add_argument("--tag", default="")
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--batch_size", type=int, default=0, help="PPO minibatch (0=default)")
    p.add_argument("--n_steps", type=int, default=0, help="PPO rollout length (0=default)")
    # reward source
    p.add_argument("--reward", default="irl",
                   choices=["irl", "closed_form", "gail", "pb_irl"],
                   help="reward signal: MaxEnt IRL (paper §3.3, default), "
                        "closed-form §3.2 formula, GAIL discriminator, or "
                        "preference-based IRL (stage 2)")
    # closed-form reward weights (paper §3.2 lambdas)
    p.add_argument("--lambda_return", type=float, default=1.0)
    p.add_argument("--lambda_div",    type=float, default=0.1)
    p.add_argument("--lambda_pos",    type=float, default=0.1)
    p.add_argument("--lambda_neg",    type=float, default=0.1)
    p.add_argument("--m_threshold",   type=float, default=0.0,
                   help="momentum threshold m_thr in R_pos / R_neg (default 0)")
    # PB-IRL (stage 2) -----------------------------------------------------
    p.add_argument("--pb_variant", default="norm",
                   choices=["base", "mean-reward", "norm", "norm-nokl"])
    p.add_argument("--pb_prior_source", default="irl", choices=["irl", "gail"],
                   help="how to parse the stage-1 reward checkpoint")
    p.add_argument("--prior_run_dir", default="",
                   help="stage-1 run dir holding best_reward_net.pt/"
                        "best_discriminator.pt and best_model.zip")
    p.add_argument("--prior_reward_net", default="",
                   help="explicit path override for the reward checkpoint")
    p.add_argument("--init_policy", default="",
                   help="explicit path override for the PPO policy checkpoint")
    p.add_argument("--pb_horizon", type=int, default=60)
    p.add_argument("--pb_margin", type=float, default=0.5)
    p.add_argument("--pb_recency", type=float, default=1.0)
    p.add_argument("--pb_kl_coef", type=float, default=1.0)
    p.add_argument("--pb_pairs", type=int, default=256)
    return p.parse_args()


def main():
    args = get_args()
    args.ind_yn = str2bool(args.ind_yn)
    args.pos_yn = str2bool(args.pos_yn)
    args.neg_yn = str2bool(args.neg_yn)
    args.multi_reward = str2bool(args.multi_reward)
    args.prior_reward_net = args.prior_reward_net or None
    args.init_policy = args.init_policy or None
    if args.reward == "pb_irl" and not (args.prior_run_dir or args.prior_reward_net):
        raise SystemExit("pb_irl requires --prior_run_dir (or --prior_reward_net)")
    args.model_name = "SmartFolio"
    set_seed(args.seed)
    print(f"CUDA available: {torch.cuda.is_available()} | device: {args.device}")

    data_dir = f"dataset/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"preprocessed data not found: {data_dir}")

    # number of stocks is determined by the preprocessed data, not hard-coded;
    # scan past any empty early-period samples
    args.num_stocks = 0
    for sf in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        s = pickle.load(open(sf, "rb"))
        if s["labels"].shape[0] > 0:
            args.num_stocks = int(s["labels"].shape[0])
            break
    if args.num_stocks == 0:
        raise RuntimeError(f"no non-empty .pkl samples in {data_dir}")
    print(f"market={args.market}  num_stocks={args.num_stocks}  policy={args.policy}")

    # HGAT needs all three relation graphs in the observation
    if args.policy == "HGAT":
        args.ind_yn = args.pos_yn = args.neg_yn = True

    train_ds = AllGraphDataSampler(base_dir=data_dir, date=True, mode="train",
                                   train_start_date=args.train_start, train_end_date=args.train_end)
    val_ds = AllGraphDataSampler(base_dir=data_dir, date=True, mode="val",
                                 val_start_date=args.val_start, val_end_date=args.val_end)
    test_ds = AllGraphDataSampler(base_dir=data_dir, date=True, mode="test",
                                  test_start_date=args.test_start, test_end_date=args.test_end)
    print(f"days: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=len(train_ds), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), pin_memory=True)

    tag = (args.tag + "_") if args.tag else ""
    if args.reward == "irl":
        rew = ""
    elif args.reward == "pb_irl":
        src = "" if args.pb_prior_source == "irl" else f"{args.pb_prior_source}prior_"
        rew = f"pb_irl_{args.pb_variant}_{src}"
    else:
        rew = f"{args.reward}_"
    run_name = f"{args.market}_{args.policy}_{rew}{tag}seed{args.seed}"
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    print(f"tensorboard run dir: {run_dir}")

    ppo_params = dict(PPO_PARAMS)
    if args.batch_size > 0:
        ppo_params["batch_size"] = args.batch_size
    if args.n_steps > 0:
        ppo_params["n_steps"] = args.n_steps

    env_init = create_env_init(args, dataset=train_ds)
    if args.policy == "MLP":
        model = PPO(policy="MlpPolicy", env=env_init, **ppo_params,
                    seed=args.seed, device=args.device, verbose=0)
    else:
        model = PPO(policy=HGATActorCriticPolicy, env=env_init, **ppo_params,
                    seed=args.seed, device=args.device, verbose=0)

    t0 = time.time()
    model, test_metrics = train_model_and_predict(
        model, args, train_loader, val_loader, test_loader, writer=writer, run_dir=run_dir)
    elapsed = time.time() - t0

    for k, v in test_metrics.items():
        writer.add_scalar(f"final_test/{k}", v, 0)
    writer.close()
    model.save(os.path.join(run_dir, "final_model"))

    print(f"\n=== {run_name} finished in {elapsed/60:.1f} min ===")
    print("final test metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()
