#!/usr/bin/env python
"""Re-evaluate the best-validation checkpoint on the test set.

Training is noisy (RL on a portfolio task) and the *final*-epoch model is
not necessarily the *best* one. ``train_model_and_predict`` already saves
``best_model.zip`` by validation Sharpe; this script loads that checkpoint
and re-runs the test evaluation, writing
  logs/<run>/test_metrics_bestval.csv
  logs/<run>/test_net_value_bestval.csv

Usage:  python eval_best.py --market sp500 --policy MLP
        python eval_best.py --market sp500 --policy HGAT
"""
import os
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from policy.policy import HGATActorCriticPolicy
from trainer.irl_trainer import evaluate_on_loader


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="sp500")
    p.add_argument("--policy", required=True, choices=["MLP", "HGAT"])
    p.add_argument("--reward", default="irl",
                   choices=["irl", "closed_form", "gail"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--test_start", default="2024-01-01")
    p.add_argument("--test_end", default="2024-12-31")
    p.add_argument("--run_dir", default="",
                   help="explicit run dir override (else derived from "
                        "market/policy/reward/seed); needed for pb_irl run names")
    args = p.parse_args()
    # the env / reward inputs need the same flags train_model_and_predict used
    args.input_dim = 6
    args.ind_yn = args.pos_yn = args.neg_yn = True
    args.multi_reward = True
    return args


def main():
    args = build_args()

    data_dir = f"dataset/data_train_predict_{args.market}/1_hy/"
    # num_stocks from a non-empty pkl (matches main.py)
    for sf in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        s = pickle.load(open(sf, "rb"))
        if s["labels"].shape[0] > 0:
            args.num_stocks = int(s["labels"].shape[0])
            break

    rew = "" if args.reward == "irl" else f"{args.reward}_"
    run_dir = args.run_dir or f"logs/{args.market}_{args.policy}_{rew}seed{args.seed}"
    ckpt = os.path.join(run_dir, "best_model.zip")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    print(f"loading {ckpt}")
    custom = {"HGATActorCriticPolicy": HGATActorCriticPolicy} if args.policy == "HGAT" else None
    model = PPO.load(ckpt, device=args.device, custom_objects=custom)

    test_ds = AllGraphDataSampler(base_dir=data_dir, date=True, mode="test",
                                  test_start_date=args.test_start,
                                  test_end_date=args.test_end)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), pin_memory=False)

    metrics, nv = evaluate_on_loader(args, model, test_loader, verbose=True)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(run_dir, "test_metrics_bestval.csv"), index=False)
    pd.DataFrame({"net_value": nv}).to_csv(
        os.path.join(run_dir, "test_net_value_bestval.csv"), index=False)
    print("=== best-val checkpoint -> test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
