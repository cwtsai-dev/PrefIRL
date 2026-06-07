import os
import glob
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from policy.policy import HGATActorCriticPolicy
from trainer.irl_trainer import evaluate_on_loader

# Create dummy args class
class Args:
    def __init__(self, policy):
        self.market = "zz500"
        self.policy = policy
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.input_dim = 6
        self.ind_yn = True
        self.pos_yn = True
        self.neg_yn = True
        self.num_stocks = 80

def main():
    data_dir = "dataset/data_train_predict_zz500/1_hy/"
    test_ds = AllGraphDataSampler(base_dir=data_dir, date=True, mode="test",
                                  test_start_date="2024-01-01",
                                  test_end_date="2024-12-31")
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), pin_memory=False)

    models_to_eval = [
        {"dir": "logs/zz500/SmartFolio-HGAT-GAIL", "policy": "HGAT"},
        {"dir": "logs/zz500/SmartFolio-MLP-IRL", "policy": "MLP"},
        {"dir": "logs/zz500/SmartFolio-MLP-GAIL", "policy": "MLP"},
        {"dir": "logs/zz500/HGAT-IRL", "policy": "HGAT"},
    ]

    for m in models_to_eval:
        ckpt = os.path.join(m["dir"], "best_model.zip")
        if not os.path.exists(ckpt):
            print(f"Skipping {m['dir']} - best_model.zip not found")
            continue
        print(f"Evaluating best model for {m['dir']}...")
        args = Args(m["policy"])
        custom = {"HGATActorCriticPolicy": HGATActorCriticPolicy} if m["policy"] == "HGAT" else None
        model = PPO.load(ckpt, device=args.device, custom_objects=custom)
        metrics, nv = evaluate_on_loader(args, model, test_loader, verbose=True)

        pd.DataFrame([metrics]).to_csv(os.path.join(m["dir"], "test_metrics_bestval.csv"), index=False)
        pd.DataFrame({"net_value": nv}).to_csv(os.path.join(m["dir"], "test_net_value_bestval.csv"), index=False)
        print(f"Saved best-val results to {m['dir']}")

if __name__ == "__main__":
    main()
