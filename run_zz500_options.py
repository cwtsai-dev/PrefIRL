#!/usr/bin/env python3
"""Orchestrator script to run 5 specific zz500 configurations and organize results.

Options:
1. SmartFolio-HGAT-GAIL
2. SmartFolio-MLP-IRL
3. SmartFolio-MLP-GAIL
4. EqualWeight(1/N)
5. Momentum-topk

Results are saved directly in:
logs/zz500/[option]/
"""
import os
import shutil
import subprocess
import pandas as pd

# Paths
DATA_DIR = "dataset/data_train_predict_zz500/1_hy"
LOGS_DIR = "logs/zz500"
TMP_LOGS = "tmp_logs"

def run_command(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    # 1. Build dataset if it doesn't exist
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(">>> Building zz500 dataset...")
        run_command(["python", "gen_data/build_dataset.py", "--market", "zz500"])
    else:
        print(">>> zz500 dataset already built.")

    # Clean previous temp logs if any
    if os.path.exists(TMP_LOGS):
        shutil.rmtree(TMP_LOGS)

    # 2. Run Learning Agents
    agents = [
        {
            "name": "SmartFolio-HGAT-GAIL",
            "policy": "HGAT",
            "reward": "gail",
            "src_dir": "zz500_HGAT_gail_seed0"
        },
        {
            "name": "SmartFolio-MLP-IRL",
            "policy": "MLP",
            "reward": "irl",
            "src_dir": "zz500_MLP_seed0"
        },
        {
            "name": "SmartFolio-MLP-GAIL",
            "policy": "MLP",
            "reward": "gail",
            "src_dir": "zz500_MLP_gail_seed0"
        }
    ]

    for agent in agents:
        dest_dir = os.path.join(LOGS_DIR, agent["name"])
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"\n>>> Training Agent: {agent['name']}")
        cmd = [
            "python", "main.py",
            "--market", "zz500",
            "--policy", agent["policy"],
            "--reward", agent["reward"],
            "--seed", "0",
            "--logdir", TMP_LOGS,
            "--max_epochs", "200",
            "--eval_every", "20"
        ]
        run_command(cmd)

        # Move results to target folder
        src_path = os.path.join(TMP_LOGS, agent["src_dir"])
        if os.path.exists(src_path):
            print(f">>> Moving results from {src_path} to {dest_dir}")
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dest_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(src_path)
        else:
            print(f"WARNING: Source path {src_path} not found!")

    # 3. Run Baselines
    print("\n>>> Running baselines...")
    run_command(["python", "baselines.py", "--market", "zz500"])

    # 4. Extract and Organize Baselines
    bm_file = "results/baselines_metrics_zz500.csv"
    bc_file = "results/baselines_curves_zz500.csv"

    if os.path.exists(bm_file) and os.path.exists(bc_file):
        df_metrics = pd.read_csv(bm_file)
        df_curves = pd.read_csv(bc_file)

        baselines = [
            {"name": "EqualWeight_1_N", "strategy_name": "EqualWeight(1/N)"},
            {"name": "Momentum-topk", "strategy_name": "Momentum-topk"}
        ]

        for bl in baselines:
            dest_dir = os.path.join(LOGS_DIR, bl["name"])
            os.makedirs(dest_dir, exist_ok=True)

            print(f"\n>>> Saving baseline results to: {dest_dir}")
            # Extract metrics
            metric_row = df_metrics[df_metrics["strategy"] == bl["strategy_name"]]
            if not metric_row.empty:
                metric_row.to_csv(os.path.join(dest_dir, "test_metrics.csv"), index=False)
            
            # Extract net values
            if bl["strategy_name"] in df_curves.columns:
                curve_col = df_curves[[bl["strategy_name"]]].rename(columns={bl["strategy_name"]: "net_value"})
                curve_col.to_csv(os.path.join(dest_dir, "test_net_value.csv"), index=False)

    # Clean up
    if os.path.exists(TMP_LOGS):
        shutil.rmtree(TMP_LOGS)

    print("\n>>> All tasks completed successfully! Results structured under logs/zz500/")

if __name__ == "__main__":
    main()
