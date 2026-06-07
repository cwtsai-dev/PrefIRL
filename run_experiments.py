#!/usr/bin/env python
"""Multi-GPU experiment runner.

PPO with a single DummyVecEnv is not data-parallel, so "multi-GPU" here means
launching independent experiments (markets x policies x seeds) concurrently,
spread across the available GPUs. Each run logs to its own tensorboard dir.

Example:
    python run_experiments.py --markets hs300 --policies MLP HGAT \\
        --seeds 0 1 2 --gpus 0 1 2 3 --jobs-per-gpu 1 --epochs 80
"""
import os
import sys
import time
import argparse
import subprocess

VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")


def build_jobs(markets, policies, seeds, epochs, extra):
    jobs = []
    for m in markets:
        for p in policies:
            for s in seeds:
                name = f"{m}_{p}_seed{s}"
                args = ["--market", m, "--policy", p, "--seed", str(s),
                        "--max_epochs", str(epochs)]
                args += extra
                jobs.append({"name": name, "args": args})
    return jobs


def run(jobs, gpus, jobs_per_gpu, logdir):
    os.makedirs(logdir, exist_ok=True)
    slots = [g for g in gpus for _ in range(jobs_per_gpu)]
    active = [None] * len(slots)
    pending = list(jobs)
    done = []
    print(f"scheduling {len(jobs)} jobs over {len(slots)} slots "
          f"({len(gpus)} GPUs x {jobs_per_gpu})")

    while pending or any(active):
        for i, gpu in enumerate(slots):
            if active[i] is None and pending:
                job = pending.pop(0)
                logpath = os.path.join(logdir, job["name"] + ".log")
                logf = open(logpath, "w")
                cmd = ([VENV_PYTHON, "-u", "main.py"] + job["args"]
                       + ["--device", f"cuda:{gpu}"])
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                active[i] = {"proc": proc, "job": job, "logf": logf,
                             "gpu": gpu, "start": time.time()}
                print(f"  [start] {job['name']:<28s} gpu{gpu}  (log: {logpath})")
            elif active[i] is not None:
                a = active[i]
                if a["proc"].poll() is not None:
                    a["logf"].close()
                    dur = (time.time() - a["start"]) / 60.0
                    rc = a["proc"].returncode
                    status = "ok" if rc == 0 else f"FAILED rc={rc}"
                    print(f"  [done ] {a['job']['name']:<28s} {status}  ({dur:.1f} min)")
                    done.append((a["job"]["name"], rc))
                    active[i] = None
        time.sleep(5)

    print("\n=== all jobs finished ===")
    n_ok = sum(1 for _, rc in done if rc == 0)
    print(f"{n_ok}/{len(done)} succeeded")
    for name, rc in done:
        if rc != 0:
            print(f"  FAILED: {name}  (see {logdir}/{name}.log)")
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=["hs300"])
    p.add_argument("--policies", nargs="+", default=["MLP", "HGAT"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--logdir", default="run_logs")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra args passed through to main.py")
    args = p.parse_args()

    if not os.path.exists(VENV_PYTHON):
        sys.exit(f"venv python not found at {VENV_PYTHON}; run `uv sync` first")

    jobs = build_jobs(args.markets, args.policies, args.seeds, args.epochs, args.extra)
    t0 = time.time()
    run(jobs, args.gpus, args.jobs_per_gpu, args.logdir)
    print(f"total wall time: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
