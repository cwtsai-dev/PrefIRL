#!/usr/bin/env python
"""One-shot progress reporter for the sp500 sweep: prints one line per run
with current epoch, rate, and ETA. Designed to be called repeatedly by a
Monitor."""
import glob
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

now = time.time()
rows = []
for d in sorted(glob.glob("logs/sp500_*seed0")):
    try:
        ea = EventAccumulator(d)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        # 'epoch' is logged every epoch in every mode now; falls back if missing
        for tag in ("epoch", "irl/loss", "gail/loss", "test/SR", "val/SR"):
            if tag in tags and len(ea.Scalars(tag)) > 0:
                s = ea.Scalars(tag)
                cur = s[-1].step
                if len(s) > 1 and cur > 0:
                    rate = (s[-1].wall_time - s[0].wall_time) / cur
                else:
                    rate = 0.0
                eta_m = (199 - cur) * rate / 60.0 if rate > 0 else 0.0
                age = now - s[-1].wall_time
                rows.append((d.split("/")[-1], cur, rate, eta_m, age))
                break
        else:
            rows.append((d.split("/")[-1], -1, 0.0, 0.0, 0.0))
    except Exception:
        pass

ts = time.strftime("%H:%M")
if not rows:
    print(f"[{ts}] (no logs yet)")
else:
    for name, cur, rate, eta_m, age in rows:
        if cur < 0:
            print(f"[{ts}] {name:40s} no data")
        elif cur >= 199:
            print(f"[{ts}] {name:40s} DONE (ep {cur})")
        else:
            print(f"[{ts}] {name:40s} ep {cur:3d}/199  "
                  f"{rate:5.0f}s/ep  ETA {eta_m:5.0f} min  "
                  f"(last {age:.0f}s ago)")
