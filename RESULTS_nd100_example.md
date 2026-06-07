# SmartFolio — ND100 Results

Heuristic-guided IRL portfolio optimization (IJCAI-25 #1054), reproduced and evaluated on the ND100 test period (2024).

## Setup

- Market: **ND100**, train 2018–2022 / val 2023 / **test 2024**.
- Training config (paper 4.1): lr 1e-4, batch 128 (HGAT 32 — memory), 128-d hidden, 8 heads, 200 epochs, seed 0.
- Policies: **HGAT** (paper full model) and **MLP** (paper's *w/o HGAT* ablation).
- Baselines: non-learning strategies on the identical test set.

## Metrics (ND100, 2024 test)

ARR = annualised return, AVol = annualised volatility, SR = Sharpe, MDD = max drawdown, CR = Calmar, IR = information ratio (vs. equal-weight market).

| Strategy | ARR | AVol | SR | MDD | CR | IR |
|---|---|---|---|---|---|---|
| SmartFolio-HGAT-IRL (final) | 0.0812 | 0.1375 | 0.5677 | -0.0909 | 0.8930 | -0.2828 |
| SmartFolio-HGAT-IRL (best-val) | 0.1331 | 0.1788 | 0.6991 | -0.1314 | 1.0134 | 0.1556 |
| SmartFolio-HGAT-GAIL (final) | -0.0746 | 0.2008 | -0.3861 | -0.1660 | -0.4494 | -1.2856 |
| SmartFolio-HGAT-GAIL (best-val) | 0.1412 | 0.1811 | 0.7296 | -0.1242 | 1.1364 | 0.2336 |
| SmartFolio-MLP-IRL (final) | 0.0134 | 0.1365 | 0.0972 | -0.0999 | 0.1336 | -0.8100 |
| SmartFolio-MLP-IRL (best-val) | 0.0774 | 0.1543 | 0.4833 | -0.1037 | 0.7467 | -0.3303 |
| SmartFolio-MLP-GAIL (final) | -0.0508 | 0.1652 | -0.3158 | -0.1081 | -0.4701 | -1.6261 |
| SmartFolio-MLP-GAIL (best-val) | 0.1022 | 0.1571 | 0.6194 | -0.1017 | 1.0046 | -0.1558 |
| EqualWeight(1/N) | 0.1187 | 0.1473 | 0.7619 | -0.0982 | 1.2089 | 0.0000 |
| BuyAndHold | 0.1212 | 0.1519 | 0.7532 | -0.1048 | 1.1565 | 0.1166 |
| Momentum-topk | -0.0110 | 0.2668 | -0.0416 | -0.2734 | -0.0404 | -0.6471 |
| Random-topk | 0.2034 | 0.1714 | 1.0801 | -0.1071 | 1.8984 | 0.7296 |

Paper reference (Table 1, ND100, *Ours*): ARR 0.250, AVol 0.117, SR 1.906, MDD −0.058, CR 4.293, IR 1.184.

## Cumulative PnL

![PnL curve](results/pnl_curve_nd100.png)

Final cumulative wealth (start = 1.0):

- SmartFolio-HGAT-IRL (final): **1.0707**
- SmartFolio-HGAT-IRL (best-val): **1.1147**
- SmartFolio-HGAT-GAIL (final): **0.9073**
- SmartFolio-HGAT-GAIL (best-val): **1.1222**
- SmartFolio-MLP-IRL (final): **1.0040**
- SmartFolio-MLP-IRL (best-val): **1.0644**
- SmartFolio-MLP-GAIL (final): **0.9366**
- SmartFolio-MLP-GAIL (best-val): **1.0884**
- EqualWeight(1/N): **1.1062**
- BuyAndHold: **1.1079**
- Momentum-topk: **0.9544**
- Random-topk: **1.1851**

## Did it learn anything?

The PPO test metric varied a lot from epoch to epoch (typical for RL on a portfolio task) so two checkpoints are reported: **final-epoch** (paper convention) and **best-val** (the validation-Sharpe-maximising checkpoint, fairer for noisy training).

- **SmartFolio-HGAT-IRL (final)** (SR +0.568) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-HGAT-IRL (best-val)** (SR +0.699) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-HGAT-GAIL (final)** (SR -0.386) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-HGAT-GAIL (best-val)** (SR +0.730) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-MLP-IRL (final)** (SR +0.097) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-MLP-IRL (best-val)** (SR +0.483) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-MLP-GAIL (final)** (SR -0.316) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).
- **SmartFolio-MLP-GAIL (best-val)** (SR +0.619) matches/loses to 1/N (SR +0.762); matches/loses to random (SR +1.080).

