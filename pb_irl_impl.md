# PB-IRL 方法說明

## 背景：原版 SmartFolio 的 Reward 學習

SmartFolio 使用 **Inverse Reinforcement Learning（IRL）** 從專家策略反推 reward function，再用 PPO 訓練 portfolio policy。原版用 **MaxEnt IRL**（最大熵 IRL）。

我們的改進：把 reward 學習這一步換成 **Preference-Based IRL（PB-IRL）**，HGAT policy 和 PPO 訓練迴圈完全不動。

---

## PB-IRL 核心想法

**問題：** MaxEnt IRL 需要假設 reward 的函數形式，對絕對量敏感。
**改法：** 改用「比較哪個策略比較好」代替「直接算每個策略的 reward 值」。

具體步驟：

### Step 1 — 建立 Preference Dataset（偏好資料集）

- 對訓練期每一天 `t`，用現有專家策略選股，組成一個 portfolio
- 計算該 portfolio 在未來 60 天的 **Sharpe ratio**（品質衡量）
- 兩兩配對：若 Sharpe_A > Sharpe_B + margin，則記錄「A 比 B 好」
- 過濾掉差距太小的對（margin filter）、對近期資料給更高權重（recency weight）

### Step 2 — Bradley-Terry Loss（BT Loss）訓練 Reward Net

用 Bradley-Terry 配對比較模型：

```
P(A ≻ B) = σ(R_φ(τ_A) − R_φ(τ_B))
```

Loss = 讓 reward net 預測的偏好方向與真實 Sharpe 排序一致。

### Step 3 — KL 正則化（Prior Anchoring）

- 先跑一次 IRL 得到 `φ_prior`（基礎 reward net checkpoint）
- PB-IRL 訓練時加入 KL 懲罰項，讓新的 reward net 不要偏離 IRL prior 太遠

```
Loss_total = BT_loss + λ_KL × ||φ − φ_prior||²
```

這讓 PB-IRL 在學偏好資訊的同時保留 IRL 學到的結構知識。

### Step 4 — Reward Normalization（norm 變體）

訓練過程中 reward net 輸出量級容易爆炸（BT loss 只約束兩個 trajectory 的差值，不約束絕對值），導致 PPO 的 value function 估計不穩定。

解法：每個 epoch 結束後，用上一個 epoch 的 frozen reward net 算出 `μ, σ`，對當前 reward 做 z-score normalization：

```
R_normalized = (R_raw − μ) / σ
```

類似 DQN 的 target network 概念——用延遲更新的快照當基準，避免 reward 自我膨脹。

---

## 三個實驗變體

| 變體 | 差異 |
|---|---|
| **mean-reward** | trajectory reward 用 `.mean()` 而非 `.sum()`，防止長 window 放大 reward 差距 |
| **norm** | 加上 RewardNormalizer（z-score），解決 BT loss saturation |
| **norm-nokl** | norm + 拿掉 KL prior，看 prior anchoring 有多重要 |

---

## 兩階段訓練流程

```
Stage 1: 跑 IRL → 得到 best_reward_net.pt（φ_prior）
Stage 2: 跑 PB-IRL，用 φ_prior 做 KL 正則化錨點
```

---

## 初步結果

### SP500（2024 測試期）

| 策略 | SR | ARR |
|---|---|---|
| HGAT-GAIL | **1.942** | 25.3% |
| HGAT-PB-IRL-norm | 1.460 | 19.7% |
| MLP-PB-IRL-mean-reward | 1.491 | 19.2% |
| MLP-GAIL | 1.584 | 19.1% |
| HGAT-IRL | 1.231 | 16.4% |
| MLP-IRL | 1.204 | 15.2% |
| HGAT-PB-IRL-norm-nokl | 0.988 | 12.8% |
| MLP-PB-IRL-norm | 0.814 | 11.1% |
| EqualWeight（1/N） | 1.158 | 14.4% |

### ND100（2024 測試期）

| 策略 | SR | ARR |
|---|---|---|
| MLP-PB-IRL-norm | **0.925** | 18.6% |
| HGAT-PB-IRL-mean-reward | 0.752 | 16.5% |
| HGAT-GAIL | 0.730 | 14.1% |
| MLP-GAIL | 0.619 | 10.2% |
| HGAT-IRL | 0.699 | 13.3% |
| MLP-IRL | 0.483 | 7.7% |
| HGAT-PB-IRL-norm | 0.508 | 10.2% |
| HGAT-PB-IRL-norm-nokl | -0.465 | -7.9% |
| EqualWeight（1/N） | 0.762 | 11.9% |
| Random-topk | 1.080 | 20.3% |

### 關鍵觀察

- **SP500：GAIL 最強**（SR 1.94），PB-IRL-norm 對 HGAT 有效（SR 1.46 > IRL 1.23）
- **ND100：市場較難預測**，Random-topk（SR 1.08）竟勝過大多數學習方法
- **MLP vs HGAT 逆轉**：SP500 上 HGAT 領先，ND100 上 MLP-PB-IRL-norm 最佳（SR 0.925）
- **移除 KL prior 關鍵**：ND100 HGAT-norm-nokl 崩潰至 SR -0.465；SP500 影響較小（SR 0.988）→ prior anchoring 在高雜訊市場更重要
- HS300 / ZZ500 實驗進行中，跨市場結論待補充

