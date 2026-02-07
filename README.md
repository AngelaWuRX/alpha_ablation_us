# Multi-Model Equity Signal Research & Alpha Ablation (S&P 500)

This repo implements an end-to-end **equity signal research pipeline** on an S&P 500 universe:

- raw prices → factor panel → `1-day` forward-return labels  
- model zoo: **IC-weighted linear baseline**, **MLP**, **XGBoost**, **Transformer**  
- cost-aware long–short backtest with Sharpe / total return / max drawdown / turnover  
- hooks for robustness tests, alpha ablation, and **RL / graph-based** extensions

The core question:

> Given a fixed factor set and cost model, **how much edge does each model really have**, and how does that edge behave once you include turnover and drawdowns?

---

## 1. Data & experiment setup

- **Universe:** S&P 500 constituents (Kaggle dataset)
- **Frequency:** daily bars
- **Date range:** 2016-01-01 – 2026-01-16
- **Label:** `1-day` forward return, stored as `fwd_ret`
- **Split (time-based):**
  - Train / validation: **2016-01-01 – 2021-12-31**
  - Out-of-sample (OOS): **2022-01-01 – 2026-01-16**

The time split is controlled by `config.yaml`:

The main modeling scripts expect a single panel CSV with at least:
	•	date (timestamp)
	•	ticker (equity identifier)
	•	close (closing price)
	•	factor columns from features.list
	•	fwd_ret (forward return label)

## 2. First experiment results

### Out-of-sample performance (2022–2026, after costs)

| Model        | Sharpe | Total Return | Max Drawdown | Avg Turnover |
| :----------- | -----: | -----------: | -----------: | -----------: |
| Linear       | 0.2908 | +17.84%      | −36.29%      | 60.1%        |
| MLP          | 0.0716 | −2.32%       | −25.27%      | 0.0%         |
| Transformer  | 0.3228 | +20.86%      | −35.18%      | 141.4%       |
| **XGBoost ⭐** | **0.4796** | **+53.50%** | −46.25%      | 99.31%       |

> 🟢 **Best risk/return trade-off:** XGBoost.  
> 🔴 **Failed baseline:** MLP (negative OOS return and essentially zero turnover).

Interpretation: 
- Tree-based XGBoost is the clear winner on this factor set:
- Highest Sharpe and total return, despite a relatively high turnover.
- Transformer beats linear but does not beat XGBoost, and trades even more (turnover ≈ 141%).
- MLP collapsed:
	- Negative OOS return and essentially zero turnover — more a sign of training/regularization issues than alpha.
- Turnover and costs are first-order effects:
	- Naively comparing raw returns would miss how sensitive some models are to trading frictions.

Output:
	![Strategy Performace Comparison.png](https://github.com/AngelaWuRX/alpha_ablation_us/blob/0856cd9316502021093d820cebf450a36d1fd5a5/Strategy%20Performace%20Comparison.png)

## 3. Models

All models follow the same basic pattern:

1. Load the panel from `data_path` (CSV with factors + `fwd_ret`).
2. Convert `date` to `datetime`.
3. Split into **train** (`date < train_split_date`) and **test/OOS** (`date ≥ train_split_date`).
4. Train on the **train** window only.
5. Write out OOS `score` and realized `fwd_ret` for each `(date, ticker)`.

Each model writes a file of the form:

```text
date, ticker, close, score, fwd_ret
```

### 3.1 IC-weighted linear baseline

Implementation: run_linear_baseline(data_path, output_path)

File: typically src/models_linear.py (or wherever you put the function).

Logic:
	•	Uses config['features']['list'] as the factor set.
	•	Uses config['models']['train_split_date'] to define train vs test.

Training:
	1.	Compute daily cross-sectional IC (Spearman rank correlation between each factor and fwd_ret) on the train sample:
```text
daily_ic = train_df.groupby("date").apply(
    lambda x: x[col].rank().corr(x["fwd_ret"].rank())
)
ic_values[col] = daily_ic.mean()
```
	2.	Stack the mean ICs into a weight vector w (one weight per factor).
	3.	On the test sample, compute a composite linear score:

```text
test_df['score'] = test_df[factor_cols].values @ weights
```
Output:
	•	output_path CSV with columns ['date', 'ticker', 'close', 'score', 'fwd_ret'].

Interpretation:
	•	This is not a generic regression; it’s an IC-weighted factor portfolio.
It gives you a clean, interpretable linear baseline built from cross-sectional information content.

### 3.2 MLP

Implementation: run_mlp_model(data_path, output_path)

Pipeline:
	1.	Read CSV, parse date.
	2.	Train = date < train_split_date, Test = date ≥ train_split_date.
	3.	Features = config['features']['list'], label = config['features']['label'] (usually fwd_ret).
	4.	Standardize features with StandardScaler fitted on train only.
	5.	Train with MSE loss and Adam optimizer for mlp_cfg['epochs'].
	6.	Predict OOS scores on the test set and write to CSV.

This is a pure cross-sectional MLP: each (date, ticker) row is treated independently.
Temporal structure is not used yet; that’s deliberate and documented as a limitation.

### 3.3 Transformer (degenerate 1-step TS encoder)

Implementation: run_transformer_model(data_path, output_path)

Model class: TS_Transformer
	•	Embeds factor vector → d_model
	•	Passes through a nn.TransformerEncoder with n_layers and n_heads
	•	Takes the last time step and maps to a scalar via fc
```text
Current usage:
	•	The code reshapes each factor vector as a sequence of length 1:
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
```
So this Transformer is effectively a fancy nonlinear map on the factor vector, not a true multi-step time-series model. That’s fine for a first pass, but it’s explicitly noted as a limitation.

The rest mirrors the MLP:
	•	same time split by train_split_date
	•	same StandardScaler on features
	•	MSE loss, Adam optimizer
	•	OOS scores written to CSV.

### 3.4 XGBoost (tree-based nonlinear model)

Implementation: run_xgb_model(data_path, output_path)
	•	Hyperparameters come from config['models']['xgb']:
	•	n_estimators, max_depth, learning_rate, etc.
	•	Train / test split is again controlled by models.train_split_date.

Pipeline:
	1.	Load panel CSV, parse date.
	2.	train_df = df[df['date'] < train_split_date].dropna()
	3.	test_df = df[df['date'] >= train_split_date].copy()
	4.	Fit xgb.XGBRegressor on (train_df[factor_cols], train_df['fwd_ret']).
	5.	Predict on test_df[factor_cols] and write score + fwd_ret to CSV.

This model is where the nonlinear interactions in factor space really show up.

## 4. Backtest & evaluation

Given a signals file, the backtest:
	1.	For each day, ranks stocks by score.
	2.	Goes long the top quantile and short the bottom quantile (equal weight within legs).
	3.	Applies a transaction cost proportional to turnover.
	4.	Produces a daily PnL series and turnover for each model:

src/summary_backtest.py aggregates these into a summary table:
	•	Sharpe (after costs)
	•	Total return
	•	Max drawdown (MDD)
	•	Avg daily turnover
	•	IC metrics

## 5. Robustness & alpha ablation

Planned robustness work (wired in run_robustness.py, alpha_ablation_improved.py):
	1.	Factor-group ablation
	•	Tag factors as value / momentum / quality / technical.
	•	Retrain XGBoost while dropping one group at a time.
	•	Measure ΔSharpe, ΔMDD, and Δturnover.
	•	Goal: identify which factor groups actually carry edge vs redundancy/noise.
	2.	Noise / perturbation tests
	•	Add noise to factors and/or labels.
	•	Increase transaction costs to see how quickly each model breaks.
	•	This directly tests fragility of each model under realistic stress.
	3.	Synthetic / AI-generated factors (optional)
	•	Add synthetic features from model-generated transforms.
	•	Check whether they create real out-of-sample edge or just overfit the train window.

## 6. Runtime & efficiency

Training time is a constraint in this project. On my actual hardware (single laptop, full 2016–2026 history, daily S&P 500 universe), the wall-clock looks like:

- **Linear IC baseline:** fast (seconds)  
- **XGBoost:** on the order of minutes for a full train + OOS prediction run  
- **MLP:** ~3 hours per full train on the full sample  
- **Transformer:** ~12 hours per full train on the full sample  

A few important consequences:

- **Neural models are expensive here.**  
  With the current configuration, a single Transformer run is ~4× slower than the MLP and massively slower than XGBoost, while delivering **worse Sharpe** than XGBoost.

- **XGBoost hits the best accuracy / compute trade-off.**  
  It gives the strongest out-of-sample performance in this setup while being cheap enough to re-run for robustness and ablation experiments.

- **Experiment design is compute-limited.**  
  Because the neural models take hours, most robustness and alpha ablation work is run on XGBoost (and sometimes the linear baseline), **by design**: that’s where you can realistically afford many variants.

## 7. RL & graph-based extensions (research direction)

This repo is the base for two “next-step” research directions:

7.1 RL on top of signals

Prototype code (e.g. rl_training/discrete_dqn.py) sets up:
	•	State: factor/signal vector for a given (date, ticker) or portfolio.
	•	Action: discrete position (short, flat, long) or discretized size.
	•	Reward: after-cost PnL.

Baseline comparison:
	•	Simple rule-based threshold strategies vs DQN policies.
	•	Key questions:
	•	Can RL reduce drawdowns or turnover while preserving edge?
	•	Does it adapt better to regime changes than fixed rules?

7.2 From trees to graphs (stock networks)

XGBoost winning in the first experiment suggests that structure and nonlinear interactions matter.
The next step is to add structure to the universe itself:
	•	Build a stock graph:
	•	nodes = tickers
	•	edges = sector relationships, correlation clusters, co-movement, etc.
	•	Either:
	•	apply simple graph algorithms on top of signals, or
	•	plug a GNN into the same pipeline to output score per stock.
