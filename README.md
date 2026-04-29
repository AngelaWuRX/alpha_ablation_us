# Alpha Ablation (S&P 500)

Compare four models on the same factor set and cost model.

## Setup

- Universe: S&P 500, daily bars
- Train: 2016–2021 · OOS: 2022–2026
- Label: 1-day forward return
- Long top quantile, short bottom, after costs

## Results (OOS)

| Model       | Sharpe | Total Return | Max DD  | Turnover |
|-------------|-------:|-------------:|--------:|---------:|
| Linear      |  0.29  |   +17.8%     | −36.3%  |   60.1%  |
| MLP         |  0.07  |    −2.3%     | −25.3%  |    0.0%  |
| Transformer |  0.32  |   +20.9%     | −35.2%  |  141.4%  |
| XGBoost     |  0.48  |   +53.5%     | −46.3%  |   99.3%  |

**XGBoost wins.** Transformer trades too much for its edge. MLP failed to learn.

![comparison](Strategy%20Performace%20Comparison.png)
