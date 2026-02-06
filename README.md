# Multi-Model Equity Signal Research & Alpha Ablation (US S&P 500)

This repo implements an end-to-end research pipeline for **daily US equity signals** on the S&P 500:

- raw price data → factor construction → forward-return labels  
- model zoo: **linear**, **MLP**, **XGBoost**, **Transformer**  
- cost-aware long–short backtest with Sharpe / IC / drawdown / turnover  
- robustness tests and **alpha ablation** (including synthetic / AI-generated factors)  
- hooks for **RL training** and future **graph-based models** on stock relationships

The goal is to answer, quantitatively:

> *Given a fixed factor set and cost model, how much edge does each model really have, and how stable is that edge under perturbations and distribution shift?*

---

## 1. Experiment setup

- **Universe:** S&P 500 constituents (≈`N` stocks) from Kaggle  
- **Frequency:** daily bars  
- **Date range:** `YYYY-MM-DD` – `YYYY-MM-DD`  
- **Label:** `k`-day forward return (e.g. 5-day)  
- **Split:**  
  - Train / validation: `< YYYY-MM-DD`  
  - Out-of-sample (OOS): `≥ YYYY-MM-DD`  

All configuration is stored in [`config.yaml`](config.yaml).

---

## 2. Data pipeline

Scripts:

- [`src/make_factors.py`](src/make_factors.py)  
  - Builds value / momentum / quality / technical factors from raw S&P 500 prices.
  - Outputs `data/factors.csv` and `data/factors/` (per-ticker panels).

- [`src/make_labels.py`](src/make_labels.py)  
  - Computes `k`-day forward returns and alignment with factor dates.
  - Outputs `data/labels.csv`.

Raw prices live in `data/prices/` (Kaggle S&P 500 dataset, pre-cleaned).

To regenerate factors + labels:

```bash
python src/make_factors.py --config config.yaml
python src/make_labels.py  --config config.yaml