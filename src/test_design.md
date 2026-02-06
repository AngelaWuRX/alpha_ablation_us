# AI Test & Sanity Check Design

|> This document describes the design of the AI test / sanity-check layer for the ALPHA_ABLATION_US project.

The purpose of this layer is simple:

AI is allowed to generate code. It is not allowed to generate unchecked results.

**Everything in this directory exists to attack the pipeline and expose bugs, hallucinations, and leakage before any performance numbers are taken seriously.**


## 1. Motivation

This section is aimed for investigating: 
	1.	Silent hallucinations – the model can invent logic, metrics, or assumptions that look plausible but are wrong.
	2.	Subtle data/label leakage – anything involving time series joins, label generation, and backtesting is easy to get wrong in ways that still “look good” in backtest.

The AI test layer treats the rest of the codebase as untrusted and tries to falsify it with:
	•	negative controls (where the correct answer is “no edge”),
	•	invariance tests (properties that must hold under sign/scale/permute),
	•	small synthetic examples where the ground truth is known exactly.

If any of these tests fail, performance claims are not credible.

## 2. Scope
The AI test layer covers four main components of the pipeline:
	1.	Data pipeline
	•	factor construction (make_factors.py)
	•	label construction (make_labels.py)
	•	data utilities (utils.py)
	2.	Models
	•	linear / MLP / transformer / XGB (models_*.py, train_all.py)
	3.	Backtesting & metrics
	•	backtest engine (backtest.py)
	•	metric calculations (metrics.py)
	•	summary/report generation (make_summary.py, generate_plots.py)
	4.	Robustness & falsification
	•	robustness experiments (run_robustness.py)
	•	cross-model comparisons and ablations

The tests are written with pytest and can be run standalone or via the sanity_check.py entrypoint.

## 3. Directory Layout
ALPHA_ABLATION_US/
  src/
    ...
    sanity_check.py      
	entrypoint to run all AI tests
    test_data_pipeline.py
    test_models.py
    test_backtest.py
    test_robustness.py


## 4. Test Philosophy

The tests are designed around properties, not specific numeric outputs.

For each part of the pipeline we ask:

“If the world works the way we claim, what must always be true here?”

Then we encode that as a test. The focus is on:
	•	Leakage detection – does performance collapse when labels or returns are randomized?
	•	Invariance – do results behave correctly under known symmetries (sign flips, scaling, permutations)?
	•	Degenerate cases – do trivial strategies (no-trade, all-zero features) behave in the obvious way?
	•	Sanity comparisons – do different models trained on the same data at least agree more than random?

If the code passes these, it’s not “proved correct,” but it becomes much harder for a hallucinated bug to hide.


## 5. Test Types

### 5.1 Data / Leakage Tests (test_data_pipeline.py)

Goal: ensure there is no unintended look-ahead or misalignment between features, labels, and returns.

Key properties:
	•	Monotone time index
	•	Timestamps are strictly increasing with no duplicates.
	•	Feature–label alignment
	•	Each label at time t depends only on information at or before the intended horizon relative to t.
	•	Spot checks: recompute labels on a tiny slice from raw prices and compare.
	•	No look-ahead joins
	•	Shuffling the time order of factors before label construction must not change labels.
	•	Any change indicates use of global order information.
	•	Feature sanity
	•	No NaNs or infinities.
	•	No all-zero or all-constant feature columns.
	•	Basic distribution checks (reasonable means / standard deviations for z-scored factors).

If these tests fail, the dataset itself is untrustworthy.

### 5.2 Model Tests (test_models.py)

Goal: verify that models behave as expected on controlled data and degenerate scenarios.

Key properties:
	•	Random labels → random performance
	•	Shuffling labels destroys predictive structure.
	•	Metrics such as IC / Sharpe should collapse near 0.
	•	Permutation invariance
	•	Reordering samples in the training set (with labels) should not fundamentally change learned mapping for deterministic models.
	•	Degenerate features
	•	Training a model on all-zero features should yield constant predictions.
	•	Linear model sanity
	•	On small synthetic data where y = Xβ + ε, the linear model recovers β within tolerance.

Failing these tests usually means the model code is not solving the problem you think it is (e.g., using indices, mis-wired tensors, or wrong optimization).

### 5.3 Backtest & Metrics Tests (test_backtest.py)

Goal: ensure that position accounting and metrics do not fabricate alpha.

Key properties:
	•	No-trade baseline
	•	A zero-signal strategy must generate ~zero PnL and zero turnover.
	•	Random strategy baseline
	•	Random signals should give IC ≈ 0, Sharpe ≈ 0, hit rate ≈ 50% over long horizons.
	•	Sign flip invariance
	•	Flipping the sign of all signals should flip PnL but keep magnitude-based statistics (e.g., |Sharpe|) and turnover unchanged.
	•	Scaling invariance
	•	Multiplying signals by a positive constant should not change rank-based metrics (IC, rank-IC, hit rate). Position sizes and PnL should scale in a predictable way.
	•	Hand-checkable toy example
	•	On a tiny synthetic dataset (few dates, known returns), PnL, IC, and hit rate computed by metrics.py must match hand-calculated values.

If any of this fails, performance metrics are unreliable regardless of model quality.

### 5.4 Robustness & Falsification Tests (test_robustness.py)

Goal: check that reported performance is robust and that the system reacts correctly to deliberate manipulations.

Key properties:
	•	Leakage “spike” control
	•	If labels are deliberately shifted into the future (true leak), performance should explode.
	•	If real results are close to this inflated benchmark, something is wrong.
	•	Factor ablation
	•	Removing clearly predictive factors should harm performance; removing obviously useless ones should not magically improve it.
	•	Stability under small perturbations
	•	Adding tiny noise to features should not flip prediction signs everywhere; prediction correlation before/after noise should remain high.
	•	Cross-model agreement
	•	Different model families (linear, tree, MLP, transformer) trained on the same data should agree on direction more than random (sign agreement > 50%).
	•	Time-split consistency
	•	Performance should be in the same ballpark across reasonable train/test splits, not only for one cherry-picked window.

These tests don’t guarantee the strategy is good, but they sharply reduce the chance that an impressive backtest is just a quirk of the data or a bug.


## 6. How This Connects to the AI Report

The AI report that accompanies this project will explicitly reference this test suite. For each major result, it will state:
	•	which commit / config was tested,
	•	that the full AI test suite passed, and
	•	any known limitations (e.g., tests not yet implemented for a new factor family).

The message is: results are not just “good looking plots”; they are outputs of a system that has survived a deliberately hostile test environment.

## 7. Limitations and Future Work

This test layer has limits:
	•	It does not prove absence of all leakage or modeling errors.
	•	It does not guarantee out-of-sample commercial viability.
	•	It assumes a reasonably stationary environment for baselines like random labels.

Future directions:
	•	Add stress tests under extreme market regimes (crisis periods).
	•	Add tests for transaction cost modeling and slippage assumptions.
	•	Extend tests to multi-asset / portfolio-level constraints.

Even with these limits, the AI test layer is non-negotiable: no experiment result in this repo is considered meaningful unless it passes this suite.
