# 🧩 Time Series Feature Engineering

> Status: ✅ Complete — [Open the notebook →](13_time_series_feature_engineering.ipynb)

Topic 13 of the Time Series Analysis repo — the bridge back to the [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) repo's tabular models. Every topic so far modeled a series directly (ARIMA's own recursive structure, a state-space model's hidden states); this notebook instead turns a time series into an ordinary supervised-learning table — lag features, causal rolling statistics, and cyclical calendar features — so that any regressor (here, `RandomForestRegressor`) can be pointed at it. Built and validated carefully, this reveals one of the most important and under-discussed pitfalls in applied time series ML: a tree-based model **cannot extrapolate** beyond the range of values it saw during training, and on a genuinely trending series like AirPassengers this failure is severe enough to score worse than a naive forecast — until a single feature-engineering fix (targeting the seasonal difference instead of the raw level) turns the same model into the most accurate one built anywhere in this repo so far.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Formula Reference](#-formula-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
A time series y_1, y_2, ..., y_n becomes an ordinary supervised-learning table:

  Lag features:      y_(t-1), y_(t-2), ..., y_(t-k)        -- "what happened recently"
  Rolling stats:      mean/std/min/max of y_(t-w..t-1)       -- "recent typical behavior"
  Calendar cycles:    sin/cos(2*pi*period_position/period)   -- "where in the cycle are we"

  =>  X_t = [y_(t-1), y_(t-2), roll_mean_w(t), dow_sin(t), ...]   ->   any regressor   ->   y_hat_t

CRITICAL RULE: every feature at time t may use ONLY data strictly before t (Topic 11's leakage lesson).

CRITICAL PITFALL: tree-based models (Random Forest, GBM) predict by averaging leaf-node training
targets -- they can NEVER output a value above their training-set maximum. On a trending series,
this is a structural failure, not a tuning problem.
```

---

## 🎯 Why This Topic Matters

- **From-scratch causal feature-engineering functions were built and validated** — §1 generated 18 features (lags, rolling mean/std/min/max, day-of-week and day-of-year cyclical encodings) from 2 raw columns on synthetic data, every one shifted to use only past information.
- **Topic 11's leakage lesson was reconfirmed on a tree model, not just linear regression** — §2 found a centered (non-causal) rolling-window feature making a `RandomForestRegressor` look **74.3%** more accurate on its own (MAE 0.4840 vs. the causal version's 1.8843) purely from boundary leakage, with zero genuine predictive improvement.
- **A severe, real, and often-missed pitfall was found and precisely quantified on real data** — §4 found a `RandomForestRegressor` fit directly on AirPassengers' raw level scoring walk-forward **MASE = 1.0319** (Topic 11's own 4-fold methodology) — *worse than a naive same-month-last-year forecast* — because every fold's test-period maximum exceeded the model's training-set maximum, and a tree model is structurally unable to predict above values it has already seen.
- **A one-line feature-engineering fix (not a model change) resolved the pitfall completely** — §5 found retargeting the same `RandomForestRegressor` at the year-over-year seasonal difference (Topic 02's differencing idea, repurposed) instead of the raw level cutting walk-forward MAE by **65.9%**, to **MASE = 0.3518** — the lowest of any real-data forecast built anywhere in this repo, including Topic 07/11's SARIMA (MASE 0.6695) and Topic 12's Kalman-filter BSM (MASE 0.6403).
- **That large improvement was honestly stress-tested with Topic 11's own significance test, not just reported** — §6 found the detrended Random Forest vs. SARIMA Diebold-Mariano test giving **p = 0.1083** — not significant at the 5% level with only 48 pooled observations, an honest "promising, not proven" conclusion despite the striking raw MAE gap.
- **Feature importances gave a concrete, practical diagnostic no ARIMA-family model in this repo can offer** — §7 found the lag-1 seasonal-difference feature alone accounting for **36.5%** of the fitted model's total importance.

---

## 🧮 Mathematical Explanation

### 1. Lag features

$$X_t^{(\text{lag}_k)} = y_{t-k}, \qquad k \in \{1, 2, 3, \dots\}$$

### 2. Causal rolling-window statistics

$$X_t^{(\text{roll}_w,\ \text{mean})} = \frac{1}{w}\sum_{i=1}^{w} y_{t-i}, \qquad X_t^{(\text{roll}_w,\ \text{std})} = \sqrt{\frac{1}{w}\sum_{i=1}^{w}\left(y_{t-i}-X_t^{(\text{roll}_w,\ \text{mean})}\right)^2}$$

the window is shifted by 1 before aggregating, so the statistic at $t$ never includes $y_t$ itself.

### 3. Cyclical calendar encoding

$$X_t^{(\sin)} = \sin\!\left(\frac{2\pi \cdot \text{position}_t}{\text{period}}\right), \qquad X_t^{(\cos)} = \cos\!\left(\frac{2\pi \cdot \text{position}_t}{\text{period}}\right)$$

the sin/cos pair (rather than a raw integer like month-of-year) ensures December (12) and January (1) are encoded as adjacent, not distant — the same Fourier-basis idea used for Prophet's seasonality in Topic 10, applied per-timestamp instead of as a whole-series regressor.

### 4. The extrapolation ceiling of a tree-based model

For a regression tree, every prediction is the mean of training targets in some leaf $\mathcal{L}$:

$$\hat y_t = \frac{1}{|\mathcal{L}|}\sum_{i \in \mathcal{L}} y_i \qquad \Longrightarrow \qquad \min_i y_i \ \le\ \hat y_t \ \le\ \max_i y_i \ \ \text{over all training targets}$$

so no tree-based prediction can ever exceed $\max(y_{\text{train}})$, regardless of the input features.

### 5. The detrending fix

Instead of modeling $y_t$ directly, model the bounded seasonal difference and reconstruct afterward exactly:

$$z_t = y_t - y_{t-12}, \qquad \hat y_t = y_{t-12} + \hat z_t$$

$z_t$ has no trend, so its range stays roughly constant over time even while $y_t$ keeps growing — removing the extrapolation ceiling at its source.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Lag feature | $X_t^{(\text{lag}_k)}=y_{t-k}$ | §1, §3 |
| Causal rolling mean/std | window shifted by 1, then aggregated over $w$ past points | §1, §3 |
| Cyclical calendar encoding | $\sin/\cos(2\pi \cdot \text{position}/\text{period})$ | §1, §3 |
| Tree prediction bound | $\hat y_t \in [\min(y_{\text{train}}), \max(y_{\text{train}})]$ | §4 |
| Seasonal-difference detrending | $z_t=y_t-y_{t-12}$, reconstruct $\hat y_t=y_{t-12}+\hat z_t$ | §5 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Fitting a tree-based model directly on a trending series' raw level** — §4 found this scoring MASE=1.0319 (worse than naive) purely because the model can never predict above its training-set maximum, regardless of feature quality.
2. **Computing rolling/window features on the full series before splitting** — §2 found a centered rolling feature giving a 74.3%-better-looking (but entirely fake) test score through boundary leakage alone.
3. **Assuming a feature-engineering fix needs a complicated model change** — §5's fix was a one-line change (predict the seasonal difference, not the level) that cut MAE by 65.9% without touching the model itself.
4. **Reading an impressive MAE improvement as automatically "proven"** — §6 found the detrended RF's large edge over SARIMA not reaching statistical significance (p=0.1083) with only 48 pooled observations; Topic 11's Diebold-Mariano discipline applies here just as much as it did to comparing two classical models.
5. **Treating feature importance as unconditional truth about the real world** — §7's importances are specific to this exact feature set and this exact model; a different lag/window choice or a different regressor could redistribute the same predictive signal across different-looking features.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| Custom `make_lag_features()` | From-scratch lag-column generator |
| Custom `make_rolling_features()` | From-scratch causal rolling mean/std/min/max generator |
| Custom `make_calendar_features()` | From-scratch sin/cos cyclical calendar encoder |
| `sklearn.ensemble.RandomForestRegressor` | Tree-based regressor used throughout to expose the extrapolation pitfall and its fix |
| `RandomForestRegressor.feature_importances_` | Library feature-importance diagnostic |
| `statsmodels.tsa.statespace.sarimax.SARIMAX` | Real SARIMA fitting, reused directly from Topics 07/11 for comparison |
| Custom `diebold_mariano_test()` | Topic 11's forecast-comparison test, reused directly |

---

## 📝 Self-Test Exercises

1. Section 2 found a centered rolling-window feature making a Random Forest look 74.3% more accurate purely from leakage. Using the causal rolling-mean formula in §2 of the Mathematical Explanation, explain precisely which specific test-set rows would be affected if the window size were increased from 11 to 51, and why.
2. Section 4 found every fold's Random Forest maximum prediction falling below that fold's true test maximum. Using the tree-prediction-bound formula in §4 of the Mathematical Explanation, explain why increasing `n_estimators` (the number of trees) would **not** fix this ceiling, while increasing `max_depth` alone also would not.
3. Section 5 found detrending by the seasonal difference $z_t=y_t-y_{t-12}$ fixing the extrapolation problem. Propose what would happen if the series had a strong non-seasonal (e.g., linear) trend instead — would $z_t=y_t-y_{t-12}$ still remove the ceiling, and if not, what differencing scheme from Topic 02 or Topic 06 would you use instead?
4. Section 6 found the detrended Random Forest's large MAE edge over SARIMA not reaching significance (p=0.1083) with 48 pooled observations. Using the Diebold-Mariano test's dependence on $n$ from Topic 11, explain what would need to happen to the pooled sample size (e.g., more folds or a longer series) for the same effect size to become statistically significant.
5. Section 7 found the lag-1 seasonal-difference feature dominating feature importance (36.5%). Propose one additional feature (not already in the notebook's feature set) that might capture information the current lag and rolling features miss, and explain what specific real-world pattern in AirPassengers it would be designed to detect.

---

## 📓 Notebook

12 executed code cells: from-scratch causal lag, rolling-statistic, and cyclical calendar feature-engineering functions; a direct reconfirmation of Topic 11's leakage lesson on a tree-based model (74.3% fake improvement from a centered rolling window); a rigorous walk-forward demonstration (reusing Topic 11's exact 4-fold methodology) that a `RandomForestRegressor` fit on AirPassengers' raw level cannot extrapolate past its training maximum and scores worse than naive (MASE 1.0319); the one-line seasonal-differencing fix that cuts its MAE by 65.9% (MASE 0.3518, the best real-data result in this repo); and an honest Diebold-Mariano significance check against Topic 07/11's SARIMA:

➡️ **[13_time_series_feature_engineering.ipynb](13_time_series_feature_engineering.ipynb)**

---

## 📚 Further Reading

- [Hyndman & Athanasopoulos: Forecasting: Principles and Practice — Ch. on Time Series Regression Features](https://otexts.com/fpp3/)
- [Why Tree-Based Models Struggle to Extrapolate (discussion of the leaf-averaging mechanism)](https://scikit-learn.org/stable/modules/tree.html)
- [`scikit-learn` RandomForestRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Bergmeir, Hyndman & Koo: A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction](https://doi.org/10.1016/j.csda.2017.11.003)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.13_Time_Series_Feature_Engineering&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
