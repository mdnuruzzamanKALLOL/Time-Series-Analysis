# Multi-Step Forecasting Strategies & Ensemble Forecasting

> Status: ✅ Complete

Recursive vs. direct forecasting, and combining multiple forecasts — the final, capstone topic of this repo

## Concept & Intuition

Every model built across this repo — ARIMA, SARIMA, VAR, GARCH, Prophet, Kalman filters, Random Forests, LSTMs — has produced forecasts more than one step ahead, but *how* it does so has been left implicit. There are two fundamentally different strategies. **Recursive (iterated)** forecasting fits a single one-step-ahead model and feeds its own forecasts back in as if they were real data to reach further horizons — this is what ARIMA/SARIMA/VAR/Kalman-filter forecasts have all been doing under the hood. **Direct** forecasting instead fits a *separate* model for each horizon, regressing the $h$-steps-ahead target directly on information available today. Direct avoids compounding one-step errors, but pays for it with less efficient parameter estimation. Neither strategy is a universal winner — this notebook proves that with real simulations, not assertions.

The topic closes with the natural follow-up question: if several models are available, why pick just one? **Ensemble forecasting** combines multiple forecasts, and this notebook builds a genuine ensemble from three models constructed *earlier in this repo* — SARIMA (Topics 07/11), a Basic Structural Model (Topic 12), and a direct Random Forest (Topic 13) — testing the well-documented **forecast combination puzzle**: simple averages of decent forecasts are surprisingly hard to beat.

## Mathematical Explanation

**Forecast uncertainty inherently grows with horizon.** For a correctly specified AR(1) process $y_t=\phi y_{t-1}+\varepsilon_t$, the $h$-step-ahead forecast error variance has closed form:

$$\text{Var}(e_h) = \sigma^2\sum_{i=0}^{h-1}\phi^{2i} \;\longrightarrow\; \frac{\sigma^2}{1-\phi^2} \text{ as } h\to\infty$$

This growth is unavoidable for *any* strategy — it reflects genuine, irreducible uncertainty about the future, not a flaw in how the forecast is produced.

**Recursive strategy.** Fit one model $\hat y_t = f(y_{t-1},\dots,y_{t-p})$; forecast $h$ steps by iterating: $\hat y_{t+1}=f(y_t,\dots)$, then $\hat y_{t+2}=f(\hat y_{t+1}, y_t,\dots)$, and so on, feeding forecasts back in as inputs.

**Direct strategy.** For each horizon $h=1,\dots,H$, fit a separate model $\hat y_{t+h} = g_h(y_t,\dots,y_{t-p})$ using only information available at time $t$ — no iteration, no compounding.

**Forecast combination.** Given $K$ forecasts $\hat y^{(1)},\dots,\hat y^{(K)}$, a combined forecast is a weighted average $\sum_k w_k \hat y^{(k)}$. The simplest choice is equal weights $w_k=1/K$; a more "sophisticated" choice (Bates & Granger, 1969) is inverse-error weighting, $w_k \propto 1/\text{MAE}_k$, estimated from past performance. The **forecast combination puzzle** (Clemen, 1989) is the repeated empirical finding that equal weights are extremely hard to beat, because estimating better weights introduces its own estimation noise.

## Code Implementation (`20_multi_step_forecasting_strategies_ensemble_forecasting.ipynb`)

### 1. Forecast uncertainty grows with horizon (validated)

A Monte Carlo simulation (20,000 repetitions) of an AR(1) with $\phi=0.7,\ \sigma=1$ matches the theoretical closed-form variance almost exactly at every horizon (e.g. $h=1$: 1.0000 theoretical vs. 1.0147 empirical; $h=10$: 1.9592 vs. 1.9538), confirming the growth toward the unconditional variance (1.96) as $h\to\infty$.

### 2. Recursive vs. direct: correctly specified model

500 Monte Carlo repetitions of a true AR(2) process ($\phi_1=0.5,\ \phi_2=0.3$), $n=150$ training points, both strategies given the correct 2 lags:

| $h$ | RMSE recursive | RMSE direct |
|---|---|---|
| 1 | 1.1144 | 1.1144 |
| 2 | 1.1546 | 1.1608 |
| 5 | 1.3204 | 1.3383 |
| 10 | 1.4356 | 1.4648 |

**Recursive wins at 9 of 10 horizons** (tied at $h=1$, where they're identical by construction). Even with a correctly specified model, direct's per-horizon regressions have higher finite-sample estimation variance than iterating a single, more parsimonious one-step model — direct's "no compounding" advantage doesn't materialize when there's no misspecification to compound in the first place.

### 3. Recursive vs. direct: misspecified model

The same experiment, but both strategies are now deliberately underspecified — fit with only 1 lag (AR(1)) when the truth is AR(2):

| $h$ | RMSE recursive | RMSE direct | Direct better? |
|---|---|---|---|
| 1 | 1.1276 | 1.1276 | no (tied) |
| 2 | 1.1823 | 1.1604 | **yes** |
| 4 | 1.3473 | 1.3219 | **yes** |
| 6 | 1.3707 | 1.3665 | **yes** |
| 7 | 1.4877 | 1.4900 | no |
| 10 | 1.4530 | 1.4550 | no |

**Direct wins at 5 of 10 horizons** — a genuinely mixed result. Under misspecification, direct's ability to target each horizon's loss directly helps at mid-range horizons, but recursive catches back up at longer horizons. This matches real findings in the forecasting literature (Marcellino, Stock & Watson, 2006): there is no universal rule, and both strategies should be tried and validated out-of-sample for any specific problem.

### 4. Real data: Random Forest, recursive vs. direct, on AirPassengers

A walk-forward comparison (7 origins, Topic 11 methodology, 12-month horizon, seasonally-differenced features from Topic 13) shows a clear overall winner:

| Strategy | Overall mean MASE |
|---|---|
| RF recursive | 0.7302 |
| RF direct | **0.6594** |

Direct wins clearly at 8 of 12 individual horizons. This reflects Topic 13's finding that Random Forests cannot extrapolate: once a recursive forecast drifts even slightly, subsequent iterations feed that bias back in as if it were real data, compounding it with no correction mechanism. Fitting a separate model per horizon sidesteps this entirely, at the cost of training 12 models instead of 1.

### 5. Ensemble: combining this repo's own models

SARIMA(1,1,1)(1,1,1)$_{12}$ (Topics 07/11), a Basic Structural Model (local level + seasonal, Topic 12), and the direct Random Forest from Section 4 are combined on the same 7 walk-forward folds:

| Model | Overall MASE |
|---|---|
| Seasonal naive | 1.1187 |
| SARIMA | 0.5537 |
| Basic Structural Model | 0.9053 |
| Direct Random Forest | 0.6594 |
| **Simple average (equal weights)** | **0.5125** |
| Weighted average (sequential inverse-MAE) | 0.5267 |

The **simple equal-weighted average of just 3 models beats every individual model, including the best one (SARIMA)** — and it also slightly beats the more "sophisticated" sequentially-weighted average. The weighting scheme is genuinely causal: weights for fold $i$ use only mean-absolute-error history from folds $0,\dots,i-1$ (equal weights for the very first fold), so this is not a look-ahead artifact.

A Diebold-Mariano test comparing the simple-average ensemble against SARIMA alone (pooling errors across all folds and horizons, $h=12$) gives **stat = -0.6965, p = 0.4881** — not statistically significant. The ensemble is numerically better and directionally consistent with the forecast-combination literature, but 7 walk-forward folds is too small a sample to confidently rule out that SARIMA alone is just as good.

## Common Pitfalls

1. **"Direct avoids compounding errors" is true but incomplete.** Section 2 showed recursive can still beat direct on a correctly specified model, purely from better finite-sample statistical efficiency. Direct's advantage only shows up once misspecification or nonlinearity is severe enough to make error compounding the dominant problem.
2. **The recursive-vs-direct comparison is genuinely mixed under misspecification** (Section 3) — direct won at some horizons and lost at others on identical data. There's no universal rule; test both empirically for any specific problem.
3. **Tree-based models are especially vulnerable to recursive compounding**, because they cannot extrapolate (Topic 13): a recursive forecast that drifts outside the training range gets fed back in with no mechanism to correct it. Direct is the safer default for these model classes.
4. **Simple equal-weight averaging is a genuinely strong baseline, not a naive fallback.** It beat every individual model and a more sophisticated (but still honestly causal) weighting scheme in Section 5 — one of the most replicated findings in the forecasting literature.
5. **A numerically better ensemble is not automatically a statistically proven one.** The DM test found the ensemble's improvement over SARIMA was not significant with only 7 folds — small evaluation samples can't support strong claims even when the point estimate looks convincing.
6. **Combination weights must be estimated causally**, using only past-fold performance. Calibrating weights on the same folds being evaluated would silently leak look-ahead information into the "sophisticated" weighting scheme's apparent advantage.

## Function Reference

| Function | Purpose |
|---|---|
| `simulate_ar2` (from scratch) | Simulate an AR(2) process for Monte Carlo experiments |
| `fit_ols` (from scratch) | OLS fit used for both recursive one-step and direct per-horizon regressions |
| `make_feat_row` (from scratch) | Lag + rolling-mean feature engineering (Topic 13 methodology), reused for RF recursive and direct |
| `RandomForestRegressor` | Recursive and direct multi-step RF forecasting |
| `SARIMAX`, `UnobservedComponents` | SARIMA and Basic Structural Model components of the ensemble |
| `diebold_mariano_test` (from scratch) | Statistical test for whether the ensemble's improvement is significant |

## Self-Test

1. Why does forecast error variance grow with horizon even for a correctly specified, perfectly estimated AR(1) model? What does it converge to as $h\to\infty$, and why?
2. Explain why recursive forecasting beat direct forecasting at 9 of 10 horizons in Section 2, even though both strategies used the *correct* model specification.
3. Under model misspecification (Section 3), why might direct forecasting help at some horizons but not others? What does this imply about picking a strategy without testing it?
4. Why are Random Forests specifically vulnerable to recursive-forecasting error compounding, more so than a linear AR model?
5. What is the "forecast combination puzzle," and what did Section 5 show that reproduces it?
6. Why was it important that the weighted-average ensemble in Section 5 used only *past*-fold errors to compute its weights? What would have gone wrong if it had used all folds, including the one being evaluated?
7. The ensemble beat SARIMA numerically (0.51 vs. 0.55 MASE) but the Diebold-Mariano test found this insignificant (p=0.49). What would you need to do to get a more conclusive answer?

## Notebook

[`20_multi_step_forecasting_strategies_ensemble_forecasting.ipynb`](./20_multi_step_forecasting_strategies_ensemble_forecasting.ipynb)

## Further Reading

- Bates, J.M., Granger, C.W.J. (1969). *The Combination of Forecasts.* Operational Research Quarterly, 20(4), 451-468. (Original forecast combination paper.)
- Clemen, R.T. (1989). *Combining Forecasts: A Review and Annotated Bibliography.* International Journal of Forecasting, 5(4), 559-583. (Documents the forecast combination puzzle.)
- Marcellino, M., Stock, J.H., Watson, M.W. (2006). *A Comparison of Direct and Iterated Multistep AR Methods for Forecasting Macroeconomic Time Series.* Journal of Econometrics, 135(1-2), 499-526.
- Taieb, S.B., Hyndman, R.J. (2012). *Recursive and Direct Multi-Step Forecasting: The Best of Both Worlds.* Monash University working paper. (Proposes hybrid DirRec strategies.)
- Diebold, F.X., Mariano, R.S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 13(3), 253-263.

---
[← Back to Time Series Analysis](../README.md)
