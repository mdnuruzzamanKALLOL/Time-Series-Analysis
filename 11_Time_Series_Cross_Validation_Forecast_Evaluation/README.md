# ✅ Time Series Cross-Validation & Forecast Evaluation

> Status: ✅ Complete — [Open the notebook →](11_time_series_cross_validation_forecast_evaluation.ipynb)

Topic 11 of the Time Series Analysis repo — the discipline topic the repo's own introduction promised: "the Model Evaluation & Tuning topic's `TimeSeriesSplit` warning gets its full treatment here." Every model in Topics 05-10 was validated with a single train/test split; this notebook builds proper walk-forward cross-validation from scratch, matches it exactly to `sklearn.model_selection.TimeSeriesSplit`, demonstrates concretely why shuffled k-fold CV gives a misleading picture of forecast difficulty, builds a full suite of forecast evaluation metrics (MAE, RMSE, MAPE, sMAPE, MASE) and the Diebold-Mariano test for statistically comparing two forecasts, then applies all of it to a genuine multi-fold walk-forward evaluation of SARIMA on AirPassengers — finding that a single train/test split (the evaluation style used throughout Topics 05-10) can hide a large amount of fold-to-fold variation in real forecast difficulty.

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
Classical ML CV:  shuffle rows -> k-fold -> every fold's train set surrounds its test set in time
                  (fine for i.i.d. rows, WRONG for time-ordered rows)

Time series CV:   walk forward only -> every fold's train set is entirely BEFORE its test set
                  (the only way to simulate what a real forecaster actually knows at each point)

Expanding window: train = [0 ......................... t]   test = [t+1 .. t+h]   (grows every fold)
Rolling window:   train =     [t-w ............. t]         test = [t+1 .. t+h]   (fixed size, slides)
```

---

## 🎯 Why This Topic Matters

- **Shuffled k-fold CV was shown hiding a genuinely escalating forecast-difficulty trend, while walk-forward CV exposed it directly** — §1 found a series with real, growing volatility giving flat, uninformative shuffled-k-fold errors (**0.86–0.96** across all 5 folds) while walk-forward CV's per-fold errors climbed steadily (**0.59 → 1.64**), with the walk-forward last fold alone (relative error **22.3%**) far closer to the true unseen-future error than either method's overall average (relative errors **52.7%** and **57.5%**).
- **A from-scratch expanding-window splitter matched `sklearn.model_selection.TimeSeriesSplit` exactly on every fold** — §2 found identical train/test index arrays across all 5 folds, byte-for-byte.
- **A real, concrete feature-leakage bug was demonstrated, not just described** — §3 found a *centered* rolling-window feature (computed on the full series before splitting — an easy, common mistake) giving a test MAE **25.5% better** than the honest, causally-computed version, purely through boundary leakage rather than genuine skill.
- **MASE gave an absolute, scale-independent reference point that MAE/RMSE alone cannot** — §4 found two synthetic models both scoring MASE **> 1** (2.22 and 5.36), immediately flagging both as worse than a naive one-step forecast — information invisible in their raw MAE/RMSE numbers alone.
- **The Diebold-Mariano test, built from scratch, correctly distinguished a genuinely-better forecast from two statistically-identical ones** — §5 found p≈**0.000000** (correctly rejecting equal accuracy) when one model was truly more accurate, and p=**0.6287** (correctly failing to reject) when two models had identical error distributions.
- **A full walk-forward re-evaluation of SARIMA on real AirPassengers data revealed a 4.4x spread in fold-level accuracy purely from which 12 months were held out** — §6 found per-fold MAE ranging from **7.05 to 31.27** across four expanding-window folds — variability a single train/test split (the style used throughout Topics 05-10) can never reveal.
- **A textbook-lower MAE did not automatically mean a statistically proven difference** — §7 found the Diebold-Mariano test's conclusion flipping from highly significant (p=0.0001 at horizon h=1) to not significant at the 5% level (p=0.1392 at the methodologically correct h=12) purely from correctly accounting for within-fold forecast-error autocorrelation, with only 48 pooled observations.

---

## 🧮 Mathematical Explanation

### 1. Expanding-window (walk-forward) split

$$\text{Fold }k:\quad \text{train}=\{1,\dots,t_k\}, \quad \text{test}=\{t_k+1,\dots,t_k+h\}, \quad t_1<t_2<\dots<t_K$$

Every fold's training set is a strict prefix of history relative to its test set — the only split structure that mirrors what a real forecaster actually knows at each point in time.

### 2. Forecast evaluation metrics

$$\text{MAE}=\frac1n\sum|y_t-\hat y_t|, \qquad \text{RMSE}=\sqrt{\frac1n\sum(y_t-\hat y_t)^2}$$

$$\text{MAPE}=\frac{100}{n}\sum\left|\frac{y_t-\hat y_t}{y_t}\right|, \qquad \text{sMAPE}=\frac{100}{n}\sum\frac{2|y_t-\hat y_t|}{|y_t|+|\hat y_t|}$$

$$\text{MASE}=\frac{\frac1n\sum|y_t-\hat y_t|}{\frac1{n-m}\sum_{i=m+1}^{n}|y_i-y_{i-m}|}$$

MASE's denominator is the in-sample naive (lag-$m$) forecast's own MAE — the only metric here with a universal, scale-independent reference point: **MASE > 1 always means "worse than naive," regardless of the series' units.**

### 3. The Diebold-Mariano test

$$d_t = L(e_{1,t}) - L(e_{2,t}), \qquad DM=\frac{\bar d}{\sqrt{\widehat{\text{Var}}(\bar d)}}\sim t_{n-1} \text{ (Harvey-Leybourne-Newbold corrected)}$$

with $\widehat{\text{Var}}(\bar d)$ estimated via a Newey-West long-run variance using $h-1$ lags, where $h$ is the forecast horizon — accounting for the fact that multi-step-ahead forecast errors are themselves autocorrelated, unlike i.i.d. residuals.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Expanding-window split | train=$\{1,\dots,t_k\}$, test=$\{t_k+1,\dots,t_k+h\}$ | §1, §2, §6 |
| MASE | $\text{MAE}/\text{MAE}_{\text{naive, in-sample}}$ | §4, §6 |
| Diebold-Mariano statistic | $DM=\bar d/\sqrt{\widehat{\text{Var}}(\bar d)}$, Newey-West with $h-1$ lags | §5, §7 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Using shuffled `KFold` on time-ordered data** — §1 found it producing flat, uninformative per-fold errors (~0.86-0.96) that completely hid a genuinely escalating difficulty trend that walk-forward CV exposed directly (0.59 → 1.64 across folds).
2. **Computing rolling/window features on the full series before splitting** — §3 found a centered rolling-window feature giving an artificially better test MAE (0.1999 vs. the honest 0.2684) purely through boundary leakage, not genuine skill.
3. **Trusting a single train/test split's error as "the" expected accuracy** — §6 found SARIMA's walk-forward fold MAEs ranging from 7.05 to 31.27 (a 4.4x spread) purely depending on which 12 months were held out; a single split (as every prior topic in this repo used) reports only one point from that range as if it were the whole story.
4. **Comparing models with MAE/RMSE alone across series of different scales** — §4 showed MASE giving an absolute, interpretable reference point (>1 means worse than a naive lag forecast) that raw MAE/RMSE cannot provide.
5. **Reading a lower MAE as automatic proof of a better model** — §7 found the DM test's conclusion flipping from highly significant to not-significant depending on the (methodologically meaningful) choice of `h`, with only 48 pooled observations; a point-estimate difference is not the same as a demonstrated statistical difference.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.model_selection.TimeSeriesSplit` | Library expanding-window walk-forward CV splitter |
| `sklearn.model_selection.KFold(shuffle=True)` | The (wrong, for time series) shuffled baseline used for contrast |
| `sklearn.metrics.mean_absolute_error`, `mean_squared_error` | Library MAE/RMSE, matched against the from-scratch versions |
| `statsmodels.tsa.statespace.sarimax.SARIMAX` | Real SARIMA fitting for the walk-forward AirPassengers application |
| Custom `manual_expanding_window_split()` / `manual_rolling_window_split()` | From-scratch walk-forward splitters |
| Custom `mase_fn()`, `smape_fn()`, `mape_fn()` | From-scratch scale-independent forecast metrics |
| Custom `diebold_mariano_test()` | From-scratch statistical forecast-comparison test |

---

## 📝 Self-Test Exercises

1. Section 1 found shuffled K-Fold's per-fold errors staying flat (~0.86-0.96) even though the underlying series' volatility was clearly growing over time. Using the expanding-window formula in §1 of the Mathematical Explanation, explain precisely why a shuffled fold's training set can never reveal this kind of time-dependent difficulty the way a walk-forward fold's training set can.
2. Section 3 found a centered rolling-window feature producing a 25.5% "improvement" that was actually leakage. Propose a modification to the rolling-window CV splitters from Section 2 (a `gap` parameter) that would protect against this specific leakage even if a centered window were used by mistake, and explain why it only partially fixes the problem.
3. Section 4 found both synthetic models scoring MASE > 1 on a strongly autocorrelated series. Using the MASE formula in §2 of the Mathematical Explanation, explain why a random-walk-like series makes it especially hard for ANY model to beat the naive benchmark, regardless of how sophisticated the model is.
4. Section 6 found SARIMA's fold-level MAE varying from 7.05 to 31.27 depending purely on which 12 months were tested. Looking at the fold test periods in the notebook, propose a real-world explanation (something about the AirPassengers series itself) for why some 12-month windows were so much harder to forecast than others.
5. Section 7 found the Diebold-Mariano test's conclusion changing from "reject H0" to "fail to reject H0" purely by changing `h` from 1 to 12. Explain why `h=1` is the methodologically WRONG choice here (hint: think about what generated the 48 pooled errors being tested), even though it produces the more "exciting" (significant) result.

---

## 📓 Notebook

16 executed code cells: shuffled k-fold CV shown hiding a genuinely escalating real forecast-difficulty trend that walk-forward CV exposes directly; a from-scratch expanding-window splitter matched exactly to `sklearn.model_selection.TimeSeriesSplit` on every fold, alongside a from-scratch rolling-window alternative; a real, concrete feature-leakage bug (a centered rolling-window feature) demonstrated and fixed; a full forecast-metric suite (MAE, RMSE, MAPE, sMAPE, MASE) built from scratch and matched to `sklearn` where applicable; a from-scratch, Newey-West-corrected Diebold-Mariano test validated on controlled synthetic cases; and a genuine multi-fold walk-forward evaluation of SARIMA on real AirPassengers data revealing a 4.4x fold-to-fold accuracy spread and a forecast-horizon-sensitive statistical significance test against a naive-seasonal benchmark:

➡️ **[11_time_series_cross_validation_forecast_evaluation.ipynb](11_time_series_cross_validation_forecast_evaluation.ipynb)**

---

## 📚 Further Reading

- [Diebold & Mariano (1995): Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599)
- [Harvey, Leybourne & Newbold (1997): Testing the Equality of Prediction Mean Squared Errors](https://doi.org/10.1016/S0169-2070(96)00719-4)
- [Hyndman & Koehler (2006): Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001) (the original MASE paper)
- [`sklearn.model_selection.TimeSeriesSplit` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.11_Time_Series_Cross_Validation_Forecast_Evaluation&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
