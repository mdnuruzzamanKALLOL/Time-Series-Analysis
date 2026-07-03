# 📊 Moving Averages & Exponential Smoothing

> Status: ✅ Complete — [Open the notebook →](04_moving_averages_exponential_smoothing.ipynb)

Topic 04 of the Time Series Analysis repo. Before the AR/MA/ARMA model family (Topic 05), the simplest useful forecasting tools are moving averages and exponential smoothing — this notebook builds Simple Moving Average, Simple Exponential Smoothing (SES), Holt's linear (double) method, and Holt-Winters (triple) seasonal smoothing entirely from scratch, matches all three recursions to `statsmodels` at floating-point precision, and closes with a real forecasting bake-off on AirPassengers against a naive persistence baseline.

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
SMA:            average of the last k observations           -- no trend/seasonal awareness
SES (single):   l_t                                           -- level only
Holt (double):  l_t + b_t                                     -- level + trend
Holt-Winters (triple): l_t + b_t + s_t                        -- level + trend + seasonality

Each adds one more smoothed component on top of the last.
```

---

## 🎯 Why This Topic Matters

- **All three exponential smoothing recursions were matched to `statsmodels` at exact floating-point precision, not approximately** — §2, §4, and §5 found maximum differences of **0.00e+00** (SES), **0.00e+00** (Holt, both level and trend), and **1.42e-14 / 1.78e-15 / 5.33e-15** (Holt-Winters level/trend/seasonal) — including a genuine forecast-value match for Holt at **0.00e+00**.
- **Getting Holt-Winters to match exactly required reading `statsmodels`' actual source code, not just the textbook formula** — §5 discovered that `statsmodels`' seasonal update term uses the **previous** level+trend forecast ($l_{t-1}+b_{t-1}$), not the newly-updated level $l_t$ — a genuine implementation detail invisible from the standard textbook equations alone, and the fix that took the match from a 0.37 discrepancy down to 5.33e-15.
- **A manual alpha grid search cross-validated closely against `statsmodels`' continuous optimizer** — §3 found the grid search's best alpha at **0.95** versus `statsmodels`' optimizer at **0.9458** — independent confirmation from a cruder (0.01-step) method.
- **SMA's bias-variance tradeoff was shown directly with real in-sample numbers** — §1 found SMA(5)'s in-sample MAE at **0.3071** versus SMA(20)'s **0.5507** on a slowly-drifting series — the shorter window tracks genuine level movement more closely, at the cost of reacting more to noise.
- **A genuinely surprising, honestly-reported real-data finding: SES's optimizer chose alpha=1.0000, making it numerically identical to the naive baseline** — §8 found SES's optimized $\alpha$ on the AirPassengers training data landing at exactly **1.0000**, producing an MAE of **115.250** — identical to the naive forecast's MAE, because SES with $\alpha=1$ *is* the naive forecast; without a trend term, weighting entirely on the most recent point minimized in-sample one-step-ahead squared error on this strongly-trending series.
- **Holt's optimizer also found beta=0.0000 on the same data — a real, if counter-intuitive, result** — §8 found Holt's optimized trend-smoothing parameter at exactly **0.0000** despite the series' obvious upward trend, meaning the trend component's smoothing collapsed to a constant rather than adapting — Holt still beat SES/naive (MAE 91.616 vs 115.250) purely from its different optimized alpha, not from genuinely tracking a changing trend rate.
- **Holt-Winters, using multiplicative seasonality per Topic 01's finding, was the clear winner on genuinely unseen test data** — §8 found Holt-Winters at MAE=**28.977**, RMSE=**32.489**, MAPE=**6.391%** — a **74.9%** MAE improvement over the naive baseline, the first method in this section to meaningfully outperform the trivial "tomorrow = today" comparison.

---

## 🧮 Mathematical Explanation

### 1. Simple Moving Average

$$\hat Y_{t+1} = \frac{1}{k}\sum_{i=0}^{k-1}Y_{t-i}$$

### 2. Simple Exponential Smoothing

$$l_t = \alpha Y_t + (1-\alpha)l_{t-1}$$

### 3. Holt's linear method

$$l_t = \alpha Y_t + (1-\alpha)(l_{t-1}+b_{t-1}), \qquad b_t = \beta(l_t-l_{t-1})+(1-\beta)b_{t-1}$$

$h$-step forecast: $l_t + h\,b_t$ — a straight line extrapolated from the current level and trend.

### 4. Holt-Winters (additive seasonal)

$$l_t=\alpha(Y_t-s_{t-m})+(1-\alpha)(l_{t-1}+b_{t-1}), \quad b_t=\beta(l_t-l_{t-1})+(1-\beta)b_{t-1}, \quad s_t=\gamma\left(Y_t-(l_{t-1}+b_{t-1})\right)+(1-\gamma)s_{t-m}$$

The seasonal term $s_t$ crucially uses the *old* forecast $l_{t-1}+b_{t-1}$, not the just-updated $l_t$ — the exact detail that had to be reverse-engineered from `statsmodels`' source to achieve an exact match in §5.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| SMA | $\frac1k\sum_{i=0}^{k-1}Y_{t-i}$ | §1 |
| SES | $\alpha Y_t+(1-\alpha)l_{t-1}$ | §2-§3 |
| Holt level/trend | $\alpha Y_t+(1-\alpha)(l_{t-1}+b_{t-1})$; $\beta(l_t-l_{t-1})+(1-\beta)b_{t-1}$ | §4 |
| Holt-Winters seasonal | $\gamma(Y_t-(l_{t-1}+b_{t-1}))+(1-\gamma)s_{t-m}$ | §5 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Assuming the textbook Holt-Winters formula is exactly what any given library implements** — §5's discovery (old forecast, not new level, in the seasonal update) shows a "textbook-equivalent" formula can still differ in a specific term; always verify against source or documentation before assuming numerical equivalence.
2. **Trusting an optimizer's chosen smoothing parameter without checking what it implies** — §8's alpha=1.0000 for SES and beta=0.0000 for Holt are real optimizer outputs, not bugs, but both mean those components contributed nothing beyond the previous simplest model; always compare against baselines (as done here) rather than assuming a fancier-named method is automatically better.
3. **Using additive seasonality on data with growing seasonal amplitude** — Topic 01 already established AirPassengers' seasonal swings grow with the trend level; §8 correctly used `seasonal="mul"` for the real-data Holt-Winters fit, consistent with that earlier finding.
4. **Comparing smoothing methods only in-sample** — §8's train/test split is essential; SES's in-sample fit can look reasonable while its test-set MAE ties the naive baseline exactly, a distinction only a genuine holdout period reveals.
5. **Choosing SMA's window size without checking the tradeoff directly** — §1's MAE comparison (0.3071 vs 0.5507) is a concrete, quantified version of the general "shorter window = more responsive but noisier" rule of thumb.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.holtwinters.SimpleExpSmoothing` | Library SES implementation |
| `statsmodels.tsa.holtwinters.Holt` | Library double exponential smoothing |
| `statsmodels.tsa.holtwinters.ExponentialSmoothing` | Library Holt-Winters (triple) smoothing |
| Custom `manual_ses()`, `manual_holt()`, `manual_holt_winters_additive()` | From-scratch recursions |
| Custom `mae()`, `rmse()`, `mape()` | From-scratch forecast evaluation metrics |

---

## 📝 Self-Test Exercises

1. Section 5 found that matching Holt-Winters exactly required using the *previous* level+trend forecast in the seasonal update, not the new level. Using the formula in §4 of the Mathematical Explanation, explain what would go wrong (conceptually) if the seasonal term used the newly-updated level $l_t$ instead — what information would be "leaking" into the seasonal estimate?
2. Section 8 found SES's optimized alpha at exactly 1.0000 on the AirPassengers training data. Using the SES formula in §2, explain algebraically why alpha=1 makes SES's forecast identical to the naive persistence forecast.
3. Section 8 found Holt's optimized beta at exactly 0.0000, yet Holt still beat SES and naive on MAE. Using the Holt formulas in §3, explain what beta=0 implies about the trend component $b_t$ over time, and why Holt could still outperform SES despite this.
4. Section 1 found SMA(5)'s in-sample MAE (0.3071) lower than SMA(20)'s (0.5507) on a slowly-drifting series. Would you expect this ranking to reverse on a series with a constant level and large random noise (no drift)? Explain using the bias-variance framing in §1's discussion.
5. Section 8 found Holt-Winters improving MAE by 74.9% over the naive baseline, while SES and Holt showed much smaller (or zero) improvement. Given AirPassengers' known strong seasonality (Topics 01 and 03), which of Holt-Winters' three components (level, trend, or seasonal) do you think contributed most to this gap, and how could you test that hypothesis using the tools built in this notebook?

---

## 📓 Notebook

32 executed code cells: SMA's bias-variance tradeoff quantified directly (MAE 0.307 vs 0.551), SES, Holt, and Holt-Winters all built from scratch and matched to `statsmodels` at floating-point precision (including a genuine source-code-level discovery about Holt-Winters' seasonal update formula), a manual alpha grid search cross-validated against `statsmodels`' optimizer, and a full real-data forecasting bake-off on AirPassengers comparing naive, SMA, SES, Holt, and Holt-Winters against a 24-month genuinely-unseen holdout — honestly reporting that SES and Holt's optimizers found degenerate parameters (alpha=1, beta=0) on this data, while Holt-Winters delivered a real 74.9% MAE improvement over the naive baseline:

➡️ **[04_moving_averages_exponential_smoothing.ipynb](04_moving_averages_exponential_smoothing.ipynb)**

---

## 📚 Further Reading

- [Holt (1957, reprinted 2004): Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages](https://www.sciencedirect.com/science/article/pii/S0169207003001134)
- [Winters (1960): Forecasting Sales by Exponentially Weighted Moving Averages](https://www.jstor.org/stable/2627346)
- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Ch. 8: Exponential Smoothing](https://otexts.com/fpp3/expsmooth.html)
- [statsmodels: Exponential Smoothing documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.04_Moving_Averages_Exponential_Smoothing&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
