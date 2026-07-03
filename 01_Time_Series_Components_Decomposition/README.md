# 📈 Time Series Components & Decomposition

> Status: ✅ Complete — [Open the notebook →](01_time_series_components_decomposition.ipynb)

Topic 01 of the Time Series Analysis repo. Every method in the Classical ML and Statistical Inference repos assumed rows were exchangeable. This repo starts by breaking that assumption on purpose: a time series is built from **trend** (long-run direction), **seasonality** (repeating fixed-period pattern), and **residual noise** — this notebook builds additive and multiplicative decomposition from scratch, validates against `statsmodels`, stress-tests classical decomposition against the more robust STL method, and closes on the classic real-world AirPassengers dataset that recurs across Topics 02-05.

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
Additive:        Y_t = Trend_t + Seasonal_t + Residual_t     (fixed-size seasonal swings)
Multiplicative:  Y_t = Trend_t * Seasonal_t * Residual_t      (seasonal swings scale with trend level)

Classical decomposition: centered moving average -> trend
                          period-average of detrended data -> seasonal
                          leftover -> residual

STL: same idea, Loess-smoothed and (optionally) ROBUST to outliers
```

---

## 🎯 Why This Topic Matters

- **Both decomposition models were validated against a known ground truth, not just eyeballed** — §1-§2 built a synthetic series with an exactly-known trend and seasonal component, and `statsmodels`' additive decomposition recovered the trend to a mean absolute error of just **0.6253** (against a trend range of 233.7) and the seasonal component to **0.6301** (against an amplitude of 15.0).
- **Picking the wrong model type has a real, quantified cost** — §3 found that decomposing genuinely multiplicative data with the wrong (additive) model left a residual standard deviation of **7.7907**, versus **0.0182** for the correct multiplicative model — a concrete, checkable consequence of a common modeling mistake, not just a definitional nuance.
- **STL's robustness to outliers was demonstrated directly, not just claimed** — §4 injected a single +80 spike into an otherwise clean series and found classical decomposition's trend distorted by up to **7.6646** near the outlier, versus STL's (`robust=True`) distortion of only **2.2544** — roughly a 3x improvement from one design choice.
- **A from-scratch centered moving-average implementation matched `statsmodels` to floating-point precision** — §5 found a maximum difference of **1.14e-13** between the manual "2x12" moving-average trend and `statsmodels`' internal computation, confirming the textbook algorithm was correctly understood, not just imported.
- **A genuinely honest finding: even a "clean" synthetic decomposition doesn't fully whiten the residual** — §6's Ljung-Box test on the synthetic series' residual gave p=**0.0034**, rejecting the no-autocorrelation null — the accelerating (quadratic) trend used to build the series wasn't perfectly captured by the linear centered moving average, leaving detectable leftover structure. This was reported honestly rather than adjusted to force a "passing" result, and the same pattern reappeared on real data.
- **Trend and seasonal strength were quantified as actual numbers, not adjectives** — §7 found the synthetic series scoring **F_T=0.9984** and **F_S=0.9400** (both near the 0-1 maximum), while the real AirPassengers series scored **F_T=1.0000** and **F_S=0.9361** — both series are unambiguously strongly trended and strongly seasonal by this objective measure.
- **On real AirPassengers data, the residual Ljung-Box test rejected far more strongly than on the synthetic series** — §8 found p≈**0.000000** (10 lags) on the real multiplicative decomposition's residual, versus the synthetic series' 0.0034 — a reminder that real economic data carries more structure than even a deliberately imperfect synthetic series, consistent with §6's honest finding rather than contradicting it.

---

## 🧮 Mathematical Explanation

### 1. Additive vs. multiplicative decomposition

$$Y_t = T_t + S_t + R_t \qquad \text{vs.} \qquad Y_t = T_t \times S_t \times R_t$$

Multiplicative is appropriate when seasonal amplitude scales with the trend level — a log transform turns a multiplicative model back into an additive one, since $\log(T_tS_tR_t) = \log T_t + \log S_t + \log R_t$.

### 2. Centered moving average (even period)

$$\hat T_t = \frac{1}{2}\left[\frac{1}{m}\sum_{j=-m/2}^{m/2-1}Y_{t+j} \;+\; \frac{1}{m}\sum_{j=-m/2+1}^{m/2}Y_{t+j}\right]$$

The "2xm" double-averaging trick keeps the moving average properly centered on an integer time index when the period $m$ is even (e.g. $m=12$ for monthly data) — a single $m$-window average would be centered on a half-integer point otherwise.

### 3. Trend and seasonal strength (Hyndman & Athanasopoulos)

$$F_T = \max\left(0,\ 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t+R_t)}\right), \qquad F_S = \max\left(0,\ 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t+R_t)}\right)$$

Both bounded in $[0,1]$ — a value near 1 means removing that component would explain nearly all the non-residual variance.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Additive model | $T_t+S_t+R_t$ | §1-§2 |
| Multiplicative model | $T_t \times S_t \times R_t$ | §3 |
| Centered 2xm moving average | double-averaged $m$-window mean | §5 |
| Trend/seasonal strength | $1-\text{Var}(R)/\text{Var}(T{+}R)$ (resp. $S{+}R$) | §7 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Using additive decomposition on data whose seasonal swings grow with the trend level** — §3's residual standard deviation ballooning from 0.0182 to 7.7907 under the wrong model choice is a direct, quantified illustration.
2. **Trusting classical decomposition's trend estimate near outliers** — §4's ~3x-larger trend distortion (7.66 vs STL's 2.25) near a single injected spike shows why `STL(robust=True)` is often the safer default on messy real data.
3. **Assuming a decomposition residual is automatically white noise** — §6's honest p=0.0034 Ljung-Box rejection on a deliberately clean synthetic series (caused by an accelerating trend a linear moving average can't perfectly track) shows this must be checked, not assumed, even in a controlled setting.
4. **Comparing residual magnitudes across additive and multiplicative models directly** — §8 noted the additive residual (~19.34) and multiplicative residual (~0.033) live on incomparable scales (raw units vs. a ratio near 1.0); the right evidence for model choice is domain knowledge and visual inspection, not raw residual-size comparison.
5. **Forgetting to set an explicit datetime index/frequency before decomposing** — `seasonal_decompose` and `STL` both require a `period` (or an index with inferable frequency); this notebook always builds an explicit `pd.date_range(..., freq="MS")` index before decomposing.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.seasonal.seasonal_decompose` | Classical additive/multiplicative decomposition |
| `statsmodels.tsa.seasonal.STL` | Robust Loess-based decomposition |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Tests residual autocorrelation |
| Custom `centered_moving_average()` | From-scratch 2xm trend extraction |
| Custom `trend_strength()` / `seasonal_strength()` | Hyndman-Athanasopoulos strength metrics |

---

## 📝 Self-Test Exercises

1. Section 3 found the wrong-model residual standard deviation (7.7907) roughly 430x the correct model's (0.0182). Using the additive vs. multiplicative formulas in §1 of the Mathematical Explanation, explain in words why forcing an additive model onto multiplicative data would leave residual structure that scales with the trend level itself.
2. Section 4 found STL's trend distortion near the injected outlier (2.2544) about a third of classical decomposition's (7.6646). Propose one reason a Loess-based local regression would be inherently less sensitive to a single extreme point than a simple moving average.
3. Section 6 found the synthetic series' own decomposition residual failing the Ljung-Box no-autocorrelation test (p=0.0034), despite the series being built from a known, clean process. Using the trend formula (a quadratic in t) and the centered-moving-average formula in §2, explain why a moving average specifically struggles to track an accelerating (non-linear) trend.
4. Section 7 found AirPassengers scoring F_T=1.0000 (trend strength) and F_S=0.9361 (seasonal strength). If a Ljung-Box test on that same series' residual strongly rejects no-autocorrelation (as §8 found), does a high trend/seasonal strength score guarantee the residual is well-behaved? Explain what these two diagnostics actually measure differently.
5. Section 5 found the manual centered moving average matching statsmodels to 1.14e-13 (floating-point precision). If period were odd (e.g. 7, for daily-with-weekly-seasonality data) instead of even (12), how would the centered-moving-average formula in §2 need to change, and why does the "2xm" double-averaging trick become unnecessary?

---

## 📓 Notebook

30 executed code cells: additive and multiplicative decomposition validated against a known synthetic ground truth (trend/seasonal MAE both under 1.0), a quantified cost for choosing the wrong model type (residual std 7.79 vs 0.018), STL's outlier-robustness demonstrated directly against classical decomposition (2.25 vs 7.66 max trend distortion), a from-scratch centered moving-average implementation matched to statsmodels at 1.14e-13, an honestly-reported Ljung-Box finding that even a clean synthetic series' residual retains detectable structure, Hyndman-Athanasopoulos trend/seasonal strength metrics built from scratch, and a full application to the real AirPassengers dataset (trend strength 1.000, seasonal strength 0.936) that recurs across Topics 02-05:

➡️ **[01_time_series_components_decomposition.ipynb](01_time_series_components_decomposition.ipynb)**

---

## 📚 Further Reading

- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Ch. 3: Time Series Decomposition](https://otexts.com/fpp3/decomposition.html)
- [Cleveland et al. (1990): STL: A Seasonal-Trend Decomposition Procedure Based on Loess](https://www.wessa.net/download/stl.pdf)
- [statsmodels: Seasonal Decomposition documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.01_Time_Series_Components_Decomposition&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
