# 🌊 SARIMA

> Status: ✅ Complete — [Open the notebook →](07_sarima.ipynb)

Topic 07 of the Time Series Analysis repo — extending Topic 06's ARIMA with the seasonal machinery Topic 06 honestly proved was missing. This notebook generalizes Topic 06's differencing/integration pair to an arbitrary seasonal lag, simulates and recovers a multiplicative seasonal AR process, builds and honestly orders-selects a full SARIMA(p,d,q)(P,D,Q)$_s$ process, catches two more real `statsmodels` gotchas along the way (a trend-term restriction and a wrong-period silent failure), and closes by fitting the full model to log(AirPassengers) — finally letting the seasonal term Topic 06 was missing do its job, then honestly comparing the result against every prior topic's forecast, including two that still edge it out.

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
SARIMA(p,d,q)(P,D,Q)_s = ARIMA(p,d,q)  *multiplied by*  a seasonal ARIMA(P,D,Q) at lag s

  (1 - phi(L))   (1 - Phi(L^s))   (1-L)^d (1-L^s)^D  Y_t  =  (1 + theta(L)) (1 + Theta(L^s)) e_t
   `-- regular --´ `-- seasonal --´  `-- both integrations --´    `-- regular --´ `-- seasonal --´

d, D chosen via ADF at each differencing combination (Topic 02/06's discipline, generalized)
(p,q,P,Q) chosen via AIC/BIC grid search at fixed (d,D,s), cross-checked against auto_arima

The classic example: monthly data needing one regular AND one seasonal (lag-12)
difference, plus one regular AND one seasonal MA term -- SARIMA(0,1,1)(0,1,1)_12,
the "Box-Jenkins Airline Model," independently rediscovered on real AirPassengers
data in this exact notebook.
```

---

## 🎯 Why This Topic Matters

- **Topic 06's differencing/integration pair was generalized to an arbitrary seasonal lag and proven exactly invertible** — §1 found the from-scratch `manual_seasonal_diff` matching `pandas.diff(12)` at **0.00e+00**, and reconstructing the original series to within **1.78e-15** (floating-point roundoff) even after a double seasonal difference (D=2).
- **A multiplicative seasonal AR process had both its regular and seasonal components recovered from a single fit** — §2 found `phi_hat=0.4955` (true 0.5) and `Phi_hat=0.4104` (true 0.4) from one `ARIMA(1,0,0)(1,0,0)_12` fit, confirming the $(1-\phi L)(1-\Phi L^{12})$ multiplicative structure is estimated correctly, not just additively approximated.
- **d and D were chosen the honest way on a full integrated process — and neither difference alone was enough** — §3 found ADF failing to reject a unit root after *either* a regular difference alone (p=0.8765) *or* a seasonal difference alone (p=0.8578), with only the *combined* d=1,D=1 transform achieving stationarity (p≈0.0000) — correctly recovering the true simulated order without assuming it.
- **AIC didn't crown the true generating order on synthetic data — an honest, expected result, not a bug** — §3 found the true SARIMA(1,1,1)(1,1,1)$_{12}$ order ranked **#6 of 35** candidates by AIC, consistent with Topic 05's own finding that AIC doesn't always recover the exact generating order among several close alternatives.
- **A stricter version of Topic 06's trend-default gotcha was caught directly as a real `ValueError`, not just described** — §5 found `statsmodels` outright refusing a constant trend once `d+D>0`, rather than silently fitting something misleading.
- **A wrong seasonal period was shown fitting without error, but decisively worse on every metric** — §6 found `m=6` (wrong) giving AIC=**-237.38** and forecast MAE=**75.998**, versus `m=12` (correct) at AIC=**-389.01** and MAE=**39.443** — `statsmodels` has no way to know monthly data repeats every 12 steps unless told.
- **A real-data grid search independently agreed with `pmdarima`'s `auto_arima` on the exact classic Box-Jenkins structure** — §7 found both the manual (p,q,P,Q) grid search and `auto_arima` selecting **SARIMA(0,1,1)(0,1,1)$_{12}$** — the textbook "Airline Model" — out of 35 candidates, after excluding 4 that failed to converge.
- **The selected model's forecast dramatically fixed Topic 06's seasonal blindness, but two prior topics' techniques still won on this holdout — an honest, informative result** — §8 found SARIMA's reconstructed 24-month forecast MAE of **39.443** cutting Topic 06's non-seasonal error (81.871) by **52%**, yet still trailing Topic 04's Holt-Winters (28.977) and Topic 05's anchored ARMA reconstruction (17.792) — architectural completeness didn't automatically win this particular accuracy comparison.

---

## 🧮 Mathematical Explanation

### 1. The seasonal differencing operator

$$\nabla_s^D Y_t = (1-L^s)^D Y_t$$

Generalizes Topic 06's $\nabla^d$ to lag $s$; $D=1,s=12$ gives $Y_t - Y_{t-12}$, exactly Topic 02's seasonal-differencing transform.

### 2. The full multiplicative SARIMA equation

$$\phi(L)\Phi(L^s)(1-L)^d(1-L^s)^D Y_t = \theta(L)\Theta(L^s)\varepsilon_t$$

where $\phi(L),\theta(L)$ are the regular AR/MA polynomials (Topic 06) and $\Phi(L^s),\Theta(L^s)$ are their seasonal counterparts, evaluated at lag $s$ and *multiplied* — not added — into the regular polynomials.

### 3. Why multiplication, not addition

$$(1-\phi L)(1-\Phi L^{12}) = 1 - \phi L - \Phi L^{12} + \phi\Phi L^{13}$$

Expanding the product introduces a cross term at lag $p+s$ (here, lag 13) — a real interaction between the regular and seasonal dynamics that a naively *additive* model ($1-\phi L-\Phi L^{12}$, no cross term) would miss entirely.

### 4. Seasonal integration (forecast reconstruction)

$$\hat Y_{t+h} = Y_{t+h-s} + \hat W_{t+h} \qquad (D=1 \text{ case})$$

Undoing one seasonal difference needs the value exactly $s$ steps back, not one step back — §1's `manual_seasonal_integrate` needs $s$ seed values per differencing order, generalizing Topic 06's single-seed regular case.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Seasonal differencing | $\nabla_s^D Y_t=(1-L^s)^DY_t$ | §1 |
| Full SARIMA(p,d,q)(P,D,Q)$_s$ | $\phi(L)\Phi(L^s)(1-L)^d(1-L^s)^DY_t=\theta(L)\Theta(L^s)\varepsilon_t$ | §2-§3 |
| Multiplicative cross term | $(1-\phi L)(1-\Phi L^s)=1-\phi L-\Phi L^s+\phi\Phi L^{p+s}$ | §2 |
| Seasonal integration | $\hat Y_{t+h}=Y_{t+h-s}+\hat W_{t+h}$ | §1, §7-§8 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Testing only a regular OR only a seasonal difference and concluding neither is needed** — §3 found *both* individually insufficient (ADF p=0.88 and p=0.86) on a series that genuinely needed both; always test every relevant combination, not just each transform in isolation.
2. **Requesting a constant trend once `d+D>0`** — §5 found `statsmodels` raising a `ValueError` outright (a stricter safeguard than Topic 06's silently-different-default gotcha) — read the error message; it explains exactly which trend orders remain valid.
3. **Assuming a wrong seasonal period will "just fail" loudly** — §6 found `m=6` fitting without any error at all, just a decisively worse AIC and forecast; a period mismatch is a silent, not a loud, mistake.
4. **Trusting a grid search's lowest-AIC row without checking convergence** — §7 found 4 of 35 real-data candidates failing to converge; excluding them before selection (per Topic 06's precedent) is what actually produced the trustworthy classic Airline-model result.
5. **Assuming a more complete model architecture automatically forecasts better** — §8's SARIMA fixed Topic 06's seasonal blindness but still lost to Topics 04-05's simpler, anchored techniques on this specific holdout; a model with more correctly-specified structure is not the same guarantee as a model with lower forecast error on every dataset and horizon.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.arima.model.ARIMA` (with `seasonal_order`) | Full MLE fitting for SARIMA(p,d,q)(P,D,Q)$_s$ |
| `statsmodels.tsa.stattools.adfuller` | ADF unit-root test, applied at every (d,D) combination |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Residual white-noise check across multiple lags |
| `pmdarima.auto_arima` (`seasonal=True, m=12`) | Independent stepwise (p,d,q)(P,D,Q) search |
| Custom `manual_seasonal_diff()` / `manual_seasonal_integrate()` | From-scratch, exactly-invertible seasonal differencing and its inverse |

---

## 📝 Self-Test Exercises

1. Section 1 generalized Topic 06's `manual_integrate` (one seed value) to `manual_seasonal_integrate` (s seed values). Using the seasonal integration formula in §4 of the Mathematical Explanation, explain why undoing a lag-12 seasonal difference needs 12 seed values instead of 1, while undoing a regular (lag-1) difference only ever needs 1.
2. Section 3 found ADF failing to reject a unit root after *either* a regular difference alone (p=0.8765) *or* a seasonal difference alone (p=0.8578) on the same series. Propose a series where only a regular difference would be needed and a series where only a seasonal difference would be needed, and explain what feature of each series would produce that result.
3. Section 5's `ValueError` explains that a constant is only valid when the model's trend order exceeds `d+D`. Using the multiplicative equation in §2, explain in your own words why integrating a series with `d=1` or `D=1` at all already removes the information a constant term would otherwise capture.
4. Section 6 found `m=6` fitting without any error, just worse AIC and forecast accuracy. Propose one situation where an incorrect `m` might still produce a *deceptively* reasonable-looking AIC (hint: think about what happens if the wrong period happens to be a divisor or multiple of the true one).
5. Section 8 found SARIMA(0,1,1)(0,1,1)$_{12}$'s forecast MAE (39.443) worse than both Topic 04's Holt-Winters (28.977) and Topic 05's anchored ARMA reconstruction (17.792), despite SARIMA being architecturally the most complete model in the series so far. Using the notebook's own explanation of Topic 05's "anchoring" technique, propose a modification to how SARIMA's forecast is reconstructed here that might close some of that gap, without changing the fitted model itself.

---

## 📓 Notebook

25 executed code cells: a from-scratch, exactly-invertible seasonal differencing/integration pair generalizing Topic 06's regular-lag version to arbitrary lag `s`; a multiplicative seasonal AR(1)x(1)$_{12}$ process with both components recovered from a single fit; a full SARIMA(1,1,1)(1,1,1)$_{12}$ process with `d` and `D` chosen via ADF at every combination (not assumed), and a (p,q,P,Q) grid search cross-checked against `pmdarima`'s independent stepwise search while honestly reporting AIC didn't rank the true order first; two new real `statsmodels` gotchas caught directly (a stricter trend-term `ValueError` once `d+D>0`, and a silently-fittable-but-decisively-worse wrong seasonal period); and a full real-data application that independently rediscovers the classic Box-Jenkins Airline Model — SARIMA(0,1,1)(0,1,1)$_{12}$ — via both a manual grid search and `auto_arima`, checks its residuals and forecast confidence interval, and honestly reports its forecast beating Topic 06 by 52% while still trailing Topics 04-05 on this specific 24-month holdout:

➡️ **[07_sarima.ipynb](07_sarima.ipynb)**

---

## 📚 Further Reading

- [Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)
- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — Seasonal ARIMA models](https://otexts.com/fpp3/seasonal-arima.html)
- [statsmodels: ARIMA documentation (seasonal_order)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [pmdarima: auto_arima documentation](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.07_SARIMA&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
