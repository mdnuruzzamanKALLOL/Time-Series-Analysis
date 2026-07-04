# 🧩 ARIMA

> Status: ✅ Complete — [Open the notebook →](06_arima.ipynb)

Topic 06 of the Time Series Analysis repo — combining Topic 02's differencing, Topic 03's autocorrelation diagnostics, and Topic 05's AR/MA/ARMA estimation into a single model class. This notebook builds a from-scratch differencing/integration pair and proves it is exactly what `ARIMA`'s `d` parameter does internally (catching a real default-trend gotcha along the way), simulates and recovers a true ARIMA(1,1,1) process, chooses the differencing order `d` honestly via ADF rather than by assumption, cross-checks a manual (p,q) grid search against `pmdarima`'s `auto_arima`, demonstrates why AIC cannot be compared across different values of `d`, adds forecast confidence intervals (a capability Topic 05 didn't use), and closes by fitting ARIMA directly to log(AirPassengers) — without Topic 02's seasonal differencing — to honestly show that clean residual diagnostics don't guarantee a good forecast when real seasonal structure goes unmodeled.

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
ARIMA(p,d,q) = Integrated(d) + ARMA(p,q)

  1. Difference the series d times until it's stationary (Topic 02's ADF/KPSS decide d)
  2. Fit ARMA(p,q) on the differenced series (Topic 05's estimation machinery)
  3. Forecast on the differenced scale, then integrate (cumulatively sum) d times
     back to the original scale

d is a PREPROCESSING STEP baked into one model class, not new math --
ARIMA(p,d,q) fit on raw Y_t is mechanically identical to ARMA(p,q) fit on
diff(Y_t, d), once the trend term is matched between the two.
```

---

## 🎯 Why This Topic Matters

- **A from-scratch differencing/integration pair was built and proven exactly invertible** — §1 found `manual_diff` matching `np.diff` at **0.00e+00** for both d=1 and d=2, and reconstructing the original random walk from either differenced series to within **4.44e-15** (floating-point roundoff, not a real error).
- **`ARIMA`'s `d` parameter was proven mechanically identical to manual pre-differencing — but only after catching a real gotcha** — §2 first found ARIMA(1,1,1) fit directly (phi=0.6480, theta=-0.3508) disagreeing with ARMA(1,1) fit on the manually-differenced series (phi=0.6376, theta=-0.3426) by a suspicious 0.0175 — traced to `ARIMA(p,0,q)` defaulting to a constant trend while `ARIMA(p,d,q)` with d>0 defaults to none. Matching the trend explicitly (`trend="n"`) brought the parameter difference down to **1.41e-07**, confirming true mechanical equivalence.
- **d was chosen the honest way — from an ADF test on the data — not copied from the simulation's known ground truth** — §3 found ADF p-values of **0.9653** (d=0, non-stationary) collapsing to **~0.0000** at d=1, correctly recovering the true simulated integration order without peeking at how the series was generated.
- **Over-differencing was shown directly inflating variance, not just asserted as a rule** — §3 found variance at d=2 (**1.4892**) exceeding d=1 (**1.1238**) on the same series, the concrete cost of differencing past the point stationarity was already achieved.
- **An exhaustive manual grid search and `pmdarima`'s independent stepwise `auto_arima` agreed on the same order from two different search strategies** — §4 found both selecting **(1,1,1)** (manual AIC=2823.37, auto_arima AIC=2822.74) out of 15 candidates, with the fitted model's AR/MA polynomial roots (moduli **1.543** and **2.851**) directly confirming stationarity and invertibility rather than assuming `statsmodels` got it right.
- **A genuine, easy-to-miss pitfall was demonstrated with real numbers: AIC is not comparable across different `d`** — §5 found `ARIMA(1,0,1)` on the raw series (AIC=2882.31) and `ARIMA(1,1,1)` on the same series (AIC=2823.37) both reporting `nobs=1000`, yet the underlying likelihoods are computed on different data — the lower AIC does *not* mean "better model" the way a fixed-d comparison validly would.
- **Forecast confidence intervals — a first for this series — were checked for honest calibration, not just plotted** — §6 found **100%** of a 24-step synthetic holdout falling inside the reported 95% interval.
- **A non-converged candidate was caught mid-grid-search and excluded before model selection, exactly the discipline Topic 05's pitfalls called for** — §7 found 1 of 15 real-data candidates (the AIC-"best"-looking (3,1,3)) failing to converge, and excluded it in favor of the best *converged* order, (2,1,1).
- **The selected real-data model passed every Ljung-Box check cleanly, yet its forecast badly underperformed — an honest, important finding, not a bug to hide** — §7 found Ljung-Box p-values of **0.98–1.00** at every lag up to 24 (clean one-step-ahead residuals), while the reconstructed 24-month forecast MAE of **81.871** was **2.8x worse** than Topic 04's Holt-Winters (28.977) and **4.6x worse** than Topic 05's ARMA-on-seasonally-differenced model (17.792) — a non-seasonal model simply has no term to reproduce a 12-month cycle, no matter how clean its residuals look.

---

## 🧮 Mathematical Explanation

### 1. The integrated (I) component

$$\nabla^d Y_t = (1-L)^d Y_t$$

where $L$ is the lag operator ($LY_t = Y_{t-1}$). $d=1$ gives the familiar $Y_t - Y_{t-1}$; $d=2$ differences that result again.

### 2. ARIMA(p,d,q) as a single equation

$$\phi(L)(1-L)^d Y_t = \theta(L)\varepsilon_t, \qquad \phi(L) = 1-\phi_1L-\dots-\phi_pL^p, \quad \theta(L)=1+\theta_1L+\dots+\theta_qL^q$$

Fitting this is exactly Topic 05's ARMA machinery applied to $W_t = \nabla^d Y_t$, then un-differencing the forecast of $W_t$ back to $Y_t$'s scale.

### 3. Integration (the forecast-reconstruction direction)

$$\hat Y_{t+h} = Y_t + \sum_{i=1}^{h}\hat W_{t+i} \qquad (d=1 \text{ case})$$

Each differencing step needs exactly one "seed" value from the previous, less-differenced level to invert; $d$ differences need $d$ seed values, applied in reverse order.

### 4. Why AIC needs a fixed d

$$AIC = -2\ln\hat L(\text{model} \mid \text{data}) + 2k$$

$\hat L$ is a likelihood over whichever series was actually fit. `statsmodels` reports `nobs` after accounting for the state-space representation, but the *information content* differencing removes (long-run level, trend) is not directly comparable across different $d$ — only the $(p,q)$ dimension can be safely swept at a fixed, pre-chosen $d$.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Differencing operator | $\nabla^d Y_t=(1-L)^dY_t$ | §1 |
| ARIMA(p,d,q) | $\phi(L)(1-L)^dY_t=\theta(L)\varepsilon_t$ | §2 |
| Integration (forecast reconstruction) | $\hat Y_{t+h}=Y_t+\sum_{i=1}^h \hat W_{t+i}$ | §1, §6-§7 |
| AIC | $-2\ln\hat L+2k$ | §4-§5, §7 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Assuming `ARIMA(p,0,q)` and `ARIMA(p,d,q)` (d>0) share the same default trend** — §2 found a real 0.0175 parameter gap purely from `ARIMA(p,0,q)` defaulting to a constant while `ARIMA(p,d,q)` with d>0 defaults to none; always set `trend` explicitly when comparing the two forms directly.
2. **Choosing d from what you already know about the data instead of testing for it** — §3 deliberately re-derived d=1 via ADF on a series whose true order was already known from the simulation, exactly the discipline needed on real data where the true order is never known.
3. **Comparing AIC/BIC across different values of d** — §5's `ARIMA(1,0,1)` (AIC=2882.31) vs `ARIMA(1,1,1)` (AIC=2823.37) both report `nobs=1000`, but the comparison is still invalid; fix d with a stationarity test first, then sweep (p,q).
4. **Trusting the "best AIC" candidate in a grid search without checking convergence** — §7's (3,1,3) had the lowest AIC of all 15 real-data candidates but failed to converge; it was excluded in favor of the best *converged* order, (2,1,1), a strictly weaker-looking but trustworthy result.
5. **Treating a clean Ljung-Box residual check as proof of a good forecast** — §7's (2,1,1) model passed Ljung-Box at every lag up to 24 (p≥0.98) yet forecast 2.8-4.6x worse than Topics 04-05's seasonally-aware models; Ljung-Box only confirms *one-step-ahead* residuals look uncorrelated, not that a *multi-step* forecast captures structure (like seasonality) the model has no term for.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.arima.model.ARIMA` | Full MLE fitting for ARIMA(p,d,q), with `.arroots`/`.maroots` and `.get_forecast()` |
| `statsmodels.tsa.stattools.adfuller` | ADF unit-root test (Topic 02, reused to choose d) |
| `statsmodels.tsa.stattools.kpss` | KPSS complementary-null test (Topic 02, reused) |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Residual white-noise check across multiple lags |
| `pmdarima.auto_arima` | Independent stepwise (p,d,q) search, cross-checked against the manual grid |
| Custom `manual_diff()` / `manual_integrate()` | From-scratch, exactly-invertible differencing and its inverse |

---

## 📝 Self-Test Exercises

1. Section 2 found ARIMA(1,1,1) fit directly and ARMA(1,1) fit on the manually-differenced series disagreeing by 0.0175 *before* the trend term was matched, then agreeing to 1.41e-07 *after*. Using the ARIMA(p,d,q) equation in §2 of the Mathematical Explanation, explain why adding a constant term to a model fit on already-differenced data implies something different about the original series than adding it to the undifferenced model.
2. Section 3 found variance increasing from 1.1238 (d=1) to 1.4892 (d=2) on the same series. Propose a rule, based only on ADF/KPSS results at successive d, for when to stop differencing — and explain what evidence in this notebook would tell you if you differenced one time too many.
3. Section 5 showed AIC=2882.31 (d=0) vs AIC=2823.37 (d=1) both computed with `nobs=1000`. If `nobs` matches, why is directly comparing these two AIC values still invalid? (Hint: consider what "non-stationary starting parameters" warnings during the d=0 fit imply about what that likelihood is actually measuring.)
4. Section 7 found the AIC-"best" candidate (3,1,3) failed to converge, while the best converged candidate was (2,1,1) — a full 8 AIC points worse. Explain why using a non-converged model's AIC to justify its complexity would be a worse mistake than simply picking a slightly-higher-AIC model that actually converged.
5. Section 7 found ARIMA(2,1,1)'s residuals passing Ljung-Box cleanly at every lag up to 24 (including the seasonal lag 12), yet its forecast MAE (81.871) was far worse than Topics 04-05. Using this notebook's own explanation, describe what a residual autocorrelation check can and cannot tell you about a forecasting model's seasonal accuracy, and what additional diagnostic (hint: think about what's plotted in the forecast chart) would have flagged the problem earlier.

---

## 📓 Notebook

30 executed code cells: a from-scratch, exactly-invertible differencing/integration pair matched to `numpy`; a true ARIMA(1,1,1) process simulated and recovered two ways, catching and correcting a real default-trend mismatch between `ARIMA(p,0,q)` and `ARIMA(p,d,q)`; ADF-based (not ground-truth-based) selection of `d`, including a direct demonstration of over-differencing's variance cost; a manual (p,q) grid search cross-checked against `pmdarima`'s independent `auto_arima`, plus a direct AR/MA root stationarity/invertibility check; a real demonstration that AIC is not comparable across different `d`; forecast confidence intervals checked for honest coverage; and a full real-data application to log(AirPassengers) that catches a non-converged grid-search candidate, selects a clean-diagnostics model, and honestly reports that model's forecast badly underperforming Topics 04-05 — the direct, data-driven case for Topic 07 (SARIMA):

➡️ **[06_arima.ipynb](06_arima.ipynb)**

---

## 📚 Further Reading

- [Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)
- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — ARIMA models](https://otexts.com/fpp3/arima.html)
- [statsmodels: ARIMA documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [pmdarima: auto_arima documentation](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.06_ARIMA&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
