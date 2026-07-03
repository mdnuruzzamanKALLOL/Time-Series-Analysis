# 🔁 Autocorrelation & Partial Autocorrelation

> Status: ✅ Complete — [Open the notebook →](03_autocorrelation_partial_autocorrelation.ipynb)

Topic 03 of the Time Series Analysis repo. ACF and PACF are the Box-Jenkins methodology's core diagnostic tool: their *shape* (not just significance) identifies which classical model — AR, MA, or ARMA — fits a series, directly setting up Topic 05. This notebook builds the sample ACF formula and the Durbin-Levinson PACF recursion from scratch, matches both to `statsmodels` exactly, demonstrates the textbook AR-vs-MA signatures on synthetic data with known ground truth, and closes on the real AirPassengers series from Topics 01-02.

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
Box-Jenkins identification rule:

              ACF                       PACF
AR(p):   decays gradually          cuts off sharply after lag p
MA(q):   cuts off sharply after lag q    decays gradually
ARMA:    both decay gradually (Topic 05 combines the two)

PACF = correlation between Y_t and Y_t-k AFTER removing the
       linear effect of everything in between (lags 1..k-1)
```

---

## 🎯 Why This Topic Matters

- **Both the sample ACF formula and the Durbin-Levinson PACF recursion were built entirely from scratch and matched `statsmodels` to machine precision** — §1-§3 found a maximum difference of **2.22e-16** (ACF) and **1.25e-16** (PACF) against `statsmodels`' own implementations, confirming the textbook algorithms were correctly understood, not just imported.
- **AR(1)'s theoretical geometric-decay ACF was validated against the sample estimate** — §2 found the theoretical $\phi^k$ values (0.7, 0.49, 0.343...) versus the sample ACF (0.6826, 0.459, 0.3036...) agreeing with a mean absolute error of just **0.0357** across 15 lags.
- **The Box-Jenkins AR signature was demonstrated directly, not just asserted** — §3 found AR(1)'s PACF cutting off sharply after lag 1 (only lag 1 significant at the 95% confidence band), while its ACF decayed gradually across many lags — exactly the diagnostic pattern that will identify AR order in Topic 05.
- **MA(1) showed the exact mirror-image signature, confirming the rule works both directions** — §4 found MA(1)'s ACF cutting off after lag 1 (theoretical value 0.4412 via $\theta/(1+\theta^2)$, sample 0.4384) while its PACF stayed significant across several lags (1, 2, 3, and 9) — the opposite pattern from AR(1).
- **Confidence-band crossing rates on true white noise were checked directly against the theoretical rate** — §5 found 0 of 40 lags crossing the 95% band on genuine white noise, consistent with (if somewhat under, from ordinary sampling variability) the expected ~5% false-crossing rate.
- **Ljung-Box cleanly separated white noise from AR(1) using a single joint statistic** — §6 found p=**0.7009** for white noise (correctly fails to reject) versus p=**2.08e-83** for AR(1) (overwhelmingly rejects) — reusing Topic 01's portmanteau test on a genuinely different kind of structure.
- **Real AirPassengers data showed the textbook combined trend+seasonal ACF signature, and a genuinely honest finding after full transformation** — §7 found the raw series' ACF still at **0.7604** by lag 12 (slow decay plus repeated seasonal spikes), while even after log+seasonal+first differencing (Topic 02's recommended transformation), the Ljung-Box test still rejected no-autocorrelation at p=**0.0013** — real economic data didn't fully reduce to white noise from differencing alone, honestly foreshadowing why Topics 06-07 (ARIMA/SARIMA) exist as more complete models.

---

## 🧮 Mathematical Explanation

### 1. Sample autocorrelation function (ACF)

$$r_k = \frac{\sum_{t=1}^{n-k}(Y_t-\bar Y)(Y_{t+k}-\bar Y)}{\sum_{t=1}^{n}(Y_t-\bar Y)^2}$$

### 2. AR(1) theoretical ACF

$$\rho_k = \phi^k$$

A direct consequence of AR(1)'s defining recursion — geometric decay, never cutting off exactly to zero.

### 3. Durbin-Levinson recursion (PACF)

$$\phi_{kk} = \frac{r_k - \sum_{j=1}^{k-1}\phi_{k-1,j}r_{k-j}}{1-\sum_{j=1}^{k-1}\phi_{k-1,j}r_j}, \qquad \phi_{kj}=\phi_{k-1,j}-\phi_{kk}\phi_{k-1,k-j}$$

Computes each partial autocorrelation $\phi_{kk}$ recursively from the ACF values alone, without needing to re-fit a full $k$-th order regression at every step.

### 4. MA(1) theoretical ACF

$$\rho_1 = \frac{\theta}{1+\theta^2}, \qquad \rho_k = 0 \ \text{for } k>1$$

The sharp cutoff after lag 1 is exact for a true MA(1) process — it has no memory beyond a one-step-lagged noise term.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Sample ACF | $\sum(Y_t-\bar Y)(Y_{t+k}-\bar Y)/\sum(Y_t-\bar Y)^2$ | §1 |
| AR(1) theoretical ACF | $\phi^k$ | §2 |
| Durbin-Levinson PACF | recursive $\phi_{kk}$ | §3 |
| MA(1) theoretical ACF | $\theta/(1+\theta^2)$, else 0 | §4 |
| 95% confidence band | $\pm 1.96/\sqrt n$ | §5 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Reading ACF/PACF shape without knowing the Box-Jenkins rule** — §3-§4's directly-contrasted AR(1) vs MA(1) signatures (decay-then-cutoff vs cutoff-then-decay) are the entire diagnostic logic Topic 05 will rely on; misreading which one decays and which one cuts off leads to the wrong model order.
2. **Treating a single significant lag as proof of real structure** — §5's white-noise confidence-band check is the reason a *pattern* across several lags (or a joint Ljung-Box test, §6) is more trustworthy than any single crossing.
3. **Assuming differencing alone fully whitens a real series' ACF** — §7's honest finding (Ljung-Box p=0.0013 even after log+seasonal+first differencing) shows real data can retain structure a simple transformation doesn't remove — exactly the gap ARIMA/SARIMA (Topics 06-07) are built to close.
4. **Forgetting ACF and PACF answer different questions** — PACF's "removing the effect of intermediate lags" (§3's Durbin-Levinson formula) is what makes it useful for AR order identification specifically; using ACF alone for that purpose would misread an AR(2)+ process's gradually-decaying pattern as ambiguous.
5. **Using the wrong PACF `method` in statsmodels without knowing why** — §3 used `method='ywm'`, matching the manual Durbin-Levinson recursion built on *unadjusted* ACF values; other methods (`'ols'`, `'ywadjusted'`) give closely similar but not numerically identical results, worth knowing before comparing against a from-scratch implementation.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.stattools.acf` | Sample autocorrelation function |
| `statsmodels.tsa.stattools.pacf` | Partial autocorrelation function |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Joint multi-lag autocorrelation test (Topic 01, reused) |
| Custom `manual_acf()` | From-scratch sample ACF |
| Custom `durbin_levinson_pacf()` | From-scratch PACF recursion |

---

## 📝 Self-Test Exercises

1. Section 2 found the AR(1) sample ACF (0.6826 at lag 1) somewhat below the theoretical value (0.7). Using the sample ACF formula in §1, explain why a finite sample (n=500) would be expected to show some deviation from the exact theoretical $\phi^k$, and whether you'd expect this gap to shrink or grow as n increases.
2. Section 3 found AR(1)'s PACF significant only at lag 1, while section 4 found MA(1)'s PACF significant at lags 1, 2, 3, AND 9. Using the Durbin-Levinson formula in §3 of the Mathematical Explanation, explain in words why an MA process (which has no finite AR representation in general) would be expected to show a PACF that doesn't cut off cleanly.
3. Section 5 found 0 of 40 lags crossing the 95% confidence band on true white noise, when about 2 would be expected under the null. Is this result surprising, concerning, or unremarkable? Justify your answer using the confidence-band formula and basic sampling-variability reasoning.
4. Section 7 found the raw AirPassengers ACF still at 0.7604 by lag 12, while the fully-transformed series' Ljung-Box test still rejected at p=0.0013. Given Topic 02 already confirmed the transformed series passes the ADF stationarity test, does "stationary" (per ADF) guarantee "no significant autocorrelation" (per Ljung-Box)? Explain what these two diagnostics are actually each checking.
5. Section 7 found the fully-transformed AirPassengers series' significant PACF lags at 1, 3, 9, and 12. If you were choosing a candidate AR order for Topic 05's model-building based on this PACF alone, which lag would you consider most important, and why might lag 12 specifically deserve separate treatment (hint: think about what Topic 01 and Topic 02 already established about this series).

---

## 📓 Notebook

30 executed code cells: a from-scratch sample ACF and Durbin-Levinson PACF recursion both matched to `statsmodels` at machine precision, AR(1)'s theoretical geometric-decay ACF validated against the sample estimate, the classic Box-Jenkins AR-vs-MA ACF/PACF signatures demonstrated directly on synthetic data with known parameters, a confidence-band crossing-rate check on true white noise, Ljung-Box cleanly separating white noise from AR(1) with a single joint statistic, and a full real-data application to AirPassengers comparing raw vs. log+seasonal+first-differenced ACF/PACF — honestly reporting that differencing alone didn't fully eliminate detectable autocorrelation, setting up Topics 05-07's more complete models:

➡️ **[03_autocorrelation_partial_autocorrelation.ipynb](03_autocorrelation_partial_autocorrelation.ipynb)**

---

## 📚 Further Reading

- [Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)
- [Durbin (1960): The Fitting of Time-Series Models](https://www.jstor.org/stable/1401322)
- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Ch. 9.5: ACF and PACF Plots](https://otexts.com/fpp3/non-seasonal-arima.html)
- [statsmodels: ACF/PACF documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.03_Autocorrelation_Partial_Autocorrelation&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
