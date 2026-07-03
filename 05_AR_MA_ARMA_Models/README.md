# 🧮 AR, MA, ARMA Models

> Status: ✅ Complete — [Open the notebook →](05_ar_ma_arma_models.ipynb)

Topic 05 of the Time Series Analysis repo — the building blocks of the classical forecasting model family, and the direct payoff of Topics 02-04's preparation. This notebook builds AR coefficient estimation via the Yule-Walker equations from scratch and matches it to `statsmodels` exactly, estimates an MA(1) parameter via method-of-moments and compares it honestly against full MLE, uses AIC/BIC to recover a known true model order, and closes by fitting a real ARMA model to the transformed AirPassengers series from Topics 01-04, forecasting against the same holdout period Topic 04 used.

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
AR(p):   Y_t = phi_1*Y_t-1 + ... + phi_p*Y_t-p + e_t         -- regresses on its OWN past values
MA(q):   Y_t = e_t + theta_1*e_t-1 + ... + theta_q*e_t-q      -- regresses on past FORECAST ERRORS
ARMA(p,q): combines both

Estimation:
  AR:  Yule-Walker (closed-form, from the ACF) or full MLE
  MA:  no closed-form linear solution -- method-of-moments (MA(1) only) or full MLE
  ARMA: full MLE (the only practical option once both are present)

Order selection: AIC/BIC grid search over candidate (p,q), guided by
                 Topic 03's ACF/PACF signatures as a starting point.
```

---

## 🎯 Why This Topic Matters

- **AR coefficient estimation via Yule-Walker was built from scratch and matched `statsmodels` to exact floating-point precision** — §1 found a maximum difference of **0.00e+00** between the manual Toeplitz-system solver and `statsmodels`' `yule_walker` function on the same AR(2) data.
- **Yule-Walker and full MLE were compared honestly as two legitimate but different estimators, not forced to agree** — §2 found Yule-Walker's total absolute error against the true $\phi=[0.6,-0.2]$ at **0.0063** versus full MLE's **0.0073** — Yule-Walker happened to win on this particular sample, a reminder that "more sophisticated" doesn't guarantee "closer" on any single finite draw.
- **MA(1)'s method-of-moments estimator, built from the closed-form invertible-root solution, agreed closely (not exactly) with full MLE** — §3 found method-of-moments at $\hat\theta=$**0.4696** versus MLE's **0.5035** (true $\theta=0.5$) — a real 0.0340 gap, honestly reported as expected estimator disagreement rather than a bug.
- **A synthetic ARMA(1,1) model's residuals passed the Ljung-Box white-noise check convincingly** — §4 found p=**0.9966** (10 lags) on the fitted model's residuals, strong evidence the model successfully captured the underlying structure.
- **AIC and BIC both correctly recovered the true (1,1) order from an 8-candidate grid search** — §5 found both criteria selecting **(1,1)** — AIC at 2815.38, BIC at 2835.01 — out of orders ranging from (0,1) up to (2,2).
- **The complexity penalty was shown directly outweighing a deliberately overfit model's marginal fit improvement** — §6 found the true-order (1,1) model's AIC (**2815.38**) beating a needlessly complex (5,5) model's AIC (**2822.06**), despite the larger model having far more free parameters to fit the training data with.
- **On real, Topic-02-transformed AirPassengers data, AIC selected a pure-AR(2) model over the intuitively-expected ARMA(1,1)** — §7 found order **(2,0)** winning the grid search (AIC=-368.89) narrowly over (1,1) (AIC=-368.18), and its residuals passed Ljung-Box cleanly (p=**0.6479**) — a genuinely data-driven result, not the order a naive PACF-only reading of Topic 03 might have suggested.
- **Reconstructed back to the original passenger-count scale, ARMA(2,0)'s forecast MAE beat Topic 04's best method** — §7 found a reconstruction MAE of **17.792** on the identical 24-month holdout Topic 04 used, versus Holt-Winters' **28.977** — a genuine, apples-to-apples improvement from the more principled Box-Jenkins model-building process this repo has built up to.

---

## 🧮 Mathematical Explanation

### 1. AR(p) and the Yule-Walker equations

$$Y_t = \sum_{i=1}^p \phi_i Y_{t-i} + \varepsilon_t \qquad\Longrightarrow\qquad \mathbf{R}\boldsymbol\phi = \mathbf{r}$$

where $\mathbf{R}$ is the $p\times p$ Toeplitz matrix of sample autocorrelations $r_0,\dots,r_{p-1}$ and $\mathbf r=(r_1,\dots,r_p)^T$ — turning AR estimation into a linear system solvable in closed form.

### 2. MA(1) method-of-moments

$$\rho_1 = \frac{\theta}{1+\theta^2} \quad\Longrightarrow\quad \theta = \frac{1-\sqrt{1-4\rho_1^2}}{2\rho_1}$$

The invertible root (the one satisfying $|\theta|<1$) is the unique economically-meaningful solution to this quadratic.

### 3. ARMA(p,q)

$$Y_t = \sum_{i=1}^p\phi_iY_{t-i} + \varepsilon_t + \sum_{j=1}^q\theta_j\varepsilon_{t-j}$$

### 4. Information criteria

$$AIC = -2\ln(\hat L) + 2k, \qquad BIC = -2\ln(\hat L) + k\ln(n)$$

BIC's $\ln(n)$ penalty per parameter grows with sample size, making it asymptotically more conservative (favoring simpler models) than AIC's fixed penalty of 2 per parameter.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Yule-Walker system | $\mathbf R\boldsymbol\phi=\mathbf r$ | §1 |
| MA(1) method-of-moments | $\theta=(1-\sqrt{1-4\rho_1^2})/(2\rho_1)$ | §3 |
| ARMA(p,q) | $\sum\phi_iY_{t-i}+\varepsilon_t+\sum\theta_j\varepsilon_{t-j}$ | §4 |
| AIC / BIC | $-2\ln\hat L+2k$ / $-2\ln\hat L+k\ln n$ | §5-§7 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Expecting Yule-Walker and MLE to match exactly** — §2's honest 0.0063-vs-0.0073 comparison (Yule-Walker actually closer here) contrasts with §1's exact 0.00e+00 match against `statsmodels`' *own* Yule-Walker function — matching a library's implementation of the *same* algorithm should be exact; matching a *different, legitimate* algorithm should only be expected to be close.
2. **Assuming ACF/PACF alone dictates the final model order** — §7's AIC-selected (2,0) beating (1,1) on real data is a reminder that Topic 03's ACF/PACF signatures are a starting point for candidate orders, not a guarantee; a proper grid search with AIC/BIC (and residual diagnostics) is still required.
3. **Ignoring optimizer convergence warnings during a grid search** — several candidate orders in §5 and §7's searches triggered `statsmodels` convergence/non-stationary-starting-parameter warnings; these don't necessarily invalidate the fitted AIC/BIC values but are worth checking when a specific candidate order looks suspiciously good or bad.
4. **Trusting AIC over BIC (or vice versa) without knowing the tradeoff** — §5's both-agree result doesn't mean they always will; BIC's heavier per-parameter penalty makes it systematically more likely to prefer simpler models as sample size grows, per the formula in §4 of the Mathematical Explanation.
5. **Skipping residual diagnostics after model selection** — §7's Ljung-Box check (p=0.6479) on the selected model's residuals is what actually confirms the AIC-chosen order adequately captured the structure, not just that it had the lowest AIC among the candidates tried.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.arima.model.ARIMA` | Full MLE fitting for AR/MA/ARMA models |
| `statsmodels.regression.linear_model.yule_walker` | Library Yule-Walker AR estimator |
| `statsmodels.tsa.stattools.acf` | Sample ACF (Topic 03, reused for method-of-moments) |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Residual white-noise check (Topics 01/03, reused) |
| Custom `manual_yule_walker()` | From-scratch Toeplitz-system AR solver |
| Custom `mom_ma1_theta()` | From-scratch MA(1) method-of-moments estimator |

---

## 📝 Self-Test Exercises

1. Section 1 found the manual Yule-Walker solver matching `statsmodels`' `yule_walker` function to exactly 0.00e+00. Using the Toeplitz system formula in §1 of the Mathematical Explanation, explain why solving a $p\times p$ linear system is sufficient for AR estimation, but no equivalent closed-form linear system exists for MA coefficients.
2. Section 3 found method-of-moments (0.4696) and MLE (0.5035) disagreeing by 0.0340 on the same MA(1) data, while Section 1's Yule-Walker matched statsmodels' Yule-Walker exactly. Explain why these are fundamentally different kinds of comparisons — one validates an *implementation*, the other compares two different *estimation methods*.
3. Section 7 found AIC selecting order (2,0) — a pure AR model with no MA component — on the real, transformed AirPassengers data, even though Topic 03 found several significant PACF lags beyond just 1-2. Propose one reason a grid search restricted to p,q ≤ 3 might miss a better higher-order model, and how you would extend the search to check.
4. Section 6 found the overfit (5,5) model's AIC (2822.06) worse than the true-order (1,1) model's (2815.38), despite the overfit model's greater flexibility. Using the AIC formula in §4, calculate roughly how many more log-likelihood units the (5,5) model would have needed to gain (from its extra 8 parameters) to actually beat the (1,1) model's AIC.
5. Section 7 found ARMA(2,0)'s reconstructed forecast MAE (17.792) beating Topic 04's Holt-Winters MAE (28.977) on the identical 24-month holdout. Both models used information from the same underlying data. Propose one reason the Box-Jenkins (ACF/PACF/AIC-driven) modeling process used in this notebook might produce a better forecast than exponential smoothing on this specific series.

---

## 📓 Notebook

31 executed code cells: AR(2) coefficient estimation via a from-scratch Yule-Walker solver matched to `statsmodels` at exact floating-point precision, an honest Yule-Walker-vs-MLE comparison as two legitimate but different estimators, MA(1)'s method-of-moments estimator built from the closed-form invertible-root solution and compared against full MLE, a synthetic ARMA(1,1) model's residuals passing Ljung-Box cleanly, an 8-candidate AIC/BIC grid search correctly recovering the true (1,1) order, a deliberately overfit (5,5) model correctly penalized by both information criteria, and a full real-data application selecting an ARMA(2,0) model for the Topic-02-transformed AirPassengers series that — reconstructed back to the original passenger-count scale — beat Topic 04's best exponential smoothing forecast (MAE 17.792 vs 28.977) on the identical 24-month holdout:

➡️ **[05_ar_ma_arma_models.ipynb](05_ar_ma_arma_models.ipynb)**

---

## 📚 Further Reading

- [Yule (1927): On a Method of Investigating Periodicities in Disturbed Series](https://royalsocietypublishing.org/doi/10.1098/rsta.1927.0007)
- [Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)
- [Akaike (1974): A New Look at the Statistical Model Identification](https://ieeexplore.ieee.org/document/1100705)
- [statsmodels: ARIMA documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.05_AR_MA_ARMA_Models&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
