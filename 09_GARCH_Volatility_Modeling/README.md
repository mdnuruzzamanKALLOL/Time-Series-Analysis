# 📉 GARCH (Volatility Modeling)

> Status: ✅ Complete — [Open the notebook →](09_garch_volatility_modeling.ipynb)

Topic 09 of the Time Series Analysis repo — a different target entirely. Topics 01-08 modeled the conditional *mean* of one or several series; GARCH models the conditional *variance*, the phenomenon of volatility clustering ("large changes tend to be followed by large changes, of either sign") that a mean-only model like Topics 05-08 cannot see at all. This notebook builds ARCH(1) and GARCH(1,1) log-likelihood estimation from scratch and matches both to the `arch` package, builds Engle's ARCH LM test from scratch and matches it to `statsmodels`, then applies the full toolkit to real S&P 500 returns (1999-2018) — finding textbook volatility clustering, a decisive preference for fat-tailed Student-t innovations over Normal, and a real 2018 volatility-forecasting backtest.

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
Topics 01-08: model the conditional MEAN     E[Y_t | past]
Topic 09:     model the conditional VARIANCE Var[e_t | past]  <- volatility, not level

ARCH(1):    sigma_t^2 = omega + alpha * e_t-1^2                 (depends on past SHOCKS)
GARCH(1,1): sigma_t^2 = omega + alpha * e_t-1^2 + beta*sigma_t-1^2  (+ past VARIANCE itself)

Volatility clustering: a raw return series can look like pure noise (flat ACF)
while its SQUARED series is strongly autocorrelated -- big moves cluster together,
regardless of direction. This is invisible to every model in Topics 05-08.
```

---

## 🎯 Why This Topic Matters

- **ARCH(1) maximum-likelihood estimation was built entirely from scratch and matched the `arch` package on simulated data with known parameters** — §1 found a manual `scipy.optimize` MLE giving $\hat\omega=$**0.0976** (true 0.1) and $\hat\alpha=$**0.5014** (true 0.5), differing from the `arch` package's own fit by only **4.53e-04** total.
- **GARCH(1,1) MLE was built the same way and matched just as closely, recovering all three parameters from one optimization** — §2 found persistence ($\alpha+\beta$) of **0.9633** (manual) vs. **0.9634** (`arch` package), a **1.70e-05** difference, against a true value of 0.95.
- **Volatility clustering was demonstrated directly as autocorrelation in squared, not raw, values — invisible to every prior topic's tools** — §3 found the simulated GARCH series' raw-value Ljung-Box test failing to reject no-autocorrelation (p=**0.160**) while its squared values overwhelmingly rejected it (p=**7.4e-65**) on the *same* series.
- **Engle's ARCH LM test was built from scratch and matched `statsmodels` to near machine precision, then shown correctly telling homoskedastic from heteroskedastic data apart** — §4 found a maximum test-statistic difference of **4.55e-13**, with white noise giving p=0.0016 (borderline at 5 lags on this specific draw) versus a true ARCH(1) series giving p=**1.23e-102**.
- **Real S&P 500 returns (1999-2018) showed the identical clustering signature, and overwhelming evidence of ARCH effects** — §6 found squared-return autocorrelation of **0.21-0.38** across 10 lags (versus near-zero for raw returns), and an Engle's-test p-value of **4.6e-245**.
- **Student-t innovations decisively beat Normal on real returns — not a marginal preference, a 180-point AIC gap** — §7 found GARCH(1,1)-Normal at AIC=**13200.34** versus GARCH(1,1)-Student-t at AIC=**13020.63**, with a fitted **6.68** degrees of freedom confirming real fat tails no amount of GARCH tuning under a Normal assumption could represent.
- **A narrow AIC "win" for a more complex model was found and explicitly not followed, on the same principle established in Topics 05-06** — §7 found GARCH(2,1)/(2,2) narrowly beating GARCH(1,1) by AIC (13005/13004.6 vs. 13020.6), but the industry-standard, more interpretable GARCH(1,1) was kept for the rest of the analysis.
- **Real daily equity volatility showed the well-documented near-unit persistence, checked directly rather than assumed** — §8 found $\alpha+\beta=$**0.9995** — below 1 (a valid, stationary variance process) but close enough to confirm volatility shocks decay far more slowly than return shocks ever do.
- **A real volatility forecast was backtested through a genuinely volatile year and landed close to, not exactly at, its target** — §9 found a rolling 1-step-ahead GARCH(1,1)-t forecast through 2018 (spanning the February "Volmageddon" spike and the December selloff) producing a 5% VaR breach rate of **6.35%** (16 of 252 days) — an honest, realistic near-calibration, not a manufactured perfect result.

---

## 🧮 Mathematical Explanation

### 1. ARCH(1)

$$\varepsilon_t = \sigma_t z_t, \quad z_t\sim N(0,1), \qquad \sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2$$

Variance depends only on the most recent squared shock; a large $|\varepsilon_{t-1}|$ mechanically raises $\sigma_t^2$, regardless of sign.

### 2. GARCH(1,1)

$$\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$$

Adds memory of the variance's own recent level, not just the latest shock — why GARCH(1,1) can match long, slowly-decaying real volatility with only 3 parameters where ARCH would need many lags.

### 3. Gaussian conditional log-likelihood

$$\ln L=-\frac12\sum_{t=1}^{n}\left[\ln(2\pi\sigma_t^2)+\frac{\varepsilon_t^2}{\sigma_t^2}\right]$$

Both §1 and §2's manual estimators maximize this directly via `scipy.optimize.minimize` on the negated form.

### 4. Persistence and the unconditional variance

$$E[\sigma_t^2]=\frac{\omega}{1-\alpha-\beta} \qquad (\text{requires } \alpha+\beta<1)$$

$\alpha+\beta$ (§8's "persistence") controls how slowly a variance shock decays; values near 1 (as found on real data) imply very long volatility memory without technically violating stationarity.

### 5. Engle's ARCH LM test

$$LM = nR^2 \sim \chi^2_q, \qquad R^2 \text{ from regressing } \varepsilon_t^2 \text{ on } \varepsilon_{t-1}^2,\dots,\varepsilon_{t-q}^2$$

A large $LM$ (small p-value) rejects the null of no ARCH effect — exactly the test built from scratch in §4.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| ARCH(1) variance | $\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2$ | §1 |
| GARCH(1,1) variance | $\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$ | §2, §7-§9 |
| Gaussian log-likelihood | $-\frac12\sum[\ln(2\pi\sigma_t^2)+\varepsilon_t^2/\sigma_t^2]$ | §1-§2 |
| Unconditional variance | $\omega/(1-\alpha-\beta)$ | §2, §8 |
| Engle's ARCH LM test | $LM=nR^2\sim\chi^2_q$ | §4, §6 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Fitting GARCH to price levels instead of returns** — GARCH assumes a (near) zero- or constant-mean input; prices are trending and non-stationary (Topic 02's lesson), so only log returns (§5) are ever modeled here.
2. **Assuming Normal innovations by default** — §7 found Student-t beating Normal by 180 AIC points on real returns; using Normal innovations would understate the true probability of extreme moves.
3. **Reading `alpha+beta` close to 1 as automatically "broken"** — §8's persistence of ~0.9995 is a well-documented stylized fact of daily equity data, not a sign of misspecification, provided it stays below 1.
4. **Trusting the single lowest-AIC order without weighing interpretability** — §7 found GARCH(2,1)/(2,2) narrowly beating GARCH(1,1) by AIC; the industry-standard GARCH(1,1) was still used, consistent with Topic 05-06's own precedent that a narrow AIC win isn't an automatic mandate.
5. **Treating a single VaR backtest as proof of a well-calibrated model** — §9's 6.35% breach rate against a 5% target is close but not exact; a real risk model would be backtested across many non-overlapping windows before being trusted operationally.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `arch.arch_model` | ARCH/GARCH(p,q) MLE fitting, with Normal or Student-t innovations |
| `arch.data.sp500.load` | Real, bundled S&P 500 daily price data (1999-2018), no download needed |
| `statsmodels.stats.diagnostic.het_arch` | Library Engle's ARCH LM test |
| `statsmodels.tsa.stattools.acf` | Sample ACF (Topic 03, reused on squared values) |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Residual/squared-residual white-noise check |
| Custom `arch1_negloglik()` / `garch11_negloglik()` | From-scratch Gaussian conditional log-likelihoods |
| Custom `manual_arch_lm_test()` | From-scratch Engle's ARCH LM test |

---

## 📝 Self-Test Exercises

1. Section 3 found the same GARCH(1,1) series giving Ljung-Box p=0.160 on its raw values but p=7.4e-65 on its squared values. Using the ARCH(1) formula in §1 of the Mathematical Explanation, explain why a series can have literally zero linear autocorrelation in its level while still being highly predictable in its squared values.
2. Section 7 found GARCH(1,1)-Student-t beating GARCH(1,1)-Normal by 180 AIC points, with a fitted `nu` of 6.68. Propose what would happen to that AIC gap (larger or smaller) if the fitted `nu` had instead come out to 30, and explain why using the relationship between the Student-t and Normal distributions.
3. Section 8 found persistence ($\alpha+\beta$) of 0.9995 on real S&P 500 returns. Using the unconditional variance formula in §4 of the Mathematical Explanation, explain what happens to that formula's usefulness as $\alpha+\beta$ approaches 1, and why this matters for how far ahead a GARCH volatility forecast can be trusted.
4. Section 7 found GARCH(2,1) narrowly beating GARCH(1,1) by AIC (13005.1 vs. 13020.6) but GARCH(1,1) was used anyway. Propose one quantitative criterion (not just "interpretability") that could justify preferring the simpler model even when it doesn't have the single lowest AIC.
5. Section 9's VaR backtest found 16 breaches out of 252 days (6.35%) against a 5% target. Using the fact that this window includes both the February 2018 volatility spike and the December 2018 selloff, propose whether you'd expect the breach rate to be closer to 5% on a calmer historical window, and what that implies about evaluating a single backtest window in isolation.

---

## 📓 Notebook

24 executed code cells: ARCH(1) and GARCH(1,1) maximum-likelihood estimation built entirely from scratch via `scipy.optimize` and matched to the `arch` package on simulated data with known ground-truth parameters; volatility clustering demonstrated directly as autocorrelation in squared (not raw) values, invisible to every mean-based tool from Topics 01-08; Engle's ARCH LM test built from scratch and matched to `statsmodels` to near machine precision, shown correctly distinguishing homoskedastic from heteroskedastic data; and a full real-data application to S&P 500 daily returns (1999-2018) — confirming overwhelming real ARCH effects, a decisive 180-AIC-point preference for Student-t over Normal innovations, a persistence near (but below) 1 consistent with known equity-market behavior, and a real volatility-forecast backtest through the 2018 "Volmageddon" spike and December selloff landing close to its 5% VaR target:

➡️ **[09_garch_volatility_modeling.ipynb](09_garch_volatility_modeling.ipynb)**

---

## 📚 Further Reading

- [Engle (1982): Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation](https://www.jstor.org/stable/1912773)
- [Bollerslev (1986): Generalized Autoregressive Conditional Heteroskedasticity](https://doi.org/10.1016/0304-4076%2886%2990063-1)
- [Sheppard et al., `arch` package documentation](https://arch.readthedocs.io/)
- [RiskMetrics/J.P. Morgan Technical Document — Value at Risk](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.09_GARCH_Volatility_Modeling&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
