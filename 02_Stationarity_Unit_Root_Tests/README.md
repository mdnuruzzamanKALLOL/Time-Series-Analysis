# 📉 Stationarity & Unit Root Tests

> Status: ✅ Complete — [Open the notebook →](02_stationarity_unit_root_tests.ipynb)

Topic 02 of the Time Series Analysis repo. Nearly every classical forecasting model built later in this series (AR, MA, ARMA, ARIMA) assumes the series is **stationary** — constant mean, constant variance, autocovariance depending only on lag, not on absolute time. This notebook builds the Augmented Dickey-Fuller test's regression from scratch and matches it exactly to `statsmodels`, contrasts it with the complementary-null KPSS test, demonstrates the classic spurious-regression trap on two independent random walks (300-trial simulation), and closes on the real AirPassengers series from Topic 01.

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
ADF:  H0 = unit root (non-stationary)     H1 = stationary
KPSS: H0 = stationary                     H1 = unit root (non-stationary)
      -> opposite nulls, used TOGETHER for cross-validation

Random walk (non-stationary): Y_t = Y_{t-1} + noise   -- variance grows with t
White noise (stationary):     Y_t = noise              -- constant variance

Fix for a unit root: DIFFERENCING  ->  Delta_Y_t = Y_t - Y_{t-1}

DANGER: regressing one non-stationary series on another can look "significant"
        with ZERO true relationship -- always check stationarity before regressing.
```

---

## 🎯 Why This Topic Matters

- **The Augmented Dickey-Fuller regression was built entirely from scratch and matched `statsmodels` to six decimal places** — §2 found the manual OLS-based Dickey-Fuller t-statistic at **-22.343815** for white noise and **-0.282015** for a random walk, both matching `adfuller`'s returned statistic exactly.
- **ADF correctly separated the two known-answer cases** — §2 found white noise's ADF p-value at **0.000000** (correctly rejects the unit-root null) versus the random walk's **0.928013** (correctly fails to reject) — and §2 additionally confirmed the random walk's statistic (-0.2820) doesn't beat the 5% critical value (**-2.8673**), the actual decision rule behind the p-value.
- **The "augmented" part of ADF was shown to matter, not just be a formality** — §3 built a genuine AR(2) process and found the *unaugmented* Dickey-Fuller regression's residuals still significantly autocorrelated (Ljung-Box p=**0.000000**, 5 lags) — exactly the structure that adding 2 lagged difference terms is designed to absorb.
- **KPSS's opposite-null framing agreed with ADF on both synthetic series** — §4 found KPSS correctly failing to reject stationarity for white noise (p=0.100) and correctly rejecting it for the random walk (p=0.010) — cross-validating two independently-derived tests, the same discipline used throughout this series.
- **A genuine, quantified pitfall: the ADF regression specification itself determines the answer** — §5 found a deliberately trend-stationary series (noise around a deterministic line) *incorrectly* failing to reject the unit-root null (p=0.9603) when tested without a trend term, versus *correctly* rejecting it (p≈0.000000) once `regression='ct'` was used — the exact same data, opposite conclusions, purely from a specification choice.
- **Differencing was shown directly converting a random walk to stationary** — §6 found the raw random walk's ADF p-value at 0.9280 dropping to **0.000000** after a single first difference — the textbook I(1)→I(0) transformation, not just asserted.
- **Spurious regression was measured properly across 300 repeated trials, not asserted from one lucky draw** — §7's single-run demo produced only a modest R²=0.009 (a reminder that any one realization can understate the effect), but the follow-up 300-trial simulation found the raw-level regression **falsely "significant" (p<0.05) 79.3% of the time** between two truly *independent* random walks — nearly 16x the nominal 5% rate — collapsing to a correctly-calibrated **6.7%** once both series were first-differenced.
- **On real AirPassengers data, only seasonal differencing (not log+first-differencing alone) achieved ADF stationarity** — §8 found the raw series at p=0.9919, log+first-differenced at p=0.0711 (still not significant at 0.05), plain seasonal differencing (lag 12) at p=**0.0116** (stationary), while log+seasonal differencing combined landed at p=0.0724 — a genuinely mixed, honestly-reported real-data result rather than a clean textbook outcome.

---

## 🧮 Mathematical Explanation

### 1. The Augmented Dickey-Fuller regression

$$\Delta Y_t = \alpha + \gamma Y_{t-1} + \sum_{i=1}^{p}\delta_i \Delta Y_{t-i} + \varepsilon_t$$

$H_0: \gamma=0$ (unit root) vs $H_1: \gamma<0$ (stationary). The test statistic $\hat\gamma/SE(\hat\gamma)$ is an ordinary OLS t-statistic, but its null distribution isn't the standard t-distribution — it follows the (non-standard) Dickey-Fuller distribution, requiring its own tabulated critical values.

### 2. KPSS test statistic

$$KPSS = \frac{\sum_{t=1}^{T} S_t^2}{T^2\hat\sigma^2}, \qquad S_t = \sum_{i=1}^{t}\hat e_i$$

Built from the cumulative sum of residuals from a regression on a constant (or constant+trend) — large values indicate a growing "random walk" component in the residual, i.e. non-stationarity.

### 3. Differencing removes one unit root

$$\Delta Y_t = Y_t - Y_{t-1} = \epsilon_t \quad \text{(if } Y_t \text{ is a pure random walk)}$$

### 4. Spurious regression (Granger & Newbold, 1974)

Regressing two independent I(1) (unit-root) series produces a t-statistic whose distribution does **not** converge to standard normal even as $T\to\infty$ — it diverges, meaning the false-rejection rate does not shrink with more data, unlike genuine model misspecification.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| ADF regression | $\Delta Y_t=\alpha+\gamma Y_{t-1}+\sum\delta_i\Delta Y_{t-i}+\varepsilon_t$ | §2-§3 |
| KPSS statistic | $\sum S_t^2/(T^2\hat\sigma^2)$ | §4 |
| First difference | $Y_t-Y_{t-1}$ | §6 |
| Seasonal difference | $Y_t-Y_{t-12}$ (monthly data) | §8 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting the ADF "augmentation" on autocorrelated data** — §3's Ljung-Box p=0.000000 on the unaugmented regression's residuals shows real leftover structure when the process has genuine AR dynamics beyond a single lag.
2. **Running ADF without a trend term on trend-stationary data** — §5's flipped conclusion (p=0.9603 vs p≈0.000000) from the SAME data is a direct, quantified illustration; always consider whether `regression='c'`, `'ct'`, or `'n'` matches the series' actual structure.
3. **Trusting a single spurious-regression demo instead of the repeated-trial rate** — §7's single realization (R²=0.009) looked unremarkable, while the real 300-trial false-rejection rate (79.3%) told the true, alarming story; the same single-run-vs-many-trials lesson applies here as in the Statistical Inference repo's Topic 07.
4. **Regressing two non-stationary series directly** — §7's 79.3% false-positive rate (vs a correct 6.7% after differencing) is the direct, measured cost of skipping a stationarity check before running any regression on time-ordered data.
5. **Assuming one transformation (e.g. log+difference) always achieves stationarity** — §8's honest finding that log+seasonal differencing did NOT clear the 0.05 threshold (p=0.0724) while plain seasonal differencing did (p=0.0116) shows real data doesn't always cooperate with the "textbook" transformation choice; always re-check with ADF/KPSS after transforming, never assume.
6. **Treating ADF's failure to reject as proof of a unit root** — ADF's null is non-stationarity; failing to reject only means insufficient evidence against it. This is exactly why §4 ran KPSS (opposite null) as a cross-check rather than relying on ADF alone.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.stattools.adfuller` | Augmented Dickey-Fuller unit-root test |
| `statsmodels.tsa.stattools.kpss` | KPSS stationarity test (opposite null) |
| `statsmodels.stats.diagnostic.acorr_ljungbox` | Residual autocorrelation check (Topic 01, reused) |
| Custom `manual_adf_statistic()` | From-scratch Dickey-Fuller regression |
| Custom `spurious_regression_trial()` | Repeated-trial spurious regression simulator |

---

## 📝 Self-Test Exercises

1. Section 2 found the manual and statsmodels ADF statistics matching to six decimal places on both series. Using the ADF regression formula in §1 of the Mathematical Explanation, explain which regression coefficient corresponds to $\gamma$ and why its t-statistic (not a t-distribution p-value in the usual sense) is what ADF actually tests.
2. Section 5 found the SAME trend-stationary series either failing or correctly rejecting the unit-root null purely based on the `regression` argument (`'c'` vs `'ct'`). Explain in words what a deterministic trend does to a constant-only ADF regression's residuals, and why that would bias the test toward failing to reject a unit root.
3. Section 7 found a single-realization spurious regression R-squared of just 0.009, while the 300-trial simulation found a 79.3% false-rejection rate. Reconcile these two findings — how can the *typical* R-squared be modest while the false-positive rate is still dramatically inflated? (Hint: think about what "R-squared=0.009 but p<0.05" implies about sample size and the null distribution of the t-statistic under non-stationarity.)
4. Section 8 found log+seasonal differencing (p=0.0724) failing to clear the 0.05 threshold while seasonal differencing alone (p=0.0116) succeeded, on the same AirPassengers series. Propose one plausible reason adding the log transform on top of seasonal differencing could make the ADF test slightly less conclusive here, rather than more.
5. Section 4 found ADF and KPSS agreeing on both synthetic series (white noise: stationary/stationary; random walk: non-stationary/non-stationary). If a real series produced ADF "fails to reject" (suggesting non-stationary) AND KPSS "rejects" (also suggesting non-stationary), would that agreement be stronger or weaker evidence than if ADF said "stationary" while KPSS said "non-stationary"? Explain what a disagreement between the two tests would imply.

---

## 📓 Notebook

33 executed code cells: a from-scratch Augmented Dickey-Fuller regression matched to `statsmodels` at six decimal places on both a stationary and non-stationary series, a demonstrated need for augmentation via Ljung-Box residual testing on a real AR(2) process, KPSS cross-validated against ADF on both synthetic cases, a genuine trend-term-specification pitfall that flips the ADF conclusion on identical data, differencing shown converting a random walk to stationary, a 300-trial repeated simulation revealing spurious regression's true 79.3% false-rejection rate (vs a correct 6.7% after differencing) rather than relying on one anecdote, and a full real-data stationarity workup on AirPassengers testing four different transformations, honestly reporting that log+seasonal differencing didn't clear significance while seasonal differencing alone did:

➡️ **[02_stationarity_unit_root_tests.ipynb](02_stationarity_unit_root_tests.ipynb)**

---

## 📚 Further Reading

- [Dickey & Fuller (1979): Distribution of the Estimators for Autoregressive Time Series with a Unit Root](https://www.jstor.org/stable/2286348)
- [Kwiatkowski, Phillips, Schmidt & Shin (1992): Testing the Null Hypothesis of Stationarity](https://www.sciencedirect.com/science/article/abs/pii/030440769290104Y)
- [Granger & Newbold (1974): Spurious Regressions in Econometrics](https://www.sciencedirect.com/science/article/abs/pii/0304407674900347)
- [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Ch. 9.1: Stationarity and Differencing](https://otexts.com/fpp3/stationarity.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.02_Stationarity_Unit_Root_Tests&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
