# 🕸️ Vector Autoregression (VAR)

> Status: ✅ Complete — [Open the notebook →](08_vector_autoregression_var.ipynb)

Topic 08 of the Time Series Analysis repo — the first departure from a single series. Every prior topic modeled one variable's own past; VAR models several interdependent variables jointly, each regressed on lagged values of *all* of them. This notebook builds VAR(1) OLS estimation and a Granger-causality F-test entirely from scratch and matches both exactly to `statsmodels`, checks system stability via the coefficient matrix's eigenvalues, then applies the full toolkit (order selection, Granger causality, impulse responses, forecast error variance decomposition) to real US macroeconomic data — closing with an intentional stress test: forecasting straight through the 2008 financial crisis, where a linear model trained on calmer history is honestly beaten by a naive flat forecast.

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
Univariate AR(p) (Topics 05-07):  Y_t depends on ITS OWN past values only

VAR(p): a SYSTEM of k variables, each depending on the PAST OF ALL k variables

  y1_t = a11*y1_t-1 + a12*y2_t-1 + ... + a1k*yk_t-1 + e1_t
  y2_t = a21*y1_t-1 + a22*y2_t-1 + ... + a2k*yk_t-1 + e2_t
  ...
  yk_t = ak1*y1_t-1 + ak2*y2_t-1 + ... + akk*yk_t-1 + ek_t

  <=>  Y_t = A1 Y_t-1 + ... + Ap Y_t-p + E_t   (matrix form)

New tools this system unlocks that a single series never could:
  - Granger causality: does one series' past help predict another's future?
  - Impulse response functions: how does a shock to one variable ripple through the rest?
  - Forecast error variance decomposition: how much of each variable's uncertainty
    is self-driven vs. driven by the others?
```

---

## 🎯 Why This Topic Matters

- **VAR(1) estimation was built from scratch as plain multivariate OLS and matched `statsmodels` to exact floating-point precision** — §1 found a manual least-squares solve on a simulated, deliberately asymmetric bivariate system matching `statsmodels`' `VAR` class at **0.00e+00** maximum difference.
- **A Granger-causality F-test was built from scratch (restricted vs. unrestricted SSR) and matched `statsmodels` exactly, then correctly recovered a known one-directional causal structure** — §2 found the manual F-statistic matching `statsmodels`' `grangercausalitytests` to **6.39e-14**, and correctly detecting that $y_2$ Granger-causes $y_1$ (p=**9.1e-15**, true coefficient 0.3) while $y_1$ does *not* Granger-cause $y_2$ (p=**0.690**, true coefficient 0.0) — exactly the asymmetry built into the simulation.
- **System stability was checked via the coefficient matrix's eigenvalues, the direct multivariate generalization of Topics 06-07's AR-root check** — §3 found eigenvalue moduli of **0.482** and **0.414** (both < 1), cross-validated against `statsmodels`' `.is_stable()` and its reciprocal-convention `.roots`.
- **All three real macro variables were confirmed non-stationary at the level and stationary only after a log-difference, exactly Topic 02's discipline applied three times over** — §4 found ADF p-values of **0.65-1.00** at the level collapsing to **p<0.0001** after log-differencing, for real GDP, consumption, and investment (1959Q1-2009Q3).
- **All four order-selection criteria agreed on the same lag order, a clean and satisfying real-data result** — §5 found AIC, BIC, FPE, and HQIC all selecting **lag 1** out of 8 candidates.
- **Granger causality tests on real macro data uncovered genuine, economically sensible asymmetries without assuming them** — §7 found consumption growth Granger-causing GDP growth (p<0.0001) but not the reverse (p=0.1225), and GDP growth Granger-causing investment growth (p=0.0008, the textbook "accelerator effect") but not the reverse (p=0.1964).
- **Impulse response functions showed a GDP shock's effect on investment decaying to near-zero within the stability the eigenvalues predicted** — §8 found the cumulative effect dropping from **0.0300** at impact to **0.000002** by quarter 10, direct visual confirmation that a stable system's shocks must decay.
- **Forecast error variance decomposition quantified exactly how externally-driven investment is compared to GDP** — §9 found GDP growth's own shocks explaining **86.8%** of its own forecast error variance, while investment growth's own shocks explained only **21.9%** of its own — consistent with §7's finding that investment is Granger-caused by GDP but not vice versa.
- **A deliberate stress test — forecasting straight through the 2008 financial crisis — found VAR(1) honestly beaten by a naive flat forecast on all three variables** — §10 found real investment falling **31.4%** over the 8-quarter test window, and a "no change from last quarter" naive forecast beating VAR(1)'s forecast MAE on real GDP (192 vs. 570), real consumption (77 vs. 415), and real investment (344 vs. 378) — a stable, correctly-specified, well-validated model still failed once the future stopped resembling its training regime.

---

## 🧮 Mathematical Explanation

### 1. The VAR(p) system

$$Y_t = c + A_1Y_{t-1} + \dots + A_pY_{t-p} + \varepsilon_t, \qquad Y_t\in\mathbb{R}^k$$

Each $A_i$ is a $k\times k$ matrix; $A_i[m,n]$ is the effect of variable $n$'s lag-$i$ value on variable $m$'s current value. A VAR(1) with $k=2$ has 4 free coefficients (plus 2 constants); a VAR(2) with $k=3$ already has 18.

### 2. OLS estimation, equation by equation

Every equation shares the same $k\times p$ regressors (all variables' lagged values); only the target column differs. This is exactly why a single `lstsq` call against a stacked regressor matrix reproduces `statsmodels`' per-equation OLS exactly (§1).

### 3. Granger causality as a nested F-test

$$F = \frac{(SSR_r - SSR_u)/q}{SSR_u/(n-k)} \sim F_{q,\,n-k}$$

$SSR_r$ is the restricted model's (target on its own lags only) sum of squared residuals; $SSR_u$ adds the candidate cause's lags. A large $F$ (small p-value) means the extra lags significantly reduced residual variance.

### 4. Stability via eigenvalues

$$\det(I_k - A_1z - \dots - A_pz^p) = 0 \text{ has all roots } |z|>1 \iff \text{all eigenvalues of the companion matrix have modulus} <1$$

For a VAR(1), the companion matrix is $A_1$ itself — hence §3's direct eigenvalue check of the fitted $A_1$.

### 5. Orthogonalized impulse responses

$$Y_t = \sum_{i=0}^{\infty}\Psi_i\varepsilon_{t-i}, \qquad \Psi_i \text{ from a Cholesky decomposition of } \text{Cov}(\varepsilon_t)$$

The Cholesky step orthogonalizes correlated shocks so each one can be interpreted as an independent, one-standard-deviation impulse (§8).

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| VAR(p) system | $Y_t=c+A_1Y_{t-1}+\dots+A_pY_{t-p}+\varepsilon_t$ | §1, §6 |
| Granger causality F-test | $F=\frac{(SSR_r-SSR_u)/q}{SSR_u/(n-k)}$ | §2, §7 |
| Stability condition | all eigenvalues of $A_1$ (or companion matrix) have modulus $<1$ | §3, §6 |
| Orthogonalized IRF | $Y_t=\sum_i\Psi_i\varepsilon_{t-i}$ | §8 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Assuming one stationary variable is enough** — §4 checked *all three* real variables individually; a VAR with even one non-stationary input risks the same spurious-regression trap Topic 02 demonstrated, now hiding inside a multivariate system.
2. **Letting parameter count grow unchecked** — a VAR(p) has $k^2p$ AR coefficients; §5's order-selection criteria exist specifically to stop this from exploding as more variables or lags are added.
3. **Treating Granger causality as proven causation** — §7's consumption→GDP and GDP→investment results are consistent with economic theory, but the test only shows predictive precedence in the data, not a verified causal mechanism.
4. **Confusing `statsmodels`' `.roots` convention with a raw eigenvalue check** — §3 found `.roots` reporting the *reciprocal* of the eigenvalues (expected >1 for stability, the same convention as Topics 06-07's AR roots); mixing the two conventions up would flip the stability conclusion.
5. **Trusting a stable, well-specified model to forecast through a regime change** — §10's crisis stress test found VAR(1) beaten by a naive flat forecast on every variable; correct specification and clean historical fit are necessary, not sufficient, for good forecasts when the future looks nothing like the training data.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.vector_ar.var_model.VAR` | VAR(p) estimation, order selection, stability, IRF, FEVD |
| `statsmodels.tsa.stattools.grangercausalitytests` | Library Granger-causality F/chi2/LR tests |
| `statsmodels.tsa.stattools.adfuller` | ADF unit-root test (Topic 02, reused per variable) |
| `statsmodels.datasets.macrodata` | Real US quarterly macroeconomic data (1959Q1-2009Q3) |
| Custom `manual_granger_f()` | From-scratch restricted-vs-unrestricted F-test |
| `VAR.irf()` / `VAR.fevd()` | Impulse response functions / forecast error variance decomposition |

---

## 📝 Self-Test Exercises

1. Section 1 found a single `lstsq` call reproducing all of `statsmodels`' per-equation VAR OLS coefficients exactly. Using the VAR(p) equation in §1 of the Mathematical Explanation, explain why every equation in the system can share the same regressor matrix, unlike a general system of unrelated regressions.
2. Section 2 built `true_A1` with a 0.0 coefficient specifically so $y_1$'s lag would have no effect on $y_2$. If that coefficient had instead been a small but nonzero 0.05, propose what you'd expect to happen to the p-value of the $y_1\to y_2$ Granger test as the sample size `n_var` grows, and why.
3. Section 3 found `statsmodels`' `.roots` reporting values that are the reciprocals of the eigenvalues computed directly. Using the stability formula in §4 of the Mathematical Explanation, explain why "eigenvalue modulus < 1" and "root modulus > 1" are the same stability condition stated two different ways.
4. Section 9's FEVD found investment growth's own shocks explaining only 21.9% of its own 10-quarter-ahead forecast error variance. Propose what a FEVD result of 100% self-explained variance for some variable would imply about that variable's relationship to the rest of the system, and whether it would be consistent with the Granger causality results in Section 7.
5. Section 10 found a naive flat forecast beating VAR(1) during the 2008 crisis on all three variables. Propose one modification to the forecasting approach (not the model itself) that might have reduced VAR(1)'s error during the crisis window, and explain what information it would need that a model trained only on pre-2007 data doesn't have.

---

## 📓 Notebook

23 executed code cells: a from-scratch VAR(1) OLS estimator matched exactly to `statsmodels` on a simulated, deliberately asymmetric bivariate system; a from-scratch Granger-causality F-test matched exactly to `statsmodels` and shown correctly recovering the known one-directional causal structure; a multivariate stability check via coefficient-matrix eigenvalues, cross-validated against `statsmodels`' reciprocal-convention roots; a full real-data application to US GDP, consumption, and investment (1959Q1-2009Q3) — stationarity confirmed per variable, order selection agreeing across all four criteria, genuine asymmetric Granger causality uncovered (consumption→GDP, GDP→investment), impulse response functions and forecast error variance decomposition characterizing shock propagation — and a deliberate stress test forecasting straight through the 2008 financial crisis, where VAR(1) is honestly outperformed by a naive flat forecast on every variable:

➡️ **[08_vector_autoregression_var.ipynb](08_vector_autoregression_var.ipynb)**

---

## 📚 Further Reading

- [Sims (1980): Macroeconomics and Reality](https://www.jstor.org/stable/1912017) — the original VAR paper
- [Lütkepohl, *New Introduction to Multiple Time Series Analysis*](https://link.springer.com/book/10.1007/978-3-540-27752-1)
- [Granger (1969): Investigating Causal Relations by Econometric Models and Cross-Spectral Methods](https://www.jstor.org/stable/1912791)
- [statsmodels: VAR documentation](https://www.statsmodels.org/stable/vector_ar.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.08_Vector_Autoregression_VAR&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
