# Cointegration & Vector Error Correction Models

> Status: ✅ Complete

Long-run equilibrium relationships between non-stationary series

## Concept & Intuition

Most real-world time series that matter economically — GDP, price levels, exchange rates, stock prices, interest rates — are **non-stationary**: they drift, trend, and have no fixed mean. Topic 08 (VAR) sidestepped this by differencing everything to stationarity before modeling. But differencing throws information away: if two non-stationary series are tied together by a genuine long-run economic relationship, differencing destroys that relationship, leaving only short-run noise.

**Cointegration** asks a sharper question: even though $y_{1t}$ and $y_{2t}$ individually wander off to infinity (they're I(1)), is there some linear combination $y_{1t} - \beta y_{2t}$ that stays put (I(0), stationary)? If so, the two series share a **common stochastic trend** and are bound by a long-run equilibrium — they can drift apart temporarily, but a restoring force always pulls them back together. Two examples that motivate the topic:

1. **Spurious regression.** If you regress two *unrelated* random walks on each other, standard OLS theory is violated and you get "statistically significant" relationships far more often than the nominal 5% rate would suggest — a well-known trap (Granger & Newbold, 1974) for anyone doing macro/finance regressions on levels without checking stationarity first.
2. **Vector Error Correction Models (VECM).** If two series really are cointegrated, the Granger Representation Theorem guarantees they have a VECM representation: a VAR in differences *plus* an error-correction term that measures how far the system is from its long-run equilibrium and pulls it back. Ignoring that term (i.e., fitting a plain VAR-in-differences) throws away real information and costs forecast accuracy.

## Mathematical Explanation

**Cointegration.** $y_{1t}, y_{2t}$ are each I(1). They are cointegrated if there exists $\beta$ such that

$$z_t = y_{1t} - \beta y_{2t}$$

is I(0). $z_t$ is the cointegrating residual (equilibrium error), $\beta$ the cointegrating vector.

**Engle-Granger two-step test (1987).**
1. Regress $y_1$ on $y_2$ by OLS to get $\hat\beta$.
2. Run an ADF test on the residuals $\hat z_t$. If stationary, the series are cointegrated.

Simple but limited: only detects a single relationship, is sensitive to which variable is the regressor, and needs special MacKinnon critical values (not standard ADF tables) because $\hat z_t$ is estimated, not observed.

**Johansen trace test (1991).** Works directly with a VAR-in-levels and tests the **rank** $r$ of the long-run coefficient matrix $\Pi = \alpha\beta'$ via a sequence of likelihood-ratio tests: $H_0: r=0$ vs. $r\ge1$, then $r\le1$ vs. $r\ge2$, etc. Handles more than two series and multiple cointegrating relationships; generally considered the more reliable test.

**Vector Error Correction Model (VECM).** If $y_t$ is cointegrated with rank $r$, the Granger Representation Theorem gives:

$$\Delta y_t = \alpha \beta' y_{t-1} + \sum_{i=1}^{k-1}\Gamma_i \Delta y_{t-i} + \varepsilon_t$$

- $\beta' y_{t-1}$: the lagged equilibrium error.
- $\alpha$ (loading/adjustment matrix): how fast **each** series corrects back toward equilibrium. Near-zero $\alpha_i$ means variable $i$ is **weakly exogenous** — it drives the system without reacting; large $|\alpha_i|$ means that variable does the correcting.
- $\Gamma_i \Delta y_{t-i}$: ordinary short-run VAR dynamics.

A plain VAR fit on $\Delta y_t$ alone (ignoring $\beta' y_{t-1}$) is a special case that assumes $\alpha=0$ — no error correction at all.

## Code Implementation (`17_cointegration_vector_error_correction_models.ipynb`)

### 1. Spurious regression (the motivating danger)

Two **independent** random walks (500 steps) were regressed on each other:

| Quantity | Value |
|---|---|
| OLS slope | 1.1213 |
| t-statistic | 21.05 |
| p-value | 8.0×10⁻⁷¹ |
| R² | 0.4708 |
| ADF on regression residuals | p = 0.4093 (non-stationary) |

A textbook "highly significant" relationship between two series that share nothing but non-stationarity — and the tell: the regression residuals themselves are non-stationary, meaning the fitted relationship doesn't actually hold up over time. Repeating this 200 times with fresh random-walk pairs, a "significant" (p<0.05) regression appeared **181/200 = 90.5%** of the time — 18x the nominal 5% false-positive rate. This is the entire reason cointegration testing exists: **never regress levels of two I(1) series without checking cointegration first.**

### 2. Synthetic validation: recovering a known cointegrating relationship

A bivariate system was built by construction: a shared random-walk common trend, plus a stationary AR(1) "spread" so the true cointegrating vector ($\beta_{true}=1.5$) is known exactly.

| Series | ADF level p-value | ADF diff p-value |
|---|---|---|
| y1 | 0.371 | 0.0 |
| y2 | 0.504 | 7×10⁻⁶ |

Both confirmed I(1). Testing for cointegration:

- **Engle-Granger:** stat = -9.07, p = 5.9×10⁻¹⁴ → cointegrated. Recovered relationship: $y_1 = -0.081 + 1.451 y_2$ (true $\beta=1.5$). ADF on the residual: p = 4.7×10⁻¹⁵ (clearly stationary).
- **Johansen:** trace stat for $r=0$ is 71.6 (vs. 95% critical value 15.5) → strongly rejects, confirming ≥1 cointegrating relationship. Trace stat for $r\le1$ is 2.6 (vs. crit. 3.8) → correctly fails to reject, stopping at **exactly rank 1**, as built.

Both tests agree cleanly here — this is the well-behaved case.

### 3. VECM vs. plain VAR-in-differences (synthetic, known ground truth)

A VECM (`k_ar_diff=1`, `coint_rank=1`) was compared against a VAR fit only on differences, forecasting 20 steps out of sample:

| Horizon | VECM MAE | VAR-in-differences MAE |
|---|---|---|
| 1 | 0.3024 | 0.4008 |
| 5 | 0.6970 | 0.8229 |
| 10 | 0.8505 | 0.7852 |
| 20 (overall) | 0.8030 | 0.8518 |

VECM wins at 3 of 4 horizons and overall (5.7% MAE reduction), but is *worse* at horizon 10 — a useful, honest reminder that "the theoretically correct model always wins" is not a guarantee on any single realization; cumulative forecast error is itself a noisy quantity, so rankings between two reasonable models can flip at particular horizons even when one has a genuine structural edge on average.

### 4. Real data: US real GDP vs. real consumption (1959Q1-2009Q3, quarterly, `statsmodels` macrodata)

Both series (log scale) confirmed I(1): ADF level p-values 0.383 (GDP) and 0.460 (consumption); ADF diff p-values ≈0 and 2.3×10⁻⁵.

**An honest test disagreement:**

| Test | Result |
|---|---|
| Engle-Granger (cons on gdp) | stat=-3.535, p=0.0294 → rejects at 5% |
| Engle-Granger (gdp on cons) | stat=-3.562, p=0.0273 → rejects at 5% (direction-robust) |
| Johansen, k_ar_diff=1 | trace=12.88 vs. 95% crit 15.49 → fails to reject |
| Johansen, k_ar_diff=2 | trace=13.69 vs. 95% crit 15.49 → fails to reject |
| Johansen, k_ar_diff=3 | trace=14.41 vs. 95% crit 15.49 → fails to reject |
| Johansen, k_ar_diff=4 | trace=13.25 vs. 95% crit 15.49 → fails to reject |

Engle-Granger rejects "no cointegration" at the 5% level (consistently across regression direction); Johansen never does, at any of 4 lag orders (VAR lag-order selection on levels: AIC picks 5, BIC picks 2, HQIC picks 3 — none flip the Johansen conclusion). This is a real, documented phenomenon, not a bug: Engle-Granger is known to have size distortions (over-rejects) in samples of this size (~200 quarters), while Johansen is generally the more trustworthy test. With economic theory (the permanent-income hypothesis) providing a strong prior that GDP and consumption share a long-run relationship, and one test confirming it, proceeding with a VECM is defensible — but this is exactly the kind of ambiguous case that requires judgment, not a rubber-stamped test.

**VECM fit on the full sample** (`k_ar_diff=2`, `coint_rank=1`):

- Cointegrating vector: $\log(\text{gdp}) - 0.9494 \times \log(\text{cons}) \approx$ stationary.
- Adjustment speed, GDP equation: $\alpha=0.0061$ (p=0.532, **not significant**) → GDP is weakly exogenous, it drives the system without reacting to past disequilibrium.
- Adjustment speed, consumption equation: $\alpha=0.0471$ (p<0.001, **significant**) → when consumption strays from its long-run relationship with GDP, it is **consumption** that corrects back — consistent with the permanent-income hypothesis (consumption adapts to income/output, not the reverse).

**Walk-forward forecast comparison** (Topic 11 methodology: 8 folds, 4-quarter-ahead forecasts each):

| Fold | VECM MAE | VAR-diff MAE |
|---|---|---|
| 0 | 0.00357 | 0.00295 |
| 1 | 0.00490 | 0.00553 |
| 2 | 0.00448 | 0.00617 |
| 3 | 0.00362 | 0.00199 |
| 4 | 0.00376 | 0.00522 |
| 5 | 0.00210 | 0.00273 |
| 6 | 0.01313 | 0.01543 |
| 7 | 0.01861 | 0.02199 |
| **Overall** | **0.00677** | **0.00775** |

VECM wins 6 of 8 folds and reduces overall MAE by **12.63%**. A Diebold-Mariano test (h=4, absolute loss, callback to Topics 08/11) confirms this is not noise: **stat = -2.29, p = 0.0253**, significant at the 5% level.

**The nuanced finding:** even though the Johansen test could not confidently confirm cointegration at conventional significance, imposing the theoretically-motivated long-run relationship via VECM still produced a statistically significant forecast improvement over a VAR-in-differences. Statistical significance of a specification test and practical forecasting value are related but not identical — the only reliable way to know which model to trust is out-of-sample evaluation.

## Common Pitfalls

1. **Regressing levels of two I(1) series without testing for cointegration first.** ~90% of independent random-walk pairs produced a "significant" OLS relationship at the 5% level — 18x the nominal false-positive rate. Always test for cointegration (or difference first) before trusting a regression between two trending/non-stationary series.
2. **Engle-Granger and Johansen can disagree, and Engle-Granger is the less reliable of the two.** It has known size distortions in finite samples and depends on an arbitrary choice of dependent variable (though here both directions happened to agree). Prefer Johansen, especially with more than 2 series.
3. **Cointegration rank matters as much as its existence.** With more than 2 series, Johansen's sequential trace test ($r=0$, then $r\le1$, etc.) determines *how many* independent long-run equilibria exist; using the wrong rank misspecifies the VECM.
4. **A VECM only helps if the series really are cointegrated.** Imposing a nonexistent equilibrium constraint biases forecasts. Test first, and weigh both statistical evidence and domain theory when tests disagree — as they did for GDP/consumption here.
5. **Adjustment speeds ($\alpha$) reveal causal structure the levels alone hide.** In GDP/consumption, only consumption's $\alpha$ was significant — GDP is weakly exogenous. This is often the most economically meaningful output of the whole exercise.
6. **Deterministic-term choice matters.** `deterministic="ci"` (constant restricted to the cointegrating relation) fits the case where series trend but the equilibrium itself has no trend. Other choices (`"co"`, `"lo"`, `"li"`) encode different structural assumptions and will change both the test outcome and the fitted $\beta$.

## Function Reference

| Function | Purpose |
|---|---|
| `statsmodels.tsa.stattools.adfuller` | ADF unit-root test (Topic 02) |
| `statsmodels.tsa.stattools.coint` | Engle-Granger cointegration test with correct MacKinnon critical values |
| `statsmodels.tsa.vector_ar.vecm.coint_johansen` | Johansen trace/max-eigenvalue cointegration rank test |
| `statsmodels.tsa.vector_ar.vecm.VECM` | Vector Error Correction Model estimation |
| `statsmodels.tsa.api.VAR` | Vector Autoregression, used here on differences as the baseline (Topic 08) |

## Self-Test

1. Why is OLS on the levels of two independent random walks statistically invalid, even though it "runs" without error?
2. What does it mean for a linear combination of two I(1) series to be I(0), and why is that the defining property of cointegration?
3. Name two limitations of the Engle-Granger two-step method that the Johansen test avoids.
4. In the VECM equation $\Delta y_t = \alpha\beta' y_{t-1} + \ldots$, what does a near-zero $\alpha_i$ for series $i$ imply about that series?
5. Why did the notebook still fit and evaluate a VECM on GDP/consumption despite the Johansen test failing to reject "no cointegration"? Was that a good decision?
6. What does the horizon-10 result (VECM slightly worse than VAR-in-differences) in the synthetic experiment tell you about interpreting single out-of-sample comparisons?

## Notebook

[`17_cointegration_vector_error_correction_models.ipynb`](./17_cointegration_vector_error_correction_models.ipynb)

## Further Reading

- Granger, C.W.J. & Newbold, P. (1974). *Spurious Regressions in Econometrics.* Journal of Econometrics.
- Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction: Representation, Estimation, and Testing.* Econometrica.
- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.* Econometrica.
- Hamilton, J.D. (1994). *Time Series Analysis*, Chapters 19-20 (Cointegration).
- [statsmodels VECM documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html)

---
[← Back to Time Series Analysis](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.17_Cointegration_Vector_Error_Correction_Models&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
