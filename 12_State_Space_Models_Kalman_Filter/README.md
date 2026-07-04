# 🧭 State Space Models & Kalman Filter

> Status: ✅ Complete — [Open the notebook →](12_state_space_models_kalman_filter.ipynb)

Topic 12 of the Time Series Analysis repo — the "unifying framework" the learning path promised. A state-space model represents a series through unobserved (latent) states evolving over time, observed only through noisy measurements; the Kalman filter is the optimal recursive algorithm for estimating those states. This notebook builds the Kalman filter and its backward-pass companion, the RTS smoother, entirely from scratch and matches both to `statsmodels.tsa.statespace.structural.UnobservedComponents`, then reveals two structural connections hiding in plain sight: a local-level Kalman filter's steady-state gain **is** Topic 04's simple exponential smoothing constant, and a state-space Basic Structural Model achieves accuracy statistically indistinguishable from Topic 07's SARIMA on the same real AirPassengers walk-forward evaluation from Topic 11 — using a completely different mathematical framework to get there.

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
Every observed series y_t is treated as a noisy window onto a HIDDEN state x_t:

  state equation:        x_t = F x_(t-1) + w_t     (how the hidden truth evolves)
  observation equation:  y_t = H x_t     + v_t     (how we noisily observe it)

Kalman filter:  predict x_t from x_(t-1), then correct using y_t  -> "filtered" estimate (uses only the past)
RTS smoother:   run the filter forward, then a second pass backward -> "smoothed" estimate (uses the whole series)

Local level model  (1 hidden state: level)          <-> Topic 04's Simple Exponential Smoothing
Local linear trend (2 hidden states: level + trend) <-> Topic 04's Holt's Linear Method
Basic Structural Model (level + trend + seasonal)   <-> a state-space alternative to Topic 07's SARIMA
```

---

## 🎯 Why This Topic Matters

- **The Kalman filter's MLE-fitted variances matched `statsmodels`' own state-space model to 4 decimal places** — §2 found a manual `scipy.optimize` fit giving $(\hat Q,\hat R)=($**0.1877**, **1.0410**$)$, identical to `UnobservedComponents(level="local level")`'s own fitted $(\sigma^2_{\text{level}},\sigma^2_{\text{irregular}})$, with filtered-state correlation **1.00000000** and a max absolute difference of **1.3e-05**.
- **A hidden connection between two "different" topics was found, not just asserted** — §3 found the Kalman filter's steady-state gain ($K_{ss}=$**0.3439**) essentially identical to Topic 04's independently-derived, SSE-optimal simple exponential smoothing constant ($\alpha=$**0.3434**) — a difference of just **0.0005**, despite the two being fit by completely different procedures (state-space MLE vs. one-step-ahead SSE minimization).
- **The 2-state local linear trend model — the Kalman-filter analogue of Holt's linear method — matched `statsmodels` just as closely** — §4 found filtered level and trend correlations of **1.000000** against `UnobservedComponents(level="local linear trend")`, with all three fitted variances matching to within rounding.
- **The RTS smoother, built from scratch, matched `statsmodels`' `smoothed_state` output exactly, and quantified a real accuracy gain from using the future** — §5 found a smoothed-state correlation of **1.000000** (max diff **1.3e-05**) against `statsmodels`, with smoothing reducing MAE by **19.7%** over filtering alone by using information the filter, by construction, cannot.
- **The Kalman filter's native handling of missing data was shown decisively beating naive imputation** — §6 found reconstructing 30 artificially-missing points at **2.5x** the accuracy of simple mean-fill, with no separate imputation step at all — just skipping the update step when an observation is absent.
- **A real state-space model and a real ARIMA-family model were shown to be statistically indistinguishable on the same real data, using the same rigorous methodology from Topic 11** — §7-§8 found a Basic Structural Model's walk-forward MAE (**18.699**, MASE **0.6403**) narrowly beating Topic 07/11's SARIMA (**19.551**, MASE **0.6695**) on AirPassengers, but Topic 11's own Diebold-Mariano test found this difference **not statistically significant** (p=**0.7638**) — two entirely different mathematical frameworks reaching the same practical destination.

---

## 🧮 Mathematical Explanation

### 1. The general linear-Gaussian state-space model

$$x_t = Fx_{t-1}+w_t,\ w_t\sim N(0,Q) \qquad y_t = Hx_t+v_t,\ v_t\sim N(0,R)$$

### 2. Kalman filter recursion

$$\text{Predict: } \hat x_{t|t-1}=F\hat x_{t-1|t-1}, \quad P_{t|t-1}=FP_{t-1|t-1}F^\top+Q$$

$$\text{Update: } K_t=\frac{P_{t|t-1}H^\top}{HP_{t|t-1}H^\top+R}, \quad \hat x_{t|t}=\hat x_{t|t-1}+K_t(y_t-H\hat x_{t|t-1}), \quad P_{t|t}=(I-K_tH)P_{t|t-1}$$

### 3. RTS smoother (backward pass)

$$J_t=\frac{P_{t|t}}{P_{t+1|t}}, \qquad \hat x_{t|n}=\hat x_{t|t}+J_t(\hat x_{t+1|n}-\hat x_{t+1|t})$$

Uses the entire series (both past and future relative to $t$), unlike the filter which only uses data up to $t$.

### 4. Steady-state gain = exponential smoothing

For the scalar local-level model, the discrete Riccati recursion converges to a fixed point $P_{ss}$, giving a constant gain:

$$K_{ss}=\frac{P_{ss}}{P_{ss}+R} \qquad \Longrightarrow \qquad \hat x_{t|t}=\hat x_{t-1|t-1}+K_{ss}\left(y_t-\hat x_{t-1|t-1}\right)$$

which is *identical in form* to Topic 04's simple exponential smoothing update with $\alpha=K_{ss}$.

### 5. Prediction-error decomposition log-likelihood

$$\ln L=-\frac12\sum_{t=1}^n\left[\ln(2\pi S_t)+\frac{e_t^2}{S_t}\right], \qquad e_t=y_t-H\hat x_{t|t-1},\ S_t=HP_{t|t-1}H^\top+R$$

used throughout to fit $Q$ and $R$ by maximum likelihood, exactly as `statsmodels` does internally.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| State/observation equations | $x_t=Fx_{t-1}+w_t,\ y_t=Hx_t+v_t$ | §1, §4 |
| Kalman gain | $K_t=P_{t\|t-1}H^\top/(HP_{t\|t-1}H^\top+R)$ | §1, §2 |
| RTS smoother gain | $J_t=P_{t\|t}/P_{t+1\|t}$ | §5 |
| Steady-state gain = SES alpha | $K_{ss}=P_{ss}/(P_{ss}+R)$ | §3 |
| Prediction-error log-likelihood | $-\frac12\sum[\ln(2\pi S_t)+e_t^2/S_t]$ | §2, §4 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Initializing $P_0$ too small** — an overconfident initial state variance makes the filter slow to correct early mistakes; this notebook used a large diffuse-like $P_0=10^6$ throughout, matching how `statsmodels` handles diffuse initialization for a genuinely unknown starting state.
2. **Treating the Kalman filter and exponential smoothing as unrelated techniques** — §3 found the MLE-fitted local-level filter's steady-state gain (0.3439) matching Topic 04's independently-optimized SES alpha (0.3434) to within 0.0005; they are the same estimator viewed through two different derivations.
3. **Forgetting that a lower MAE needs a significance test** — §8 found the Kalman-filter-based BSM's slightly better raw MAE against SARIMA failing to reach significance under Topic 11's own Diebold-Mariano test (p=0.7638); architectural novelty is not automatically a proven forecasting advantage.
4. **Using the filtered state where the smoothed state is available and appropriate** — §5 found a 19.7% MAE improvement from smoothing; the filtered state is only the right choice for genuinely real-time/online applications where future data truly isn't available yet.
5. **Imputing missing values before modeling instead of letting the filter handle them** — §6 found the Kalman filter's native missing-data handling (skip the update step) beating naive mean-imputation by 2.5x, with no separate imputation step required at all.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `statsmodels.tsa.statespace.structural.UnobservedComponents` | Library Kalman-filter-based structural time series models (local level, local linear trend, seasonal) |
| `statsmodels.tsa.statespace.sarimax.SARIMAX` | Real SARIMA fitting, reused directly from Topics 07/11 for comparison |
| `UnobservedComponents.filtered_state` / `.smoothed_state` | Library filtered and RTS-smoothed state estimates |
| Custom `kalman_filter_local_level()` / `kalman_filter_local_linear_trend()` | From-scratch Kalman filter for 1D and 2D states |
| Custom `rts_smoother()` | From-scratch RTS backward-pass smoother |
| Custom `steady_state_gain()` | From-scratch discrete Riccati equation fixed-point solver |
| Custom `diebold_mariano_test()` | Topic 11's forecast-comparison test, reused directly |

---

## 📝 Self-Test Exercises

1. Section 2 found the Kalman filter's filtered-state correlation with `statsmodels` at 1.00000000, essentially perfect. Using the Kalman gain formula in §2 of the Mathematical Explanation, explain why two implementations using the *same* $(Q,R)$ must produce the same filtered states regardless of implementation details, as long as both correctly follow the recursion.
2. Section 3 found the Kalman filter's steady-state gain matching Topic 04's SES-optimal alpha almost exactly. Propose what would happen to this match if the underlying series' true $Q$ (state noise) were much larger relative to $R$ (observation noise) — would $K_{ss}$ (and therefore the implied optimal alpha) move toward 0 or toward 1, and why?
3. Section 5 found the RTS smoother improving on the filter by 19.7%. Using the smoother gain formula $J_t=P_{t|t}/P_{t+1|t}$ in §3 of the Mathematical Explanation, explain what happens to $J_t$ (and therefore how much the smoother corrects the filtered estimate) as $t$ approaches the very end of the series.
4. Section 6 found the Kalman filter handling missing data by simply skipping the update step. Using the predict-step formula in §2 of the Mathematical Explanation, explain what happens to $P_{t|t-1}$ during a long run of consecutive missing observations, and what that implies about how much you should trust the filter's state estimate after such a gap.
5. Section 8 found a Basic Structural Model and SARIMA statistically indistinguishable on AirPassengers (p=0.7638). Propose one practical (non-accuracy) reason an analyst might still prefer the state-space model over SARIMA for this specific series, referencing what the state-space model's explicit level/trend/seasonal decomposition gives you that SARIMA's differencing-based approach does not.

---

## 📓 Notebook

13 executed code cells: a from-scratch Kalman filter and RTS smoother for both a local level and a local linear trend state-space model, matching `statsmodels`' `UnobservedComponents` to near machine precision on every check (MLE-fitted variances, filtered states, and smoothed states); a direct numerical demonstration that the Kalman filter's steady-state gain **is** Topic 04's simple exponential smoothing constant; native missing-data handling beating naive imputation by 2.5x; and a real walk-forward comparison (reusing Topic 11's exact methodology and Diebold-Mariano test) showing a state-space Basic Structural Model statistically indistinguishable in accuracy from Topic 07's SARIMA on real AirPassengers data:

➡️ **[12_state_space_models_kalman_filter.ipynb](12_state_space_models_kalman_filter.ipynb)**

---

## 📚 Further Reading

- [Kalman (1960): A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552)
- [Rauch, Tung & Striebel (1965): Maximum Likelihood Estimates of Linear Dynamic Systems](https://doi.org/10.2514/3.3166)
- [Durbin & Koopman: Time Series Analysis by State Space Methods](https://academic.oup.com/book/26707)
- [`statsmodels` UnobservedComponents documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.12_State_Space_Models_Kalman_Filter&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
