# 🚨 Anomaly Detection in Time Series

> Status: ✅ Complete — [Open the notebook →](14_anomaly_detection_in_time_series.ipynb)

Topic 14 of the Time Series Analysis repo. Anomaly detection asks a fundamentally different question than the forecasting-focused topics so far: not "what will happen next," but "does this point (or period) fit the pattern the rest of the series establishes?" This notebook builds threshold-based detectors from scratch (z-score and its robust MAD-based cousin), shows why *contextual* anomalies on trending/seasonal data require decomposition first, reuses Topic 12's Kalman filter innovations as a naturally causal detector, and closes with a real S&P 500 application that reuses Topic 09's GARCH model to show that whether a given return counts as "anomalous" depends entirely on the volatility regime it occurred in — not on its raw size.

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
Point anomaly:        a single observation far from the rest       (a one-day 10-sigma spike)
Contextual anomaly:   normal in general, abnormal for its CONTEXT   (68F is fine in July, bizarre in January)
Collective anomaly:   individually normal points, abnormal as a GROUP (a flatline where variation is expected)

A global threshold can only ever catch POINT anomalies.
Contextual anomalies require removing whatever defines "context" first:
  - trend/season   -> STL decomposition, then threshold the residual
  - current level   -> Kalman filter, then threshold the standardized innovation
  - volatility regime -> GARCH, then threshold the standardized residual
```

---

## 🎯 Why This Topic Matters

- **The "masking effect" was demonstrated and precisely quantified, not just described** — §2 found a 30-point contiguous anomaly cluster inflating the standard deviation from ~1 to **5.698**, which completely masked 3 separate, genuinely large isolated anomalies (z-score recall dropped to **0.485**), while the robust MAD-based detector — immune to up to 50% contamination — caught every single one (precision=recall=F1=**1.000**).
- **Global and local thresholds were shown failing completely on real contextual anomalies** — §3 found both a global z-score and a 13-month centered rolling z-score detecting **zero of four** injected 30-35% shocks on real AirPassengers data, because the series' own seasonal swing dwarfs the shock in both raw and locally-windowed terms.
- **STL decomposition recovered detection, but only after confronting a real, non-obvious pitfall** — §3 found the default MAD threshold (3.5) catching all 4 true anomalies but at precision **0.364** due to LOESS smoothing spreading each anomaly's influence to its neighbors; a clear numerical gap in the sorted scores (true anomalies at |z|>14, largest false positive at |z|~6) justified a higher, data-informed threshold that recovered perfect precision and recall.
- **Topic 12's Kalman filter was reused as a causal anomaly detector requiring zero threshold tuning** — §4 found its standardized one-step-ahead innovations catching all 4 anomalies at precision=recall=F1=**1.000** using the generic threshold of 3.0 — no data-specific calibration needed, and unlike STL, genuinely usable in real-time monitoring since it needs no future data.
- **A real, decisive contextual-anomaly finding was uncovered on real S&P 500 data by reusing Topic 09's GARCH model** — §5 found a naive static return threshold flagging 29 days (28 of them clustered almost entirely inside the already-volatile 2008 crisis), while the GARCH-conditional threshold flagged a different set of 9 days — including Feb 2018's Volmageddon (-4.10%, cond. vol only 0.93) and the 2007 Chinese-market-correction spillover (-3.47%, cond. vol only 0.49) — both events the naive method missed entirely because they occurred during unusually *calm* periods.

---

## 🧮 Mathematical Explanation

### 1. Standard z-score (non-robust)

$$z_t = \frac{y_t - \bar y}{s}, \qquad \bar y = \frac{1}{n}\sum_i y_i,\ \ s = \sqrt{\frac{1}{n}\sum_i (y_i-\bar y)^2}$$

flag $|z_t| > \tau$; both $\bar y$ and $s$ can themselves be corrupted by the anomalies being searched for.

### 2. Robust modified z-score (MAD-based)

$$M_t = \frac{0.6745\,(y_t - \tilde y)}{\text{MAD}}, \qquad \tilde y = \text{median}(y), \qquad \text{MAD} = \text{median}(|y_i - \tilde y|)$$

the constant $0.6745$ makes MAD comparable to the standard deviation under normality; median and MAD have a 50% breakdown point.

### 3. STL residual-based detection

$$y_t = T_t + S_t + R_t \qquad \Longrightarrow \qquad \text{apply the z-score or MAD z-score to } R_t \text{ instead of } y_t$$

### 4. Kalman filter standardized innovations (Topic 12 reused)

$$e_t = y_t - \hat y_{t|t-1}, \qquad S_t = \text{Var}(e_t) \qquad \Longrightarrow \qquad \nu_t = \frac{e_t}{\sqrt{S_t}} \sim N(0,1) \text{ under a correct model}$$

flag $|\nu_t| > 3$; causal by construction since $\hat y_{t|t-1}$ uses only data strictly before $t$.

### 5. GARCH-conditional standardized residual (Topic 09 reused)

$$\hat\varepsilon_t = \frac{r_t - \hat\mu}{\hat\sigma_t} \qquad \text{where } \hat\sigma_t^2=\hat\omega+\hat\alpha \varepsilon_{t-1}^2+\hat\beta\hat\sigma_{t-1}^2$$

flag $|\hat\varepsilon_t| > \tau$; the same raw return $r_t$ can be flagged or not depending entirely on $\hat\sigma_t$, the *current* volatility regime.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Standard z-score | $z_t=(y_t-\bar y)/s$ | §2 |
| Robust MAD z-score | $M_t=0.6745(y_t-\tilde y)/\text{MAD}$ | §2, §3 |
| STL decomposition | $y_t=T_t+S_t+R_t$ | §3 |
| Kalman filter innovation | $\nu_t=e_t/\sqrt{S_t}$ | §4 |
| GARCH-standardized residual | $\hat\varepsilon_t=(r_t-\hat\mu)/\hat\sigma_t$ | §5 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Using the sample mean/std for thresholding when anomalies themselves may be present** — §2 found a 30-point anomaly cluster inflating the standard deviation enough to completely mask 3 separate, genuinely large (6-sigma-if-clean) isolated anomalies; the MAD-based detector caught all of them.
2. **Applying a global or short rolling-window threshold to trending/seasonal data** — §3 found both approaches missing 100% of four injected 30%+ shocks on AirPassengers, because the series' own seasonal swing dwarfs the shock in both absolute and locally-windowed terms.
3. **Using a fixed, generic threshold (like 3.5 MAD) on a decomposition residual without checking it** — §3 found STL's LOESS smoothing spreading a real anomaly's influence into neighboring points, producing extra false positives at the generic threshold; sorting the anomaly scores and looking for a genuine gap (14+ vs. ~6 here) is more reliable than a fixed rule.
4. **Preferring a non-causal detector for a task that needs real-time monitoring** — §4 found STL's decomposition requiring the full series (including future points) to compute a residual at time $t$, while the Kalman filter's standardized innovations are one-step-ahead by construction and available immediately as new data arrives, at no cost in accuracy on this series.
5. **Judging "anomalous" by the raw size of a change alone** — §5 found a naive threshold flagging almost exclusively 2008-crisis days (huge moves during an already-volatile regime) while completely missing days like Feb 2018's Volmageddon and the 2007 Chinese-market-correction spillover, both unusual specifically because they occurred during unusually calm periods; the GARCH-conditional approach is exactly Topic 09's volatility model repurposed as a contextual anomaly detector.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| Custom `zscore_detector()` | From-scratch non-robust z-score point-anomaly detector |
| Custom `mad_zscore_detector()` | From-scratch robust MAD-based modified z-score detector |
| Custom `rolling_zscore()` | From-scratch centered rolling-window contextual detector (offline use only) |
| `statsmodels.tsa.seasonal.STL` | Library seasonal-trend decomposition via LOESS |
| `statsmodels.tsa.statespace.structural.UnobservedComponents` | Kalman-filter-based structural model, reused from Topic 12, for `.standardized_forecasts_error` |
| `arch.arch_model` | GARCH(1,1)-t volatility model, reused directly from Topic 09 |
| Custom `precision_recall_f1()` | Ground-truth evaluation metric for detector comparison |

---

## 📝 Self-Test Exercises

1. Section 2 found the standard z-score's recall dropping to 0.485 once a 30-point anomaly cluster inflated the standard deviation. Using the z-score formula in §1 of the Mathematical Explanation, explain algebraically why a *larger* contaminating cluster (fixed anomaly magnitude, increasing count) eventually starts to reduce even the MAD-based detector's performance, once contamination exceeds its 50% breakdown point.
2. Section 3 found both a global and a rolling z-score missing all four injected AirPassengers shocks. Propose what rolling-window size (relative to the 12-month seasonal period) would be needed for a rolling z-score to have any chance of detecting a contextual anomaly, and explain the trade-off that window size choice creates against detecting genuinely slow, collective anomalies.
3. Section 3 found STL residual spillover creating false positives at neighboring points. Using the STL decomposition formula in §3 of the Mathematical Explanation, explain mechanically why a LOESS-smoothed trend or seasonal component would be pulled toward a single extreme point at $t$, thereby contaminating the residual $R_{t-1}$ and $R_{t+1}$ as well.
4. Section 4 found the Kalman filter's standardized innovations needing no threshold tuning, unlike STL's residual. Referencing Topic 12's log-likelihood formula (prediction-error decomposition), explain why $\nu_t=e_t/\sqrt{S_t}$ is expected to be approximately $N(0,1)$ under a correctly specified model, while STL's residual has no equivalent theoretical guarantee.
5. Section 5 found 8 days flagged only by the GARCH-conditional method, several occurring in the mid-2010s. Propose one alternative real-world dataset (outside finance) where a "calm regime" vs. "volatile regime" distinction like Topic 09's GARCH model would matter just as much for anomaly detection, and describe what conditional-variance model you would reuse or adapt to detect anomalies in it.

---

## 📓 Notebook

13 executed code cells, 3 plots: from-scratch z-score and MAD-based robust anomaly detectors validated on synthetic data (demonstrating and precisely quantifying the "masking effect"); a real demonstration that global and rolling thresholds miss 100% of injected contextual anomalies on trending/seasonal AirPassengers data; STL decomposition recovering perfect detection after confronting a genuine LOESS-spillover pitfall; Topic 12's Kalman filter innovations reused as a causal, tuning-free alternative; and a real-world S&P 500 application reusing Topic 09's exact GARCH(1,1)-t model to show that naive and volatility-conditional anomaly detection disagree on the vast majority of flagged days, each catching genuinely different, real historical events:

➡️ **[14_anomaly_detection_in_time_series.ipynb](14_anomaly_detection_in_time_series.ipynb)**

---

## 📚 Further Reading

- [Hochenbaum, Vallis & Kejariwal (Twitter): Automatic Anomaly Detection in the Cloud via Statistical Learning](https://arxiv.org/abs/1704.07706)
- [Iglewicz & Hoaglin: Volume 16: How to Detect and Handle Outliers (the modified z-score / MAD method)](https://asq.org/quality-press/display-item?item=H1210)
- [Cleveland et al. (1990): STL: A Seasonal-Trend Decomposition Procedure Based on Loess](https://www.wessa.net/download/stl.pdf)
- [Numenta Anomaly Benchmark (NAB) — real labeled time series anomaly datasets](https://github.com/numenta/NAB)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.14_Anomaly_Detection_in_Time_Series&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
