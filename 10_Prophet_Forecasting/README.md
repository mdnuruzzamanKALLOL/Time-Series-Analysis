# 🔮 Prophet Forecasting

> Status: ✅ Complete — [Open the notebook →](10_prophet_forecasting.ipynb)

Topic 10 of the Time Series Analysis repo — the first "practical toolkit" topic, and a shift in approach from Topics 05-09. Prophet is an **additive** decomposable model: $y(t)=g(t)+s(t)+h(t)+\varepsilon_t$, where $g(t)$ is a piecewise-linear (or logistic) trend, $s(t)$ is seasonality built from Fourier series (allowing *multiple simultaneous* seasonal periods — something SARIMA in Topic 07 fundamentally cannot do with its single seasonal period), and $h(t)$ is holiday/event effects. This notebook builds the piecewise-linear trend and Fourier seasonality from scratch, validates them against Prophet itself on synthetic data with known ground truth, then applies the full toolkit to the real, famous Peyton Manning Wikipedia-pageviews dataset — finding genuine football-season seasonality, real Super Bowl spikes via the holiday-effects API, and an honest look at what Prophet's changepoint regularization actually buys (and costs).

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
Topics 05-09: ARIMA-family models          -> ONE seasonal period at a time (SARIMA), stationarity required
Topic 10:     Prophet's additive model     -> ANY number of seasonal periods at once, no stationarity required

y(t) = g(t) + s(t) + h(t) + noise
       |       |      |
       |       |      +-- one-off events (e.g. a Super Bowl appearance)
       |       +--------- Fourier series, one per seasonal period (e.g. weekly AND yearly together)
       +----------------- piecewise-linear trend, slope allowed to change at a set of candidate changepoints

Both g(t) and s(t) are LINEAR in their parameters given fixed changepoints,
so a point estimate is just one least-squares fit -- no need for the
ADF/ACF/PACF order-selection machinery of Topics 02-07.
```

---

## 🎯 Why This Topic Matters

- **Prophet's piecewise-linear trend and Fourier seasonality were built entirely from scratch and validated against Prophet itself on synthetic data with known ground truth** — §2 found a manual `numpy.lstsq` fit recovering a true trend-slope change of 0.02→0.08 as **0.0200→0.0800** exactly, with fitted values correlating with an actual `Prophet` fit at **0.99999994** and MAE differing by only **8.84e-05**.
- **On real data, the manually-built weekly and yearly seasonal components matched Prophet's to near-perfection** — §4 found a weekly-component correlation of **1.0000** and yearly-component correlation of **0.9940** against the real Peyton Manning Wikipedia-pageviews series, with the only meaningful gap coming from the trend term (regularized in Prophet, not in the manual OLS fit).
- **A genuine, interpretable weekly-seasonality finding, not just "there is a weekly pattern"** — §5 found Wikipedia searches for a football player peaking on **Monday** (+0.352 on the log scale), not Sunday (the actual game day), because searches spike *after* game recaps circulate — with Saturday, the day before games, the lowest (-0.312).
- **Real one-off event effects were captured directly through the `holidays` API, using Peyton Manning's actual, documented Super Bowl appearances** — §6 found an estimated effect of **+3.42 on the log scale** the day after the February 2014 Super Bowl (a historically lopsided 8-43 loss that dominated the news), implying roughly **31×** the normal page-view count that day.
- **A component ablation quantified exactly how much each piece of the additive model contributes, and confirmed the full model beats a simple benchmark** — §7 found held-out MAE of **0.785** (trend only) → **0.760** (+ weekly) → **0.417** (+ yearly, the single biggest jump, reflecting football season's dominance) — decisively beating a same-day-last-year naive benchmark's **0.598**.
- **Prophet's changepoint sparsity (its Laplace-prior regularization) was confirmed directly, not just assumed from the docs** — §8 found only **11 of 25** candidate changepoints exceeding a small delta threshold at the default `changepoint_prior_scale=0.05`, with several deltas landing at exactly **0.0000**.
- **An honest, dataset-specific finding that "more regularization" isn't automatically better** — §8 found that loosening `changepoint_prior_scale` from the default 0.05 to 5.0 (activating all 25 changepoints) *improved* held-out MAE from **0.417 to 0.337** on this real series — a real athlete's career genuinely has more trend flexibility than the smooth default assumes, echoing the same "don't trust one metric/default blindly" lesson from Topics 05-06.

---

## 🧮 Mathematical Explanation

### 1. Piecewise-linear trend

$$g(t) = \left(k+\sum_{j:\,t>s_j}\delta_j\right)t + \left(m+\sum_{j:\,t>s_j}\gamma_j\right), \qquad \gamma_j=-s_j\delta_j$$

Reparameterized here via a hinge (ReLU) basis, $g(t)=m+kt+\sum_j\delta_j\max(0,t-s_j)$, which is automatically continuous at every candidate changepoint $s_j$ and linear in $(m,k,\delta_1,\dots,\delta_J)$.

### 2. Fourier-series seasonality

$$s(t)=\sum_{n=1}^{N}\left(a_n\cos\frac{2\pi nt}{P}+b_n\sin\frac{2\pi nt}{P}\right)$$

One such block per seasonal period $P$ (7 for weekly, 365.25 for yearly) — additively combined, which is exactly what lets Prophet model both at once where SARIMA's single seasonal period cannot.

### 3. Full additive model as one linear regression

$$y(t)=g(t)+s_{\text{weekly}}(t)+s_{\text{yearly}}(t)+h(t)+\varepsilon_t$$

Every term above is linear in its own parameters, so stacking all their basis columns into one design matrix $X$ and solving $\hat\beta=\arg\min_\beta\|y-X\beta\|^2$ recovers a full point estimate — the manual `numpy.lstsq` fit used throughout §2 and §4.

### 4. Laplace-prior changepoint regularization

$$\delta_j\sim\text{Laplace}(0,\tau), \qquad \tau=\texttt{changepoint\_prior\_scale}$$

Prophet's actual fitting (MAP estimation under this Laplace prior) shrinks most $\delta_j$ toward exactly zero — the sparsity directly confirmed in §8.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Piecewise-linear trend (hinge basis) | $g(t)=m+kt+\sum_j\delta_j\max(0,t-s_j)$ | §1, §2, §4 |
| Fourier seasonality | $s(t)=\sum_n a_n\cos(2\pi nt/P)+b_n\sin(2\pi nt/P)$ | §1, §2, §4 |
| Full additive model | $y(t)=g(t)+s_w(t)+s_y(t)+h(t)+\varepsilon_t$ | §2, §4, §7 |
| Laplace-prior changepoint shrinkage | $\delta_j\sim\text{Laplace}(0,\tau)$ | §8 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Treating the default `changepoint_prior_scale=0.05` as always correct** — §8 found a real series where loosening it from 0.05 to 5.0 *improved* held-out MAE (0.417 → 0.337); it is a hyperparameter to tune per-series, not a fixed constant.
2. **Reading "better in-sample fit" as "better model"** — §4 found the unregularized manual OLS trend fitting training data slightly better than Prophet's regularized trend; that gap is regularization doing its job, not a bug in Prophet.
3. **Ignoring that many "seasonal" real-world series have holiday-like one-off events that seasonality alone can't capture** — §6's Super Bowl spikes (+1.4 to +3.4 on a log scale) sit far outside what any smooth Fourier seasonality could represent; the `holidays` argument exists specifically for this.
4. **Applying Prophet to series shorter than ~2 full seasonal cycles** — an earlier synthetic run in developing this notebook with only 2 years of daily data produced a badly confounded trend/yearly-seasonality fit (a changepoint placed exactly at the 1-year mark aliased with the yearly Fourier term); §2 deliberately used 4 years to keep the seasonal component properly identified.
5. **Assuming Prophet's automatic seasonality/holiday components are additive on the original scale when `y` is already log-transformed (as here)** — every effect discussed in this notebook (weekly, yearly, holiday) is additive **in log-page-view space**, so converting a holiday effect back to "how many more page views" requires exponentiating, as done in §6.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `prophet.Prophet` | Fits the full additive trend + seasonality + holidays model via MAP/MCMC |
| `Prophet.fit`, `Prophet.predict` | Fit on historical `(ds, y)` data, predict components + `yhat` |
| `Prophet(changepoints=...)` / `n_changepoints=` | Manually specify or auto-place candidate trend changepoints |
| `Prophet(holidays=...)` | Custom one-off event effects with `lower_window`/`upper_window` |
| `Prophet(changepoint_prior_scale=...)` | Controls the Laplace-prior regularization strength on trend flexibility |
| Custom `relu_basis()` / `fourier_series()` | From-scratch hinge trend and Fourier seasonality basis functions |

---

## 📝 Self-Test Exercises

1. Section 2 found the manual OLS trend and Prophet's trend matching almost exactly (correlation 0.99999994) on synthetic data with a single, obvious changepoint. Section 4 found a bigger gap (correlation 0.9805) on the real, noisier Peyton Manning data with 25 candidate changepoints. Using the Laplace-prior formula in §4 of the Mathematical Explanation, explain why more candidate changepoints and more real-world noise make the two fits diverge more.
2. Section 5 found Monday, not Sunday, as the peak day for a football player's Wikipedia searches. Propose what weekly pattern you would instead expect from a topic like "grocery store hours" or "restaurant reservations," and why the driving mechanism (search timing vs. the event itself) would differ.
3. Section 7's component ablation found "trend + weekly" (MAE 0.760) barely better than "trend only" (MAE 0.785), while adding yearly seasonality dropped MAE to 0.417. Using the day-of-week effect magnitudes from Section 5 (roughly ±0.35 on a log scale) versus the implied yearly amplitude, explain quantitatively why yearly seasonality dominates this particular series.
4. Section 8 found that loosening `changepoint_prior_scale` from 0.05 to 5.0 improved this series' held-out MAE. Propose a rule of thumb (not "try every value") for when a real-world series is likely to need looser trend regularization than Prophet's default, based on what you know about Peyton Manning's actual career.
5. Section 6 noted the 2007 and 2016 Super Bowl dates fall outside this dataset's range and contribute nothing to the fitted holiday effect. If you were forecasting *forward* past 2016-01-20 with this fitted model, propose what would happen to a future Super Bowl date that was included in the `holidays` dataframe but never appeared in the training data.

---

## 📓 Notebook

16 executed code cells: Prophet's piecewise-linear trend and Fourier-series seasonality built entirely from scratch and validated against an actual `Prophet` fit on synthetic data with known ground truth (correlation 0.99999994); applied to the real, canonical Peyton Manning Wikipedia-pageviews dataset (2007-2016, downloaded directly from Prophet's own GitHub repository) — matching Prophet's weekly and yearly seasonal components almost exactly, finding a genuine Monday-peak weekly pattern, capturing real Super Bowl page-view spikes through the `holidays` API, quantifying each model component's contribution via ablation against a naive benchmark, and honestly showing that Prophet's default changepoint regularization is not automatically optimal for every real series:

➡️ **[10_prophet_forecasting.ipynb](10_prophet_forecasting.ipynb)**

*Dataset: [`peyton_manning_wiki_pageviews.csv`](peyton_manning_wiki_pageviews.csv) — daily log(Wikipedia page views), 2007-12-10 to 2016-01-20, downloaded from the official [`facebook/prophet`](https://github.com/facebook/prophet) example data and saved locally for reproducibility.*

---

## 📚 Further Reading

- [Taylor & Letham (2018): Forecasting at Scale](https://peerj.com/preprints/3190/)
- [Prophet official documentation](https://facebook.github.io/prophet/)
- [Prophet GitHub repository](https://github.com/facebook/prophet)
- [Prophet's own quick-start example (the same Peyton Manning dataset used here)](https://facebook.github.io/prophet/docs/quick_start.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.10_Prophet_Forecasting&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
