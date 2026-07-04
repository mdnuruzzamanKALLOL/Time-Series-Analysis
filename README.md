# 📈 Statistical Machine Learning — Time Series Analysis

Part of the [📘 Statistical Machine Learning for Noob](https://github.com/mdnuruzzamanKALLOL) series — a from-scratch, math-first, fully-executed ML study series designed to be read on GitHub, run locally, and actually learned from (not just skimmed).

Every method in the [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) repo assumed independent, identically distributed rows — this repo is where that assumption breaks on purpose. Time-ordered data needs its own toolkit (stationarity, autocorrelation, seasonal structure), and its own validation discipline (the Model Evaluation & Tuning topic's `TimeSeriesSplit` warning gets its full treatment here).

## 📑 Table of Contents

1. [Why This Repo Exists](#-why-this-repo-exists)
2. [Learning Path](#-learning-path)
3. [Topics](#-topics)
4. [Repository Structure](#-repository-structure)
5. [Prerequisites](#-prerequisites)
6. [How to Use](#-how-to-use)
7. [Datasets](#-datasets)
8. [The Full Series](#-the-full-series)
9. [Self-Test Philosophy](#-self-test-philosophy)
10. [Feedback & Corrections](#-feedback--corrections)
11. [Related](#-related)

---

## 🎯 Why This Repo Exists

Shuffling time-ordered data before cross-validating, as almost every generic ML tutorial does by default, silently lets a model "see the future" — a leakage bug specific to this data type. This repo builds classical time series methods from the ground up (decomposition, ARIMA family, GARCH, state-space models) alongside the validation and forecasting-evaluation discipline that keeps results honest, ending with a bridge into neural sequence models for forecasting.

## 🗺️ Learning Path

```
01-04 Structure & Preparation        ->  decomposition, stationarity, autocorrelation, smoothing
        v
05-09 Classical Forecasting Models   ->  AR/MA/ARMA, ARIMA, SARIMA, VAR, GARCH
        v
10-14 Practical Forecasting Toolkit  ->  Prophet, proper CV, Kalman filters, feature engineering, anomaly detection
        v
15-20 Advanced & Neural Methods      ->  LSTM/GRU forecasting, spectral analysis, cointegration, DTW,
                                          change point detection, multi-step/ensemble forecasting
```

Each topic explicitly says what it's setting up for — read the "Why This Topic Matters" section at the top of every README once topics are built out.

## 📚 Topics

| # | Topic | Key Concepts | Status |
|---|-------|-------------|:---:|
| 01 | [Time Series Components & Decomposition](01_Time_Series_Components_Decomposition/) | Trend, seasonality, and residual decomposition | ✅ |
| 02 | [Stationarity & Unit Root Tests](02_Stationarity_Unit_Root_Tests/) | ADF and KPSS tests for a stable statistical structure over time | ✅ |
| 03 | [Autocorrelation & Partial Autocorrelation](03_Autocorrelation_Partial_Autocorrelation/) | ACF/PACF plots and what they reveal about model order | ✅ |
| 04 | [Moving Averages & Exponential Smoothing](04_Moving_Averages_Exponential_Smoothing/) | Simple, double, and triple (Holt-Winters) smoothing | ✅ |
| 05 | [AR, MA, ARMA Models](05_AR_MA_ARMA_Models/) | The building blocks of classical time series models | ✅ |
| 06 | [ARIMA](06_ARIMA/) | Combining autoregression, differencing, and moving averages | ✅ |
| 07 | [SARIMA](07_SARIMA/) | ARIMA extended with seasonal components | ✅ |
| 08 | [Vector Autoregression (VAR)](08_Vector_Autoregression_VAR/) | Modeling multiple interdependent time series jointly | ✅ |
| 09 | [GARCH (Volatility Modeling)](09_GARCH_Volatility_Modeling/) | Modeling time-varying variance in financial and other series | ✅ |
| 10 | [Prophet Forecasting](10_Prophet_Forecasting/) | Facebook's additive forecasting model for business time series | ✅ |
| 11 | [Time Series Cross-Validation & Forecast Evaluation](11_Time_Series_Cross_Validation_Forecast_Evaluation/) | Why shuffled k-fold CV is wrong for time-ordered data | ✅ |
| 12 | [State Space Models & Kalman Filter](12_State_Space_Models_Kalman_Filter/) | A unifying framework for time series with hidden states | ✅ |
| 13 | [Time Series Feature Engineering](13_Time_Series_Feature_Engineering/) | Lag features, rolling statistics, and Fourier features | ✅ |
| 14 | [Anomaly Detection in Time Series](14_Anomaly_Detection_in_Time_Series/) | Finding points and periods that don't fit the pattern | ✅ |
| 15 | [LSTM/GRU for Time Series Forecasting](15_LSTMGRU_for_Time_Series_Forecasting/) | Neural sequence models for forecasting | ✅ |
| 16 | [Spectral Analysis & Fourier Transforms](16_Spectral_Analysis_Fourier_Transforms/) | Finding periodicities hidden in a time series | ✅ |
| 17 | [Cointegration & Vector Error Correction Models](17_Cointegration_Vector_Error_Correction_Models/) | Long-run equilibrium relationships between non-stationary series | 🚧 |
| 18 | [Dynamic Time Warping & Time Series Clustering](18_Dynamic_Time_Warping_Time_Series_Clustering/) | Comparing and grouping series that are shifted or stretched in time | 🚧 |
| 19 | [Change Point Detection](19_Change_Point_Detection/) | Finding where a series' statistical behavior shifts | 🚧 |
| 20 | [Multi-Step Forecasting Strategies & Ensemble Forecasting](20_Multi_Step_Forecasting_Strategies_Ensemble_Forecasting/) | Recursive vs. direct forecasting, and combining multiple forecasts | 🚧 |

**20 topics planned** — built one at a time, each with a deep-dive `README.md` (full math derivation in LaTeX, pitfalls, self-test exercises) and a fully-executed Jupyter notebook (30+ code cells), following the exact standard established in [Foundation](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation) and [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML).

## 📁 Repository Structure

```
Time-Series-Analysis/
├── README.md                          ← you are here
├── 01_Time_Series_Components_Decomposition/
│   └── README.md
├── 02_Stationarity_Unit_Root_Tests/
│   └── README.md
├── 03_Autocorrelation_Partial_Autocorrelation/
│   └── README.md
├── 04_Moving_Averages_Exponential_Smoothing/
│   └── README.md
├── 05_AR_MA_ARMA_Models/
│   └── README.md
├── 06_ARIMA/
│   └── README.md
├── 07_SARIMA/
│   └── README.md
├── 08_Vector_Autoregression_VAR/
│   └── README.md
├── 09_GARCH_Volatility_Modeling/
│   └── README.md
├── 10_Prophet_Forecasting/
│   └── README.md
├── 11_Time_Series_Cross_Validation_Forecast_Evaluation/
│   └── README.md
├── 12_State_Space_Models_Kalman_Filter/
│   └── README.md
├── 13_Time_Series_Feature_Engineering/
│   └── README.md
├── 14_Anomaly_Detection_in_Time_Series/
│   └── README.md
├── 15_LSTMGRU_for_Time_Series_Forecasting/
│   └── README.md
├── 16_Spectral_Analysis_Fourier_Transforms/
│   └── README.md
├── 17_Cointegration_Vector_Error_Correction_Models/
│   └── README.md
├── 18_Dynamic_Time_Warping_Time_Series_Clustering/
│   └── README.md
├── 19_Change_Point_Detection/
│   └── README.md
└── 20_Multi_Step_Forecasting_Strategies_Ensemble_Forecasting/
    └── README.md
```

Every topic folder will be self-contained once built: read the `README.md` for the theory, open the `.ipynb` for the hands-on implementation.

## 🧰 Prerequisites

- Python 3.9+
- `statsmodels` and `pmdarima` do the classical heavy lifting; `torch` is used for topic 15 (LSTM/GRU forecasting).

```bash
pip install numpy pandas matplotlib seaborn statsmodels pmdarima arch scikit-learn prophet torch jupyter
```

## 🚀 How to Use

**Just reading?** Every notebook (once built) renders directly on GitHub with full output — click any `.ipynb` link in the table above.

**Running it yourself:**

```bash
git clone https://github.com/mdnuruzzamanKALLOL/Time-Series-Analysis.git
cd Time-Series-Analysis
pip install numpy pandas matplotlib seaborn statsmodels pmdarima arch scikit-learn prophet torch jupyter
jupyter notebook
```

## 📦 Datasets

Real-data topics draw from the central **[Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets)** repo (162-entry catalog, 418 files) shared across this entire series.

## 🔭 The Full Series

| Repo | Covers | Status |
|---|---|---|
| [Foundation](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation) | Python/NumPy/Pandas, Visualization, Preprocessing, Feature Engineering, Math | ✅ Complete |
| [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) | Regression, Classification, Ensembles, Unsupervised, Model Evaluation & Tuning (29 algorithms) | ✅ Complete |
| [Statistical Inference & Hypothesis Testing](https://github.com/mdnuruzzamanKALLOL/Statistical-Inference-Hypothesis-Testing) | Probability through causal inference (20 topics) | 🚧 In progress |
| [Time Series Analysis](https://github.com/mdnuruzzamanKALLOL/Time-Series-Analysis) | Decomposition through neural forecasting (20 topics) | 🚧 In progress |
| [Deep Learning Foundations](https://github.com/mdnuruzzamanKALLOL/Deep-Learning-Foundations) | Perceptron through GANs and interpretability (20 topics) | 🚧 In progress |
| [Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets) | 418 datasets (162 cataloged + personal collection) backing the whole series | ✅ Complete |

## 📝 Self-Test Philosophy

Every topic README (once built) ends with a **Self-Test Exercises** section — deliberately *not* answered inline. The point is to predict the answer before running the corresponding notebook cell, not to read a solution.

## 💬 Feedback & Corrections

Found a bug, a math error, or a broken dataset link? Open an issue or a pull request — this series documents its own mistakes on purpose, so a caught error is a feature, not an embarrassment.

## 🔗 Related

- [Foundation →](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation)
- [Classical ML →](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML)
- [Datasets →](https://github.com/mdnuruzzamanKALLOL/Datasets)
- [Author's GitHub profile](https://github.com/mdnuruzzamanKALLOL)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.root&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
