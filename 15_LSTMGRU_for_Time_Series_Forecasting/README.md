# 🔁 LSTM/GRU for Time Series Forecasting

> Status: ✅ Complete — [Open the notebook →](15_lstmgru_for_time_series_forecasting.ipynb)

Topic 15 of the Time Series Analysis repo — the first neural-network topic. LSTMs and GRUs are recurrent architectures designed to solve the vanishing-gradient problem that cripples vanilla RNNs on long sequences, using learned gates to control what information is kept, forgotten, or output at each time step. This notebook builds both cells from scratch in NumPy and matches them to PyTorch's `nn.LSTM`/`nn.GRU` to machine precision, confirms an LSTM can learn genuine AR dynamics given enough data, and then applies both architectures honestly to real AirPassengers data — where the central, expected finding is that with only ~100 monthly training points, neither network comes close to the classical statistical models or the feature-engineered Random Forest built in earlier topics, a well-documented real-world pattern (echoed in large forecasting competitions like the M4) rather than a failure of implementation.

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
Vanilla RNN:  h_t = tanh(W_x x_t + W_h h_(t-1))
              gradients backpropagated through many steps repeatedly multiply by W_h -> vanish or explode

LSTM adds a separate CELL STATE (a "conveyor belt") plus three learned gates:
  forget gate f_t: how much of the old cell state to keep
  input  gate i_t: how much of the new candidate to add
  output gate o_t: how much of the cell state to expose as the hidden state

GRU merges the cell state into the hidden state and uses only two gates:
  reset  gate r_t: how much of the past hidden state to use when computing the candidate
  update gate z_t: how much to keep the old hidden state vs. the new candidate

Fewer gates -> fewer parameters -> GRU trains faster and needs less data than LSTM,
at the (usually small) cost of slightly less expressive power.
```

---

## 🎯 Why This Topic Matters

- **From-scratch LSTM and GRU cells were validated against PyTorch to machine precision** — §2-§3 found max absolute differences of **5.39e-08** (LSTM output), **4.56e-08** (LSTM cell state), and **4.64e-08** (GRU output) between a from-scratch NumPy recursion using PyTorch's own trained weights and PyTorch's native `nn.LSTM`/`nn.GRU` output.
- **A small LSTM was shown genuinely learning AR(1) dynamics, not memorizing noise, given enough data** — §4 found an LSTM trained on 1,000 synthetic points reaching test MAE **0.3796**, essentially matching the theoretical Bayes-optimal one-step-ahead MAE of **0.3989** computed directly from the known true noise variance.
- **Overfitting on small real-world-sized data was demonstrated concretely, not just asserted** — §5 found training loss monotonically shrinking to **0.0021** over 400 epochs on just 65 training sequences, while validation loss bottomed out at **0.2399** (epoch 143) and then more than doubled to **0.5569** by the final epoch.
- **A rigorous, honest walk-forward evaluation (Topic 11's exact methodology) found neural networks barely beating a naive forecast on raw AirPassengers** — §6 found both LSTM (MASE **1.2572**) and GRU (MASE **1.2324**) landing almost exactly at the naive-seasonal baseline (MASE 1.2663), dramatically behind SARIMA (0.6695) and the Kalman-filter BSM (0.6403).
- **Topic 13's detrending fix helped meaningfully but did not close the gap to classical models** — §7 found the GRU's MASE improving from 1.2324 to **0.8167** when targeting the seasonal difference instead of the level, still well behind SARIMA, the BSM, and especially Topic 13's detrended Random Forest (MASE 0.3518).
- **Pure random weight initialization was shown changing the result by a large margin, all else held fixed** — §8 found identical data, architecture, and training producing a MAE ranging from **17.56 to 30.50** (a 74% relative spread) across 5 random seeds alone.

---

## 🧮 Mathematical Explanation

### 1. LSTM gate equations

$$i_t=\sigma(W_{ii}x_t+W_{hi}h_{t-1}+b_i), \quad f_t=\sigma(W_{if}x_t+W_{hf}h_{t-1}+b_f), \quad o_t=\sigma(W_{io}x_t+W_{ho}h_{t-1}+b_o)$$

$$\tilde c_t=\tanh(W_{ig}x_t+W_{hg}h_{t-1}+b_g), \qquad c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t, \qquad h_t=o_t\odot\tanh(c_t)$$

### 2. GRU gate equations

$$r_t=\sigma(W_{ir}x_t+W_{hr}h_{t-1}+b_r), \qquad z_t=\sigma(W_{iz}x_t+W_{hz}h_{t-1}+b_z)$$

$$\tilde h_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})), \qquad h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$$

### 3. Bayes-optimal one-step MAE (known-model benchmark)

For $y_t=\phi y_{t-1}+\varepsilon_t,\ \varepsilon_t\sim N(0,\sigma^2)$, the optimal one-step forecast is $\hat y_t=\phi y_{t-1}$, giving expected absolute error:

$$E|\varepsilon_t| = \sigma\sqrt{2/\pi}$$

### 4. Seasonal-difference reconstruction (Topic 13 reused)

$$z_t = y_t - y_{t-12} \qquad \Longrightarrow \qquad \hat y_t = y_{t-12} + \hat z_t$$

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| LSTM cell/hidden update | $c_t=f_t c_{t-1}+i_t\tilde c_t,\ h_t=o_t\tanh(c_t)$ | §2 |
| GRU hidden update | $h_t=(1-z_t)\tilde h_t+z_t h_{t-1}$ | §3 |
| Bayes-optimal AR(1) MAE | $E\vert\varepsilon_t\vert=\sigma\sqrt{2/\pi}$ | §4 |
| Seasonal-difference reconstruction | $\hat y_t=y_{t-12}+\hat z_t$ | §7 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting to scale inputs before training a neural network** — unlike every ARIMA-family or tree-based model in this repo, LSTMs/GRUs train on gradient descent through `tanh`/`sigmoid` gates that saturate outside roughly $[-3,3]$; §6-§7 fit the scaler on training data only in each walk-forward fold, exactly matching Topic 11's leakage discipline.
2. **Judging a small-data neural network's potential by its training loss** — §5 found training loss shrinking to near-zero long after validation loss had already bottomed out and started rising; on 65 training sequences, near-perfect training fit is a red flag, not a good sign.
3. **Trusting early stopping on a tiny internal validation set** — an early-stopping loop was attempted for this notebook and found *unreliable* here (the ~15-26 sequence internal validation set is itself too small and noisy to reliably identify the right stopping point); a fixed, modest epoch budget informed by the overfitting curve in §5 worked more consistently.
4. **Expecting a neural network to beat classical models by default on short series** — §6 found both LSTM and GRU landing almost exactly at the naive-seasonal baseline (MASE≈1.23-1.26) on raw AirPassengers, dramatically behind SARIMA (0.6695), the Kalman-filter BSM (0.6403), and especially the feature-engineered Random Forest (0.3518) from Topic 13 — recurrent networks generally need far more than ~100-150 data points to reliably outperform well-specified classical models, a pattern also documented in large forecasting competitions like the M4.
5. **Reporting a single-seed deep learning result as if it were precise** — §8 found identical data, architecture, and training procedure producing a MAE ranging from 17.56 to 30.50 (a 74% relative spread) purely from different random weight initializations; any single run should be treated as one noisy sample, ideally averaged over multiple seeds when reporting a final number.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `torch.nn.LSTM` / `torch.nn.GRU` | Library recurrent cell implementations, validated against from-scratch NumPy |
| Custom `lstm_forward_numpy()` / `gru_forward_numpy()` | From-scratch gate recursions using PyTorch's own trained weights |
| Custom `SimpleLSTM` / `SeqModel` (`torch.nn.Module`) | Small sequence-to-one forecasting networks used throughout |
| `torch.optim.Adam` | Gradient-based optimizer used for all training |
| Custom `make_windows()` | From-scratch sequence-windowing function, reused from Topic 13's feature-engineering pattern |
| Custom `walk_forward_nn()` | Topic 11's exact walk-forward methodology, adapted for recursive neural-network forecasting |

---

## 📝 Self-Test Exercises

1. Section 2 found the from-scratch LSTM matching PyTorch to within 5.39e-08. Using the gate equations in §1 of the Mathematical Explanation, explain why this tiny residual difference is consistent with floating-point rounding rather than an algorithmic error, and what result you would expect if both implementations used float64 instead of float32.
2. Section 4 found an LSTM's test MAE (0.3796) very close to the Bayes-optimal bound (0.3989) with 1,000 training points. Propose what would happen to this gap if the training set were shrunk to 100 points instead, and connect your answer to what Section 5's overfitting curve found on a similarly small AirPassengers training set.
3. Section 6 found both LSTM and GRU landing almost exactly at the naive-seasonal MASE (1.2663) on raw AirPassengers. Using Topic 13's tree-model extrapolation argument as a contrast, explain why a neural network with a plain linear output layer is *not* structurally bounded by its training range the way a Random Forest is, and propose why it still failed to substantially beat naive here despite that theoretical advantage.
4. Section 7 found the seasonal-difference target improving the GRU's MASE from 1.2324 to 0.8167, a smaller relative gain than Topic 13 found for the Random Forest (1.0319 to 0.3518). Propose one architectural or training change (not a change of target) that might close more of this remaining gap, and explain the specific mechanism by which it would help.
5. Section 8 found a 74% relative MAE spread across 5 random seeds. Using this notebook's fixed 150-epoch training procedure, propose a concrete experimental design (how many seeds, what you would report) that would let you responsibly claim "Model A is more accurate than Model B" on this series, and explain what evidence would be needed to support that claim given the seed variance observed.

---

## 📓 Notebook

12 executed code cells, 1 plot: from-scratch LSTM and GRU cells matched to PyTorch's native implementations to machine precision; a synthetic AR(1) sanity check confirming an LSTM can learn genuine dynamics given ample data; a concrete overfitting demonstration on a 65-sequence training set; a full walk-forward evaluation (Topic 11's exact methodology) of LSTM and GRU on real AirPassengers data, both on the raw level and on Topic 13's seasonal-difference target; a 5-seed reproducibility check; and a final side-by-side comparison table against every real-data model built in this repo so far:

➡️ **[15_lstmgru_for_time_series_forecasting.ipynb](15_lstmgru_for_time_series_forecasting.ipynb)**

---

## 📚 Further Reading

- [Hochreiter & Schmidhuber (1997): Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Cho et al. (2014): Learning Phrase Representations using RNN Encoder-Decoder (introduces the GRU)](https://arxiv.org/abs/1406.1078)
- [Makridakis, Spiliotis & Assimakopoulos (2018): The M4 Competition — statistical vs. ML forecasting methods](https://doi.org/10.1016/j.ijforecast.2018.06.001)
- [PyTorch `nn.LSTM` / `nn.GRU` documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.15_LSTMGRU_for_Time_Series_Forecasting&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
