# 🌊 Spectral Analysis & Fourier Transforms

> Status: ✅ Complete — [Open the notebook →](16_spectral_analysis_fourier_transforms.ipynb)

Topic 16 of the Time Series Analysis repo. Every topic so far analyzed a series in the *time* domain — how $y_t$ relates to $y_{t-1}, y_{t-2}, \ldots$. Spectral analysis moves to the *frequency* domain instead, decomposing a series into the sinusoids that compose it and asking which periodicities carry the most energy. This notebook builds the discrete Fourier transform and periodogram from scratch, proves the classical Wiener-Khinchin link back to Topic 03's autocorrelation function, confronts two real estimation pitfalls (spectral leakage and periodogram inconsistency) with their standard fixes, and closes with two genuine real-data discoveries: recovering AirPassengers' 12-month cycle only after removing its trend, and recovering the famous ~11-year solar (Schwabe) sunspot cycle first identified by Arthur Schuster's 1906 periodogram analysis.

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
Any series of length n can be written EXACTLY as a sum of n/2 sinusoids (the Fourier basis):

  y_t = sum_k [ a_k cos(2*pi*f_k*t) + b_k sin(2*pi*f_k*t) ],   f_k = k/n,  k = 0, 1, ..., n/2

The Discrete Fourier Transform (DFT) computes these amplitudes directly.
The PERIODOGRAM |Y(f)|^2/n estimates how much VARIANCE each frequency f contributes -- a
"frequency-domain histogram" of where the series' energy lives.

Nyquist frequency = 0.5 cycles/sample: the highest frequency distinguishable from sampled data;
anything faster aliases (masquerades) as a lower frequency.
```

---

## 🎯 Why This Topic Matters

- **The from-scratch DFT was validated to near machine precision** — §2 found a naive $O(n^2)$ matrix-based DFT matching `numpy.fft.fft`'s $O(n\log n)$ fast algorithm to within **4.09e-12**.
- **The periodogram was shown correctly recovering two true hidden periodicities from noisy data** — §3 found the top two peaks landing at periods **20.00** and **7.04**, matching the injected true periods of 20 and 7 almost exactly.
- **The Wiener-Khinchin theorem was proven numerically, not just asserted, directly linking back to Topic 03** — §4 found the periodogram and the discrete Fourier transform of the sample autocovariance function (Topic 03's ACF) matching to within **1.04e-12** — the time-domain and frequency-domain views of a series are provably the same information.
- **Spectral leakage was quantified precisely, and a concrete fix validated** — §5 found a rectangular window leaking **6.72%** of total spectral power away from a sinusoid's true (non-integer-period) frequency, reduced to just **0.07%** — a **94x** reduction — under a Hann window.
- **The periodogram's classical statistical inconsistency was demonstrated and fixed** — §6 found a single periodogram ordinate's variance staying large (std=3.61 around a true value of 4.0) regardless of averaging, while Welch's method cut that variance by **15.4x** through segment averaging.
- **Two genuine real-world discoveries closed the notebook** — §7 found AirPassengers' raw-level periodogram putting **79.0%** of its power into "trend-like" periods over 24 months, completely hiding the 12-month seasonal cycle until detrending dropped that figure to **10.0%** and revealed it (plus 6- and 3-month harmonics); §8 found the exact same from-scratch periodogram applied to 309 years of real Wolf sunspot data recovering a dominant **11.04-year** cycle, matching the historically documented ~11.0-11.1-year solar (Schwabe) cycle that Arthur Schuster first identified with this technique in 1906.

---

## 🧮 Mathematical Explanation

### 1. Discrete Fourier Transform

$$X_k = \sum_{j=0}^{n-1} x_j\, e^{-2\pi i jk/n}, \qquad k = 0, 1, \ldots, n-1$$

### 2. Periodogram

$$I(f_k) = \frac{1}{n}\left|\sum_{j=0}^{n-1}(x_j-\bar x)\, e^{-2\pi i j f_k}\right|^2, \qquad f_k = k/n$$

### 3. Wiener-Khinchin theorem

$$I(f) = \sum_{k=-(n-1)}^{n-1} \hat\gamma(k)\, e^{-2\pi i fk}, \qquad \hat\gamma(k) = \frac{1}{n}\sum_{t=1}^{n-k}(x_t-\bar x)(x_{t+k}-\bar x)$$

the periodogram is exactly the Fourier transform of the sample autocovariance function $\hat\gamma(k)$ from Topic 03.

### 4. Welch's method

$$\hat S_{\text{Welch}}(f) = \frac{1}{K}\sum_{i=1}^{K} I_i(f)$$

the average of $K$ periodograms computed on overlapping, windowed segments of the series; variance shrinks roughly as $1/K$ at the cost of frequency resolution.

---

## 📋 Formula Reference

| Concept | Formula | Notebook Section |
|---|---|---|
| Discrete Fourier Transform | $X_k=\sum_j x_j e^{-2\pi ijk/n}$ | §2 |
| Periodogram | $I(f_k)=\lvert X_k\rvert^2/n$ | §3, §5-§8 |
| Wiener-Khinchin theorem | $I(f)=\sum_k\hat\gamma(k)e^{-2\pi ifk}$ | §4 |
| Welch's method | $\hat S_{\text{Welch}}(f)=\frac{1}{K}\sum_i I_i(f)$ | §6 |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Running a periodogram on a series with a strong trend without removing it first** — §7 found 79.0% of AirPassengers' raw-level spectral power sitting at periods longer than 24 months, completely obscuring the true 12-month seasonal cycle until the trend was removed.
2. **Assuming more data always makes a periodogram estimate more precise** — §6 found a single periodogram ordinate's variance staying large regardless of sample size (a well-known theoretical result: the periodogram is asymptotically unbiased but *not* consistent); Welch's method traded frequency resolution for a 15.4x variance reduction by averaging over segments.
3. **Ignoring spectral leakage when a true period does not divide evenly into the sample length** — §5 found a rectangular window leaking 6.72% of total power away from the true frequency, nearly 100x more than the 0.07% leaked under a Hann window.
4. **Treating the periodogram and the autocorrelation function as unrelated diagnostics** — §4 found the periodogram and the discrete Fourier transform of Topic 03's sample autocovariance function matching to within 1.04e-12: they are the exact same information about a series, viewed in two different domains.
5. **Reading a periodogram peak as proof of a true underlying cycle without checking its plausibility** — §8's ~11.04-year peak in real sunspot data matches over a century of independently confirmed solar physics; a peak in a shorter, noisier series should be treated with far more caution, since noise alone can and does produce spurious-looking peaks (as several smaller peaks near, but not at, the AirPassengers and sunspot dominant frequencies illustrate).

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| Custom `naive_dft()` | From-scratch $O(n^2)$ discrete Fourier transform |
| `numpy.fft.fft` / `numpy.fft.fftfreq` | Library fast Fourier transform, used for validation and all periodograms |
| Custom `periodogram()` | From-scratch periodogram estimator |
| Custom `autocovariance()` | From-scratch sample autocovariance function, reused to verify Wiener-Khinchin |
| Custom `periodogram_windowed()` | From-scratch windowed periodogram (rectangular vs. Hann) |
| Custom `welch_from_scratch()` | From-scratch Welch's method (segmented, windowed, averaged periodogram) |
| `scipy.signal.detrend` | Library linear detrending, applied before spectral analysis |
| `statsmodels.datasets.sunspots` | Real 309-year Wolf sunspot number dataset |

---

## 📝 Self-Test Exercises

1. Section 2 found the from-scratch DFT matching NumPy's FFT to within 4.09e-12. Using the DFT formula in §1 of the Mathematical Explanation, explain why both algorithms must produce identical results in exact arithmetic, and what makes the FFT algorithmically faster without changing the answer.
2. Section 5 found a non-integer-period sinusoid leaking 6.72% of its power away from its true frequency under a rectangular window. Propose what would happen to that leakage percentage if the sinusoid's period were changed to exactly 10 samples (dividing evenly into the 100-sample series), and explain why using the periodogram formula in §2.
3. Section 6 found the periodogram's variance not shrinking with sample size, unlike most statistical estimators. Using Welch's method formula in §4 of the Mathematical Explanation, explain the specific trade-off between segment length and the number of segments $K$, and what happens to frequency resolution as $K$ increases with a fixed total sample size.
4. Section 7 found harmonics at 6 and 3 months alongside the fundamental 12-month AirPassengers cycle. Explain, using the Fourier basis intuition in §1 of the Concept section, why a real, non-sinusoidal seasonal pattern (e.g., a sharp summer peak rather than a smooth sine wave) necessarily produces energy at integer multiples of the fundamental frequency, not just at the fundamental itself.
5. Section 8 found several periodogram peaks near, but not exactly at, 11.04 years (e.g., 9.97 and 10.66 years) in the sunspot data. Propose an experiment (using either Welch's method from §6 or a longer synthetic simulation) that would help determine whether these nearby peaks reflect genuine additional periodicities in solar activity or are simply noise-driven artifacts of a single true ~11-year cycle.

---

## 📓 Notebook

13 executed code cells, 4 plots: a from-scratch DFT and periodogram validated against NumPy to near machine precision; recovery of two known hidden periodicities from noisy synthetic data; numerical proof of the Wiener-Khinchin theorem linking the periodogram to Topic 03's autocorrelation function; a concrete spectral-leakage demonstration and its Hann-window fix; a variance-reduction demonstration for Welch's method; and two real-data applications — AirPassengers' 12-month cycle hidden by its trend until detrending, and the famous ~11-year solar cycle recovered from 309 years of real Wolf sunspot data:

➡️ **[16_spectral_analysis_fourier_transforms.ipynb](16_spectral_analysis_fourier_transforms.ipynb)**

---

## 📚 Further Reading

- [Schuster (1906): On the Periodicities of Sunspots](https://doi.org/10.1098/rsta.1906.0018)
- [Welch (1967): The Use of Fast Fourier Transform for the Estimation of Power Spectra](https://doi.org/10.1109/TAU.1967.1161901)
- [Wiener (1930) / Khinchin (1934): the Wiener-Khinchin theorem](https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem)
- [`scipy.signal.periodogram` / `scipy.signal.welch` documentation](https://docs.scipy.org/doc/scipy/reference/signal.html#spectral-analysis)

---
[← Back to Time Series Analysis](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TimeSeriesAnalysis.16_Spectral_Analysis_Fourier_Transforms&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
