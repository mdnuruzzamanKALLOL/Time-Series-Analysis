# Dynamic Time Warping & Time Series Clustering

> Status: ✅ Complete

Comparing and grouping series that are shifted or stretched in time

## Concept & Intuition

Every distance-based comparison used so far in this repo (forecast errors, Diebold-Mariano tests, anomaly scores) implicitly compares two series **index-by-index**: point $t$ of series A against point $t$ of series B. That assumption silently breaks the moment two series have the same underlying *shape* but are shifted in time, recorded at different speeds, or simply have different lengths — extremely common for gestures, heartbeats, seasonal cycles, or any real-world process that doesn't repeat like clockwork.

**Dynamic Time Warping (DTW)** replaces the rigid index-to-index comparison with a flexible, non-linear alignment that stretches and compresses the time axis to line up matching features wherever they occur, then measures the remaining (unavoidable) difference. This makes it possible to correctly recognize that two signals have "the same shape" even when a naive point-by-point distance says otherwise — and to compare series of genuinely different lengths without forcing them into a common grid first.

## Mathematical Explanation

For series $x_{1:n}$ and $y_{1:m}$, define the cumulative cost matrix via dynamic programming:

$$D(i,j) = (x_i - y_j)^2 + \min\big[D(i-1,j),\ D(i,j-1),\ D(i-1,j-1)\big]$$

with $D(0,0)=0$ and boundary rows/columns at $+\infty$. The **DTW distance** is $\sqrt{D(n,m)}$ (squared-Euclidean local cost, square root at the end — the standard convention). Filling the matrix is $O(nm)$; backtracking from $(n,m)$ to $(0,0)$, always stepping to the cheapest of the three neighbors, recovers the **optimal warping path** — the actual alignment between the two series.

**Sakoe-Chiba band.** Unconstrained DTW can in principle warp arbitrarily far, sometimes producing degenerate alignments. Restricting the path to $|i-j|\le w$ both speeds up computation (only cells inside the band are filled) and prevents pathological warps — but it is a genuine modeling choice: too narrow a band can forbid the true optimal alignment entirely.

**DTW is not a proper metric** (it can violate the triangle inequality), but it works perfectly well as an input to hierarchical clustering or nearest-neighbor ranking, which only require a distance/dissimilarity value, not a metric.

## Code Implementation (`18_dynamic_time_warping_time_series_clustering.ipynb`)

### 1. Why Euclidean distance fails

Two identical bell-shaped "bumps," one centered at $t=40$ and the other at $t=50$ (a pure 10-step shift, otherwise identical): Euclidean distance = **3.26** — a large number assigned to two signals that are, by construction, the exact same shape.

### 2. From-scratch DTW, validated to machine precision

The $O(nm)$ dynamic-programming recursion was implemented directly and checked against the `dtaidistance` reference library on random series (n=30, m=35): **both report 5.449461, max abs diff = 0.00e+00.** Applied to the shifted-bump example, DTW correctly reports a distance of **0.0000** (vs. Euclidean's 3.26) since a pure shift can be perfectly compensated by warping.

### 3. The warping path

Backtracking the cost matrix recovers the actual alignment: the plot shows bump A's rising edge matched to bump B's rising edge, peak-to-peak, and falling edge to falling edge — not a straight one-to-one index correspondence, but a genuinely shape-aware match.

### 4. The Sakoe-Chiba band: speed and correctness trade-off

On two independent 400-step random walks:

| | Distance | Time |
|---|---|---|
| Unconstrained DTW | 316.85 | 147.1 ms |
| Sakoe-Chiba band (w=20) | 566.03 | 14.5 ms |

**~10x speedup**, closely matching the ~10x reduction in cells computed. Critically, the **distance itself changed** (566 vs. 317) — the band didn't just make computation faster, it forbade the true optimal (unconstrained) alignment. Window size is a real modeling decision, not just a performance knob.

### 5. A case where Euclidean distance is backwards

| Comparison | Euclidean | DTW |
|---|---|---|
| sine vs. shifted sine (same shape) | 8.2753 | 2.5165 |
| sine vs. square wave (different shape) | 4.7392 | 4.5750 |

Euclidean distance ranks the **same-shape, merely-shifted** pair as *more* different (8.28) than the **genuinely different-shaped** pair (4.74) — backwards. DTW correctly ranks them the other way (2.52 < 4.58): the shifted sine is recognized as far more similar to the original than the square wave ever is.

### 6. Clustering by shape: DTW vs. Euclidean

Three synthetic shape families — single bump, double bump, ramp-plateau — were generated (10 examples each) with a random time shift, random stretch, and noise. Hierarchical clustering (average linkage, $k=3$) was scored against the true family labels with the Adjusted Rand Index (ARI, 1.0=perfect, 0=random):

| Distance | ARI |
|---|---|
| Euclidean | 0.0655 |
| DTW (unconstrained) | **1.0000** |

DTW recovers the three true shape families **perfectly**; Euclidean distance is essentially no better than random assignment.

### 7. Pitfall: the warping window determines what's possible

Repeating the clustering above with different Sakoe-Chiba band widths:

| Window | DTW ARI |
|---|---|
| 10 | 0.0771 |
| 15 | 0.4694 |
| 20 | 0.5007 |
| 30 | 1.0000 |
| None (unconstrained) | 1.0000 |

With too narrow a band, DTW cannot find the alignment needed to compensate for the random shifts and performs about as poorly as Euclidean (0.077). Only once the band is wide enough (≥30) does DTW recover the true families perfectly. **The window is not a mere speed optimization — it determines which alignments are even reachable.**

### 8. Real data: solar cycle shapes (sunspot data, 1700-2008)

The sunspot dataset (Topic 16) was automatically segmented into 26 individual solar cycles (trough-to-trough), which naturally have **different durations** — 9 to 15 years, mean 10.9 — a case where Euclidean distance cannot even be computed without first forcing every cycle onto a common length via resampling, while DTW compares native-length series directly.

Ranking all 26 cycles by average pairwise DTW distance (a shape-based outlier score, callback to Topic 14's anomaly detection):

- **Most shape-atypical:** 1711-1722 (avg DTW dist 0.469) — the first fully-observed cycle after the Maunder Minimum grand solar minimum ended (~1715), plausibly still an atypical recovery-phase cycle.
- **Most typical:** 1932-1943 (avg DTW dist 0.270).
- **Most similar pair by shape:** 1877-1888 and 1922-1932 (DTW dist 0.162) — nearly identical shape despite being 45 years apart.
- The **longest cycle in the record, 1783-1798 (15 years)**, ranks **#7 of 26** in shape-atypicality — notably above the median. This is the historically documented **"Solar Cycle 4"**, long known in solar physics as anomalously long with an unusual declining phase; a body of research (Usoskin et al., 2009, *ApJ* 700:L154) argues that sparse sunspot observations during the French-Revolution-era Dalton Minimum actually hid a second, weak cycle inside what was recorded as one long cycle. An unsupervised, purely shape-based DTW ranking independently flags this scientifically debated period as unusual.

Forcing a $k=2$ hierarchical clustering on these 26 cycles was **not informative** — it merely isolated the single biggest outlier (1711-1722) into its own cluster, leaving all 25 others together. A reminder that real data doesn't always have well-separated cluster structure; ranking/nearest-neighbor approaches can be more useful than forced partitioning.

### 9. A verifiable physical pattern: the Waldmeier effect

$$r(\text{peak amplitude},\ \text{trough-to-peak rise time}) = -0.7170,\quad p = 3.8\times10^{-5}$$

A strong, statistically significant confirmation of the **Waldmeier effect**: stronger solar cycles rise to their peak significantly *faster* than weak ones. This is independent, verifiable evidence that cycle **shape** (not just amplitude) carries genuine physical information — exactly the kind of structure a shape-aware distance measure like DTW is built to compare.

## Common Pitfalls

1. **Euclidean distance can rank a shifted copy of a signal as more different than a genuinely different-shaped signal.** Section 5 demonstrated this concretely (8.28 vs. 4.74) — a real, not hypothetical, failure mode for any comparison involving imperfectly-aligned real-world series.
2. **The Sakoe-Chiba window is a real modeling decision, not just a speed optimization.** Too narrow a band forbids the very alignment that makes DTW useful (Section 7); too wide a band on noisy data risks pathological warps. Choose it based on how much timing variation is physically plausible.
3. **DTW is not a proper metric** (triangle inequality can fail). Fine for hierarchical clustering and nearest-neighbor ranking; be cautious using it with algorithms that assume a true metric space.
4. **Comparing variable-length series with Euclidean distance requires resampling first**, silently imposing a choice (interpolation method, common length) that can distort the shapes being compared. DTW avoids this preprocessing step and its hidden assumptions entirely.
5. **Real-data clustering doesn't always yield clean, evenly-sized partitions.** Forcing $k=2$ on the 26 solar cycles just isolated one outlier rather than producing a meaningful split — ranking/nearest-neighbor approaches are often more informative than forced partitioning when data has one dominant mode plus a few outliers.
6. **DTW's $O(nm)$ per-pair cost becomes $O(N^2nm)$ for an $N$-series distance matrix** — expensive at scale. Banding, lower-bound pruning (e.g., LB_Keogh, used internally by `dtaidistance`), or DTW-barycenter-based centroid methods are the standard ways to scale this up in practice.

## Function Reference

| Function | Purpose |
|---|---|
| `dtw_distance` (from scratch) | $O(nm)$ dynamic-programming DTW distance, with optional Sakoe-Chiba band |
| `dtw_path` (from scratch) | Backtrack the cost matrix to recover the optimal warping path |
| `dtaidistance.dtw.distance` | Reference DTW implementation used for validation |
| `scipy.cluster.hierarchy.linkage` / `fcluster` | Hierarchical clustering from a precomputed distance matrix |
| `sklearn.metrics.adjusted_rand_score` | Clustering-quality metric against known true labels |
| `scipy.signal.find_peaks` | Used (on negated, smoothed data) to detect solar-cycle troughs |

## Self-Test

1. Why does Euclidean distance judge a phase-shifted copy of a sine wave to be more different from the original than a square wave is — and why is that the wrong answer?
2. Write out the DTW recurrence $D(i,j)$ and explain what each of the three terms being minimized corresponds to (insertion, deletion, match).
3. What does the Sakoe-Chiba band constrain, and why is choosing its width a "real modeling decision" rather than just a speed/accuracy trade-off?
4. Why does DTW not need any resampling to compare a 9-year and a 15-year solar cycle, while Euclidean distance does?
5. Why is DTW distance not a proper metric, and does that limitation matter for the hierarchical clustering used in this notebook?
6. The $k=2$ clustering of solar cycles just isolated a single outlier rather than splitting the data meaningfully. What does that suggest about applying forced-k clustering to real, potentially single-mode data?

## Notebook

[`18_dynamic_time_warping_time_series_clustering.ipynb`](./18_dynamic_time_warping_time_series_clustering.ipynb)

## Further Reading

- Sakoe, H. & Chiba, S. (1978). *Dynamic Programming Algorithm Optimization for Spoken Word Recognition.* IEEE Trans. ASSP.
- Keogh, E. & Ratanamahatana, C.A. (2005). *Exact Indexing of Dynamic Time Warping.* Knowledge and Information Systems.
- Usoskin, I.G., Mursula, K., Arlt, R., Kovaltsov, G.A. (2009). *A Solar Cycle Lost in 1793-1800: Early Sunspot Observations Resolve the Old Mystery.* The Astrophysical Journal, 700(2), L154.
- Waldmeier, M. (1935). *Neue Eigenschaften der Sonnenfleckenkurve.* Astronomische Mitteilungen der Eidgenössischen Sternwarte Zürich (the original description of the Waldmeier effect).
- [dtaidistance documentation](https://dtaidistance.readthedocs.io/)

---
[← Back to Time Series Analysis](../README.md)
