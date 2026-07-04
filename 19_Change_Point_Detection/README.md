# Change Point Detection

> Status: ✅ Complete

Finding where a series' statistical behavior shifts

## Concept & Intuition

Topic 14 (Anomaly Detection) found individual points or short bursts that don't fit the surrounding pattern — the series reverts to normal right after. **Change point detection** asks a related but different question: where does a series' underlying statistical behavior (mean, variance, trend) shift **permanently** to a new regime? A single flood year is an anomaly; a dam that permanently reduces river flow from that point onward is a change point. The distinction matters because the two call for different tools and different interpretations — mistaking one for the other leads to real errors (e.g., treating a one-off event as evidence the "new normal" has changed, or vice versa).

This notebook builds change point detection from scratch: a single-split test based on minimizing within-segment variance, greedy **binary segmentation** for multiple change points, BIC for automatically choosing how many, **CUSUM** for sequential/online monitoring, and `ruptures`' exact **PELT** algorithm — validating every from-scratch piece against the library, and applying all of it to the single most-cited real change point dataset in the field: the Nile river's annual flow record.

## Mathematical Explanation

**Single change point (offline).** For a candidate split at index $t$, the reduction in total sum-of-squared-errors from splitting is:

$$\text{gain}(t) = \text{SSE}(x_{1:n}) - \text{SSE}(x_{1:t}) - \text{SSE}(x_{t+1:n})$$

The best split maximizes this gain — equivalent, under a Gaussian-mean-shift assumption, to a likelihood-ratio test for where the mean changes.

**Binary segmentation** extends this to multiple change points greedily: find the best single split, then recurse on each resulting segment, repeating until a stopping rule (e.g., a fixed number of change points, or a penalty threshold) is reached. It is fast ($O(n\log n)$ typical) but **not guaranteed globally optimal** for a given number of change points, since earlier splits are never revisited.

**BIC** selects the number of change points $k$ by penalizing the log-likelihood improvement each additional split buys:

$$\text{BIC}(k) = n\log(\text{RSS}_k/n) + (k+1)\log(n)$$

**PELT** (Pruned Exact Linear Time) finds the segmentation that exactly minimizes $\sum_i \text{cost}(\text{segment}_i) + \beta \cdot k$ for a given penalty $\beta$, in expected $O(n)$ time via dynamic programming with pruning — genuinely optimal for that penalty, unlike greedy binary segmentation.

**CUSUM** (cumulative sum control chart) is an *online/sequential* method: it monitors a stream and raises an alarm once evidence for a shift accumulates past a threshold $h$:

$$S^+_t = \max\big(0,\ S^+_{t-1} + (x_t-\mu_0) - k\big), \qquad S^-_t = \max\big(0,\ S^-_{t-1} - (x_t-\mu_0) - k\big)$$

where $\mu_0$ is the in-control target and $k$ is a slack parameter (typically half the shift magnitude to be detected). The threshold $h$ trades off the **ARL0** (average run length under no shift — false-alarm robustness) against detection delay once a real shift occurs.

## Code Implementation (`19_change_point_detection.ipynb`)

### 1. Change point vs. anomaly

A single transient spike injected into otherwise-flat noise: a naive single-change-point detector places a split exactly at the spike, but the two resulting segment means are nearly identical (-0.294 vs. 0.356) — **not a real level shift**. Allowing 2 change points instead of 1 correctly isolates the spike into its own **1-point segment** (mean 15.0, length 1), leaving the true before/after means at ~0. Segment *duration*, not the mere existence of a split, is what distinguishes an anomaly from a genuine regime change.

### 2. Single change point, validated

On synthetic data with a known mean shift (0→3 at $t=50$), the from-scratch SSE-gain search finds $t=50$ exactly, matching `ruptures`' `Binseg` (with `jump=1`) precisely.

### 3. Multiple change points: binary segmentation + BIC

On a 4-segment synthetic series (means 0, 3, -2, 4; true change points at 60, 120, 180), from-scratch binary segmentation recovers **[60, 120, 180]** exactly, matching `ruptures` exactly. BIC-based automatic selection over $k=0$ to $8$ correctly identifies **$k=3$** as the minimum (BIC=-2.87), exactly matching the true number and location of change points.

### 4. CUSUM: the ARL0 / detection-delay trade-off

| Threshold $h$ | ARL0 (avg. run length, no shift) | Mean detection delay | Detected |
|---|---|---|---|
| 3 | 56.1 | 4.60 | 53/300 |
| 5 | 421.7 | 8.94 | 238/300 |
| 8 | 952.5 | 14.51 | 297/300 |
| 10 | 986.4 | 18.81 | 300/300 |
| 15 | 1000.0 | 28.47 | 300/300 |
| 20 | 1000.0 | 38.82 | 300/300 |

Raising $h$ from 3 to 20 pushes the false-alarm-free run length from 56 to 1000+ steps, at the direct cost of increasing detection delay from ~5 to ~39 steps once a real shift occurs. There is no value of $h$ that minimizes both simultaneously — the right choice depends on which error costs more in the application.

### 5. Real data: the Nile river's famous change point

The Nile's annual flow volume at Aswan (1871-1970) is the most-cited example dataset in the change point literature (used in Killick, Fearnhead & Eckley's original 2012 PELT paper). Both the from-scratch method and `ruptures` (with `jump=1`) find a single change point at **1899** (mean before 1097.8, mean after 850.0), matching the historically documented **1898-1902 construction of the Aswan Low Dam**, which permanently regulated and reduced Nile flow — a genuine, verifiable real-world validation, found entirely from the data.

**A library-defaults pitfall surfaced here:** `ruptures`' `Binseg` with its *default* `jump=5` (only checks every 5th candidate point, for speed) returned **1901** instead — a visibly different, less accurate year. Always check what a library's defaults are actually doing.

### 6. Pitfall: greedy binary segmentation + BIC can overfit

Repeating the BIC-based automatic selection from Section 3 on the real Nile data:

| $k$ | RSS | BIC |
|---|---|---|
| 0 | 2,835,157 | 1029.85 |
| 1 | 1,597,457 | 977.09 |
| 2 | 1,542,327 | 978.18 |
| 3 | 1,452,060 | 976.75 |
| 4 | 1,396,298 | 977.44 |
| 5 | 1,310,797 | **975.73 (min)** |
| 6 | 1,261,120 | 976.47 |

BIC nominally picks $k=5$ (years 1877, 1878, 1881, 1890, 1899) — but the BIC values from $k=1$ through $k=6$ are nearly **flat** (975-978, a razor-thin range). BIC is not confidently distinguishing between 1 and 5 change points here; the greedy algorithm is adding spurious splits beyond the single, historically real one, because BIC computed along a greedy binary-segmentation path is not the same as BIC of the true globally-optimal segmentation for each $k$.

### 7. Pitfall: PELT's optimality doesn't save you from penalty choice

PELT is exactly optimal *for a given penalty* — but the penalty itself remains a free, highly consequential choice:

| Penalty basis | Value | Change points found |
|---|---|---|
| Raw variance of the series (inflated by the level shift itself) | 130,564 | **1**: [1899] |
| Robust noise-variance estimate (from first differences) | 61,240 | **9**: [1881, 1890, 1899, 1908, 1911, 1916, 1918, 1954, 1966] |

Using the (inflated) raw variance to scale the BIC-style penalty happens to reproduce the single historically-known change point. Using a more textbook-"correct" robust noise estimate (median-absolute-deviation of first differences) finds **9** change points — almost certainly overfit to short-run fluctuations. PELT's exactness guarantees the best segmentation *for a given penalty*; it says nothing about whether that penalty is the right one. **There is no free lunch: domain knowledge (here, knowing the Aswan Dam's construction date) is what actually validates the result.**

## Common Pitfalls

1. **A change point is not the same thing as an anomaly.** A naive single-split detector can seize on a transient spike (Section 1); the tell is that an anomaly ends up isolated in a very short segment once more splits are allowed, while a genuine change point produces two large, stable segments.
2. **Library defaults matter.** `ruptures`' `Binseg` default `jump=5` gave a visibly wrong year (1901 vs. the correct 1899) on real data purely from subsampling candidate points for speed.
3. **Greedy binary segmentation + BIC can overfit real data.** BIC values were nearly flat across $k=1$ to $6$ on the Nile series, and the greedy path added likely-spurious extra change points beyond the one real, historically documented shift.
4. **PELT's optimality is conditional on the penalty, and the penalty is not free.** Two defensible penalty scales gave 1 vs. 9 change points on identical real data. Always sanity-check automatic results against domain knowledge.
5. **CUSUM's threshold $h$ trades off false alarms against detection delay** — there is no value that minimizes both; the right choice depends on which error is costlier for the specific application.
6. **All cost functions used here assume shifts in the MEAN under roughly Gaussian, constant-variance noise.** A pure variance shift or a more complex distributional change needs a different cost function (`ruptures` supports several, e.g. `"rbf"` for arbitrary distributional shifts) — using the wrong cost model for the kind of change you're actually looking for will silently miss it.

## Function Reference

| Function | Purpose |
|---|---|
| `best_split` (from scratch) | Find the single change point maximizing SSE-reduction |
| `binary_segmentation` / `binary_segmentation_path` (from scratch) | Greedy multi-change-point detection, with BIC selection path |
| `cusum` (from scratch) | Two-sided sequential CUSUM with configurable slack $k$ and threshold $h$ |
| `ruptures.Binseg` | Reference greedy binary segmentation, used for validation |
| `ruptures.Pelt` | Exact optimal segmentation for a given penalty |

## Self-Test

1. Why did the naive single-change-point detector "find" a split at the location of a transient spike, and what evidence reveals that this isn't a genuine change point?
2. Write the SSE-gain formula for a candidate split point and explain why maximizing it is equivalent to a likelihood-ratio test for a mean shift under Gaussian noise.
3. Why is binary segmentation not guaranteed to find the globally optimal segmentation for a given number of change points, while PELT is?
4. In the CUSUM ARL0/delay table, why does increasing $h$ from 3 to 20 help with false alarms but hurt detection speed? What real-world consideration should drive the choice of $h$?
5. The Nile data example showed two "reasonable" penalty choices for PELT giving 1 vs. 9 change points. What does this imply about trusting an automatic change point count without any domain knowledge?
6. Why did `ruptures`' default `jump=5` setting give a different (less accurate) change point year than `jump=1` on the Nile data?

## Notebook

[`19_change_point_detection.ipynb`](./19_change_point_detection.ipynb)

## Further Reading

- Killick, R., Fearnhead, P., Eckley, I.A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost.* Journal of the American Statistical Association, 107(500), 1590-1598. (Introduces PELT; uses the Nile dataset as a running example.)
- Page, E.S. (1954). *Continuous Inspection Schemes.* Biometrika, 41(1/2), 100-115. (Original CUSUM paper.)
- Cobb, G.W. (1978). *The Problem of the Nile: Conditional Solution to a Changepoint Problem.* Biometrika, 65(2), 243-251. (Classic early analysis of this exact dataset.)
- Truong, C., Oudre, L., Vayatis, N. (2020). *Selective Review of Offline Change Point Detection Methods.* Signal Processing, 167. (Survey; also the paper behind the `ruptures` package.)
- [ruptures documentation](https://centre-borelli.github.io/ruptures-docs/)

---
[← Back to Time Series Analysis](../README.md)
