# Methodology Notes

## Problem Framing

Replicate Liu et al. (2025, *Knowledge-Based Systems*) — a social media sentiment spatial model — on Taiwan's TAIEX index, and propose an LLM-enhanced Bullish Index to address PTT's irony-heavy comment culture.

---

## Data Sources

| Index | Source | Measures |
|-------|--------|---------|
| Attention Index (AI) | Google Trends | Investor search intensity |
| Bullish Index (BI) | PTT Stock board | Retail sentiment direction |
| Propagation Index (PI) | PTT Stock board | Community engagement (PCA) |

**Training**: 2021-01-04 – 2024-12-31 (970 trading days)  
**Test**: 2025-01-02 – 2026-01-30 (262 trading days)

### Why YouTube Was Rejected

During scoping, YouTube was evaluated as a PI data source but rejected due to data sparsity:

| Metric | PTT | YouTube |
|--------|-----|---------|
| Day coverage | 100% | 42.8% |
| Daily avg | 78.4 posts | 2.17 videos |
| 2021 monthly avg | 1,600+ | 5–9 |

With median 1 video/day, PCA dimensionality reduction on a 1-sample/day input would produce statistically meaningless components. Both BI and PI therefore use PTT — consistent with the original paper's single-platform design.

---

## Temporal Alignment

Non-trading days (weekends, holidays) exist in PTT data but not in TAIEX. Strategy: sum PTT metrics for all non-trading days and attribute to the next trading day. This mirrors how retail traders process weekend information on Monday's open.

```
Friday data  →  Friday row
Saturday + Sunday → Monday row  (aggregated)
```

---

## Sentiment Space Model

Three z-scored indices (AI, BI, PI) form a 3D coordinate per trading day. The model searches for a **Sentiment-Driven Subspace (SDS)** — a region where the historical majority of points belong to the same market trend (up or down).

### HDS Boundary (Hyperdimensional Sphere)

The SDS boundary is an **axis-aligned ellipsoid** in 3D space. A point is *inside* the ellipsoid if:

```
(AI/r_AI)² + (BI/r_BI)² + (PI/r_PI)² ≤ 1
```

Radii `(r_AI, r_BI, r_PI)` are optimized by PSO over the trailing window of training data.

### PSO Optimization

- 30 particles × 100 iterations  
- Fitness = `coverage × accuracy` over the training window  
- Four sub-models: A (AI×BI), B (AI×PI), C (BI×PI), D (AI×BI×PI)

### Multi-Filter Prediction

A prediction is issued only when all active filters pass:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Support | ≥ 2 historical matches | Avoid one-shot patterns |
| Gini impurity | < 0.48 | Require dominant trend |
| k-NN alignment | majority agree | Local consistency check |

---

## Methodological Fixes

Two common but subtle issues were identified and corrected:

### 1. Z-Score Data Leakage

**Problem**: Normalizing with full-period μ/σ leaks test-period statistics into training.  
**Fix**: Compute μ and σ on training data only; apply to test data with training statistics.

```python
mu = train_df[col].mean()
sigma = train_df[col].std()
df[f"{col}_zscore"] = (df[col] - mu) / sigma
```

### 2. Look-Ahead Bias

**Problem**: If target label `y[t] = 1` when `close[t] > close[t-1]`, and features are also from day `t`, the model predicts today using today's data.  
**Fix**: `y = returns.shift(-1)` — use day `t` features to predict day `t+1` return.

---

## LLM Enhancement

PTT's comment culture creates noise in push/boo counts:
- "推 早安大爆崩" — social greeting, not bearish signal  
- "噓 又漲了" — sarcastic boo on a rising stock  

**Solution**: Use Claude Opus 4.6 to semantically classify each article's comment thread as +1 (bullish), 0 (neutral), or -1 (bearish), replacing the raw push/boo ratio.

**Hybrid BI design** (avoids re-labeling 970 training days):
- Training period: original push/boo ratio BI  
- Test period: LLM-classified BI  

PSO trains on ratio-BI patterns; at inference, LLM-BI is the input signal.

---

## Grid Search Results (96 combinations)

Parameters searched: α ∈ {0.20, 0.25, 0.30}, θ ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}, window ∈ {20, 40, 60} days, version ∈ {Baseline, LLM-BI}

| Configuration | Version | Coverage | Accuracy |
|---------------|---------|----------|----------|
| w=40, α=0.25, θ=0.75 | **LLM-BI** | 6.2% | **81.2%** |
| w=40, α=0.25, θ=0.75 | Baseline | 6.1% | 68.8% |
| w=20, α=0.20, θ=0.65 | **LLM-BI** | **30.4%** | **62.0%** |
| w=20, α=0.20, θ=0.65 | Baseline | 27.2% | 59.2% |

LLM-BI outperforms Baseline on accuracy across nearly all 48 parameter combinations.

---

## Comparison with Original Paper

| Aspect | Liu et al. (2025) | This Study |
|--------|-------------------|------------|
| Market | Weibo + A-share | PTT + TAIEX |
| BI source | Weibo sentiment | PTT push/boo ratio → LLM |
| Data leakage | Not discussed | Explicitly corrected |
| Best accuracy | ~72% | 81.2% (low coverage) |
| Practical accuracy | ~65% | 62.0% (30% coverage) |
