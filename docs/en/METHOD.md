# Technical Method Details

**Status:** Exploratory technical report. Not peer-reviewed.

This document describes the complete NCT method pipeline for detecting discontinuities in RGB-D images.

**Main finding:** The discrete 3D motif representation shows statistically significant signal (p < 0.01) although with modest magnitude (ΔAUC ≈ +0.004 vs random).

---

## Complete Pipeline

### 1. Input Preprocessing

```
RGB (optional) + Depth → 3D Surface P(X,Y,Z)
```

| Step | Operation | Formula/Notes |
|------|-----------|---------------|
| Load | Read RGB-D pairs | NYU Depth V2 dataset |
| Normalize | Convert to meters | `depth_m = depth_uint16 / 1000.0` |
| Project | Back-project to 3D | Using camera intrinsics |
| Normals | Compute local surface normals | `n = ∇S / |∇S|` |

### 2. NCT Descriptor Extraction

Computation of directional components:

- **Sx**: Laplacian in X direction of the surface
- **Sy**: Laplacian in Y direction of the surface  
- **Sz**: Laplacian in Z direction of the surface

### 3. State Discretization

Each component is quantized into 4 states:

| State | Condition | Meaning |
|-------|-----------|---------|
| `+` | value > +threshold | Strong positive curvature |
| `-` | value < -threshold | Strong negative curvature |
| `0` | \|value\| ≤ threshold | Flat region |
| `~` | in transition band | Transition zone |

**Total motifs:** 4³ = 64 possible `(Sx, Sy, Sz)` combinations

### 4. Target Definition

The rupture target combines three signals:

```
target = λ₁·depth_edges + λ₂·normal_edges + λ₃·plane_residual
```

| Component | Description |
|-----------|-------------|
| `depth_edges` | High gradients in depth map |
| `normal_edges` | Sharp changes in normal orientation |
| `plane_residual` | Local plane fitting error |

### 5. Weight Learning (Training)

For each motif `m`:

```
lift(m) = E[target | motif = m]  (with Bayesian shrinkage)
weight[m] = normalize(lift(m)) ∈ [-1, 1]
```

**Shrinkage applied:** Motifs with few observations (`< min_count`) are shrunk towards the global mean.

### 6. Prediction (Testing)

```
delta = classical_depth_edge_score
ambiguity_gate = triangular_gate(delta)  # [0,1]
score = delta + alpha · ambiguity_gate · weight[motif]
```

The `gate` suppresses NCT correction where classical delta already discriminates well.

---

## Hyperparameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `alpha` | NCT correction weight | 0.02 - 0.04 | 0.03 |
| `state_threshold` | Threshold for `+`/`-` states | 0.1 - 0.5 | 0.2 |
| `tilde_band` | Width of transition band for `~` | 0.05 - 0.2 | 0.1 |
| `min_count` | Minimum occurrences for weight confidence | 50 - 200 | 100 |
| `shrinkage_power` | Bayesian shrinkage strength | 0.5 - 2.0 | 1.0 |
| `random_baselines` | Number of permutations for p-value | 256 - 1024 | 256 |

---

## Evaluation Metrics

All metrics are computed in the **AMBIGUOUS_ONLY** zone (where `delta` is in intermediate band and doesn't separate well).

### Spearman ρ
Rank correlation between predicted score and real target.

```
ρ = corr_rank(score, target)
```

### AUC top-20%
Area under ROC curve restricted to top 20% of scores.

```
AUC@20 = ROC_AUC(score > percentile(score, 80), target)
```

### F1 top-20%
F1-score between top-20% predicted pixels vs top-20% target pixels.

```
F1@20 = F1(pred_top20, target_top20)
```

**Note:** The NON_AMBIGUOUS zone is excluded because classical delta already discriminates correctly there.

---

## Empirical p-value

Statistical significance computation against random baselines:

```
p_metric = (1 + #{random_runs : metric(random) ≥ metric(model)}) / (1 + N_random)
```

| N_random | Minimum p-value | Interpretation |
|----------|-----------------|----------------|
| 256 | 1/257 ≈ 0.0039 | Model exceeds ALL random runs |
| 1024 | 1/1025 ≈ 0.0010 | Maximum confidence |

---

## Evaluation Zones

```
Zone            | Delta Condition        | NCT Correction
----------------|------------------------|----------------
NON_AMBIGUOUS   | delta < low_th         | Suppressed (not applicable)
AMBIGUOUS_ONLY  | low_th ≤ delta ≤ high_th | Active (gate ≈ 1)
NON_AMBIGUOUS   | delta > high_th        | Suppressed (not applicable)
```

The logic: don't fix what isn't broken. If delta already separates well (extreme zones), don't apply correction.

---

## Output Files

| File | Description |
|------|-------------|
| `*_results.json` | Detailed results per run |
| `*_summary.csv` | Aggregated statistical summary |
| `*_weights.csv` | Learned weight tables per seed |
| `*_pairs.csv` | RGB-D pairs processed information |

---

For setup and execution instructions, see [REPRODUCING.md](REPRODUCING.md).
