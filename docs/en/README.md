# NCT — Discrete Symbolic Motif Validation on Depth Maps

Experimental validation of a discrete symbolic representation of local depth descriptors, evaluated on the NYU Depth V2 dataset.

**Status:** Exploratory technical report. Not peer-reviewed.  
Statistically significant but modest magnitude results.  
Not yet compared against classical state-of-the-art edge detectors (Canny, Sobel, HED).

---

## Quick Links

- [Method Details](METHOD.md) — Algorithm pipeline, hyperparameters, metrics
- [Interpreting Results](INTERPRETATION.md) — How to read p-values, effect sizes
- [Reproducing Results](REPRODUCING.md) — Step-by-step setup guide
- [Project History](HISTORY.md) — Origins, validation process, pruning decisions
- [Spanish Version](../README.md) — Versión en español

---

## Project Origin

This work emerged from a symbolic framework called **NCT (Números Cuánticos Tridimensionales / 3D Quantum Numbers)** that I developed autodidactically over several months, without prior formal mathematical training.

The original intuition: work with ordered tuples over a discrete alphabet of four symbols `{+, -, 0, ~}` and binary operations between them to represent geometric states.

I arrived at this representation through direct experimentation, not from the mathematical literature. This means many ideas in NCT have known counterparts in formal disciplines:

| NCT Concept | Standard Technical Equivalent |
|-------------|------------------------------|
| Ordered tuple `(x, y, z)` with `x, y, z ∈ {+, -, 0, ~}` | Cartesian product of a 4-element finite set |
| Operations `⊕`, `⊗` between states | Binary operations on a finite commutative magma |
| Discretization of gradients and laplacian into 4 levels | Vector quantization / codebook learning |
| Weight table by motif | Lookup-table classifier over quantized features |
| "Motif survival" | Per-feature lift estimation with shrinkage |
| State `~` as transition marker | Tri-state / multi-valued logic (Belnap, Łukasiewicz) |

**Recognizing these equivalences doesn't invalidate my path of discovery, but contextualizes it: what is validated here is not "new mathematics", but a concrete discrete representation technique whose specific form came from NCT.**

What is genuinely original to this work:

- The choice of four specific states `{+, -, 0, ~}` with the fourth symbol (`~`) explicitly handling transition.
- The axes of discretization (x gradient, y gradient, local laplacian) and their thresholds.
- The triangular ambiguity gate over the classical delta.
- The validation procedure with empirical p-values against random permutations of learned weights.

---

## Hypotheses

**H1:** The learned motif weights in training, applied to an independent test set, outperform random permutations of the same weights in geometric rupture detection metrics (Spearman, AUC top-20%, F1 top-20%) with statistical significance.

**H0 (null):** The learned weights are indistinguishable from randomly permuted weights (the motif → rupture association is noise).

---

## Method

1. **Input:** Depth maps from NYU Depth V2 (16-bit PNG, converted to meters).

2. **Local descriptors:** For each pixel, compute horizontal gradient (`Sx`), vertical gradient (`Sy`), and local laplacian (`Sz`).

3. **Discretization:** Map each continuous component to one of four states `{+, -, 0, ~}` using fixed thresholds. Each pixel is labeled with a motif `(Sx, Sy, Sz)` belonging to one of 4³ = 64 possible motifs.

4. **Target:** "Geometric rupture" derived from the residual between the fitted global plane and the local plane, combined with local plane improvement over global.

5. **Training:** Over training frames, compute the mean lift of each motif with respect to the target, with shrinkage to regularize rare motifs. Result: table of 64 weights.

6. **Testing:** Apply the weight table as residual correction `delta + alpha * gate * weight_table[motif]` over test frames. Compare against:
   - `classical_delta`: delta alone without correction.
   - `random`: randomly permuted weights (256 permutations).

7. **Significance:** Empirical p-value = fraction of random permutations that match or exceed the learned model.

---

## Results

### Grouped numeric block split (30 runs)

3 alphas × 10 seeds, blocks of 50 consecutive frames, 654 frames from NYU Depth V2 test set, 256 random baselines:

| Model | p_F1 | p_AUC | p_Spearman | < 0.05 in |
|-------|------|-------|------------|-----------|
| `motif_survival` | 0.0039 | 0.0079 | 0.0039 | 30/30 |
| `motif_survival_binary` | 0.0039 | 0.0039 | 0.0039 | 30/30 |
| `motif_survival_pos_only` | 0.0042 | 0.0926 | 0.0525 | F1 30/30, AUC 1/30 |
| `motif_survival_neg_only` | 0.1540 | 0.2432 | 0.0660 | 0/30 |

ΔAUC vs random: +0.0038 (motif_survival), +0.0070 (binary).  
ΔF1 vs random: +0.0052 (motif_survival), +0.0054 (binary).

### Leave-one-cluster-out (24 runs)

3 alphas × 8 RGB clusters (k-means with mean confidence 0.21), 654 frames, 256 random baselines:

| Model | p_F1 | p_AUC | p_Spearman | < 0.05 in |
|-------|------|-------|------------|-----------|
| `motif_survival` | 0.0087 | 0.0168 | 0.0084 | 24/24 |
| `motif_survival_binary` | 0.0090 | 0.0078 | 0.0078 | 24/24 |

ΔAUC vs random: +0.0038. ΔF1 vs random: +0.0052. Magnitude stable across clusters (range +0.0034 to +0.0043).

### Top Stable Motifs

Motifs with highest lift and lowest variance across independent splits:

| ID | Motif (Sx,Sy,Sz) | Weight (mean ± std) | Lift | Appears in |
|----|-------------------|---------------------|------|------------|
| 22 | (+, +, -) | 0.998 ± 0.006 | +0.17 | 30/30 splits |
| 26 | (+, -, -) | 0.985 ± 0.006 | +0.17 | 30/30 splits |
| 38 | (-, +, -) | 0.968 ± 0.008 | +0.17 | 30/30 splits |
| 32 | (-, 0, 0) | -0.755 ± 0.008 | -0.13 | 30/30 splits |

The high stability of top weights across independent splits suggests the representation captures reproducible structure.

---

## What This IS

- A reproducible validation pipeline with empirical p-values.
- Statistical evidence that the discrete 3D motif representation is not random noise in this domain.
- A reusable harness for auditing other ML/CV pipelines.

## What This Is NOT

- Not a state-of-the-art edge detector.
- Not "new mathematics". It is vector quantization applied to depth maps, arrived at through a personal path.
- Not a peer-reviewed paper. It is a technical report published for external audit.

---

## Limitations

1. **Small absolute magnitude.** ΔF1 ≈ +0.005 over random. Statistically significant, practically modest.

2. **No comparison against Canny, Sobel, HED.** Pending.

3. **Cluster scenes, not semantic scenes.** The 8 leave-one-out clusters are k-means over RGB with mean confidence 0.21, not the official NYU semantic labels.

   ![Auto scene clusters](../../docs/assets/scenes_auto_contact_sheet.png)
   
   *Contact sheet with the 8 RGB clusters generated automatically. Visual validation allows auditing that the cluster-based split is interpretable.*
   
   Validation with official `scene_types.txt` is pending.

4. **Only NYU Depth V2.** No cross-dataset validation (ScanNet, KITTI, SUN RGB-D).

5. **Target derived from input.** The "rupture target" is computed from the depth map itself (combination of plane_gap + improvement), not from manually annotated edge ground truth.

---

## Reproducing the Results

### Clone

```bash
git clone https://github.com/Hanzzel-corp/nct-depth-motif.git
cd nct-depth-motif
```

### Requirements

- Python 3.9+
- numpy, pillow, scipy
- PyTorch + CUDA (optional, to accelerate from ~38 hours to ~1 hour)

```bash
pip install -r requirements.txt
```

### Download Dataset

NYU Depth V2 from the official site:  
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

Expected structure:
```
dataset/
├── rgb/
│   ├── 000001.png
│   ├── 000002.png
│   └── ... (1449 RGB images)
└── depth/
    ├── 000001.png
    ├── 000002.png
    └── ... (1449 depth maps)

results/
├── grouped_split_30runs_summary.csv
├── scene_loo_24runs_summary.csv
├── top_motifs_grouped.csv
└── top_motifs_scene_loo.csv

src/
├── motif_survival_grouped.py
└── motif_survival_scene_loo.py
```

⚠️ **Data policy:** The NYU Depth V2 dataset is **not included** in this repository. Download it from the official source and place RGB/depth files under `dataset/rgb` and `dataset/depth`.

---

## Quick Reproduction Commands

### Step 1: Verify dataset structure
```bash
ls dataset/rgb | head -5
ls dataset/depth | head -5
python3 -c "from PIL import Image; import numpy as np; arr=np.array(Image.open('dataset/depth/000001.png')); print('shape:', arr.shape, 'dtype:', arr.dtype, 'range:', arr.min(), '-', arr.max())"
```

### Step 2: Run grouped split validation (30 runs, ~1 hour GPU)

Using the values reported in the table (3 alphas × 10 seeds):

```bash
python3 src/motif_survival_grouped.py \
  --depth ./dataset/depth \
  --target combined \
  --alpha 0.02,0.03,0.04 \
  --seeds 11,22,33,44,55,66,77,88,99,111 \
  --random-baselines 256 \
  --device cuda \
  --split-mode grouped \
  --group-strategy numeric_block \
  --group-size 50 \
  --depth-scale 1000 \
  --fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362
```

### Step 3: Run scene LOO validation (24 runs, ~2-3 hours GPU)

Using the values reported in the table (3 alphas × 8 clusters):

```bash
python3 src/motif_survival_scene_loo.py \
  --depth ./dataset/depth \
  --target combined \
  --alpha 0.02,0.03,0.04 \
  --seeds 11,22,33 \
  --random-baselines 256 \
  --device cuda \
  --split-mode scene_loo \
  --scene-map ./results/scenes_auto.csv \
  --depth-scale 1000 \
  --max-size 160 \
  --fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362
```

---

## Quick Results Summary

| Model | ΔAUC vs random | ΔF1 vs random | p(AUC) | p(F1) | Significant runs |
|-------|---------------|---------------|--------|-------|------------------|
| **motif_survival_binary** | **+0.0071** | **+0.0054** | **0.0078** | **0.0090** | **24/24** |
| motif_survival | +0.0038 | +0.0052 | 0.0168 | 0.0087 | 24/24 |
| motif_survival_pos_only | +0.0024 | +0.0042 | 0.1105 | 0.0123 | F1 only |
| motif_survival_neg_only | +0.0015 | +0.0013 | 0.2542 | 0.1744 | no |

**Note:** `motif_survival_binary` (sign only +1/-1/0) outperforms the full model, confirming that weight direction is more informative than magnitude.

### Alternative: Run example scripts

**Option A: Grouped split (30 runs, ~1 hour with GPU)**
```bash
bash examples/run_grouped_split.sh
```

**Option B: Scene leave-one-out (24 runs)**
```bash
bash examples/run_scene_loo.sh
```

### Verify Output

Compare your generated `9B02_grouped_split_summary.csv` with `results/grouped_split_30runs_summary.csv`.  
The columns `p_sp`, `p_auc`, `p_f1` should be identical (with same seed).  
The `d_*` columns may vary slightly due to GPU operation ordering but should maintain sign and magnitude.

---

## Citation

```bibtex
@software{nct_depth_motif,
  author = {Jose Zamora},
  title = {NCT Depth Motif: Discrete Symbolic Motif Validation on Depth Maps},
  year = {2026},
  url = {https://github.com/Hanzzel-corp/nct-depth-motif}
}
```

---

**Author:** Jose Zamora  
**Year:** 2026  
**License:** MIT
