# Reproduction Guide

Steps to reproduce the results reported in the README.

**Status:** Exploratory technical report. Results validated on NYU Depth V2 only.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | Not required | CUDA-capable (accelerates ~38x: from ~38h to ~1h) |
| Storage | 10 GB | 50 GB (full dataset) |
| Python | 3.9+ | 3.10+ |
| OS | Linux/macOS/Windows | Ubuntu 20.04+ |

**Dependencies:** numpy, pillow, scipy, PyTorch (optional, for acceleration)

---

## Step 1: Clone Repository

```bash
git clone https://github.com/Hanzzel-corp/nct-depth-motif.git
cd nct-depth-motif
```

---

## Step 2: Setup Environment

```bash
# Create and install dependencies
bash setup_env.sh

# Activate environment
source .venv/bin/activate

# Verify installation
python3 -c "import torch, numpy, scipy, PIL; print('✓ All installed')"
```

**GPU Note:** If you have CUDA, install PyTorch specifically:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: Download Dataset

### NYU Depth V2 (Labeled)

1. Visit: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Download **Labeled dataset** (~2.8 GB)
3. Extract to `dataset/`:

```
dataset/
├── rgb/
│   ├── 000001.png
│   └── ... (1449 images)
└── depth/
    ├── 000001.png
    └── ... (1449 depth maps)
```

### Alternative: Use NYU Toolbox

To extract synchronized RGB-D pairs from raw data:

```bash
# Requires MATLAB
# Follow instructions on official NYU site
```

See [`dataset/README.md`](../../dataset/README.md) for troubleshooting.

---

## Step 4: Run Experiments

### Option A: Grouped Split (30 runs)

**Estimated time:** ~1 hour (GPU) / ~38 hours (CPU)

```bash
bash examples/run_grouped_split.sh
```

**Generated outputs:**
| File | Description |
|------|-------------|
| `9B02_grouped_split_results.json` | Detailed results per run |
| `9B02_grouped_split_summary.csv` | Aggregated statistical summary |
| `9B02_grouped_split_weights.csv` | Learned weight tables |

### Option B: Scene Leave-One-Out (24 runs)

**Estimated time:** ~2-3 hours (GPU) / ~80+ hours (CPU)

```bash
bash examples/run_scene_loo.sh
```

**Note:** Requires `results/scenes_auto.csv` (included in repo).

---

## Step 5: Verify Reproduction

### Compare with Reference Results

```bash
# Compare your summary with reference
diff 9B02_grouped_split_summary.csv results/grouped_split_30runs_summary.csv
```

### What Should Match

| Column | Tolerance |
|--------|-----------|
| `p_sp`, `p_auc`, `p_f1` | **Exact** (with same seed) |
| `d_sp`, `d_auc`, `d_f1` | ±0.001 (due to GPU operation ordering) |
| Metric signs | Must be preserved |

### What Can Vary (acceptable)

- Exact `d_*` values due to GPU operation ordering differences
- Timestamps in files
- Row order in CSV (if there are ties)

---

## Troubleshooting

### "No module named torch"

```bash
source .venv/bin/activate
pip install torch numpy pillow scipy
```

### "CUDA out of memory"

```bash
# Reduce chunk size for random baselines
python3 src/motif_survival_grouped.py \
    ... \
    --gpu-random-chunk 16  # default: 32
```

### "No images found"

```bash
# Verify structure
ls dataset/rgb | head -5
ls dataset/depth | head -5

# Check dataset README
cat dataset/README.md
```

### Very Different Results

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| High p-values | Different seeds | Use same seeds as example |
| NaN metrics | Incomplete dataset | Verify all images present |
| Much slower | CPU vs GPU | Check `torch.cuda.is_available()` |

---

## Quick Validation (1 run)

For quick testing without waiting for 30 runs:

```bash
python3 src/motif_survival_grouped.py \
    --depth ./dataset/depth \
    --target combined \
    --alpha 0.03 \
    --seeds 11 \
    --random-baselines 256 \
    --device cuda \
    --split-mode grouped \
    --group-strategy numeric_block \
    --group-size 50
```

Time: ~2 minutes (GPU) / ~1 hour (CPU)

You should see `p < 0.05` on main metrics.

---

## Reporting Issues

If you find unexplained discrepancies:

1. Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
2. Save full log: `bash examples/run_grouped_split.sh 2>&1 | tee run.log`
3. Open issue with:
   - Operating system and version
   - PyTorch and CUDA versions (if applicable)
   - Log file
   - Generated summary.csv file
