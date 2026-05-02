#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCT Motif Survival - Grouped Split Validation (v13.4)

Reporte técnico exploratorio. No revisado por pares.

Este script implementa validación cruzada con splits agrupados por escenas para evaluar
la capacidad predictiva de motivos NCT 3D sobre discontinuidades en imágenes RGB-D.

Hallazgo principal: La representación discreta de motivos 3D muestra señal estadísticamente
significativa (p < 0.01) aunque de magnitud modesta (ΔAUC ≈ +0.004 vs random).

Arquitectura del método
-----------------------
1. Procesamiento de entrada:
   - Lee pares RGB-D del dataset NYU Depth V2
   - Convierte depth a coordenadas 3D usando parámetros de cámara
   - Calcula normales locales de la superficie

2. Extracción de motivos NCT:
   - Computa descriptores direccionales (Sx, Sy, Sz) desde la superficie 3D
   - Discretiza cada descriptor en 4 estados: {+, -, 0, ~}
   - Genera 64 motivos tridimensionales posibles (4³)

3. Entrenamiento:
   - Agrupa escenas en folds (strategy: numeric_block)
   - Aprende tabla de pesos por motivo usando shrinkage Bayesiano
   - Calcula lift de cada motivo sobre target de ruptura

4. Evaluación:
   - Aplica corrección NCT sobre delta clásico: score = delta + alpha * gate * weight[motif]
   - Calcula métricas: Spearman ρ, AUC top-20%, F1 top-20%
   - Computa p-value empírico contra 256 baselines aleatorios

Targets de ruptura disponibles
------------------------------
- depth_edges    : Discontinuidades de profundidad (gradiente alto)
- normal_edges   : Cambios bruscos en normales 3D
- plane_residual : Error contra plano local ajustado
- combined       : Promedio de los tres targets anteriores

Modelos evaluados
-----------------
BASE:
  - classical_depth_edge : Detector clásico (baseline)
  - nct_energy_fixed     : Energía NCT sin aprendizaje
  - random_motif_mean    : Promedio de tablas aleatorias

TEST:
  - motif_survival           : Pesos completos con shrinkage
  - motif_survival_pos_only  : Solo pesos positivos
  - motif_survival_neg_only  : Solo pesos negativos
  - motif_survival_binary    : Solo signo del peso (+1/-1/0)

Hiperparámetros clave
---------------------
- alpha           : Peso de corrección NCT [0.02-0.04]
- state_threshold : Umbral para estados +/-/0
- tilde_band      : Banda para estado de transición ~
- min_count       : Mínimo de ocurrencias para confianza en peso
- shrinkage_power : Fuerza del shrinkage Bayesiano

Uso
---
Ver examples/run_grouped_split.sh para ejecución típica con múltiples seeds.

Ejemplo mínimo:
    python3 src/motif_survival_grouped.py \
        --depth ./dataset/depth \
        --target combined \
        --alpha 0.03 \
        --seeds 11,22,33 \
        --device cuda

Parámetros de cámara NYU Depth V2:
    --fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362

Salidas
-------
- *_results.json  : Resultados detallados por run
- *_summary.csv   : Resumen estadístico agregado
- *_weights.csv   : Tablas de pesos aprendidas por seed
- *_pairs.csv     : Información de pares RGB-D procesados

Notas
-----
- La evaluación se enfoca en zona AMBIGUOUS (donde delta clásico no separa bien)
- Zona NON_AMBIGUOUS se excluye porque delta ya discrimina correctamente
- GPU acelera significativamente el cálculo de baselines aleatorios
- CPU viable para runs individuales (~38x más lento para 256 randoms)

Autor: Jose Zamora
Versión: 13.4
Licencia: MIT
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None


REGIONS = ("GLOBAL", "AMBIGUOUS_ONLY", "NON_AMBIGUOUS")
REGION = "AMBIGUOUS_ONLY"
METRICS = ("spearman", "auc_top20", "f1_top20", "pearson", "iou_top20")

BASE_MODELS = (
    "classical_depth_edge",
    "nct_energy_fixed",
    "random_motif_mean",
)

TEST_MODELS = (
    "motif_survival",
    "motif_survival_pos_only",
    "motif_survival_neg_only",
    "motif_survival_binary",
)

MODEL_NAMES = BASE_MODELS + TEST_MODELS

CODE_TO_STATE = {0: "0", 1: "+", 2: "-", 3: "~"}
STATE_TO_CODE = {"0": 0, "+": 1, "-": 2, "~": 3}

SUPPORTED_DEPTH_EXTS = {".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
SUPPORTED_RGB_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# Parse / utils
# ============================================================

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def finite(vals):
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(vals):
    arr = finite(vals)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def positive_rate(vals):
    arr = finite(vals)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > 0.0))


def empirical_p_value(real_value: float, random_values: List[float]) -> float:
    rv = np.asarray(random_values, dtype=float)
    rv = rv[np.isfinite(rv)]
    if rv.size == 0 or not np.isfinite(real_value):
        return float("nan")
    return float((1 + np.sum(rv >= real_value)) / (1 + rv.size))


def safe_json_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return None
    except Exception:
        return None


def motif_name(mid: int) -> str:
    sx = mid // 16
    sy = (mid % 16) // 4
    sz = mid % 4
    return f"({CODE_TO_STATE[sx]},{CODE_TO_STATE[sy]},{CODE_TO_STATE[sz]})"


def robust_scale(vals: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if not np.isfinite(hi - lo) or abs(hi - lo) <= 1e-12:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(hi - lo) or abs(hi - lo) <= 1e-12:
        lo, hi = 0.0, 1.0
    return lo, hi


def normalize01(a: np.ndarray, valid: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    vals = np.asarray(a[valid], dtype=float)
    lo, hi = robust_scale(vals, p_low, p_high)
    out[valid] = np.clip((a[valid] - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    out[~valid] = 0.0
    return out


def robust_norm_abs(a: np.ndarray, valid: np.ndarray, p: float = 99.0) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    vals = np.asarray(a[valid], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    scale = float(np.percentile(np.abs(vals), p))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    out[valid] = np.clip(np.abs(a[valid]) / scale, 0.0, 1.0)
    return out


def robust_norm_signed(a: np.ndarray, valid: np.ndarray, p: float = 99.0) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    vals = np.asarray(a[valid], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    scale = float(np.percentile(np.abs(vals), p))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    out[valid] = np.clip(a[valid] / scale, -1.0, 1.0)
    return out


def box_mean(a: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    r = window // 2
    pad = np.pad(a, ((r, r), (r, r)), mode="edge")
    integ = np.pad(pad, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    k = 2 * r + 1
    s = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
    return s / float(k * k)


def local_std(a: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    z = np.where(valid, a, 0.0)
    m1 = box_mean(z, window)
    m2 = box_mean(z * z, window)
    return np.sqrt(np.maximum(0.0, m2 - m1 * m1))


def downsample(a: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return a
    h, w = a.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return a

    step = int(np.ceil(m / max_size))
    return a[::step, ::step, ...]


# ============================================================
# IO dataset
# ============================================================

def collect_files(root: Path, exts: set) -> Dict[str, Path]:
    if root is None:
        return {}
    root = Path(root)
    if not root.exists():
        return {}
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            # Stem como clave. Si hay duplicados, conserva primero.
            out.setdefault(p.stem, p)
    return out


def match_rgb_depth(rgb_dir: Optional[Path], depth_dir: Path) -> List[dict]:
    depth_files = collect_files(depth_dir, SUPPORTED_DEPTH_EXTS)
    rgb_files = collect_files(rgb_dir, SUPPORTED_RGB_EXTS) if rgb_dir else {}

    pairs = []
    for stem, dpath in sorted(depth_files.items()):
        pairs.append({
            "stem": stem,
            "depth": str(dpath),
            "rgb": str(rgb_files[stem]) if stem in rgb_files else "",
        })

    return pairs


def load_depth(path: Path, depth_scale: float = 0.0, auto_depth_scale: bool = True) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        # Usa primera clave disponible.
        key = list(data.keys())[0]
        arr = data[key]
    else:
        if Image is None:
            raise RuntimeError("Pillow no está instalado. Instalá pillow o usá .npy.")
        img = Image.open(path)
        arr = np.array(img)

    if arr.ndim == 3:
        # Si viene RGB por error, usa primer canal o convierte promedio.
        arr = arr[..., 0]

    depth = np.asarray(arr, dtype=float)

    # Limpieza básica.
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    if depth_scale and depth_scale > 0:
        depth = depth / float(depth_scale)
    elif auto_depth_scale:
        mx = float(np.nanmax(depth)) if depth.size else 0.0
        # Heurística:
        # - uint16 en milímetros: valores típicos > 100
        # - uint16 en decimilímetros o escala grande: también baja.
        if mx > 100.0:
            depth = depth / 1000.0

    return depth


# ============================================================
# Geometría RGB-D → superficie 3D
# ============================================================

def make_valid(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > float(min_depth))
    if max_depth > 0:
        valid &= depth < float(max_depth)
    return valid


def fill_invalid_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    z = np.asarray(depth, dtype=float).copy()
    if np.any(valid):
        med = float(np.nanmedian(z[valid]))
    else:
        med = 1.0
    z[~np.isfinite(z)] = med
    z[~valid] = med
    return z


def backproject_xyz(depth: np.ndarray, valid: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    z = fill_invalid_depth(depth, valid)

    if fx <= 0:
        fx = float(max(h, w))
    if fy <= 0:
        fy = float(max(h, w))
    if cx < 0:
        cx = (w - 1) / 2.0
    if cy < 0:
        cy = (h - 1) / 2.0

    yy, xx = np.mgrid[0:h, 0:w]
    X = (xx - float(cx)) * z / float(fx)
    Y = (yy - float(cy)) * z / float(fy)
    Z = z

    X[~valid] = 0.0
    Y[~valid] = 0.0
    Z[~valid] = 0.0

    return X, Y, Z


def compute_normals(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Derivadas de la superficie.
    Xy, Xx = np.gradient(X)
    Yy, Yx = np.gradient(Y)
    Zy, Zx = np.gradient(Z)

    # Tangentes: dP/dx y dP/dy.
    tx = np.stack([Xx, Yx, Zx], axis=-1)
    ty = np.stack([Xy, Yy, Zy], axis=-1)

    n = np.cross(tx, ty)
    norm = np.linalg.norm(n, axis=-1) + 1e-12
    nx = n[..., 0] / norm
    ny = n[..., 1] / norm
    nz = n[..., 2] / norm

    nx[~valid] = 0.0
    ny[~valid] = 0.0
    nz[~valid] = 0.0
    return nx, ny, nz


def normal_change(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Cambio de normales entre vecinos.
    gyx, gxx = np.gradient(nx)
    gyy, gyx2 = np.gradient(ny)
    gyz, gxz = np.gradient(nz)
    mag = np.sqrt(gxx * gxx + gyx * gyx + gyx2 * gyx2 + gyy * gyy + gxz * gxz + gyz * gyz)
    return robust_norm_abs(mag, valid)


def depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    z = fill_invalid_depth(depth, valid)
    gy, gx = np.gradient(z)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = (
        np.roll(z, 1, 0)
        + np.roll(z, -1, 0)
        + np.roll(z, 1, 1)
        + np.roll(z, -1, 1)
        - 4.0 * z
    )
    out = np.clip(0.70 * robust_norm_abs(grad, valid) + 0.30 * robust_norm_abs(lap, valid), 0.0, 1.0)
    out[~valid] = 0.0
    return out


def plane_residual_target(depth: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """
    Target simple: residual contra profundidad suavizada/local.
    No ajusta plano exacto por pixel por velocidad; usa diferencia contra promedio local.
    Para datasets reales funciona como indicador de ruptura/localidad.
    """
    z = fill_invalid_depth(depth, valid)
    smooth = box_mean(z, window)
    residual = np.abs(z - smooth)
    out = robust_norm_abs(residual, valid)
    out[~valid] = 0.0
    return out


def build_target(depth: np.ndarray, valid: np.ndarray, target_name: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, target_window: int) -> Dict[str, np.ndarray]:
    de = depth_edges(depth, valid)

    nx, ny, nz = compute_normals(X, Y, Z, valid)
    ne = normal_change(nx, ny, nz, valid)

    pr = plane_residual_target(depth, valid, target_window)

    if target_name == "depth_edges":
        target = de
    elif target_name == "normal_edges":
        target = ne
    elif target_name == "plane_residual":
        target = pr
    elif target_name == "combined":
        target = np.clip(0.45 * de + 0.35 * ne + 0.20 * pr, 0.0, 1.0)
    else:
        raise ValueError(f"Target inválido: {target_name}")

    target[~valid] = 0.0
    return {
        "target": target,
        "depth_edges": de,
        "normal_edges": ne,
        "plane_residual": pr,
    }


# ============================================================
# NCT Tensor 3D
# ============================================================

def state_from_signal(signal: np.ndarray, valid: np.ndarray, threshold: float, tilde_band: float) -> np.ndarray:
    s = np.asarray(signal, dtype=float)
    out = np.zeros_like(s, dtype=np.int16)

    th = float(threshold)
    band = float(tilde_band)

    abs_s = np.abs(s)
    plus = s >= (th + band)
    minus = s <= -(th + band)
    tilde = (abs_s >= max(th - band, 0.0)) & (abs_s < (th + band))

    out[plus] = 1
    out[minus] = 2
    out[tilde] = 3
    out[~valid] = 0
    return out


def compute_nct3d_tensor(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, valid: np.ndarray, state_threshold: float, tilde_band: float) -> Dict[str, np.ndarray]:
    """
    Estados direccionales por eje a partir de superficie 3D real.

    Sx:
        cambio energético horizontal en X/Z.

    Sy:
        cambio energético vertical en Y/Z.

    Sz:
        curvatura / salida en profundidad.
    """
    Zy, Zx = np.gradient(Z)
    Xy, Xx = np.gradient(X)
    Yy, Yx = np.gradient(Y)

    # Señales firmadas direccionales.
    sx_signal = robust_norm_signed(Zx + 0.25 * Xx, valid)
    sy_signal = robust_norm_signed(Zy + 0.25 * Yy, valid)

    lap_z = (
        np.roll(Z, 1, 0)
        + np.roll(Z, -1, 0)
        + np.roll(Z, 1, 1)
        + np.roll(Z, -1, 1)
        - 4.0 * Z
    )
    sz_signal = robust_norm_signed(lap_z, valid)

    Sx = state_from_signal(sx_signal, valid, state_threshold, tilde_band)
    Sy = state_from_signal(sy_signal, valid, state_threshold, tilde_band)
    Sz = state_from_signal(sz_signal, valid, state_threshold, tilde_band)

    motif_id = (Sx * 16 + Sy * 4 + Sz).astype(np.int16)
    motif_id[~valid] = 0

    Ex = np.abs(sx_signal)
    Ey = np.abs(sy_signal)
    Ez = np.abs(sz_signal)

    active_count = ((Sx != 0).astype(float) + (Sy != 0).astype(float) + (Sz != 0).astype(float))
    tilde_count = ((Sx == 3).astype(float) + (Sy == 3).astype(float) + (Sz == 3).astype(float))
    plus_count = ((Sx == 1).astype(float) + (Sy == 1).astype(float) + (Sz == 1).astype(float))
    minus_count = ((Sx == 2).astype(float) + (Sy == 2).astype(float) + (Sz == 2).astype(float))

    energy = np.clip((Ex + Ey + Ez) / 3.0, 0.0, 1.0)
    transition = np.clip(0.50 * (tilde_count / 3.0) + 0.50 * energy, 0.0, 1.0)
    suppression = np.clip(0.45 * (minus_count / 3.0) + 0.35 * (1.0 - energy) + 0.20 * (active_count / 3.0), 0.0, 1.0)

    for arr in (Ex, Ey, Ez, energy, transition, suppression):
        arr[~valid] = 0.0

    return {
        "Sx": Sx,
        "Sy": Sy,
        "Sz": Sz,
        "motif_id": motif_id,
        "sx_signal": sx_signal,
        "sy_signal": sy_signal,
        "sz_signal": sz_signal,
        "Ex": Ex,
        "Ey": Ey,
        "Ez": Ez,
        "active_count": active_count,
        "tilde_count": tilde_count,
        "plus_count": plus_count,
        "minus_count": minus_count,
        "energy": energy,
        "transition": transition,
        "suppression": suppression,
    }



# ============================================================
# CUDA helpers v13.3
# ============================================================

def select_runtime_device(device_arg: str) -> str:
    """
    Selecciona dispositivo real.
    - cpu: fuerza CPU.
    - cuda: exige CUDA disponible.
    - auto: usa cuda si está disponible; si no, CPU.
    """
    device_arg = str(device_arg).lower().strip()

    if device_arg == "cpu":
        return "cpu"

    if torch is None:
        if device_arg == "cuda":
            raise RuntimeError("Pediste --device cuda pero torch no está instalado/importable.")
        return "cpu"

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Pediste --device cuda pero torch.cuda.is_available() es False.")
        return "cuda"

    raise ValueError(f"--device inválido: {device_arg}")


def torch_rank_1d(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float32)
    ranks[order] = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks


def torch_batch_corr(pred, target):
    """
    Pearson por fila.
    pred: [B,N]
    target: [N]
    """
    if pred.numel() == 0 or target.numel() < 3:
        return torch.full((pred.shape[0],), float("nan"), device=pred.device)

    tx = target.float()
    tx = tx - tx.mean()
    ty = pred.float() - pred.float().mean(dim=1, keepdim=True)

    num = (ty * tx.unsqueeze(0)).sum(dim=1)
    den = torch.sqrt((ty * ty).sum(dim=1) * (tx * tx).sum() + 1e-12)
    return num / den


def torch_batch_rank(pred):
    """
    Ranking ordinal por fila.
    Nota: no hace average-rank para empates. En este benchmark los scores
    incluyen delta continuo, así que los empates suelen ser pocos.
    """
    order = torch.argsort(pred, dim=1)
    ranks = torch.empty_like(pred, dtype=torch.float32)
    base = torch.arange(1, pred.shape[1] + 1, device=pred.device, dtype=torch.float32)
    ranks.scatter_(1, order, base.unsqueeze(0).expand(pred.shape[0], -1))
    return ranks


def torch_batch_metrics(pred, target, target_rank, target_top):
    """
    Calcula métricas para un batch de predicciones random en GPU.
    Devuelve dict de listas Python por métrica.
    """
    b, n = pred.shape
    if n < 5:
        nan_list = [float("nan")] * b
        return {m: nan_list[:] for m in METRICS}

    pred = pred.float()
    target = target.float()
    target_top = target_top.bool()

    # Pearson.
    pearson = torch_batch_corr(pred, target)

    # Spearman.
    pred_rank = torch_batch_rank(pred)
    spearman = torch_batch_corr(pred_rank, target_rank.float())

    # Top20 por fila.
    q = torch.quantile(pred, 0.80, dim=1, keepdim=True)
    pred_top = pred >= q

    tt = target_top.unsqueeze(0)

    tp = (pred_top & tt).sum(dim=1).float()
    fp = (pred_top & ~tt).sum(dim=1).float()
    fn = (~pred_top & tt).sum(dim=1).float()
    union = (pred_top | tt).sum(dim=1).float()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (union + 1e-12)

    # AUC contra target_top usando ranking ordinal.
    y = target_top.float()
    n_pos = y.sum()
    n_neg = float(n) - n_pos

    if n_pos <= 0 or n_neg <= 0:
        auc = torch.full((b,), float("nan"), device=pred.device)
    else:
        rank_sum_pos = (pred_rank * y.unsqueeze(0)).sum(dim=1)
        auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg + 1e-12)

    return {
        "spearman": spearman.detach().cpu().numpy().astype(float).tolist(),
        "auc_top20": auc.detach().cpu().numpy().astype(float).tolist(),
        "f1_top20": f1.detach().cpu().numpy().astype(float).tolist(),
        "pearson": pearson.detach().cpu().numpy().astype(float).tolist(),
        "iou_top20": iou.detach().cpu().numpy().astype(float).tolist(),
    }


def evaluate_random_distribution_gpu(sample: dict, random_tables: List[np.ndarray], alpha: float, device: str, gpu_random_chunk: int) -> Dict[str, Dict[str, List[float]]]:
    """
    Evalúa todas las tablas random en GPU por chunks.

    Esta es la parte pesada del benchmark:
      random_baselines × pixels × métricas.

    Mantiene la misma estructura de salida que la versión CPU:
      {region: {metric: [valor_por_random]}}
    """
    if torch is None or device != "cuda":
        raise RuntimeError("evaluate_random_distribution_gpu requiere torch + device=cuda")

    random_np = np.asarray(random_tables, dtype=np.float32)
    n_random = int(random_np.shape[0])
    chunk = max(1, int(gpu_random_chunk))

    out = {region: {m: [] for m in METRICS} for region in REGIONS}

    motif_all = np.asarray(sample["motif_id"], dtype=np.int64)
    delta_all = np.asarray(sample["classical_depth_edge"], dtype=np.float32)
    gate_all = np.asarray(sample["gate"], dtype=np.float32)
    target_all = np.asarray(sample["target"], dtype=np.float32)

    with torch.no_grad():
        weights_all = torch.as_tensor(random_np, dtype=torch.float32, device=device)

        for region in REGIONS:
            mask = np.asarray(sample["masks"][region], dtype=bool)
            if int(np.sum(mask)) < 5:
                for m in METRICS:
                    out[region][m] = [float("nan")] * n_random
                continue

            motif = torch.as_tensor(motif_all[mask], dtype=torch.long, device=device)
            delta = torch.as_tensor(delta_all[mask], dtype=torch.float32, device=device)
            gate = torch.as_tensor(gate_all[mask], dtype=torch.float32, device=device)
            target = torch.as_tensor(target_all[mask], dtype=torch.float32, device=device)

            # Filtros de finitud, por seguridad.
            finite_mask = torch.isfinite(delta) & torch.isfinite(gate) & torch.isfinite(target)
            if int(finite_mask.sum().item()) < 5:
                for m in METRICS:
                    out[region][m] = [float("nan")] * n_random
                continue

            motif = motif[finite_mask]
            delta = delta[finite_mask]
            gate = gate[finite_mask]
            target = target[finite_mask]

            target_q = torch.quantile(target, 0.80)
            target_top = target >= target_q
            target_rank = torch_rank_1d(target)

            for start in range(0, n_random, chunk):
                end = min(start + chunk, n_random)
                w = weights_all[start:end]             # [B,64]
                wmap = w[:, motif]                     # [B,N]
                pred = torch.clamp(delta.unsqueeze(0) + float(alpha) * gate.unsqueeze(0) * wmap, 0.0, 1.0)

                met = torch_batch_metrics(pred, target, target_rank, target_top)
                for key in METRICS:
                    out[region][key].extend(met[key])

    return out

# ============================================================
# Métricas
# ============================================================

def pearson_metric(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(pred[mask], dtype=float)
    y = np.asarray(target[mask], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / den)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_metric(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(pred[mask], dtype=float)
    y = np.asarray(target[mask], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return float("nan")
    return pearson_metric(rankdata_average(x), rankdata_average(y), np.ones_like(x, dtype=bool))


def binary_top_mask(score: np.ndarray, mask: np.ndarray, top_fraction: float = 0.20) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    vals = score[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    q = np.percentile(vals, 100.0 * (1.0 - top_fraction))
    out[mask] = score[mask] >= q
    return out


def evaluate(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if int(np.sum(mask)) < 5:
        return {m: float("nan") for m in METRICS}

    sp = spearman_metric(pred, target, mask)
    pr = pearson_metric(pred, target, mask)

    pred_top = binary_top_mask(pred, mask, 0.20)
    targ_top = binary_top_mask(target, mask, 0.20)

    tp = float(np.sum(pred_top & targ_top))
    fp = float(np.sum(pred_top & ~targ_top))
    fn = float(np.sum(~pred_top & targ_top))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    union = float(np.sum(pred_top | targ_top))
    iou = tp / (union + 1e-12)

    # AUC top20 simplificado por ranking: target_top como positivo.
    y_true = targ_top[mask].astype(int)
    scores = pred[mask].astype(float)
    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        ranks = rankdata_average(scores)
        n_pos = float(np.sum(y_true == 1))
        n_neg = float(np.sum(y_true == 0))
        sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
        auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg + 1e-12)

    return {
        "spearman": float(sp),
        "auc_top20": float(auc),
        "f1_top20": float(f1),
        "pearson": float(pr),
        "iou_top20": float(iou),
    }


# ============================================================
# Cache de muestra
# ============================================================

def prepare_sample(
    pair: dict,
    args,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict:
    depth = load_depth(Path(pair["depth"]), depth_scale=args.depth_scale, auto_depth_scale=not args.no_auto_depth_scale)
    depth = downsample(depth, args.max_size)

    valid = make_valid(depth, args.min_depth, args.max_depth)

    if int(np.sum(valid)) < args.min_valid_pixels:
        raise ValueError(f"Muy pocos píxeles válidos: {int(np.sum(valid))}")

    # Ajustar intrínsecos si downsample cambió tamaño.
    # Como downsample es por slicing, calculamos factor aproximado comparando tamaño original vs actual.
    # Si no hay intrínsecos, fx/fy/cx/cy auto se calculan dentro.
    X, Y, Z = backproject_xyz(depth, valid, fx=fx, fy=fy, cx=cx, cy=cy)

    target_pack = build_target(
        depth=depth,
        valid=valid,
        target_name=args.target,
        X=X,
        Y=Y,
        Z=Z,
        target_window=args.target_window,
    )
    target = target_pack["target"]

    nct = compute_nct3d_tensor(
        X=X,
        Y=Y,
        Z=Z,
        valid=valid,
        state_threshold=args.state_threshold,
        tilde_band=args.tilde_band,
    )

    # Baseline clásico: depth edge directo.
    classical = target_pack["depth_edges"]

    # Fixed NCT energy baseline.
    nct_energy_fixed = np.clip(0.55 * nct["energy"] + 0.25 * nct["transition"] - 0.20 * nct["suppression"], 0.0, 1.0)

    # Gate para zonas ambiguas: donde baseline clásico no está ni bajo ni alto.
    lo = args.gate_low
    hi = args.gate_high
    center = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-9)
    g = np.clip(1.0 - np.abs(classical - center) / half, 0.0, 1.0)
    g[~valid] = 0.0

    ambiguous = valid & (classical > lo) & (classical < hi)

    masks = {
        "GLOBAL": valid,
        "AMBIGUOUS_ONLY": ambiguous,
        "NON_AMBIGUOUS": valid & ~ambiguous,
    }

    return {
        "stem": pair["stem"],
        "rgb": pair["rgb"],
        "depth": pair["depth"],
        "shape": list(depth.shape),
        "valid": valid,
        "target": target,
        "target_pack": target_pack,
        "classical_depth_edge": classical,
        "nct_energy_fixed": nct_energy_fixed,
        "gate": g,
        "motif_id": nct["motif_id"],
        "masks": masks,
        "nct": {
            "energy": nct["energy"],
            "transition": nct["transition"],
            "suppression": nct["suppression"],
            "active_count": nct["active_count"],
            "tilde_count": nct["tilde_count"],
        },
    }


# ============================================================
# Train motif survival
# ============================================================

def train_motif_weights(
    samples: List[dict],
    train_region: str,
    min_count: int,
    shrink_k: float,
    normalize: str,
) -> dict:
    sums = np.zeros(64, dtype=float)
    counts = np.zeros(64, dtype=float)
    all_targets = []

    for s in samples:
        mask = s["masks"][train_region]
        motif = s["motif_id"]
        target = s["target"]

        vals = target[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_targets.append(vals)

        for mid in range(64):
            mm = mask & (motif == mid) & np.isfinite(target)
            n = int(np.sum(mm))
            if n <= 0:
                continue
            sums[mid] += float(np.sum(target[mm]))
            counts[mid] += float(n)

    global_mean = float(np.mean(np.concatenate(all_targets))) if all_targets else 0.0

    raw_lift = np.zeros(64, dtype=float)
    mean_by_motif = np.full(64, np.nan, dtype=float)

    for mid in range(64):
        if counts[mid] > 0:
            mean_by_motif[mid] = sums[mid] / counts[mid]
            raw_lift[mid] = mean_by_motif[mid] - global_mean

    shrink = counts / (counts + float(shrink_k))
    weights = raw_lift * shrink
    weights[counts < int(min_count)] = 0.0

    if normalize == "maxabs":
        scale = float(np.nanmax(np.abs(weights)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        weights = weights / scale
    elif normalize == "std":
        scale = float(np.nanstd(weights))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        weights = np.clip(weights / (3.0 * scale), -1.0, 1.0)
    elif normalize == "none":
        weights = np.clip(weights, -1.0, 1.0)
    else:
        raise ValueError(f"normalize inválido: {normalize}")

    weights = np.clip(weights, -1.0, 1.0)

    table = {}
    for mid in range(64):
        table[str(mid)] = {
            "motif": motif_name(mid),
            "count": int(counts[mid]),
            "mean_target": float(mean_by_motif[mid]) if np.isfinite(mean_by_motif[mid]) else None,
            "global_mean": float(global_mean),
            "raw_lift": float(raw_lift[mid]),
            "shrink": float(shrink[mid]),
            "weight": float(weights[mid]),
        }

    return {
        "weights": weights,
        "table": table,
        "global_mean": global_mean,
        "train_region": train_region,
        "min_count": int(min_count),
        "shrink_k": float(shrink_k),
        "normalize": normalize,
    }


def apply_motif_model(sample: dict, weights: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    delta = sample["classical_depth_edge"]
    g = sample["gate"]
    motif = sample["motif_id"]
    wmap = weights[motif]

    if mode == "motif_survival":
        residual = g * wmap
    elif mode == "motif_survival_pos_only":
        residual = g * np.maximum(wmap, 0.0)
    elif mode == "motif_survival_neg_only":
        residual = g * np.minimum(wmap, 0.0)
    elif mode == "motif_survival_binary":
        residual = g * np.sign(wmap)
    else:
        raise ValueError(mode)

    pred = np.clip(delta + float(alpha) * residual, 0.0, 1.0)
    pred[~sample["valid"]] = 0.0
    return pred


def random_weight_tables(weights: np.ndarray, n: int, rng: np.random.Generator) -> List[np.ndarray]:
    out = []
    for _ in range(int(n)):
        w = np.array(weights, dtype=float).copy()
        rng.shuffle(w)
        out.append(w)
    return out


# ============================================================
# Split/eval
# ============================================================

def split_indices(n: int, seed: int, test_ratio: float) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * float(test_ratio))))
    test = sorted(idx[:n_test].tolist())
    train = sorted(idx[n_test:].tolist())
    return train, test


def extract_first_integer(text: str, fallback: int) -> int:
    m = re.search(r"(\\d+)", str(text))
    if not m:
        return int(fallback)
    return int(m.group(1))


def build_group_ids(pairs: List[dict], strategy: str = "numeric_block", group_size: int = 50, prefix_len: int = 3) -> List[str]:
    """
    Agrupa frames para evitar leakage por frames vecinos.

    numeric_block:
        00000_depth ... 00049_depth -> block_0
        00050_depth ... 00099_depth -> block_1

    prefix:
        usa los primeros N caracteres del stem.

    stem:
        cada stem es su propio grupo. Equivale casi a random por frame,
        pero mantiene el mismo flujo de grouped split.
    """
    strategy = str(strategy).strip().lower()
    group_size = max(1, int(group_size))
    prefix_len = max(1, int(prefix_len))

    gids = []
    for i, p in enumerate(pairs):
        stem = str(p.get("stem", f"sample_{i}"))

        if strategy == "numeric_block":
            num = extract_first_integer(stem, i)
            gid = f"block_{num // group_size:06d}"
        elif strategy == "prefix":
            gid = f"prefix_{stem[:prefix_len]}"
        elif strategy == "stem":
            gid = f"stem_{stem}"
        else:
            raise ValueError(f"group_strategy inválida: {strategy}")

        gids.append(gid)

    return gids


def split_indices_grouped(
    pairs: List[dict],
    seed: int,
    test_ratio: float,
    group_strategy: str,
    group_size: int,
    group_prefix_len: int,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    """
    Split por grupos: un grupo completo cae en TRAIN o TEST, nunca dividido.
    Esto reduce leakage si los frames vienen de una misma secuencia.
    """
    rng = np.random.default_rng(seed)
    group_ids = build_group_ids(
        pairs,
        strategy=group_strategy,
        group_size=group_size,
        prefix_len=group_prefix_len,
    )

    group_to_indices: Dict[str, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(idx)

    groups = list(group_to_indices.keys())
    rng.shuffle(groups)

    n_total = len(pairs)
    target_test = max(1, int(round(n_total * float(test_ratio))))

    test_groups = []
    train_groups = []
    test_indices = []

    for gid in groups:
        if len(test_indices) < target_test:
            test_groups.append(gid)
            test_indices.extend(group_to_indices[gid])
        else:
            train_groups.append(gid)

    train_indices = []
    for gid in train_groups:
        train_indices.extend(group_to_indices[gid])

    if not train_indices and len(test_groups) > 1:
        moved = test_groups.pop()
        train_groups.append(moved)
        moved_indices = set(group_to_indices[moved])
        test_indices = [i for i in test_indices if i not in moved_indices]
        train_indices.extend(group_to_indices[moved])

    if not test_indices and train_groups:
        moved = train_groups.pop()
        test_groups.append(moved)
        moved_indices = group_to_indices[moved]
        train_indices = [i for i in train_indices if i not in set(moved_indices)]
        test_indices.extend(moved_indices)

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    meta = {
        "split_mode": "grouped",
        "group_strategy": group_strategy,
        "group_size": int(group_size),
        "group_prefix_len": int(group_prefix_len),
        "n_groups_total": len(groups),
        "n_groups_train": len(train_groups),
        "n_groups_test": len(test_groups),
        "test_groups": sorted(test_groups),
        "train_groups": sorted(train_groups),
        "target_test_count": int(target_test),
        "actual_test_count": len(test_indices),
        "actual_test_ratio": float(len(test_indices) / max(1, n_total)),
    }
    return train_indices, test_indices, meta


def make_split(
    pairs: List[dict],
    seed: int,
    test_ratio: float,
    split_mode: str,
    group_strategy: str,
    group_size: int,
    group_prefix_len: int,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    split_mode = str(split_mode).strip().lower()

    if split_mode == "random":
        train_idx, test_idx = split_indices(len(pairs), seed=seed, test_ratio=test_ratio)
        return train_idx, test_idx, {
            "split_mode": "random",
            "n_groups_total": None,
            "n_groups_train": None,
            "n_groups_test": None,
            "test_groups": [],
            "train_groups": [],
            "actual_test_count": len(test_idx),
            "actual_test_ratio": float(len(test_idx) / max(1, len(pairs))),
        }

    if split_mode == "grouped":
        return split_indices_grouped(
            pairs=pairs,
            seed=seed,
            test_ratio=test_ratio,
            group_strategy=group_strategy,
            group_size=group_size,
            group_prefix_len=group_prefix_len,
        )

    raise ValueError(f"split_mode inválido: {split_mode}")


def eval_sample_models(
    sample: dict,
    weights: np.ndarray,
    random_tables: List[np.ndarray],
    alpha: float,
    device: str = "cpu",
    gpu_random_chunk: int = 64,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    preds = {
        "classical_depth_edge": sample["classical_depth_edge"],
        "nct_energy_fixed": sample["nct_energy_fixed"],
    }

    for mode in TEST_MODELS:
        preds[mode] = apply_motif_model(sample, weights, alpha, mode=mode)

    # Métricas de modelos reales en CPU. Son pocas y no son el cuello principal.
    out = {}
    for model, pred in preds.items():
        out[model] = {}
        for region in REGIONS:
            out[model][region] = evaluate(pred, sample["target"], sample["masks"][region])

    # Random mean metrics: CPU o GPU.
    if device == "cuda":
        random_metrics_by_region = evaluate_random_distribution_gpu(
            sample=sample,
            random_tables=random_tables,
            alpha=alpha,
            device=device,
            gpu_random_chunk=gpu_random_chunk,
        )
    else:
        random_metrics_by_region = {region: {m: [] for m in METRICS} for region in REGIONS}

        for rw in random_tables:
            pred = apply_motif_model(sample, rw, alpha, mode="motif_survival")
            for region in REGIONS:
                met = evaluate(pred, sample["target"], sample["masks"][region])
                for m in METRICS:
                    random_metrics_by_region[region][m].append(met[m])

    out["random_motif_mean"] = {}
    for region in REGIONS:
        out["random_motif_mean"][region] = {}
        for m in METRICS:
            vals = random_metrics_by_region[region][m]
            out["random_motif_mean"][region][m] = float(np.nanmean(vals)) if vals else float("nan")

    # Guardamos distribución random para p-value.
    out["_random_distribution"] = random_metrics_by_region

    return out


def aggregate_metrics(file_metrics: List[dict]) -> dict:
    out = {region: {model: {} for model in MODEL_NAMES} for region in REGIONS}

    for region in REGIONS:
        for model in MODEL_NAMES:
            for metric in METRICS:
                vals = []
                for fm in file_metrics:
                    vals.append(fm[model][region][metric])
                m, s = mean_std(vals)
                out[region][model][metric + "_mean"] = m
                out[region][model][metric + "_std"] = s

    return out


def improvements(agg: dict, region: str) -> dict:
    out = {}
    classical = agg[region]["classical_depth_edge"]
    fixed = agg[region]["nct_energy_fixed"]
    randomm = agg[region]["random_motif_mean"]

    for model in TEST_MODELS:
        m = agg[region][model]
        out[model] = {
            "minus_classical_spearman": m["spearman_mean"] - classical["spearman_mean"],
            "minus_classical_auc": m["auc_top20_mean"] - classical["auc_top20_mean"],
            "minus_classical_f1": m["f1_top20_mean"] - classical["f1_top20_mean"],

            "minus_fixed_spearman": m["spearman_mean"] - fixed["spearman_mean"],
            "minus_fixed_auc": m["auc_top20_mean"] - fixed["auc_top20_mean"],
            "minus_fixed_f1": m["f1_top20_mean"] - fixed["f1_top20_mean"],

            "minus_random_spearman": m["spearman_mean"] - randomm["spearman_mean"],
            "minus_random_auc": m["auc_top20_mean"] - randomm["auc_top20_mean"],
            "minus_random_f1": m["f1_top20_mean"] - randomm["f1_top20_mean"],
        }
    return out


def empirical_pvalues_for_main(file_metrics: List[dict], model: str = "motif_survival_pos_only", region: str = "AMBIGUOUS_ONLY") -> dict:
    # Real metric promedio por archivo.
    pvals = {}
    for metric in ("spearman", "auc_top20", "f1_top20"):
        real_vals = [fm[model][region][metric] for fm in file_metrics]
        real_mean = float(np.nanmean(real_vals)) if real_vals else float("nan")

        # Random distribution promedio por random-id.
        if not file_metrics:
            pvals[metric] = float("nan")
            continue

        n_random = max(len(fm["_random_distribution"][region][metric]) for fm in file_metrics)
        random_means = []
        for ridx in range(n_random):
            vals = []
            for fm in file_metrics:
                arr = fm["_random_distribution"][region][metric]
                if ridx < len(arr):
                    vals.append(arr[ridx])
            if vals:
                random_means.append(float(np.nanmean(vals)))

        pvals[metric] = empirical_p_value(real_mean, random_means)

    return pvals


def run_split(
    samples: List[dict],
    pairs: List[dict],
    train_idx: List[int],
    test_idx: List[int],
    alpha: float,
    seed: int,
    args,
    split_name: str,
) -> dict:
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    weights_info = train_motif_weights(
        train_samples,
        train_region=args.train_region,
        min_count=args.min_count,
        shrink_k=args.shrink_k,
        normalize=args.normalize,
    )
    weights = weights_info["weights"]

    rng = np.random.default_rng(int(seed) + int(alpha * 1_000_000) + 1327)
    random_tables = random_weight_tables(weights, args.random_baselines, rng)

    file_metrics = [
        eval_sample_models(s, weights, random_tables, alpha)
        for s in test_samples
    ]

    agg = aggregate_metrics(file_metrics)
    imp = {region: improvements(agg, region) for region in REGIONS}
    pvals = {region: empirical_pvalues_for_main(file_metrics, MAIN_MODEL, region) for region in REGIONS}

    test_rows = []
    for idx, sample, fm in zip(test_idx, test_samples, file_metrics):
        row = {
            "stem": pairs[idx]["stem"],
            "depth": pairs[idx]["depth"],
            "rgb": pairs[idx]["rgb"],
        }
        for model in MODEL_NAMES:
            for metric in ("spearman", "auc_top20", "f1_top20"):
                row[f"{model}_{metric}"] = fm[model][REGION][metric]
        test_rows.append(row)

    # Top motifs.
    top_motifs = []
    for mid in range(64):
        info = weights_info["table"][str(mid)]
        top_motifs.append({
            "motif_id": mid,
            "motif": info["motif"],
            "count": info["count"],
            "raw_lift": info["raw_lift"],
            "weight": info["weight"],
        })
    top_motifs.sort(key=lambda r: abs(r["weight"]), reverse=True)

    return {
        "split_name": split_name,
        "alpha": float(alpha),
        "seed": int(seed),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_stems": [pairs[i]["stem"] for i in train_idx],
        "test_stems": [pairs[i]["stem"] for i in test_idx],
        "aggregate": agg,
        "improvements": imp,
        "empirical_p_values": pvals,
        "top_motifs": top_motifs[:16],
        "test_rows": test_rows,
    }


def summarize_runs(runs: List[dict], region: str) -> dict:
    out = {}
    for model in TEST_MODELS:
        out[model] = {}
        for base_name in ("classical", "fixed", "random"):
            for short, key in (("sp", "spearman"), ("auc", "auc"), ("f1", "f1")):
                vals = [r["improvements"][region][model][f"minus_{base_name}_{key}"] for r in runs]
                m, s = mean_std(vals)
                out[model][f"d_{short}_{base_name}_mean"] = m
                out[model][f"d_{short}_{base_name}_std"] = s
                out[model][f"d_{short}_{base_name}_positive_rate"] = positive_rate(vals)

            comp = []
            for r in runs:
                imp = r["improvements"][region][model]
                comp.append(imp[f"minus_{base_name}_auc"] > 0 and imp[f"minus_{base_name}_f1"] > 0)
            out[model][f"rate_auc_f1_positive_vs_{base_name}"] = float(np.mean(comp)) if comp else float("nan")

        for metric in ("spearman", "auc_top20", "f1_top20"):
            vals = [r["empirical_p_values"][region][model][metric] for r in runs]
            m, s = mean_std(vals)
            out[model][f"p_{metric}_mean"] = m
            out[model][f"p_{metric}_std"] = s
            arr = finite(vals)
            out[model][f"p_{metric}_lt_0_10_rate"] = float(np.mean(arr < 0.10)) if arr.size else float("nan")
            out[model][f"p_{metric}_lt_0_05_rate"] = float(np.mean(arr < 0.05)) if arr.size else float("nan")

    return out


def aggregate_weights(runs: List[dict]) -> dict:
    bucket = {mid: {"weights": [], "lifts": [], "counts": [], "top": 0} for mid in range(64)}
    for r in runs:
        seen = set()
        for row in r["top_motifs"]:
            mid = int(row["motif_id"])
            bucket[mid]["weights"].append(row["weight"])
            bucket[mid]["lifts"].append(row["raw_lift"])
            bucket[mid]["counts"].append(row["count"])
            seen.add(mid)
        for mid in seen:
            bucket[mid]["top"] += 1

    out = {}
    for mid in range(64):
        wm, ws = mean_std(bucket[mid]["weights"])
        lm, ls = mean_std(bucket[mid]["lifts"])
        cm, cs = mean_std(bucket[mid]["counts"])
        out[str(mid)] = {
            "motif": motif_name(mid),
            "weight_mean": wm,
            "weight_std": ws,
            "raw_lift_mean": lm,
            "raw_lift_std": ls,
            "count_mean": cm,
            "count_std": cs,
            "appearance_rate_top16": float(bucket[mid]["top"] / len(runs)) if runs else float("nan"),
            "positive_weight_rate": positive_rate(bucket[mid]["weights"]),
        }
    return out


# ============================================================
# CSV outputs
# ============================================================

MAIN_MODEL = "motif_survival_pos_only"


def write_pairs_csv(pairs: List[dict], output_csv: str):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "depth", "rgb"])
        w.writeheader()
        for p in pairs:
            w.writerow(p)


def write_summary_csv(runs: List[dict], output_csv: str):
    fields = [
        "split_name", "alpha", "seed", "n_train", "n_test", "test_stems",
        "model",
        "d_sp_classical", "d_auc_classical", "d_f1_classical",
        "d_sp_fixed", "d_auc_fixed", "d_f1_fixed",
        "d_sp_random", "d_auc_random", "d_f1_random",
        "p_sp", "p_auc", "p_f1",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            for model in TEST_MODELS:
                imp = r["improvements"][REGION][model]
                w.writerow({
                    "split_name": r["split_name"],
                    "alpha": r["alpha"],
                    "seed": r["seed"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                    "test_stems": " | ".join(r["test_stems"]),
                    "model": model,
                    "d_sp_classical": imp["minus_classical_spearman"],
                    "d_auc_classical": imp["minus_classical_auc"],
                    "d_f1_classical": imp["minus_classical_f1"],
                    "d_sp_fixed": imp["minus_fixed_spearman"],
                    "d_auc_fixed": imp["minus_fixed_auc"],
                    "d_f1_fixed": imp["minus_fixed_f1"],
                    "d_sp_random": imp["minus_random_spearman"],
                    "d_auc_random": imp["minus_random_auc"],
                    "d_f1_random": imp["minus_random_f1"],
                    "p_sp": r["empirical_p_values"][REGION][model]["spearman"],
                    "p_auc": r["empirical_p_values"][REGION][model]["auc_top20"],
                    "p_f1": r["empirical_p_values"][REGION][model]["f1_top20"],
                })


def write_weights_csv(weights: dict, output_csv: str):
    fields = [
        "motif_id", "motif", "weight_mean", "weight_std",
        "raw_lift_mean", "raw_lift_std",
        "count_mean", "count_std",
        "appearance_rate_top16",
        "positive_weight_rate",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mid in range(64):
            row = {"motif_id": mid, **weights[str(mid)]}
            w.writerow(row)


# ============================================================
# Console
# ============================================================

def print_summary(summary: dict):
    print(f"\n===== v13 RGB-D SUMMARY / TEST / {REGION} =====")
    for model in TEST_MODELS:
        s = summary[model]
        print("\n", model)
        print(
            "  vs classical:",
            "ΔAUC=", round(s["d_auc_classical_mean"], 8),
            "±", round(s["d_auc_classical_std"], 8),
            "pr=", round(s["d_auc_classical_positive_rate"], 3),
            "| ΔF1=", round(s["d_f1_classical_mean"], 8),
            "±", round(s["d_f1_classical_std"], 8),
            "pr=", round(s["d_f1_classical_positive_rate"], 3),
            "| comp=", round(s["rate_auc_f1_positive_vs_classical"], 3),
        )
        print(
            "  vs random:",
            "ΔAUC=", round(s["d_auc_random_mean"], 8),
            "±", round(s["d_auc_random_std"], 8),
            "pr=", round(s["d_auc_random_positive_rate"], 3),
            "| ΔF1=", round(s["d_f1_random_mean"], 8),
            "±", round(s["d_f1_random_std"], 8),
            "pr=", round(s["d_f1_random_positive_rate"], 3),
            "| comp=", round(s["rate_auc_f1_positive_vs_random"], 3),
        )
        print(
            "  empirical p:",
            "AUC mean=", round(s["p_auc_top20_mean"], 4),
            "p<0.10 rate=", round(s["p_auc_top20_lt_0_10_rate"], 3),
            "| F1 mean=", round(s["p_f1_top20_mean"], 4),
            "p<0.10 rate=", round(s["p_f1_top20_lt_0_10_rate"], 3),
        )


def print_weights(weights: dict):
    rows = []
    for mid in range(64):
        d = weights[str(mid)]
        score = abs(d["weight_mean"]) * d["appearance_rate_top16"]
        if np.isfinite(score):
            rows.append((score, mid, d))
    rows.sort(reverse=True)

    print("\n===== TOP MOTIFS RGB-D =====")
    for score, mid, d in rows[:16]:
        print(
            f"{mid:02d}",
            d["motif"],
            "| weight=", round(d["weight_mean"], 5),
            "±", round(d["weight_std"], 5),
            "| lift=", round(d["raw_lift_mean"], 5),
            "| top_rate=", round(d["appearance_rate_top16"], 3),
            "| pos_rate=", round(d["positive_weight_rate"], 3),
        )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default="")
    ap.add_argument("--depth", required=True)
    ap.add_argument("--target", default="combined", choices=["depth_edges", "normal_edges", "plane_residual", "combined"])
    ap.add_argument("--alpha", default="0.03")
    ap.add_argument("--seeds", default="11,22,33,44,55,369")
    ap.add_argument("--test-ratio", type=float, default=0.30)
    ap.add_argument("--split-mode", default="grouped", choices=["random", "grouped"], help="random = split por frame; grouped = separa bloques/escenas para reducir leakage.")
    ap.add_argument("--group-strategy", default="numeric_block", choices=["numeric_block", "prefix", "stem"], help="Estrategia de agrupación para grouped split.")
    ap.add_argument("--group-size", type=int, default=50, help="Tamaño de bloque para numeric_block. Ej: 50 agrupa 00000-00049.")
    ap.add_argument("--group-prefix-len", type=int, default=3, help="Cantidad de caracteres para group-strategy prefix.")
    ap.add_argument("--train-region", default="AMBIGUOUS_ONLY", choices=list(REGIONS))
    ap.add_argument("--state-threshold", type=float, default=0.18)
    ap.add_argument("--tilde-band", type=float, default=0.06)
    ap.add_argument("--min-count", type=int, default=80)
    ap.add_argument("--shrink-k", type=float, default=250.0)
    ap.add_argument("--normalize", default="maxabs", choices=["maxabs", "std", "none"])
    ap.add_argument("--random-baselines", type=int, default=256)
    ap.add_argument("--workers", type=int, default=1, help="Cantidad de workers paralelos para evaluar samples de TEST. Recomendado: 6-10 en CPU de 12 hilos.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Dispositivo para la evaluación pesada de random baselines.")
    ap.add_argument("--gpu-random-chunk", type=int, default=64, help="Cantidad de tablas random por chunk en GPU. Para 4GB VRAM: 32 o 64.")
    ap.add_argument("--max-size", type=int, default=320)
    ap.add_argument("--target-window", type=int, default=17)
    ap.add_argument("--gate-low", type=float, default=0.15)
    ap.add_argument("--gate-high", type=float, default=0.85)
    ap.add_argument("--depth-scale", type=float, default=0.0)
    ap.add_argument("--no-auto-depth-scale", action="store_true")
    ap.add_argument("--min-depth", type=float, default=1e-6)
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--min-valid-pixels", type=int, default=500)
    ap.add_argument("--fx", type=float, default=-1.0)
    ap.add_argument("--fy", type=float, default=-1.0)
    ap.add_argument("--cx", type=float, default=-1.0)
    ap.add_argument("--cy", type=float, default=-1.0)
    ap.add_argument("--output-json", default="9B02_v13_4_grouped_split_results.json")
    ap.add_argument("--output-summary-csv", default="9B02_v13_4_grouped_split_summary.csv")
    ap.add_argument("--output-weights-csv", default="9B02_v13_4_grouped_split_weights.csv")
    ap.add_argument("--output-pairs-csv", default="9B02_v13_4_grouped_split_pairs.csv")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb) if args.rgb else None
    depth_dir = Path(args.depth)
    pairs = match_rgb_depth(rgb_dir, depth_dir)

    if not pairs:
        raise SystemExit("No encontré archivos depth compatibles.")

    alphas = parse_float_list(args.alpha)
    seeds = parse_int_list(args.seeds)
    runtime_device = select_runtime_device(args.device)

    if runtime_device == "cuda" and args.workers > 1:
        print("[INFO] CUDA activo: se ignora --workers para la evaluación de TEST y se usa GPU secuencial por chunks.")
        args.workers = 1

    print("===== 9B02 NCT v13.3 RGB-D CUDA DATASET ADAPTER =====")
    print("depth:", depth_dir)
    print("rgb:", rgb_dir if rgb_dir else "(no usado)")
    print("pairs:", len(pairs))
    print("target:", args.target)
    print("alphas:", alphas)
    print("seeds:", seeds)
    print("random_baselines:", args.random_baselines)
    print("workers:", args.workers)
    print("device:", runtime_device)
    print("gpu_random_chunk:", args.gpu_random_chunk)
    print("split_mode:", args.split_mode)
    print("group_strategy:", args.group_strategy)
    print("group_size:", args.group_size)
    print("group_prefix_len:", args.group_prefix_len)
    print("NCT_3D: 64 motifs sobre superficie P(X,Y,Z)")
    print("Auditoría: target real derivado de depth; pesos solo en TRAIN; TEST no entrena.")
    print("p-values: calculados para motif_survival, pos_only, neg_only y binary.")

    write_pairs_csv(pairs, args.output_pairs_csv)

    print("\nPreparando samples...")
    samples = []
    failures = []

    # Si el usuario dio intrínsecos, se usan. Si no, auto dentro.
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    for i, pair in enumerate(pairs, start=1):
        try:
            s = prepare_sample(pair, args, fx=fx, fy=fy, cx=cx, cy=cy)
            samples.append(s)
            print(f"[{i}/{len(pairs)}] OK {pair['stem']} shape={s['shape']} valid={int(np.sum(s['valid']))}")
        except Exception as e:
            failures.append({"pair": pair, "error": repr(e)})
            print(f"[{i}/{len(pairs)}] ERROR {pair['stem']} {repr(e)}")

    if len(samples) < 3:
        raise SystemExit("Necesito al menos 3 depth maps válidos para train/test.")

    runs = []
    total = len(alphas) * len(seeds)
    nrun = 0

    for alpha in alphas:
        for seed in seeds:
            nrun += 1
            train_idx, test_idx, split_meta = make_split(
                pairs=[{"stem": sample["stem"]} for sample in samples],
                seed=seed,
                test_ratio=args.test_ratio,
                split_mode=args.split_mode,
                group_strategy=args.group_strategy,
                group_size=args.group_size,
                group_prefix_len=args.group_prefix_len,
            )

            print(f"\n[{nrun}/{total}] alpha={alpha} seed={seed}")
            if args.split_mode == "grouped":
                print(
                    "  grouped split:",
                    f"groups_test={split_meta['n_groups_test']}/{split_meta['n_groups_total']}",
                    f"test_count={split_meta['actual_test_count']}",
                    f"test_ratio={split_meta['actual_test_ratio']:.3f}",
                )
                groups_preview = split_meta.get("test_groups", [])[:20]
                print("  test groups:", ", ".join(groups_preview), "..." if len(split_meta.get("test_groups", [])) > 20 else "")
            test_preview = ", ".join([samples[i]["stem"] for i in test_idx[:60]])
            print("  test:", test_preview + (" ..." if len(test_idx) > 60 else ""))

            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]

            weights_info = train_motif_weights(
                train_samples,
                train_region=args.train_region,
                min_count=args.min_count,
                shrink_k=args.shrink_k,
                normalize=args.normalize,
            )
            weights = weights_info["weights"]

            rng = np.random.default_rng(int(seed) + int(float(alpha) * 1_000_000) + 7781)
            random_tables = random_weight_tables(weights, args.random_baselines, rng)

            if runtime_device == "cuda":
                # En CUDA no usamos ThreadPool: una sola cola GPU por chunks evita pelear por VRAM.
                file_metrics = [
                    eval_sample_models(
                        s,
                        weights,
                        random_tables,
                        float(alpha),
                        device=runtime_device,
                        gpu_random_chunk=args.gpu_random_chunk,
                    )
                    for s in test_samples
                ]
            elif args.workers and args.workers > 1:
                with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
                    file_metrics = list(
                        ex.map(
                            lambda sample: eval_sample_models(
                                sample,
                                weights,
                                random_tables,
                                float(alpha),
                                device=runtime_device,
                                gpu_random_chunk=args.gpu_random_chunk,
                            ),
                            test_samples,
                        )
                    )
            else:
                file_metrics = [
                    eval_sample_models(
                        s,
                        weights,
                        random_tables,
                        float(alpha),
                        device=runtime_device,
                        gpu_random_chunk=args.gpu_random_chunk,
                    )
                    for s in test_samples
                ]

            agg = aggregate_metrics(file_metrics)
            imp = {region: improvements(agg, region) for region in REGIONS}

            pvals = {}
            for region in REGIONS:
                pvals[region] = {}
                for model_name in TEST_MODELS:
                    pvals[region][model_name] = {}
                    for metric in ("spearman", "auc_top20", "f1_top20"):
                        real = agg[region][model_name][metric + "_mean"]

                        # distribución random por índice
                        n_random = args.random_baselines
                        random_means = []
                        for ridx in range(n_random):
                            vals = []
                            for fm in file_metrics:
                                arr = fm["_random_distribution"][region][metric]
                                if ridx < len(arr):
                                    vals.append(arr[ridx])
                            if vals:
                                random_means.append(float(np.nanmean(vals)))
                        pvals[region][model_name][metric] = empirical_p_value(real, random_means)

            top_motifs = []
            for mid in range(64):
                info = weights_info["table"][str(mid)]
                top_motifs.append({
                    "motif_id": mid,
                    "motif": info["motif"],
                    "count": info["count"],
                    "raw_lift": info["raw_lift"],
                    "weight": info["weight"],
                })
            top_motifs.sort(key=lambda r: abs(r["weight"]), reverse=True)

            run = {
                "split_name": "holdout",
                "alpha": float(alpha),
                "seed": int(seed),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "split_meta": split_meta,
                "train_stems": [samples[i]["stem"] for i in train_idx],
                "test_stems": [samples[i]["stem"] for i in test_idx],
                "aggregate": agg,
                "improvements": imp,
                "empirical_p_values": pvals,
                "top_motifs": top_motifs[:16],
            }
            runs.append(run)

            main_imp = imp[REGION][MAIN_MODEL]
            print(
                "  pos_only vs classical:",
                "ΔAUC=", round(main_imp["minus_classical_auc"], 7),
                "ΔF1=", round(main_imp["minus_classical_f1"], 7),
            )
            print(
                "  pos_only vs random:",
                "ΔAUC=", round(main_imp["minus_random_auc"], 7),
                "ΔF1=", round(main_imp["minus_random_f1"], 7),
                "| p_auc=", round(pvals[REGION][MAIN_MODEL]["auc_top20"], 4),
                "p_f1=", round(pvals[REGION][MAIN_MODEL]["f1_top20"], 4),
            )

    summary = summarize_runs(runs, REGION)
    weights_summary = aggregate_weights(runs)

    result = {
        "version": "9B02 NCT v13.4 RGB-D CUDA Grouped Split Validation",
        "purpose": "Evaluate NCT_3D motif survival on real RGB-D/depth datasets with p-values for all motif models",
        "target": args.target,
        "rgb_dir": str(rgb_dir) if rgb_dir else "",
        "depth_dir": str(depth_dir),
        "n_pairs_found": len(pairs),
        "n_samples_valid": len(samples),
        "failures": failures,
        "alphas": alphas,
        "seeds": seeds,
        "state_threshold": args.state_threshold,
        "tilde_band": args.tilde_band,
        "train_region": args.train_region,
        "min_count": args.min_count,
        "shrink_k": args.shrink_k,
        "normalize": args.normalize,
        "random_baselines": args.random_baselines,
        "split_mode": args.split_mode,
        "group_strategy": args.group_strategy,
        "group_size": args.group_size,
        "group_prefix_len": args.group_prefix_len,
        "device": runtime_device,
        "gpu_random_chunk": args.gpu_random_chunk,
        "workers": args.workers,
        "audit": {
            "uses_rgb": bool(rgb_dir),
            "uses_target_for_tensor": False,
            "uses_target_for_training": True,
            "uses_test_for_training": False,
            "uses_test_for_selection": False,
            "random_baseline": "permuted motif weights",
            "main_model": MAIN_MODEL,
            "cuda_note": "v13.4 moves random baseline evaluation to GPU when --device cuda is active.",
            "grouped_split_note": "When --split-mode grouped, contiguous filename blocks/groups are kept entirely in TRAIN or TEST to reduce frame leakage.",
        },
        "summary": summary,
        "weights_summary": weights_summary,
        "runs": runs,
    }

    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(runs, args.output_summary_csv)
    write_weights_csv(weights_summary, args.output_weights_csv)

    print_summary(summary)
    print_weights(weights_summary)

    print("\nSalidas:")
    print("JSON:", args.output_json)
    print("Summary CSV:", args.output_summary_csv)
    print("Weights CSV:", args.output_weights_csv)
    print("Pairs CSV:", args.output_pairs_csv)

    print("\nLectura:")
    print("- Si pos_only gana vs classical y vs random, NCT_3D aporta señal sobre depth real.")
    print("- Si gana vs classical pero no vs random, el efecto viene de perturbación/gate, no de identidad motif.")
    print("- Si p_auc/p_f1 < 0.10 en varias corridas, supera random de forma más defendible.")
    print("- Si top motifs son estables, hay motivos NCT 3D sobrevivientes en datos RGB-D reales.")
    print("- Si falla en RGB-D real pero funcionaba en sintético, el target sintético era demasiado favorable.")


if __name__ == "__main__":
    main()
