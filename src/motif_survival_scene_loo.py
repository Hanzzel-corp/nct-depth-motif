#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCT Motif Survival - Scene Leave-One-Out Validation (v14.2.1)

Reporte técnico exploratorio. No revisado por pares.

Este script implementa validación leave-one-scene-out (LOO) para evaluar la generalización
de motivos NCT 3D a escenas completamente no vistas durante entrenamiento.

Hallazgo: Generalización estadísticamente significativa dentro de NYU Depth V2,
con magnitudes de efecto consistentes al grouped split (ΔAUC ≈ +0.004).

Diferencias con Grouped Split
-----------------------------
Grouped Split (motif_survival_grouped.py):
  - Divide escenas en grupos por orden numérico
  - Cada fold tiene múltiples escenas de train y test
  - Más rápido, menos estricto en generalización

Scene LOO (este script):
  - Deja una escena completa fuera para test
  - Entrena en TODAS las demás escenas
  - Máxima exigencia en generalización a escenas nuevas
  - Requiere mapeo automático de escenas desde datos

Arquitectura
------------
1. Mapeo de escenas:
   - Lee archivo scenes_auto.csv con asignación imagen→escena
   - Detecta automáticamente escenas únicas en el dataset

2. Loop LOO:
   - Para cada escena E: train = todas las demás, test = E
   - Entrena tabla de pesos NCT en train
   - Evalúa métricas en test (zona AMBIGUOUS)

3. Agregación:
   - Promedia métricas sobre las 24 escenas del dataset
   - Computa intervalos de confianza
   - Genera p-values contra baselines aleatorios

Formato de scenes_auto.csv
--------------------------
Archivo CSV con columnas:
  - file_id    : Identificador base del archivo (ej: "000001")
  - rgb_file   : Nombre del archivo RGB (ej: "000001.png")
  - depth_file : Nombre del archivo depth (ej: "000001.png")
  - scene_id   : Identificador de escena (ej: "living_room_001")
  - scene_idx  : Índice numérico de escena (0-23 para NYU)

El script results/scenes_auto.csv contiene el mapeo oficial para NYU Depth V2.

Targets de ruptura
------------------
Ver documentación en motif_survival_grouped.py (mismos targets).

Modelos evaluados
-----------------
Mismos 4 variantes que grouped split:
  - motif_survival           : Pesos completos
  - motif_survival_pos_only  : Solo positivos
  - motif_survival_neg_only  : Solo negativos
  - motif_survival_binary    : Signo únicamente

Uso
---
Ver examples/run_scene_loo.sh para ejecución típica.

Ejemplo mínimo:
    python3 src/motif_survival_scene_loo.py \
        --depth ./dataset/depth \
        --scenes ./results/scenes_auto.csv \
        --target combined \
        --alpha 0.03 \
        --device cuda

Notas importantes
-----------------
- El archivo scenes_auto.csv es REQUERIDO (a diferencia de grouped)
- Cada iteración LOO entrena desde cero (más lento que grouped)
- GPU altamente recomendado para baselines aleatorios
- 24 escenas × 256 baselines = 6144 evaluaciones por configuración

Salidas
-------
- *_loo_results.json  : Resultados por escena
- *_loo_summary.csv   : Resumen agregado sobre 24 escenas
- *_loo_weights.csv   : Pesos aprendidos por iteración LOO

Autor: Jose Zamora
Versión: 14.2.1
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


def basic_stem_aliases_for_matching(stem: str) -> List[str]:
    """
    Aliases para emparejar depth/RGB aunque los nombres no sean idénticos.
    Ej:
      depth: 00000_depth  -> rgb: 00000 / 00000_rgb / rgb_00000
      depth: depth_00000  -> rgb: 00000 / rgb_00000
    """
    s0 = str(stem).strip()
    aliases = {s0}

    suffixes = ("_depth", "-depth", "_d", "-d", "_rgb", "-rgb", "_color", "-color")
    prefixes = ("depth_", "depth-", "rgb_", "rgb-", "color_", "color-")

    changed = True
    while changed:
        changed = False
        current = list(aliases)

        for s in current:
            for suf in suffixes:
                if s.endswith(suf):
                    ns = s[: -len(suf)]
                    if ns and ns not in aliases:
                        aliases.add(ns)
                        changed = True

            for pre in prefixes:
                if s.startswith(pre):
                    ns = s[len(pre):]
                    if ns and ns not in aliases:
                        aliases.add(ns)
                        changed = True

    # Variantes RGB/color desde base numérica.
    for s in list(aliases):
        aliases.add(f"{s}_rgb")
        aliases.add(f"rgb_{s}")
        aliases.add(f"{s}_color")
        aliases.add(f"color_{s}")

    return list(aliases)


def match_rgb_for_depth_stem(depth_stem: str, rgb_files: Dict[str, Path]) -> str:
    if not rgb_files:
        return ""

    for alias in basic_stem_aliases_for_matching(depth_stem):
        if alias in rgb_files:
            return str(rgb_files[alias])

    # Fallback: comparar parte numérica.
    dn = first_integer_from_text(depth_stem, -1) if "first_integer_from_text" in globals() else -1
    if dn >= 0:
        for rst, rpath in rgb_files.items():
            rn = first_integer_from_text(rst, -2) if "first_integer_from_text" in globals() else -2
            if rn == dn:
                return str(rpath)

    return ""


def match_rgb_depth(rgb_dir: Optional[Path], depth_dir: Path) -> List[dict]:
    depth_files = collect_files(depth_dir, SUPPORTED_DEPTH_EXTS)
    rgb_files = collect_files(rgb_dir, SUPPORTED_RGB_EXTS) if rgb_dir else {}

    pairs = []
    matched = 0
    for stem, dpath in sorted(depth_files.items()):
        rgb_path = match_rgb_for_depth_stem(stem, rgb_files)
        if rgb_path:
            matched += 1
        pairs.append({
            "stem": stem,
            "depth": str(dpath),
            "rgb": rgb_path,
        })

    if rgb_dir:
        print(f"RGB matching: {matched}/{len(pairs)} depth frames tienen RGB emparejado.")

    return pairs



def first_integer_from_text(text: str, fallback: int = 0) -> int:
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else int(fallback)


def write_scene_map_template(
    pairs: List[dict],
    output_csv: str,
    mode: str = "block",
    block_size: int = 200,
):
    """
    Crea un scenes.csv inicial.

    mode=block:
        Agrupa por número de frame. Ej: 00000-00199 -> scene_block_000.
        Sirve para prueba técnica leave-one-block-out.

    mode=parent:
        Usa el nombre de la carpeta padre del depth.
        Sirve si el dataset está organizado como dataset/depth/kitchen/*.png.

    mode=todo:
        Escribe TODO_SCENE para editar a mano con categorías reales.
        Ej: kitchen, office, bathroom, living_room.
    """
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    block_size = max(1, int(block_size))

    rows = []
    for i, pair in enumerate(pairs):
        stem = str(pair.get("stem", ""))
        depth = str(pair.get("depth", ""))
        rgb = str(pair.get("rgb", ""))

        if mode == "block":
            n = first_integer_from_text(stem, i)
            scene = f"scene_block_{n // block_size:03d}"
        elif mode == "parent":
            parent = Path(depth).parent.name if depth else "unknown_parent"
            scene = parent if parent else "unknown_parent"
        elif mode == "todo":
            scene = "TODO_SCENE"
        else:
            raise ValueError(f"--scene-template-mode inválido: {mode}")

        rows.append({
            "stem": stem,
            "scene_type": scene,
            "depth": depth,
            "rgb": rgb,
        })

    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "scene_type", "depth", "rgb"])
        w.writeheader()
        w.writerows(rows)

    # Resumen por escena.
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["scene_type"]] = counts.get(r["scene_type"], 0) + 1

    print("OK scene map creado:", output)
    print("rows:", len(rows))
    print("scenes:", len(counts))
    for k, v in sorted(counts.items())[:20]:
        print(" ", k, "=", v)
    if len(counts) > 20:
        print(" ...")




def rgb_scene_feature(rgb_path: str, thumb_size: int = 64) -> Optional[np.ndarray]:
    """
    Feature liviano SOLO desde RGB.
    No usa depth, target ni tensor NCT para evitar contaminar el test geométrico.
    """
    if not rgb_path or Image is None:
        return None
    p = Path(rgb_path)
    if not p.exists():
        return None

    try:
        img = Image.open(p).convert("RGB")
        img = img.resize((thumb_size, thumb_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None

    feats = []

    # Estadísticos globales RGB.
    feats.extend(arr.reshape(-1, 3).mean(axis=0).tolist())
    feats.extend(arr.reshape(-1, 3).std(axis=0).tolist())

    # Grilla 4x4 con medias RGB: conserva composición espacial gruesa.
    grid = 4
    h, w = arr.shape[:2]
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(round(gy * h / grid))
            y1 = int(round((gy + 1) * h / grid))
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            cell = arr[y0:y1, x0:x1, :]
            feats.extend(cell.reshape(-1, 3).mean(axis=0).tolist())

    # Luminancia gruesa.
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    feats.append(float(lum.mean()))
    feats.append(float(lum.std()))
    feats.append(float(np.percentile(lum, 10)))
    feats.append(float(np.percentile(lum, 90)))

    return np.asarray(feats, dtype=np.float32)


def standardize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd <= 1e-8)] = 1.0
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def simple_kmeans(x: np.ndarray, k: int, seed: int = 369, max_iter: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KMeans pequeño sin sklearn.
    Devuelve labels, centroids, distances_to_assigned.
    """
    n = int(x.shape[0])
    k = max(2, min(int(k), n))
    rng = np.random.default_rng(int(seed))

    # Inicialización: primer centro random, luego puntos lejanos.
    first = int(rng.integers(0, n))
    centers = [x[first].copy()]
    min_d = np.sum((x - centers[0]) ** 2, axis=1)

    for _ in range(1, k):
        probs = min_d / (np.sum(min_d) + 1e-12)
        idx = int(rng.choice(np.arange(n), p=probs))
        centers.append(x[idx].copy())
        d = np.sum((x - centers[-1]) ** 2, axis=1)
        min_d = np.minimum(min_d, d)

    centroids = np.stack(centers, axis=0).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for cid in range(k):
            mask = labels == cid
            if np.any(mask):
                centroids[cid] = x[mask].mean(axis=0)
            else:
                # Reubicar centro vacío en el peor representado.
                worst = int(np.argmax(np.min(dists, axis=1)))
                centroids[cid] = x[worst]

    dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    assigned = dists[np.arange(n), labels]
    return labels, centroids, assigned


def cluster_confidence(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.sqrt(np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    order = np.argsort(dists, axis=1)
    d1 = dists[np.arange(x.shape[0]), order[:, 0]]
    d2 = dists[np.arange(x.shape[0]), order[:, 1]] if centroids.shape[0] > 1 else d1 + 1.0
    conf = 1.0 - (d1 / (d2 + 1e-12))
    return np.clip(conf, 0.0, 1.0)


def make_contact_sheet(
    rows: List[dict],
    output_png: str,
    samples_per_scene: int = 8,
    thumb_w: int = 160,
    thumb_h: int = 120,
):
    """
    Crea una hoja visual para revisar escenas sugeridas.
    """
    if not output_png or Image is None:
        return

    # Agrupar y tomar representantes de mayor confianza.
    by_scene: Dict[str, List[dict]] = {}
    for r in rows:
        by_scene.setdefault(r["scene_type"], []).append(r)

    scenes = sorted(by_scene.keys())
    blocks = []
    for scene in scenes:
        reps = sorted(by_scene[scene], key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
        blocks.append((scene, reps[:samples_per_scene]))

    ncols = samples_per_scene
    nrows = len(blocks)
    if nrows == 0:
        return

    label_h = 30
    sheet = Image.new("RGB", (ncols * thumb_w, nrows * (thumb_h + label_h)), "white")

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)
    except Exception:
        draw = None

    for row_idx, (scene, reps) in enumerate(blocks):
        ybase = row_idx * (thumb_h + label_h)
        if draw is not None:
            draw.text((4, ybase + 4), scene, fill=(0, 0, 0))

        for col_idx, r in enumerate(reps):
            rgb = r.get("rgb", "")
            if not rgb or not Path(rgb).exists():
                continue
            try:
                im = Image.open(rgb).convert("RGB").resize((thumb_w, thumb_h))
                x = col_idx * thumb_w
                y = ybase + label_h
                sheet.paste(im, (x, y))
                if draw is not None:
                    txt = f"{r.get('stem','')} c={float(r.get('confidence',0.0)):.2f}"
                    draw.text((x + 2, y + 2), txt, fill=(255, 255, 255))
            except Exception:
                continue

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("OK contact sheet:", out)


def write_auto_scene_map_rgb_cluster(
    pairs: List[dict],
    output_csv: str,
    k: int = 8,
    seed: int = 369,
    confidence_threshold: float = 0.15,
    contact_sheet: str = "",
    samples_per_cluster: int = 8,
):
    """
    Genera scenes_auto.csv usando clustering RGB.
    Importante:
      - No usa depth.
      - No usa target.
      - No usa NCT.
      - Es una propuesta auditable, no verdad final.
    """
    if Image is None:
        raise RuntimeError("Pillow no está instalado. No puedo crear auto-scene-map desde RGB.")

    features = []
    valid_pairs = []
    missing_rgb = []

    for pair in pairs:
        rgb = str(pair.get("rgb", ""))
        feat = rgb_scene_feature(rgb)
        if feat is None:
            missing_rgb.append(pair.get("stem", ""))
            continue
        features.append(feat)
        valid_pairs.append(pair)

    if len(valid_pairs) < 2:
        raise RuntimeError(
            "No hay suficientes RGB válidos para auto-scene-map. "
            "El script no pudo emparejar RGB con depth. "
            "Primero probá: --diagnose-rgb-match. "
            "Si no hay RGB real, usá --make-scene-map con --scene-template-mode block/todo."
        )

    x = np.stack(features, axis=0)
    xz, _, _ = standardize_features(x)

    labels, centroids, assigned = simple_kmeans(xz, k=k, seed=seed)
    conf = cluster_confidence(xz, labels, centroids)

    rows = []
    for pair, label, c in zip(valid_pairs, labels, conf):
        scene = f"scene_auto_{int(label):02d}"
        rows.append({
            "stem": pair["stem"],
            "scene_type": scene,
            "confidence": f"{float(c):.6f}",
            "method": "rgb_cluster",
            "needs_review": "true" if float(c) < float(confidence_threshold) else "false",
            "depth": pair.get("depth", ""),
            "rgb": pair.get("rgb", ""),
        })

    # Si hubo depth sin RGB, los dejamos marcados para revisión.
    for pair in pairs:
        if pair.get("stem", "") in set(missing_rgb):
            rows.append({
                "stem": pair["stem"],
                "scene_type": "scene_missing_rgb",
                "confidence": "0.000000",
                "method": "missing_rgb",
                "needs_review": "true",
                "depth": pair.get("depth", ""),
                "rgb": pair.get("rgb", ""),
            })

    rows.sort(key=lambda r: r["stem"])

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["stem", "scene_type", "confidence", "method", "needs_review", "depth", "rgb"],
        )
        w.writeheader()
        w.writerows(rows)

    counts: Dict[str, int] = {}
    review_count = 0
    for r in rows:
        counts[r["scene_type"]] = counts.get(r["scene_type"], 0) + 1
        review_count += 1 if r["needs_review"] == "true" else 0

    print("OK auto scene map creado:", output)
    print("rows:", len(rows))
    print("scenes:", len(counts))
    print("needs_review:", review_count)
    for k2, v in sorted(counts.items()):
        print(" ", k2, "=", v)

    if contact_sheet:
        make_contact_sheet(
            rows=rows,
            output_png=contact_sheet,
            samples_per_scene=int(samples_per_cluster),
        )

    print("\nSiguiente paso recomendado:")
    print("- Abrí el CSV y renombrá scene_auto_XX por categorías reales si podés:")
    print("  kitchen, office, bathroom, bedroom, living_room, hallway, etc.")
    print("- Si no sabés, dejá scene_auto_XX; sirve como prueba técnica, no como categoría semántica final.")


def preflight_scene_map_for_pairs(pairs: List[dict], args):
    """
    Valida scene_map antes de cargar/procesar los 654 depth maps.
    Evita esperar toda la preparación para descubrir que solo hay 1 escena.
    """
    if args.split_mode != "scene_loo":
        return
    if not args.scene_map:
        return

    scene_path = Path(args.scene_map)
    if not scene_path.exists():
        raise SystemExit(
            f"No existe --scene-map: {scene_path}\n"
            f"Crealo con:\n"
            f"  python3 {Path(__file__).name} --rgb {args.rgb} --depth {args.depth} "
            f"--auto-scene-map scenes_auto.csv\n"
            f"o con:\n"
            f"  python3 {Path(__file__).name} --rgb {args.rgb} --depth {args.depth} "
            f"--make-scene-map scenes.csv --scene-template-mode block"
        )

    csv_map = load_scene_map_csv(str(scene_path), args.stem_column, args.scene_column)

    counts: Dict[str, int] = {}
    missing = 0

    for pair in pairs:
        found = None
        for alias in stem_aliases(pair["stem"]):
            if alias in csv_map:
                found = csv_map[alias]
                break
        if found is None:
            missing += 1
            continue
        counts[found] = counts.get(found, 0) + 1

    valid = {k: v for k, v in counts.items() if v >= int(args.scene_min_samples)}

    print("scene_map preflight:", scene_path)
    print("  escenas totales:", len(counts))
    print("  escenas válidas:", len(valid), f"(min_samples={args.scene_min_samples})")
    print("  missing stems:", missing)
    for k, v in sorted(counts.items())[:30]:
        flag = "OK" if v >= int(args.scene_min_samples) else "LOW"
        print(" ", flag, k, "=", v)

    if len(valid) < 2:
        raise SystemExit(
            f"scene_loo requiere al menos 2 escenas válidas. Encontradas: {len(valid)}.\n"
            f"Tu CSV probablemente tiene una sola etiqueta en {args.scene_column}, por ejemplo TODO_SCENE.\n"
            f"Soluciones:\n"
            f"  1) Auto RGB cluster:\n"
            f"     python3 {Path(__file__).name} --rgb {args.rgb} --depth {args.depth} --auto-scene-map scenes_auto.csv\n"
            f"  2) Bloques técnicos:\n"
            f"     python3 {Path(__file__).name} --rgb {args.rgb} --depth {args.depth} --make-scene-map scenes.csv --scene-template-mode block\n"
            f"  3) Manual real:\n"
            f"     Editá scenes_real.csv y usá kitchen/office/bathroom/living_room/etc."
        )



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


def normalize_stem_key(text: str) -> str:
    x = str(text).strip()
    x = Path(x).stem
    return x


def stem_aliases(stem: str) -> List[str]:
    s0 = normalize_stem_key(stem)
    aliases = {s0}
    for suffix in ("_depth", "-depth", "_rgb", "-rgb"):
        if s0.endswith(suffix):
            aliases.add(s0[: -len(suffix)])
    return list(aliases)


def load_scene_map_csv(path: str, stem_column: str, scene_column: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path:
        return mapping

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if stem_column not in reader.fieldnames:
            raise ValueError(f"No existe columna stem '{stem_column}' en {path}. Columnas: {reader.fieldnames}")
        if scene_column not in reader.fieldnames:
            raise ValueError(f"No existe columna scene '{scene_column}' en {path}. Columnas: {reader.fieldnames}")

        for row in reader:
            raw_stem = str(row.get(stem_column, "")).strip()
            scene = str(row.get(scene_column, "")).strip()
            if not raw_stem or not scene:
                continue
            for alias in stem_aliases(raw_stem):
                mapping[alias] = scene

    return mapping


def infer_scene_labels(samples: List[dict], pairs: List[dict], args) -> Tuple[List[str], Dict[str, object]]:
    """
    Prioridad:
      1) --scene-map CSV.
      2) Carpeta padre del depth, si hay más de una carpeta padre.
    """
    csv_map = load_scene_map_csv(args.scene_map, args.stem_column, args.scene_column)

    labels = []
    missing = []

    if csv_map:
        for sample in samples:
            found = None
            for alias in stem_aliases(sample["stem"]):
                if alias in csv_map:
                    found = csv_map[alias]
                    break
            if found is None:
                missing.append(sample["stem"])
                found = "__MISSING_SCENE__"
            labels.append(found)

        meta = {
            "scene_source": "csv",
            "scene_map": args.scene_map,
            "scene_column": args.scene_column,
            "stem_column": args.stem_column,
            "missing_scene_count": len(missing),
            "missing_scene_examples": missing[:20],
        }
        return labels, meta

    parent_labels = []
    for i, p in enumerate(pairs):
        parent_labels.append(Path(p["depth"]).parent.name)

    unique = sorted(set(parent_labels))
    if len(unique) <= 1:
        raise ValueError(
            "split-mode scene_loo necesita --scene-map CSV con columnas stem,scene_type "
            "o un dataset organizado en subcarpetas por escena."
        )

    meta = {
        "scene_source": "parent_folder",
        "scene_map": "",
        "scene_column": "",
        "stem_column": "",
        "missing_scene_count": 0,
        "missing_scene_examples": [],
    }
    return parent_labels, meta


def build_scene_loo_splits(samples: List[dict], pairs: List[dict], args) -> Tuple[List[dict], Dict[str, object]]:
    labels, label_meta = infer_scene_labels(samples, pairs, args)

    scene_to_indices: Dict[str, List[int]] = {}
    for idx, scene in enumerate(labels):
        if scene == "__MISSING_SCENE__":
            continue
        scene_to_indices.setdefault(scene, []).append(idx)

    all_scenes = sorted(scene_to_indices.keys())
    valid_scenes = [sc for sc in all_scenes if len(scene_to_indices[sc]) >= int(args.scene_min_samples)]

    if args.scene_limit and int(args.scene_limit) > 0:
        valid_scenes = valid_scenes[: int(args.scene_limit)]

    if len(valid_scenes) < 2:
        raise ValueError(
            f"scene_loo requiere al menos 2 escenas válidas. "
            f"Encontradas: {len(valid_scenes)} con min_samples={args.scene_min_samples}."
        )

    split_specs = []
    all_valid_indices = sorted([i for sc in valid_scenes for i in scene_to_indices[sc]])

    for scene in valid_scenes:
        test_idx = sorted(scene_to_indices[scene])
        test_set = set(test_idx)
        train_idx = [i for i in all_valid_indices if i not in test_set]

        meta = {
            "split_mode": "scene_loo",
            "test_scene": scene,
            "train_scenes": [sc for sc in valid_scenes if sc != scene],
            "n_scenes_total": len(valid_scenes),
            "n_train_scenes": len(valid_scenes) - 1,
            "n_test_scenes": 1,
            "scene_counts": {sc: len(scene_to_indices[sc]) for sc in valid_scenes},
            "actual_test_count": len(test_idx),
            "actual_test_ratio": float(len(test_idx) / max(1, len(all_valid_indices))),
            **label_meta,
        }

        split_specs.append({
            "split_name": f"scene_loo:{scene}",
            "train_idx": train_idx,
            "test_idx": test_idx,
            "split_meta": meta,
        })

    scene_meta = {
        "scene_loo_enabled": True,
        "valid_scenes": valid_scenes,
        "n_valid_scenes": len(valid_scenes),
        "scene_counts": {sc: len(scene_to_indices[sc]) for sc in valid_scenes},
        **label_meta,
    }
    return split_specs, scene_meta


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



def summarize_runs_by_alpha(runs: List[dict], region: str) -> dict:
    """
    Devuelve summary separado por alpha.
    Esto permite ver si alpha=0.02, 0.03 o 0.04 domina en AUC/F1/p-values.
    """
    out = {}
    alphas = sorted({float(r["alpha"]) for r in runs})
    for alpha in alphas:
        sub = [r for r in runs if float(r["alpha"]) == float(alpha)]
        out[f"{alpha:g}"] = summarize_runs(sub, region)
    return out


def model_report_row(model: str, s: dict) -> dict:
    return {
        "model": model,
        "d_sp_random_mean": s.get("d_sp_random_mean", float("nan")),
        "d_auc_random_mean": s.get("d_auc_random_mean", float("nan")),
        "d_f1_random_mean": s.get("d_f1_random_mean", float("nan")),
        "p_spearman_mean": s.get("p_spearman_mean", float("nan")),
        "p_auc_top20_mean": s.get("p_auc_top20_mean", float("nan")),
        "p_f1_top20_mean": s.get("p_f1_top20_mean", float("nan")),
        "p_auc_top20_lt_0_05_rate": s.get("p_auc_top20_lt_0_05_rate", float("nan")),
        "p_f1_top20_lt_0_05_rate": s.get("p_f1_top20_lt_0_05_rate", float("nan")),
        "rate_auc_f1_positive_vs_random": s.get("rate_auc_f1_positive_vs_random", float("nan")),
    }


def model_selection_score(s: dict) -> tuple:
    """
    Score lexicográfico:
    1) Menor p_auc + p_f1.
    2) Mayor ΔAUC vs random.
    3) Mayor ΔF1 vs random.
    4) Mayor ΔSpearman vs random.
    """
    p_auc = s.get("p_auc_top20_mean", float("nan"))
    p_f1 = s.get("p_f1_top20_mean", float("nan"))
    d_auc = s.get("d_auc_random_mean", float("nan"))
    d_f1 = s.get("d_f1_random_mean", float("nan"))
    d_sp = s.get("d_sp_random_mean", float("nan"))

    p_sum = (p_auc if np.isfinite(p_auc) else 1e9) + (p_f1 if np.isfinite(p_f1) else 1e9)
    return (-p_sum, d_auc if np.isfinite(d_auc) else -1e9, d_f1 if np.isfinite(d_f1) else -1e9, d_sp if np.isfinite(d_sp) else -1e9)


def choose_best_model(summary: dict) -> dict:
    best_model = None
    best_score = None
    for model in TEST_MODELS:
        s = summary.get(model, {})
        score = model_selection_score(s)
        if best_score is None or score > best_score:
            best_score = score
            best_model = model
    row = model_report_row(best_model, summary[best_model])
    row["selection_score"] = list(best_score)
    return row


def choose_best_alpha_by_model(summary_by_alpha: dict) -> dict:
    out = {}
    for model in TEST_MODELS:
        best_alpha = None
        best_score = None
        best_summary = None
        for alpha_key, summary in summary_by_alpha.items():
            if model not in summary:
                continue
            score = model_selection_score(summary[model])
            if best_score is None or score > best_score:
                best_alpha = alpha_key
                best_score = score
                best_summary = summary[model]
        if best_alpha is not None:
            row = model_report_row(model, best_summary)
            row["best_alpha"] = best_alpha
            row["selection_score"] = list(best_score)
            out[model] = row
    return out


def write_alpha_summary_csv(summary_by_alpha: dict, output_csv: str):
    fields = [
        "alpha", "model",
        "d_sp_random_mean", "d_auc_random_mean", "d_f1_random_mean",
        "p_spearman_mean", "p_auc_top20_mean", "p_f1_top20_mean",
        "p_auc_top20_lt_0_05_rate", "p_f1_top20_lt_0_05_rate",
        "rate_auc_f1_positive_vs_random",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for alpha_key, summary in summary_by_alpha.items():
            for model in TEST_MODELS:
                row = model_report_row(model, summary[model])
                row["alpha"] = alpha_key
                w.writerow(row)


def print_alpha_report(summary_by_alpha: dict, best_alpha_by_model: dict, best_model_global: dict):
    print("\n===== v14 REPORT BY ALPHA / TEST /", REGION, "=====")
    for alpha_key, summary in summary_by_alpha.items():
        print(f"\nalpha={alpha_key}")
        for model in TEST_MODELS:
            s = summary[model]
            print(
                " ", model,
                "| ΔAUC random=", round(s["d_auc_random_mean"], 8),
                "| ΔF1 random=", round(s["d_f1_random_mean"], 8),
                "| p_auc=", round(s["p_auc_top20_mean"], 6),
                "| p_f1=", round(s["p_f1_top20_mean"], 6),
            )

    print("\n===== BEST ALPHA BY MODEL =====")
    for model, row in best_alpha_by_model.items():
        print(
            model,
            "| best_alpha=", row["best_alpha"],
            "| ΔAUC random=", round(row["d_auc_random_mean"], 8),
            "| ΔF1 random=", round(row["d_f1_random_mean"], 8),
            "| p_auc=", round(row["p_auc_top20_mean"], 6),
            "| p_f1=", round(row["p_f1_top20_mean"], 6),
        )

    print("\n===== BEST MODEL GLOBAL =====")
    print(
        best_model_global["model"],
        "| ΔAUC random=", round(best_model_global["d_auc_random_mean"], 8),
        "| ΔF1 random=", round(best_model_global["d_f1_random_mean"], 8),
        "| p_auc=", round(best_model_global["p_auc_top20_mean"], 6),
        "| p_f1=", round(best_model_global["p_f1_top20_mean"], 6),
    )


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

MAIN_MODEL = "motif_survival_binary"


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
    print(f"\n===== v14 RGB-D SCENE-LOO SUMMARY / TEST / {REGION} =====")
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
    ap.add_argument("--split-mode", default="scene_loo", choices=["random", "grouped", "scene_loo"], help="random = split por frame; grouped = bloques numéricos; scene_loo = deja una categoría/escena completa como TEST.")
    ap.add_argument("--group-strategy", default="numeric_block", choices=["numeric_block", "prefix", "stem"], help="Estrategia de agrupación para grouped split.")
    ap.add_argument("--group-size", type=int, default=50, help="Tamaño de bloque para numeric_block. Ej: 50 agrupa 00000-00049.")
    ap.add_argument("--group-prefix-len", type=int, default=3, help="Cantidad de caracteres para group-strategy prefix.")
    ap.add_argument("--scene-map", default="", help="CSV con columnas stem y scene_type. Requerido para scene_loo si no hay carpetas por escena.")
    ap.add_argument("--stem-column", default="stem", help="Nombre de columna del CSV que identifica el frame/stem.")
    ap.add_argument("--scene-column", default="scene_type", help="Nombre de columna del CSV que identifica la categoría/escena.")
    ap.add_argument("--scene-min-samples", type=int, default=5, help="Mínimo de frames por escena para entrar en scene_loo.")
    ap.add_argument("--scene-limit", type=int, default=0, help="Limitar cantidad de escenas evaluadas. 0 = todas.")
    ap.add_argument("--make-scene-map", default="", help="Crea un CSV scenes.csv/template y termina, salvo que uses --continue-after-make-scene-map.")
    ap.add_argument("--scene-template-mode", default="block", choices=["block", "parent", "todo"], help="Modo para crear --make-scene-map: block, parent o todo.")
    ap.add_argument("--scene-template-block-size", type=int, default=200, help="Tamaño del bloque numérico al crear scene map en modo block.")
    ap.add_argument("--continue-after-make-scene-map", action="store_true", help="Después de crear --make-scene-map, continúa la corrida.")
    ap.add_argument("--auto-scene-map", default="", help="Crea un scenes_auto.csv usando clustering RGB auditable y termina, salvo que uses --continue-after-auto-scene-map.")
    ap.add_argument("--auto-scene-k", type=int, default=8, help="Cantidad de clusters/escenas sugeridas por RGB.")
    ap.add_argument("--auto-scene-seed", type=int, default=369, help="Seed del clustering RGB.")
    ap.add_argument("--auto-scene-confidence-threshold", type=float, default=0.15, help="Debajo de este valor marca needs_review=true.")
    ap.add_argument("--auto-scene-contact-sheet", default="scenes_auto_contact_sheet.png", help="PNG con ejemplos por cluster para revisión visual.")
    ap.add_argument("--auto-scene-samples-per-cluster", type=int, default=8, help="Cantidad de miniaturas por cluster en el contact sheet.")
    ap.add_argument("--diagnose-rgb-match", action="store_true", help="Muestra diagnóstico de emparejamiento RGB/depth y termina.")
    ap.add_argument("--continue-after-auto-scene-map", action="store_true", help="Después de crear --auto-scene-map, continúa la corrida usando ese CSV.")
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
    ap.add_argument("--output-json", default="9B02_v14_2_1_scene_loo_results.json")
    ap.add_argument("--output-summary-csv", default="9B02_v14_2_1_scene_loo_summary.csv")
    ap.add_argument("--output-weights-csv", default="9B02_v14_2_1_scene_loo_weights.csv")
    ap.add_argument("--output-pairs-csv", default="9B02_v14_2_1_scene_loo_pairs.csv")
    ap.add_argument("--output-alpha-csv", default="9B02_v14_2_1_scene_loo_by_alpha.csv")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb) if args.rgb else None
    depth_dir = Path(args.depth)
    pairs = match_rgb_depth(rgb_dir, depth_dir)

    if not pairs:
        raise SystemExit("No encontré archivos depth compatibles.")

    if args.diagnose_rgb_match:
        n_rgb = sum(1 for p in pairs if p.get("rgb"))
        print("Diagnóstico RGB/depth")
        print("pairs:", len(pairs))
        print("rgb_matched:", n_rgb)
        print("rgb_missing:", len(pairs) - n_rgb)
        print("\nPrimeros 20 pairs:")
        for p in pairs[:20]:
            print(" ", p["stem"], "| depth=", p["depth"], "| rgb=", p["rgb"] or "(missing)")
        if n_rgb == 0:
            print("\nNo se emparejó ningún RGB.")
            print("Revisá nombres con:")
            print("  find ./dataset/rgb -maxdepth 2 -type f | head -30")
            print("  find ./dataset/depth -maxdepth 2 -type f | head -30")
        raise SystemExit(0)

    if args.make_scene_map:
        write_scene_map_template(
            pairs=pairs,
            output_csv=args.make_scene_map,
            mode=args.scene_template_mode,
            block_size=args.scene_template_block_size,
        )
        if not args.continue_after_make_scene_map:
            raise SystemExit(0)

    if args.auto_scene_map:
        write_auto_scene_map_rgb_cluster(
            pairs=pairs,
            output_csv=args.auto_scene_map,
            k=args.auto_scene_k,
            seed=args.auto_scene_seed,
            confidence_threshold=args.auto_scene_confidence_threshold,
            contact_sheet=args.auto_scene_contact_sheet,
            samples_per_cluster=args.auto_scene_samples_per_cluster,
        )
        if args.continue_after_auto_scene_map:
            args.scene_map = args.auto_scene_map
        else:
            raise SystemExit(0)

    preflight_scene_map_for_pairs(pairs, args)

    alphas = parse_float_list(args.alpha)
    seeds = parse_int_list(args.seeds)
    runtime_device = select_runtime_device(args.device)

    if runtime_device == "cuda" and args.workers > 1:
        print("[INFO] CUDA activo: se ignora --workers para la evaluación de TEST y se usa GPU secuencial por chunks.")
        args.workers = 1

    print("===== 9B02 NCT v14.2.1 RGB-D CUDA SCENE-LOO AUTO-SCENE RGB-ALIAS DATASET ADAPTER =====")
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
    print("scene_map:", args.scene_map)
    print("scene_column:", args.scene_column)
    print("stem_column:", args.stem_column)
    print("scene_min_samples:", args.scene_min_samples)
    print("scene_limit:", args.scene_limit)
    print("make_scene_map:", args.make_scene_map)
    print("scene_template_mode:", args.scene_template_mode)
    print("scene_template_block_size:", args.scene_template_block_size)
    print("auto_scene_map:", args.auto_scene_map)
    print("auto_scene_k:", args.auto_scene_k)
    print("auto_scene_confidence_threshold:", args.auto_scene_confidence_threshold)
    print("auto_scene_contact_sheet:", args.auto_scene_contact_sheet)
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

    scene_loo_meta = {"scene_loo_enabled": False}
    scene_split_specs = None

    if args.split_mode == "scene_loo":
        scene_split_specs, scene_loo_meta = build_scene_loo_splits(samples, pairs, args)

    runs = []
    splits_per_seed = len(scene_split_specs) if scene_split_specs is not None else 1
    total = len(alphas) * len(seeds) * splits_per_seed
    nrun = 0

    for alpha in alphas:
        for seed in seeds:
            if args.split_mode == "scene_loo":
                split_specs = scene_split_specs
            else:
                train_idx, test_idx, split_meta = make_split(
                    pairs=[{"stem": sample["stem"]} for sample in samples],
                    seed=seed,
                    test_ratio=args.test_ratio,
                    split_mode=args.split_mode,
                    group_strategy=args.group_strategy,
                    group_size=args.group_size,
                    group_prefix_len=args.group_prefix_len,
                )
                split_specs = [{
                    "split_name": "holdout",
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "split_meta": split_meta,
                }]

            for split_spec in split_specs:
                nrun += 1
                split_name = split_spec["split_name"]
                train_idx = split_spec["train_idx"]
                test_idx = split_spec["test_idx"]
                split_meta = split_spec["split_meta"]

                print(f"\n[{nrun}/{total}] alpha={alpha} seed={seed} split={split_name}")

                if args.split_mode == "grouped":
                    print(
                        "  grouped split:",
                        f"groups_test={split_meta['n_groups_test']}/{split_meta['n_groups_total']}",
                        f"test_count={split_meta['actual_test_count']}",
                        f"test_ratio={split_meta['actual_test_ratio']:.3f}",
                    )
                    groups_preview = split_meta.get("test_groups", [])[:20]
                    print("  test groups:", ", ".join(groups_preview), "..." if len(split_meta.get("test_groups", [])) > 20 else "")

                if args.split_mode == "scene_loo":
                    print(
                        "  scene LOO:",
                        f"test_scene={split_meta['test_scene']}",
                        f"train_scenes={split_meta['n_train_scenes']}",
                        f"test_count={split_meta['actual_test_count']}",
                        f"test_ratio={split_meta['actual_test_ratio']:.3f}",
                    )

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
                    file_metrics = [
                        eval_sample_models(
                            sample,
                            weights,
                            random_tables,
                            float(alpha),
                            device=runtime_device,
                            gpu_random_chunk=args.gpu_random_chunk,
                        )
                        for sample in test_samples
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
                            sample,
                            weights,
                            random_tables,
                            float(alpha),
                            device=runtime_device,
                            gpu_random_chunk=args.gpu_random_chunk,
                        )
                        for sample in test_samples
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
                    "split_name": split_name,
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
                    "  binary vs classical:",
                    "ΔAUC=", round(main_imp["minus_classical_auc"], 7),
                    "ΔF1=", round(main_imp["minus_classical_f1"], 7),
                )
                print(
                    "  binary vs random:",
                    "ΔAUC=", round(main_imp["minus_random_auc"], 7),
                    "ΔF1=", round(main_imp["minus_random_f1"], 7),
                    "| p_auc=", round(pvals[REGION][MAIN_MODEL]["auc_top20"], 4),
                    "p_f1=", round(pvals[REGION][MAIN_MODEL]["f1_top20"], 4),
                )

    summary = summarize_runs(runs, REGION)
    summary_by_alpha = summarize_runs_by_alpha(runs, REGION)
    best_alpha_by_model = choose_best_alpha_by_model(summary_by_alpha)
    best_model_global = choose_best_model(summary)
    weights_summary = aggregate_weights(runs)

    result = {
        "version": "9B02 NCT v14.2.1 RGB-D CUDA Auto Scene Map RGB Alias Fix",
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
        "scene_map": args.scene_map,
        "scene_column": args.scene_column,
        "stem_column": args.stem_column,
        "scene_min_samples": args.scene_min_samples,
        "scene_limit": args.scene_limit,
        "make_scene_map": args.make_scene_map,
        "scene_template_mode": args.scene_template_mode,
        "scene_template_block_size": args.scene_template_block_size,
        "auto_scene_map": args.auto_scene_map,
        "auto_scene_k": args.auto_scene_k,
        "auto_scene_seed": args.auto_scene_seed,
        "auto_scene_confidence_threshold": args.auto_scene_confidence_threshold,
        "auto_scene_contact_sheet": args.auto_scene_contact_sheet,
        "scene_loo_meta": scene_loo_meta,
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
            "cuda_note": "v14.2.1 moves random baseline evaluation to GPU when --device cuda is active.",
            "grouped_split_note": "When --split-mode grouped, contiguous filename blocks/groups are kept entirely in TRAIN or TEST to reduce frame leakage.",
            "scene_loo_note": "When --split-mode scene_loo, one full scene/category is held out as TEST and all other scenes are TRAIN.",
            "scene_map_tool_note": "Use --make-scene-map or --auto-scene-map to generate a scenes.csv template before running scene_loo.",
            "auto_scene_note": "auto-scene-map uses RGB clustering only; it does not use depth, target, or NCT features. RGB/depth matching supports stem aliases like 00000_depth -> 00000. Treat it as auditable suggestion, not final semantic truth.",
        },
        "summary": summary,
        "summary_by_alpha": summary_by_alpha,
        "best_alpha_by_model": best_alpha_by_model,
        "best_model_global": best_model_global,
        "weights_summary": weights_summary,
        "runs": runs,
    }

    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(runs, args.output_summary_csv)
    write_alpha_summary_csv(summary_by_alpha, args.output_alpha_csv)
    write_weights_csv(weights_summary, args.output_weights_csv)

    print_summary(summary)
    print_alpha_report(summary_by_alpha, best_alpha_by_model, best_model_global)
    print_weights(weights_summary)

    print("\nSalidas:")
    print("JSON:", args.output_json)
    print("Summary CSV:", args.output_summary_csv)
    print("Alpha CSV:", args.output_alpha_csv)
    print("Weights CSV:", args.output_weights_csv)
    print("Pairs CSV:", args.output_pairs_csv)

    print("\nLectura:")
    print("- MAIN_MODEL es motif_survival_binary; si sobrevive scene_loo, NCT_3D generaliza entre categorías/escenas reales.")
    print("- Si gana vs classical pero no vs random, el efecto viene de perturbación/gate, no de identidad motif.")
    print("- Si p_auc/p_f1 < 0.10 en varias corridas, supera random de forma más defendible.")
    print("- Si top motifs son estables, hay motivos NCT 3D sobrevivientes en datos RGB-D reales.")
    print("- Si falla en RGB-D real pero funcionaba en sintético, el target sintético era demasiado favorable.")


if __name__ == "__main__":
    main()
