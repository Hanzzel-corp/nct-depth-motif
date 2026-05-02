#!/bin/bash
# =============================================================================
# NCT Motif Survival - Scene Leave-One-Out Validation (24 runs)
# =============================================================================
#
# Este script reproduce la validación leave-one-scene-out con 24 corridas
# (una por cada escena de NYU Depth V2).
#
# QUÉ HACE:
#   - Lee mapeo de escenas desde results/scenes_auto.csv
#   - Para cada escena i: train = todas las demás, test = escena i
#   - Evalúa 3 valores de alpha (0.02, 0.03, 0.04)
#   - Computa 256 baselines aleatorios para p-value empírico
#
# DIFERENCIAS CON GROUPED SPLIT:
#   - Máxima exigencia en generalización (escenas no vistas)
#   - Más lento: cada iteración entrena desde cero
#   - Requiere archivo scenes_auto.csv
#
# TIEMPO ESTIMADO:
#   - GPU (CUDA): ~2-3 horas
#   - CPU: ~80+ horas (no recomendado)
#
# REQUISITOS:
#   - Dataset NYU Depth V2 en ./dataset/
#   - Archivo scenes_auto.csv en ./results/
#   - Entorno virtual activado
#   - PyTorch con CUDA (altamente recomendado)
#
# SALIDAS:
#   - 9B02_scene_loo_results.json   (resultados por escena)
#   - 9B02_scene_loo_summary.csv    (resumen agregado)
#   - 9B02_scene_loo_weights.csv    (pesos por iteración)
#
# VERIFICACIÓN:
#   Comparar con results/scene_loo_24runs_summary.csv
#   Las columnas p_sp, p_auc, p_f1 deben coincidir.
#
# =============================================================================

set -e

echo "Iniciando scene leave-one-out validation (24 runs)..."
echo "Hora inicio: $(date)"
echo ""
echo "Advertencia: Este script es ~2x más lento que grouped split"
echo "porque cada iteración entrena desde cero en 23 escenas."
echo ""

# Configuración de threading para evitar contención
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 src/motif_survival_scene_loo.py \
  --rgb ./dataset/rgb \
  --depth ./dataset/depth \
  --scenes ./results/scenes_auto.csv \
  --target combined \
  --alpha 0.02,0.03,0.04 \
  --random-baselines 256 \
  --device cuda \
  --depth-scale 1000 \
  --fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362 \
  --max-size 160

echo ""
echo "Completado. Hora fin: $(date)"
echo ""
echo "Resultados generados:"
ls -lh 9B02_scene_loo_*.{json,csv} 2>/dev/null || echo "Revisar archivos de salida"
