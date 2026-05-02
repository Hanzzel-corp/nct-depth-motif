#!/bin/bash
# =============================================================================
# NCT Motif Survival - Grouped Split Validation (30 runs)
# =============================================================================
#
# Este script reproduce la validación principal del paper con 30 corridas
# sobre NYU Depth V2 usando grouped split cross-validation.
#
# QUÉ HACE:
#   - Divide las escenas en grupos por orden numérico (50 escenas por grupo)
#   - Para cada seed: entrena en N-1 grupos, test en el grupo restante
#   - Evalúa 3 valores de alpha (0.02, 0.03, 0.04)
#   - Computa 256 baselines aleatorios para p-value empírico
#
# TIEMPO ESTIMADO:
#   - GPU (CUDA): ~1 hora
#   - CPU: ~38 horas
#
# REQUISITOS:
#   - Dataset NYU Depth V2 en ./dataset/
#   - Entorno virtual activado (source .venv/bin/activate)
#   - PyTorch con CUDA (opcional pero recomendado)
#
# SALIDAS:
#   - 9B02_grouped_split_results.json   (resultados detallados)
#   - 9B02_grouped_split_summary.csv    (resumen estadístico)
#   - 9B02_grouped_split_weights.csv    (tablas de pesos)
#
# VERIFICACIÓN:
#   Comparar con results/grouped_split_30runs_summary.csv
#   Las columnas p_sp, p_auc, p_f1 deben coincidir.
#
# =============================================================================

set -e

echo "Iniciando grouped split validation (30 runs)..."
echo "Hora inicio: $(date)"

# Configuración de threading para evitar contención
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 src/motif_survival_grouped.py \
  --rgb ./dataset/rgb \
  --depth ./dataset/depth \
  --target combined \
  --alpha 0.02,0.03,0.04 \
  --seeds 11,22,33,44,55,66,77,88,99,111 \
  --random-baselines 256 \
  --device cuda \
  --gpu-random-chunk 32 \
  --split-mode grouped \
  --group-strategy numeric_block \
  --group-size 50 \
  --depth-scale 1000 \
  --fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362 \
  --max-size 160

echo ""
echo "Completado. Hora fin: $(date)"
echo ""
echo "Resultados generados:"
ls -lh 9B02_grouped_split_*.{json,csv} 2>/dev/null || echo "Revisar archivos de salida"
