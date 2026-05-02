# Guía de Reproducción

Pasos para reproducir los resultados reportados en el README.

**Estado:** Reporte técnico exploratorio. Resultados validados sobre NYU Depth V2 únicamente.

---

## Requisitos

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | No requerida | CUDA-capable (acelera ~38x: de ~38h a ~1h) |
| Almacenamiento | 10 GB | 50 GB (dataset completo) |
| Python | 3.9+ | 3.10+ |
| OS | Linux/macOS/Windows | Ubuntu 20.04+ |

**Dependencias:** numpy, pillow, scipy, PyTorch (opcional, para acelerar)

---

## Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/Hanzzel-corp/nct-depth-motif.git
cd nct-depth-motif
```

---

## Paso 2: Configurar Entorno

```bash
# Crear e instalar dependencias
bash setup_env.sh

# Activar entorno
source .venv/bin/activate

# Verificar instalación
python3 -c "import torch, numpy, scipy, PIL; print('✓ Todo instalado')"
```

**Nota para GPU**: Si tienes CUDA, instala PyTorch específico:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Paso 3: Descargar Dataset

### NYU Depth V2 (Labeled)

1. Visitar: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Descargar **Labeled dataset** (~2.8 GB)
3. Extraer en `dataset/`:

```
dataset/
├── rgb/
│   ├── 000001.png
│   └── ... (1449 imágenes)
└── depth/
    ├── 000001.png
    └── ... (1449 mapas)
```

### Alternativa: Usar Toolbox NYU

Para extraer pares RGB-D sincronizados desde raw data:

```bash
# Requiere MATLAB
# Seguir instrucciones en sitio oficial de NYU
```

Ver [`dataset/README.md`](../dataset/README.md) para solución de problemas.

---

## Paso 4: Ejecutar Experimentos

### Opción A: Grouped Split (30 runs)

**Tiempo estimado**: ~1 hora (GPU) / ~38 horas (CPU)

```bash
bash examples/run_grouped_split.sh
```

**Salidas generadas**:
| Archivo | Descripción |
|---------|-------------|
| `9B02_grouped_split_results.json` | Resultados detallados por run |
| `9B02_grouped_split_summary.csv` | Resumen estadístico agregado |
| `9B02_grouped_split_weights.csv` | Tablas de pesos aprendidos |

### Opción B: Scene Leave-One-Out (24 runs)

**Tiempo estimado**: ~2-3 horas (GPU) / ~80+ horas (CPU)

```bash
bash examples/run_scene_loo.sh
```

**Nota**: Requiere `results/scenes_auto.csv` (incluido en repo).

---

## Paso 5: Verificar Reproducción

### Comparación con Resultados de Referencia

```bash
# Comparar tu summary con el reference
diff 9B02_grouped_split_summary.csv results/grouped_split_30runs_summary.csv
```

### Qué debe coincidir

| Columna | Tolerancia |
|---------|------------|
| `p_sp`, `p_auc`, `p_f1` | **Exacto** (con misma seed) |
| `d_sp`, `d_auc`, `d_f1` | ±0.001 (por orden GPU) |
| Signo de métricas | Debe conservarse |

### Qué puede variar (aceptable)

- Valores exactos de `d_*` por diferencias en orden de operaciones GPU
- Timestamps en archivos
- Orden de filas en CSV (si hay empates)

---

## Troubleshooting

### "No module named torch"

```bash
source .venv/bin/activate
pip install torch numpy pillow scipy
```

### "CUDA out of memory"

```bash
# Reducir chunk size para random baselines
python3 src/motif_survival_grouped.py \
    ... \
    --gpu-random-chunk 16  # default: 32
```

### "No se encontraron imágenes"

```bash
# Verificar estructura
ls dataset/rgb | head -5
ls dataset/depth | head -5

# Verificar README en dataset/
cat dataset/README.md
```

### Resultados muy diferentes

| Síntoma | Posible causa | Solución |
|---------|---------------|----------|
| p-values muy altos | Seeds diferentes | Usar mismas seeds del ejemplo |
| Métricas NaN | Dataset incompleto | Verificar todas las imágenes |
| Mucho más lento | CPU vs GPU | Verificar `torch.cuda.is_available()` |

---

## Validación Rápida (1 run)

Para prueba rápida sin esperar 30 runs:

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

Tiempo: ~2 minutos (GPU) / ~1 hora (CPU)

Deberías ver `p < 0.05` en las métricas principales.

---

## Reportar Problemas

Si encuentras discrepancias no explicadas:

1. Verificar versión de PyTorch: `python3 -c "import torch; print(torch.__version__)"`
2. Guardar log completo: `bash examples/run_grouped_split.sh 2>&1 | tee run.log`
3. Abrir issue con:
   - Sistema operativo y versión
   - Versión de PyTorch y CUDA (si aplica)
   - Archivo de log
   - Archivo summary.csv generado
