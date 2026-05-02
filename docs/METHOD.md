# Detalle Técnico del Método

**Estado:** Reporte técnico exploratorio. No revisado por pares.

Este documento describe el pipeline completo del método NCT evaluado sobre NYU Depth V2.

**Hallazgo principal:** La representación discreta de motivos 3D muestra señal estadísticamente significativa (p < 0.01) aunque de magnitud modesta (ΔAUC ≈ +0.004).

---

## Pipeline Completo

### 1. Preprocesamiento de Entrada

```
RGB (opcional) + Depth → Superficie 3D P(X,Y,Z)
```

| Paso | Operación | Fórmula |
|------|-----------|---------|
| Carga | Leer pares RGB-D | NYU Depth V2 |
| Normalización | Convertir a metros | `depth_m = depth_uint16 / 1000.0` |
| Proyección | Back-project a 3D | Usando intrínsecos de cámara |
| Normales | Calcular normales locales | `n = ∇S / |∇S|` |

### 2. Extracción de Descriptores NCT

Cálculo de componentes direccionales:

- **Sx**: Gradiente direccional en X (primera derivada espacial)
- **Sy**: Gradiente direccional en Y (primera derivada espacial)
- **Sz**: Laplaciano local (segunda derivada, curvatura escalar)

### 3. Discretización de Estados

Cada componente se cuantiza en 4 estados:

| Estado | Condición | Significado |
|--------|-----------|-------------|
| `+` | valor > +threshold | Curvatura positiva fuerte |
| `-` | valor < -threshold | Curvatura negativa fuerte |
| `0` | \|valor\| ≤ threshold | Región plana |
| `~` | en banda transición | Zona de transición |

**Total de motivos**: 4³ = 64 combinaciones posibles de `(Sx, Sy, Sz)`

### 4. Definición de Target

El target de ruptura combina tres señales:

```
target = λ₁·depth_edges + λ₂·normal_edges + λ₃·plane_residual
```

| Componente | Descripción |
|------------|-------------|
| `depth_edges` | Gradientes altos en mapa de profundidad |
| `normal_edges` | Cambios bruscos en orientación de normales |
| `plane_residual` | Error de ajuste a plano local |

### 5. Aprendizaje de Pesos (Train)

Para cada motivo `m`:

```
lift(m) = E[target | motif = m]  (con shrinkage Bayesiano)
weight[m] = normalize(lift(m)) ∈ [-1, 1]
```

**Shrinkage aplicado**: motivos con pocas observaciones (`< min_count`) se shrinkean hacia la media global.

### 6. Predicción (Test)

```
delta = classical_depth_edge_score
ambiguity_gate = triangular_gate(delta)  # [0,1]
score = delta + alpha · ambiguity_gate · weight[motif]
```

El `gate` suprime la corrección NCT donde el delta clásico ya discrimina bien.

---

## Hiperparámetros

| Parámetro | Descripción | Rango típico | Default |
|-----------|-------------|--------------|---------|
| `alpha` | Peso de corrección NCT | 0.02 - 0.04 | 0.03 |
| `state_threshold` | Umbral para estados `+`/`-` | 0.1 - 0.5 | 0.2 |
| `tilde_band` | Ancho de banda para estado `~` | 0.05 - 0.2 | 0.1 |
| `min_count` | Mínimo de ocurrencias para confianza | 50 - 200 | 100 |
| `shrinkage_power` | Fuerza del shrinkage Bayesiano | 0.5 - 2.0 | 1.0 |
| `random_baselines` | Número de permutaciones para p-value | 256 - 1024 | 256 |

---

## Métricas de Evaluación

Todas las métricas se calculan en la zona **AMBIGUOUS_ONLY** (donde `delta` está en banda intermedia y no separa bien).

### Spearman ρ
Correlación de ranking entre score predicho y target real.

```
ρ = corr_rank(score, target)
```

### AUC top-20%
Área bajo la curva ROC restringida al 20% superior de scores.

```
AUC@20 = ROC_AUC(score > percentile(score, 80), target)
```

### F1 top-20%
F1-score entre píxeles top-20% de score vs top-20% de target.

```
F1@20 = F1(pred_top20, target_top20)
```

**Nota**: La zona NON_AMBIGUOUS se excluye porque el delta clásico ya discrimina correctamente allí.

---

## p-value Empírico

Cálculo de significancia estadística contra baselines aleatorios:

```
p_metric = (1 + #{random_runs : metric(random) ≥ metric(model)}) / (1 + N_random)
```

| N_random | p-value mínimo | Interpretación |
|----------|----------------|----------------|
| 256 | 1/257 ≈ 0.0039 | Modelo supera a TODOS los azares |
| 1024 | 1/1025 ≈ 0.0010 | Máxima confianza |

---

## Zonas de Evaluación

```
Zona            | Condición delta      | Corrección NCT
----------------|----------------------|------------------
NON_AMBIGUOUS   | delta < low_th       | No aplicable (suprimida)
AMBIGUOUS_ONLY  | low_th ≤ delta ≤ high_th | Activa (gate ≈ 1)
NON_AMBIGUOUS   | delta > high_th      | No aplicable (suprimida)
```

La lógica: no arreglar lo que no está roto. Si delta ya separa bien (zonas extremas), no aplicar corrección.
