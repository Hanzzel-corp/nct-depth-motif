# Cómo Interpretar los Resultados

Guía de lectura de los archivos de resultados generados por los experimentos NCT.

---

## Cuatro Modelos Evaluados

Cada ejecución entrena y evalúa cuatro variantes de la tabla de pesos:

| Modelo | Descripción | Uso |
|--------|-------------|-----|
| `motif_survival` | Pesos completos con shrinkage | Modelo principal |
| `motif_survival_pos_only` | Solo pesos > 0 | Identificar motivos "protectores" |
| `motif_survival_neg_only` | Solo pesos < 0 | Identificar motivos "de ruptura" |
| `motif_survival_binary` | Solo signo (+1/-1/0) | Test de información mínima |

## Resultados de Referencia

Valores esperados de ejecuciones exitosas sobre NYU Depth V2:

### Grouped numeric block split (30 runs)

3 alphas × 10 seeds, bloques de 50 frames, 256 random baselines:

| Modelo | p_F1 | p_AUC | p_Spearman | < 0.05 en |
|--------|------|-------|------------|-----------|
| `motif_survival` | 0.0039 | 0.0079 | 0.0039 | 30/30 |
| `motif_survival_binary` | 0.0039 | 0.0039 | 0.0039 | 30/30 |
| `motif_survival_pos_only` | 0.0042 | 0.0926 | 0.0525 | F1 30/30, AUC 1/30 |
| `motif_survival_neg_only` | 0.1540 | 0.2432 | 0.0660 | 0/30 |

**Magnitudes**: ΔAUC ≈ +0.004, ΔF1 ≈ +0.005 vs random

---

## Lectura de p-values

El p-value indica la probabilidad de que el resultado sea azar.

| Rango | Interpretación |
|-------|----------------|
| **p < 0.0039** | Significancia máxima. Con 256 randoms, el modelo supera a TODOS los azares posibles |
| **p < 0.01** | Muy significativo. Fuerte evidencia contra H0 |
| **p < 0.05** | Significativo (clásico). Evidencia estándar contra H0 |
| **p = 0.05-0.10** | Tendencia. Requiere más datos |
| **p > 0.10** | No significativo. Indistinguible de azar |

**Fórmula**: `p = (1 + n_mejores_random) / (1 + N_random)`

---

## Hallazgo Clave: Binary = Full

### Observación

`motif_survival_binary` alcanza la **misma significancia** que `motif_survival` completo (a veces mayor AUC).

### Interpretación

| Aspecto | Conclusión |
|---------|------------|
| **Información útil** | Está en la **dirección** de asociación (+/-) |
| **Magnitud** | Agrega ruido más que valor discriminativo |
| **Implicación** | La cuantización NCT captura estructura geométrica real |

### Por qué ocurre

La dirección indica:
- **Peso positivo**: motivo asociado con supervivencia (no ruptura)
- **Peso negativo**: motivo asociado con ruptura

La magnitud exacta depende del shrinkage y del tamaño de muestra, introduciendo variabilidad innecesaria.

---

## Magnitudes del Efecto

### Tamaño del efecto observado

| Métrica | Mejora vs Random | Interpretación |
|---------|------------------|----------------|
| ΔAUC | ~+0.004 | Pequeño pero consistente |
| ΔF1 | ~+0.005 | Pequeño pero consistente |

### Contextualización

```
Efecto NCT vs random:    +0.004 en AUC  (p < 0.01)
Efecto vs Canny/Sobel:   NO MEDIDO en este trabajo (pendiente)
```

**Veredicto**:
- ✅ Estadísticamente significativo (rechaza H0)
- ⚠️ Prácticamente modesto (no reemplaza detectores clásicos)
- 💡 Complementario: corrige delta en zona ambigua

---

## Límites de Interpretación

### Lo que SÍ se puede concluir

- Los motivos NCT contienen información estadísticamente significativa sobre discontinuidades 3D
- La dirección del peso es más informativa que su magnitud
- El método generaliza dentro de NYU Depth V2 (scene LOO)

### Lo que NO se puede concluir (aún)

| Afirmación | Razón |
|------------|-------|
| "Superior a Canny/Sobel/HED" | No se comparó directamente en el mismo benchmark |
| "Generaliza a ScanNet/KITTI" | Solo probado en NYU Depth V2 |
| "NCT es la cuantización óptima" | Otras cuantizaciones no fueron exploradas sistemáticamente |
| "Listo para producción" | Mejoras marginales sobre baseline clásico |

---

## Checklist de Reproducción

Para confirmar que tus resultados son consistentes:

- [ ] `p_sp`, `p_auc`, `p_f1` coinciden (o son muy similares) con reference
- [ ] `motif_survival_binary` es competitivo con `motif_survival`
- [ ] Todos los modelos test superan a `random_motif_mean`
- [ ] Efecto concentrado en zona AMBIGUOUS_ONLY
- [ ] Signo de pesos es consistente entre runs (ej: `(+,+,-)` suele ser negativo)

---

## Ejemplo de Lectura

Fragmento típico de `summary.csv`:

```csv
model,alpha,p_sp,p_auc,p_f1
motif_survival,0.03,0.0039,0.0078,0.0039
motif_survival_binary,0.03,0.0039,0.0039,0.0039
random_motif_mean,0.03,0.2500,0.4500,0.3200
```

**Lectura**: Ambos modelos NCT alcanzan significancia máxima/muy alta, mientras el promedio de azares no supera el azar esperado (p ≈ 0.5).
