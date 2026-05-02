# Dataset

⚠️ **El dataset NYU Depth V2 NO está incluido en este repositorio.**  
Debe descargarse manualmente desde la fuente oficial (ver instrucciones abajo).

Esta carpeta debe contener el dataset NYU Depth V2 para ejecutar los experimentos.

## Estructura requerida

```
dataset/
├── rgb/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── depth/
    ├── 000001.png
    ├── 000002.png
    └── ...
```

## Formato de archivos

- **RGB**: Imágenes en formato `.png`, `.jpg` o `.jpeg`
- **Depth**: Mapas de profundidad en formato:
  - `.png` (uint16 en milímetros, usa `--depth-scale 1000`)
  - `.npy` o `.npz` (arrays NumPy)
  - `.tif` o `.tiff`

## Descarga

### NYU Depth V2 (Labeled)

1. Visitar: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Descargar el dataset "Labeled" (~2.8GB)
3. Extraer las carpetas `rgb/` y `depth/` aquí

### Alternativa: NYU Depth V2 (Raw)

Para más imágenes, usar el dataset raw (~428GB) o el toolbox oficial de MATLAB para extraer pares RGB-D sincronizados.

## Notas importantes

- Los archivos RGB y depth deben tener el **mismo nombre base** para ser emparejados automáticamente
- Ejemplo: `rgb/000001.png` se empareja con `depth/000001.png`
- La profundidad en formato PNG uint16 típicamente usa escala de 1000 (milímetros)

## Parámetros de cámara (NYU)

Para resultados óptimos, usar estos intrínsecos:

```bash
--fx 518.8579 --fy 519.4696 --cx 325.5824 --cy 253.7362
```

## Solución de problemas

| Problema | Solución |
|----------|----------|
| "No se encontraron imágenes" | Verificar que los nombres de archivo coincidan en rgb/ y depth/ |
| Profundidad incorrecta | Verificar `--depth-scale` (1000 para mm, 1 para metros) |
| Pocos pares detectados | Algunas imágenes pueden no tener depth disponible en el dataset original |

---

Para más detalles sobre el formato del dataset, consultar la documentación oficial de NYU Depth V2.
