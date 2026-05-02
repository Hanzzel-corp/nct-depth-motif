# Trayectoria del Proyecto

Este repositorio contiene una versión consolidada de un trabajo exploratorio más largo. Para contexto:

---

## Origen Personal

Este trabajo nació de un marco simbólico llamado **NCT (Números Cuánticos Tridimensionales)** que desarrollé de manera **autodidacta** a lo largo de varios meses, **sin formación matemática formal previa**.

La intuición original: trabajar con tuplas ordenadas sobre un alfabeto discreto de cuatro símbolos `{+, -, 0, ~}` y operaciones binarias entre ellas para representar estados geométricos.

Llegué a esta representación desde la **experimentación directa**, no desde la literatura matemática. Eso significa que muchas de las ideas que aparecen en NCT tienen contraparte conocida en disciplinas formales (ver tabla de equivalencias en el README).

**El reconocimiento de estas equivalencias no me las quita como ruta de descubrimiento, pero las contextualiza:** lo que se valida acá no es una "matemática nueva", sino una técnica concreta de representación discreta cuya forma específica vino de NCT.

---

## Proceso de Validación Experimental

A través de meses de experimentos sistemáticos, fui sometiendo cada componente a prueba:

| Componente | Hipótesis | Resultado |
|------------|-----------|-----------|
| Operaciones ⊕, ⊗ | Aportarían señal discriminativa | ❌ No superaron baselines simples |
| Cuantización {+, -, 0, ~} | Capturaría estructura geométrica | ✅ Mostró señal consistente |
| Tabla de pesos | Aprender asociaciones estado→ruptura | ✅ Generalizó en validación cruzada |
| Unificación física | Equivalencia con modelos teóricos | ❌ No validable con este pipeline |
| AGI | Base para razonamiento simbólico | ❌ Sin evidencia experimental |
| "Fase 3-6-9" y metáforas | Elementos decorativos | ❌ Sin valor predictivo |

---

## Lo que se descartó en el camino

- **Operaciones binarias ⊕, ⊗** entre estados como motor de detección
- **Aplicaciones a unificación física** (no validables con este pipeline)
- **Aplicaciones a AGI** (sin base experimental)
- **Capas decorativas** como "fase 3-6-9"

## Lo que sobrevivió a validación

| Elemento | Justificación |
|----------|---------------|
| **Discretización en 4 estados {+, -, 0, ~}** | Captura información geométrica esencial |
| **Tabla de pesos por motivo 3D** | Permite adaptación a datos reales |
| **Estado `~` como marcador de transición** | Útil para identificar zonas ambiguas |
| **Gate de ambigüedad triangular sobre delta clásico** | No corregir donde delta ya funciona bien |

Mantener el nombre "NCT" en este reporte es decisión personal: es la marca interna del proyecto desde su concepción. Las equivalencias con técnicas estándar están documentadas abiertamente en el README.

**Equivalencias conocidas**:
- Los descriptores NCT son variantes de descriptores de curvatura local
- La cuantización es similar a LBP (Local Binary Patterns) en 3D
- El gate de ambigüedad funciona como un mecanismo de supresión selectiva (similar en espíritu a un attention mechanism, aunque mucho más simple)

---

## Cronología de Versiones

| Versión | Foco | Estado |
|---------|------|--------|
| v1-v11 | Exploración teórica y operaciones binarias | ❌ Descartado |
| v12-v12.1 | Motivos NCT en depth sintético | Transición |
| v13 | RGB-D reales (NYU Depth V2) | ✅ **Base actual** |
| v13.4 | Grouped split validation | ✅ Consolidado |
| v14.2.1 | Scene leave-one-out | ✅ Consolidado |

---

## Filosofía del Proyecto

> **"Lo que no se puede falsar experimentalmente no pertenece a este repositorio."**

Este proyecto adopta principios de:

- **Empirismo fuerte**: Solo componentes que mejoran métricas reales
- **Minimalismo**: La representación más simple que capture el fenómeno
- **Honestidad**: Documentar tanto hallazgos positivos como negativos
- **Reproducibilidad**: Cualquier claim debe ser verificable por terceros

---

## Reconocimientos

El método final es el resultado de un proceso iterativo de "pruning agresivo" donde la mayoría de las ideas originales (incluyendo operaciones binarias, aplicaciones a física unificada y AGI) fueron descartadas tras no mostrar señal empírica. Este proceso fue crucial para llegar a un sistema que realmente funciona en datos reales.
