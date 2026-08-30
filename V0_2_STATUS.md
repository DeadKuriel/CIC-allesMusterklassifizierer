# Estado verificable de AMK 0.2

La base 0.2 es ejecutable, pero **todavía no satisface todos los criterios de aceptación** y no debe
publicarse como versión metodológicamente terminada.

## Implementado y comprobado

- Configuración Pydantic versionada y validación previa a artefactos.
- Carga y validación de CSV/TSV; rutas de Parquet/Excel mediante extras instalables.
- Auditoría y políticas trazables de redundantes e indiscernibles.
- Preprocesamiento por tipo mediante `ColumnTransformer` y samplers mediante `imblearn.Pipeline`.
- Holdout, K-fold, LOO, grupos y tiempo con invariantes de intersección, grupos y precedencia.
- Registro de 13 familias de modelos, Grid/Randomized Search y pipeline persistido.
- Métricas principales y matrices estrictas contra etiquetas desconocidas.
- CLI 0.2, compatibilidad aislada con `validate`/`classify` 0.1, manifiesto y reporte HTML.
- Correcciones del typo de matplotlib, alias de distancia, loop singleton, NaN/inf, esquema legado y
  omisión silenciosa de etiquetas.

## Pendiente antes de declarar 0.2 completa

- Pruebas instrumentadas que demuestren fit por fold y que SMOTE nunca observa test.
- Evaluación out-of-fold formal para estrategias CV sin holdout; actualmente la ruta estable y
  demostrada es holdout final.
- Separar semánticamente `train` y `evaluate`; hoy delegan al orquestador completo.
- Pruebas de CSV/TSV/Parquet/Excel, todos los tipos semánticos, unknown categories, limpieza,
  tuning sin test, comparación con splits compartidos y CLI end-to-end.
- Validación anidada y successive halving real (el valor `halving` no debe documentarse como
  terminado hasta integrar `HalvingGridSearchCV`).
- SMOTENC automático correcto para datos mixtos; actualmente exige índices transformados explícitos.
- Selección de características, transformaciones por unidades/log/clipping y antigüedad configurable.
- Curvas ROC/PR, imágenes de matrices y reporte HTML enriquecido.
- Ocho configuraciones de ejemplo ejecutables solicitadas y documentación exhaustiva por flujo.
- Caracterización completa del prototipo legado y paridad demostrada.
- Elevar cobertura global desde 22.06% a un umbral apropiado para liberación.
